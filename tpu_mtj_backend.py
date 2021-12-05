import multiprocessing
from typing import Any, Dict, List, Optional
import progressbar
import time
import os
import requests
import random
import jax
from jax.config import config
from jax.experimental import maps
import jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import transformers
from mesh_transformer.checkpoint import read_ckpt_lowmem
from mesh_transformer.transformer_shard import CausalTransformer, CausalTransformerShard


params: Dict[str, Any] = {}


def show_spinner():
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength, widgets=[progressbar.Timer(), '  ', progressbar.BouncingBar(left='[', right=']', marker='█')])
    i = 0
    while True:
        bar.update(i)
        time.sleep(0.1)
        i += 1

def apply_repetition_penalty(logits, tokens, repetition_penalty):
    '''
    This gets called by generate_loop_fn to apply repetition penalty
    to the 1D array logits using the provided 1D array of tokens to penalize
    '''
    # Make a new array with the same length as the tokens array but with
    # each element replaced by the value at the corresponding index in the
    # logits array; e.g.
    # if logits is [77, 5, 3, 98] and tokens is [0, 1, 2, 3, 2, 3, 1],
    # then penalty_logits will be [77, 5, 3, 98, 3, 98, 5]
    penalty_logits = jnp.take(logits, tokens)
    # Divide positive values by repetition_penalty and multiply negative
    # values by repetition_penalty (the academic publication that described
    # this technique actually just only divided, but that would cause tokens
    # with negative logits to become more likely, which is obviously wrong)
    penalty_logits = jnp.where(
        penalty_logits > 0,
        penalty_logits/repetition_penalty,
        penalty_logits*repetition_penalty,
    )
    # Finally, put those penalized logit values back into their original
    # positions in the logits array
    return logits.at[tokens].set(penalty_logits)

def kobold_sample(key, logits, top_p=0.9, temp=0.5, top_k=0, tfs=1.0):
    '''
    This gets called by generate_loop_fn to apply a series of 4 filters
    to the logits (top-k, then top-p, then TFS, then temperature) before
    picking one token using the modified logits
    '''
    # Top-k (keep only the k tokens with the highest logits and remove
    # the rest, by setting their logits to negative infinity)
    def top_k_filter(logits):
        # After sorting the logits array in descending order,
        # sorted_indices_to_remove is a 1D array that is True for tokens
        # in the sorted logits array we want to remove and False for ones
        # we want to keep, in this case the first top_k elements will be
        # False and the rest will be True
        sorted_indices_to_remove = jnp.arange(len(logits)) >= top_k
        # Unsort the logits array back to its original configuration and
        # remove tokens we need to remove
        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(-logits),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, logits)
    logits = jax.lax.cond(top_k > 0, top_k_filter, lambda x: x, logits)
    # Top-p (after sorting the remaining tokens again in descending order of
    # logit, remove the ones that have cumulative softmax probability
    # greater than p)
    def top_p_filter(logits):
        # Sort the logits array in descending order, replace every element
        # with e (Euler's number) to the power of that element, and divide
        # each element of the new array by the sum of the elements in the
        # new array
        sorted_logits = -jnp.sort(-logits)
        probabilities = jax.nn.softmax(sorted_logits)
        # Calculate cumulative_probabilities as the prefix-sum array of
        # probabilities
        cumulative_probabilities = jnp.cumsum(probabilities, axis=-1)
        # We want to remove tokens with cumulative probability higher
        # than top_p
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Don't ever remove the token with the highest logit, even if
        # the probability is higher than top_p
        sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
        # Unsort and remove
        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(-logits),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, logits)
    logits = jax.lax.cond(top_p < 1.0, top_p_filter, lambda x: x, logits)
    # Tail free sampling (basically top-p a second time on remaining tokens
    # except it's the "cumulative normalized absolute second finite
    # differences of the softmax probabilities" instead of just the
    # cumulative softmax probabilities)
    def tail_free_filter(logits):
        # Sort in descending order
        sorted_logits = -jnp.sort(-logits)
        # Softmax again
        probabilities = jax.nn.softmax(sorted_logits)
        # Calculate the second finite differences of that array (i.e.
        # calculate the difference array and then calculate the difference
        # array of the difference array)
        d2 = jnp.diff(jnp.diff(probabilities))
        # Get the absolute values of all those second finite differences
        d2 = jnp.abs(d2)
        # Normalize (all elements in the array are divided by the sum of the
        # array's elements)
        d2 = d2 / d2.sum(axis=-1, keepdims=True)
        # Get the prefix-sum array
        cumulative_d2 = jnp.cumsum(d2, axis=-1)
        # We will remove the tokens with a cumulative normalized absolute
        # second finite difference larger than the TFS value
        sorted_indices_to_remove = cumulative_d2 > tfs
        # Don't remove the token with the highest logit
        sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
        # Since the d2 array has two fewer elements than the logits array,
        # we'll add two extra Trues to the end
        sorted_indices_to_remove = jnp.pad(
            sorted_indices_to_remove,
            (0, 2),
            constant_values=True,
        )
        # Unsort and remove
        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(-logits),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, logits)
    logits = jax.lax.cond(tfs < 1.0, tail_free_filter, lambda x: x, logits)
    # Temperature (just divide the logits by the temperature)
    def temp_filter(logits):
        return logits / temp
    logits = jax.lax.cond(True, temp_filter, lambda x: x, logits)
    # Finally, pick one token using the softmax thingy again (it gives
    # an array whose elements sum to 1 so it can be used nicely as a
    # probability distribution)
    return jax.random.categorical(key, logits, -1).astype(jnp.uint32)[jnp.newaxis]

pad_token_id = 50256

class PenalizingCausalTransformer(CausalTransformer):
    def __init__(self, config):
        # Initialize
        super().__init__(config)
        def generate(state, key, ctx, ctx_length, gen_length, numseqs_aux, sampler_options, soft_embeddings=None):
            numseqs = numseqs_aux.shape[0]
            # These are the tokens that we don't want the AI to ever write
            self.badwords = jnp.array([6880, 50256, 42496, 4613, 17414, 22039, 16410, 27, 29, 38430, 37922, 15913, 24618, 28725, 58, 47175, 36937, 26700, 12878, 16471, 37981, 5218, 29795, 13412, 45160, 3693, 49778, 4211, 20598, 36475, 33409, 44167, 32406, 29847, 29342, 42669, 685, 25787, 7359, 3784, 5320, 33994, 33490, 34516, 43734, 17635, 24293, 9959, 23785, 21737, 28401, 18161, 26358, 32509, 1279, 38155, 18189, 26894, 6927, 14610, 23834, 11037, 14631, 26933, 46904, 22330, 25915, 47934, 38214, 1875, 14692, 41832, 13163, 25970, 29565, 44926, 19841, 37250, 49029, 9609, 44438, 16791, 17816, 30109, 41888, 47527, 42924, 23984, 49074, 33717, 31161, 49082, 30138, 31175, 12240, 14804, 7131, 26076, 33250, 3556, 38381, 36338, 32756, 46581, 17912, 49146])
            def generate_sample(context, ctx_length):
                # Give the initial context to the transformer
                transformer = CausalTransformerShard(config)
                def generate_initial_scan_fn(sequence_index, _):
                    _, initial_state = transformer.generate_initial(context, ctx_length, soft_embeddings=soft_embeddings)
                    # The "generated" array will contain the tokens from the
                    # context as well as the tokens picked by the sampler at
                    # each stage, padded with a bunch of 50256s, so we know
                    # which tokens have to be repetition penalized
                    generated = jnp.pad(context, (0, config["seq"]), constant_values=pad_token_id)  # Let it start off with just the 2048 context tokens, plus some 50256s which will be eventually filled with sampler-chosen tokens
                    generated_index = config["seq"]
                    # Add that information to generate_loop_fn's starting state
                    initial_state = (generated, generated_index, sequence_index) + initial_state
                    return sequence_index+1, initial_state
                _, initial_states = jax.lax.scan(generate_initial_scan_fn, 0, None, numseqs)
                sample_key = initial_states[-1][0]
                initial_states = list(jax.tree_map(lambda x: x[i], initial_states[:-1]) for i in range(numseqs))
                # Get repetition penalty from the arguments
                repetition_penalty = sampler_options.pop('repetition_penalty', None)
                # This is the main generation loop
                def generate_loop_fn(carry):
                    # Unpack current generate_loop_fn state
                    generated, generated_index, sequence_index, next_token, decode_state = carry[0][0]
                    sample_key = carry[1]
                    # Get the pseudo-random number generator key that will
                    # be used by kobold_sample to randomly pick a token
                    sample_key, new_key = jax.random.split(sample_key)
                    # Give the context to the model and get the logits it
                    # spits out
                    # (a 2D array with 1 row and 50400 columns representing
                    # how strongly it thinks each of the 50257 tokens in its
                    # vocabulary should be appended to the context, followed
                    # by 143 apparently useless columns ???)
                    logits, new_state = transformer.generate_once(next_token, decode_state, soft_embeddings=soft_embeddings)
                    # Verify that logits does indeed have that many rows and
                    # columns (if you get an error here, pray for mercy)
                    assert logits.shape == (1, config["n_vocab"])
                    # Flatten it into a 1D array to make it easier to use
                    logits = logits[0]
                    # Apply repetition penalty to all tokens that are
                    # currently inside the "generated" array
                    if repetition_penalty is not None:
                        logits = apply_repetition_penalty(
                            logits,
                            generated,
                            repetition_penalty
                        )
                    # Remove any tokens in the badwords list by setting
                    # their logits to negative infinity which effectively
                    # makes their probabilities of being chosen zero
                    logits = logits.at[self.badwords].set(-jnp.inf)
                    # Use the sampler (kobold_sample) to pick one token
                    # based on the logits array as a 1D array with 1 element
                    # (higher logit means higher probability of being
                    # picked, non-linearly)
                    next_token = kobold_sample(
                        sample_key,
                        logits,
                        **sampler_options,
                    )
                    # Remember what token was picked
                    generated = generated.at[generated_index].set(next_token[0])
                    generated_index += 1
                    # Re-pack the current generate_loop_fn's state so we can
                    # get back the same variables the next time
                    carry[0][0] = (generated, generated_index, sequence_index, next_token, new_state)
                    carry[0].append(carry[0].pop(0))
                    return carry[0], new_key
                final_state = jax.lax.while_loop(
                    lambda carry: carry[0][0][1] - config["seq"] < gen_length,
                    generate_loop_fn,
                    (initial_states, sample_key),
                )
                return final_state
            generate_fn = hk.transform(generate_sample).apply
            return generate_fn(state["params"], key, ctx, ctx_length)
        self.generate_xmap = jax.experimental.maps.xmap(
            fun=generate,
            in_axes=(
                ["shard", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["shard", ...],
            ),
            out_axes=["shard", "batch", ...],
            axis_resources={'shard': 'mp', 'batch': 'dp'},
        )
    def generate(self, ctx, ctx_length, gen_length, numseqs, sampler_options, return_logits=False, soft_embeddings=None):
        assert not return_logits
        key = hk.PRNGSequence(random.randint(0, 2 ** 60))
        batch_size = ctx.shape[0]
        self.batch_size = batch_size
        return self.generate_xmap(
            self.state,
            jnp.array(key.take(batch_size)),
            ctx,
            np.array(ctx_length, dtype=np.uint32),
            np.array(gen_length, dtype=np.uint32),
            np.empty((batch_size, numseqs), dtype=np.uint8),
            sampler_options,
            soft_embeddings,
        )


def infer(
    context: str,
    top_p=0.9,
    temp=0.5,
    top_k=0,
    tfs=1.0,
    repetition_penalty=1.0,
    numseqs=1,
    gen_len=80,
    soft_embeddings: Optional[np.array] = None,
    soft_tokens: Optional[np.array] = None,
) -> List[str]:
    maps.thread_resources.env = thread_resources_env
    total_batch = 1
    tokens = np.uint32(tokenizer.encode(context, max_length=params["seq"] - (soft_tokens.shape[0] if soft_tokens is not None else 0), truncation=True))
    if(soft_tokens is not None):
        tokens = np.uint32(np.concatenate((soft_tokens, tokens)))
    provided_ctx = tokens.shape[0]
    pad_amount = seq - provided_ctx
    padded_tokens = np.pad(tokens, ((pad_amount, 0),), constant_values=pad_token_id)
    batched_tokens = np.array([padded_tokens] * total_batch)
    samples = []
    batched_generator_params = {
        "temp": temp * np.ones(total_batch),
        "top_p": top_p * np.ones(total_batch),
        "tfs": tfs * np.ones(total_batch),
        "repetition_penalty": repetition_penalty * np.ones(total_batch),
        "top_k": np.full(total_batch, top_k, dtype=np.uint32)
    }
    output = network.generate(
        batched_tokens,
        np.ones(total_batch, dtype=np.uint32) * provided_ctx,
        np.ones(total_batch, dtype=np.uint32) * gen_len,
        numseqs,
        batched_generator_params,
        soft_embeddings=soft_embeddings,
    )[0]
    for o in output:
        samples.append(tokenizer.decode(o[0][0, 0, params["seq"] : params["seq"] + gen_len]))
    return samples


def load_model(path: str, driver_version="tpu_driver0.1_dev20210607", **kwargs) -> None:
    global thread_resources_env, seq, tokenizer, network, params

    default_params = {
        "compat": "j",
        "layers": 28,
        "d_model": 4096,
        "n_heads": 16,
        "n_vocab": 50400,
        "n_vocab_padding": 0,
        "norm": "layernorm",
        "pe": "rotary",
        "pe_rotary_dims": 64,
        "seq": 2048,
        "cores_per_replica": 8,
    }
    params = kwargs
    for param in default_params:
        if param not in params:
            params[param] = default_params[param]

    # Disable JAX warnings about these two functions having been renamed
    jax.host_count = jax.process_count
    jax.host_id = jax.process_index

    print("Connecting to your Colab instance's TPU", flush=True)
    spinner = multiprocessing.Process(target=show_spinner, args=())
    spinner.start()
    colab_tpu_addr = os.environ['COLAB_TPU_ADDR'].split(':')[0]
    url = f'http://{colab_tpu_addr}:8475/requestversion/{driver_version}'
    requests.post(url)
    spinner.terminate()
    print()
    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']

    cores_per_replica = params["cores_per_replica"]
    seq = params["seq"]
    params["optimizer"] = optax.scale(0)
    mesh_shape = (1, cores_per_replica)
    devices = np.array(jax.devices()[:cores_per_replica]).reshape(mesh_shape)
    thread_resources_env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')), ())
    maps.thread_resources.env = thread_resources_env
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

    if not path.endswith("/"):
        path += "/"

    network = PenalizingCausalTransformer(params)
    network.state = read_ckpt_lowmem(network.state, path, devices.shape[1])
    network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))
