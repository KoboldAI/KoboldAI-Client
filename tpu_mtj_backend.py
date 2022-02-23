'''
This file is AGPL-licensed.

Some of the code in this file is from Clover Edition:
https://github.com/cloveranon/Clover-Edition/blob/master/aidungeon/gpt2generator.py

The license for Clover Edition is shown below:

Copyright (c) 2019 Nick Walton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import multiprocessing
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
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


def warper_callback(logits) -> np.array:
    raise NotImplementedError("`tpu_mtj_backend.warper_callback()` needs to be defined")

def stopping_callback(generated, n_generated, excluded_world_info) -> Tuple[List[set], bool, bool]:
    raise NotImplementedError("`tpu_mtj_backend.stopping_callback()` needs to be defined")

def settings_callback() -> dict:
    return {
        "top_p": 0.9,
        "temp": 0.5,
        "top_k": 0,
        "tfs": 1.0,
        "repetition_penalty": 1.0,
        "rpslope": 0.0,
        "rprange": 0,
    }

def started_compiling_callback() -> None:
    pass

def stopped_compiling_callback() -> None:
    pass

def compiling_callback() -> None:
    pass


def show_spinner():
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength, widgets=[progressbar.Timer(), '  ', progressbar.BouncingBar(left='[', right=']', marker='█')])
    i = 0
    while True:
        bar.update(i)
        time.sleep(0.1)
        i += 1


__F = TypeVar("__F", bound=Callable)
__T = TypeVar("__T")

def __move_xmap(f: __F, out_axis: str) -> __F:
    return maps.xmap(
        f,
        in_axes=(["shard", ...], ["batch", ...]),
        out_axes=[out_axis, ...],
        axis_resources={'shard': 'mp', 'batch': 'dp'},
    )

def __shard_xmap(batch_dim=1):
    xmap = __move_xmap(lambda s, b: s, "shard")
    def inner(x: __T) -> __T:
        return xmap(x, np.empty(batch_dim))
    return inner

def __batch_xmap(shard_dim=1):
    xmap = __move_xmap(lambda s, b: b, "batch")
    def inner(x: __T) -> __T:
        return xmap(np.empty(shard_dim), x)
    return inner


def apply_repetition_penalty_dynamic(logits, tokens, repetition_penalty, generated_index, gen_length, rpslope, rprange):
    '''
    This gets called by generate_loop_fn to apply repetition penalty
    to the 1D array logits using the provided 1D array of tokens to penalize
    '''
    tokens = np.minimum(tokens, params["n_vocab"]-1)  # https://github.com/google/jax/issues/3774
    rpslope = np.int32(rpslope)
    rprange = np.int32(rprange)
    clipped_rprange = rprange if rprange > 0 else tokens.shape[-1]
    penalty_arange = np.roll(np.arange(tokens.shape[-1]) + (clipped_rprange - tokens.shape[-1]), generated_index, axis=-1)
    # Make a new array with the same length as the tokens array but with
    # each element replaced by the value at the corresponding index in the
    # logits array; e.g.
    # if logits is [77, 5, 3, 98] and tokens is [0, 1, 2, 3, 2, 3, 1],
    # then penalty_logits will be [77, 5, 3, 98, 3, 98, 5]
    penalty_logits = np.take(logits, tokens)
    # Repetition penalty slope
    if rpslope != 0.0 and rprange > 0:
        _penalty = (penalty_arange/(rprange - 1)) * 2 - 1
        _penalty = (rpslope * _penalty) / (1 + np.abs(_penalty) * (rpslope - 1))
        _penalty = 1 + ((_penalty + 1) / 2) * (repetition_penalty - 1)
        repetition_penalty = _penalty
    # Divide positive values by repetition_penalty and multiply negative
    # values by repetition_penalty (the academic publication that described
    # this technique actually just only divided, but that would cause tokens
    # with negative logits to become more likely, which is obviously wrong)
    penalty_logits = np.where(
        penalty_arange >= 0,
        np.where(
            penalty_logits > 0,
            penalty_logits/repetition_penalty,
            penalty_logits*repetition_penalty,
        ),
        penalty_logits,
    )
    # Finally, put those penalized logit values back into their original
    # positions in the logits array
    logits[tokens] = penalty_logits
    return logits

def kobold_sample_dynamic(key, logits, top_p=0.9, temp=0.5, top_k=0, tfs=1.0):
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
        sorted_indices_to_remove = np.arange(len(logits)) >= top_k
        # Unsort the logits array back to its original configuration and
        # remove tokens we need to remove
        _, indices_to_remove = jax.lax.sort_key_val(
            np.argsort(-logits),
            sorted_indices_to_remove,
        )
        return np.where(indices_to_remove, -np.inf, logits)
    if top_k > 0:
        logits = top_k_filter(logits)
    # Top-p (after sorting the remaining tokens again in descending order of
    # logit, remove the ones that have cumulative softmax probability
    # greater than p)
    def top_p_filter(logits):
        # Sort the logits array in descending order, replace every element
        # with e (Euler's number) to the power of that element, and divide
        # each element of the new array by the sum of the elements in the
        # new array
        sorted_logits = -np.sort(-logits)
        probabilities = np.array(jax.nn.softmax(sorted_logits), copy=True)
        # Calculate cumulative_probabilities as the prefix-sum array of
        # probabilities
        cumulative_probabilities = np.cumsum(probabilities, axis=-1)
        # We want to remove tokens with cumulative probability higher
        # than top_p
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Don't ever remove the token with the highest logit, even if
        # the probability is higher than top_p
        sorted_indices_to_remove[0] = False
        # Unsort and remove
        _, indices_to_remove = jax.lax.sort_key_val(
            np.argsort(-logits),
            sorted_indices_to_remove,
        )
        return np.where(indices_to_remove, -np.inf, logits)
    if top_p < 1.0:
        logits = top_p_filter(logits)
    # Tail free sampling (basically top-p a second time on remaining tokens
    # except it's the "cumulative normalized absolute second finite
    # differences of the softmax probabilities" instead of just the
    # cumulative softmax probabilities)
    def tail_free_filter(logits):
        # Sort in descending order
        sorted_logits = -np.sort(-logits)
        # Softmax again
        probabilities = np.array(jax.nn.softmax(sorted_logits), copy=True)
        # Calculate the second finite differences of that array (i.e.
        # calculate the difference array and then calculate the difference
        # array of the difference array)
        d2 = np.diff(np.diff(probabilities))
        # Get the absolute values of all those second finite differences
        d2 = np.abs(d2)
        # Normalize (all elements in the array are divided by the sum of the
        # array's elements)
        d2 = d2 / d2.sum(axis=-1, keepdims=True)
        # Get the prefix-sum array
        cumulative_d2 = np.cumsum(d2, axis=-1)
        # We will remove the tokens with a cumulative normalized absolute
        # second finite difference larger than the TFS value
        sorted_indices_to_remove = cumulative_d2 > tfs
        # Don't remove the token with the highest logit
        sorted_indices_to_remove[0] = False
        # Since the d2 array has two fewer elements than the logits array,
        # we'll add two extra Trues to the end
        sorted_indices_to_remove = np.pad(
            sorted_indices_to_remove,
            (0, 2),
            constant_values=True,
        )
        # Unsort and remove
        _, indices_to_remove = jax.lax.sort_key_val(
            np.argsort(-logits),
            sorted_indices_to_remove,
        )
        return np.where(indices_to_remove, -np.inf, logits)
    if tfs < 1.0:
        logits = tail_free_filter(logits)
    # Temperature (just divide the logits by the temperature)
    logits /= temp
    # Finally, pick one token using the softmax thingy again (it gives
    # an array whose elements sum to 1 so it can be used nicely as a
    # probability distribution)
    return jax.random.categorical(key, logits, -1).astype(np.uint32)

def apply_repetition_penalty_static(logits, tokens, repetition_penalty, generated_index, gen_length, rpslope, rprange):
    '''
    This gets called by generate_loop_fn to apply repetition penalty
    to the 1D array logits using the provided 1D array of tokens to penalize
    '''
    rpslope = jnp.int32(rpslope)
    rprange = jnp.int32(rprange)
    clipped_rprange = jax.lax.cond(rprange > 0, lambda x: x, lambda x: tokens.shape[-1], rprange)
    penalty_arange = jnp.roll(jnp.arange(tokens.shape[-1]) + (clipped_rprange - tokens.shape[-1]), generated_index, axis=-1)
    # Make a new array with the same length as the tokens array but with
    # each element replaced by the value at the corresponding index in the
    # logits array; e.g.
    # if logits is [77, 5, 3, 98] and tokens is [0, 1, 2, 3, 2, 3, 1],
    # then penalty_logits will be [77, 5, 3, 98, 3, 98, 5]
    penalty_logits = jnp.take(logits, tokens)
    # Repetition penalty slope
    def apply_slope(carry):
        repetition_penalty, rprange = carry
        _penalty = (penalty_arange/(rprange - 1)) * 2 - 1
        _penalty = (rpslope * _penalty) / (1 + jnp.abs(_penalty) * (rpslope - 1))
        _penalty = 1 + ((_penalty + 1) / 2) * (repetition_penalty - 1)
        return _penalty
    repetition_penalty = jax.lax.cond(
        (rpslope != 0.0) & (rprange > 0),  # Not a typo; do not use `and` here, it makes JAX crash
        apply_slope,
        lambda carry: jnp.full(tokens.shape, carry[0]),
        (repetition_penalty, rprange),
    )
    # Divide positive values by repetition_penalty and multiply negative
    # values by repetition_penalty (the academic publication that described
    # this technique actually just only divided, but that would cause tokens
    # with negative logits to become more likely, which is obviously wrong)
    penalty_logits = jnp.where(
        penalty_arange >= 0,
        jnp.where(
            penalty_logits > 0,
            penalty_logits/repetition_penalty,
            penalty_logits*repetition_penalty,
        ),
        penalty_logits,
    )
    # Finally, put those penalized logit values back into their original
    # positions in the logits array
    return logits.at[tokens].set(penalty_logits)

def kobold_sample_static(key, logits, top_p=0.9, temp=0.5, top_k=0, tfs=1.0):
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
    return jax.random.categorical(key, logits, -1).astype(jnp.uint32)

pad_token_id = 50256

def sample_func(data, key, numseqs_aux, badwords, repetition_penalty, generated_index, gen_length, rpslope, rprange, sampler_options):
    numseqs = numseqs_aux.shape[0]
    gi = data[0][1]
    def sample_loop_fn(carry):
        generated, generated_index, logits, _ = carry[0][0]
        sample_key = carry[1]
        # Get the pseudo-random number generator key that will
        # be used by kobold_sample_dynamic to randomly pick a token
        sample_key, new_key = jax.random.split(sample_key, num=2)
        # Apply repetition penalty to all tokens that are
        # currently inside the "generated" array
        logits = apply_repetition_penalty_dynamic(
            logits,
            generated,
            repetition_penalty,
            generated_index, 
            gen_length,
            rpslope,
            rprange,
        )
        # Remove any tokens in the badwords list by setting
        # their logits to negative infinity which effectively
        # makes their probabilities of being chosen zero
        logits[badwords] = -np.inf
        # Use the sampler (kobold_sample_dynamic) to pick one token
        # based on the logits array as a 0D uint32 array
        # (higher logit means higher probability of being
        # picked, non-linearly)
        next_token = kobold_sample_dynamic(
            sample_key,
            logits,
            **sampler_options,
        )
        # Remember what token was picked
        generated[generated_index] = next_token
        generated_index += 1
        # Re-pack the current sample_loop_fn's state so we can
        # get back the same variables the next time
        carry[0][0] = [generated, generated_index, logits, next_token]
        carry[0].append(carry[0].pop(0))
        return carry[0], new_key
    # return jax.lax.while_loop(
    #     lambda carry: carry[0][0][1] == gi,
    #     sample_loop_fn,
    #     (data, key),
    # )
    carry = (data, key)
    while carry[0][0][1] == gi:
        carry = sample_loop_fn(carry)
    return carry

class PenalizingCausalTransformer(CausalTransformer):
    def __init__(self, config, **kwargs):
        # Initialize
        super().__init__(config, **kwargs)
        def generate_static(state, key, ctx, ctx_length, gen_length, numseqs_aux, sampler_options, soft_embeddings=None):
            compiling_callback()
            numseqs = numseqs_aux.shape[0]
            # These are the tokens that we don't want the AI to ever write
            self.badwords = jnp.array(vars.badwordsids).squeeze()
            @hk.transform
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
                rpslope = sampler_options.pop('rpslope', None)
                rprange = sampler_options.pop('rprange', None)
                # This is the main generation loop
                def generate_loop_fn(carry):
                    # Unpack current generate_loop_fn state
                    generated, generated_index, sequence_index, next_token, decode_state = carry[0][0]
                    sample_key = carry[1]
                    # Get the pseudo-random number generator key that will
                    # be used by kobold_sample_static to randomly pick a token
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
                        logits = apply_repetition_penalty_static(
                            logits,
                            generated,
                            repetition_penalty,
                            generated_index,
                            gen_length,
                            rpslope,
                            rprange,
                        )
                    # Remove any tokens in the badwords list by setting
                    # their logits to negative infinity which effectively
                    # makes their probabilities of being chosen zero
                    logits = logits.at[self.badwords].set(-jnp.inf)
                    # Use the sampler (kobold_sample_static) to pick one token
                    # based on the logits array as a 0D uint32 array
                    # (higher logit means higher probability of being
                    # picked, non-linearly)
                    next_token = kobold_sample_static(
                        sample_key,
                        logits,
                        **sampler_options,
                    )
                    # Remember what token was picked
                    generated = generated.at[generated_index].set(next_token)
                    generated_index += 1
                    # Re-pack the current generate_loop_fn's state so we can
                    # get back the same variables the next time
                    carry[0][0] = (generated, generated_index, sequence_index, next_token[jnp.newaxis], new_state)
                    carry[0].append(carry[0].pop(0))
                    return carry[0], new_key
                return jax.lax.while_loop(
                    lambda carry: carry[0][0][1] - config["seq"] < gen_length,
                    generate_loop_fn,
                    (initial_states, sample_key),
                )
            return generate_sample.apply(state["params"], key, ctx, ctx_length)
        self.generate_static_xmap = jax.experimental.maps.xmap(
            fun=generate_static,
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
        def generate_initial(state, key, ctx, ctx_length, numseqs_aux, soft_embeddings=None):
            compiling_callback()
            numseqs = numseqs_aux.shape[0]
            @hk.transform
            def generate_initial_inner(context, ctx_length):
                # Give the initial context to the transformer
                transformer = CausalTransformerShard(config)
                def generate_initial_scan_fn(sequence_index, c):
                    _, initial_state = transformer.generate_initial(c, ctx_length, soft_embeddings=soft_embeddings)
                    generated_index = config["seq"]
                    # Add that information to generate_loop_fn's starting state
                    initial_state = (jnp.empty(config["n_vocab"], dtype=jnp.float32), generated_index, sequence_index) + initial_state
                    return sequence_index+1, initial_state
                _, initial_states = jax.lax.scan(generate_initial_scan_fn, 0, context, numseqs)
                sample_key = initial_states[-1][0]
                initial_states = list(list(jax.tree_map(lambda x: x[i], initial_states[:-1])) for i in range(numseqs))
                return initial_states, sample_key
            return generate_initial_inner.apply(state["params"], key, ctx, ctx_length)
        self.generate_initial_xmap = jax.experimental.maps.xmap(
            fun=generate_initial,
            in_axes=(
                ["shard", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["batch", ...],
                ["shard", ...],
            ),
            out_axes=["shard", "batch", ...],
            axis_resources={'shard': 'mp', 'batch': 'dp'},
        )
        def generate_once(data, state, numseqs_aux, soft_embeddings=None):
            numseqs = numseqs_aux.shape[0]
            @hk.without_apply_rng
            @hk.transform
            def generate_once_inner():
                gi = data[0][1]
                # Give the initial context to the transformer
                transformer = CausalTransformerShard(config)
                # This is the main generation loop
                def generate_loop_fn(carry):
                    # Unpack current generate_loop_fn state
                    _, generated_index, sequence_index, next_token, decode_state = carry[0][0]
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
                    assert logits.dtype == jnp.float32
                    # Flatten it into a 1D array to make it easier to use
                    logits = logits[0]
                    # Re-pack the current generate_loop_fn's state so we can
                    # get back the same variables the next time
                    generated_index += 1
                    carry[0][0] = [logits, generated_index, sequence_index, next_token, new_state]
                    carry[0].append(carry[0].pop(0))
                    return carry[0],
                return jax.lax.while_loop(
                    lambda carry: carry[0][0][1] == gi,
                    generate_loop_fn,
                    (data,),
                )
            return generate_once_inner.apply(state["params"])
        self.generate_once_xmap = jax.experimental.maps.xmap(
            fun=generate_once,
            in_axes=(
                ["shard", "batch", ...],
                ["shard", ...],
                ["batch", ...],
                ["shard", ...],
            ),
            out_axes=["shard", "batch", ...],
            axis_resources={'shard': 'mp', 'batch': 'dp'},
        )
    def generate_dynamic(self, ctx, ctx_length, gen_length, numseqs, return_logits=False, soft_embeddings=None, excluded_world_info=None, use_callback=True):
        assert excluded_world_info is not None
        assert not return_logits
        assert gen_length.ndim == 1
        assert soft_embeddings is not None
        key = hk.PRNGSequence(random.randint(0, 2 ** 60))
        batch_size = ctx.shape[0]
        self.batch_size = batch_size
        _numseqs_aux = jnp.empty((batch_size, numseqs), dtype=np.uint32)
        numseqs_aux = batch_xmap(_numseqs_aux)
        sample_data = [
            [
                np.pad(ctx[0][i], (0, params["seq"]), constant_values=pad_token_id),
                params["seq"],
                None,
                np.empty((), dtype=np.uint32),
            ]
            for i in range(numseqs)
        ]
        n_generated = 0
        regeneration_required = False
        halt = False
        started_compiling_callback()
        generate_data, sample_key = self.generate_initial_xmap(self.state, jnp.array(key.take(batch_size)), ctx, ctx_length, numseqs_aux, soft_embeddings)
        sample_key = np.asarray(sample_key[0, 0])
        while True:
            generate_data, = self.generate_once_xmap(generate_data, self.state, numseqs_aux, soft_embeddings)
            for i in range(numseqs):
                sample_data[i][2] = np.array(generate_data[i][0][0, 0], copy=True)
            if use_callback:
                logits = np.float32(tuple(d[2] for d in sample_data))
                logits = warper_callback(logits)
                for i in range(numseqs):
                    sample_data[i][2] = logits[i]
            sampler_options = settings_callback()
            repetition_penalty = sampler_options.pop("repetition_penalty", 1.0)
            rpslope = sampler_options.pop("rpslope", 0.0)
            rprange = sampler_options.pop("rprange", 0)
            sample_data, sample_key = sample_func(sample_data, sample_key, _numseqs_aux, badwords, repetition_penalty, params["seq"] + n_generated, gen_length, rpslope, rprange, sampler_options)
            n_generated += 1
            for i in range(numseqs):
                generate_data[i][3] = np.tile(sample_data[i][0][sample_data[i][1]-1][np.newaxis, np.newaxis], (params["cores_per_replica"], 1, 1))
            if use_callback:
                generated = np.uint32(tuple(d[0] for d in sample_data))
                excluded_world_info, regeneration_required, halt = stopping_callback(generated, n_generated, excluded_world_info)
                if regeneration_required or halt:
                    break
            else:
                break
        stopped_compiling_callback()
        return sample_data, n_generated, regeneration_required, halt
    def generate_static(self, ctx, ctx_length, gen_length, numseqs, sampler_options, return_logits=False, soft_embeddings=None):
        assert not return_logits
        key = hk.PRNGSequence(random.randint(0, 2 ** 60))
        batch_size = ctx.shape[0]
        self.batch_size = batch_size
        started_compiling_callback()
        result = self.generate_static_xmap(
            self.state,
            jnp.array(key.take(batch_size)),
            ctx,
            np.array(ctx_length, dtype=np.uint32),
            np.array(gen_length, dtype=np.uint32),
            np.empty((batch_size, numseqs), dtype=np.uint8),
            sampler_options,
            soft_embeddings,
        )
        stopped_compiling_callback()
        return result


def infer_dynamic(
    context: np.array,
    numseqs=1,
    gen_len=80,
    soft_embeddings: Optional[np.array] = None,
    soft_tokens: Optional[np.array] = None,
    excluded_world_info = None,
    use_callback=True,
) -> Tuple[List[np.array], int, bool, bool]:
    assert excluded_world_info is not None
    maps.thread_resources.env = thread_resources_env
    total_batch = 1
    tokens = context
    if(soft_tokens is not None):
        tokens = np.uint32(np.concatenate((np.tile(soft_tokens, (tokens.shape[0], 1)), tokens), axis=-1))
    provided_ctx = tokens.shape[-1]
    pad_amount = seq - provided_ctx
    padded_tokens = np.pad(tokens, ((0, 0), (pad_amount, 0)), constant_values=pad_token_id)
    batched_tokens = np.array([padded_tokens] * total_batch)
    samples = []
    output = network.generate_dynamic(
        batched_tokens,
        np.ones(total_batch, dtype=np.uint32) * provided_ctx,
        np.ones(total_batch, dtype=np.uint32) * gen_len,
        numseqs,
        soft_embeddings=soft_embeddings,
        excluded_world_info=excluded_world_info,
        use_callback=use_callback,
    )
    for out in output[0]:
        samples.append(out[0][params["seq"] : params["seq"] + gen_len])
    return (samples,) + output[1:]

def infer_static(
    context: np.array,
    top_p=0.9,
    temp=0.5,
    top_k=0,
    tfs=1.0,
    repetition_penalty=1.0,
    rpslope=0.0,
    rprange=0,
    numseqs=1,
    gen_len=80,
    soft_embeddings: Optional[np.array] = None,
    soft_tokens: Optional[np.array] = None,
) -> List[np.array]:
    maps.thread_resources.env = thread_resources_env
    total_batch = 1
    tokens = context
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
        "rpslope": rpslope * np.ones(total_batch),
        "rprange": np.full(total_batch, rprange, dtype=np.uint32),
        "top_k": np.full(total_batch, top_k, dtype=np.uint32)
    }
    output = network.generate_static(
        batched_tokens,
        np.ones(total_batch, dtype=np.uint32) * provided_ctx,
        np.ones(total_batch, dtype=np.uint32) * gen_len,
        numseqs,
        batched_generator_params,
        soft_embeddings=soft_embeddings,
    )[0]
    for o in output:
        samples.append(o[0][0, 0, params["seq"] : params["seq"] + gen_len])
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
        "tokenizer_class": "GPT2TokenizerFast",
        "tokenizer": "gpt2",
    }
    params = kwargs
    if "compat" in params:
        default_params["compat"] = params["compat"]
    if default_params["compat"] == "fairseq_lm":
        default_params["tokenizer"] = "KoboldAI/fairseq-dense-125M"
    for param in default_params:
        if param not in params:
            params[param] = default_params[param]

    # Load tokenizer
    if not isinstance(params["tokenizer_class"], str) or not any(params["tokenizer_class"].endswith(s) for s in ("Tokenizer", "TokenizerFast")):
        raise ValueError("`tokenizer_class` must be a string ending in 'Tokenizer' or 'TokenizerFast'")
    tokenizer_class = getattr(__import__("transformers"), params["tokenizer_class"])
    tokenizer = tokenizer_class.from_pretrained(params["tokenizer"])

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

    global shard_xmap, batch_xmap
    shard_xmap = __shard_xmap()
    batch_xmap = __batch_xmap(shard_dim=cores_per_replica)

    global badwords
    # These are the tokens that we don't want the AI to ever write
    badwords = jnp.array(vars.badwordsids).squeeze()

    if not path.endswith("/"):
        path += "/"

    network = PenalizingCausalTransformer(params, dematerialized=True)
    network.state = read_ckpt_lowmem(network.state, path, devices.shape[1])
    network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))
