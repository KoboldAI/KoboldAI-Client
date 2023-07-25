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

import utils

import multiprocessing
import threading
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TypeVar
import progressbar
import time
import os
import sys
import json
import zipfile
import requests
import random
import jax
import jax.dlpack
from jax.config import config
from jax.experimental import maps
import jax.numpy as jnp
import numpy as np
import haiku as hk
from transformers import AutoTokenizer, GPT2Tokenizer, AutoModelForCausalLM, GPTNeoForCausalLM
from tokenizers import Tokenizer
from mesh_transformer.checkpoint import read_ckpt_lowmem
from mesh_transformer.transformer_shard import CausalTransformer, CausalTransformerShard, PlaceholderTensor
from mesh_transformer.util import to_bf16
import time

import modeling.warpers as warpers

socketio = None

params: Dict[str, Any] = {}

__seed = random.randrange(2**64)
rng = random.Random(__seed)


def get_rng_seed():
    return __seed

def set_rng_seed(seed: int):
    global __seed, rng
    rng = random.Random(seed)
    __seed = seed
    return seed

def randomize_rng_seed():
    return set_rng_seed(random.randrange(2**64))

def get_rng_state():
    return rng

def set_rng_state(state):
    global rng
    rng = state

def new_rng_state(seed: int):
    return random.Random(seed)

def warper_callback(logits) -> np.array:
    raise NotImplementedError("`tpu_mtj_backend.warper_callback()` needs to be defined")

def stopping_callback(generated, n_generated) -> Tuple[bool, bool]:
    raise NotImplementedError("`tpu_mtj_backend.stopping_callback()` needs to be defined")

def settings_callback() -> dict:
    return {
        "sampler_order": utils.default_sampler_order.copy(),
        "top_p": 0.9,
        "temp": 0.5,
        "top_k": 0,
        "tfs": 1.0,
        "typical": 1.0,
        "top_a": 0.0,
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


def show_spinner(queue):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength, widgets=[progressbar.Timer(), '  ', progressbar.BouncingBar(left='[', right=']', marker='█')])
    i = 0
    while True:
        if i % 2 == 0:
            queue.put(["from_server", {'cmd': 'model_load_status', 'data': "Connecting to TPU..." }, {"broadcast":True, "room":"UI_1"}])
        else:
            queue.put(["from_server", {'cmd': 'model_load_status', 'data': "Connecting to TPU...." }, {"broadcast":True, "room":"UI_1"}])
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


class _EmptyState(NamedTuple):
    pass

class _DummyOptimizer:
    def init(*args, **kwargs):
        return _EmptyState()


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
    if koboldai_vars.use_alt_rep_pen:
        penalty_logits = np.where(
            penalty_arange >= 0,
            penalty_logits - np.log(repetition_penalty),
            penalty_logits,
        )

    else:
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

def kobold_sample_dynamic(key, logits, rpargs, sampler_order: Optional[np.ndarray] = None, top_p=0.9, temp=0.5, top_k=0, tfs=1.0, typical=1.0, top_a=0.0):
    '''
    This gets called by generate_loop_fn to apply a series of 6 filters
    to the logits (top-k, then top-a, then top-p, then TFS, then typical, then temperature)
    before picking one token using the modified logits
    '''
    for sid in jnp.array(sampler_order, int):
        sid = int(sid)
        warper = warpers.Warper.from_id(sid)

        if not warper.value_is_valid():
            continue

        # Repetition Penalty needs more info about the context
        if warper == warpers.RepetitionPenalty:
            logits = warper.jax_dynamic(logits, *rpargs)
        else:
            logits = warper.jax_dynamic(logits)

    # Finally, pick one token using the softmax thingy again (it gives
    # an array whose elements sum to 1 so it can be used nicely as a
    # probability distribution)
    return jax.random.categorical(key, logits, -1).astype(np.uint32)

def kobold_sample_static(
    key,
    logits,
    rpargs,
    sampler_order: Optional[np.ndarray] = None,
    top_p=0.9,
    temp=0.5,
    top_k=0,
    tfs=1.0,
    typical=1.0,
    top_a=0.0,
):
    '''
    This gets called by generate_loop_fn to apply a series of 6 filters
    to the logits (top-k, then top-a, then top-p, then TFS, then typical, then temperature)
    before picking one token using the modified logits
    '''

    # Lame to have these here instead of modeling/warpers.py but JAX JIT stuff >:(
    # For documentation see modeling/warpers.py
    def sample_top_k(scores: jnp.array) -> jnp.array:
        sorted_indices_to_remove = jnp.arange(len(scores)) >= top_k

        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(-scores),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, scores)


    def sample_top_a(scores: jnp.array) -> jnp.array:
        probabilities = jax.nn.softmax(scores)
        probs_max = probabilities.max()
        return jnp.where(
            probabilities < probs_max * probs_max * top_a, -jnp.inf, scores
        )


    def sample_top_p(scores: jnp.array) -> jnp.array:
        sorted_logits = -jnp.sort(-scores)
        probabilities = jax.nn.softmax(sorted_logits)

        cumulative_probabilities = jnp.cumsum(probabilities, axis=-1)
        sorted_indices_to_remove = cumulative_probabilities > top_p

        sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(-scores),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, scores)


    def sample_tail_free(scores: jnp.array) -> jnp.array:
        sorted_logits = -jnp.sort(-scores)
        probabilities = jax.nn.softmax(sorted_logits)

        d2 = jnp.diff(jnp.diff(probabilities))
        d2 = jnp.abs(d2)
        d2 = d2 / d2.sum(axis=-1, keepdims=True)

        cumulative_d2 = jnp.cumsum(d2, axis=-1)
        sorted_indices_to_remove = cumulative_d2 > tfs
        sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
        sorted_indices_to_remove = jnp.pad(
            sorted_indices_to_remove,
            (0, 2),
            constant_values=True,
        )

        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(-scores),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, scores)


    def sample_typical(scores: jnp.array) -> jnp.array:
        probs = jax.nn.softmax(scores)
        log_probs = jnp.log(probs)

        neg_entropy = jnp.nansum(probs * log_probs, axis=-1, keepdims=True)
        entropy_deviation = jnp.abs(neg_entropy - log_probs)

        _, sorted_logits = jax.lax.sort_key_val(entropy_deviation, probs)
        sorted_indices_to_remove = jnp.cumsum(sorted_logits, axis=-1) >= typical
        sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)

        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(entropy_deviation),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, scores)


    def sample_temperature(scores: jnp.array) -> jnp.array:
        return scores / temp


    def sample_repetition_penalty(
        logits: jnp.array,
        tokens: jnp.array,
        repetition_penalty,
        generated_index,
        rpslope,
        rprange
    ) -> jnp.array:
        """
        This gets called to apply repetition penalty to the 1D array logits
        using the provided 1D array of tokens to penalize
        """
        rpslope = jnp.int32(rpslope)
        rprange = jnp.int32(rprange)

        clipped_rprange = jax.lax.cond(
            rprange > 0, lambda x: x, lambda x: tokens.shape[-1], rprange
        )

        penalty_arange = jnp.roll(
            jnp.arange(tokens.shape[-1]) + (clipped_rprange - tokens.shape[-1]),
            generated_index,
            axis=-1,
        )
        # Make a new array with the same length as the tokens array but with
        # each element replaced by the value at the corresponding index in the
        # logits array; e.g.
        # if logits is [77, 5, 3, 98] and tokens is [0, 1, 2, 3, 2, 3, 1],
        # then penalty_logits will be [77, 5, 3, 98, 3, 98, 5]
        penalty_logits = jnp.take(logits, tokens)

        # Repetition penalty slope
        def apply_slope(carry):
            repetition_penalty, rprange = carry
            _penalty = (penalty_arange / (rprange - 1)) * 2 - 1
            _penalty = (rpslope * _penalty) / (1 + jnp.abs(_penalty) * (rpslope - 1))
            _penalty = 1 + ((_penalty + 1) / 2) * (repetition_penalty - 1)
            return _penalty

        repetition_penalty = jax.lax.cond(
            (rpslope != 0.0)
            & (rprange > 0),  # Not a typo; do not use `and` here, it makes JAX crash
            apply_slope,
            lambda carry: jnp.full(tokens.shape, carry[0]),
            (repetition_penalty, rprange),
        )
        # Divide positive values by repetition_penalty and multiply negative
        # values by repetition_penalty (the academic publication that described
        # this technique actually just only divided, but that would cause tokens
        # with negative logits to become more likely, which is obviously wrong)
        if koboldai_vars.use_alt_rep_pen:
            penalty_logits = jnp.where(
                penalty_arange >= 0,
                penalty_logits - jnp.log(repetition_penalty),
                penalty_logits,
            )
        else:
            penalty_logits = jnp.where(
                penalty_arange >= 0,
                jnp.where(
                    penalty_logits > 0,
                    penalty_logits / repetition_penalty,
                    penalty_logits * repetition_penalty,
                ),
                penalty_logits,
            )
        # Finally, put those penalized logit values back into their original
        # positions in the logits array
        return logits.at[tokens].set(penalty_logits)


    for k in sampler_order:
        logits = jax.lax.cond(jnp.logical_and(k == 0, top_k > 0), sample_top_k, lambda x: x, logits)
        logits = jax.lax.cond(jnp.logical_and(k == 1, top_a > 0.0), sample_top_a, lambda x: x, logits)
        logits = jax.lax.cond(jnp.logical_and(k == 2, top_p < 1.0), sample_top_p, lambda x: x, logits)
        logits = jax.lax.cond(jnp.logical_and(k == 3, tfs < 1.0), sample_tail_free, lambda x: x, logits)
        logits = jax.lax.cond(jnp.logical_and(k == 4, typical < 1.0), sample_typical, lambda x: x, logits)
        logits = jax.lax.cond(jnp.logical_and(k == 5, temp != 1.0), sample_temperature, lambda x: x, logits)
        logits = jax.lax.cond(jnp.logical_and(k == 6, rpargs[1] != 1.0), lambda x: sample_repetition_penalty(*x), lambda x: x[0], (logits, *rpargs))
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
            (
                generated,
                generated_index, 
            ),
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
    def __init__(self, badwordsids, config, **kwargs):
        # Initialize
        super().__init__(config, **kwargs)
        def generate_static(state, key, ctx, ctx_length, gen_length, numseqs_aux, sampler_options, soft_embeddings=None):
            compiling_callback()
            numseqs = numseqs_aux.shape[0]
            # These are the tokens that we don't want the AI to ever write
            badwords = jnp.array(badwordsids).squeeze()
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
                    # Remove any tokens in the badwords list by setting
                    # their logits to negative infinity which effectively
                    # makes their probabilities of being chosen zero
                    logits = logits.at[badwords].set(-jnp.inf)
                    # Use the sampler (kobold_sample_static) to pick one token
                    # based on the logits array as a 0D uint32 array
                    # (higher logit means higher probability of being
                    # picked, non-linearly)
                    next_token = kobold_sample_static(
                        sample_key,
                        logits,
                        (
                            generated,
                            repetition_penalty,
                            generated_index,
                            rpslope,
                            rprange,
                        ),
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
    def generate_dynamic(self, ctx, ctx_length, gen_length, numseqs, return_logits=False, soft_embeddings=None, use_callback=True):
        assert not return_logits
        assert gen_length.ndim == 1
        assert soft_embeddings is not None
        key = hk.PRNGSequence(rng.randint(0, 2 ** 60))
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
                regeneration_required, halt = stopping_callback(generated, n_generated)
                if regeneration_required or halt:
                    break
            else:
                break
        stopped_compiling_callback()
        return sample_data, n_generated, regeneration_required, halt
    def generate_static(self, ctx, ctx_length, gen_length, numseqs, sampler_options, return_logits=False, soft_embeddings=None):
        assert not return_logits
        key = hk.PRNGSequence(rng.randint(0, 2 ** 60))
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
    use_callback=True,
) -> Tuple[List[np.array], int, bool, bool]:
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
    typical=1.0,
    top_a=0.0,
    repetition_penalty=1.0,
    rpslope=0.0,
    rprange=0,
    numseqs=1,
    gen_len=80,
    soft_embeddings: Optional[np.array] = None,
    soft_tokens: Optional[np.array] = None,
    sampler_order: Optional[List[int]] = None,
) -> List[np.array]:
    maps.thread_resources.env = thread_resources_env
    if sampler_order is None:
        sampler_order = utils.default_sampler_order.copy()
    sampler_order = sampler_order[:]
    if len(sampler_order) < 7:  # Add repetition penalty at beginning if it's not present
        sampler_order = [6] + sampler_order
    sampler_order = np.uint32(sampler_order)
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
        "sampler_order": np.repeat(sampler_order[np.newaxis], total_batch, axis=0),
        "temp": temp * np.ones(total_batch),
        "top_p": top_p * np.ones(total_batch),
        "tfs": tfs * np.ones(total_batch),
        "typical": typical * np.ones(total_batch),
        "top_a": top_a * np.ones(total_batch),
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


def reshard_reverse(x, total_shards, old_shape):
    assert len(x.shape) != 1
    if len(x.shape) == 2:
        if old_shape[1] == x.shape[1]:
            out = x[0:1].tile((total_shards, 1))
        else:
            out = x.reshape(old_shape)
    elif len(x.shape) == 3:
        if x.shape[0] * x.shape[2] == old_shape[2]:
            out = x.reshape(old_shape)
        elif x.shape[0] * x.shape[1] == old_shape[1]:
            out = x.reshape((old_shape[1], old_shape[0], old_shape[2])).permute((1, 0, 2))
        else:
            assert False
    else:
        assert False
    return out


def get_old_shape(t, total_shards, dim=2):
    if len(t.shape) == 2:
        shard_shape = t.shape
        if dim == 1:
            assert shard_shape[0] % total_shards == 0
            return (shard_shape[0] // total_shards, shard_shape[1])
        elif dim == 2:
            assert shard_shape[1] % total_shards == 0
            return (shard_shape[0], shard_shape[1] // total_shards)
        else:
            raise ValueError(f"Unsupported dim {dim}")
    if len(t.shape) == 1:
        assert t.shape[0] % total_shards == 0
        return (t.shape[0] // total_shards,)
    else:
        raise ValueError(f"Unsupported shape {t.shape}")


def read_neox_checkpoint(state, path, config, checkpoint_shards=2):
    assert config["cores_per_replica"] % checkpoint_shards == 0
    output_shards = config["cores_per_replica"] // checkpoint_shards

    import torch
    import torch.utils.dlpack
    import modeling.lazy_loader as lazy_loader
    from tqdm.auto import tqdm

    move_xmap = jax.experimental.maps.xmap(
        fun=lambda x, _: to_bf16(x),
        in_axes=(["shard", ...], ["batch", ...]),
        out_axes=["shard", ...],
        axis_resources={'shard': 'mp', 'batch': 'dp'}
    )

    path_template = os.path.join(path, "layer_{layer:02d}-model_{shard:02d}-model_states.pt")

    static_mapping = {
        "word_embeddings.weight": {"module": "embedding_shard/~/linear", "param": "w", "axis": 1},
        "final_linear.weight": {"module": "projection_shard/~/linear", "param": "w", "axis": 2},
        "norm.weight": {"module": "projection_shard/~/replicated_layer_norm", "param": "scale", "axis": None},
        "norm.bias": {"module": "projection_shard/~/replicated_layer_norm", "param": "offset", "axis": None},
    }

    layer_mapping = {
        "attention.query_key_value.weight": {"module": "combined_qkv", "param": "w", "axis": 2},
        "attention.query_key_value.bias": {"module": "combined_qkv", "param": "b", "axis": 1},
        "attention.dense.weight": {"module": "linear_3", "param": "w", "axis": 1},
        "attention.dense.bias": {"module": "linear_3", "param": "b", "axis": None},
        "mlp.dense_h_to_4h.weight": {"module": "linear_4", "param": "w", "axis": 2},
        "mlp.dense_h_to_4h.bias": {"module": "linear_4", "param": "b", "axis": 1},
        "mlp.dense_4h_to_h.weight": {"module": "linear_5", "param": "w", "axis": 1},
        "mlp.dense_4h_to_h.bias": {"module": "linear_5", "param": "b", "axis": None},
        "input_layernorm.weight": {"module": "replicated_layer_norm", "param": "scale", "axis": None},
        "input_layernorm.bias": {"module": "replicated_layer_norm", "param": "offset", "axis": None},
        "post_attention_layernorm.weight": {"module": "replicated_layer_norm_1", "param": "scale", "axis": None},
        "post_attention_layernorm.bias": {"module": "replicated_layer_norm_1", "param": "offset", "axis": None},
    }

    tqdm_length = len(static_mapping) + config["layers"]*len(layer_mapping)
    if socketio is None:
        bar = tqdm(total=tqdm_length, desc="Loading from NeoX checkpoint")
    else:
        bar = tqdm(total=tqdm_length, desc="Loading from NeoX checkpoint", file=utils.UIProgressBarFile(socketio.emit))
    koboldai_vars.status_message = "Loading TPU"
    koboldai_vars.total_layers = tqdm_length
    koboldai_vars.loaded_layers = 0

    for checkpoint_layer in range(config["layers"] + 5):
        if checkpoint_layer in (1, config["layers"] + 2):
            continue
        layer = checkpoint_layer - 2
        shards = []
        with lazy_loader.use_custom_unpickler(lazy_loader.RestrictedUnpickler):
            for checkpoint_shard in range(checkpoint_shards):
                shards.append(torch.load(path_template.format(layer=checkpoint_layer, shard=checkpoint_shard), map_location="cpu"))
        for key in shards[0]:
            if key == "attention.rotary_emb.inv_freq":
                continue
            elif key in static_mapping:
                target_module = "causal_transformer_shard/~/" + static_mapping[key]["module"]
                target_param = static_mapping[key]["param"]
                target_axis = static_mapping[key]["axis"]
            elif key in layer_mapping:
                target_module = f"causal_transformer_shard/~/layer_{layer}/~/" + layer_mapping[key]["module"]
                target_param = layer_mapping[key]["param"]
                target_axis = layer_mapping[key]["axis"]
            else:
                error = f"{repr(key)} not found in mapping"
                print("\n\nERROR: ", error, file=sys.stderr)
                raise RuntimeError(error)
            original_shape = shards[0][key].shape
            for checkpoint_shard in range(checkpoint_shards):
                if key in ("attention.dense.bias", "mlp.dense_4h_to_h.bias"):
                    shards[checkpoint_shard][key] /= output_shards
                if key != "word_embeddings.weight" and shards[checkpoint_shard][key].ndim == 2:
                    shards[checkpoint_shard][key] = shards[checkpoint_shard][key].T
                tensor = shards[checkpoint_shard][key]
                if target_axis is not None:
                    target_shape = (output_shards,) + get_old_shape(tensor, total_shards=output_shards, dim=target_axis)
                else:
                    target_shape = (output_shards, tensor.shape[0])
                shards[checkpoint_shard][key] = reshard_reverse(tensor.unsqueeze_(0), output_shards, target_shape)
            #print(key, ":", original_shape, "->", shards[0][key].shape)
            tensor = torch.cat([shards[s][key] for s in range(checkpoint_shards)], dim=0)
            target_shape = state["params"][target_module][target_param].shape
            if tensor.shape != target_shape:
                error = f"Weight {repr(key)} has shape {tensor.shape} in checkpoint but shape {target_shape} was requested by MTJ for {target_module} {target_param}"
                print("\n\nERROR: ", error, file=sys.stderr)
                raise RuntimeError(error)
            if tensor.dtype is torch.float16 or tensor.dtype is torch.float32:
                tensor = tensor.bfloat16()
            state["params"][target_module][target_param] = move_xmap(
                jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(tensor)).copy(),
                np.zeros(config["cores_per_replica"]),
            )
            bar.update(1)
            koboldai_vars.loaded_layers+=1
    for mk, mv in state["params"].items():
        for pk, pv in mv.items():
            if isinstance(pv, PlaceholderTensor):
                error = f"{mk} {pk} could not be found in the model checkpoint"
                print("\n\nERROR:  " + error, file=sys.stderr)
                raise RuntimeError(error)

    koboldai_vars.status_message = ""

import koboldai_settings

def load_model(path: str, model_type: str, badwordsids=koboldai_settings.badwordsids_default, driver_version="tpu_driver_20221109", hf_checkpoint=False, socketio_queue=None, initial_load=False, logger=None, **kwargs) -> None:
    global thread_resources_env, seq, tokenizer, network, params, pad_token_id

    if kwargs.get("pad_token_id"):
        pad_token_id = kwargs["pad_token_id"]
    elif kwargs.get("eos_token_id"):
        pad_token_id = kwargs["eos_token_id"]

    if not hasattr(koboldai_vars, "sampler_order") or not koboldai_vars.sampler_order:
        koboldai_vars.sampler_order = utils.default_sampler_order.copy()

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
        "tokenizer_class": "GPT2Tokenizer",
        "tokenizer": "gpt2",
    }
    params = kwargs

    if koboldai_vars.model == "TPUMeshTransformerGPTNeoX":
        default_params = {
            "compat": "neox",
            "layers": 44,
            "d_model": 6144,
            "n_heads": 64,
            "n_vocab": 50432,
            "n_vocab_padding": 0,
            "norm": "doublelayernorm",
            "pe": "neox_rotary",
            "pe_rotary_dims": 24,
            "seq": 2048,
            "cores_per_replica": 8,
            "tokenizer_class": "GPT2Tokenizer",
            "tokenizer": "gpt2",
        }


    # Try to convert HF config.json to MTJ config
    if hf_checkpoint:
        spec_path = os.path.join("maps", model_type + ".json")
        if not os.path.isfile(spec_path):
            raise NotImplementedError(f"Unsupported model type {repr(model_type)}")
        with open(spec_path) as f:
            lazy_load_spec = json.load(f)

        if "mtj_compat" in lazy_load_spec:
            params["compat"] = lazy_load_spec["mtj_compat"]
        if "mtj_pe" in lazy_load_spec:
            params["pe"] = lazy_load_spec["mtj_pe"]
        for k, v in lazy_load_spec.get("mtj_config_map", {}).items():
            if type(v) is not list:
                params[k] = params[v]
                continue
            for i in range(len(v)):
                if i == len(v) - 1:
                    params[k] = v[i]
                elif v[i] in params:
                    params[k] = params[v[i]]
                    break

        params["n_vocab"] = params["vocab_size"]

        if "activation_function" in params:
            params["activation"] = params["activation_function"]

        # Both the number of attention heads in the model and the embedding
        # dimension of the model need to be divisible by the number of TPU cores
        # that we use, and JAX also requires the number of TPU cores used to be
        # an even number if we're using more than one core, so logically we try
        # to pick the largest possible even number of TPU cores such that the
        # number of attention heads and embedding dimension are both divisible
        # by the number of TPU cores, and fall back to one core if an even
        # number of TPU cores is not possible.
        for c in (8, 6, 4, 2, 1):
            if 0 == params["n_heads"] % c == params.get("d_embed", params["d_model"]) % c:
                params["cores_per_replica"] = c
                break

        # The vocabulary size of the model also has to be divisible by the
        # number of TPU cores, so we pad the vocabulary with the minimum
        # possible number of dummy tokens such that it's divisible.
        params["n_vocab_padding"] = -(params["n_vocab"] % -params["cores_per_replica"])

    if "compat" in params:
        default_params["compat"] = params["compat"]
    if default_params["compat"] == "fairseq_lm":
        default_params["tokenizer"] = "KoboldAI/fairseq-dense-125M"
    for param in default_params:
        if param not in params:
            params[param] = default_params[param]

    # Use an optimization that will allow us to avoid one extra transpose operation
    if hf_checkpoint:
        params["transposed_linear"] = True

    # Load tokenizer
    if koboldai_vars.model == "TPUMeshTransformerGPTNeoX":
        tokenizer = Tokenizer.from_file(os.path.join(path, "20B_tokenizer.json"))
        def new_encode(old_encode):
            def encode(s, *args, **kwargs):
                return old_encode(s).ids
            return encode
        tokenizer.encode = new_encode(tokenizer.encode)
        tokenizer._koboldai_header = []
    elif not hf_checkpoint:
        if not isinstance(params["tokenizer_class"], str) or not any(params["tokenizer_class"].endswith(s) for s in ("Tokenizer", "TokenizerFast")):
            raise ValueError("`tokenizer_class` must be a string ending in 'Tokenizer' or 'TokenizerFast'")
        tokenizer_class = getattr(__import__("transformers"), params["tokenizer_class"])
        tokenizer = tokenizer_class.from_pretrained(params["tokenizer"])

    # Disable JAX warnings about these two functions having been renamed
    jax.host_count = jax.process_count
    jax.host_id = jax.process_index

    print("Connecting to your Colab instance's TPU", flush=True)
    old_ai_busy = koboldai_vars.aibusy
    koboldai_vars.status_message = "Connecting to TPU"
    if os.environ.get('COLAB_TPU_ADDR', '') != '':
        tpu_address = os.environ['COLAB_TPU_ADDR']  # Colab
    else:
        tpu_address = os.environ['TPU_NAME']  # Kaggle
    tpu_address = tpu_address.replace("grpc://", "")
    tpu_address_without_port = tpu_address.split(':', 1)[0]
    url = f'http://{tpu_address_without_port}:8475/requestversion/{driver_version}'
    def check_status(url, queue):
        requests.post(url)
        queue.put("Done")
        
    queue = multiprocessing.Queue()
    spinner = multiprocessing.Process(target=check_status, args=(url, queue))
    spinner.start()
    i = 0
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength, widgets=[progressbar.Timer(), '  ', progressbar.BouncingBar(left='[', right=']', marker='█')])
    while True:
        if not queue.empty():
            queue.get()
            break
        if i % 20 == 0:
        #    socketio.emit("from_server", {'cmd': 'model_load_status', 'data': "Connecting to TPU..." }, broadcast=True, room="UI_1")
            socketio_queue.put(["from_server", {'cmd': 'model_load_status', 'data': "Connecting to TPU..." }, {"broadcast":True, "room":"UI_1"}])
        elif i % 10 == 0:
        #    socketio.emit("from_server", {'cmd': 'model_load_status', 'data': "Connecting to TPU...." }, broadcast=True, room="UI_1")
           socketio_queue.put(["from_server", {'cmd': 'model_load_status', 'data': "Connecting to TPU...." }, {"broadcast":True, "room":"UI_1"}])
        bar.update(i)
        time.sleep(0.1)
        i += 1
        
    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = "grpc://" + tpu_address
    koboldai_vars.aibusy = old_ai_busy
    print()

    start_time = time.time()
    cores_per_replica = params["cores_per_replica"]
    seq = params["seq"]
    params["optimizer"] = _DummyOptimizer()
    print("to line 1246 {}s".format(time.time()-start_time))
    start_time = time.time()
    mesh_shape = (1, cores_per_replica)
    devices = jax.devices()
    devices = np.array(devices[:cores_per_replica]).reshape(mesh_shape)
    thread_resources_env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')), ())
    maps.thread_resources.env = thread_resources_env
    if initial_load:
        logger.message(f"KoboldAI has still loading your model but available at the following link for UI 1: {koboldai_vars.cloudflare_link}")
        logger.message(f"KoboldAI has still loading your model but available at the following link for UI 2: {koboldai_vars.cloudflare_link}/new_ui")
        logger.message(f"KoboldAI has still loading your model but available at the following link for KoboldAI Lite: {koboldai_vars.cloudflare_link}/lite")
        logger.message(f"KoboldAI has still loading your model but available at the following link for the API: [Loading Model...]")
        logger.message(f"While the model loads you can use the above links to begin setting up your session, for generations you must wait until after its done loading.")

    global badwords
    # These are the tokens that we don't want the AI to ever write
    badwords = jnp.array(badwordsids).squeeze()

    if not path.endswith("/"):
        path += "/"

    network = PenalizingCausalTransformer(badwordsids, params, dematerialized=True)

    if not hf_checkpoint and koboldai_vars.model != "TPUMeshTransformerGPTNeoX":
        network.state = read_ckpt_lowmem(network.state, path, devices.shape[1])
        #network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))
        return

    if koboldai_vars.model == "TPUMeshTransformerGPTNeoX":
        print("\n\n\nThis model has  ", f"{hk.data_structures.tree_size(network.state['params']):,d}".replace(",", " "), "  parameters.\n")
        read_neox_checkpoint(network.state, path, params)
        return

    # Convert from HF checkpoint

    move_xmap = jax.experimental.maps.xmap(
        fun=lambda x, _: to_bf16(x),
        in_axes=(["shard", ...], ["batch", ...]),
        out_axes=["shard", ...],
        axis_resources={'shard': 'mp', 'batch': 'dp'}
    )

    model_spec = {}
    for key, spec in lazy_load_spec.get("static_weights", {}).items():
        if spec.get("mtj") is not None:
            model_spec[key] = spec["mtj"].copy()
            model_spec[key]["module"] = "causal_transformer_shard/~/" + model_spec[key]["module"]
    for _key, spec in lazy_load_spec.get("layer_weights", {}).items():
        for layer in range(params["layers"]):
            if spec.get("mtj") is not None:
                key = _key.format(layer=layer)
                model_spec[key] = spec["mtj"].copy()
                model_spec[key]["module"] = "causal_transformer_shard/~/" + model_spec[key]["module"].format(layer=layer)

    import modeling.lazy_loader as lazy_loader
    import torch
    from tqdm.auto import tqdm
    import functools

    
    def callback(model_dict, f, **_):
        if callback.nested:
            return
        callback.nested = True
        try:
            if utils.current_shard == 0:
                print("\n\n\nThis model has  ", f"{hk.data_structures.tree_size(network.state['params']):,d}".replace(",", " "), "  parameters.\n")

            if utils.num_shards is None or utils.current_shard == 0:
                if utils.num_shards is not None:
                    num_tensors = len(utils.get_sharded_checkpoint_num_tensors(utils.from_pretrained_model_name, utils.from_pretrained_index_filename, **utils.from_pretrained_kwargs))
                else:
                    num_tensors = len(model_dict)

                if socketio is None:
                    utils.bar = tqdm(total=num_tensors, desc="Loading model tensors")
                else:
                    utils.bar = tqdm(total=num_tensors, desc="Loading model tensors", file=utils.UIProgressBarFile(socketio.emit))
                koboldai_vars.status_message = "Loading model"
                koboldai_vars.loaded_layers = 0
                koboldai_vars.total_layers = num_tensors

            if utils.num_shards is not None:
                utils.current_shard += 1

            for key in sorted(model_dict.keys(), key=lambda k: (model_dict[k].key, model_dict[k].seek_offset)):
                model_spec_key = max((k for k in model_spec.keys() if key.endswith(k)), key=len, default=None)

                # Some model weights are used by transformers but not by MTJ.
                # We have to materialize these weights anyways because
                # transformers will throw a tantrum otherwise.  To attain
                # the least possible memory usage, we create them as meta
                # tensors, which don't take up any actual CPU or TPU memory.
                if model_spec_key is None:
                    model_dict[key] = torch.empty(model_dict[key].shape, dtype=model_dict[key].dtype, device="meta")
                    utils.bar.update(1)
                    koboldai_vars.loaded_layers += 1
                    continue

                spec = model_spec[model_spec_key]
                transforms = set(spec.get("transforms", ()))

                if not isinstance(model_dict[key], lazy_loader.LazyTensor):
                    error = f"Duplicate key {repr(key)}"
                    print("\n\nERROR:  " + error, file=sys.stderr)
                    raise RuntimeError(error)

                tensor = model_dict[key].materialize(map_location="cpu")
                model_dict[key] = tensor.to("meta")

                # MTJ requires certain mathematical operations to be performed
                # on tensors in order for them to be in the correct format
                if "remove_first_two_rows" in transforms:
                    tensor = tensor[2:]
                if "divide_by_shards" in transforms:
                    tensor /= params["cores_per_replica"]
                if "vocab_pad" in transforms:
                    tensor = torch.nn.functional.pad(tensor, (0,) * (tensor.ndim * 2 - 1) + (params["n_vocab_padding"],))
                # We don't need to transpose linear module weights anymore because MTJ will do it for us if `transposed_linear` is set to True in the config
                #if "no_transpose" not in transforms and tensor.ndim == 2:
                #    tensor = tensor.T
                tensor.unsqueeze_(0)
                

                # Shard the tensor so that parts of the tensor can be used
                # on different TPU cores
                tensor = reshard_reverse(
                    tensor,
                    params["cores_per_replica"],
                    network.state["params"][spec["module"]][spec["param"]].shape,
                )
                tensor = tensor.detach()
                # numpy does not support bfloat16
                if tensor.dtype is torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                tensor = jnp.array(tensor)
                if tensor.dtype is torch.float16 or tensor.dtype is torch.float32:
                    tensor = tensor.bfloat16()
                network.state["params"][spec["module"]][spec["param"]] = move_xmap(
                    tensor,
                    np.empty(params["cores_per_replica"]),
                )
                
                koboldai_vars.loaded_layers += 1
                try:
                    time.sleep(0.01)
                except:
                    pass
                utils.bar.update(1)

            if utils.num_shards is not None and utils.current_shard < utils.num_shards:
                return

            # Check for tensors that MTJ needs that were not provided in the
            # HF model
            for mk, mv in network.state["params"].items():
                for pk, pv in mv.items():
                    if isinstance(pv, PlaceholderTensor):
                        # The transformers GPT-J models apparently do not
                        # have embedding bias, whereas MTJ GPT-J models do,
                        # so we have to supplement an embedding bias tensor
                        # by creating a tensor with the necessary shape, filled
                        # with zeros.
                        if mk == "causal_transformer_shard/~/embedding_shard/~/linear" and pk == "b":
                            mv[pk] = move_xmap(jnp.zeros(mv[pk].shape, dtype=jnp.bfloat16), np.empty(params["cores_per_replica"]))

                        else:
                            error = f"{mk} {pk} could not be found in the model checkpoint"
                            print("\n\nERROR:  " + error, file=sys.stderr)
                            raise RuntimeError(error)
        finally:
            if utils.num_shards is None or utils.current_shard >= utils.num_shards:
                utils.bar.close()
                utils.bar = None
                koboldai_vars.status_message = ""
            callback.nested = False
    callback.nested = False

    if os.path.isdir(koboldai_vars.model.replace('/', '_')):
        import shutil
        shutil.move(koboldai_vars.model.replace('/', '_'), "models/{}".format(koboldai_vars.model.replace('/', '_')))
    print("\n", flush=True)
    with lazy_loader.use_lazy_load(callback=callback, dematerialized_modules=True):
        if(os.path.isdir(koboldai_vars.custmodpth)):
            try:
                tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
            except Exception as e:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                except Exception as e:
                    try:
                        tokenizer = GPT2Tokenizer.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                    except Exception as e:
                        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
            try:
                model     = AutoModelForCausalLM.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
            except Exception as e:
                model     = GPTNeoForCausalLM.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
        elif(os.path.isdir("models/{}".format(koboldai_vars.model.replace('/', '_')))):
            try:
                tokenizer = AutoTokenizer.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
            except Exception as e:
                try:
                    tokenizer = AutoTokenizer.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
                except Exception as e:
                    try:
                        tokenizer = GPT2Tokenizer.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
                    except Exception as e:
                        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
            try:
                model     = AutoModelForCausalLM.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
            except Exception as e:
                model     = GPTNeoForCausalLM.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
            except Exception as e:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache")
                except Exception as e:
                    try:
                        tokenizer = GPT2Tokenizer.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache")
                    except Exception as e:
                        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
            try:
                model     = AutoModelForCausalLM.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache")
            except Exception as e:
                model     = GPTNeoForCausalLM.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache")

    #network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))
    global shard_xmap, batch_xmap
    shard_xmap = __shard_xmap()
    batch_xmap = __batch_xmap(shard_dim=cores_per_replica)
