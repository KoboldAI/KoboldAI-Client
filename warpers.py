"""
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

---

Some of the code in this file is also from Hugging Face logitsTransformers:
https://github.com/huggingface/transformers

Transformers is licensed under the Apache-2.0 License. The changes made to this
file are mostly porting warper code to the torch methods.
"""
# Comments mostly taken from tpu_mtj_backend.py

from __future__ import annotations

import utils
import torch
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import tpu_mtj_backend
except ImportError as e:
    print(e)
    if utils.koboldai_vars.use_colab_tpu:
        raise e


def update_settings():
    # This feels like a bad way to structure this
    koboldai_vars = utils.koboldai_vars
    Temperature.temperature = koboldai_vars.temp
    TopP.top_p = koboldai_vars.top_p
    TopK.top_k = koboldai_vars.top_k
    TopA.top_a = koboldai_vars.top_a
    TailFree.tfs = koboldai_vars.tfs
    Typical.typical = koboldai_vars.typical
    RepetitionPenalty.rep_pen = koboldai_vars.rep_pen
    RepetitionPenalty.rep_pen_range = koboldai_vars.rep_pen_range
    RepetitionPenalty.rep_pen_slope = koboldai_vars.rep_pen_slope
    RepetitionPenalty.use_alt_rep_pen = koboldai_vars.use_alt_rep_pen


class Warper:
    @staticmethod
    def from_id(warper_id: int) -> Warper:
        return {
            0: TopK,
            1: TopA,
            2: TopP,
            3: TailFree,
            4: Typical,
            5: Temperature,
            6: RepetitionPenalty,
        }[warper_id]


class Temperature(Warper):
    """Temperature (just divide the logits by the temperature)"""

    temperature: float = 0.5

    @classmethod
    def torch(cls, scores: torch.Tensor) -> torch.Tensor:
        return scores / cls.temperature

    @classmethod
    def jax_dynamic(cls, scores: np.array) -> np.array:
        return scores / cls.temperature

    @classmethod
    def jax_static(cls, scores: jnp.array) -> jnp.array:
        return scores / cls.temperature

    @classmethod
    def value_is_valid(cls) -> bool:
        return cls.temperature != 1.0


class TopP(Warper):
    """
    Top-p (after sorting the remaining tokens again in descending order of
    logit, remove the ones that have cumulative softmax probability
    greater than p)
    """

    top_p: float = 0.9

    @classmethod
    def torch(cls, scores: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - cls.top_p)

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        return scores.masked_fill(indices_to_remove, -np.inf)

    @classmethod
    def jax_dynamic(cls, scores: np.array) -> np.array:
        # Sort the logits array in descending order, replace every element
        # with e (Euler's number) to the power of that element, and divide
        # each element of the new array by the sum of the elements in the
        # new array
        sorted_logits = -np.sort(-scores)
        probabilities = np.array(jax.nn.softmax(sorted_logits), copy=True)
        # Calculate cumulative_probabilities as the prefix-sum array of
        # probabilities
        cumulative_probabilities = np.cumsum(probabilities, axis=-1)
        # We want to remove tokens with cumulative probability higher
        # than top_p
        sorted_indices_to_remove = cumulative_probabilities > cls.top_p
        # Don't ever remove the token with the highest logit, even if
        # the probability is higher than top_p
        sorted_indices_to_remove[0] = False
        # Unsort and remove
        _, indices_to_remove = jax.lax.sort_key_val(
            np.argsort(-scores),
            sorted_indices_to_remove,
        )
        return np.where(indices_to_remove, -np.inf, scores)

    @classmethod
    def jax_static(cls, scores: jnp.array) -> jnp.array:
        # Sort the logits array in descending order, replace every element
        # with e (Euler's number) to the power of that element, and divide
        # each element of the new array by the sum of the elements in the
        # new array
        sorted_logits = -jnp.sort(-scores)
        probabilities = jax.nn.softmax(sorted_logits)
        # Calculate cumulative_probabilities as the prefix-sum array of
        # probabilities
        cumulative_probabilities = jnp.cumsum(probabilities, axis=-1)
        # We want to remove tokens with cumulative probability higher
        # than top_p
        sorted_indices_to_remove = cumulative_probabilities > cls.top_p
        # Don't ever remove the token with the highest logit, even if
        # the probability is higher than top_p
        sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)
        # Unsort and remove
        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(-scores),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, scores)

    @classmethod
    def value_is_valid(cls) -> bool:
        return cls.top_p < 1.0


class TopK(Warper):
    """
    Top-k (keep only the k tokens with the highest logits and remove the rest,
    by setting their logits to negative infinity)
    """

    top_k: int = 0

    @classmethod
    def torch(cls, scores: torch.Tensor) -> torch.Tensor:
        top_k = min(max(cls.top_k, 1), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, -np.inf)
        return scores

    @classmethod
    def jax_dynamic(cls, scores: np.array) -> np.array:
        # After sorting the logits array in descending order,
        # sorted_indices_to_remove is a 1D array that is True for tokens
        # in the sorted logits array we want to remove and False for ones
        # we want to keep, in this case the first top_k elements will be
        # False and the rest will be True
        sorted_indices_to_remove = np.arange(len(scores)) >= cls.top_k
        # Unsort the logits array back to its original configuration and
        # remove tokens we need to remove
        _, indices_to_remove = jax.lax.sort_key_val(
            np.argsort(-scores),
            sorted_indices_to_remove,
        )
        return np.where(indices_to_remove, -np.inf, scores)

    @classmethod
    def jax_static(cls, scores: jnp.array) -> jnp.array:
        sorted_indices_to_remove = jnp.arange(len(scores)) >= cls.top_k

        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(-scores),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, scores)

    @classmethod
    def value_is_valid(cls) -> bool:
        return cls.top_p > 0


class TailFree(Warper):
    """
    Tail free sampling (basically top-p a second time on remaining tokens except
    it's the "cumulative normalized absolute second finite differences of the
    softmax probabilities" instead of just the cumulative softmax probabilities)
    """

    tfs: float = 1.0

    @classmethod
    def torch(cls, scores: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold (token with 0 are kept)
        sorted_indices_to_remove = normalized_d2_cdf > cls.tfs

        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, -np.inf)
        return scores

    @classmethod
    def jax_dynamic(cls, scores: np.array) -> np.array:
        # Sort in descending order
        sorted_logits = -np.sort(-scores)

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
        sorted_indices_to_remove = cumulative_d2 > cls.tfs

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
            np.argsort(-scores),
            sorted_indices_to_remove,
        )
        return np.where(indices_to_remove, -np.inf, scores)

    @classmethod
    def jax_static(cls, scores: jnp.array) -> jnp.array:
        sorted_logits = -jnp.sort(-scores)
        probabilities = jax.nn.softmax(sorted_logits)

        d2 = jnp.diff(jnp.diff(probabilities))
        d2 = jnp.abs(d2)
        d2 = d2 / d2.sum(axis=-1, keepdims=True)

        cumulative_d2 = jnp.cumsum(d2, axis=-1)
        sorted_indices_to_remove = cumulative_d2 > cls.tfs
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

    @classmethod
    def value_is_valid(cls) -> bool:
        return cls.tfs < 1.0


class Typical(Warper):
    """Typical sampling, described in https://arxiv.org/pdf/2202.00666.pdf"""

    typical: float = 1.0

    @classmethod
    def torch(cls, scores: torch.Tensor) -> torch.Tensor:
        # Compute softmax probabilities and the natural logarithms of them
        probs = scores.softmax(dim=-1)
        log_probs = probs.log()

        # Compute the negative of entropy, which is the sum of p*ln(p) for all p
        # in the set of softmax probabilities of the logits
        neg_entropy = (probs * log_probs).nansum(dim=-1, keepdim=True)

        # Determine absolute difference between the negative entropy and the
        # log probabilities
        entropy_deviation = (neg_entropy - log_probs).abs()

        # Keep certain tokens such that the sum of the entropy_deviation of the
        # kept tokens is the smallest possible value such that the sum of the
        # softmax probabilities of the kept tokens is at least the threshold
        # value (by sorting the tokens in ascending order of entropy_deviation
        # and then keeping the smallest possible number of tokens from the
        # beginning such that sum of softmax probabilities is at or above the
        # threshold)
        _, sorted_indices = torch.sort(entropy_deviation)
        sorted_logits = probs.gather(-1, sorted_indices)
        sorted_indices_to_remove = sorted_logits.cumsum(dim=-1) >= cls.typical
        sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)

        min_tokens_to_keep = 1
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, -np.inf)
        return scores

    @classmethod
    def jax_dynamic(cls, scores: np.array) -> np.array:
        # Compute softmax probabilities and the natural logarithms of them
        probs = jax.nn.softmax(scores)
        with np.errstate(divide="ignore"):
            log_probs = np.log(probs)

        # Compute the negative of entropy, which is the sum of p*ln(p) for all p
        # in the set of softmax probabilities of the logits
        neg_entropy = np.nansum(probs * log_probs, axis=-1, keepdims=True)

        # Determine absolute difference between the negative entropy and the
        # log probabilities
        entropy_deviation = np.abs(neg_entropy - log_probs)

        # Keep certain tokens such that the sum of the entropy_deviation of the
        # kept tokens is the smallest possible value such that the sum of the
        # softmax probabilities of the kept tokens is at least the threshold
        # value (by sorting the tokens in ascending order of entropy_deviation
        # and then keeping the smallest possible number of tokens from the
        # beginning such that sum of softmax probabilities is at or above the
        # threshold)
        _, sorted_logits = jax.lax.sort_key_val(entropy_deviation, probs)
        sorted_indices_to_remove = np.cumsum(sorted_logits, axis=-1) >= cls.typical
        sorted_indices_to_remove = np.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove[0] = False

        # Unsort and remove
        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(entropy_deviation),
            sorted_indices_to_remove,
        )
        return np.where(indices_to_remove, -jnp.inf, scores)

    @classmethod
    def jax_static(cls, scores: jnp.array) -> jnp.array:
        probs = jax.nn.softmax(scores)
        log_probs = jnp.log(probs)

        neg_entropy = jnp.nansum(probs * log_probs, axis=-1, keepdims=True)
        entropy_deviation = jnp.abs(neg_entropy - log_probs)

        _, sorted_logits = jax.lax.sort_key_val(entropy_deviation, probs)
        sorted_indices_to_remove = jnp.cumsum(sorted_logits, axis=-1) >= cls.typical
        sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)

        _, indices_to_remove = jax.lax.sort_key_val(
            jnp.argsort(entropy_deviation),
            sorted_indices_to_remove,
        )
        return jnp.where(indices_to_remove, -jnp.inf, scores)

    @classmethod
    def value_is_valid(cls) -> bool:
        return cls.typical < 1.0


class TopA(Warper):
    """
    Top-a (remove all tokens that have softmax probability less than *m^2 where
    m is the maximum softmax probability)
    """

    top_a: float = 0.0

    @classmethod
    def torch(cls, scores: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Remove tokens with probability less than top_a*(max(probs))^2 (token with 0 are kept)
        probs_max = probs[..., 0, None]
        sorted_indices_to_remove = probs < probs_max * probs_max * cls.top_a

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, -np.inf)
        return scores

    @classmethod
    def jax_dynamic(cls, scores: np.array) -> np.array:
        # Replace every element in the logits array
        # with e (Euler's number) to the power of that element, and divide
        # each element of the new array by the sum of the elements in the
        # new array
        probabilities = np.array(jax.nn.softmax(scores), copy=True)
        # Find the largest probability
        probs_max = probabilities.max()
        # Remove tokens
        return np.where(
            probabilities < probs_max * probs_max * cls.top_a, -np.inf, scores
        )

    @classmethod
    def jax_static(cls, scores: jnp.array) -> jnp.array:
        probabilities = jax.nn.softmax(scores)
        probs_max = probabilities.max()
        return jnp.where(
            probabilities < probs_max * probs_max * cls.top_a, -jnp.inf, scores
        )

    @classmethod
    def value_is_valid(cls) -> bool:
        return cls.top_a > 0.0


class RepetitionPenalty(Warper):
    rep_pen: float = 1.0
    rep_pen_slope: float = 0.0
    rep_pen_range: int = 0
    use_alt_rep_pen: bool = False

    @classmethod
    def torch(cls, scores: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        cls.rep_pen_range = int(cls.rep_pen_range)
        clipped_penalty_range = min(input_ids.shape[-1], cls.rep_pen_range)

        if cls.rep_pen != 1.0:
            if cls.rep_pen_range > 0:
                if clipped_penalty_range < input_ids.shape[1]:
                    input_ids = input_ids[..., -clipped_penalty_range:]

                if cls.rep_pen_slope != 0:
                    _penalty = (
                        torch.arange(
                            cls.rep_pen_range, dtype=scores.dtype, device=scores.device
                        )
                        / (cls.rep_pen_range - 1)
                    ) * 2.0 - 1
                    _penalty = (cls.rep_pen_slope * _penalty) / (
                        1 + torch.abs(_penalty) * (cls.rep_pen_slope - 1)
                    )
                    _penalty = 1 + ((_penalty + 1) / 2).unsqueeze(0) * (cls.rep_pen - 1)
                    cls.rep_pen = _penalty[..., -clipped_penalty_range:]

            score = torch.gather(scores, 1, input_ids)
            if cls.use_alt_rep_pen:
                score = score - torch.log(cls.rep_pen)
            else:
                score = torch.where(
                    score <= 0, score * cls.rep_pen, score / cls.rep_pen
                )
            scores.scatter_(1, input_ids, score)

        return scores

    @classmethod
    def jax_static(
        cls,
        logits: jnp.array,
        tokens: jnp.array,
        generated_index,
    ) -> jnp.array:
        """
        This gets called by generate_loop_fn to apply repetition penalty
        to the 1D array logits using the provided 1D array of tokens to penalize
        """
        rpslope = jnp.int32(cls.rep_pen_slope)
        rprange = jnp.int32(cls.rep_pen_range)
        repetition_penalty = cls.rep_pen

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
        if cls.use_alt_rep_pen:
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

    @classmethod
    def jax_dynamic(
        cls,
        scores: jnp.array,
        tokens: jnp.array,
        generated_index,
    ) -> jnp.array:
        """
        This gets called by generate_loop_fn to apply repetition penalty
        to the 1D array logits using the provided 1D array of tokens to penalize
        """
        tokens = np.minimum(
            tokens, tpu_mtj_backend.params["n_vocab"] - 1
        )  # https://github.com/google/jax/issues/3774

        rpslope = np.int32(cls.rep_pen_slope)
        rprange = np.int32(cls.rep_pen_range)
        repetition_penalty = cls.rep_pen

        clipped_rprange = rprange if rprange > 0 else tokens.shape[-1]
        penalty_arange = np.roll(
            np.arange(tokens.shape[-1]) + (clipped_rprange - tokens.shape[-1]),
            generated_index,
            axis=-1,
        )
        # Make a new array with the same length as the tokens array but with
        # each element replaced by the value at the corresponding index in the
        # logits array; e.g.
        # if logits is [77, 5, 3, 98] and tokens is [0, 1, 2, 3, 2, 3, 1],
        # then penalty_logits will be [77, 5, 3, 98, 3, 98, 5]
        penalty_logits = np.take(scores, tokens)
        # Repetition penalty slope
        if rpslope != 0.0 and rprange > 0:
            _penalty = (penalty_arange / (rprange - 1)) * 2 - 1
            _penalty = (rpslope * _penalty) / (1 + np.abs(_penalty) * (rpslope - 1))
            _penalty = 1 + ((_penalty + 1) / 2) * (repetition_penalty - 1)
            repetition_penalty = _penalty
        # Divide positive values by repetition_penalty and multiply negative
        # values by repetition_penalty (the academic publication that described
        # this technique actually just only divided, but that would cause tokens
        # with negative logits to become more likely, which is obviously wrong)
        if cls.use_alt_rep_pen:
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
                    penalty_logits / repetition_penalty,
                    penalty_logits * repetition_penalty,
                ),
                penalty_logits,
            )
        # Finally, put those penalized logit values back into their original
        # positions in the logits array
        scores[tokens] = penalty_logits
        return scores

    @classmethod
    def value_is_valid(cls) -> bool:
        return cls.rep_pen != 1.0
