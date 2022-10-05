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

import torch
from transformers import LogitsWarper


class AdvancedRepetitionPenaltyLogitsProcessor(LogitsWarper):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.penalty_range = int(self.penalty_range)
        clipped_penalty_range = min(input_ids.shape[-1], self.penalty_range)

        if self.penalty != 1.0:
            if self.penalty_range > 0:
                if clipped_penalty_range < input_ids.shape[1]:
                    input_ids = input_ids[..., -clipped_penalty_range:]

                if self.penalty_slope != 0:
                    _penalty = (torch.arange(self.penalty_range, dtype=scores.dtype, device=scores.device)/(self.penalty_range - 1)) * 2. - 1
                    _penalty = (self.penalty_slope * _penalty) / (1 + torch.abs(_penalty) * (self.penalty_slope - 1))
                    _penalty = 1 + ((_penalty + 1) / 2).unsqueeze(0) * (self.penalty - 1)
                    self.penalty = _penalty[..., -clipped_penalty_range:]

            score = torch.gather(scores, 1, input_ids)
            score = torch.where(score <= 0, score * self.penalty, score / self.penalty)
            scores.scatter_(1, input_ids, score)

        return scores


class TailFreeLogitsWarper(LogitsWarper):

    def __init__(self, tfs: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"`tfs` has to be a float >= 0 and <= 1, but is {tfs}")
        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.filter_value >= 1.0:
            return scores
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold (token with 0 are kept)
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs

        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TypicalLogitsWarper(LogitsWarper):
    '''
    Typical sampling, described in https://arxiv.org/pdf/2202.00666.pdf
    '''

    def __init__(self, typical: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        typical = float(typical)
        if typical < 0 or typical > 1.0:
            raise ValueError(f"`typical` has to be a float >= 0 and <= 1, but is {typical}")
        self.typical = typical
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.filter_value >= 1.0:
            return scores

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
        sorted_indices_to_remove = sorted_logits.cumsum(dim=-1) >= self.typical
        sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)

        min_tokens_to_keep = max(self.min_tokens_to_keep, 1)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., : min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopALogitsWarper(LogitsWarper):
    def __init__(self, top_a: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_a = float(top_a)
        if top_a < 0 or top_a > 1.0:
            raise ValueError(f"`top_a` has to be a float >= 0 and <= 1, but is {top_a}")
        self.top_a = top_a
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.filter_value >= 1.0:
            return scores

        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Remove tokens with probability less than top_a*(max(probs))^2 (token with 0 are kept)
        probs_max = probs[..., 0, None]
        sorted_indices_to_remove = probs < probs_max * probs_max * self.top_a

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

# Returns counts of tokens for given `scores`. If `count_pruned_tokens` is true, pruned tokens are included in the counts; otherwise pruned tokens are removed before counting.
def get_count(scores: torch.FloatTensor, keepdim: bool = False, count_pruned_tokens: bool = False) -> torch.FloatTensor:
    if count_pruned_tokens:
        count = scores.count_nonzero(dim=-1)
    else:
        count = scores.gt(negative_infinity).count_nonzero(dim=-1)
    if keepdim:
        sequences = scores.size(dim=0)
        return count.reshape(sequences, 1)
    else:
        return count
    
# Returns entropy values of given `scores`.
def get_entropy(scores: torch.FloatTensor, keepdim: bool = False) -> torch.FloatTensor:
    probs = scores.softmax(dim=-1)
    infos = probs.log().negative()
    return infos.multiply(probs).nansum(dim=-1, keepdim=keepdim)

# Returns normalized entropy values of given `scores`
# Normalized entropy is `entropy/max_entropy`, where `max_entropy` is `log(token_count)`
# If `count_pruned_tokens` is true, pruned tokens are included in determining max_entropy; otherwise pruned tokens are removed before calculating max_entropy.
def get_normalized_entropy(scores: torch.FloatTensor, keepdim: bool = False, count_pruned_tokens: bool = True) -> torch.FloatTensor:
    max_entropy = get_count(scores, keepdim, count_pruned_tokens).log()
    entropy = get_entropy(scores, keepdim)
    return entropy / max_entropy

# Returns the temperature values that, when applied to `scores`, will approximate the given `entropy` value
# If a temperature required to achieve the entropy is greater than `max_temp` 
# or less than `min_temp`, then `max_temp` or `min_temp` will be returned, 
# respectively
def get_temp_for_entropy(scores, entropy):
    max_temp = 8.0 # @param {type: "number"}
    min_temp = 0.001 # @param {type: "number"}
    max_error = 0.001 # @param {type: "number"}

    curr_entropy = get_entropy(scores, keepdim=True)

    count_equals_one = get_count(scores, keepdim=-1) == 1
    no_temp_needed = (curr_entropy - entropy).abs() <= max_error
    max_entropy = get_entropy(scores/max_temp, keepdim=True)
    use_max_temps = (max_entropy <= entropy)
    min_entropy = get_entropy(scores/min_temp, keepdim=True)
    use_min_temps = (min_entropy >= entropy)
    
    high_temps = torch.ones(curr_entropy.shape).to(scores.device) * max_temp
    high_temps = high_temps.masked_fill(use_max_temps, max_temp)
    high_temps = high_temps.masked_fill(use_min_temps, min_temp)
    high_temps = high_temps.masked_fill(no_temp_needed, 1.0)
    high_temps = high_temps.masked_fill(count_equals_one, 1.0)

    low_temps = torch.ones(curr_entropy.shape).to(scores.device) * min_temp
    low_temps = low_temps.masked_fill(use_max_temps, max_temp)
    low_temps = low_temps.masked_fill(use_min_temps, min_temp)
    low_temps = low_temps.masked_fill(no_temp_needed, 1.0)
    low_temps = low_temps.masked_fill(count_equals_one, 1.0)

    while True:
        temps = (high_temps + low_temps)/2.0
        estimates = get_entropy(scores/temps, keepdim=True)
        diffs = estimates - entropy

        # Get indices for high and low temps (temps within error range are in both)
        high_mask = (diffs >= -max_error)
        low_mask = (diffs <= max_error)

        high_temp_selection = temps.masked_select(high_mask)
        low_temp_selection = temps.masked_select(low_mask)

        high_temps = high_temps.masked_scatter(high_mask, high_temp_selection)
        low_temps = low_temps.masked_scatter(low_mask, low_temp_selection)

        assert (high_temps >= low_temps).all()

        if high_temps.eq(low_temps).all():
            break
    return temps

# Returns the temperature values that, when applied to `scores`, will approximate the given `normalized_entropy` value
# This is a wrapper for `get_temp_for_entropy` using normalized_entropy instead of entropy
# If `count_pruned_tokens` is True, pruned tokens are included in determining max_entropy; otherwise pruned tokens are removed before calculating max_entropy.
# Note: it works better when `count_pruned_tokens` is True
def get_temp_for_normalized_entropy(scores, normalized_entropy, count_pruned_tokens: bool = True):
    max_entropy = get_count(scores, keepdim=True, count_pruned_tokens=count_pruned_tokens).log()
    entropy = normalized_entropy * max_entropy
    return get_temp_for_entropy(scores, entropy)

class EntropyLogitsWarper(LogitsWarper):
    def __init__(self, target_normalized_entropy: float = 0.0):
        assert target_normalized_entropy >= 0
        assert target_normalized_entropy <= 1.0
        self.normalized_entropy = target_normalized_entropy

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        normalized_scores = scores.log_softmax(dim=-1)
        normalized_entropy = get_normalized_entropy(normalized_scores, keepdim=True)
        temps = get_temp_for_normalized_entropy(scores, self.target_normalized_entropy
        return scores.div(temps).log_softmax(dim=-1)
