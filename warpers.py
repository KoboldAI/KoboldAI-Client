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
