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
from transformers import LogitsWarper, LogitsProcessor


class AdvancedRepetitionPenaltyLogitsProcessor(LogitsProcessor):
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
            raise ValueError(f"`tfs` has to be a float > 0 and < 1, but is {tfs}")
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
