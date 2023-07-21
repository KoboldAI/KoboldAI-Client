from __future__ import annotations

from typing import Dict, List
import torch
from torch.nn import functional as F

import utils

# Weird annotations to avoid cyclic import
from modeling import inference_model


class ProbabilityVisualization:
    def __call__(
        self,
        model: inference_model.InferenceModel,
        scores: torch.FloatTensor,
        input_ids: torch.longLongTensor,
    ) -> torch.FloatTensor:
        assert scores.ndim == 2

        if utils.koboldai_vars.numseqs > 1 or not utils.koboldai_vars.show_probs:
            return scores

        if not utils.koboldai_vars.show_probs:
            return scores

        option_offset = 0
        if (
            utils.koboldai_vars.actions.action_count + 1
            in utils.koboldai_vars.actions.actions
        ):
            for x in range(
                len(
                    utils.koboldai_vars.actions.actions[
                        utils.koboldai_vars.actions.action_count + 1
                    ]["Options"]
                )
            ):
                option = utils.koboldai_vars.actions.actions[
                    utils.koboldai_vars.actions.action_count + 1
                ]["Options"][x]
                if option["Pinned"] or option["Previous Selection"] or option["Edited"]:
                    option_offset = x + 1
        batch_offset = (
            int((utils.koboldai_vars.generated_tkns - 1) / utils.koboldai_vars.genamt)
            if utils.koboldai_vars.alt_multi_gen
            else 0
        )
        for batch_index, batch in enumerate(scores):
            probs = F.softmax(batch, dim=-1).cpu().numpy()

            token_prob_info = []
            for token_id, score in sorted(
                enumerate(probs), key=lambda x: x[1], reverse=True
            )[:8]:
                token_prob_info.append(
                    {
                        "tokenId": token_id,
                        "decoded": utils.decodenewlines(
                            model.tokenizer.decode(token_id)
                        ),
                        "score": float(score),
                    }
                )

            if utils.koboldai_vars.numseqs == 1:
                utils.koboldai_vars.actions.set_probabilities(token_prob_info)
            else:
                utils.koboldai_vars.actions.set_option_probabilities(
                    token_prob_info, batch_index + option_offset + batch_offset
                )

        return scores


class LuaIntegration:
    def __call__(
        self,
        model: inference_model.InferenceModel,
        scores: torch.FloatTensor,
        input_ids: torch.longLongTensor,
    ) -> torch.FloatTensor:
        assert scores.ndim == 2
        assert input_ids.ndim == 2
        model.gen_state["regeneration_required"] = False
        model.gen_state["halt"] = False

        if utils.koboldai_vars.standalone:
            return scores

        scores_shape = scores.shape
        scores_list = scores.tolist()
        utils.koboldai_vars.lua_koboldbridge.logits = (
            utils.koboldai_vars.lua_state.table()
        )
        for r, row in enumerate(scores_list):
            utils.koboldai_vars.lua_koboldbridge.logits[
                r + 1
            ] = utils.koboldai_vars.lua_state.table(*row)
        utils.koboldai_vars.lua_koboldbridge.vocab_size = scores_shape[-1]

        utils.koboldai_vars.lua_koboldbridge.execute_genmod()

        scores = torch.tensor(
            tuple(
                tuple(row.values())
                for row in utils.koboldai_vars.lua_koboldbridge.logits.values()
            ),
            device=scores.device,
            dtype=scores.dtype,
        )
        assert scores.shape == scores_shape

        return scores


class PhraseBiasLogitsProcessor:
    def __init__(self) -> None:
        # Hack
        self.model = None

    def _find_intersection(self, big: List, small: List) -> int:
        """Find the maximum overlap between the beginning of small and the end of big.
        Return the index of the token in small following the overlap, or 0.

        big: The tokens in the context (as a tensor)
        small: The tokens in the phrase to bias (as a list)

        Both big and small are in "oldest to newest" order.
        """
        # There are asymptotically more efficient methods for determining the overlap,
        # but typically there will be few (0-1) instances of small[0] in the last len(small)
        # elements of big, plus small will typically be fairly short. So this naive
        # approach is acceptable despite O(N^2) worst case performance.

        num_small = len(small)
        # The small list can only ever match against at most num_small tokens of big,
        # so create a slice.  Typically, this slice will be as long as small, but it
        # may be shorter if the story has just started.
        # We need to convert the big slice to list, since natively big is a tensor
        # and tensor and list don't ever compare equal.  It's better to convert here
        # and then use native equality tests than to iterate repeatedly later.
        big_slice = list(big[-num_small:])

        # It's possible that the start token appears multiple times in small
        # For example, consider the phrase:
        # [ fair is foul, and foul is fair, hover through the fog and filthy air]
        # If we merely look for the first instance of [ fair], then we would
        # generate the following output:
        # " fair is foul, and foul is fair is foul, and foul is fair..."
        start = small[0]
        for i, t in enumerate(big_slice):
            # Strictly unnecessary, but it's marginally faster to test the first
            # token before creating slices to test for a full match.
            if t == start:
                remaining = len(big_slice) - i
                if big_slice[i:] == small[:remaining]:
                    # We found a match.  If the small phrase has any remaining tokens
                    # then return the index of the next token.
                    if remaining < num_small:
                        return remaining
                    # In this case, the entire small phrase matched, so start over.
                    return 0

        # There were no matches, so just begin at the beginning.
        return 0

    def _allow_leftwards_tampering(self, phrase: str) -> bool:
        """Determines if a phrase should be tampered with from the left in
        the "soft" token encoding mode."""

        if phrase[0] in [".", "?", "!", ";", ":", "\n"]:
            return False
        return True

    def _get_token_sequence(self, phrase: str) -> List[List]:
        """Convert the phrase string into a list of encoded biases, each
        one being a list of tokens. How this is done is determined by the
        phrase's format:

        - If the phrase is surrounded by square brackets ([]), the tokens
            will be the phrase split by commas (,). If a "token" isn't
            actually a number, it will be skipped. NOTE: Tokens output by
            this may not be in the model's vocabulary, and such tokens
            should be ignored later in the pipeline.
        - If the phrase is surrounded by curly brackets ({}), the phrase
            will be directly encoded with no synonym biases and no fancy
            tricks.
        - Otherwise, the phrase will be encoded, with close deviations
            being included as synonym biases.
        """

        # TODO: Cache these tokens, invalidate when model or bias is
        # changed.

        # Handle direct token id input
        if phrase.startswith("[") and phrase.endswith("]"):
            no_brackets = phrase[1:-1]
            ret = []
            for token_id in no_brackets.split(","):
                try:
                    ret.append(int(token_id))
                except ValueError:
                    # Ignore non-numbers. Rascals!
                    pass
            return [ret]

        # Handle direct phrases
        if phrase.startswith("{") and phrase.endswith("}"):
            no_brackets = phrase[1:-1]
            return [self.model.tokenizer.encode(no_brackets)]

        # Handle untamperable phrases
        if not self._allow_leftwards_tampering(phrase):
            return [self.model.tokenizer.encode(phrase)]

        # Handle slight alterations to original phrase
        phrase = phrase.strip(" ")
        ret = []

        for alt_phrase in [phrase, f" {phrase}"]:
            ret.append(self.model.tokenizer.encode(alt_phrase))

        return ret

    def _get_biased_tokens(self, input_ids: List) -> Dict:
        # TODO: Different "bias slopes"?

        ret = {}
        for phrase, _bias in utils.koboldai_vars.biases.items():
            bias_score, completion_threshold = _bias
            token_seqs = self._get_token_sequence(phrase)
            variant_deltas = {}
            for token_seq in token_seqs:
                if not token_seq:
                    continue
                bias_index = self._find_intersection(input_ids, token_seq)

                # Ensure completion after completion_threshold tokens
                # Only provide a positive bias when the base bias score is positive.
                if bias_score > 0 and bias_index + 1 > completion_threshold:
                    bias_score = 999

                token_to_bias = token_seq[bias_index]
                variant_deltas[token_to_bias] = bias_score

            # If multiple phrases bias the same token, add the modifiers
            # together. This should NOT be applied to automatic variants
            for token_to_bias, bias_score in variant_deltas.items():
                if token_to_bias in ret:
                    ret[token_to_bias] += bias_score
                else:
                    ret[token_to_bias] = bias_score
        return ret

    def __call__(
        self,
        model: inference_model.InferenceModel,
        scores: torch.FloatTensor,
        input_ids: torch.longLongTensor,
    ) -> torch.FloatTensor:
        self.model = model

        assert scores.ndim == 2
        assert input_ids.ndim == 2

        scores_shape = scores.shape

        for batch in range(scores_shape[0]):
            for token, bias in self._get_biased_tokens(input_ids[batch]).items():
                if bias > 0 and bool(scores[batch][token].isneginf()):
                    # Adding bias to -inf will do NOTHING!!! So just set it for
                    # now. There may be more mathishly correct way to do this
                    # but it'll work. Also, make sure the bias is actually
                    # positive. Don't give a -inf token more chance by setting
                    # it to -0.5!
                    scores[batch][token] = bias
                else:
                    scores[batch][token] += bias

        return scores
