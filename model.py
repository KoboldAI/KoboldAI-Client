# TODO:
# - Intertwine stoppers and streaming and such
from __future__ import annotations

import bisect
import copy
import requests
from dataclasses import dataclass
from eventlet import tpool
import gc
import shutil
import contextlib
import functools
import itertools
import json
import os
import time
import traceback
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import zipfile
from tqdm.auto import tqdm
from logger import logger
import torch_lazy_loader

import torch
from torch.nn import Embedding
import numpy as np
import accelerate.utils
import transformers
from transformers import (
    StoppingCriteria,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    GPTNeoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    modeling_utils,
    AutoModelForTokenClassification,
    AutoConfig,
)

import utils
import breakmodel
import koboldai_settings

HACK_currentmodel = None

try:
    import tpu_mtj_backend
except ModuleNotFoundError as e:
    # Not on TPU... hopefully
    if utils.koboldai_vars.use_colab_tpu:
        raise e

# HACK: Tttttttterrrible structure hack
class colors:
    PURPLE = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    UNDERLINE = "\033[4m"


class OpenAIAPIError(Exception):
    def __init__(self, error_type: str, error_message) -> None:
        super().__init__(f"{error_type}: {error_message}")


class HordeException(Exception):
    pass


class ColabException(Exception):
    pass


class APIException(Exception):
    pass


class GenerationSettings:
    def __init__(self, **overrides) -> None:
        for setting in [
            "temp",
            "top_p",
            "top_k",
            "tfs",
            "typical",
            "top_a",
            "rep_pen",
            "rep_pen_slope",
            "rep_pen_range",
            "sampler_order",
        ]:
            setattr(
                self,
                setting,
                overrides.get(setting, getattr(utils.koboldai_vars, setting)),
            )


class Stoppers:
    @staticmethod
    def core_stopper(
        model: InferenceModel,
        input_ids: torch.LongTensor,
    ) -> bool:
        if not utils.koboldai_vars.inference_config.do_core:
            return False

        utils.koboldai_vars.generated_tkns += 1

        if (
            not utils.koboldai_vars.standalone
            and utils.koboldai_vars.lua_koboldbridge.generated_cols
            and utils.koboldai_vars.generated_tkns
            != utils.koboldai_vars.lua_koboldbridge.generated_cols
        ):
            raise RuntimeError(
                f"Inconsistency detected between KoboldAI Python and Lua backends ({utils.koboldai_vars.generated_tkns} != {utils.koboldai_vars.lua_koboldbridge.generated_cols})"
            )

        if utils.koboldai_vars.abort or (
            utils.koboldai_vars.inference_config.stop_at_genamt
            and utils.koboldai_vars.generated_tkns >= utils.koboldai_vars.genamt
        ):
            utils.koboldai_vars.abort = False
            model.gen_state["regeneration_required"] = False
            model.gen_state["halt"] = False
            return True

        if utils.koboldai_vars.standalone:
            return False

        assert input_ids.ndim == 2

        model.gen_state[
            "regeneration_required"
        ] = utils.koboldai_vars.lua_koboldbridge.regeneration_required
        model.gen_state["halt"] = not utils.koboldai_vars.lua_koboldbridge.generating
        utils.koboldai_vars.lua_koboldbridge.regeneration_required = False

        for i in (
            range(utils.koboldai_vars.numseqs)
            if not utils.koboldai_vars.alt_multi_gen
            else range(1)
        ):
            utils.koboldai_vars.lua_koboldbridge.generated[i + 1][
                utils.koboldai_vars.generated_tkns
            ] = int(input_ids[i, -1].item())

        return model.gen_state["regeneration_required"] or model.gen_state["halt"]

    @staticmethod
    def dynamic_wi_scanner(
        model: InferenceModel,
        input_ids: torch.LongTensor,
    ) -> bool:
        if not utils.koboldai_vars.inference_config.do_dynamic_wi:
            return False

        if not utils.koboldai_vars.dynamicscan:
            return False

        if len(model.gen_state["wi_scanner_excluded_keys"]) != input_ids.shape[0]:
            model.gen_state["wi_scanner_excluded_keys"]
            print(model.tokenizer.decode(model.gen_state["wi_scanner_excluded_keys"]))
            print(model.tokenizer.decode(input_ids.shape[0]))

        assert len(model.gen_state["wi_scanner_excluded_keys"]) == input_ids.shape[0]

        tail = input_ids[..., -utils.koboldai_vars.generated_tkns :]
        for i, t in enumerate(tail):
            decoded = utils.decodenewlines(model.tokenizer.decode(t))
            _, _, _, found = utils.koboldai_vars.calc_ai_text(
                submitted_text=decoded, send_context=False
            )
            found = list(
                set(found) - set(model.gen_state["wi_scanner_excluded_keys"][i])
            )
            if found:
                print("FOUNDWI", found)
                return True
        return False

    @staticmethod
    def chat_mode_stopper(
        model: InferenceModel,
        input_ids: torch.LongTensor,
    ) -> bool:
        if not utils.koboldai_vars.chatmode:
            return False

        data = [model.tokenizer.decode(x) for x in input_ids]
        # null_character = model.tokenizer.encode(chr(0))[0]
        if "completed" not in model.gen_state:
            model.gen_state["completed"] = [False] * len(input_ids)

        for i in range(len(input_ids)):
            if (
                data[i][-1 * (len(utils.koboldai_vars.chatname) + 1) :]
                == utils.koboldai_vars.chatname + ":"
            ):
                model.gen_state["completed"][i] = True
        if all(model.gen_state["completed"]):
            utils.koboldai_vars.generated_tkns = utils.koboldai_vars.genamt
            del model.gen_state["completed"]
            return True
        return False


class PostTokenHooks:
    @staticmethod
    def stream_tokens(
        model: InferenceModel,
        input_ids: torch.LongTensor,
    ) -> None:
        if not model.gen_state["do_streaming"]:
            return

        if not utils.koboldai_vars.output_streaming:
            return

        data = [
            utils.applyoutputformatting(
                utils.decodenewlines(model.tokenizer.decode(x[-1])),
                no_sentence_trimming=True,
                no_single_line=True,
            )
            for x in input_ids
        ]
        utils.koboldai_vars.actions.stream_tokens(data)


# We only want to use logit manipulations and such on our core text model
class use_core_manipulations:
    # These must be set by wherever they get setup
    get_logits_processor: callable = None
    sample: callable = None
    get_stopping_criteria: callable = None

    # We set these automatically
    old_get_logits_processor: callable = None
    old_sample: callable = None
    old_get_stopping_criteria: callable = None

    def __enter__(self):
        if use_core_manipulations.get_logits_processor:
            use_core_manipulations.old_get_logits_processor = (
                transformers.GenerationMixin._get_logits_processor
            )
            transformers.GenerationMixin._get_logits_processor = (
                use_core_manipulations.get_logits_processor
            )

        if use_core_manipulations.sample:
            use_core_manipulations.old_sample = transformers.GenerationMixin.sample
            transformers.GenerationMixin.sample = use_core_manipulations.sample

        if use_core_manipulations.get_stopping_criteria:
            use_core_manipulations.old_get_stopping_criteria = (
                transformers.GenerationMixin._get_stopping_criteria
            )
            transformers.GenerationMixin._get_stopping_criteria = (
                use_core_manipulations.get_stopping_criteria
            )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if use_core_manipulations.old_get_logits_processor:
            transformers.GenerationMixin._get_logits_processor = (
                use_core_manipulations.old_get_logits_processor
            )
        else:
            assert (
                not use_core_manipulations.get_logits_processor
            ), "Patch leak: THE MONKEYS HAVE ESCAPED"

        if use_core_manipulations.old_sample:
            transformers.GenerationMixin.sample = use_core_manipulations.old_sample
        else:
            assert (
                not use_core_manipulations.sample
            ), "Patch leak: THE MONKEYS HAVE ESCAPED"

        if use_core_manipulations.old_get_stopping_criteria:
            transformers.GenerationMixin._get_stopping_criteria = (
                use_core_manipulations.old_get_stopping_criteria
            )
        else:
            assert (
                not use_core_manipulations.get_stopping_criteria
            ), "Patch leak: THE MONKEYS HAVE ESCAPED"


def patch_transformers_download():
    def http_get(
        url: str,
        temp_file,
        proxies=None,
        resume_size=0,
        headers=None,
        file_name=None,
    ):
        """
        Download remote file. Do not gobble up errors.
        """
        headers = copy.deepcopy(headers)
        if resume_size > 0:
            headers["Range"] = f"bytes={resume_size}-"
        r = requests.get(url, stream=True, proxies=proxies, headers=headers)
        transformers.utils.hub._raise_for_status(r)
        content_length = r.headers.get("Content-Length")
        total = (
            resume_size + int(content_length) if content_length is not None else None
        )

        # `tqdm` behavior is determined by `utils.logging.is_progress_bar_enabled()`
        # and can be set using `utils.logging.enable/disable_progress_bar()`
        if url[-11:] != "config.json":
            progress = tqdm.tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=total,
                initial=resume_size,
                desc=f"Downloading {file_name}"
                if file_name is not None
                else "Downloading",
                file=utils.UIProgressBarFile(),
            )
            utils.koboldai_vars.status_message = "Download Model"
            utils.koboldai_vars.total_download_chunks = total

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                if url[-11:] != "config.json":
                    progress.update(len(chunk))
                    utils.koboldai_vars.downloaded_chunks += len(chunk)
                temp_file.write(chunk)

        if url[-11:] != "config.json":
            progress.close()

        utils.koboldai_vars.status_message = ""

    transformers.utils.hub.http_get = http_get


def patch_transformers_loader() -> None:
    """
    Patch the Transformers loader to use aria2 and our shard tracking.
    Universal for TPU/MTJ and Torch.
    """
    old_from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def new_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        utils.koboldai_vars.fp32_model = False
        utils.num_shards = None
        utils.current_shard = 0
        utils.from_pretrained_model_name = pretrained_model_name_or_path
        utils.from_pretrained_index_filename = None
        utils.from_pretrained_kwargs = kwargs
        utils.bar = None
        if not utils.args.no_aria2:
            utils.aria2_hook(pretrained_model_name_or_path, **kwargs)
        return old_from_pretrained(
            cls, pretrained_model_name_or_path, *model_args, **kwargs
        )

    if not hasattr(PreTrainedModel, "_kai_patched"):
        PreTrainedModel.from_pretrained = new_from_pretrained
        PreTrainedModel._kai_patched = True

    if hasattr(modeling_utils, "get_checkpoint_shard_files"):
        old_get_checkpoint_shard_files = modeling_utils.get_checkpoint_shard_files

        def new_get_checkpoint_shard_files(
            pretrained_model_name_or_path, index_filename, *args, **kwargs
        ):
            utils.num_shards = utils.get_num_shards(index_filename)
            utils.from_pretrained_index_filename = index_filename
            return old_get_checkpoint_shard_files(
                pretrained_model_name_or_path, index_filename, *args, **kwargs
            )

        modeling_utils.get_checkpoint_shard_files = new_get_checkpoint_shard_files


def patch_transformers_generation() -> None:
    # Not sure why this global is needed...
    global transformers

    # Patch transformers to use our custom logit warpers -- Only HFTorchInferenceModel uses this
    from transformers import (
        LogitsProcessorList,
        LogitsWarper,
        LogitsProcessor,
        TopKLogitsWarper,
        TopPLogitsWarper,
        TemperatureLogitsWarper,
    )
    from warpers import (
        AdvancedRepetitionPenaltyLogitsProcessor,
        TailFreeLogitsWarper,
        TypicalLogitsWarper,
        TopALogitsWarper,
    )

    def dynamic_processor_wrap(cls, field_name, var_name, cond=None):
        old_call = cls.__call__

        def new_call(self, *args, **kwargs):
            if not isinstance(field_name, str) and isinstance(field_name, Iterable):
                conds = []
                for f, v in zip(field_name, var_name):
                    conds.append(getattr(utils.koboldai_vars, v))
                    setattr(self, f, conds[-1])
            else:
                conds = getattr(utils.koboldai_vars, var_name)
                setattr(self, field_name, conds)
            assert len(args) == 2
            if cond is None or cond(conds):
                return old_call(self, *args, **kwargs)
            return args[1]

        cls.__call__ = new_call

    # TODO: Make samplers generic
    dynamic_processor_wrap(
        AdvancedRepetitionPenaltyLogitsProcessor,
        ("penalty", "penalty_slope", "penalty_range", "use_alt_rep_pen"),
        ("rep_pen", "rep_pen_slope", "rep_pen_range", "use_alt_rep_pen"),
        cond=lambda x: x[0] != 1.0,
    )
    dynamic_processor_wrap(TopKLogitsWarper, "top_k", "top_k", cond=lambda x: x > 0)
    dynamic_processor_wrap(TopALogitsWarper, "top_a", "top_a", cond=lambda x: x > 0.0)
    dynamic_processor_wrap(TopPLogitsWarper, "top_p", "top_p", cond=lambda x: x < 1.0)
    dynamic_processor_wrap(TailFreeLogitsWarper, "tfs", "tfs", cond=lambda x: x < 1.0)
    dynamic_processor_wrap(
        TypicalLogitsWarper, "typical", "typical", cond=lambda x: x < 1.0
    )
    dynamic_processor_wrap(
        TemperatureLogitsWarper, "temperature", "temp", cond=lambda x: x != 1.0
    )

    class PhraseBiasLogitsProcessor(LogitsProcessor):
        def __init__(self):
            pass

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
                return [HACK_currentmodel.tokenizer.encode(no_brackets)]

            # Handle untamperable phrases
            if not self._allow_leftwards_tampering(phrase):
                return [HACK_currentmodel.tokenizer.encode(phrase)]

            # Handle slight alterations to original phrase
            phrase = phrase.strip(" ")
            ret = []

            for alt_phrase in [phrase, f" {phrase}"]:
                ret.append(HACK_currentmodel.tokenizer.encode(alt_phrase))

            return ret

        def _get_biased_tokens(self, input_ids: List) -> Dict:
            # TODO: Different "bias slopes"?

            ret = {}
            for phrase, _bias in utils.koboldai_vars.biases.items():
                bias_score, completion_threshold = _bias
                token_seqs = self._get_token_sequence(phrase)
                variant_deltas = {}
                for token_seq in token_seqs:
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
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
        ) -> torch.FloatTensor:
            assert scores.ndim == 2
            assert input_ids.ndim == 2

            scores_shape = scores.shape

            for batch in range(scores_shape[0]):
                for token, bias in self._get_biased_tokens(input_ids[batch]).items():
                    scores[batch][token] += bias

            return scores

    class LuaLogitsProcessor(LogitsProcessor):
        def __init__(self):
            pass

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
        ) -> torch.FloatTensor:
            assert scores.ndim == 2
            assert input_ids.ndim == 2
            self.regeneration_required = False
            self.halt = False

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

    from torch.nn import functional as F

    def visualize_probabilities(
        model: InferenceModel,
        scores: torch.FloatTensor,
    ) -> None:
        assert scores.ndim == 2

        if utils.koboldai_vars.numseqs > 1 or not utils.koboldai_vars.show_probs:
            return

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

    def new_get_logits_processor(*args, **kwargs) -> LogitsProcessorList:
        processors = new_get_logits_processor.old_get_logits_processor(*args, **kwargs)
        processors.insert(0, LuaLogitsProcessor())
        processors.append(PhraseBiasLogitsProcessor())
        return processors

    use_core_manipulations.get_logits_processor = new_get_logits_processor
    new_get_logits_processor.old_get_logits_processor = (
        transformers.GenerationMixin._get_logits_processor
    )

    class KoboldLogitsWarperList(LogitsProcessorList):
        def __init__(self, beams: int = 1, **kwargs):
            self.__warper_list: List[LogitsWarper] = []
            self.__warper_list.append(
                TopKLogitsWarper(top_k=1, min_tokens_to_keep=1 + (beams > 1))
            )
            self.__warper_list.append(
                TopALogitsWarper(top_a=0.5, min_tokens_to_keep=1 + (beams > 1))
            )
            self.__warper_list.append(
                TopPLogitsWarper(top_p=0.5, min_tokens_to_keep=1 + (beams > 1))
            )
            self.__warper_list.append(
                TailFreeLogitsWarper(tfs=0.5, min_tokens_to_keep=1 + (beams > 1))
            )
            self.__warper_list.append(
                TypicalLogitsWarper(typical=0.5, min_tokens_to_keep=1 + (beams > 1))
            )
            self.__warper_list.append(TemperatureLogitsWarper(temperature=0.5))
            self.__warper_list.append(AdvancedRepetitionPenaltyLogitsProcessor())

        def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            *args,
            **kwargs,
        ):
            sampler_order = utils.koboldai_vars.sampler_order[:]
            if (
                len(sampler_order) < 7
            ):  # Add repetition penalty at beginning if it's not present
                sampler_order = [6] + sampler_order
            for k in sampler_order:
                scores = self.__warper_list[k](input_ids, scores, *args, **kwargs)
            visualize_probabilities(HACK_currentmodel, scores)
            return scores

    def new_get_logits_warper(
        beams: int = 1,
    ) -> LogitsProcessorList:
        return KoboldLogitsWarperList(beams=beams)

    def new_sample(self, *args, **kwargs):
        assert kwargs.pop("logits_warper", None) is not None
        kwargs["logits_warper"] = new_get_logits_warper(
            beams=1,
        )
        if (utils.koboldai_vars.newlinemode == "s") or (
            utils.koboldai_vars.newlinemode == "ns"
        ):
            kwargs["eos_token_id"] = -1
            kwargs.setdefault("pad_token_id", 2)
        return new_sample.old_sample(self, *args, **kwargs)

    new_sample.old_sample = transformers.GenerationMixin.sample
    use_core_manipulations.sample = new_sample

    # Allow bad words filter to ban <|endoftext|> token
    import transformers.generation.logits_process

    def new_init(self, bad_words_ids: List[List[int]], eos_token_id: int):
        return new_init.old_init(self, bad_words_ids, -1)

    new_init.old_init = (
        transformers.generation.logits_process.NoBadWordsLogitsProcessor.__init__
    )
    transformers.generation.logits_process.NoBadWordsLogitsProcessor.__init__ = new_init


def patch_transformers() -> None:
    patch_transformers_download()
    patch_transformers_loader()

    # Doesn't do anything for TPU
    patch_transformers_generation()


class GenerationResult:
    def __init__(
        self,
        model: InferenceModel,
        out_batches: list,
        prompt: list,
        # Controls if generate() does it's looping thing. This should only be
        # done for HF models that use that StoppingCondition
        is_whole_generation: bool,
        # Controls if we should trim output by prompt length
        output_includes_prompt: bool = False,
        # Lazy filter to cut off extra lines where we can't manipulate
        # probabilities
        single_line: bool = False,
    ):
        # Shave prompt off of encoded response when needed (HF). Decoded does
        # not return prompt.
        if output_includes_prompt:
            self.encoded = out_batches[:, len(prompt) :]
        else:
            self.encoded = out_batches

        self.prompt = prompt
        self.is_whole_generation = is_whole_generation

        self.decoded = [
            utils.decodenewlines(model.tokenizer.decode(enc)) for enc in self.encoded
        ]

        if single_line:
            self.decoded = [x.split("\n", 1)[0] for x in self.decoded]
            self.encoded = np.array(model.tokenizer(self.decoded).input_ids)


@dataclass
class ModelCapabilities:
    embedding_manipulation: bool = False
    post_token_hooks: bool = False
    stopper_hooks: bool = False
    # TODO: Support non-live probabilities from APIs
    post_token_probs: bool = False


class InferenceModel:
    def __init__(self) -> None:
        self.gen_state = {}
        self.post_token_hooks = []
        self.stopper_hooks = []
        self.tokenizer = None
        self.capabilties = ModelCapabilities()

    def load(self, save_model: bool = False, initial_load: bool = False) -> None:
        """Main load function. Do not override this. Override _load() instead."""

        self._load(save_model=save_model, initial_load=initial_load)
        self._post_load()

        global HACK_currentmodel
        HACK_currentmodel = self

    def _post_load(self) -> None:
        pass

    def _load(self, save_model: bool, inital_load: bool) -> None:
        raise NotImplementedError

    def _get_tokenizer(self, location: str):
        # TODO: This newlinemode inference might need more scrutiny
        utils.koboldai_vars.newlinemode = "n"
        if "xglm" in location:
            # Default to </s> newline mode if using XGLM
            utils.koboldai_vars.newlinemode = "s"
        if "opt" in location or "bloom" in location:
            # Handle </s> but don't convert newlines if using Fairseq models that have newlines trained in them
            utils.koboldai_vars.newlinemode = "ns"

        std_kwargs = {"revision": utils.koboldai_vars.revision, "cache_dir": "cache"}

        suppliers = [
            # Fast tokenizer disabled by default as per HF docs:
            # > Note: Make sure to pass use_fast=False when loading
            #   OPT’s tokenizer with AutoTokenizer to get the correct
            #   tokenizer.
            lambda: AutoTokenizer.from_pretrained(
                location, use_fast=False, **std_kwargs
            ),
            lambda: AutoTokenizer.from_pretrained(location, **std_kwargs),
            # Fallback to GPT2Tokenizer
            lambda: GPT2Tokenizer.from_pretrained(location, **std_kwargs),
            lambda: GPT2Tokenizer.from_pretrained("gpt2", **std_kwargs),
        ]

        for i, try_get_tokenizer in enumerate(suppliers):
            try:
                return try_get_tokenizer()
            except Exception as e:
                # If we error on each attempt, raise the last one
                if i == len(suppliers) - 1:
                    raise e

    def core_generate(
        self,
        text: list,
        _min: int,
        _max: int,
        found_entries: set,
        is_core: bool = False,
    ):
        # This generation function is tangled with koboldai_vars intentionally. It
        # is meant for the story and nothing else.

        start_time = time.time()
        gen_in = torch.tensor(text, dtype=torch.long)[None]
        logger.debug(
            "core_generate: torch.tensor time {}s".format(time.time() - start_time)
        )

        start_time = time.time()
        if utils.koboldai_vars.is_model_torch():
            # Torch stuff
            if utils.koboldai_vars.full_determinism:
                torch.manual_seed(utils.koboldai_vars.seed)

            if utils.koboldai_vars.sp is not None:
                assert self.capabilties.embedding_manipulation
                soft_tokens = torch.arange(
                    self.model.config.vocab_size,
                    self.model.config.vocab_size + utils.koboldai_vars.sp.shape[0],
                )
                gen_in = torch.cat((soft_tokens[None], gen_in), dim=-1)
        elif utils.koboldai_vars.use_colab_tpu:
            if utils.koboldai_vars.full_determinism:
                tpu_mtj_backend.set_rng_seed(utils.koboldai_vars.seed)

        logger.debug(
            "core_generate: Model Setup (SP, etc) time {}s".format(
                time.time() - start_time
            )
        )

        if (
            gen_in.shape[-1] + utils.koboldai_vars.genamt
            > utils.koboldai_vars.max_length
        ):
            logger.error("gen_in.shape[-1]: {}".format(gen_in.shape[-1]))
            logger.error(
                "utils.koboldai_vars.genamt: {}".format(utils.koboldai_vars.genamt)
            )
            logger.error(
                "utils.koboldai_vars.max_length: {}".format(
                    utils.koboldai_vars.max_length
                )
            )
        assert (
            gen_in.shape[-1] + utils.koboldai_vars.genamt
            <= utils.koboldai_vars.max_length
        )

        start_time = time.time()
        gen_in = gen_in.to(utils.get_auxilary_device())

        logger.debug(
            "core_generate: gen_in to device time {}s".format(time.time() - start_time)
        )
        start_time = time.time()

        found_entries = found_entries or set()

        self.gen_state["wi_scanner_excluded_keys"] = found_entries

        utils.koboldai_vars._prompt = utils.koboldai_vars.prompt

        with torch.no_grad():
            already_generated = 0
            numseqs = utils.koboldai_vars.numseqs
            total_gens = None

            for i in range(
                utils.koboldai_vars.numseqs if utils.koboldai_vars.alt_multi_gen else 1
            ):
                while True:
                    # The reason this is a loop is due to how Dynamic WI works. We
                    # cannot simply add the WI to the context mid-generation, so we
                    # stop early, and then insert WI, then continue generating. That
                    # stopping and continuing is this loop.

                    start_time = time.time()
                    result = self.raw_generate(
                        gen_in[0],
                        max_new=utils.koboldai_vars.genamt,
                        do_streaming=utils.koboldai_vars.output_streaming,
                        do_dynamic_wi=utils.koboldai_vars.dynamicscan,
                        batch_count=numseqs
                        if not utils.koboldai_vars.alt_multi_gen
                        else 1,
                        # Real max length is handled by CoreStopper.
                        bypass_hf_maxlength=utils.koboldai_vars.dynamicscan,
                        is_core=True,
                    )
                    logger.debug(
                        "core_generate: run raw_generate pass {} {}s".format(
                            already_generated, time.time() - start_time
                        )
                    )

                    genout = result.encoded

                    already_generated += len(genout[0])

                    try:
                        assert (
                            already_generated
                            <= utils.koboldai_vars.genamt * utils.koboldai_vars.numseqs
                            if utils.koboldai_vars.alt_multi_gen
                            else 1
                        )
                    except AssertionError:
                        print("AlreadyGenerated", already_generated)
                        print("genamt", utils.koboldai_vars.genamt)
                        raise

                    if result.is_whole_generation:
                        break

                    # Generation stopped; why?
                    # If we have been told to halt, we have reached our target token
                    # amount (controlled by halt), or Dynamic WI has not told us to
                    # stop temporarily to insert WI, we can assume that we are done
                    # generating. We shall break.
                    if (
                        self.gen_state["halt"]
                        or not self.gen_state["regeneration_required"]
                    ):
                        break

                    # Now we are doing stuff for Dynamic WI.
                    assert genout.ndim >= 2
                    assert genout.shape[0] == utils.koboldai_vars.numseqs

                    if (
                        utils.koboldai_vars.lua_koboldbridge.generated_cols
                        and utils.koboldai_vars.generated_tkns
                        != utils.koboldai_vars.lua_koboldbridge.generated_cols
                    ):
                        raise RuntimeError(
                            f"Inconsistency detected between KoboldAI Python and Lua backends ({utils.koboldai_vars.generated_tkns} != {utils.koboldai_vars.lua_koboldbridge.generated_cols})"
                        )

                    if already_generated != utils.koboldai_vars.generated_tkns:
                        print("already_generated: {}".format(already_generated))
                        print(
                            "generated_tkns: {}".format(
                                utils.koboldai_vars.generated_tkns
                            )
                        )
                        raise RuntimeError("WI scanning error")

                    for r in range(utils.koboldai_vars.numseqs):
                        for c in range(already_generated):
                            assert (
                                utils.koboldai_vars.lua_koboldbridge.generated[r + 1][
                                    c + 1
                                ]
                                is not None
                            )
                            genout[r][
                                genout.shape[-1] - already_generated + c
                            ] = utils.koboldai_vars.lua_koboldbridge.generated[r + 1][
                                c + 1
                            ]

                    encoded = []

                    for i in range(utils.koboldai_vars.numseqs):
                        txt = utils.decodenewlines(
                            self.tokenizer.decode(genout[i, -already_generated:])
                        )
                        # winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True, actions=utils.koboldai_vars.actions)
                        # txt, _, _ = calcsubmitbudget(len(utils.koboldai_vars.actions), winfo, mem, anotetxt, utils.koboldai_vars.actions, submission=txt)
                        txt, _, _, _found_entries = utils.koboldai_vars.calc_ai_text(
                            submitted_text=txt, send_context=False
                        )
                        found_entries[i].update(_found_entries)
                        encoded.append(
                            torch.tensor(txt, dtype=torch.long, device=genout.device)
                        )

                    max_length = len(max(encoded, key=len))
                    encoded = torch.stack(
                        tuple(
                            torch.nn.functional.pad(
                                e,
                                (max_length - len(e), 0),
                                value=self.model.config.pad_token_id
                                or self.model.config.eos_token_id,
                            )
                            for e in encoded
                        )
                    )
                    genout = torch.cat(
                        (
                            encoded,
                            genout[..., -already_generated:],
                        ),
                        dim=-1,
                    )

                    if utils.koboldai_vars.sp is not None:
                        soft_tokens = torch.arange(
                            self.model.config.vocab_size,
                            self.model.config.vocab_size
                            + utils.koboldai_vars.sp.shape[0],
                            device=genout.device,
                        )
                        genout = torch.cat(
                            (soft_tokens.tile(utils.koboldai_vars.numseqs, 1), genout),
                            dim=-1,
                        )

                    assert (
                        genout.shape[-1]
                        + utils.koboldai_vars.genamt
                        - already_generated
                        <= utils.koboldai_vars.max_length
                    )
                    gen_in = genout
                    numseqs = 1
                if total_gens is None:
                    total_gens = genout
                else:
                    total_gens = torch.cat((total_gens, genout))

        return total_gens, already_generated

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
    ) -> GenerationResult:
        raise NotImplementedError

    def raw_generate(
        self,
        # prompt is either a string (text) or a list (token ids)
        prompt: Union[str, list, np.ndarray],
        max_new: int,
        do_streaming: bool = False,
        do_dynamic_wi: bool = False,
        batch_count: int = 1,
        bypass_hf_maxlength: bool = False,
        generation_settings: Optional[dict] = None,
        is_core: bool = False,
        single_line: bool = False,
        found_entries: set = (),
    ) -> GenerationResult:
        """A wrapper around _raw_generate() that handles timing and some other minute stuff."""
        # TODO: Support singleline outside of torch

        self.gen_state["do_streaming"] = do_streaming
        self.gen_state["do_dynamic_wi"] = do_dynamic_wi

        # Dynamic WI depends on this!!! This is a main gen call.
        self.gen_state["stop_at_genamt"] = do_dynamic_wi

        # Makes stopping criteria hook happy
        self.gen_state["wi_scanner_excluded_keys"] = self.gen_state.get(
            "wi_scanner_excluded_keys", set()
        )

        utils.koboldai_vars.inference_config.do_core = is_core
        gen_settings = GenerationSettings(*(generation_settings or {}))

        if isinstance(prompt, torch.Tensor):
            prompt_tokens = prompt.cpu().numpy()
        elif isinstance(prompt, list):
            prompt_tokens = np.array(prompt)
        elif isinstance(prompt, str):
            prompt_tokens = np.array(self.tokenizer.encode(prompt))
        else:
            raise ValueError(f"Prompt is {type(prompt)}. Not a fan!")

        assert isinstance(prompt_tokens, np.ndarray)
        assert len(prompt_tokens.shape) == 1

        if utils.koboldai_vars.model == "ReadOnly":
            raise NotImplementedError("No loaded model")

        time_start = time.time()

        with use_core_manipulations():
            result = self._raw_generate(
                prompt_tokens=prompt_tokens,
                max_new=max_new,
                batch_count=batch_count,
                gen_settings=gen_settings,
                single_line=single_line,
            )

        time_end = round(time.time() - time_start, 2)
        tokens_per_second = round(len(result.encoded[0]) / time_end, 2)

        if not utils.koboldai_vars.quiet:
            logger.info(
                f"Generated {len(result.encoded[0])} tokens in {time_end} seconds, for an average rate of {tokens_per_second} tokens per second."
            )

        return result

    def generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new_tokens: int,
        do_streaming: bool = False,
        do_dynamic_wi: bool = False,
        single_line: bool = False,
        batch_count: int = 1,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _post_token_gen(self, input_ids: torch.LongTensor) -> None:
        for hook in self.post_token_hooks:
            hook(self, input_ids)


class HFMTJInferenceModel:
    def __init__(
        self,
        model_name: str,
    ) -> None:
        super().__init__()

        self.model_name = model_name

        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.capabilties = ModelCapabilities(
            embedding_manipulation=False,
            post_token_hooks=False,
            stopper_hooks=False,
            post_token_probs=False,
        )

    def setup_mtj(self) -> None:
        def mtj_warper_callback(scores) -> "np.array":
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

            scores = np.array(
                tuple(
                    tuple(row.values())
                    for row in utils.koboldai_vars.lua_koboldbridge.logits.values()
                ),
                dtype=scores.dtype,
            )
            assert scores.shape == scores_shape

            return scores

        def mtj_stopping_callback(
            generated, n_generated, excluded_world_info
        ) -> Tuple[List[set], bool, bool]:
            utils.koboldai_vars.generated_tkns += 1

            assert len(excluded_world_info) == len(generated)
            regeneration_required = (
                utils.koboldai_vars.lua_koboldbridge.regeneration_required
            )
            halt = (
                utils.koboldai_vars.abort
                or not utils.koboldai_vars.lua_koboldbridge.generating
                or utils.koboldai_vars.generated_tkns >= utils.koboldai_vars.genamt
            )
            utils.koboldai_vars.lua_koboldbridge.regeneration_required = False

            global past

            for i in range(utils.koboldai_vars.numseqs):
                utils.koboldai_vars.lua_koboldbridge.generated[i + 1][
                    utils.koboldai_vars.generated_tkns
                ] = int(
                    generated[i, tpu_mtj_backend.params["seq"] + n_generated - 1].item()
                )

            if not utils.koboldai_vars.dynamicscan or halt:
                return excluded_world_info, regeneration_required, halt

            for i, t in enumerate(generated):
                decoded = utils.decodenewlines(
                    self.tokenizer.decode(past[i])
                ) + utils.decodenewlines(
                    self.tokenizer.decode(
                        t[
                            tpu_mtj_backend.params["seq"] : tpu_mtj_backend.params[
                                "seq"
                            ]
                            + n_generated
                        ]
                    )
                )
                # _, found = checkworldinfo(decoded, force_use_txt=True, actions=koboldai_vars.actions)
                _, _, _, found = utils.koboldai_vars.calc_ai_text(
                    submitted_text=decoded
                )
                found -= excluded_world_info[i]
                if len(found) != 0:
                    regeneration_required = True
                    break
            return excluded_world_info, regeneration_required, halt

        def mtj_compiling_callback() -> None:
            print(colors.GREEN + "TPU backend compilation triggered" + colors.END)
            utils.koboldai_vars.compiling = True

        def mtj_stopped_compiling_callback() -> None:
            print(colors.GREEN + "TPU backend compilation stopped" + colors.END)
            utils.koboldai_vars.compiling = False

        def mtj_settings_callback() -> dict:
            sampler_order = utils.koboldai_vars.sampler_order[:]
            if (
                len(sampler_order) < 7
            ):  # Add repetition penalty at beginning if it's not present
                sampler_order = [6] + sampler_order
            return {
                "sampler_order": utils.koboldai_vars.sampler_order,
                "top_p": float(utils.koboldai_vars.top_p),
                "temp": float(utils.koboldai_vars.temp),
                "top_k": int(utils.koboldai_vars.top_k),
                "tfs": float(utils.koboldai_vars.tfs),
                "typical": float(utils.koboldai_vars.typical),
                "top_a": float(utils.koboldai_vars.top_a),
                "repetition_penalty": float(utils.koboldai_vars.rep_pen),
                "rpslope": float(utils.koboldai_vars.rep_pen_slope),
                "rprange": int(utils.koboldai_vars.rep_pen_range),
            }

        self.load_mtj_backend()

        tpu_mtj_backend.socketio = utils.socketio

        if utils.koboldai_vars.model == "TPUMeshTransformerGPTNeoX":
            utils.koboldai_vars.badwordsids = utils.koboldai_vars.badwordsids_neox

        print(
            "{0}Initializing Mesh Transformer JAX, please wait...{1}".format(
                colors.PURPLE, colors.END
            )
        )
        if utils.koboldai_vars.model in (
            "TPUMeshTransformerGPTJ",
            "TPUMeshTransformerGPTNeoX",
        ) and (
            not utils.koboldai_vars.custmodpth
            or not os.path.isdir(utils.koboldai_vars.custmodpth)
        ):
            raise FileNotFoundError(
                f"The specified model path {repr(utils.koboldai_vars.custmodpth)} is not the path to a valid folder"
            )
        if utils.koboldai_vars.model == "TPUMeshTransformerGPTNeoX":
            tpu_mtj_backend.pad_token_id = 2

        tpu_mtj_backend.koboldai_vars = utils.koboldai_vars
        tpu_mtj_backend.warper_callback = mtj_warper_callback
        tpu_mtj_backend.stopping_callback = mtj_stopping_callback
        tpu_mtj_backend.compiling_callback = mtj_compiling_callback
        tpu_mtj_backend.stopped_compiling_callback = mtj_stopped_compiling_callback
        tpu_mtj_backend.settings_callback = mtj_settings_callback

    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.patch_transformers()
        self.setup_mtj()

        utils.koboldai_vars.allowsp = True
        # loadmodelsettings()
        # loadsettings()
        tpu_mtj_backend.load_model(
            utils.koboldai_vars.custmodpth,
            hf_checkpoint=utils.koboldai_vars.model
            not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")
            and utils.koboldai_vars.use_colab_tpu,
            socketio_queue=koboldai_settings.queue,
            initial_load=initial_load,
            logger=logger,
            **utils.koboldai_vars.modelconfig,
        )

        # tpool.execute(tpu_mtj_backend.load_model, koboldai_vars.custmodpth, hf_checkpoint=koboldai_vars.model not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and koboldai_vars.use_colab_tpu, **koboldai_vars.modelconfig)
        utils.koboldai_vars.modeldim = int(
            tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"])
        )

        self.tokenizer = tpu_mtj_backend.tokenizer
        if (
            utils.koboldai_vars.badwordsids is koboldai_settings.badwordsids_default
            and utils.koboldai_vars.model_type not in ("gpt2", "gpt_neo", "gptj")
        ):
            utils.koboldai_vars.badwordsids = [
                [v]
                for k, v in self.tokenizer.get_vocab().items()
                if any(c in str(k) for c in "<>[]")
                if utils.koboldai_vars.newlinemode != "s" or str(k) != "</s>"
            ]

    def get_soft_tokens() -> np.array:
        soft_tokens = None

        if utils.koboldai_vars.sp is None:
            tensor = np.zeros(
                (
                    1,
                    tpu_mtj_backend.params.get(
                        "d_embed", tpu_mtj_backend.params["d_model"]
                    ),
                ),
                dtype=np.float32,
            )
            rows = tensor.shape[0]
            padding_amount = (
                tpu_mtj_backend.params["seq"]
                - (
                    tpu_mtj_backend.params["seq"]
                    % -tpu_mtj_backend.params["cores_per_replica"]
                )
                - rows
            )
            tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
            tensor = tensor.reshape(
                tpu_mtj_backend.params["cores_per_replica"],
                -1,
                tpu_mtj_backend.params.get(
                    "d_embed", tpu_mtj_backend.params["d_model"]
                ),
            )
            utils.koboldai_vars.sp = tpu_mtj_backend.shard_xmap(tensor)

        soft_tokens = np.arange(
            tpu_mtj_backend.params["n_vocab"]
            + tpu_mtj_backend.params["n_vocab_padding"],
            tpu_mtj_backend.params["n_vocab"]
            + tpu_mtj_backend.params["n_vocab_padding"]
            + utils.koboldai_vars.sp_length,
            dtype=np.uint32,
        )
        return soft_tokens

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
    ) -> GenerationResult:
        soft_tokens = self.get_soft_tokens()

        genout = tpool.execute(
            tpu_mtj_backend.infer_static,
            np.uint32(prompt_tokens),
            gen_len=max_new,
            temp=gen_settings.temp,
            top_p=gen_settings.top_p,
            top_k=gen_settings.top_k,
            tfs=gen_settings.tfs,
            typical=gen_settings.typical,
            top_a=gen_settings.top_a,
            numseqs=batch_count,
            repetition_penalty=gen_settings.rep_pen,
            rpslope=gen_settings.rep_pen_slope,
            rprange=gen_settings.rep_pen_range,
            soft_embeddings=utils.koboldai_vars.sp,
            soft_tokens=soft_tokens,
            sampler_order=gen_settings.sampler_order,
        )
        genout = np.array(genout)

        return GenerationResult(
            out_batches=genout,
            prompt=prompt_tokens,
            is_whole_generation=True,
            single_line=single_line,
        )


class HFTorchInferenceModel(InferenceModel):
    def __init__(
        self,
        model_name: str,
        lazy_load: bool,
        low_mem: bool,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.lazy_load = lazy_load
        self.low_mem = low_mem

        self.post_token_hooks = [
            Stoppers.core_stopper,
            PostTokenHooks.stream_tokens,
            Stoppers.dynamic_wi_scanner,
            Stoppers.chat_mode_stopper,
        ]

        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.capabilties = ModelCapabilities(
            embedding_manipulation=True,
            post_token_hooks=True,
            stopper_hooks=True,
            post_token_probs=True,
        )
        self._old_stopping_criteria = None

    def _post_load(self) -> None:
        print("HELLLOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        # Patch stopping_criteria

        class PTHStopper(StoppingCriteria):
            def __call__(
                hf_self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
            ) -> None:
                self._post_token_gen(input_ids)

                for stopper in self.stopper_hooks:
                    do_stop = stopper(input_ids)
                    if do_stop:
                        return True
                return False

        old_gsc = transformers.GenerationMixin._get_stopping_criteria

        def _get_stopping_criteria(
            hf_self,
            *args,
            **kwargs,
        ):
            stopping_criteria = old_gsc(hf_self, *args, **kwargs)
            stopping_criteria.insert(0, PTHStopper())
            return stopping_criteria

        use_core_manipulations.get_stopping_criteria = _get_stopping_criteria

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
    ) -> GenerationResult:
        if not isinstance(prompt_tokens, torch.Tensor):
            gen_in = torch.tensor(prompt_tokens, dtype=torch.long)[None]
        else:
            gen_in = prompt_tokens

        device = utils.get_auxilary_device()
        gen_in = gen_in.to(device)

        additional_bad_words_ids = [self.tokenizer.encode("\n")] if single_line else []

        with torch.no_grad():
            start_time = time.time()
            genout = self.model.generate(
                gen_in,
                do_sample=True,
                max_length=min(
                    len(prompt_tokens) + max_new, utils.koboldai_vars.max_length
                ),
                repetition_penalty=1.0,
                bad_words_ids=utils.koboldai_vars.badwordsids
                + additional_bad_words_ids,
                use_cache=True,
                num_return_sequences=batch_count,
            )
        logger.debug(
            "torch_raw_generate: run generator {}s".format(time.time() - start_time)
        )

        return GenerationResult(
            self,
            out_batches=genout,
            prompt=prompt_tokens,
            is_whole_generation=False,
            output_includes_prompt=True,
        )

    def _get_model(self, location: str, tf_kwargs: Dict):
        try:
            return AutoModelForCausalLM.from_pretrained(
                location,
                revision=utils.koboldai_vars.revision,
                cache_dir="cache",
                **tf_kwargs,
            )
        except Exception as e:
            if "out of memory" in traceback.format_exc().lower():
                raise RuntimeError(
                    "One of your GPUs ran out of memory when KoboldAI tried to load your model."
                )
            return GPTNeoForCausalLM.from_pretrained(
                location,
                revision=utils.koboldai_vars.revision,
                cache_dir="cache",
                **tf_kwargs,
            )

    def get_local_model_path(
        self, legacy: bool = False, ignore_existance: bool = False
    ) -> Optional[str]:
        """
        Returns a string of the model's path locally, or None if it is not downloaded.
        If ignore_existance is true, it will always return a path.
        """

        basename = utils.koboldai_vars.model.replace("/", "_")
        if legacy:
            ret = basename
        else:
            ret = os.path.join("models", basename)

        if os.path.isdir(ret) or ignore_existance:
            return ret
        return None

    def get_hidden_size(self) -> int:
        return self.model.get_input_embeddings().embedding_dim

    def _move_to_devices(self) -> None:
        if not utils.koboldai_vars.breakmodel:
            if utils.koboldai_vars.usegpu:
                self.model = self.model.half().to(utils.koboldai_vars.gpu_device)
            else:
                self.model = self.model.to("cpu").float()
            return

        for key, value in self.model.state_dict().items():
            target_dtype = (
                torch.float32 if breakmodel.primary_device == "cpu" else torch.float16
            )
            if value.dtype is not target_dtype:
                accelerate.utils.set_module_tensor_to_device(
                    self.model, key, target_dtype
                )

        disk_blocks = breakmodel.disk_blocks
        gpu_blocks = breakmodel.gpu_blocks
        ram_blocks = len(utils.layers_module_names) - sum(gpu_blocks)
        cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))
        device_map = {}

        for name in utils.layers_module_names:
            layer = int(name.rsplit(".", 1)[1])
            device = (
                ("disk" if layer < disk_blocks else "cpu")
                if layer < ram_blocks
                else bisect.bisect_right(cumulative_gpu_blocks, layer - ram_blocks)
            )
            device_map[name] = device

        for name in utils.get_missing_module_names(self.model, list(device_map.keys())):
            device_map[name] = breakmodel.primary_device

        breakmodel.dispatch_model_ex(
            self.model,
            device_map,
            main_device=breakmodel.primary_device,
            offload_buffers=True,
            offload_dir="accelerate-disk-cache",
        )

        gc.collect()
        return

    # Function to patch transformers to use our soft prompt
    def patch_embedding(self) -> None:
        if getattr(Embedding, "_koboldai_patch_causallm_model", None):
            Embedding._koboldai_patch_causallm_model = self.model
            return

        old_embedding_call = Embedding.__call__

        kai_model = self

        def new_embedding_call(self, input_ids, *args, **kwargs):
            # Don't touch embeddings for models other than the core inference model (that's us!)
            if (
                Embedding._koboldai_patch_causallm_model.get_input_embeddings()
                is not self
            ):
                return old_embedding_call(self, input_ids, *args, **kwargs)

            assert input_ids is not None

            if utils.koboldai_vars.sp is not None:
                shifted_input_ids = input_ids - kai_model.model.config.vocab_size

            input_ids.clamp_(max=kai_model.model.config.vocab_size - 1)
            inputs_embeds = old_embedding_call(self, input_ids, *args, **kwargs)

            if utils.koboldai_vars.sp is not None:
                utils.koboldai_vars.sp = utils.koboldai_vars.sp.to(
                    inputs_embeds.dtype
                ).to(inputs_embeds.device)
                inputs_embeds = torch.where(
                    (shifted_input_ids >= 0)[..., None],
                    utils.koboldai_vars.sp[shifted_input_ids.clamp(min=0)],
                    inputs_embeds,
                )

            return inputs_embeds

        Embedding.__call__ = new_embedding_call
        Embedding._koboldai_patch_causallm_model = self.model

    def _get_lazy_load_callback(self, n_layers: int, convert_to_float16: bool = True):
        if not self.lazy_load:
            return

        if utils.args.breakmodel_disklayers is not None:
            breakmodel.disk_blocks = utils.args.breakmodel_disklayers

        disk_blocks = breakmodel.disk_blocks
        gpu_blocks = breakmodel.gpu_blocks
        ram_blocks = ram_blocks = n_layers - sum(gpu_blocks)
        cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))

        def lazy_load_callback(
            model_dict: Dict[str, Union[torch_lazy_loader.LazyTensor, torch.Tensor]],
            f,
            **_,
        ):
            if lazy_load_callback.nested:
                return
            lazy_load_callback.nested = True

            device_map: Dict[str, Union[str, int]] = {}

            @functools.lru_cache(maxsize=None)
            def get_original_key(key):
                return max(
                    (
                        original_key
                        for original_key in utils.module_names
                        if original_key.endswith(key)
                    ),
                    key=len,
                )

            for key, value in model_dict.items():
                original_key = get_original_key(key)
                if isinstance(value, torch_lazy_loader.LazyTensor) and not any(
                    original_key.startswith(n) for n in utils.layers_module_names
                ):
                    device_map[key] = (
                        utils.koboldai_vars.gpu_device
                        if utils.koboldai_vars.hascuda and utils.koboldai_vars.usegpu
                        else "cpu"
                        if not utils.koboldai_vars.hascuda
                        or not utils.koboldai_vars.breakmodel
                        else breakmodel.primary_device
                    )
                else:
                    layer = int(
                        max(
                            (
                                n
                                for n in utils.layers_module_names
                                if original_key.startswith(n)
                            ),
                            key=len,
                        ).rsplit(".", 1)[1]
                    )
                    device = (
                        utils.koboldai_vars.gpu_device
                        if utils.koboldai_vars.hascuda and utils.koboldai_vars.usegpu
                        else "disk"
                        if layer < disk_blocks and layer < ram_blocks
                        else "cpu"
                        if not utils.koboldai_vars.hascuda
                        or not utils.koboldai_vars.breakmodel
                        else "shared"
                        if layer < ram_blocks
                        else bisect.bisect_right(
                            cumulative_gpu_blocks, layer - ram_blocks
                        )
                    )
                    device_map[key] = device

            if utils.num_shards is None or utils.current_shard == 0:
                utils.offload_index = {}
                if os.path.isdir("accelerate-disk-cache"):
                    # Delete all of the files in the disk cache folder without deleting the folder itself to allow people to create symbolic links for this folder
                    # (the folder doesn't contain any subfolders so os.remove will do just fine)
                    for filename in os.listdir("accelerate-disk-cache"):
                        try:
                            os.remove(os.path.join("accelerate-disk-cache", filename))
                        except OSError:
                            pass
                os.makedirs("accelerate-disk-cache", exist_ok=True)
                if utils.num_shards is not None:
                    num_tensors = len(
                        utils.get_sharded_checkpoint_num_tensors(
                            utils.from_pretrained_model_name,
                            utils.from_pretrained_index_filename,
                            **utils.from_pretrained_kwargs,
                        )
                    )
                else:
                    num_tensors = len(device_map)
                print(flush=True)
                utils.koboldai_vars.status_message = "Loading model"
                utils.koboldai_vars.total_layers = num_tensors
                utils.koboldai_vars.loaded_layers = 0
                utils.bar = tqdm(
                    total=num_tensors,
                    desc="Loading model tensors",
                    file=utils.UIProgressBarFile(),
                )

            with zipfile.ZipFile(f, "r") as z:
                try:
                    last_storage_key = None
                    zipfolder = os.path.basename(os.path.normpath(f)).split(".")[0]
                    f = None
                    current_offset = 0
                    able_to_pin_layers = True
                    if utils.num_shards is not None:
                        utils.current_shard += 1
                    for key in sorted(
                        device_map.keys(),
                        key=lambda k: (model_dict[k].key, model_dict[k].seek_offset),
                    ):
                        storage_key = model_dict[key].key
                        if (
                            storage_key != last_storage_key
                            or model_dict[key].seek_offset < current_offset
                        ):
                            last_storage_key = storage_key
                            if isinstance(f, zipfile.ZipExtFile):
                                f.close()
                            try:
                                f = z.open(f"archive/data/{storage_key}")
                            except:
                                f = z.open(f"{zipfolder}/data/{storage_key}")
                            current_offset = 0
                        if current_offset != model_dict[key].seek_offset:
                            f.read(model_dict[key].seek_offset - current_offset)
                            current_offset = model_dict[key].seek_offset
                        device = device_map[key]
                        size = functools.reduce(
                            lambda x, y: x * y, model_dict[key].shape, 1
                        )
                        dtype = model_dict[key].dtype
                        nbytes = (
                            size
                            if dtype is torch.bool
                            else size
                            * (
                                (
                                    torch.finfo
                                    if dtype.is_floating_point
                                    else torch.iinfo
                                )(dtype).bits
                                >> 3
                            )
                        )
                        # print(f"Transferring <{key}>  to  {f'({device.upper()})' if isinstance(device, str) else '[device ' + str(device) + ']'} ... ", end="", flush=True)
                        model_dict[key] = model_dict[key].materialize(
                            f, map_location="cpu"
                        )
                        if model_dict[key].dtype is torch.float32:
                            utils.koboldai_vars.fp32_model = True
                        if (
                            convert_to_float16
                            and breakmodel.primary_device != "cpu"
                            and utils.koboldai_vars.hascuda
                            and (
                                utils.koboldai_vars.breakmodel
                                or utils.koboldai_vars.usegpu
                            )
                            and model_dict[key].dtype is torch.float32
                        ):
                            model_dict[key] = model_dict[key].to(torch.float16)
                        if breakmodel.primary_device == "cpu" or (
                            not utils.koboldai_vars.usegpu
                            and not utils.koboldai_vars.breakmodel
                            and model_dict[key].dtype is torch.float16
                        ):
                            model_dict[key] = model_dict[key].to(torch.float32)
                        if device == "shared":
                            model_dict[key] = model_dict[key].to("cpu").detach_()
                            if able_to_pin_layers:
                                try:
                                    model_dict[key] = model_dict[key].pin_memory()
                                except:
                                    able_to_pin_layers = False
                        elif device == "disk":
                            accelerate.utils.offload_weight(
                                model_dict[key],
                                get_original_key(key),
                                "accelerate-disk-cache",
                                index=utils.offload_index,
                            )
                            model_dict[key] = model_dict[key].to("meta")
                        else:
                            model_dict[key] = model_dict[key].to(device)
                        # print("OK", flush=True)
                        current_offset += nbytes
                        utils.bar.update(1)
                        utils.koboldai_vars.loaded_layers += 1
                finally:
                    if (
                        utils.num_shards is None
                        or utils.current_shard >= utils.num_shards
                    ):
                        if utils.offload_index:
                            for name, tensor in utils.named_buffers:
                                dtype = tensor.dtype
                                if (
                                    convert_to_float16
                                    and breakmodel.primary_device != "cpu"
                                    and utils.koboldai_vars.hascuda
                                    and (
                                        utils.koboldai_vars.breakmodel
                                        or utils.koboldai_vars.usegpu
                                    )
                                ):
                                    dtype = torch.float16
                                if breakmodel.primary_device == "cpu" or (
                                    not utils.koboldai_vars.usegpu
                                    and not utils.koboldai_vars.breakmodel
                                ):
                                    dtype = torch.float32
                                if (
                                    name in model_dict
                                    and model_dict[name].dtype is not dtype
                                ):
                                    model_dict[name] = model_dict[name].to(dtype)
                                if tensor.dtype is not dtype:
                                    tensor = tensor.to(dtype)
                                if name not in utils.offload_index:
                                    accelerate.utils.offload_weight(
                                        tensor,
                                        name,
                                        "accelerate-disk-cache",
                                        index=utils.offload_index,
                                    )
                            accelerate.utils.save_offload_index(
                                utils.offload_index, "accelerate-disk-cache"
                            )
                        utils.bar.close()
                        utils.bar = None
                        utils.koboldai_vars.status_message = ""
                    lazy_load_callback.nested = False
                    if isinstance(f, zipfile.ZipExtFile):
                        f.close()

        lazy_load_callback.nested = False
        return lazy_load_callback

    @contextlib.contextmanager
    def _maybe_use_float16(self, always_use: bool = False):
        if always_use or (
            utils.koboldai_vars.hascuda
            and self.low_mem
            and (utils.koboldai_vars.usegpu or utils.koboldai_vars.breakmodel)
        ):
            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float16)
            yield True
            torch.set_default_dtype(original_dtype)
        else:
            yield False

    def breakmodel_device_list(self, n_layers, primary=None, selected=None):
        # TODO: Find a better place for this or rework this

        device_count = torch.cuda.device_count()
        if device_count < 2:
            primary = None
        gpu_blocks = breakmodel.gpu_blocks + (
            device_count - len(breakmodel.gpu_blocks)
        ) * [0]
        print(f"{colors.YELLOW}       DEVICE ID  |  LAYERS  |  DEVICE NAME{colors.END}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            if len(name) > 47:
                name = "..." + name[-44:]
            row_color = colors.END
            sep_color = colors.YELLOW
            print(
                f"{row_color}{colors.YELLOW + '->' + row_color if i == selected else '  '} {'(primary)' if i == primary else ' '*9} {i:3}  {sep_color}|{row_color}     {gpu_blocks[i]:3}  {sep_color}|{row_color}  {name}{colors.END}"
            )
        row_color = colors.END
        sep_color = colors.YELLOW
        print(
            f"{row_color}{colors.YELLOW + '->' + row_color if -1 == selected else '  '} {' '*9} N/A  {sep_color}|{row_color}     {breakmodel.disk_blocks:3}  {sep_color}|{row_color}  (Disk cache){colors.END}"
        )
        print(
            f"{row_color}   {' '*9} N/A  {sep_color}|{row_color}     {n_layers:3}  {sep_color}|{row_color}  (CPU){colors.END}"
        )

    def breakmodel_device_config(self, config):
        # TODO: Find a better place for this or rework this

        global breakmodel, generator
        import breakmodel

        n_layers = utils.num_layers(config)

        if utils.args.cpu:
            breakmodel.gpu_blocks = [0] * n_layers
            return

        elif (
            utils.args.breakmodel_gpulayers is not None
            or utils.args.breakmodel_disklayers is not None
        ):
            try:
                if not utils.args.breakmodel_gpulayers:
                    breakmodel.gpu_blocks = []
                else:
                    breakmodel.gpu_blocks = list(
                        map(int, utils.args.breakmodel_gpulayers.split(","))
                    )
                assert len(breakmodel.gpu_blocks) <= torch.cuda.device_count()
                s = n_layers
                for i in range(len(breakmodel.gpu_blocks)):
                    if breakmodel.gpu_blocks[i] <= -1:
                        breakmodel.gpu_blocks[i] = s
                        break
                    else:
                        s -= breakmodel.gpu_blocks[i]
                assert sum(breakmodel.gpu_blocks) <= n_layers
                n_layers -= sum(breakmodel.gpu_blocks)
                if utils.args.breakmodel_disklayers is not None:
                    assert utils.args.breakmodel_disklayers <= n_layers
                    breakmodel.disk_blocks = utils.args.breakmodel_disklayers
                    n_layers -= utils.args.breakmodel_disklayers
            except:
                logger.warning(
                    "--breakmodel_gpulayers is malformatted. Please use the --help option to see correct usage of --breakmodel_gpulayers. Defaulting to all layers on device 0."
                )
                breakmodel.gpu_blocks = [n_layers]
                n_layers = 0
        elif utils.args.breakmodel_layers is not None:
            breakmodel.gpu_blocks = [
                n_layers - max(0, min(n_layers, utils.args.breakmodel_layers))
            ]
            n_layers -= sum(breakmodel.gpu_blocks)
        elif utils.args.model is not None:
            logger.info("Breakmodel not specified, assuming GPU 0")
            breakmodel.gpu_blocks = [n_layers]
            n_layers = 0
        else:
            device_count = torch.cuda.device_count()
            if device_count > 1:
                print(
                    colors.CYAN
                    + "\nPlease select one of your GPUs to be your primary GPU."
                )
                print(
                    "VRAM usage in your primary GPU will be higher than for your other ones."
                )
                print("It is recommended you make your fastest GPU your primary GPU.")
                self.breakmodel_device_list(n_layers)
                while True:
                    primaryselect = input("device ID> ")
                    if (
                        primaryselect.isnumeric()
                        and 0 <= int(primaryselect) < device_count
                    ):
                        breakmodel.primary_device = int(primaryselect)
                        break
                    else:
                        print(
                            f"{colors.RED}Please enter an integer between 0 and {device_count-1}.{colors.END}"
                        )
            else:
                breakmodel.primary_device = 0

            print(
                colors.PURPLE
                + "\nIf you don't have enough VRAM to run the model on a single GPU"
            )
            print(
                "you can split the model between your CPU and your GPU(s), or between"
            )
            print("multiple GPUs if you have more than one.")
            print("By putting more 'layers' on a GPU or CPU, more computations will be")
            print(
                "done on that device and more VRAM or RAM will be required on that device"
            )
            print("(roughly proportional to number of layers).")
            print(
                "It should be noted that GPUs are orders of magnitude faster than the CPU."
            )
            print(
                f"This model has{colors.YELLOW} {n_layers} {colors.PURPLE}layers.{colors.END}\n"
            )

            for i in range(device_count):
                self.breakmodel_device_list(
                    n_layers, primary=breakmodel.primary_device, selected=i
                )
                print(
                    f"{colors.CYAN}\nHow many of the remaining{colors.YELLOW} {n_layers} {colors.CYAN}layers would you like to put into device {i}?\nYou can also enter -1 to allocate all remaining layers to this device.{colors.END}\n"
                )
                while True:
                    layerselect = input("# of layers> ")
                    if (
                        layerselect.isnumeric() or layerselect.strip() == "-1"
                    ) and -1 <= int(layerselect) <= n_layers:
                        layerselect = int(layerselect)
                        layerselect = n_layers if layerselect == -1 else layerselect
                        breakmodel.gpu_blocks.append(layerselect)
                        n_layers -= layerselect
                        break
                    else:
                        print(
                            f"{colors.RED}Please enter an integer between -1 and {n_layers}.{colors.END}"
                        )
                if n_layers == 0:
                    break

            if n_layers > 0:
                self.breakmodel_device_list(
                    n_layers, primary=breakmodel.primary_device, selected=-1
                )
                print(
                    f"{colors.CYAN}\nHow many of the remaining{colors.YELLOW} {n_layers} {colors.CYAN}layers would you like to put into the disk cache?\nYou can also enter -1 to allocate all remaining layers to this device.{colors.END}\n"
                )
                while True:
                    layerselect = input("# of layers> ")
                    if (
                        layerselect.isnumeric() or layerselect.strip() == "-1"
                    ) and -1 <= int(layerselect) <= n_layers:
                        layerselect = int(layerselect)
                        layerselect = n_layers if layerselect == -1 else layerselect
                        breakmodel.disk_blocks = layerselect
                        n_layers -= layerselect
                        break
                    else:
                        print(
                            f"{colors.RED}Please enter an integer between -1 and {n_layers}.{colors.END}"
                        )

        logger.init_ok("Final device configuration:", status="Info")
        self.breakmodel_device_list(n_layers, primary=breakmodel.primary_device)

        # If all layers are on the same device, use the old GPU generation mode
        while len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] == 0:
            breakmodel.gpu_blocks.pop()
        if len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] in (
            -1,
            utils.num_layers(config),
        ):
            utils.koboldai_vars.breakmodel = False
            utils.koboldai_vars.usegpu = True
            utils.koboldai_vars.gpu_device = len(breakmodel.gpu_blocks) - 1
            return

        if not breakmodel.gpu_blocks:
            logger.warning("Nothing assigned to a GPU, reverting to CPU only mode")
            import breakmodel

            breakmodel.primary_device = "cpu"
            utils.koboldai_vars.breakmodel = False
            utils.koboldai_vars.usegpu = False
            return


class GenericHFTorchInferenceModel(HFTorchInferenceModel):
    def _load(self, save_model: bool, initial_load: bool) -> None:
        utils.koboldai_vars.allowsp = True

        # Make model path the same as the model name to make this consistent
        # with the other loading method if it isn't a known model type. This
        # code is not just a workaround for below, it is also used to make the
        # behavior consistent with other loading methods - Henk717
        # if utils.koboldai_vars.model not in ["NeoCustom", "GPT2Custom"]:
        #     utils.koboldai_vars.custmodpth = utils.koboldai_vars.model

        if utils.koboldai_vars.model == "NeoCustom":
            utils.koboldai_vars.model = os.path.basename(
                os.path.normpath(utils.koboldai_vars.custmodpth)
            )

        # If we specify a model and it's in the root directory, we need to move
        # it to the models directory (legacy folder structure to new)
        if self.get_local_model_path(legacy=True):
            shutil.move(
                self.get_local_model_path(legacy=True, ignore_existance=True),
                self.get_local_model_path(ignore_existance=True),
            )

        # Get the model_type from the config or assume a model type if it isn't present
        try:
            model_config = AutoConfig.from_pretrained(
                self.get_local_model_path() or utils.koboldai_vars.model,
                revision=utils.koboldai_vars.revision,
                cache_dir="cache",
            )
            utils.koboldai_vars.model_type = model_config.model_type
        except ValueError as e:
            utils.koboldai_vars.model_type = {
                "NeoCustom": "gpt_neo",
                "GPT2Custom": "gpt2",
            }.get(utils.koboldai_vars.model)

            if not utils.koboldai_vars.model_type:
                logger.warning(
                    "No model type detected, assuming Neo (If this is a GPT2 model use the other menu option or --model GPT2Custom)"
                )
                utils.koboldai_vars.model_type = "gpt_neo"

        tf_kwargs = {
            "low_cpu_mem_usage": True,
        }

        if utils.koboldai_vars.model_type == "gpt2":
            # We must disable low_cpu_mem_usage and if using a GPT-2 model
            # because GPT-2 is not compatible with this feature yet.
            tf_kwargs.pop("low_cpu_mem_usage", None)

            # Also, lazy loader doesn't support GPT-2 models
            utils.koboldai_vars.lazy_load = False

        # If we're using torch_lazy_loader, we need to get breakmodel config
        # early so that it knows where to load the individual model tensors
        if (
            utils.koboldai_vars.lazy_load
            and utils.koboldai_vars.hascuda
            and utils.koboldai_vars.breakmodel
            and not utils.koboldai_vars.nobreakmodel
        ):
            self.breakmodel_device_config(model_config)

        if utils.koboldai_vars.lazy_load:
            # If we're using lazy loader, we need to figure out what the model's hidden layers are called
            with torch_lazy_loader.use_lazy_torch_load(
                dematerialized_modules=True, use_accelerate_init_empty_weights=True
            ):
                try:
                    metamodel = AutoModelForCausalLM.from_config(model_config)
                except Exception as e:
                    metamodel = GPTNeoForCausalLM.from_config(model_config)
                utils.layers_module_names = utils.get_layers_module_names(metamodel)
                utils.module_names = list(metamodel.state_dict().keys())
                utils.named_buffers = list(metamodel.named_buffers(recurse=True))

        # Download model from Huggingface if it does not exist, otherwise load locally
        with self._maybe_use_float16(), torch_lazy_loader.use_lazy_torch_load(
            enable=utils.koboldai_vars.lazy_load,
            callback=self._get_lazy_load_callback(utils.num_layers(model_config))
            if utils.koboldai_vars.lazy_load
            else None,
            dematerialized_modules=True,
        ):
            if utils.koboldai_vars.lazy_load:
                # torch_lazy_loader.py and low_cpu_mem_usage can't be used at the same time
                tf_kwargs.pop("low_cpu_mem_usage", None)

            self.tokenizer = self._get_tokenizer(self.get_local_model_path())

            if self.get_local_model_path():
                # Model is stored locally, load it.
                self.model = self._get_model(self.get_local_model_path(), tf_kwargs)
            else:
                # Model not stored locally, we need to download it.

                # _rebuild_tensor patch for casting dtype and supporting LazyTensors
                old_rebuild_tensor = torch._utils._rebuild_tensor

                def new_rebuild_tensor(
                    storage: Union[torch_lazy_loader.LazyTensor, torch.Storage],
                    storage_offset,
                    shape,
                    stride,
                ):
                    if not isinstance(storage, torch_lazy_loader.LazyTensor):
                        dtype = storage.dtype
                    else:
                        dtype = storage.storage_type.dtype
                        if not isinstance(dtype, torch.dtype):
                            dtype = storage.storage_type(0).dtype
                    if dtype is torch.float32 and len(shape) >= 2:
                        utils.koboldai_vars.fp32_model = True
                    return old_rebuild_tensor(storage, storage_offset, shape, stride)

                torch._utils._rebuild_tensor = new_rebuild_tensor
                self.model = self._get_model(utils.koboldai_vars.model, tf_kwargs)
                torch._utils._rebuild_tensor = old_rebuild_tensor

                if save_model:
                    self.tokenizer.save_pretrained(
                        self.get_local_model_path(ignore_existance=True)
                    )

                    if utils.koboldai_vars.fp32_model and not breakmodel.disk_blocks:
                        # Use save_pretrained to convert fp32 models to fp16,
                        # unless we are using disk cache because save_pretrained
                        # is not supported in that case
                        model = model.half()
                        model.save_pretrained(
                            self.get_local_model_path(ignore_existance=True),
                            max_shard_size="500MiB",
                        )

                    else:
                        # For fp16 models, we can just copy the model files directly
                        import transformers.configuration_utils
                        import transformers.modeling_utils
                        import transformers.file_utils
                        import huggingface_hub

                        legacy = packaging.version.parse(
                            transformers_version
                        ) < packaging.version.parse("4.22.0.dev0")
                        # Save the config.json
                        shutil.move(
                            os.path.realpath(
                                huggingface_hub.hf_hub_download(
                                    utils.koboldai_vars.model,
                                    transformers.configuration_utils.CONFIG_NAME,
                                    revision=utils.koboldai_vars.revision,
                                    cache_dir="cache",
                                    local_files_only=True,
                                    legacy_cache_layout=legacy,
                                )
                            ),
                            os.path.join(
                                self.get_local_model_path(ignore_existance=True),
                                transformers.configuration_utils.CONFIG_NAME,
                            ),
                        )

                        if utils.num_shards is None:
                            # Save the pytorch_model.bin or model.safetensors of an unsharded model
                            for possible_weight_name in [
                                transformers.modeling_utils.WEIGHTS_NAME,
                                "model.safetensors",
                            ]:
                                try:
                                    shutil.move(
                                        os.path.realpath(
                                            huggingface_hub.hf_hub_download(
                                                utils.koboldai_vars.model,
                                                possible_weight_name,
                                                revision=utils.koboldai_vars.revision,
                                                cache_dir="cache",
                                                local_files_only=True,
                                                legacy_cache_layout=legacy,
                                            )
                                        ),
                                        os.path.join(
                                            self.get_local_model_path(
                                                ignore_existance=True
                                            ),
                                            possible_weight_name,
                                        ),
                                    )
                                except Exception as e:
                                    if possible_weight_name == "model.safetensors":
                                        raise e
                        else:
                            # Handle saving sharded models

                            with open(utils.from_pretrained_index_filename) as f:
                                map_data = json.load(f)
                            filenames = set(map_data["weight_map"].values())
                            # Save the pytorch_model.bin.index.json of a sharded model
                            shutil.move(
                                os.path.realpath(utils.from_pretrained_index_filename),
                                os.path.join(
                                    self.get_local_model_path(ignore_existance=True),
                                    transformers.modeling_utils.WEIGHTS_INDEX_NAME,
                                ),
                            )
                            # Then save the pytorch_model-#####-of-#####.bin files
                            for filename in filenames:
                                shutil.move(
                                    os.path.realpath(
                                        huggingface_hub.hf_hub_download(
                                            utils.koboldai_vars.model,
                                            filename,
                                            revision=utils.koboldai_vars.revision,
                                            cache_dir="cache",
                                            local_files_only=True,
                                            legacy_cache_layout=legacy,
                                        )
                                    ),
                                    os.path.join(
                                        self.get_local_model_path(
                                            ignore_existance=True
                                        ),
                                        filename,
                                    ),
                                )
                    shutil.rmtree("cache/")

        if (
            utils.koboldai_vars.badwordsids is koboldai_settings.badwordsids_default
            and utils.koboldai_vars.model_type not in ("gpt2", "gpt_neo", "gptj")
        ):
            utils.koboldai_vars.badwordsids = [
                [v]
                for k, v in self.tokenizer.get_vocab().items()
                if any(c in str(k) for c in "<>[]")
                if utils.koboldai_vars.newlinemode != "s" or str(k) != "</s>"
            ]

        self.patch_embedding()

        if utils.koboldai_vars.hascuda:
            if utils.koboldai_vars.usegpu:
                # Use just VRAM
                self.model = self.model.half().to(utils.koboldai_vars.gpu_device)
            elif utils.koboldai_vars.breakmodel:
                # Use both RAM and VRAM (breakmodel)
                if not utils.koboldai_vars.lazy_load:
                    self.breakmodel_device_config(model.config)
                self._move_to_devices()
            elif breakmodel.disk_blocks > 0:
                # Use disk
                self._move_to_devices()
            elif breakmodel.disk_blocks > 0:
                self._move_to_devices()
            else:
                # Use CPU
                self.model = self.model.to("cpu").float()
        elif breakmodel.disk_blocks > 0:
            self._move_to_devices()
        else:
            self.model = self.model.to("cpu").float()
        self.model.kai_model = self
        utils.koboldai_vars.modeldim = self.get_hidden_size()


class CustomGPT2HFTorchInferenceModel(HFTorchInferenceModel):
    def _load(self, save_model: bool, initial_load: bool) -> None:
        utils.koboldai_vars.lazy_load = False

        model_path = None

        for possible_config_path in [
            utils.koboldai_vars.custmodpth,
            os.path.join("models", utils.koboldai_vars.custmodpth),
        ]:
            try:
                with open(
                    os.path.join(possible_config_path, "config.json"), "r"
                ) as file:
                    # Unused?
                    self.model_config = json.load(file)
                model_path = possible_config_path
                break
            except FileNotFoundError:
                pass

        if not model_path:
            raise RuntimeError("Empty model_path!")

        with self._maybe_use_float16():
            try:
                self.model = GPT2LMHeadModel.from_pretrained(
                    utils.koboldai_vars.custmodpth,
                    revision=utils.koboldai_vars.revision,
                    cache_dir="cache",
                )
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    utils.koboldai_vars.custmodpth,
                    revision=utils.koboldai_vars.revision,
                    cache_dir="cache",
                )
            except Exception as e:
                if "out of memory" in traceback.format_exc().lower():
                    raise RuntimeError(
                        "One of your GPUs ran out of memory when KoboldAI tried to load your model."
                    )
                raise e

        if save_model:
            self.model.save_pretrained(
                self.get_local_model_path(ignore_existance=True),
                max_shard_size="500MiB",
            )
            self.tokenizer.save_pretrained(
                self.get_local_model_path(ignore_existance=True)
            )

        utils.koboldai_vars.modeldim = self.get_hidden_size()

        # Is CUDA available? If so, use GPU, otherwise fall back to CPU
        if utils.koboldai_vars.hascuda and utils.koboldai_vars.usegpu:
            self.model = self.model.half().to(utils.koboldai_vars.gpu_device)
        else:
            self.model = self.model.to("cpu").float()

        self.patch_causal_lm()


class OpenAIAPIInferenceModel(InferenceModel):
    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.tokenizer = self._get_tokenizer("gpt2")

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
    ) -> GenerationResult:
        # Taken mainly from oairequest()

        decoded_prompt = utils.decodenewlines(self.tokenizer.decode(prompt_tokens))

        # Store context in memory to use it for comparison with generated content
        utils.koboldai_vars.lastctx = decoded_prompt

        # Build request JSON data
        # GooseAI is a subntype of OAI. So to check if it's this type, we check the configname as a workaround
        # as the koboldai_vars.model will always be OAI
        if "GooseAI" in utils.koboldai_vars.configname:
            reqdata = {
                "prompt": decoded_prompt,
                "max_tokens": max_new,
                "temperature": gen_settings.temp,
                "top_a": gen_settings.top_a,
                "top_p": gen_settings.top_p,
                "top_k": gen_settings.top_k,
                "tfs": gen_settings.tfs,
                "typical_p": gen_settings.typical,
                "repetition_penalty": gen_settings.rep_pen,
                "repetition_penalty_slope": gen_settings.rep_pen_slope,
                "repetition_penalty_range": gen_settings.rep_pen_range,
                "n": batch_count,
                # TODO: Implement streaming
                "stream": False,
            }
        else:
            reqdata = {
                "prompt": decoded_prompt,
                "max_tokens": max_new,
                "temperature": gen_settings.temp,
                "top_p": gen_settings.top_p,
                "frequency_penalty": gen_settings.rep_pen,
                "n": batch_count,
                "stream": False,
            }

        req = requests.post(
            utils.koboldai_vars.oaiurl,
            json=reqdata,
            headers={
                "Authorization": "Bearer " + utils.koboldai_vars.oaiapikey,
                "Content-Type": "application/json",
            },
        )

        j = req.json()

        if not req.ok:
            # Send error message to web client
            if "error" in j:
                error_type = j["error"]["type"]
                error_message = j["error"]["message"]
            else:
                error_type = "Unknown"
                error_message = "Unknown"
            raise OpenAIAPIError(error_type, error_message)

        outputs = [out["text"] for out in j["choices"]]
        return GenerationResult(
            model=self,
            out_batches=np.array([self.tokenizer.encode(x) for x in outputs]),
            prompt=prompt_tokens,
            is_whole_generation=True,
            single_line=single_line,
        )


class HordeInferenceModel(InferenceModel):
    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.tokenizer = self._get_tokenizer(
            utils.koboldai_vars.cluster_requested_models[0]
            if len(utils.koboldai_vars.cluster_requested_models) > 0
            else "gpt2",
        )

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
    ) -> GenerationResult:
        decoded_prompt = utils.decodenewlines(self.tokenizer.decode(prompt_tokens))

        # Store context in memory to use it for comparison with generated content
        utils.koboldai_vars.lastctx = decoded_prompt

        # Build request JSON data
        reqdata = {
            "max_length": max_new,
            "max_context_length": utils.koboldai_vars.max_length,
            "rep_pen": gen_settings.rep_pen,
            "rep_pen_slope": gen_settings.rep_pen_slope,
            "rep_pen_range": gen_settings.rep_pen_range,
            "temperature": gen_settings.temp,
            "top_p": gen_settings.top_p,
            "top_k": int(gen_settings.top_k),
            "top_a": gen_settings.top_a,
            "tfs": gen_settings.tfs,
            "typical": gen_settings.typical,
            "n": batch_count,
        }

        cluster_metadata = {
            "prompt": decoded_prompt,
            "params": reqdata,
            "models": [x for x in utils.koboldai_vars.cluster_requested_models if x],
            "trusted_workers": False,
        }

        client_agent = "KoboldAI:2.0.0:koboldai.org"
        cluster_headers = {
            "apikey": utils.koboldai_vars.horde_api_key,
            "Client-Agent": client_agent,
        }

        try:
            # Create request
            req = requests.post(
                utils.koboldai_vars.colaburl[:-8] + "/api/v2/generate/text/async",
                json=cluster_metadata,
                headers=cluster_headers,
            )
        except requests.exceptions.ConnectionError:
            errmsg = f"Horde unavailable. Please try again later"
            logger.error(errmsg)
            raise HordeException(errmsg)

        if req.status_code == 503:
            errmsg = f"KoboldAI API Error: No available KoboldAI servers found in Horde to fulfil this request using the selected models or other properties."
            logger.error(errmsg)
            raise HordeException(errmsg)
        elif not req.ok:
            errmsg = f"KoboldAI API Error: Failed to get a standard reply from the Horde. Please check the console."
            logger.error(errmsg)
            logger.error(f"HTTP {req.status_code}!!!")
            logger.error(req.text)
            raise HordeException(errmsg)

        try:
            req_status = req.json()
        except requests.exceptions.JSONDecodeError:
            errmsg = f"Unexpected message received from the Horde: '{req.text}'"
            logger.error(errmsg)
            raise HordeException(errmsg)

        request_id = req_status["id"]
        logger.debug("Horde Request ID: {}".format(request_id))

        # We've sent the request and got the ID back, now we need to watch it to see when it finishes
        finished = False

        cluster_agent_headers = {"Client-Agent": client_agent}

        while not finished:
            try:
                req = requests.get(
                    f"{utils.koboldai_vars.colaburl[:-8]}/api/v2/generate/text/status/{request_id}",
                    headers=cluster_agent_headers,
                )
            except requests.exceptions.ConnectionError:
                errmsg = f"Horde unavailable. Please try again later"
                logger.error(errmsg)
                raise HordeException(errmsg)

            if not req.ok:
                errmsg = f"KoboldAI API Error: Failed to get a standard reply from the Horde. Please check the console."
                logger.error(req.text)
                raise HordeException(errmsg)

            try:
                req_status = req.json()
            except requests.exceptions.JSONDecodeError:
                errmsg = (
                    f"Unexpected message received from the KoboldAI Horde: '{req.text}'"
                )
                logger.error(errmsg)
                raise HordeException(errmsg)

            if "done" not in req_status:
                errmsg = f"Unexpected response received from the KoboldAI Horde: '{req_status}'"
                logger.error(errmsg)
                raise HordeException(errmsg)

            finished = req_status["done"]
            utils.koboldai_vars.horde_wait_time = req_status["wait_time"]
            utils.koboldai_vars.horde_queue_position = req_status["queue_position"]
            utils.koboldai_vars.horde_queue_size = req_status["waiting"]

            if not finished:
                logger.debug(req_status)
                time.sleep(1)

        logger.debug("Last Horde Status Message: {}".format(req_status))

        if req_status["faulted"]:
            raise HordeException("Horde Text generation faulted! Please try again.")

        generations = req_status["generations"]
        gen_servers = [(cgen["worker_name"], cgen["worker_id"]) for cgen in generations]
        logger.info(f"Generations by: {gen_servers}")

        return GenerationResult(
            model=self,
            out_batches=np.array(
                [self.tokenizer.encode(cgen["text"]) for cgen in generations]
            ),
            prompt=prompt_tokens,
            is_whole_generation=True,
            single_line=single_line,
        )


class ColabInferenceModel(InferenceModel):
    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.tokenizer = self._get_tokenizer("EleutherAI/gpt-neo-2.7B")

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
    ):
        decoded_prompt = utils.decodenewlines(self.tokenizer.decode(prompt_tokens))

        # Store context in memory to use it for comparison with generated content
        utils.koboldai_vars.lastctx = decoded_prompt

        # Build request JSON data
        reqdata = {
            "text": decoded_prompt,
            "min": 0,
            "max": max_new,
            "rep_pen": gen_settings.rep_pen,
            "rep_pen_slope": gen_settings.rep_pen_slope,
            "rep_pen_range": gen_settings.rep_pen_range,
            "temperature": gen_settings.temp,
            "top_p": gen_settings.top_p,
            "top_k": gen_settings.top_k,
            "tfs": gen_settings.tfs,
            "typical": gen_settings.typical,
            "topa": gen_settings.top_a,
            "numseqs": batch_count,
            "retfultxt": False,
        }

        # Create request
        req = requests.post(utils.koboldai_vars.colaburl, json=reqdata)

        if req.status_code != 200:
            raise ColabException(f"Bad status code {req.status_code}")

        # Deal with the response
        js = req.json()["data"]

        # Try to be backwards compatible with outdated colab
        if "text" in js:
            genout = [utils.getnewcontent(js["text"], self.tokenizer)]
        else:
            genout = js["seqs"]

        return GenerationResult(
            model=self,
            out_batches=np.array([self.tokenizer.encode(x) for x in genout]),
            prompt=prompt_tokens,
            is_whole_generation=True,
            single_line=single_line,
        )


class APIInferenceModel(InferenceModel):
    def _load(self, save_model: bool, initial_load: bool) -> None:
        tokenizer_id = requests.get(
            utils.koboldai_vars.colaburl[:-8] + "/api/v1/model",
        ).json()["result"]
        self.tokenizer = self._get_tokenizer(tokenizer_id)

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
    ):
        decoded_prompt = utils.decodenewlines(self.tokenizer.decode(prompt_tokens))

        # Store context in memory to use it for comparison with generated content
        utils.koboldai_vars.lastctx = decoded_prompt

        # Build request JSON data
        reqdata = {
            "prompt": decoded_prompt,
            "max_length": max_new,
            "max_context_length": utils.koboldai_vars.max_length,
            "rep_pen": gen_settings.rep_pen,
            "rep_pen_slope": gen_settings.rep_pen_slope,
            "rep_pen_range": gen_settings.rep_pen_range,
            "temperature": gen_settings.temp,
            "top_p": gen_settings.top_p,
            "top_k": gen_settings.top_k,
            "top_a": gen_settings.top_a,
            "tfs": gen_settings.tfs,
            "typical": gen_settings.typical,
            "n": batch_count,
        }

        # Create request
        while True:
            req = requests.post(
                utils.koboldai_vars.colaburl[:-8] + "/api/v1/generate",
                json=reqdata,
            )
            if (
                req.status_code == 503
            ):  # Server is currently generating something else so poll until it's our turn
                time.sleep(1)
                continue

            js = req.json()
            if req.status_code != 200:
                logger.error(json.dumps(js, indent=4))
                raise APIException(f"Bad API status code {req.status_code}")

            genout = [obj["text"] for obj in js["results"]]
            return GenerationResult(
                model=self,
                out_batches=np.array([self.tokenizer.encode(x) for x in genout]),
                prompt=prompt_tokens,
                is_whole_generation=True,
                single_line=single_line,
            )
