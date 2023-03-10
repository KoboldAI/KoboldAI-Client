from __future__ import annotations

import gc
import os
import time
import bisect
import zipfile
import functools
import itertools
import traceback
import contextlib
from tqdm.auto import tqdm
from typing import Dict, List, Union

import torch
from torch.nn import Embedding
import transformers
from transformers import (
    StoppingCriteria,
    GPTNeoForCausalLM,
    AutoModelForCausalLM,
    LogitsProcessorList,
    LogitsProcessor,
)

import utils
import torch_lazy_loader
from logger import logger, Colors

from modeling import warpers
from modeling import inference_model
from modeling.warpers import Warper
from modeling.stoppers import Stoppers
from modeling.post_token_hooks import PostTokenHooks
from modeling.inference_models.hf import HFInferenceModel
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
    ModelCapabilities,
    use_core_manipulations,
)

try:
    import breakmodel
    import accelerate.utils
except ModuleNotFoundError as e:
    if not utils.koboldai_vars.use_colab_tpu:
        raise e

# When set to true, messages will appear in the console if samplers are not
# changing the scores. Keep in mind some samplers don't always change the
# scores for each token.
LOG_SAMPLER_NO_EFFECT = False


class HFTorchInferenceModel(HFInferenceModel):
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
            PostTokenHooks.stream_tokens,
        ]

        self.stopper_hooks = [
            Stoppers.core_stopper,
            Stoppers.dynamic_wi_scanner,
            Stoppers.singleline_stopper,
            Stoppers.chat_mode_stopper,
        ]

        self.model = None
        self.tokenizer = None
        self.capabilties = ModelCapabilities(
            embedding_manipulation=True,
            post_token_hooks=True,
            stopper_hooks=True,
            post_token_probs=True,
        )
        self._old_stopping_criteria = None

    def _apply_warpers(
        self, scores: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        warpers.update_settings()

        if LOG_SAMPLER_NO_EFFECT:
            pre = torch.Tensor(scores)

        for sid in utils.koboldai_vars.sampler_order:
            warper = Warper.from_id(sid)

            if not warper.value_is_valid():
                continue

            if warper == warpers.RepetitionPenalty:
                # Rep pen needs more data than other samplers
                scores = warper.torch(scores, input_ids=input_ids)
            else:
                scores = warper.torch(scores)

            if LOG_SAMPLER_NO_EFFECT:
                if torch.equal(pre, scores):
                    logger.info(warper, "had no effect on the scores.")
                pre = torch.Tensor(scores)
        return scores

    def _post_load(model_self) -> None:
        # Patch stopping_criteria

        class PTHStopper(StoppingCriteria):
            def __call__(
                hf_self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
            ) -> None:
                model_self._post_token_gen(input_ids)

                for stopper in model_self.stopper_hooks:
                    do_stop = stopper(model_self, input_ids)
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

        # Patch logitswarpers

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
                    return [model_self.tokenizer.encode(no_brackets)]

                # Handle untamperable phrases
                if not self._allow_leftwards_tampering(phrase):
                    return [model_self.tokenizer.encode(phrase)]

                # Handle slight alterations to original phrase
                phrase = phrase.strip(" ")
                ret = []

                for alt_phrase in [phrase, f" {phrase}"]:
                    ret.append(model_self.tokenizer.encode(alt_phrase))

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
                    for token, bias in self._get_biased_tokens(
                        input_ids[batch]
                    ).items():
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

                scores = torch.Tensor(
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
                    if (
                        option["Pinned"]
                        or option["Previous Selection"]
                        or option["Edited"]
                    ):
                        option_offset = x + 1
            batch_offset = (
                int(
                    (utils.koboldai_vars.generated_tkns - 1)
                    / utils.koboldai_vars.genamt
                )
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
            processors = new_get_logits_processor.old_get_logits_processor(
                *args, **kwargs
            )
            # TODOB4MERGE: These two
            # processors.insert(0, LuaLogitsProcessor())
            # processors.append(PhraseBiasLogitsProcessor())
            return processors

        use_core_manipulations.get_logits_processor = new_get_logits_processor
        new_get_logits_processor.old_get_logits_processor = (
            transformers.GenerationMixin._get_logits_processor
        )

        class KoboldLogitsWarperList(LogitsProcessorList):
            def __init__(self):
                pass

            def __call__(
                lw_self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                *args,
                **kwargs,
            ):
                scores = model_self._apply_warpers(scores=scores, input_ids=input_ids)
                visualize_probabilities(model_self, scores)
                return scores

        def new_get_logits_warper(
            beams: int = 1,
        ) -> LogitsProcessorList:
            return KoboldLogitsWarperList()

        def new_sample(self, *args, **kwargs):
            assert kwargs.pop("logits_warper", None) is not None
            kwargs["logits_warper"] = new_get_logits_warper(
                beams=1,
            )
            if utils.koboldai_vars.newlinemode in ["s", "ns"]:
                kwargs["eos_token_id"] = -1
                kwargs.setdefault("pad_token_id", 2)
            return new_sample.old_sample(self, *args, **kwargs)

        new_sample.old_sample = transformers.GenerationMixin.sample
        use_core_manipulations.sample = new_sample

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
        **kwargs,
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
            print("Fell back for model due to", e)

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
        print(f"{Colors.YELLOW}       DEVICE ID  |  LAYERS  |  DEVICE NAME{Colors.END}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            if len(name) > 47:
                name = "..." + name[-44:]
            row_color = Colors.END
            sep_color = Colors.YELLOW
            print(
                f"{row_color}{Colors.YELLOW + '->' + row_color if i == selected else '  '} {'(primary)' if i == primary else ' '*9} {i:3}  {sep_color}|{row_color}     {gpu_blocks[i]:3}  {sep_color}|{row_color}  {name}{Colors.END}"
            )
        row_color = Colors.END
        sep_color = Colors.YELLOW
        print(
            f"{row_color}{Colors.YELLOW + '->' + row_color if -1 == selected else '  '} {' '*9} N/A  {sep_color}|{row_color}     {breakmodel.disk_blocks:3}  {sep_color}|{row_color}  (Disk cache){Colors.END}"
        )
        print(
            f"{row_color}   {' '*9} N/A  {sep_color}|{row_color}     {n_layers:3}  {sep_color}|{row_color}  (CPU){Colors.END}"
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
                    Colors.CYAN
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
                            f"{Colors.RED}Please enter an integer between 0 and {device_count-1}.{Colors.END}"
                        )
            else:
                breakmodel.primary_device = 0

            print(
                Colors.PURPLE
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
                f"This model has{Colors.YELLOW} {n_layers} {Colors.PURPLE}layers.{Colors.END}\n"
            )

            for i in range(device_count):
                self.breakmodel_device_list(
                    n_layers, primary=breakmodel.primary_device, selected=i
                )
                print(
                    f"{Colors.CYAN}\nHow many of the remaining{Colors.YELLOW} {n_layers} {Colors.CYAN}layers would you like to put into device {i}?\nYou can also enter -1 to allocate all remaining layers to this device.{Colors.END}\n"
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
                            f"{Colors.RED}Please enter an integer between -1 and {n_layers}.{Colors.END}"
                        )
                if n_layers == 0:
                    break

            if n_layers > 0:
                self.breakmodel_device_list(
                    n_layers, primary=breakmodel.primary_device, selected=-1
                )
                print(
                    f"{Colors.CYAN}\nHow many of the remaining{Colors.YELLOW} {n_layers} {Colors.CYAN}layers would you like to put into the disk cache?\nYou can also enter -1 to allocate all remaining layers to this device.{Colors.END}\n"
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
                            f"{Colors.RED}Please enter an integer between -1 and {n_layers}.{Colors.END}"
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
