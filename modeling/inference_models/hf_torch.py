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
from typing import Dict, List, Optional, Union

import torch
from torch.nn import Embedding
import transformers
from transformers import (
    StoppingCriteria,
    GPTNeoForCausalLM,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    LogitsProcessorList,
)

import utils
import modeling.lazy_loader as lazy_loader
from logger import logger, Colors

from modeling import warpers
from modeling.warpers import Warper
from modeling.stoppers import Stoppers
from modeling.post_token_hooks import PostTokenHooks
from modeling.inference_models.hf import HFInferenceModel
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
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
    def __init__(self) -> None:
        super().__init__()
        self.hf_torch = True
        self.lazy_load = True
        self.low_mem = False
        self.nobreakmodel = False

        self.post_token_hooks = [
            PostTokenHooks.stream_tokens,
        ]

        self.stopper_hooks = [
            Stoppers.core_stopper,
            Stoppers.dynamic_wi_scanner,
            Stoppers.singleline_stopper,
            Stoppers.chat_mode_stopper,
            Stoppers.stop_sequence_stopper,
        ]

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

            assert scores is not None, f"Scores are None; warper '{warper}' is to blame"

            if LOG_SAMPLER_NO_EFFECT:
                if torch.equal(pre, scores):
                    logger.info(warper, "had no effect on the scores.")
                pre = torch.Tensor(scores)
        return scores

    def get_model_type(self) -> str:
        if not self.model_config:
            return "Read Only"

        if not isinstance(self.model_config, dict):
            return str(self.model_config.model_type)

        model_type = self.model_config.get("model_type")

        if model_type:
            return model_type

        if utils.koboldai_vars.mode.endswith("gpt2"):
            return "gpt2"
        else:
            return "Unknown"

    def get_auxilary_device(self):
        """Get device auxilary tensors like inputs should be stored on."""

        # NOTE: TPU isn't a torch device, so TPU stuff gets sent to CPU.
        if utils.koboldai_vars.hascuda and self.usegpu:
            return utils.koboldai_vars.gpu_device
        elif utils.koboldai_vars.hascuda and self.breakmodel:
            import breakmodel
            return breakmodel.primary_device
        return "cpu"

    def _post_load(m_self) -> None:

        if not utils.koboldai_vars.model_type:
            utils.koboldai_vars.model_type = m_self.get_model_type()

        # Patch stopping_criteria
        class PTHStopper(StoppingCriteria):
            def __call__(
                hf_self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
            ) -> None:
                m_self._post_token_gen(input_ids)

                for stopper in m_self.stopper_hooks:
                    do_stop = stopper(m_self, input_ids)
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

        def new_get_logits_processor(*args, **kwargs) -> LogitsProcessorList:
            processors = new_get_logits_processor.old_get_logits_processor(
                *args, **kwargs
            )
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
                scores = m_self._apply_warpers(scores=scores, input_ids=input_ids)

                for processor in m_self.logits_processors:
                    scores = processor(m_self, scores=scores, input_ids=input_ids)
                    assert (
                        scores is not None
                    ), f"Scores are None; processor '{processor}' is to blame"
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

        return super()._post_load()

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:
        if not isinstance(prompt_tokens, torch.Tensor):
            gen_in = torch.tensor(prompt_tokens, dtype=torch.long)[None]
        else:
            gen_in = prompt_tokens

        device = self.get_auxilary_device()
        gen_in = gen_in.to(device)

        additional_bad_words_ids = [self.tokenizer.encode("\n")] if single_line else []

        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            start_time = time.time()
            genout = self.model.generate(
                gen_in,
                do_sample=True,
                max_length=min(
                    len(prompt_tokens) + max_new, utils.koboldai_vars.max_length
                ),
                repetition_penalty=1.0,
                bad_words_ids=self.badwordsids
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
        tf_kwargs["revision"] = utils.koboldai_vars.revision
        tf_kwargs["cache_dir"] = "cache"

        # If we have model hints for legacy model, use them rather than fall back.
        try:
            if self.model_name == "GPT2Custom":
                return GPT2LMHeadModel.from_pretrained(location, **tf_kwargs)
            elif self.model_name == "NeoCustom":
                return GPTNeoForCausalLM.from_pretrained(location, **tf_kwargs)
        except Exception as e:
            logger.warning(f"{self.model_name} is a no-go; {e} - Falling back to auto.")

        # Try to determine model type from either AutoModel or falling back to legacy
        try:
            return AutoModelForCausalLM.from_pretrained(location, **tf_kwargs)
        except Exception as e:
            traceback_string = traceback.format_exc().lower()

            if "out of memory" in traceback_string:
                raise RuntimeError(
                    "One of your GPUs ran out of memory when KoboldAI tried to load your model."
                )

            # Model corrupted or serious loading problem. Stop here.
            if "invalid load key" in traceback_string:
                logger.error("Invalid load key! Aborting.")
                raise

            logger.warning(f"Fell back to GPT2LMHeadModel due to {e}")
            logger.debug(traceback.format_exc())

            try:
                return GPT2LMHeadModel.from_pretrained(location, **tf_kwargs)
            except Exception as e:
                logger.warning(f"Fell back to GPTNeoForCausalLM due to {e}")
                logger.debug(traceback.format_exc())
                return GPTNeoForCausalLM.from_pretrained(location, **tf_kwargs)

    def get_hidden_size(self) -> int:
        return self.model.get_input_embeddings().embedding_dim

    def _will_load_with_safetensors(self) -> bool:
        path = self.get_local_model_path()

        # TODO: This might mess up download to run
        if not path:
            return False

        if not os.path.exists(os.path.join(path, "model.safetensors")):
            return False

        return True

    def _move_to_devices(self) -> None:
        for key, value in self.model.state_dict().items():
            target_dtype = (
                torch.float32 if breakmodel.primary_device == "cpu" else torch.float16
            )
            if value.dtype is not target_dtype:
                accelerate.utils.set_module_tensor_to_device(
                    self.model,
                    tensor_name=key,
                    device=torch.device(value.device),
                    value=value,
                    dtype=target_dtype,
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


        disk_blocks = breakmodel.disk_blocks
        gpu_blocks = breakmodel.gpu_blocks
        ram_blocks = ram_blocks = n_layers - sum(gpu_blocks)
        cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))

        def lazy_load_callback(
            model_dict: Dict[str, Union[lazy_loader.LazyTensor, torch.Tensor]],
            f,
            is_safetensors: bool = False,
            **_,
        ):
            if lazy_load_callback.nested:
                return
            lazy_load_callback.nested = True

            device_map: Dict[str, Union[str, int]] = {}

            @functools.lru_cache(maxsize=None)
            def get_original_key(key) -> Optional[str]:
                key_candidates = [
                    original_key
                    for original_key in utils.module_names
                    if original_key.endswith(key)
                ]

                if not key_candidates:
                    logger.debug(f"!!! No key candidates for {key}")
                    return None

                return max(key_candidates, key=len)

            for key, value in model_dict.items():
                original_key = get_original_key(key)

                if not original_key:
                    continue

                if isinstance(value, lazy_loader.LazyTensor) and not any(
                    original_key.startswith(n) for n in utils.layers_module_names
                ):
                    device_map[key] = (
                        utils.koboldai_vars.gpu_device
                        if utils.koboldai_vars.hascuda and self.usegpu
                        else "cpu"
                        if not utils.koboldai_vars.hascuda
                        or not self.breakmodel
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
                        if utils.koboldai_vars.hascuda and self.usegpu
                        else "disk"
                        if layer < disk_blocks and layer < ram_blocks
                        else "cpu"
                        if not utils.koboldai_vars.hascuda
                        or not self.breakmodel
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
                            is_safetensors=is_safetensors,
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
                    position=1
                )

            if not is_safetensors:
                # Torch lazyload
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
                            key=lambda k: (
                                model_dict[k].key,
                                model_dict[k].seek_offset,
                            ),
                        ):
                            storage_key = model_dict[key].key
                            if (
                                storage_key != last_storage_key
                                or model_dict[key].seek_offset < current_offset
                            ):
                                last_storage_key = storage_key
                                if isinstance(f, zipfile.ZipExtFile):
                                    f.close()
                                ziproot = z.namelist()[0].split("/")[0]
                                f = z.open(f"{ziproot}/data/{storage_key}")
                                
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
                            #logger.debug(f"Transferring <{key}>  to  {f'({device.upper()})' if isinstance(device, str) else '[device ' + str(device) + ']'} ... ")
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
                                    self.breakmodel
                                    or self.usegpu
                                )
                                and model_dict[key].dtype is torch.float32
                            ):
                                model_dict[key] = model_dict[key].to(torch.float16)
                            if breakmodel.primary_device == "cpu" or (
                                not self.usegpu
                                and not self.breakmodel
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
                                            self.breakmodel
                                            or self.usegpu
                                        )
                                    ):
                                        dtype = torch.float16
                                    if breakmodel.primary_device == "cpu" or (
                                        not self.usegpu
                                        and not self.breakmodel
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
            else:
                # Loading with safetensors
                try:
                    able_to_pin_layers = True

                    if utils.num_shards is not None:
                        utils.current_shard += 1

                    for key in sorted(
                        device_map.keys(),
                        key=lambda k: model_dict[k].key,
                    ):
                        storage_key = model_dict[key].key

                        device = device_map[key]

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
                                self.breakmodel
                                or self.usegpu
                            )
                            and model_dict[key].dtype is torch.float32
                        ):
                            model_dict[key] = model_dict[key].to(torch.float16)

                        if breakmodel.primary_device == "cpu" or (
                            not self.usegpu
                            and not self.breakmodel
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
                                        self.breakmodel
                                        or self.usegpu
                                    )
                                ):
                                    dtype = torch.float16
                                if breakmodel.primary_device == "cpu" or (
                                    not self.usegpu
                                    and not self.breakmodel
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

        lazy_load_callback.nested = False
        return lazy_load_callback

    @contextlib.contextmanager
    def _maybe_use_float16(self, always_use: bool = False):
        if always_use or (
            utils.koboldai_vars.hascuda
            and self.low_mem
            and (self.usegpu or self.breakmodel)
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
        logger.debug("n_layers: {}".format(n_layers))
        logger.debug("gpu blocks: {}".format(breakmodel.gpu_blocks))
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

        logger.debug("gpu blocks before modification: {}".format(breakmodel.gpu_blocks))

        if utils.args.cpu:
            breakmodel.gpu_blocks = [0] * n_layers
            return

        elif breakmodel.gpu_blocks == []:
            logger.info("Breakmodel not specified, assuming GPU 0")
            breakmodel.gpu_blocks = [n_layers]
            n_layers = 0
        
        else:
            s = n_layers
            for i in range(len(breakmodel.gpu_blocks)):
                if breakmodel.gpu_blocks[i] <= -1:
                    breakmodel.gpu_blocks[i] = s
                    break
                else:
                    s -= breakmodel.gpu_blocks[i]
            assert sum(breakmodel.gpu_blocks) <= n_layers
            n_layers -= sum(breakmodel.gpu_blocks)
            if breakmodel.disk_blocks is not None:
                assert breakmodel.disk_blocks <= n_layers
                n_layers -= breakmodel.disk_blocks

        logger.init_ok("Final device configuration:", status="Info")
        self.breakmodel_device_list(n_layers, primary=breakmodel.primary_device)
        with open("settings/{}.breakmodel".format(self.model_name.replace("/", "_")), "w") as file:
            file.write("{}\n{}".format(",".join(map(str, breakmodel.gpu_blocks)), breakmodel.disk_blocks))

        # If all layers are on the same device, use the old GPU generation mode
        while len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] == 0:
            breakmodel.gpu_blocks.pop()
        self.breakmodel = True
        if len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] in (
            -1,
            utils.num_layers(config),
        ):
            logger.debug("All layers on same GPU. Breakmodel disabled")
            self.breakmodel = False
            self.usegpu = True
            utils.koboldai_vars.gpu_device = len(breakmodel.gpu_blocks) - 1
            return

        if not breakmodel.gpu_blocks:
            logger.warning("Nothing assigned to a GPU, reverting to CPU only mode")
            import breakmodel

            breakmodel.primary_device = "cpu"
            self.breakmodel = False
            self.usegpu = False
            return
