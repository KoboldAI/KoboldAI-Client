from __future__ import annotations
from dataclasses import dataclass

import os
import time
import bisect
import itertools
import traceback
import contextlib
from torch import nn
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

# When set to true, messages will appear in the console if samplers are not
# changing the scores. Keep in mind some samplers don't always change the
# scores for each token.
LOG_SAMPLER_NO_EFFECT = False

class BreakmodelConfig:
    def __init__(self) -> None:
        self.disk_blocks = 0
        self.gpu_blocks = []
        self.primary_device = 0 if torch.cuda.device_count() > 0 else "cpu"

    def get_device_map(self, model: nn.Module) -> dict:
        ram_blocks = len(utils.layers_module_names) - sum(self.gpu_blocks)
        cumulative_gpu_blocks = tuple(itertools.accumulate(self.gpu_blocks))
        device_map = {}

        for name in utils.layers_module_names:
            layer = int(name.rsplit(".", 1)[1])
            device = (
                ("disk" if layer < self.disk_blocks else "cpu")
                if layer < ram_blocks
                else bisect.bisect_right(cumulative_gpu_blocks, layer - ram_blocks)
            )
            device_map[name] = device

        for name in utils.get_missing_module_names(model, list(device_map.keys())):
            device_map[name] = self.primary_device

        return device_map


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
        self.breakmodel_config = BreakmodelConfig()

    def set_input_parameters(self, parameters):
        ret = super().set_input_parameters(parameters)

        # Hook onto input param setting for setting breakmodel stuff
        if self.breakmodel:
            self.breakmodel_config.gpu_blocks = self.layers
            self.breakmodel_config.disk_blocks = self.disk_layers

        return ret

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

        device = utils.get_auxilary_device()
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
                bad_words_ids=self.badwordsids + additional_bad_words_ids,
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

        if self.lazy_load:
            tf_kwargs.pop("low_cpu_mem_usage", None)

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
            if self.lazy_load:
                with lazy_loader.use_lazy_load(dematerialized_modules=True):
                    metamodel = AutoModelForCausalLM.from_config(self.model_config)
                    tf_kwargs["device_map"] = self.breakmodel_config.get_device_map(metamodel)
                    print("Rodger rodger", tf_kwargs)

            with lazy_loader.use_lazy_load(
                enable=self.lazy_load,
                # DO NOT DEMATERIALIZE MODULES / INIT WEIGHTS EMPTY!!! IT WILL EXPLODE!!!!!!!
                dematerialized_modules=False,
            ):
                model = AutoModelForCausalLM.from_pretrained(
                    location,
                    offload_folder="accelerate-disk-cache",
                    torch_dtype=torch.float16,
                    **tf_kwargs,
                )

            return model
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
        device_count = torch.cuda.device_count()
        if device_count < 2:
            primary = None

        logger.debug("n_layers: {}".format(n_layers))
        logger.debug("gpu blocks: {}".format(self.breakmodel_config.gpu_blocks))

        gpu_blocks = self.breakmodel_config.gpu_blocks + (
            device_count - len(self.breakmodel_config.gpu_blocks)
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
            f"{row_color}{Colors.YELLOW + '->' + row_color if -1 == selected else '  '} {' '*9} N/A  {sep_color}|{row_color}     {self.breakmodel_config.disk_blocks:3}  {sep_color}|{row_color}  (Disk cache){Colors.END}"
        )
        print(
            f"{row_color}   {' '*9} N/A  {sep_color}|{row_color}     {n_layers:3}  {sep_color}|{row_color}  (CPU){Colors.END}"
        )

    def breakmodel_device_config(self, config):
        n_layers = utils.num_layers(config)

        logger.debug("gpu blocks before modification: {}".format(self.breakmodel_config.gpu_blocks))

        if utils.args.cpu:
            self.breakmodel_config.gpu_blocks = [0] * n_layers
            return

        elif self.breakmodel_config.gpu_blocks == []:
            logger.info("Breakmodel not specified, assuming GPU 0")
            self.breakmodel_config.gpu_blocks = [n_layers]
            n_layers = 0

        else:
            s = n_layers
            for i in range(len(self.breakmodel_config.gpu_blocks)):
                if self.breakmodel_config.gpu_blocks[i] <= -1:
                    self.breakmodel_config.gpu_blocks[i] = s
                    break
                else:
                    s -= self.breakmodel_config.gpu_blocks[i]
            assert sum(self.breakmodel_config.gpu_blocks) <= n_layers
            n_layers -= sum(self.breakmodel_config.gpu_blocks)
            if self.breakmodel_config.disk_blocks is not None:
                assert self.breakmodel_config.disk_blocks <= n_layers
                n_layers -= self.breakmodel_config.disk_blocks

        logger.init_ok("Final device configuration:", status="Info")
        self.breakmodel_device_list(n_layers, primary=self.breakmodel_config.primary_device)
        with open(
            "settings/{}.breakmodel".format(self.model_name.replace("/", "_")), "w"
        ) as file:
            file.write(
                "{}\n{}".format(
                    ",".join(map(str, self.breakmodel_config.gpu_blocks)), self.breakmodel_config.disk_blocks
                )
            )

        # If all layers are on the same device, use the old GPU generation mode
        while len(self.breakmodel_config.gpu_blocks) and self.breakmodel_config.gpu_blocks[-1] == 0:
            self.breakmodel_config.gpu_blocks.pop()
        self.breakmodel = True
        if len(self.breakmodel_config.gpu_blocks) and self.breakmodel_config.gpu_blocks[-1] in (
            -1,
            utils.num_layers(config),
        ):
            logger.debug("All layers on same GPU. Breakmodel disabled")
            self.breakmodel = False
            self.usegpu = True
            utils.koboldai_vars.gpu_device = len(self.breakmodel_config.gpu_blocks) - 1
            return

        if not self.breakmodel_config.gpu_blocks:
            logger.warning("Nothing assigned to a GPU, reverting to CPU only mode")
            self.breakmodel = False
            self.usegpu = False
            return
