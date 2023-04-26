from __future__ import annotations

import os
import json
import traceback

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import utils
from modeling.inference_models.hf_torch import HFTorchInferenceModel


class CustomGPT2HFTorchInferenceModel(HFTorchInferenceModel):
    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.lazy_load = False

        model_path = None

        for possible_config_path in [
            utils.koboldai_vars.custmodpth,
            os.path.join("models", utils.koboldai_vars.custmodpth),
            self.model_name,
        ]:
            try:
                with open(
                    os.path.join(possible_config_path, "config.json"), "r"
                ) as file:
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
                    model_path,
                    revision=utils.koboldai_vars.revision,
                    cache_dir="cache",
                    local_files_only=True,
                )
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    model_path,
                    revision=utils.koboldai_vars.revision,
                    cache_dir="cache",
                )
            except Exception as e:
                if "out of memory" in traceback.format_exc().lower():
                    raise RuntimeError(
                        "One of your GPUs ran out of memory when KoboldAI tried to load your model."
                    ) from e
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

        self.patch_embedding()
