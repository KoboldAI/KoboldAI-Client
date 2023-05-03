import os
from typing import Optional
from transformers import AutoConfig

import utils
import koboldai_settings
from logger import logger
from modeling.inference_model import InferenceModel


class HFInferenceModel(InferenceModel):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_config = None
        self.model_name = model_name

        self.model = None
        self.tokenizer = None

    def _post_load(self) -> None:
        # Clean up tokens that cause issues
        if (
            utils.koboldai_vars.badwordsids == koboldai_settings.badwordsids_default
            and utils.koboldai_vars.model_type not in ("gpt2", "gpt_neo", "gptj")
        ):
            utils.koboldai_vars.badwordsids = [
                [v]
                for k, v in self.tokenizer.get_vocab().items()
                if any(c in str(k) for c in "[]")
            ]

            if utils.koboldai_vars.newlinemode == "n":
                utils.koboldai_vars.badwordsids.append([self.tokenizer.eos_token_id])

        # These are model specific tokenizer overrides if a model has bad defaults
        if utils.koboldai_vars.model_type == "llama":
            self.tokenizer.decode_with_prefix_space = True
            self.tokenizer.add_bos_token = False
        elif utils.koboldai_vars.model_type == "opt":
            self.tokenizer._koboldai_header = self.tokenizer.encode("")
            self.tokenizer.add_bos_token = False
            self.tokenizer.add_prefix_space = False

        # Change newline behavior to match model quirks
        if utils.koboldai_vars.model_type == "xglm":
            # Default to </s> newline mode if using XGLM
            utils.koboldai_vars.newlinemode = "s"
        elif utils.koboldai_vars.model_type in ["opt", "bloom"]:
            # Handle </s> but don't convert newlines if using Fairseq models that have newlines trained in them
            utils.koboldai_vars.newlinemode = "ns"

        return super()._post_load()

    def get_local_model_path(
        self, legacy: bool = False, ignore_existance: bool = False
    ) -> Optional[str]:
        """
        Returns a string of the model's path locally, or None if it is not downloaded.
        If ignore_existance is true, it will always return a path.
        """

        if self.model_name in ["NeoCustom", "GPT2Custom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]:
            model_path = utils.koboldai_vars.custmodpth
            assert model_path

            # Path can be absolute or relative to models directory
            if os.path.exists(model_path):
                return model_path

            model_path = os.path.join("models", model_path)

            try:
                assert os.path.exists(model_path)
            except AssertionError:
                logger.error(f"Custom model does not exist at '{utils.koboldai_vars.custmodpth}' or '{model_path}'.")
                raise

            return model_path

        basename = utils.koboldai_vars.model.replace("/", "_")
        if legacy:
            ret = basename
        else:
            ret = os.path.join("models", basename)

        if os.path.isdir(ret) or ignore_existance:
            return ret
        return None

    def init_model_config(self) -> None:
        # Get the model_type from the config or assume a model type if it isn't present
        try:
            self.model_config = AutoConfig.from_pretrained(
                self.get_local_model_path() or self.model_name,
                revision=utils.koboldai_vars.revision,
                cache_dir="cache",
            )
            utils.koboldai_vars.model_type = self.model_config.model_type
        except ValueError:
            utils.koboldai_vars.model_type = {
                "NeoCustom": "gpt_neo",
                "GPT2Custom": "gpt2",
            }.get(utils.koboldai_vars.model)

            if not utils.koboldai_vars.model_type:
                logger.warning(
                    "No model type detected, assuming Neo (If this is a GPT2 model use the other menu option or --model GPT2Custom)"
                )
                utils.koboldai_vars.model_type = "gpt_neo"
