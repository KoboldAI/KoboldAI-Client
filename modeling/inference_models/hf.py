import os
from typing import Optional
from transformers import AutoConfig

import utils
from logger import logger
from modeling.inference_model import InferenceModel


class HFInferenceModel(InferenceModel):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_config = None
        self.model_name = model_name

        self.model = None
        self.tokenizer = None

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
