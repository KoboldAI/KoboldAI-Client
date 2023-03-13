import torch
import requests
import numpy as np
from typing import List, Optional, Union

import utils
from logger import logger
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
)


class OpenAIAPIError(Exception):
    def __init__(self, error_type: str, error_message) -> None:
        super().__init__(f"{error_type}: {error_message}")


class OpenAIAPIInferenceModel(InferenceModel):
    """InferenceModel for interfacing with OpenAI's generation API."""

    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.tokenizer = self._get_tokenizer("gpt2")

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

        if seed is not None:
            logger.warning(
                "Seed is unsupported on the OpenAIAPIInferenceModel. Seed will be ignored."
            )

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
