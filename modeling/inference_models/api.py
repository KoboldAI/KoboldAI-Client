from __future__ import annotations

import time
import json
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
    ModelCapabilities,
)


class APIException(Exception):
    """To be used for errors when using the Kobold API as an interface."""


class APIInferenceModel(InferenceModel):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")

    def _load(self, save_model: bool, initial_load: bool) -> None:
        tokenizer_id = requests.get(f"{self.base_url}/api/v1/model").json()["result"]

        self.tokenizer = self._get_tokenizer(tokenizer_id)

        # Do not allow API to be served over the API
        self.capabilties = ModelCapabilities(api_host=False)

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ):

        if seed is not None:
            logger.warning(
                "Seed is unsupported on the APIInferenceModel. Seed will be ignored."
            )

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
            req = requests.post(f"{self.base_url}/api/v1/generate", json=reqdata)

            if req.status_code == 503:
                # Server is currently generating something else so poll until it's our turn
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
