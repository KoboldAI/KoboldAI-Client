from __future__ import annotations

import torch
import requests
import numpy as np
from typing import List, Optional, Union
import os

import utils
from logger import logger
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
    ModelCapabilities,
)


model_backend_name = "KoboldAI Old Colab Method"
model_backend_type = "KoboldAI Old Colab Method" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

class BasicAPIException(Exception):
    """To be used for errors when using the Basic API as an interface."""


class model_backend(InferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.colaburl = ""

        # Do not allow API to be served over the API
        self.capabilties = ModelCapabilities(api_host=False)
    
    def is_valid(self, model_name, model_path, menu_path):
        return model_name == "Colab"
    
    def get_requested_parameters(self, model_name, model_path, menu_path, parameters = {}):
        if os.path.exists("settings/api.model_backend.settings") and 'colaburl' not in vars(self):
            with open("settings/api.model_backend.settings", "r") as f:
                self.colaburl = json.load(f)['base_url']
        requested_parameters = []
        requested_parameters.append({
                                        "uitype": "text",
                                        "unit": "text",
                                        "label": "URL",
                                        "id": "colaburl",
                                        "default": self.colaburl,
                                        "check": {"value": "", 'check': "!="},
                                        "tooltip": "The URL of the Colab KoboldAI API to connect to.",
                                        "menu_path": "",
                                        "extra_classes": "",
                                        "refresh_model_inputs": False
                                    })
        return requested_parameters
        
    def set_input_parameters(self, parameters):
        self.colaburl = parameters['colaburl']

    def _initialize_model(self):
        return

    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.tokenizer = self._get_tokenizer("EleutherAI/gpt-neo-2.7B")
    
    def _save_settings(self):
        with open("settings/basic_api.model_backend.settings", "w") as f:
            json.dump({"colaburl": self.colaburl}, f, indent="")

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
        req = requests.post(self.colaburl, json=reqdata)

        if req.status_code != 200:
            raise BasicAPIException(f"Bad status code {req.status_code}")

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
