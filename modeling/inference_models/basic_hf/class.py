from __future__ import annotations

import os, time
import json
import torch
from torch.nn import Embedding
import shutil
from typing import Union
import transformers
from transformers import (
    StoppingCriteria,
    GPTNeoForCausalLM,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    AutoConfig,
    LogitsProcessorList,
)
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    ModelCapabilities,
    use_core_manipulations,
)

from modeling.stoppers import Stoppers

import utils
import koboldai_settings
from logger import logger


from modeling.inference_model import InferenceModel

model_backend_name = "Very Basic Huggingface"
model_backend_type = "Huggingface" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

LOG_SAMPLER_NO_EFFECT = False

class model_backend(InferenceModel):

    def __init__(self) -> None:
        super().__init__()
        self.model_config = None
        #self.model_name = model_name

        self.model = None
        self.tokenizer = None
        self.badwordsids = koboldai_settings.badwordsids_default
        self.usegpu = False
        
    def is_valid(self, model_name, model_path, menu_path):
        try:
            if model_path is not None and os.path.exists(model_path):
                self.model_config = AutoConfig.from_pretrained(model_path)
            elif(os.path.exists("models/{}".format(model_name.replace('/', '_')))):
                self.model_config = AutoConfig.from_pretrained("models/{}".format(model_name.replace('/', '_')), revision=utils.koboldai_vars.revision, cache_dir="cache")
            else:
                self.model_config = AutoConfig.from_pretrained(model_name, revision=utils.koboldai_vars.revision, cache_dir="cache")
            return True
        except:
            return False
        
    def get_requested_parameters(self, model_name, model_path, menu_path, parameters = {}):
        requested_parameters = []
        requested_parameters.append({
                                        "uitype": "toggle",
                                        "unit": "bool",
                                        "label": "Use GPU",
                                        "id": "use_gpu",
                                        "default": True,
                                        "tooltip": "Whether or not to use the GPU",
                                        "menu_path": "Layers",
                                        "extra_classes": "",
                                        "refresh_model_inputs": False
                                    })
        return requested_parameters
        
    def set_input_parameters(self, parameters):
        self.usegpu = parameters['use_gpu'] if 'use_gpu' in parameters else None
        self.model_name = parameters['id']
        self.path = parameters['path'] if 'path' in parameters else None

    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.model_config = AutoConfig.from_pretrained(self.model_name if self.path is None else self.path)
        self.model = AutoModelForCausalLM.from_config(self.model_config)
        self.tokenizer = self._get_tokenizer(self.model_name if self.path is None else self.path)
        
        if save_model and self.path is None:
            model_path = "models/{}".format(self.model_name.replace("/", "_"))
            if not os.path.exists(model_path):
                self.tokenizer.save_pretrained(model_path)
                self.model.save_pretrained(model_path)
            
        
        if self.usegpu:
            # Use just VRAM
            self.torch_device = utils.koboldai_vars.gpu_device
            self.model = self.model.half().to(self.torch_device)
        else:
            self.torch_device = "cpu"
            self.model = self.model.to(self.torch_device).float()
            
        utils.koboldai_vars.modeldim = self.model.get_input_embeddings().embedding_dim
        

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

        gen_in = gen_in.to(self.torch_device)

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