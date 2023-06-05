from __future__ import annotations

import time, json
import torch
import requests
import numpy as np
from typing import List, Optional, Union
import os
import glob
from pathlib import Path
import re
import warnings
import gc

import utils
from logger import logger

from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
    ModelCapabilities,
)

from modeling.tokenizer import GenericTokenizer

from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from transformers import LlamaTokenizer
from exllama.generator import ExLlamaGenerator

model_backend_name = "ExLlama"


def load_model_gptq_settings(path):
    try:
        js = json.load(open(path + "/config.json", "r"))
    except Exception as e:
        return False

    gptq_model = False
    gptq_file = False

    gptq_legacy_files = glob.glob(os.path.join(path, "4bit*.safetensors"))
    if "gptq_bits" in js:
        gptq_model = True
        gptq_file = os.path.join(path, "model.safetensors")
    elif gptq_legacy_files:
        gptq_model = True
        gptq_file = gptq_legacy_files[0]
        fname = Path(gptq_file).parts[-1]
        g = re.findall("^(?:4bit)(?:-)(\\d+)(?:g-?)", fname)

    return gptq_model, gptq_file


class model_backend(InferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.model_config = None

        self.model = None
        self.tokenizer = None
        self.cache = None
        self.generator = None

        self.model_name = ""
        self.path = None

    def is_valid(self, model_name, model_path, menu_path):
        gptq_model, _ = load_model_gptq_settings(model_path)
        try:
            self.model_config = self._load_config(model_name, model_path)
            return self.model_config and gptq_model
        except:
            return False

    def get_local_model_path(self):
        return self.path or os.path.join("models", self.model_name.replace("/", "_"))

    def _load_config(self, model_name, model_path):
        if model_path is not None and os.path.exists(model_path):
            return ExLlamaConfig(os.path.join(model_path, "config.json"))
        if(os.path.exists("models/{}".format(model_name.replace('/', '_')))):
            return ExLlamaConfig(os.path.join("models/{}".format(model_name.replace('/', '_')), "config.json"))
        return False

    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.model = self._get_model(self.get_local_model_path(), {})
        self.tokenizer = self._get_tokenizer(self.get_local_model_path()))

        self.cache = ExLlamaCache(self.model)

        self.generator = ExLlamaGenerator(self.model, self.tokenizer.tokenizer, self.cache)

    def _post_load(self) -> None:
        # Note: self.tokenizer is a GenericTokenizer, and self.tokenizer.tokenizer is the actual LlamaTokenizer
        self.tokenizer.add_bos_token = False

        # HF transformers no longer supports decode_with_prefix_space
        # We work around this by wrapping decode, encode, and __call__
        # with versions that work around the 'prefix space' misfeature
        # of sentencepiece.
        vocab = self.tokenizer.convert_ids_to_tokens(range(self.tokenizer.vocab_size))
        has_prefix_space = {i for i, tok in enumerate(vocab) if tok.startswith("▁")}

        # Wrap 'decode' with a method that always returns text starting with a space
        # when the head token starts with a space. This is what 'decode_with_prefix_space'
        # used to do, and we implement it using the same technique (building a cache of
        # tokens that should have a prefix space, and then prepending a space if the first
        # token is in this set.) We also work around a bizarre behavior in which decoding
        # a single token 13 behaves differently than decoding a squence containing only [13].
        original_decode = type(self.tokenizer.tokenizer).decode
        def decode_wrapper(self, token_ids, *args, **kwargs):
            first = None
            # Note, the code below that wraps single-value token_ids in a list
            # is to work around this wonky behavior:
            #   >>> t.decode(13)
            #   '<0x0A>'
            #   >>> t.decode([13])
            #   '\n'
            # Not doing this causes token streaming to receive <0x0A> characters
            # instead of newlines.
            if isinstance(token_ids, int):
                first = token_ids
                token_ids = [first]
            elif hasattr(token_ids, 'dim'): # Check for e.g. torch.Tensor
                # Tensors don't support the Python standard of 'empty is False'
                # and the special case of dimension 0 tensors also needs to be
                # handled separately.
                if token_ids.dim() == 0:
                    first = int(token_ids.item())
                    token_ids = [first]
                elif len(token_ids) > 0:
                    first = int(token_ids[0])
            elif token_ids is not None and len(token_ids) > 0:
                first = token_ids[0]
            result = original_decode(self, token_ids, *args, **kwargs)
            if first is not None and first in has_prefix_space:
                result = " " + result
            return result
        # GenericTokenizer overrides __setattr__ so we need to use object.__setattr__ to bypass it
        object.__setattr__(self.tokenizer, 'decode', decode_wrapper.__get__(self.tokenizer))

        # Wrap encode and __call__ to work around the 'prefix space' misfeature also.
        # The problem is that "Bob" at the start of text is encoded as if it is
        # " Bob". This creates a problem because it means you can't split text, encode
        # the pieces, concatenate the tokens, decode them, and get the original text back.
        # The workaround is to prepend a known token that (1) starts with a space; and
        # (2) is not the prefix of any other token. After searching through the vocab
        # " ," (space comma) is the only token containing only printable ascii characters
        # that fits this bill. By prepending ',' to the text, the original encode
        # method always returns [1919, ...], where the tail of the sequence is the
        # actual encoded result we want without the prefix space behavior.
        original_encode = type(self.tokenizer.tokenizer).encode
        def encode_wrapper(self, text, *args, **kwargs):
            if type(text) is str:
                text = ',' + text
                result = original_encode(self, text, *args, **kwargs)
                result = result[1:]
            else:
                result = original_encode(self, text, *args, **kwargs)
            return result
        object.__setattr__(self.tokenizer, 'encode', encode_wrapper.__get__(self.tokenizer))

        # Since 'encode' is documented as being deprecated, also override __call__.
        # This doesn't appear to currently be used by KoboldAI, but doing so
        # in case someone uses it in the future.
        original_call = type(self.tokenizer.tokenizer).__call__
        def call_wrapper(self, text, *args, **kwargs):
            if type(text) is str:
                text = ',' + text
                result = original_call(self, text, *args, **kwargs)
                result = result[1:]
            else:
                result = original_call(self, text, *args, **kwargs)
            return result
        object.__setattr__(self.tokenizer, '__call__', call_wrapper.__get__(self.tokenizer))

    def unload(self):
        self.model_config = None

        self.model = None
        self.tokenizer = None
        self.cache = None
        self.generator = None

        self.model_name = ""
        self.path = None

        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
                for tensor in gc.get_objects():
                    try:
                        if torch.is_tensor(tensor):
                            tensor.set_(torch.tensor((), device=tensor.device, dtype=tensor.dtype))
                    except:
                        pass
        gc.collect()
        try:
            with torch.no_grad():
                torch.cuda.empty_cache()
        except:
            pass

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
        if seed:
            torch.manual_seed(seed)

        if not isinstance(prompt_tokens, torch.Tensor):
            gen_in = torch.tensor(prompt_tokens, dtype=torch.long)[None]
        else:
            gen_in = prompt_tokens

        self.generator.settings.temperature = max(gen_settings.temp, 0.01)
        self.generator.settings.top_k = gen_settings.top_k if gen_settings.top_k > 0 else 10000
        self.generator.settings.top_p = gen_settings.top_p
        self.generator.settings.min_p = 0.0

        self.generator.gen_begin(gen_in)

        for i in range(max_new):
            token = self.generator.gen_single_token()
            if token.item() == self.tokenizer.eos_token_id: break

        return GenerationResult(
            model=self,
            out_batches=np.array(
                self.generator.sequence[:, gen_in.size(1):],
            ),
            prompt=prompt_tokens,
            is_whole_generation=True,
            single_line=single_line,
        )

    def _get_model(self, location: str, tf_kwargs: Dict):
        _, self.model_config.model_path = load_model_gptq_settings(location)
        return ExLlama(self.model_config)

    def _get_tokenizer(self, location: str):
        tokenizer = GenericTokenizer(LlamaTokenizer.from_pretrained(location))
        tokenizer._koboldai_header = tokenizer.encode("")
        return tokenizer

    def get_requested_parameters(self, model_name, model_path, menu_path, parameters = {}):
        requested_parameters = []
        gpu_count = torch.cuda.device_count()
        layer_count = self.model_config["n_layer"] if isinstance(self.model_config, dict) else self.model_config.num_layers if hasattr(self.model_config, "num_layers") else self.model_config.n_layer if hasattr(self.model_config, "n_layer") else self.model_config.num_hidden_layers if hasattr(self.model_config, 'num_hidden_layers') else None
        requested_parameters.append({
                                        "uitype": "Valid Display",
                                        "unit": "text",
                                        "label": "Current Allocated Layers: %1/{}".format(layer_count), #%1 will be the validation value
                                        "id": "valid_layers",
                                        "max": layer_count,
                                        "step": 1,
                                        "check": {"sum": ["{}_Layers".format(i) for i in range(gpu_count)], "value": layer_count, 'check': "="},
                                        "menu_path": "Layers",
                                        "extra_classes": "",
                                        "refresh_model_inputs": False
                                    })
        for i in range(gpu_count):
            requested_parameters.append({
                                            "uitype": "slider",
                                            "unit": "int",
                                            "label": "{} Layers".format(torch.cuda.get_device_name(i)),
                                            "id": "{}_Layers".format(i),
                                            "min": 0,
                                            "max": layer_count,
                                            "step": 1,
                                            "check": {"sum": ["{}_Layers".format(i) for i in range(gpu_count)], "value": layer_count, 'check': "="},
                                            "check_message": "The sum of assigned layers must equal {}".format(layer_count),
                                            "default": [layer_count if i == 0 else 0],
                                            "tooltip": "The number of layers to put on {}.".format(torch.cuda.get_device_name(i)),
                                            "menu_path": "Layers",
                                            "extra_classes": "",
                                            "refresh_model_inputs": False
                                        })

        return requested_parameters

    def set_input_parameters(self, parameters):
        gpu_count = torch.cuda.device_count()
        layers = []
        for i in range(gpu_count):
            if isinstance(parameters["{}_Layers".format(i)], str) and parameters["{}_Layers".format(i)].isnumeric():
                layers.append(int(parameters["{}_Layers".format(i)]))
            elif isinstance(parameters["{}_Layers".format(i)], str):
                 layers.append(None)
            else:
                layers.append(parameters["{}_Layers".format(i)])

        self.layers = layers
        for i, l in enumerate(layers):
            if l > 0:
                self.model_config.device_map.layers.extend([f"cuda:{i}"] * l)
        self.model_config.device_map.lm_head = "cuda:0"
        self.model_config.device_map.norm = "cuda:0"

        self.model_name = parameters['custom_model_name'] if 'custom_model_name' in parameters else parameters['id']
        self.path = parameters['path'] if 'path' in parameters else None
