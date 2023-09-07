import os, sys
from typing import Optional
try:
    from hf_bleeding_edge import AutoConfig
except ImportError:
    from transformers import AutoConfig

import warnings
import utils
import json
import koboldai_settings
from logger import logger
from modeling.inference_model import InferenceModel
import torch
import gc


class HFInferenceModel(InferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.model_config = None

        # TODO: model_name should probably be an instantiation parameter all the
        # way down the inheritance chain.
        self.model_name = None

        self.path = None
        self.hf_torch = False
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
        if not self.hf_torch:
            return []
        if model_name in ('customhuggingface', 'customgptq'):
            requested_parameters.append({
                                        "uitype": "text",
                                        "unit": "text",
                                        "label": "Huggingface Model Name",
                                        "id": "custom_model_name",
                                        "default": parameters["custom_model_name"] if "custom_model_name" in parameters and parameters["custom_model_name"] != "" else "",
                                        "check": {"value": "", 'check': "!="},
                                        "tooltip": "Model name from https://huggingface.co/",
                                        "menu_path": "",
                                        "refresh_model_inputs": True,
                                        "extra_classes": ""
                                    })
        
        if model_name not in ('customhuggingface', 'customgptq') or "custom_model_name" in parameters:
            model_name = parameters["custom_model_name"] if "custom_model_name" in parameters and parameters["custom_model_name"] != "" else model_name
            if model_path is not None and os.path.exists(model_path):
                self.model_config = AutoConfig.from_pretrained(model_path)
            elif(os.path.exists("models/{}".format(model_name.replace('/', '_')))):
                self.model_config = AutoConfig.from_pretrained("models/{}".format(model_name.replace('/', '_')), revision=utils.koboldai_vars.revision, cache_dir="cache")
            else:
                self.model_config = AutoConfig.from_pretrained(model_name, revision=utils.koboldai_vars.revision, cache_dir="cache")
            layer_count = self.model_config["n_layer"] if isinstance(self.model_config, dict) else self.model_config.num_layers if hasattr(self.model_config, "num_layers") else self.model_config.n_layer if hasattr(self.model_config, "n_layer") else self.model_config.num_hidden_layers if hasattr(self.model_config, 'num_hidden_layers') else None
            layer_count = None if hasattr(self, "get_model_type") and self.get_model_type() == "gpt2" else layer_count #Skip layers if we're a GPT2 model as it doesn't support breakmodel
            if layer_count is not None and layer_count >= 0 and not self.nobreakmodel:
                if os.path.exists("settings/{}.generic_hf_torch.model_backend.settings".format(model_name.replace("/", "_"))) and 'base_url' not in vars(self):
                    with open("settings/{}.generic_hf_torch.model_backend.settings".format(model_name.replace("/", "_")), "r") as f:
                        temp = json.load(f)
                        break_values = temp['layers'] if 'layers' in temp else [layer_count]
                        disk_blocks = temp['disk_layers'] if 'disk_layers' in temp else 0
                else:
                    break_values = [layer_count]
                    disk_blocks = 0
                
                break_values = [int(x) for x in break_values if x != '' and x is not None]
                gpu_count = torch.cuda.device_count()
                break_values += [0] * (gpu_count - len(break_values))
                if disk_blocks is not None:
                    break_values += [int(disk_blocks)]
                requested_parameters.append({
                                                "uitype": "Valid Display",
                                                "unit": "text",
                                                "label": "Current Allocated Layers: %1/{}".format(layer_count), #%1 will be the validation value
                                                "id": "valid_layers",
                                                "max": layer_count,
                                                "step": 1,
                                                "check": {"sum": ["{}_Layers".format(i) for i in range(gpu_count)]+['CPU_Layers']+(['Disk_Layers'] if disk_blocks is not None else []), "value": layer_count, 'check': "="},
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
                                                    "check": {"sum": ["{}_Layers".format(i) for i in range(gpu_count)]+['CPU_Layers']+(['Disk_Layers'] if disk_blocks is not None else []), "value": layer_count, 'check': "="},
                                                    "check_message": "The sum of assigned layers must equal {}".format(layer_count),
                                                    "default": break_values[i],
                                                    "tooltip": "The number of layers to put on {}.".format(torch.cuda.get_device_name(i)),
                                                    "menu_path": "Layers",
                                                    "extra_classes": "",
                                                    "refresh_model_inputs": False
                                                })
                requested_parameters.append({
                                                "uitype": "slider",
                                                "unit": "int",
                                                "label": "CPU Layers",
                                                "id": "CPU_Layers",
                                                "min": 0,
                                                "max": layer_count,
                                                "step": 1,
                                                "check": {"sum": ["{}_Layers".format(i) for i in range(gpu_count)]+['CPU_Layers']+(['Disk_Layers'] if disk_blocks is not None else []), "value": layer_count, 'check': "="},
                                                "check_message": "The sum of assigned layers must equal {}".format(layer_count),
                                                "default": layer_count - sum(break_values),
                                                "tooltip": "The number of layers to put on the CPU. This will use your system RAM. It will also do inference partially on CPU. Use if you must.",
                                                "menu_path": "Layers",
                                                "extra_classes": "",
                                                "refresh_model_inputs": False
                                            })
                if disk_blocks is not None:
                    requested_parameters.append({
                                                    "uitype": "slider",
                                                    "unit": "int",
                                                    "label": "Disk Layers",
                                                    "id": "Disk_Layers",
                                                    "min": 0,
                                                    "max": layer_count,
                                                    "step": 1,
                                                    "check": {"sum": ["{}_Layers".format(i) for i in range(gpu_count)]+['CPU_Layers']+(['Disk_Layers'] if disk_blocks is not None else []), "value": layer_count, 'check': "="},
                                                    "check_message": "The sum of assigned layers must equal {}".format(layer_count),
                                                    "default": disk_blocks,
                                                    "tooltip": "The number of layers to put on the disk. This will use your hard drive. The is VERY slow in comparison to GPU or CPU. Use as a last resort.",
                                                    "menu_path": "Layers",
                                                    "extra_classes": "",
                                                    "refresh_model_inputs": False
                                                })
            else:
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
        if self.hf_torch and hasattr(self, "get_model_type") and self.get_model_type() != "gpt2":
            layer_count = self.model_config["n_layer"] if isinstance(self.model_config, dict) else self.model_config.num_layers if hasattr(self.model_config, "num_layers") else self.model_config.n_layer if hasattr(self.model_config, "n_layer") else self.model_config.num_hidden_layers if hasattr(self.model_config, 'num_hidden_layers') else None
            if layer_count is not None and layer_count >= 0 and not self.nobreakmodel:
                gpu_count = torch.cuda.device_count()
                layers = []
                for i in range(gpu_count):
                    if isinstance(parameters["{}_Layers".format(i)], str) and parameters["{}_Layers".format(i)].isnumeric():
                        layers.append(int(parameters["{}_Layers".format(i)]))
                    elif isinstance(parameters["{}_Layers".format(i)], str):
                         layers.append(None)
                    else:
                        layers.append(parameters["{}_Layers".format(i)])
                self.cpu_layers = int(parameters['CPU_Layers']) if 'CPU_Layers' in parameters else None
                if isinstance(self.cpu_layers, str):
                    self.cpu_layers = int(self.cpu_layers) if self.cpu_layers.isnumeric() else 0
                self.layers = layers
                self.disk_layers = parameters['Disk_Layers'] if 'Disk_Layers' in parameters else 0    
                if isinstance(self.disk_layers, str):
                    self.disk_layers = int(self.disk_layers) if self.disk_layers.isnumeric() else 0
                print("TODO: Allow config")
                # self.usegpu = self.cpu_layers == 0 and breakmodel.disk_blocks == 0 and sum(self.layers)-self.layers[0] == 0
            self.model_type = self.get_model_type()
            self.breakmodel = ((self.model_type != 'gpt2') or self.model_type in ("gpt_neo", "gptj", "xglm", "opt")) and not self.nobreakmodel
            self.lazy_load = True
            logger.debug("Model type: {}".format(self.model_type))
        else:
            logger.debug("Disabling breakmodel and lazyload")
            self.usegpu = parameters['use_gpu'] if 'use_gpu' in parameters else None
            self.breakmodel = False
            self.lazy_load = False
        logger.info(parameters)
        self.model_name = parameters['custom_model_name'] if 'custom_model_name' in parameters else parameters['id']
        self.path = parameters['path'] if 'path' in parameters else None

    def unload(self):
        if hasattr(self, 'model'):
            self.model = None
        if hasattr(self, 'tokenizer'):
            self.tokenizer = None
        if hasattr(self, 'model_config'):
            self.model_config = None
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
    
    def _pre_load(self) -> None:
        # HACK: Make model instantiation work without UI parameters
        self.model_name = self.model_name or utils.koboldai_vars.model
        return super()._pre_load()

    def _post_load(self) -> None:
        self.badwordsids = koboldai_settings.badwordsids_default
        self.model_type = str(self.model_config.model_type)
        
        # These are model specific tokenizer overrides if a model has bad defaults
        if self.model_type == "llama":
            # Note: self.tokenizer is a GenericTokenizer, and self.tokenizer.tokenizer is the actual LlamaTokenizer
            self.tokenizer.add_bos_token = False
            self.tokenizer.legacy = False
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

        elif self.model_type == "opt":
            self.tokenizer._koboldai_header = self.tokenizer.encode("")
            self.tokenizer.add_bos_token = False
            self.tokenizer.add_prefix_space = False

        # Change newline behavior to match model quirks
        if self.model_type == "xglm":
            # Default to </s> newline mode if using XGLM
            utils.koboldai_vars.newlinemode = "s"
        elif self.model_type in ["opt", "bloom"]:
            # Handle </s> but don't convert newlines if using Fairseq models that have newlines trained in them
            utils.koboldai_vars.newlinemode = "ns"

        # Clean up tokens that cause issues
        if (
            self.badwordsids == koboldai_settings.badwordsids_default
            and self.model_type not in ("gpt2", "gpt_neo", "gptj")
        ):
            self.badwordsids = [
                [v]
                for k, v in self.tokenizer.get_vocab().items()
                if any(c in str(k) for c in "[]")
            ]

            try:
                self.badwordsids.remove([self.tokenizer.pad_token_id])
            except:
                pass
            
            if utils.koboldai_vars.newlinemode == "n":
                self.badwordsids.append([self.tokenizer.eos_token_id])

        return super()._post_load()

    def get_local_model_path(
        self, legacy: bool = False, ignore_existance: bool = False
    ) -> Optional[str]:
        """
        Returns a string of the model's path locally, or None if it is not downloaded.
        If ignore_existance is true, it will always return a path.
        """
        if self.path is not None:
            if os.path.exists(self.path):
                return self.path

        if self.model_name in ["NeoCustom", "GPT2Custom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]:
            model_path = self.path
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

        basename = self.model_name.replace("/", "_")
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

            self.model_type = self.model_config.model_type

            if "gptq_bits" in dir(self.model_config):
                self.gptq_model = True
                self.gptq_bits = self.model_config.gptq_bits
                self.gptq_groupsize = self.model_config.gptq_groupsize if getattr(self.model_config, "gptq_groupsize", False) else -1
                self.gptq_version = self.model_config.gptq_version if getattr(self.model_config, "gptq_version", False) else 1
                self.gptq_file = None
            else:
                self.gptq_model = False
        except ValueError:
            self.model_type = {
                "NeoCustom": "gpt_neo",
                "GPT2Custom": "gpt2",
            }.get(self.model)

            if not self.model_type:
                logger.warning(
                    "No model type detected, assuming Neo (If this is a GPT2 model use the other menu option or --model GPT2Custom)"
                )
                self.model_type = "gpt_neo"
