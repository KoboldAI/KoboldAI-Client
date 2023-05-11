import os
from typing import Optional
from transformers import AutoConfig

import utils
import koboldai_settings
from logger import logger
from modeling.inference_model import InferenceModel
import torch


class HFInferenceModel(InferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.model_config = None
        #self.model_name = model_name

        self.model = None
        self.tokenizer = None

    def is_valid(self, model_name, model_path, menu_path):
        try:
            if model_path is not None and os.path.exists(model_path):
                model_config = AutoConfig.from_pretrained(model_path)
            elif(os.path.exists("models/{}".format(model_name.replace('/', '_')))):
                model_config = AutoConfig.from_pretrained("models/{}".format(model_name.replace('/', '_')), revision=utils.koboldai_vars.revision, cache_dir="cache")
            else:
                model_config = AutoConfig.from_pretrained(model_name, revision=utils.koboldai_vars.revision, cache_dir="cache")
            return True
        except:
            return False
        
    def get_requested_parameters(self, model_name, model_path, menu_path):
        requested_parameters = []
        
        if model_path is not None and os.path.exists(model_path):
            self.model_config = AutoConfig.from_pretrained(model_path)
        elif(os.path.exists("models/{}".format(model_name.replace('/', '_')))):
            self.model_config = AutoConfig.from_pretrained("models/{}".format(model_name.replace('/', '_')), revision=utils.koboldai_vars.revision, cache_dir="cache")
        else:
            self.model_config = AutoConfig.from_pretrained(model_name, revision=utils.koboldai_vars.revision, cache_dir="cache")
        layer_count = self.model_config["n_layer"] if isinstance(self.model_config, dict) else self.model_config.num_layers if hasattr(self.model_config, "num_layers") else self.model_config.n_layer if hasattr(self.model_config, "n_layer") else self.model_config.num_hidden_layers if hasattr(self.model_config, 'num_hidden_layers') else None
        if layer_count is not None and layer_count >= 0:
            if os.path.exists("settings/{}.breakmodel".format(model_name.replace("/", "_"))):
                with open("settings/{}.breakmodel".format(model_name.replace("/", "_")), "r") as file:
                    data = [x for x in file.read().split("\n")[:2] if x != '']
                    if len(data) < 2:
                        data.append("0")
                    break_values, disk_blocks = data
                    break_values = break_values.split(",")
            else:
                break_values = [layer_count]
                disk_blocks = None
            break_values = [int(x) for x in break_values if x != '' and x is not None]
            gpu_count = torch.cuda.device_count()
            break_values += [0] * (gpu_count - len(break_values))
            if disk_blocks is not None:
                break_values += [disk_blocks]
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
                                            "default": False,
                                            "tooltip": "The number of layers to put on the disk. This will use your hard drive. The is VERY slow in comparison to GPU or CPU. Use as a last resort.",
                                            "menu_path": "Layers",
                                            "extra_classes": "",
                                            "refresh_model_inputs": False
                                        })
                                        
        
        return requested_parameters
        
    def set_input_parameters(self, parameters):
        gpu_count = torch.cuda.device_count()
        layers = []
        for i in range(gpu_count):
            layers.append(int(parameters["{}_Layers".format(i)]) if parameters["{}_Layers".format(i)].isnumeric() else None)
        self.cpu_layers = parameters['CPU_Layers'] if 'CPU_Layers' in parameters else None
        self.layers = layers
        self.disk_layers = parameters['disk_layers'] if 'disk_layers' in parameters else None
        self.use_gpu = parameters['use_gpu'] if 'use_gpu' in parameters else None
        self.model_name = parameters['id']
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

    def _post_load(self) -> None:
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
