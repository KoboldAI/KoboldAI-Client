from __future__ import annotations
try:
    import os
    import json
    import shutil
    import traceback
    from typing import Dict

    import torch
    from torch.nn import Embedding
    from transformers import AutoConfig, GPTQConfig
    from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, TF2_WEIGHTS_INDEX_NAME, TF_WEIGHTS_NAME, FLAX_WEIGHTS_NAME, FLAX_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME

    import utils
    from logger import logger
    from modeling.inference_models.hf_torch import HFTorchInferenceModel

    from bigdl.llm.transformers import AutoModelForCausalLM

    load_failed = False
except Exception:
    load_failed = True

model_backend_name = "BigDL LLM"
model_backend_type = "Huggingface" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

class model_backend(HFTorchInferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.lazy_load = False
        self.nobreakmodel = True
        self.disable = load_failed
        self.has_xpu = bool(hasattr(torch, "xpu") and torch.xpu.is_available())

    def init_model_config(self) -> None:
        # Get the model_type from the config or assume a model type if it isn't present
        try:
            self.model_config = AutoConfig.from_pretrained(
                self.get_local_model_path() or self.model_name,
                revision=utils.koboldai_vars.revision,
                cache_dir="cache",
            )

            self.model_type = self.model_config.model_type
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


    def _get_model(self, location: str, tf_kwargs: Dict):
        tf_kwargs["revision"] = utils.koboldai_vars.revision
        tf_kwargs["cache_dir"] = "cache"
        if self.quantization == "4bit":
            tf_kwargs["load_in_4bit"] = True
            tf_kwargs.pop("load_in_low_bit", None)
        else:
            tf_kwargs["load_in_low_bit"] = self.quantization
            tf_kwargs.pop("load_in_4bit", None)

        if (self.has_xpu or not self.usegpu) and hasattr(self.model_config, "quantization_config"):
            # setting disable_exllama here doesn't do anything?
            self.model_config.quantization_config.pop("disable_exllama", None)
            tf_kwargs["quantization_config"] = GPTQConfig(disable_exllama=True, **self.model_config.quantization_config)
            # BigDL breaks without this:
            tf_kwargs["quantization_config"].use_exllama = False

        tf_kwargs.pop("low_cpu_mem_usage", None)

        # Try to determine model type from either AutoModel or falling back to legacy
        try:
            model = AutoModelForCausalLM.from_pretrained(
                location,
                offload_folder="accelerate-disk-cache",
                torch_dtype=self._get_target_dtype(),
                **tf_kwargs,
            )

            # We need to move the model to the desired device
            if (not self.usegpu) or (torch.cuda.device_count() <= 0 and not self.has_xpu):
                model = model.to("cpu")
            elif self.has_xpu:
                model = model.to("xpu")
            else:
                model = model.to("cuda")

            return model
        except Exception as e:
            traceback_string = traceback.format_exc().lower()

            if "out of host memory" in traceback_string or "out of memory" in traceback_string:
                raise RuntimeError(
                    "One of your GPUs ran out of memory when KoboldAI tried to load your model."
                )

            # Model corrupted or serious loading problem. Stop here.
            if "invalid load key" in traceback_string:
                logger.error("Invalid load key! Aborting.")
                raise

            if utils.args.panic:
                raise

            logger.warning(f"Failed to load model: {e}")
            logger.debug(traceback.format_exc())

    # Function to patch transformers to use our soft prompt
    def patch_embedding(self) -> None:
        if getattr(Embedding, "_koboldai_patch_causallm_model", None):
            Embedding._koboldai_patch_causallm_model = self.model
            return

        old_embedding_call = Embedding.__call__

        kai_model = self

        def new_embedding_call(self, input_ids, *args, **kwargs):
            # Don't touch embeddings for models other than the core inference model (that's us!)
            if (
                Embedding._koboldai_patch_causallm_model.get_input_embeddings()
                is not self
            ):
                return old_embedding_call(self, input_ids, *args, **kwargs)

            assert input_ids is not None

            if utils.koboldai_vars.sp is not None:
                shifted_input_ids = input_ids - kai_model.model.vocab_size

            input_ids.clamp_(max=kai_model.model.config.vocab_size - 1)
            inputs_embeds = old_embedding_call(self, input_ids, *args, **kwargs)

            if utils.koboldai_vars.sp is not None:
                utils.koboldai_vars.sp = utils.koboldai_vars.sp.to(
                    inputs_embeds.dtype
                ).to(inputs_embeds.device)
                inputs_embeds = torch.where(
                    (shifted_input_ids >= 0)[..., None],
                    utils.koboldai_vars.sp[shifted_input_ids.clamp(min=0)],
                    inputs_embeds,
                )

            return inputs_embeds

        Embedding.__call__ = new_embedding_call
        Embedding._koboldai_patch_causallm_model = self.model

    def is_valid(self, model_name, model_path, menu_path):
        base_is_valid = super().is_valid(model_name, model_path, menu_path)
        path = False
        gen_path = "models/{}".format(model_name.replace('/', '_'))
        if model_path is not None and os.path.exists(model_path):
            path = model_path
        elif os.path.exists(gen_path):
            path = gen_path

        fnames = [WEIGHTS_NAME, WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, TF2_WEIGHTS_INDEX_NAME, TF_WEIGHTS_NAME, FLAX_WEIGHTS_NAME, FLAX_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME]

        return base_is_valid and any(os.path.exists(os.path.join(path, fname)) for fname in fnames)

    def _initialize_model(self):
        return

    def get_requested_parameters(self, model_name, model_path, menu_path, parameters = {}):
        requested_parameters = super().get_requested_parameters(model_name, model_path, menu_path, parameters)
        if os.path.exists("settings/{}.hf_bigdl.model_backend.settings".format(model_name.replace("/", "_"))) and 'base_url' not in vars(self):
            with open("settings/{}.hf_bigdl.model_backend.settings".format(model_name.replace("/", "_")), "r") as f:
                temp = json.load(f)
        else:
            temp = {}
        requested_parameters.append({
            "uitype": "dropdown",
            "unit": "text",
            "label": "Quantization",
            "id": "quantization",
            "default": temp['quantization'] if 'quantization' in temp else '4bit',
            "tooltip": "Whether or not to use BnB's 4-bit or 8-bit mode",
            "menu_path": "Layers",
            "children": [{'text': '4-bit', 'value': '4bit'}, {'text': '8-bit', 'value': 'sym_int8'},
                         {'text': '16-bit', 'value':'bf16'},  {'text': 'FP16', 'value':'fp16'},
                         {'text': 'SYM INT4', 'value':'sym_int4'}, {'text': 'ASYM INT4', 'value':'asym_int4'},
                         {'text': 'NF3', 'value':'nf3'}, {'text': 'NF4', 'value':'nf4'},
                         {'text': 'FP4', 'value':'fp4'}, {'text': 'FP8', 'value':'fp8'},
                         {'text': 'FP8 E4M3', 'value':'fp8_e4m3'}, {'text': 'FP8 E5M2', 'value':'fp8_e5m2'},
                         {'text': 'SYM INT5', 'value':'sym_int5'},{'text': 'ASYM INT5', 'value':'asym_int5'}],
            "extra_classes": "",
            "refresh_model_inputs": False
                                            })
        return requested_parameters

    def set_input_parameters(self, parameters):
        super().set_input_parameters(parameters)
        self.usegpu = parameters['use_gpu'] if 'use_gpu' in parameters else False
        self.quantization = parameters['quantization'] if 'quantization' in parameters else '4bit'

    def _load(self, save_model: bool, initial_load: bool) -> None:
        utils.koboldai_vars.allowsp = True

        # Make model path the same as the model name to make this consistent
        # with the other loading method if it isn't a known model type. This
        # code is not just a workaround for below, it is also used to make the
        # behavior consistent with other loading methods - Henk717
        # if utils.koboldai_vars.model not in ["NeoCustom", "GPT2Custom"]:
        #     utils.koboldai_vars.custmodpth = utils.koboldai_vars.model
        
        if self.model_name == "NeoCustom":
            self.model_name = os.path.basename(os.path.normpath(self.path))
        utils.koboldai_vars.model = self.model_name

        # If we specify a model and it's in the root directory, we need to move
        # it to the models directory (legacy folder structure to new)
        if self.get_local_model_path(legacy=True):
            shutil.move(
                self.get_local_model_path(legacy=True, ignore_existance=True),
                self.get_local_model_path(ignore_existance=True),
            )

        self.init_model_config()

        tf_kwargs = {
            "low_cpu_mem_usage": True,
            "use_cache": True # Workaround for models that accidentally turn cache to false
        }


        if self.model_type == "llama":
            tf_kwargs.update({
                "pretraining_tp": 1 # Workaround recommended by HF to fix their mistake on the config.json tuners adopted
            })
        
        logger.debug(
            "hasgpu: {}".format(
                utils.koboldai_vars.hascuda,
            )
        )

        # Download model from Huggingface if it does not exist, otherwise load locally
        if self.get_local_model_path():
            # Model is stored locally, load it.
            self.model = self._get_model(self.get_local_model_path(), tf_kwargs)
            self.tokenizer = self._get_tokenizer(self.get_local_model_path())
        else:
            # Model not stored locally, we need to download it.

            # _rebuild_tensor patch for casting dtype and supporting LazyTensors
            old_rebuild_tensor = torch._utils._rebuild_tensor

            def new_rebuild_tensor(
                storage: torch.Storage,
                storage_offset,
                shape,
                stride,
            ):
                dtype = storage.dtype
                if dtype is torch.float32 and len(shape) >= 2:
                    utils.koboldai_vars.fp32_model = True
                return old_rebuild_tensor(storage, storage_offset, shape, stride)

            torch._utils._rebuild_tensor = new_rebuild_tensor
            self.model = self._get_model(self.model_name, tf_kwargs)
            self.tokenizer = self._get_tokenizer(self.model_name)
            torch._utils._rebuild_tensor = old_rebuild_tensor

            if save_model:
                self.tokenizer.save_pretrained(
                    self.get_local_model_path(ignore_existance=True)
                )

                if utils.koboldai_vars.fp32_model:
                    # Use save_pretrained to convert fp32 models to fp16,
                    # unless we are using disk cache because save_pretrained
                    # is not supported in that case
                    self.model = self.model.half()
                    self.model.save_pretrained(
                        self.get_local_model_path(ignore_existance=True),
                        max_shard_size="500MiB",
                    )

                else:
                    # For fp16 models, we can just copy the model files directly
                    import transformers.configuration_utils
                    import transformers.modeling_utils
                    import transformers.file_utils
                    import huggingface_hub

                    # Save the config.json
                    shutil.move(
                        os.path.realpath(
                            huggingface_hub.hf_hub_download(
                                self.model_name,
                                transformers.configuration_utils.CONFIG_NAME,
                                revision=utils.koboldai_vars.revision,
                                cache_dir="cache",
                                local_files_only=True,
                                legacy_cache_layout=False,
                            )
                        ),
                        os.path.join(
                            self.get_local_model_path(ignore_existance=True),
                            transformers.configuration_utils.CONFIG_NAME,
                        ),
                    )

                    if utils.num_shards is None:
                        # Save the pytorch_model.bin or model.safetensors of an unsharded model
                        any_success = False
                        possible_checkpoint_names = [
                            transformers.modeling_utils.WEIGHTS_NAME,
                            "model.safetensors",
                        ]

                        for possible_checkpoint_name in possible_checkpoint_names:
                            try:
                                shutil.move(
                                    os.path.realpath(
                                        huggingface_hub.hf_hub_download(
                                            self.model_name,
                                            possible_checkpoint_name,
                                            revision=utils.koboldai_vars.revision,
                                            cache_dir="cache",
                                            local_files_only=True,
                                            legacy_cache_layout=False,
                                        )
                                    ),
                                    os.path.join(
                                        self.get_local_model_path(
                                            ignore_existance=True
                                        ),
                                        possible_checkpoint_name,
                                    ),
                                )
                                any_success = True
                            except Exception:
                                pass

                        if not any_success:
                            raise RuntimeError(
                                f"Couldn't find any of {possible_checkpoint_names} in cache for {self.model_name} @ '{utils.koboldai_vars.revisison}'"
                            )
                    else:
                        # Handle saving sharded models

                        with open(utils.from_pretrained_index_filename) as f:
                            map_data = json.load(f)
                        filenames = set(map_data["weight_map"].values())
                        # Save the pytorch_model.bin.index.json of a sharded model
                        shutil.move(
                            os.path.realpath(utils.from_pretrained_index_filename),
                            os.path.join(
                                self.get_local_model_path(ignore_existance=True),
                                transformers.modeling_utils.WEIGHTS_INDEX_NAME,
                            ),
                        )
                        # Then save the pytorch_model-#####-of-#####.bin files
                        for filename in filenames:
                            shutil.move(
                                os.path.realpath(
                                    huggingface_hub.hf_hub_download(
                                        self.model_name,
                                        filename,
                                        revision=utils.koboldai_vars.revision,
                                        cache_dir="cache",
                                        local_files_only=True,
                                        legacy_cache_layout=False,
                                    )
                                ),
                                os.path.join(
                                    self.get_local_model_path(ignore_existance=True),
                                    filename,
                                ),
                            )
                shutil.rmtree("cache/")

        self.patch_embedding()

        self.model.kai_model = self
        utils.koboldai_vars.modeldim = self.get_hidden_size()

    def _save_settings(self):
        with open(
            "settings/{}.hf_bigdl.model_backend.settings".format(
                self.model_name.replace("/", "_")
            ),
            "w",
        ) as f:
            json.dump(
                {

                    "quantization": self.quantization,
                    'use_gpu': self.usegpu,
                },
                f,
                indent="",
            )
