from __future__ import annotations

import copy
import requests
from typing import List
from tqdm.auto import tqdm

import transformers
from transformers import (
    PreTrainedModel,
    modeling_utils,
)

import torch
import modeling

import utils


def patch_transformers_download():
    def http_get(
        url: str,
        temp_file,
        proxies=None,
        resume_size=0,
        headers=None,
        file_name=None,
    ):
        """
        Download remote file. Do not gobble up errors.
        """
        headers = copy.deepcopy(headers)
        if resume_size > 0:
            headers["Range"] = f"bytes={resume_size}-"
        r = requests.get(url, stream=True, proxies=proxies, headers=headers)
        transformers.utils.hub._raise_for_status(r)
        content_length = r.headers.get("Content-Length")
        total = (
            resume_size + int(content_length) if content_length is not None else None
        )

        # `tqdm` behavior is determined by `utils.logging.is_progress_bar_enabled()`
        # and can be set using `utils.logging.enable/disable_progress_bar()`
        if url[-11:] != "config.json":
            progress = tqdm.tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=total,
                initial=resume_size,
                desc=f"Downloading {file_name}"
                if file_name is not None
                else "Downloading",
                file=utils.UIProgressBarFile(),
            )
            utils.koboldai_vars.status_message = "Download Model"
            utils.koboldai_vars.total_download_chunks = total

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                if url[-11:] != "config.json":
                    progress.update(len(chunk))
                    utils.koboldai_vars.downloaded_chunks += len(chunk)
                temp_file.write(chunk)

        if url[-11:] != "config.json":
            progress.close()

        utils.koboldai_vars.status_message = ""

    transformers.utils.hub.http_get = http_get


def patch_transformers_loader() -> None:
    """
    Patch the Transformers loader to use aria2 and our shard tracking.
    Universal for TPU/MTJ and Torch.
    """
    old_from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def new_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        utils.koboldai_vars.fp32_model = False
        utils.num_shards = None
        utils.current_shard = 0
        utils.from_pretrained_model_name = pretrained_model_name_or_path
        utils.from_pretrained_index_filename = None
        utils.from_pretrained_kwargs = kwargs
        utils.bar = None
        if not utils.args.no_aria2:
            utils.aria2_hook(pretrained_model_name_or_path, **kwargs)
        return old_from_pretrained(
            cls, pretrained_model_name_or_path, *model_args, **kwargs
        )

    if not hasattr(PreTrainedModel, "_kai_patched"):
        PreTrainedModel.from_pretrained = new_from_pretrained
        PreTrainedModel._kai_patched = True

    if hasattr(modeling_utils, "get_checkpoint_shard_files"):
        old_get_checkpoint_shard_files = modeling_utils.get_checkpoint_shard_files

        def new_get_checkpoint_shard_files(
            pretrained_model_name_or_path, index_filename, *args, **kwargs
        ):
            utils.num_shards = utils.get_num_shards(index_filename)
            utils.from_pretrained_index_filename = index_filename
            return old_get_checkpoint_shard_files(
                pretrained_model_name_or_path, index_filename, *args, **kwargs
            )

        modeling_utils.get_checkpoint_shard_files = new_get_checkpoint_shard_files


def patch_transformers_generation() -> None:
    # Not sure why this global is needed...
    global transformers

    # Allow bad words filter to ban <|endoftext|> token
    import transformers.generation.logits_process

    def new_init(self, bad_words_ids: List[List[int]], eos_token_id: int):
        return new_init.old_init(self, bad_words_ids, -1)

    new_init.old_init = (
        transformers.generation.logits_process.NoBadWordsLogitsProcessor.__init__
    )
    transformers.generation.logits_process.NoBadWordsLogitsProcessor.__init__ = new_init


class LazyloadPatches:
    class StateDictFacade(dict):
        def __init__(self, state_dict):
            self.update(state_dict)

        def __getitem__(self, name):
            return super().__getitem__(name).materialize(map_location="cuda:0")

    old_load_state_dict = transformers.modeling_utils._load_state_dict_into_meta_model
    torch_old_load_from_state_dict = torch.nn.Module._load_from_state_dict

    def __enter__() -> None:
        transformers.modeling_utils._load_state_dict_into_meta_model = (
            LazyloadPatches._load_state_dict_into_meta_model
        )
        torch.nn.Module._load_from_state_dict = LazyloadPatches._torch_load_from_state_dict

    def __exit__(exc_type, exc_value, exc_traceback) -> None:
        transformers.modeling_utils._load_state_dict_into_meta_model = LazyloadPatches.old_load_state_dict
        torch.nn.Module._load_from_state_dict = LazyloadPatches.torch_old_load_from_state_dict

    def _torch_load_from_state_dict(self, state_dict, *args, **kwargs):
        return LazyloadPatches.torch_old_load_from_state_dict(
            self,
            LazyloadPatches.StateDictFacade(state_dict),
            *args,
            **kwargs
        )

    def _load_state_dict_into_meta_model(
        model,
        state_dict,
        loaded_state_dict_keys,
        start_prefix,
        expected_keys,
        device_map=None,
        offload_folder=None,
        offload_index=None,
        state_dict_folder=None,
        state_dict_index=None,
        dtype=None,
        # PATCH: load_in_8bit was renamed to is_quantized in Transformers 4.30, keep
        # both for short term compatibility
        load_in_8bit=False,
        is_quantized=False,
        is_safetensors=False,
        keep_in_fp32_modules=None,
    ):
        """
        This is modified code from the Accelerate and Transformers projects,
        made by HuggingFace. The license for these projects are as follows:
        ---
        Copyright The HuggingFace Team. All rights reserved.

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
        """
        from accelerate.utils import offload_weight, set_module_tensor_to_device

        is_quantized = is_quantized or load_in_8bit

        if is_quantized:
            from transformers.utils.bitsandbytes import set_module_quantized_tensor_to_device

        error_msgs = []

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

# BEGIN PATCH
        utils.bar = tqdm(total=len(state_dict), desc="Loading model tensors", file=utils.UIProgressBarFile(), position=1)
        utils.koboldai_vars.total_layers = len(state_dict)
        utils.koboldai_vars.loaded_layers = 0

        for param_name, param in sorted(
            state_dict.items(),
            # State dict must be ordered in this manner to make the caching in
            # lazy_loader.py effective
            key=lambda x: (
                x[1].key,
                x[1].seek_offset,
            ),
        ):

            if isinstance(param, modeling.lazy_loader.LazyTensor):
                # Should always be true
                param = param.materialize(map_location="cpu")
            utils.bar.update(1)
            utils.koboldai_vars.loaded_layers += 1
# END PATCH

            # First part of the test is always true as load_state_dict_keys always contains state_dict keys.
            if (
                param_name not in loaded_state_dict_keys
                or param_name not in expected_keys
            ):
                continue

            if param_name.startswith(start_prefix):
                param_name = param_name[len(start_prefix) :]

            module_name = param_name

            # We convert floating dtypes to the `dtype` passed. We want to keep the buffers/params
            # in int/uint/bool and not cast them.
            if dtype is not None and torch.is_floating_point(param):
                if (
                    keep_in_fp32_modules is not None
                    and any(
                        module_to_keep_in_fp32 in param_name
                        for module_to_keep_in_fp32 in keep_in_fp32_modules
                    )
                    and dtype == torch.float16
                ):
                    param = param.to(torch.float32)
                else:
                    param = param.to(dtype)

            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            if dtype is None:
                old_param = model
                splits = param_name.split(".")
                for split in splits:
                    old_param = getattr(old_param, split)
                    if old_param is None:
                        break

                if old_param is not None:
                    param = param.to(old_param.dtype)

            if device_map is None:
                param_device = "cpu"
            else:
                # find next higher level module that is defined in device_map:
                # bert.lm_head.weight -> bert.lm_head -> bert -> ''
                while len(module_name) > 0 and module_name not in device_map:
                    module_name = ".".join(module_name.split(".")[:-1])
                if module_name == "" and "" not in device_map:
                    # TODO: group all errors and raise at the end.
                    raise ValueError(f"{param_name} doesn't have any device set.")
                param_device = device_map[module_name]

            if param_device == "disk":
                if not is_safetensors:
                    offload_index = offload_weight(
                        param, param_name, offload_folder, offload_index
                    )
            elif param_device == "cpu" and state_dict_index is not None:
                state_dict_index = offload_weight(
                    param, param_name, state_dict_folder, state_dict_index
                )
            elif not is_quantized:
                # For backward compatibility with older versions of `accelerate`
                set_module_tensor_to_device(
                    model,
                    tensor_name=param_name,
                    device=param_device,
                    value=param,
                    dtype=dtype,
                )
            else:
                if (
                    param.dtype == torch.int8
                    and param_name.replace("weight", "SCB") in state_dict.keys()
                ):
                    fp16_statistics = state_dict[param_name.replace("weight", "SCB")]
                else:
                    fp16_statistics = None

                if "SCB" not in param_name:
                    set_module_quantized_tensor_to_device(
                        model,
                        param_name,
                        param_device,
                        value=param,
                        fp16_statistics=fp16_statistics,
                    )

        utils.koboldai_vars.loaded_checkpoints += 1
        return error_msgs, offload_index, state_dict_index


def patch_transformers(use_tpu: bool) -> None:
    patch_transformers_download()
    patch_transformers_loader()

    if not use_tpu:
        patch_transformers_generation()