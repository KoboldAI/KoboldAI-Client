"""
This file is AGPL-licensed.

Some of the code in this file is copied from PyTorch.

The license for PyTorch is shown below:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""


import contextlib
from functools import reduce
import itertools
import time
import zipfile
import pickle
import torch
import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from torch.nn import Module
from torch.storage import UntypedStorage
from modeling.pickling import RestrictedUnpickler, use_custom_unpickler
from modeling.patches import LazyloadPatches

# Safetensors is a dependency for the local version, TPU/Colab doesn't
# support it yet.
try:
    import safetensors

    HAS_SAFETENSORS = True
except ModuleNotFoundError:
    HAS_SAFETENSORS = False

try:
    import accelerate
    USE_TPU_EMPTY_MODULE_METHOD = False
except ModuleNotFoundError:
    USE_TPU_EMPTY_MODULE_METHOD = True

import utils
from logger import logger

# Storage of zipfile handles for each shard
torch_checkpoint_file_handles = {}


class CheckpointChunkCache:
    """Storage for common checkpoint weight files to speed up loading. In order
    for this to be effective at all, weights must be loaded in ascending order
    of (key, seek_offset).
    """

    # There is considerable room for improvement here; we could peek into the
    # state dict and preload the N most frequent weight files or something, but
    # this first implementation is on par with the speed of whatever the
    # previous callback did.

    file_name = None
    key = None
    handle = None

    hit_data = {"hits": 0, "misses": 0}

    @classmethod
    def clear(cls, unload_model: bool = False) -> None:
        if unload_model:
            cls.hit_data["hits"] = 0
            cls.hit_data["misses"] = 0

        if cls.handle:
            cls.handle.close()

        cls.file_name = None
        cls.key = None
        cls.handle = None


class LazyTensor:
    pass


class TorchLazyTensor(LazyTensor):
    def __init__(
        self,
        storage_type,
        key: str,
        location: str,
        dtype: Optional[torch.dtype] = None,
        seek_offset: Optional[int] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
        requires_grad=False,
        backward_hooks: Any = None,
    ):
        self.storage_type = storage_type
        self.key = key
        self.location = location
        self.dtype = dtype
        self.seek_offset = seek_offset
        self.shape = shape
        self.stride = stride
        self.requires_grad = requires_grad
        self.backward_hooks = backward_hooks
        self.file_name = None

    def __view(self, f: Callable):
        return f"{type(self).__name__}(storage_type={f(self.storage_type)}, key={f(self.key)}, location={f(self.location)}, dtype={f(self.dtype)}, seek_offset={f(self.seek_offset)}, shape={f(self.shape)}, stride={f(self.stride)}, requires_grad={f(self.requires_grad)}, backward_hooks={f(self.backward_hooks)})"

    def __repr__(self):
        return self.__view(repr)

    def materialize(
        self,
        map_location=None,
        no_grad=True,
    ) -> torch.Tensor:
        checkpoint = torch_checkpoint_file_handles[self.file_name]
        filename = os.path.basename(os.path.normpath(self.file_name)).split(".")[0]

        # Often we are using the same weight file to store multiple tensors, so
        # let's cache the file handle to maintain a seek position and other
        # fast stuff.
        if (
            CheckpointChunkCache.file_name != filename
            or CheckpointChunkCache.key != self.key
            or not CheckpointChunkCache.handle
        ):
            # Cache miss. Assuming weights are loaded in order of
            # (key, seek_offset), this means we need to invalidate the cache.
            # print("!", end="", flush=True)
            CheckpointChunkCache.hit_data["misses"] += 1

            CheckpointChunkCache.clear()

            CheckpointChunkCache.file_name = filename
            CheckpointChunkCache.key = self.key
            ziproot = checkpoint.namelist()[0].split("/")[0]
            CheckpointChunkCache.handle = checkpoint.open(f"{ziproot}/data/{self.key}", "r")
        else:
            # Cache hit. Hip hip hooray! :^)
            # print(".", end="", flush=True)
            CheckpointChunkCache.hit_data["hits"] += 1

        size = reduce(lambda x, y: x * y, self.shape, 1)
        dtype = self.dtype
        nbytes = (
            size
            if dtype is torch.bool
            else size
            * (
                (torch.finfo if self.dtype.is_floating_point else torch.iinfo)(
                    self.dtype
                ).bits
                >> 3
            )
        )

        assert isinstance(checkpoint, zipfile.ZipFile)

        CheckpointChunkCache.handle.seek(self.seek_offset, os.SEEK_SET)
        storage = UntypedStorage.from_buffer(
            CheckpointChunkCache.handle.read(nbytes), "little", dtype=self.dtype
        )

        storage = torch.serialization._get_restore_location(map_location)(
            storage, self.location
        )
        tensor = torch.tensor([], dtype=self.dtype, device=storage.device)
        tensor.set_(storage, 0, self.shape, self.stride)
        tensor.requires_grad = not no_grad and self.requires_grad
        tensor._backward_hooks = self.backward_hooks
        return tensor


class SafetensorsLazyTensor(LazyTensor):
    def __init__(self, checkpoint_file: str, key: str, location: str):
        self.checkpoint_file = checkpoint_file
        self.key = key
        self.location = location

        # Stub for cache sorting
        self.seek_offset = 0

    def __view(self, f: Callable):
        return f"{type(self).__name__}(checkpoint_file={f(self.checkpoint_file)}, key={f(self.key)}, location={f(self.location)})"

    def __repr__(self):
        return self.__view(repr)

    def materialize(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return safetensors_load_tensor_independently(
            self.checkpoint_file, tensor_key=self.key, device=self.location
        )


class _LazyUnpickler(RestrictedUnpickler):
    lazy_loaded_storages: Dict[str, LazyTensor]

    def __init__(self, *args, **kwargs):
        self.lazy_loaded_storages = {}
        return super().__init__(*args, **kwargs)

    def forced_persistent_load(self, saved_id):
        assert isinstance(saved_id, tuple)
        typename = saved_id[0]
        assert (
            typename == "storage"
        ), f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, key, location, _ = saved_id[1:]
        return TorchLazyTensor(storage_type, key, location)

    def load(self, *args, **kwargs):
        retval = super().load(*args, **kwargs)
        self.lazy_loaded_storages = {}
        return retval


def _rebuild_tensor(lazy_storage: LazyTensor, storage_offset, shape, stride):
    lazy_storage.shape = shape
    lazy_storage.stride = stride
    dtype = lazy_storage.storage_type.dtype
    if not isinstance(dtype, torch.dtype):
        dtype = lazy_storage.storage_type(0).dtype
    lazy_storage.dtype = dtype
    lazy_storage.seek_offset = (
        storage_offset
        if dtype is torch.bool
        else storage_offset
        * ((torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits >> 3)
    )
    return lazy_storage


def safetensors_load_tensor_independently(
    checkpoint_file: str, tensor_key: str, device: Any
) -> torch.Tensor:
    """A hacky way to load a tensor by itself and not mmap every single tensor
    or whatever is causing that big memory spike"""

    with safetensors.safe_open(checkpoint_file, framework="pt", device=device) as f:
        return f.get_tensor(tensor_key)


def patch_safetensors(callback):
    # Safetensors load patch
    import transformers

    def safetensors_load(checkpoint_file: str) -> dict:
        # Monkeypatch applied to safetensors.torch.load_file

        if utils.koboldai_vars.hascuda:
            # Use GPU as intermediary whenever possible, lowers RAM usage
            # by a significant amount while making loading slightly slower
            # (70 tensors/s -> 65 tensor/s). The memory savings probably
            # shouldn't be the happening, maybe there's a memory leak
            # somewhere in our pipeline with CPU tensors.
            intermediary_device = "cuda:0"
        else:
            intermediary_device = "cpu"

        tensors = {}

        with safetensors.safe_open(
            checkpoint_file,
            framework="pt",
            device=intermediary_device,
        ) as f:
            for key in f.keys():
                tensors[key] = None

        for key in tensors.keys():
            tensors[key] = SafetensorsLazyTensor(
                checkpoint_file=checkpoint_file,
                key=key,
                location=intermediary_device,
            )

        if callback is not None:
            callback(
                tensors,
                f=checkpoint_file,
                map_location=None,
                pickle_module=pickle,
                is_safetensors=True,
            )

        return tensors

    transformers.modeling_utils.safe_load_file = safetensors_load
    safetensors.torch.load_file = safetensors_load


def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    for hook in self._load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
    local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]
            if not torch.overrides.is_tensor_like(input_param):
                error_msgs.append('While copying the parameter named "{}", '
                                    'expected torch.Tensor or Tensor-like object from checkpoint but '
                                    'received {}'
                                    .format(key, type(input_param)))
                continue

            # This is used to avoid copying uninitialized parameters into
            # non-lazy modules, since they dont have the hook to do the checks
            # in such case, it will error when accessing the .shape attribute.
            is_param_lazy = torch.nn.parameter.is_lazy(param)
            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                input_param = input_param[0]

            if not is_param_lazy and input_param.shape != param.shape:
                # local shape should match the one in checkpoint
                error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                    'the shape in current model is {}.'
                                    .format(key, input_param.shape, param.shape))
                continue
            try:
                with torch.no_grad():
                    #param.copy_(input_param)
                    new_param = torch.nn.Parameter(input_param, requires_grad=param.requires_grad)  # This line is new
                    if name in self._parameters:  # This line is new
                        self._parameters[name] = new_param  # This line is new
                    if name in persistent_buffers:  # This line is new
                        self._buffers[name] = new_param  # This line is new
            except Exception as ex:
                error_msgs.append('While copying the parameter named "{}", '
                                    'whose dimensions in the model are {} and '
                                    'whose dimensions in the checkpoint are {}, '
                                    'an exception occurred : {}.'
                                    .format(key, param.size(), input_param.size(), ex.args))
        elif strict:
            missing_keys.append(key)

    extra_state_key = prefix + "_extra_state"
    if hasattr(Module, "set_extra_state") and getattr(self.__class__, "set_extra_state", Module.set_extra_state) is not Module.set_extra_state:  # if getattr(self.__class__, "set_extra_state", Module.set_extra_state) is not Module.set_extra_state:
        if extra_state_key in state_dict:
            self.set_extra_state(state_dict[extra_state_key])
        elif strict:
            missing_keys.append(extra_state_key)
    elif strict and (extra_state_key in state_dict):
        unexpected_keys.append(extra_state_key)

    if strict:
        for key in state_dict.keys():
            if key.startswith(prefix) and key != extra_state_key:
                input_name = key[len(prefix):]
                input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                if input_name not in self._modules and input_name not in local_state:
                    unexpected_keys.append(key)

def get_sharded_torch_checkpoints(dir: str) -> List[str]:
    try:
        with open(os.path.join(dir, "pytorch_model.bin.index.json")) as file:
            j = json.load(file)
    except FileNotFoundError:
        return []

    try:
        return list(set(j["weight_map"].values()))
    except KeyError:
        return []

@contextlib.contextmanager
def use_lazy_load(
    enable=True,
    callback: Optional[Callable] = None,
    dematerialized_modules=False,
):
    if not enable:
        with use_custom_unpickler(RestrictedUnpickler):
            yield False
        return

    begin_time = time.time()
    utils.koboldai_vars.total_checkpoints = 0
    utils.koboldai_vars.loaded_checkpoints = 0

    try:
        LazyloadPatches.__enter__()

        old_rebuild_tensor = torch._utils._rebuild_tensor
        torch._utils._rebuild_tensor = _rebuild_tensor

        # Torch load patch
        old_torch_load = torch.load

        def torch_load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
            if not utils.koboldai_vars.total_checkpoints:
                checkpoints = get_sharded_torch_checkpoints(os.path.dirname(f))
                # `checkpoints` may be empty if there is an error--return 1 in
                # this case. Either there was no checkpoint index file (most
                # common case), or there was a compatibility issue while reading
                # it.
                utils.koboldai_vars.total_checkpoints = len(checkpoints) or 1

            model_dict = old_torch_load(
                f=f,
                map_location=map_location,
                pickle_module=pickle_module,
                **pickle_load_args,
            )

            if f not in torch_checkpoint_file_handles:
                torch_checkpoint_file_handles[f] = zipfile.ZipFile(f, "r")

            for k, v in model_dict.items():
                v.file_name = f

            if callback is not None:
                callback(
                    model_dict,
                    f=f,
                    map_location=map_location,
                    pickle_module=pickle_module,
                    is_safetensors=False,
                    **pickle_load_args,
                )

            return model_dict

        torch.load = torch_load

        if HAS_SAFETENSORS:
            patch_safetensors(callback)

        if dematerialized_modules:
            # Most devices can just use Accelerate's implementation, but the Transformers on
            # the TPU complains about emptied weights unless we use VE's custom patches
            if not USE_TPU_EMPTY_MODULE_METHOD:
                init_empty_weights = accelerate.init_empty_weights()
                init_empty_weights.__enter__()
            else:
                old_linear_init = torch.nn.Linear.__init__
                old_embedding_init = torch.nn.Embedding.__init__
                old_layernorm_init = torch.nn.LayerNorm.__init__

                def linear_init(self, *args, device=None, **kwargs):
                    return old_linear_init(self, *args, device="meta", **kwargs)

                def embedding_init(self, *args, device=None, **kwargs):
                    return old_embedding_init(self, *args, device="meta", **kwargs)

                def layernorm_init(self, *args, device=None, **kwargs):
                    return old_layernorm_init(self, *args, device="meta", **kwargs)

                torch.nn.Linear.__init__ = linear_init
                torch.nn.Embedding.__init__ = embedding_init
                torch.nn.LayerNorm.__init__ = layernorm_init
                old_load_from_state_dict = torch.nn.Module._load_from_state_dict
                torch.nn.Module._load_from_state_dict = _load_from_state_dict

        with use_custom_unpickler(_LazyUnpickler):
            yield True

    finally:
        LazyloadPatches.__exit__(None, None, None)
        torch._utils._rebuild_tensor = old_rebuild_tensor
        torch.load = old_torch_load

        post_load_cleanup()
        logger.debug(
            f"[lazy_load] Context closed in {round(time.time() - begin_time, 2)} seconds."
        )

        if dematerialized_modules:
            if not USE_TPU_EMPTY_MODULE_METHOD:
                init_empty_weights.__exit__(None, None, None)
            else:
                torch.nn.Linear.__init__ = old_linear_init
                torch.nn.Embedding.__init__ = old_embedding_init
                torch.nn.LayerNorm.__init__ = old_layernorm_init
                torch.nn.Module._load_from_state_dict = old_load_from_state_dict


def post_load_cleanup() -> None:
    """Close dangling file pointers and clear caches after the load is complete."""
    global torch_checkpoint_file_handles

    logger.debug(
        f"[lazy_load] CheckpointChunkCache Hit Data: {CheckpointChunkCache.hit_data}"
    )
    CheckpointChunkCache.clear(unload_model=True)

    # Bar is initialized in
    # patches.patch_transformers_for_lazyload._load_state_dict_into_meta_model,
    # as it has access to the state dict (for getting tensor count)
    utils.bar = None

    for v in torch_checkpoint_file_handles.values():
        v.close()
    torch_checkpoint_file_handles = {}
