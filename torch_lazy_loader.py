'''
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
'''


import contextlib
from functools import reduce
import itertools
import zipfile
import pickle
import torch
import numpy as np
import collections
import _codecs
import utils
import os
from torch.nn import Module
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union


_EXTRA_STATE_KEY_SUFFIX = '_extra_state'


STORAGE_TYPE_MAP = {
    torch.float64: torch.DoubleStorage,
    torch.float32: torch.FloatStorage,
    torch.float16: torch.HalfStorage,
    torch.int64: torch.LongStorage,
    torch.int32: torch.IntStorage,
    torch.int16: torch.ShortStorage,
    torch.int8: torch.CharStorage,
    torch.uint8: torch.ByteStorage,
    torch.bool: torch.BoolStorage,
    torch.bfloat16: torch.BFloat16Storage,
}


class LazyTensor:
    def __init__(self, storage_type, key: str, location: str, dtype: Optional[torch.dtype] = None, seek_offset: Optional[int] = None, shape: Optional[Tuple[int, ...]] = None, stride: Optional[Tuple[int, ...]] = None, requires_grad=False, backward_hooks: Any = None):
        self.storage_type = storage_type
        self.key = key
        self.location = location
        self.dtype = dtype
        self.seek_offset = seek_offset
        self.shape = shape
        self.stride = stride
        self.requires_grad = requires_grad
        self.backward_hooks = backward_hooks

    def __view(self, f: Callable):
        return f"{type(self).__name__}(storage_type={f(self.storage_type)}, key={f(self.key)}, location={f(self.location)}, dtype={f(self.dtype)}, seek_offset={f(self.seek_offset)}, shape={f(self.shape)}, stride={f(self.stride)}, requires_grad={f(self.requires_grad)}, backward_hooks={f(self.backward_hooks)})"

    def __repr__(self):
        return self.__view(repr)

    def materialize(self, checkpoint: Union[zipfile.ZipFile, zipfile.ZipExtFile], map_location=None, no_grad=True, filename="pytorch_model.bin") -> torch.Tensor:
        filename = os.path.basename(os.path.normpath(filename)).split('.')[0]
        size = reduce(lambda x, y: x * y, self.shape, 1)
        dtype = self.dtype
        nbytes = size if dtype is torch.bool else size * ((torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits >> 3)
        if isinstance(checkpoint, zipfile.ZipFile):
            try:
                f = checkpoint.open(f"archive/data/{self.key}", "r")
            except:
                f = checkpoint.open(f"{filename}/data/{self.key}", "r")
            f.read(self.seek_offset)
        else:
            f = checkpoint
        try:
            storage = STORAGE_TYPE_MAP[dtype].from_buffer(f.read(nbytes), "little")
        finally:
            if isinstance(checkpoint, zipfile.ZipFile):
                f.close()
        storage = torch.serialization._get_restore_location(map_location)(storage, self.location)
        tensor = torch.tensor([], dtype=storage.dtype, device=storage.device)
        tensor.set_(storage, 0, self.shape, self.stride)
        tensor.requires_grad = not no_grad and self.requires_grad
        tensor._backward_hooks = self.backward_hooks
        return tensor

class RestrictedUnpickler(pickle.Unpickler):
    def original_persistent_load(self, saved_id):
        return super().persistent_load(saved_id)

    def forced_persistent_load(self, saved_id):
        if saved_id[0] != "storage":
            raise pickle.UnpicklingError("`saved_id[0]` must be 'storage'")
        return self.original_persistent_load(saved_id)

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        elif module == "torch._utils" and name == "_rebuild_tensor_v2":
            return torch._utils._rebuild_tensor_v2
        elif module == "torch" and name in (
            "DoubleStorage",
            "FloatStorage",
            "HalfStorage",
            "LongStorage",
            "IntStorage",
            "ShortStorage",
            "CharStorage",
            "ByteStorage",
            "BoolStorage",
            "BFloat16Storage",
        ):
            return getattr(torch, name)
        elif module == "numpy.core.multiarray" and name == "scalar":
            return np.core.multiarray.scalar
        elif module == "numpy" and name == "dtype":
            return np.dtype
        elif module == "_codecs" and name == "encode":
            return _codecs.encode
        else:
            # Forbid everything else.
            qualified_name = name if module == "__builtin__" else f"{module}.{name}"
            raise pickle.UnpicklingError(f"`{qualified_name}` is forbidden; the model you are loading probably contains malicious code")

    def load(self, *args, **kwargs):
        self.original_persistent_load = getattr(self, "persistent_load", pickle.Unpickler.persistent_load)
        self.persistent_load = self.forced_persistent_load
        return super().load(*args, **kwargs)

class _LazyUnpickler(RestrictedUnpickler):
    lazy_loaded_storages: Dict[str, LazyTensor]

    def __init__(self, *args, **kwargs):
        self.lazy_loaded_storages = {}
        return super().__init__(*args, **kwargs)

    def forced_persistent_load(self, saved_id):
        assert isinstance(saved_id, tuple)
        typename = saved_id[0]
        assert typename == "storage", f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, key, location, _ = saved_id[1:]
        return LazyTensor(storage_type, key, location)

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
    lazy_storage.seek_offset = storage_offset if dtype is torch.bool else storage_offset * ((torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits >> 3)
    return lazy_storage


# Modified version of https://github.com/pytorch/pytorch/blob/v1.11.0-rc4/torch/nn/modules/module.py#L1346-L1438
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

    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
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


@contextlib.contextmanager
def use_custom_unpickler(unpickler: Type[pickle.Unpickler] = RestrictedUnpickler):
    try:
        old_unpickler = pickle.Unpickler
        pickle.Unpickler = unpickler

        old_pickle_load = pickle.load

        def new_pickle_load(*args, **kwargs):
            return pickle.Unpickler(*args, **kwargs).load()

        pickle.load = new_pickle_load

        yield

    finally:
        pickle.Unpickler = old_unpickler
        pickle.load = old_pickle_load

@contextlib.contextmanager
def use_lazy_torch_load(enable=True, callback: Optional[Callable] = None, dematerialized_modules=False, use_accelerate_init_empty_weights=False):
    if not enable:
        with use_custom_unpickler(RestrictedUnpickler):
            yield False
        return

    try:
        old_rebuild_tensor = torch._utils._rebuild_tensor
        torch._utils._rebuild_tensor = _rebuild_tensor

        old_torch_load = torch.load

        def torch_load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
            retval = old_torch_load(f=f, map_location=map_location, pickle_module=pickle_module, **pickle_load_args)
            if callback is not None:
                callback(retval, f=f, map_location=map_location, pickle_module=pickle_module, **pickle_load_args)
            return retval

        torch.load = torch_load

        if dematerialized_modules:
            if use_accelerate_init_empty_weights and utils.HAS_ACCELERATE:
                import accelerate
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
        torch._utils._rebuild_tensor = old_rebuild_tensor
        torch.load = old_torch_load
        if dematerialized_modules:
            if use_accelerate_init_empty_weights and utils.HAS_ACCELERATE:
                init_empty_weights.__exit__(None, None, None)
            else:
                torch.nn.Linear.__init__ = old_linear_init
                torch.nn.Embedding.__init__ = old_embedding_init
                torch.nn.LayerNorm.__init__ = old_layernorm_init
                torch.nn.Module._load_from_state_dict = old_load_from_state_dict
