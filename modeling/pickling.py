from __future__ import annotations

import collections
import contextlib
import pickle

import _codecs
import numpy as np
import torch
from torch import Tensor

import modeling


def _patched_rebuild_from_type_v2(func, new_type, args, state):
    """A patched version of torch._tensor._rebuild_from_type_v2 that
    does not attempt to convert `LazyTensor`s to `torch.Tensor`s."""

    ret = func(*args)

    # BEGIN PATCH
    transformation_ok = isinstance(ret, modeling.lazy_loader.LazyTensor) and new_type == Tensor
    if type(ret) is not new_type and not transformation_ok:
        # END PATCH
        ret = ret.as_subclass(new_type)

    # Tensor does define __setstate__ even though it doesn't define
    # __getstate__. So only use __setstate__ if it is NOT the one defined
    # on Tensor
    if (
        getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
        is not Tensor.__setstate__
    ):
        ret.__setstate__(state)
    else:
        ret = torch._utils._set_obj_state(ret, state)
    return ret


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
        elif module == "torch._utils" and name in (
            "_rebuild_tensor_v2",
            "_rebuild_meta_tensor_no_storage",
        ):
            return getattr(torch._utils, name)
        elif module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return _patched_rebuild_from_type_v2
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
            "Tensor",
            "float16",
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
            raise pickle.UnpicklingError(
                f"`{qualified_name}` is forbidden; the model you are loading probably contains malicious code. If you think this is incorrect ask the developer to unban the ability for {module} to execute {name}"
            )

    def load(self, *args, **kwargs):
        self.original_persistent_load = getattr(
            self, "persistent_load", pickle.Unpickler.persistent_load
        )
        self.persistent_load = self.forced_persistent_load
        return super().load(*args, **kwargs)


@contextlib.contextmanager
def use_custom_unpickler(unpickler: pickle.Unpickler = RestrictedUnpickler):
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
