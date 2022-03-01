import contextlib
import pickle
import torch
from typing import Any, Callable, Dict, Optional, Tuple, Type


class LazyTensor:
    def __init__(self, storage_type: Type[torch._StorageBase], key: str, location: str, nelements: int, storage_offset: Optional[int] = None, shape: Optional[Tuple[int, ...]] = None, stride: Optional[Tuple[int, ...]] = None, requires_grad=False, backward_hooks: Any = None):
        self.storage_type = storage_type
        self.key = key
        self.location = location
        self.nelements = nelements
        self.storage_offset = storage_offset
        self.shape = shape
        self.stride = stride
        self.requires_grad = requires_grad
        self.backward_hooks = backward_hooks

    def __view(self, f: Callable):
        return f"{type(self).__name__}(storage_type={f(self.storage_type)}, key={f(self.key)}, location={f(self.location)}, nelements={f(self.nelements)}, storage_offset={f(self.storage_offset)}, shape={f(self.shape)}, stride={f(self.stride)}, requires_grad={f(self.requires_grad)}, backward_hooks={f(self.backward_hooks)})"

    def __repr__(self):
        return self.__view(repr)

    def materialize(self, checkpoint: torch._C.PyTorchFileReader, map_location=None) -> torch.Tensor:
        storage_dtype = self.storage_type(0).dtype
        storage = checkpoint.get_storage_from_record(f"data/{self.key}", self.nelements, storage_dtype).storage()
        storage = torch.serialization._get_restore_location(map_location)(storage, self.location)
        tensor = torch.tensor([], dtype=storage.dtype, device=storage.device)
        tensor.set_(storage, self.storage_offset, self.shape, self.stride)
        tensor.requires_grad = self.requires_grad
        tensor._backward_hooks = self.backward_hooks
        return tensor


class _LazyUnpickler(pickle.Unpickler):
    lazy_loaded_storages: Dict[str, LazyTensor]

    def __init__(self, *args, **kwargs):
        self.lazy_loaded_storages = {}
        return super().__init__(*args, **kwargs)

    def forced_persistent_load(self, saved_id):
        assert isinstance(saved_id, tuple)
        typename = saved_id[0]
        assert typename == "storage", f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"

        storage_type, key, location, nelements = saved_id[1:]

        if key not in self.lazy_loaded_storages:
            self.lazy_loaded_storages[key] = LazyTensor(storage_type, key, location, nelements)
        
        return self.lazy_loaded_storages[key]

    def load(self, *args, **kwargs):
        self.persistent_load = self.forced_persistent_load
        retval = super().load(*args, **kwargs)
        self.lazy_loaded_storages = {}
        return retval


def _rebuild_tensor(lazy_storage: LazyTensor, storage_offset, shape, stride):
    lazy_storage.storage_offset = storage_offset
    lazy_storage.shape = shape
    lazy_storage.stride = stride
    return lazy_storage


@contextlib.contextmanager
def use_lazy_torch_load(enable=True, callback: Optional[Callable] = None):
    if not enable:
        yield False
        return

    old_unpickler = pickle.Unpickler
    pickle.Unpickler = _LazyUnpickler

    old_rebuild_tensor = torch._utils._rebuild_tensor
    torch._utils._rebuild_tensor = _rebuild_tensor

    old_torch_load = torch.load

    def torch_load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
        retval = old_torch_load(f=f, map_location=map_location, pickle_module=pickle_module, **pickle_load_args)
        if callback is not None:
            callback(retval, f=f, map_location=map_location, pickle_module=pickle_module, **pickle_load_args)
        return retval

    torch.load = torch_load

    yield True

    pickle.Unpickler = old_unpickler
    torch._utils._rebuild_tensor = old_rebuild_tensor
    torch.load = old_torch_load
