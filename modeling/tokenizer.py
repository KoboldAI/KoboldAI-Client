from typing import Any, List, Union
from tokenizers import Tokenizer
import torch
from transformers import PreTrainedTokenizer


class GenericTokenizer:
    """Bridges the gap between Transformers tokenizers and Tokenizers tokenizers. Why they aren't the same, I don't know."""

    def __init__(self, tokenizer: Union[Tokenizer, PreTrainedTokenizer]) -> None:
        self.tokenizer = tokenizer
        self.valid_tokens = set(self.tokenizer.vocab.values())

    def __getattr__(self, name: str) -> Any:
        # Fall back to tokenizer for non-generic stuff
        return getattr(self.tokenizer, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # To prevent infinite recursion on __init__ setting
        if name == "tokenizer":
            super().__setattr__(name, value)
            return
        setattr(self.tokenizer, name, value)

    def encode(self, text: str) -> list:
        ret = self.tokenizer.encode(text)
        if isinstance(ret, list):
            return ret
        return ret.ids

    def decode(self, tokens: Union[int, List[int], torch.Tensor]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()

        if isinstance(tokens, int):
            tokens = [tokens]
        
        for t in tokens:
            if t not in self.valid_tokens:
                print(f"WHAT ON EARTH IS {t}")
        tokens = [t for t in tokens if t in self.valid_tokens]

        return self.tokenizer.decode(tokens)
