from typing import List, Union
from tokenizers import Tokenizer
import torch
from transformers import PreTrainedTokenizer


class GenericTokenizer:
    """Bridges the gap between Transformers tokenizers and Tokenizers tokenizers. Why they aren't the same, I don't know."""

    def __init__(self, tokenizer: Union[Tokenizer, PreTrainedTokenizer]) -> None:
        self.tokenizer = tokenizer

        # TODO: Get rid of this
        self._koboldai_header = []

        self.get_vocab = tokenizer.get_vocab

    def encode(self, text: str) -> list:
        if isinstance(self.tokenizer, PreTrainedTokenizer):
            return self.tokenizer.encode(text)
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: Union[int, List[int], torch.Tensor]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()

        if isinstance(tokens, int):
            tokens = [tokens]

        return self.tokenizer.decode(tokens)
