import os
from typing import Optional
from transformers import AutoConfig

import utils
import koboldai_settings
from logger import logger
from modeling.inference_model import InferenceModel


class HFInferenceModel(InferenceModel):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_config = None
        self.model_name = model_name

        self.model = None
        self.tokenizer = None

    def _post_load(self) -> None:
        # These are model specific tokenizer overrides if a model has bad defaults
        if utils.koboldai_vars.model_type == "llama":
            # Note: self.tokenizer is a GenericTokenizer, and self.tokenizer.tokenizer is the actual LlamaTokenizer
            self.tokenizer.add_bos_token = False

            # HF transformers no longer supports decode_with_prefix_space
            # We work around this by wrapping decode, encode, and __call__
            # with versions that work around the 'prefix space' misfeature
            # of sentencepiece.
            vocab = self.tokenizer.convert_ids_to_tokens(range(self.tokenizer.vocab_size))
            has_prefix_space = {i for i, tok in enumerate(vocab) if tok.startswith("▁")}

            # Wrap 'decode' with a method that always returns text starting with a space
            # when the head token starts with a space. This is what 'decode_with_prefix_space'
            # used to do, and we implement it using the same technique (building a cache of
            # tokens that should have a prefix space, and then prepending a space if the first
            # token is in this set.) We also work around a bizarre behavior in which decoding
            # a single token 13 behaves differently than decoding a squence containing only [13].
            original_decode = type(self.tokenizer.tokenizer).decode
            def decode_wrapper(self, token_ids, *args, **kwargs):
                first = None
                # Note, the code below that wraps single-value token_ids in a list
                # is to work around this wonky behavior:
                #   >>> t.decode(13)
                #   '<0x0A>'
                #   >>> t.decode([13])
                #   '\n'
                # Not doing this causes token streaming to receive <0x0A> characters
                # instead of newlines.
                if isinstance(token_ids, int):
                    first = token_ids
                    token_ids = [first]
                elif hasattr(token_ids, 'dim'): # Check for e.g. torch.Tensor
                    # Tensors don't support the Python standard of 'empty is False'
                    # and the special case of dimension 0 tensors also needs to be
                    # handled separately.
                    if token_ids.dim() == 0:
                        first = int(token_ids.item())
                        token_ids = [first]
                    elif len(token_ids) > 0:
                        first = int(token_ids[0])
                elif token_ids:
                    first = token_ids[0]
                result = original_decode(self, token_ids, *args, **kwargs)
                if first is not None and first in has_prefix_space:
                    result = " " + result
                return result
            # GenericTokenizer overrides __setattr__ so we need to use object.__setattr__ to bypass it
            object.__setattr__(self.tokenizer, 'decode', decode_wrapper.__get__(self.tokenizer))

            # Wrap encode and __call__ to work around the 'prefix space' misfeature also.
            # The problem is that "Bob" at the start of text is encoded as if it is
            # " Bob". This creates a problem because it means you can't split text, encode
            # the pieces, concatenate the tokens, decode them, and get the original text back.
            # The workaround is to prepend a known token that (1) starts with a space; and
            # (2) is not the prefix of any other token. After searching through the vocab
            # " ," (space comma) is the only token containing only printable ascii characters
            # that fits this bill. By prepending ',' to the text, the original encode
            # method always returns [1919, ...], where the tail of the sequence is the
            # actual encoded result we want without the prefix space behavior.
            original_encode = type(self.tokenizer.tokenizer).encode
            def encode_wrapper(self, text, *args, **kwargs):
                if type(text) is str:
                    text = ',' + text
                    result = original_encode(self, text, *args, **kwargs)
                    result = result[1:]
                else:
                    result = original_encode(self, text, *args, **kwargs)
                return result
            object.__setattr__(self.tokenizer, 'encode', encode_wrapper.__get__(self.tokenizer))

            # Since 'encode' is documented as being deprecated, also override __call__.
            # This doesn't appear to currently be used by KoboldAI, but doing so
            # in case someone uses it in the future.
            original_call = type(self.tokenizer.tokenizer).__call__
            def call_wrapper(self, text, *args, **kwargs):
                if type(text) is str:
                    text = ',' + text
                    result = original_call(self, text, *args, **kwargs)
                    result = result[1:]
                else:
                    result = original_call(self, text, *args, **kwargs)
                return result
            object.__setattr__(self.tokenizer, '__call__', call_wrapper.__get__(self.tokenizer))

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

        basename = utils.koboldai_vars.model.replace("/", "_")
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
