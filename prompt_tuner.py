import abc
import os
import sys
import math
import numpy as np
import termcolor
import contextlib
import traceback
import random
import torch
import torch.nn.functional as F
from torch.nn import Embedding, CrossEntropyLoss
import transformers
from transformers import AutoTokenizer, GPT2TokenizerFast
from mkultra.tuning import GPTPromptTuningMixin, GPTNeoPromptTuningLM
from mkultra.soft_prompt import SoftPrompt
from typing import List, Optional, TextIO, Union


_PromptTuningPreTrainedModel = Union["UniversalPromptTuningMixin", GPTPromptTuningMixin, transformers.PreTrainedModel]

class _WTEMixin:
    @property
    def wte(self: Union["_WTEMixin", transformers.PreTrainedModel]):
        return self.get_input_embeddings()

    @wte.setter
    def wte(self: Union["_WTEMixin", transformers.PreTrainedModel], v):
        self.set_input_embeddings(v)


class UniversalPromptTuningMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model: _PromptTuningPreTrainedModel = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        if not hasattr(model, "transformer"):
            model.transformer = _WTEMixin()
        elif not hasattr(model.transformer, "wte"):
            assert isinstance(model.transformer, type)
            model.transformer.__class__ = type("_UniversalPromptTuning" + model.transformer.__class__.__name__, (_WTEMixin, model.transformer.__class__), {})

        model.__class__ = type("_UniversalPromptTuning" + model.__class__.__name__, (UniversalPromptTuningMixin, model.__class__), {})

        for param in model.parameters():
            param.requires_grad = False
        model.initialize_soft_prompt()

        return model

    def forward(
        self: _PromptTuningPreTrainedModel,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        assert input_ids is not None
        assert input_ids.ndim == 2

        input_ids = F.pad(input_ids, (self.learned_embedding.size(0), 0, 0, 0), value=self.transformer.wte.weight.size(0) // 2)

        if labels is not None:
            labels = self._extend_labels(labels)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        old_embedding_call = Embedding.__call__
        model = self

        def new_embedding_call(self, input_ids, *args, **kwargs):
            inputs_embeds = old_embedding_call(self, input_ids, *args, **kwargs)
            if model.transformer.wte is self:
                assert inputs_embeds.ndim == 3
                inputs_embeds[:, :model.learned_embedding.size(0), :] = model.learned_embedding[None]
            return inputs_embeds

        Embedding.__call__ = new_embedding_call

        try:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                return_dict=return_dict,
            )
        finally:
            Embedding.__call__ = old_embedding_call

for k in dir(GPTPromptTuningMixin):
    v = getattr(GPTPromptTuningMixin, k)
    _v = getattr(UniversalPromptTuningMixin, k, None)
    if _v is None or (_v is getattr(object, k, None) and callable(_v) and not isinstance(_v, type)):
        setattr(UniversalPromptTuningMixin, k, v)


class AutoPromptTuningLM(UniversalPromptTuningMixin, transformers.AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)


default_quiet = False


def get_tokenizer(model_id, revision=None) -> transformers.PreTrainedTokenizerBase:
    if(os.path.isdir(model_id)):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir="cache")
        except Exception as e:
            pass
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir="cache", use_fast=False)
        except Exception as e:
            try:
                tokenizer = GPT2TokenizerFast.from_pretrained(model_id, revision=revision, cache_dir="cache")
            except Exception as e:
                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=revision, cache_dir="cache")
    elif(os.path.isdir("models/{}".format(vars.model.replace('/', '_')))):
        try:
            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(vars.model.replace('/', '_')), revision=revision, cache_dir="cache")
        except Exception as e:
            pass
        try:
            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(vars.model.replace('/', '_')), revision=revision, cache_dir="cache", use_fast=False)
        except Exception as e:
            try:
                tokenizer = GPT2TokenizerFast.from_pretrained("models/{}".format(vars.model.replace('/', '_')), revision=revision, cache_dir="cache")
            except Exception as e:
                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=revision, cache_dir="cache")
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(vars.model, revision=revision, cache_dir="cache")
        except Exception as e:
            pass
        try:
            tokenizer = AutoTokenizer.from_pretrained(vars.model, revision=revision, cache_dir="cache", use_fast=False)
        except Exception as e:
            try:
                tokenizer = GPT2TokenizerFast.from_pretrained(vars.model, revision=revision, cache_dir="cache")
            except Exception as e:
                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=revision, cache_dir="cache")

    @contextlib.contextmanager
    def _kai_no_prefix():
        add_bos_token = getattr(tokenizer, "add_bos_token", False)
        add_prefix_space = getattr(tokenizer, "add_prefix_space", False)
        tokenizer.add_bos_token = False
        tokenizer.add_prefix_space = False
        try:
            yield
        finally:
            tokenizer.add_bos_token = add_bos_token
            tokenizer.add_prefix_space = add_prefix_space

    tokenizer._kai_no_prefix = _kai_no_prefix
    return tokenizer


class ConfigurationError(Exception):
    def __init__(self, msg: str = "Unknown error", code: int = 1, quiet: Optional[bool] = None):
        if quiet is None:
            quiet = default_quiet
        super().__init__(msg)
        self.code = code
        self.quiet = quiet


class TrainerBase(abc.ABC):
    @abc.abstractmethod
    def startup(self, step: int) -> None:
        ...

    @abc.abstractmethod
    def get_batch(self, step: int, size: int) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_num_sequences(self) -> int:
        ...

    @abc.abstractmethod
    def get_initial_soft_embeddings(self, model: transformers.PreTrainedModel) -> SoftPrompt:
        ...

    @abc.abstractmethod
    def tokenize_dataset_callback(self, tokenizer: transformers.PreTrainedTokenizerBase, text: str) -> List[int]:
        ...

    class TrainerData:
        def __init__(self):
            self.__lazy_load_spec: Optional[dict] = None
            self.model_spec: Optional[dict] = None
            self.tokenizer_id: Optional[str] = None
            self.newlinemode: Optional[str] = None
            self.ckpt_path: Optional[str] = None
            self.save_file: Optional[str] = None
            self.params: Optional[dict] = None
            self.stparams: Optional[dict] = None
            self.gradient_accumulation_steps = -1
            self.soft_in_dim = -1
            self.prompt_method = "tokens"
            self.prompt_seed = 42

        @property
        def lazy_load_spec(self):
            print("WARNING:  `TrainerData.lazy_load_spec` is currently unused", file=sys.stderr)
            return self.__lazy_load_spec

        @lazy_load_spec.setter
        def lazy_load_spec(self, value: Optional[dict]):
            print("WARNING:  `TrainerData.lazy_load_spec` is currently unused", file=sys.stderr)
            self.__lazy_load_spec = value

        @property
        def kaiming_size(self):  # backwards compatibility
            return self.soft_in_dim

        @kaiming_size.setter
        def kaiming_size(self, value: int):  # backwards compatibility
            self.prompt_method = "kaiming"
            self.soft_in_dim = value

    data: TrainerData

    def __init__(self, universe: Optional[int] = None, quiet=False):
        self.quiet = quiet
        self.universe = universe
        self.data = self.TrainerData()
        self._spmodule: Optional[str] = None
        if universe is not None:
            print("WARNING:  The `universe` argument of `TrainerBase.__init__` is currently unused", file=sys.stderr)

    def raise_configuration_error(self, msg, **kwargs):
        if "quiet" not in kwargs:
            kwargs["quiet"] = self.quiet
        raise ConfigurationError(msg, **kwargs)

    def get_hf_checkpoint_metadata(self) -> bool:
        return True

    def get_tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        return get_tokenizer(self.ckpt_path)

    def export_to_kobold(self, output_file: str, name: str, author: str, supported: str, description: str):
        pass

    def export_to_mkultra(self, output_file: str, soft_prompt_name: str, soft_prompt_description: str):
        pass

    def tokenize_dataset(
        self,
        dataset_path: Union[str, TextIO],
        output_file: Union[str, TextIO],
        batch_size=2048,
        epochs=1,
        use_ftfy=True,
        shuffle_seed: Optional[Union[int, float, str, bytes, bytearray]] = 1729,
    ):
        dataset_path = dataset_path.replace("\\", "/")
        output_file = output_file.replace("\\", "/")
        if not isinstance(batch_size, int) or batch_size < 1:
            self.raise_configuration_error(
                "batch_size must be an integer greater than zero.", code=9
            )
        if (
            not isinstance(epochs, int) and not isinstance(epochs, float)
        ) or epochs <= 0:
            self.raise_configuration_error(
                "epochs must be an int or float greater than zero.", code=10
            )
        if isinstance(output_file, str) and output_file.endswith("/"):
            self.raise_configuration_error(
                "output_file should be the path to a file, not a directory.", code=11
            )
        if isinstance(dataset_path, str) and not os.path.exists(dataset_path):
            self.raise_configuration_error(
                "dataset_path is not set to a valid file or directory.", code=12
            )

        if use_ftfy:
            import ftfy

        tokenizer = self.get_tokenizer()

        batch_size = min(
            batch_size,
            self.data.params["max_batch_size"] - self.data.soft_in_dim,
        )
        assert batch_size >= 0
        print(
            termcolor.colored(
                "\nIf you see a warning somewhere below about token indices, ignore it.  That warning is normal.\n",
                "magenta",
            )
        )
        print("Batch size:", batch_size)
        print(termcolor.colored("Tokenizing your dataset...\n", "magenta"))

        if not isinstance(dataset_path, str):
            files = [dataset_path]
        elif os.path.isfile(dataset_path):
            files = [dataset_path]
        else:
            files = sorted(
                os.path.join(dataset_path, filename)
                for filename in os.listdir(dataset_path)
            )
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(files)
        tokens = []
        eos = tokenizer.decode(self.data.params["eos_token"])
        for path in files:
            if isinstance(path, str):
                f = open(path)
            else:
                f = path
            try:
                text = f.read()
                if use_ftfy:
                    text = ftfy.fix_text(text)
                text = text.replace("<|endoftext|>", eos)
                tokens.extend(self.tokenize_dataset_callback(tokenizer, text))
            finally:
                if isinstance(path, str):
                    f.close()

        print("Dataset size (in tokens):", len(tokens))
        if len(tokens) < batch_size + 1:
            self.raise_configuration_error(
                "Your dataset is too small!  The number of tokens has to be greater than the batch size.  Try increasing the epochs.",
                code=13,
            )
        tail = len(tokens) % (batch_size + 1)
        if tail:
            print(
                f"We're removing the last {tail} tokens from your dataset to make the length a multiple of {batch_size+1}."
            )
            tokens = tokens[:-tail]

        tokens = np.array(tokens, dtype=np.uint16).reshape((-1, batch_size + 1))
        sequences_per_epoch = tokens.shape[0]
        _epochs = math.ceil(epochs)
        if _epochs > 1:
            rng = np.random.Generator(np.random.PCG64(1729))
            tokens = np.concatenate(
                (
                    tokens,
                    *(rng.permutation(tokens, axis=0) for i in range(_epochs - 1)),
                ),
                axis=0,
            )
        tokens = tokens[: math.ceil(epochs * sequences_per_epoch)]
        print(f"Total sequences in your dataset: {tokens.shape[0]}")

        if isinstance(output_file, str):
            f = open(output_file, "w")
        else:
            f = output_file
        try:
            np.save(output_file, tokens)
        finally:
            if isinstance(output_file, str):
                f.close()

    def train(self):
        if self.data.params is not None and "max_batch_size" not in self.data.params:
            self.data.params["max_batch_size"] = 2048

        if not os.path.exists(self.data.save_file):
            print("We are starting a brand new soft-tuning session.\n")
            self.startup(step=-1)
            if self.data.soft_in_dim <= 0:
                self.raise_configuration_error(
                    "You have not set a soft prompt size.", code=6
                )
        else:
            # If we're resuming a soft-tuning session, the soft prompt tensor is
            # already in the save file and we just have to decode it.
            try:
                z = torch.load(self.data.save_file)
                assert z["step"] > 0
                assert z["tensor"].ndim == 2 and "opt_state" in z
                assert z["tensor"].shape[0] < self.data.params["max_batch_size"]
                self.data.soft_in_dim = z["tensor"].shape[0]
                step = z["step"]
                opt_state = z["opt_state"]
            except AssertionError:
                self.raise_configuration_error("MTJSP file is corrupted.", code=14)
            print(f"We're resuming a previous soft-tuning session at step {step+1}.\n")
            self.startup(step=step + 1)
            soft_embeddings = z["tensor"]

        REVISION = None

        tokenizer = self.get_tokenizer()
        model: _PromptTuningPreTrainedModel

        if(os.path.isdir(self.data.ckpt_path)):
            try:
                model     = AutoPromptTuningLM.from_pretrained(self.data.ckpt_path, revision=REVISION, cache_dir="cache")
            except Exception as e:
                if("out of memory" in traceback.format_exc().lower()):
                    raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                model     = GPTNeoPromptTuningLM.from_pretrained(self.data.ckpt_path, revision=REVISION, cache_dir="cache")
        elif(os.path.isdir("models/{}".format(vars.model.replace('/', '_')))):
            try:
                model     = AutoPromptTuningLM.from_pretrained("models/{}".format(vars.model.replace('/', '_')), revision=REVISION, cache_dir="cache")
            except Exception as e:
                if("out of memory" in traceback.format_exc().lower()):
                    raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                model     = GPTNeoPromptTuningLM.from_pretrained("models/{}".format(vars.model.replace('/', '_')), revision=REVISION, cache_dir="cache")
        else:
            try:
                model     = AutoPromptTuningLM.from_pretrained(vars.model, revision=REVISION, cache_dir="cache")
            except Exception as e:
                if("out of memory" in traceback.format_exc().lower()):
                    raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                model     = GPTNeoPromptTuningLM.from_pretrained(vars.model, revision=REVISION, cache_dir="cache")

        if step == 0:
            soft_embeddings = self.get_initial_soft_embeddings(model)
        else:
            soft_embeddings = SoftPrompt.from_inputs_embeds(soft_embeddings)
        model.set_soft_prompt(soft_embeddings)

        steps = self.get_num_sequences() // self.data.gradient_accumulation_steps
        warmup_steps = max(1, round(steps * self.data.stparams["warmup"]))

        beta1: Optional[float] = self.data.stparams.get("beta1", 0.0)
        if beta1 == 0.0:
            beta1 = None
        optimizer = transformers.Adafactor(
            params=model.get_soft_params(),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=self.data.stparams["lr"],
            beta1=beta1,
            decay_rate=self.data.stparams.get("decay_rate", -0.8),
            weight_decay=self.data.stparams.get("weight_decay", 0.1),
        )
        if step != 0:
            optimizer.load_state_dict(opt_state)
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=steps - warmup_steps,
            num_cycles=(steps - warmup_steps) // self.data.stparams.get("training_steps_per_cycle", 56),
        )

        torch.cuda.empty_cache()
        optimizer.state['step'] = step
        cross_entropy_loss = CrossEntropyLoss()

        while step < steps:
            model.train()

            total_loss = total_grad = total_grad_norm = 0

            for i in range(self.data.gradient_accumulation_steps):
                # Get the next sequence from the dataset
                block = self.get_batch(step, self.data.gradient_accumulation_steps).to(model.transformer.wte.weight.device)

                # input_ids is the context to the model (without the soft prompt) and labels is what we expect the model to generate (the -100s represent soft prompt tokens for which loss is not calculated)
                input_ids = block[:-1].unsqueeze(0).detach()
                labels = torch.cat((torch.full((model.get_soft_params().size(0) - 1,), -100, device=block.device), block)).unsqueeze(0).cuda().detach()

                # Give the context to the model and compare the model's output logits with the labels to compute the loss
                logits = model(input_ids=input_ids, labels=input_ids).logits
                loss: torch.Tensor = cross_entropy_loss(logits.view(-1, model.transformer.wte.weight.size(1)), labels.view(-1))
                total_loss += loss.detach()

                # Compute the gradient of the loss function and add it to model.get_soft_params().grad (model.get_soft_params().grad += gradient)
                loss.backward()

                total_grad_norm += torch.linalg.norm(model.get_soft_params().grad.detach() - total_grad)
                total_grad = model.get_soft_params().grad.detach()

                del input_ids
                del labels
                del logits
                torch.cuda.empty_cache()

            mean_loss = (total_loss / self.data.gradient_accumulation_steps).item()
            mean_grad_norm = (total_grad_norm / self.data.gradient_accumulation_steps).item()

            # Apply the optimization algorithm using the accumulated gradients, which changes the contents of the soft prompt matrix very slightly to reduce the loss
            optimizer.step()
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            optimizer.zero_grad()

            # Save checkpoint every few steps
            pass

            step += 1
