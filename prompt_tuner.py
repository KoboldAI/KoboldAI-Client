import abc
import os
import sys
import math
import numpy as np
import termcolor
import contextlib
import traceback
import random
import zipfile
import json
import uuid
import datetime
import base64
import pickle
import hashlib
import itertools
import functools
import bisect
import eventlet
import packaging
import gc
import time
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Embedding, CrossEntropyLoss
import transformers
from transformers import __version__ as transformers_version
from transformers import AutoTokenizer, GPT2Tokenizer, AutoConfig, AutoModelForCausalLM, GPTNeoForCausalLM, PreTrainedModel, modeling_utils
import accelerate
import accelerate.utils
from mkultra.tuning import GPTPromptTuningMixin, GPTNeoPromptTuningLM
from mkultra.soft_prompt import SoftPrompt
from typing import Dict, List, Optional, TextIO, Union

import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)

import breakmodel
import torch_lazy_loader
import utils

use_breakmodel = True


class colors:
    PURPLE    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    END       = '\033[0m'
    UNDERLINE = '\033[4m'

class Send_to_socketio(object):
    def write(self, bar):
        print(bar, end="")
        time.sleep(0.01)
        try:
            if utils.emit is not None:
                utils.emit('from_server', {'cmd': 'model_load_status', 'data': bar.replace(" ", "&nbsp;")}, broadcast=True)
        except:
            pass

def patch_transformers_download():
    global transformers
    import copy, requests, tqdm, time
    class Send_to_socketio(object):
        def write(self, bar):
            bar = bar.replace("\r", "").replace("\n", "")
            if bar != "":
                try:
                    print(bar, end="\r")
                    if utils.emit is not None:
                        utils.emit('from_server', {'cmd': 'model_load_status', 'data': bar.replace(" ", "&nbsp;")}, broadcast=True)
                    eventlet.sleep(seconds=0)
                except:
                    pass
    def http_get(
        url: str,
        temp_file: transformers.utils.hub.BinaryIO,
        proxies=None,
        resume_size=0,
        headers: transformers.utils.hub.Optional[transformers.utils.hub.Dict[str, str]] = None,
        file_name: transformers.utils.hub.Optional[str] = None,
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
        total = resume_size + int(content_length) if content_length is not None else None
        # `tqdm` behavior is determined by `utils.logging.is_progress_bar_enabled()`
        # and can be set using `utils.logging.enable/disable_progress_bar()`
        if url[-11:] != 'config.json':
            progress = tqdm.tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=total,
                initial=resume_size,
                desc=f"Downloading {file_name}" if file_name is not None else "Downloading",
                file=Send_to_socketio(),
            )
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                if url[-11:] != 'config.json':
                    progress.update(len(chunk))
                temp_file.write(chunk)
        if url[-11:] != 'config.json':
            progress.close()

    transformers.utils.hub.http_get = http_get


def patch_transformers():
    global transformers
    
    patch_transformers_download()
    
    old_from_pretrained = PreTrainedModel.from_pretrained.__func__
    @classmethod
    def new_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        utils.num_shards = None
        utils.current_shard = 0
        utils.from_pretrained_model_name = pretrained_model_name_or_path
        utils.from_pretrained_index_filename = None
        utils.from_pretrained_kwargs = kwargs
        utils.bar = None
        if utils.args is None or not utils.args.no_aria2:
            utils.aria2_hook(pretrained_model_name_or_path, **kwargs)
        return old_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    if(not hasattr(PreTrainedModel, "_kai_patched")):
        PreTrainedModel.from_pretrained = new_from_pretrained
        PreTrainedModel._kai_patched = True
    if(hasattr(modeling_utils, "get_checkpoint_shard_files")):
        old_get_checkpoint_shard_files = modeling_utils.get_checkpoint_shard_files
        def new_get_checkpoint_shard_files(pretrained_model_name_or_path, index_filename, *args, **kwargs):
            utils.num_shards = utils.get_num_shards(index_filename)
            utils.from_pretrained_index_filename = index_filename
            return old_get_checkpoint_shard_files(pretrained_model_name_or_path, index_filename, *args, **kwargs)
        modeling_utils.get_checkpoint_shard_files = new_get_checkpoint_shard_files
        
    # Some versions of transformers 4.17.0.dev0 are affected by
    # https://github.com/huggingface/transformers/issues/15736
    # This is a workaround for those versions of transformers.
    if(transformers_version == "4.17.0.dev0"):
        try:
            from transformers.models.xglm.modeling_xglm import XGLMSinusoidalPositionalEmbedding
        except ImportError:
            pass
        else:
            @torch.no_grad()
            def new_forward(self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0):
                bsz, seq_len = inputs_embeds.size()[:-1]
                input_shape = inputs_embeds.size()[:-1]
                sequence_length = input_shape[1]
                position_ids = torch.arange(
                    past_key_values_length + self.padding_idx + 1, past_key_values_length + sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
                ).unsqueeze(0).expand(input_shape).contiguous()
                max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
                if max_pos > self.weights.size(0):
                    self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)
                return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()
            XGLMSinusoidalPositionalEmbedding.forward = new_forward


    # Fix a bug in OPTForCausalLM where self.lm_head is the wrong size
    if(packaging.version.parse("4.19.0.dev0") <= packaging.version.parse(transformers_version) < packaging.version.parse("4.20.0")):
        try:
            from transformers import OPTForCausalLM, OPTModel
        except ImportError:
            pass
        else:
            # This is the same as the original __init__ but with
            # config.hidden_size
            # replaced with
            # config.word_embed_proj_dim
            def new_init(self, config):
                super(OPTForCausalLM, self).__init__(config)
                self.model = OPTModel(config)
                self.lm_head = torch.nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
                self.post_init()
            OPTForCausalLM.__init__ = new_init


def device_list(n_layers, primary=None, selected=None):
    device_count = torch.cuda.device_count()
    if(device_count < 2):
        primary = None
    gpu_blocks = breakmodel.gpu_blocks + (device_count - len(breakmodel.gpu_blocks))*[0]
    print(f"{colors.YELLOW}       DEVICE ID  |  LAYERS  |  DEVICE NAME{colors.END}")
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        if(len(name) > 47):
            name = "..." + name[-44:]
        row_color = colors.END
        sep_color = colors.YELLOW
        print(f"{row_color}{colors.YELLOW + '->' + row_color if i == selected else '  '} {'(primary)' if i == primary else ' '*9} {i:3}  {sep_color}|{row_color}     {gpu_blocks[i]:3}  {sep_color}|{row_color}  {name}{colors.END}")
    row_color = colors.END
    sep_color = colors.YELLOW
    print(f"{row_color}{colors.YELLOW + '->' + row_color if -1 == selected else '  '} {' '*9} N/A  {sep_color}|{row_color}     {breakmodel.disk_blocks:3}  {sep_color}|{row_color}  (Disk cache){colors.END}")
    print(f"{row_color}   {' '*9} N/A  {sep_color}|{row_color}     {n_layers:3}  {sep_color}|{row_color}  (CPU){colors.END}")


def move_model_to_devices(model, usegpu, gpu_device):
    global generator

    if(not use_breakmodel):
        if(usegpu):
            model = model.half().to(gpu_device)
        else:
            model = model.to('cpu').float()
        generator = model.generate
        return

    for key, value in model.state_dict().items():
        target_dtype = torch.float32 if breakmodel.primary_device == "cpu" else torch.float16
        if(value.dtype is not target_dtype):
            accelerate.utils.set_module_tensor_to_device(model, key, target_dtype)
    disk_blocks = breakmodel.disk_blocks
    gpu_blocks = breakmodel.gpu_blocks
    ram_blocks = len(utils.layers_module_names) - sum(gpu_blocks)
    cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))
    device_map = {}
    for name in utils.layers_module_names:
        layer = int(name.rsplit(".", 1)[1])
        device = ("disk" if layer < disk_blocks else "cpu") if layer < ram_blocks else bisect.bisect_right(cumulative_gpu_blocks, layer - ram_blocks)
        device_map[name] = device
    for name in utils.get_missing_module_names(model, list(device_map.keys())):
        device_map[name] = breakmodel.primary_device
    breakmodel.dispatch_model_ex(model, device_map, main_device=breakmodel.primary_device, offload_buffers=True, offload_dir="accelerate-disk-cache")
    gc.collect()
    generator = model.generate
    return


_PromptTuningPreTrainedModel = Union["UniversalPromptTuningMixin", GPTPromptTuningMixin, transformers.PreTrainedModel]

class _WTEDummy:
    def __init__(self, model: transformers.PreTrainedModel):
        self.model = model

    @property
    def wte(self: "_WTEDummy"):
        return self.model.get_input_embeddings()

    @wte.setter
    def wte(self: "_WTEDummy", v):
        self.model.set_input_embeddings(v)

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
            model.transformer = _WTEDummy(model)
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
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir="cache", use_fast=False)
        except Exception as e:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir="cache")
            except Exception as e:
                try:
                    tokenizer = GPT2Tokenizer.from_pretrained(model_id, revision=revision, cache_dir="cache")
                except Exception as e:
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=revision, cache_dir="cache")
    elif(os.path.isdir("models/{}".format(model_id.replace('/', '_')))):
        try:
            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(model_id.replace('/', '_')), revision=revision, cache_dir="cache", use_fast=False)
        except Exception as e:
            try:
                tokenizer = AutoTokenizer.from_pretrained("models/{}".format(model_id.replace('/', '_')), revision=revision, cache_dir="cache")
            except Exception as e:
                try:
                    tokenizer = GPT2Tokenizer.from_pretrained("models/{}".format(model_id.replace('/', '_')), revision=revision, cache_dir="cache")
                except Exception as e:
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=revision, cache_dir="cache")
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir="cache", use_fast=False)
        except Exception as e:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir="cache")
            except Exception as e:
                try:
                    tokenizer = GPT2Tokenizer.from_pretrained(model_id, revision=revision, cache_dir="cache")
                except Exception as e:
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=revision, cache_dir="cache")

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
    
    def _get_model_config(self) -> transformers.configuration_utils.PretrainedConfig:
        REVISION = None
        if(os.path.isdir(self.data.ckpt_path)):
            model_config     = AutoConfig.from_pretrained(self.data.ckpt_path, revision=REVISION, cache_dir="cache")
        elif(os.path.isdir("models/{}".format(self.data.ckpt_path.replace('/', '_')))):
            model_config     = AutoConfig.from_pretrained("models/{}".format(self.data.ckpt_path.replace('/', '_')), revision=REVISION, cache_dir="cache")
        else:
            model_config     = AutoConfig.from_pretrained(self.data.ckpt_path, revision=REVISION, cache_dir="cache")
        return model_config

    def get_hf_checkpoint_metadata(self) -> bool:
        params = {}
        model_config = self._get_model_config()
        params["tokenizer_id"] = self.data.ckpt_path
        tokenizer = get_tokenizer(self.data.ckpt_path)
        params["newlinemode"] = params.get(
            "newlinemode", "s" if model_config.model_type == "xglm" else "n"
        )
        params["max_batch_size"] = 2048
        with tokenizer._kai_no_prefix():
            params["eos_token"] = (
                [50259, 50259] if model_config.model_type == "xglm" and model_config.eos_token_id == 50259 else [model_config.eos_token_id]
            )
        params["seq"] = 2048
        self.data.params = params
        return True

    def get_tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        return get_tokenizer(self.data.ckpt_path)
    
    def save_data(self):
        pass

    def export_to_kobold(self, output_file: str, name: str, author: str, supported: str, description: str):
        try:
            z = torch.load(self.data.save_file)
            assert z["step"] > 0
            assert z["tensor"].ndim == 2 and "opt_state" in z
            assert z["tensor"].shape[0] < self.data.params["max_batch_size"]
            self.data.soft_in_dim = z["tensor"].shape[0]
        except AssertionError:
            self.raise_configuration_error("MKUSP file is corrupted.", code=14)

        tensor = z["tensor"]

        meta = {
            "name": name,
            "author": author,
            "supported": supported,
            "description": description,
        }
        if len(meta["author"].strip()) == 0:
            meta.pop("author")
        meta["supported"] = list(map(lambda m: m.strip(), supported.split(",")))

        with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_LZMA) as z:
            with z.open("tensor.npy", "w") as f:
                np.save(f, tensor.detach().cpu().numpy(), allow_pickle=False)
        with zipfile.ZipFile(output_file, "a", compression=zipfile.ZIP_STORED) as z:
            with z.open("meta.json", "w") as f:
                f.write(json.dumps(meta, indent=2).encode("utf-8"))

    def export_to_mkultra(self, output_file: str, soft_prompt_name: str, soft_prompt_description: str):
        try:
            z = torch.load(self.data.save_file)
            assert z["step"] > 0
            assert z["tensor"].ndim == 2 and "opt_state" in z
            assert z["tensor"].shape[0] < self.data.params["max_batch_size"]
            self.data.soft_in_dim = z["tensor"].shape[0]
            _step = z["step"]
        except AssertionError:
            self.raise_configuration_error("MKUSP file is corrupted.", code=14)

        tensor = z["tensor"]

        with open(output_file, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "step": _step,
                        "loss": float(z["loss"]),
                        "uuid": str(uuid.uuid4()),
                        "name": soft_prompt_name,
                        "description": soft_prompt_description,
                        "epoch": datetime.datetime.now().timestamp(),
                    },
                    "tensor": base64.b64encode(
                        pickle.dumps(
                            tensor.detach().cpu(),
                            protocol=4,
                        ),
                    ).decode("ascii"),
                },
                f,
            )

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

    def train(
        self,
        breakmodel_primary_device: Optional[Union[str, int, torch.device]] = None,
        breakmodel_gpulayers: Optional[List[int]] = None,
        breakmodel_disklayers = 0,
    ):
        if breakmodel_gpulayers is None:
            breakmodel_gpulayers = []
        if breakmodel_primary_device is None:
            breakmodel_primary_device = 0 if sum(x if x >= 0 else 1 for x in breakmodel_gpulayers) else "cpu"

        if self.data.params is not None and "max_batch_size" not in self.data.params:
            self.data.params["max_batch_size"] = 2048

        if not os.path.exists(self.data.save_file):
            print("We are starting a brand new soft-tuning session.\n")
            self.startup(step=-1)
            if self.data.soft_in_dim <= 0:
                self.raise_configuration_error(
                    "You have not set a soft prompt size.", code=6
                )
            step = 0
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
                self.raise_configuration_error("MKUSP file is corrupted.", code=14)
            print(f"We're resuming a previous soft-tuning session at step {step+1}.\n")
            self.startup(step=step + 1)
            soft_embeddings = z["tensor"]

        REVISION = None

        patch_transformers()

        model: _PromptTuningPreTrainedModel

        model_config = self._get_model_config()
        n_layers = utils.num_layers(model_config)
        breakmodel_gpulayers = [x if x >= 0 else n_layers for x in breakmodel_gpulayers]

        convert_to_float16 = True
        hascuda = torch.cuda.is_available()
        usegpu = hascuda and not breakmodel_disklayers and len(breakmodel_gpulayers) == 1 and breakmodel_gpulayers[0] == n_layers
        gpu_device = breakmodel_primary_device
        use_breakmodel = bool(hascuda or breakmodel_disklayers or sum(breakmodel_gpulayers))

        assert len(breakmodel_gpulayers) <= torch.cuda.device_count()
        assert sum(breakmodel_gpulayers) + breakmodel_disklayers <= n_layers

        breakmodel.gpu_blocks = breakmodel_gpulayers
        breakmodel.disk_blocks = breakmodel_disklayers
        disk_blocks = breakmodel.disk_blocks
        gpu_blocks = breakmodel.gpu_blocks
        ram_blocks = ram_blocks = n_layers - sum(gpu_blocks)
        cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))

        device_list(ram_blocks, primary=breakmodel.primary_device)

        def lazy_load_callback(model_dict: Dict[str, Union[torch_lazy_loader.LazyTensor, torch.Tensor]], f, **_):
            if lazy_load_callback.nested:
                return
            lazy_load_callback.nested = True

            device_map: Dict[str, Union[str, int]] = {}

            @functools.lru_cache(maxsize=None)
            def get_original_key(key):
                return max((original_key for original_key in utils.module_names if original_key.endswith(key)), key=len)

            for key, value in model_dict.items():
                original_key = get_original_key(key)
                if isinstance(value, torch_lazy_loader.LazyTensor) and not any(original_key.startswith(n) for n in utils.layers_module_names):
                    device_map[key] = gpu_device if hascuda and usegpu else "cpu" if not hascuda or not use_breakmodel else breakmodel.primary_device
                else:
                    layer = int(max((n for n in utils.layers_module_names if original_key.startswith(n)), key=len).rsplit(".", 1)[1])
                    device = gpu_device if hascuda and usegpu else "disk" if layer < disk_blocks and layer < ram_blocks else "cpu" if not hascuda or not use_breakmodel else "shared" if layer < ram_blocks else bisect.bisect_right(cumulative_gpu_blocks, layer - ram_blocks)
                    device_map[key] = device

            if utils.num_shards is None or utils.current_shard == 0:
                utils.offload_index = {}
                if os.path.isdir("accelerate-disk-cache"):
                    # Delete all of the files in the disk cache folder without deleting the folder itself to allow people to create symbolic links for this folder
                    # (the folder doesn't contain any subfolders so os.remove will do just fine)
                    for filename in os.listdir("accelerate-disk-cache"):
                        try:
                            os.remove(os.path.join("accelerate-disk-cache", filename))
                        except OSError:
                            pass
                os.makedirs("accelerate-disk-cache", exist_ok=True)
                if utils.num_shards is not None:
                    num_tensors = len(utils.get_sharded_checkpoint_num_tensors(utils.from_pretrained_model_name, utils.from_pretrained_index_filename, **utils.from_pretrained_kwargs))
                else:
                    num_tensors = len(device_map)
                print(flush=True)
                utils.bar = tqdm(total=num_tensors, desc="Loading model tensors", file=Send_to_socketio())

            with zipfile.ZipFile(f, "r") as z:
                try:
                    last_storage_key = None
                    f = None
                    current_offset = 0
                    able_to_pin_layers = True
                    if utils.num_shards is not None:
                        utils.current_shard += 1
                    for key in sorted(device_map.keys(), key=lambda k: (model_dict[k].key, model_dict[k].seek_offset)):
                        storage_key = model_dict[key].key
                        if storage_key != last_storage_key or model_dict[key].seek_offset < current_offset:
                            last_storage_key = storage_key
                            if isinstance(f, zipfile.ZipExtFile):
                                f.close()
                            f = z.open(f"archive/data/{storage_key}")
                            current_offset = 0
                        if current_offset != model_dict[key].seek_offset:
                            f.read(model_dict[key].seek_offset - current_offset)
                            current_offset = model_dict[key].seek_offset
                        device = device_map[key]
                        size = functools.reduce(lambda x, y: x * y, model_dict[key].shape, 1)
                        dtype = model_dict[key].dtype
                        nbytes = size if dtype is torch.bool else size * ((torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits >> 3)
                        #print(f"Transferring <{key}>  to  {f'({device.upper()})' if isinstance(device, str) else '[device ' + str(device) + ']'} ... ", end="", flush=True)
                        model_dict[key] = model_dict[key].materialize(f, map_location="cpu")
                        # if model_dict[key].dtype is torch.float32:
                        #     fp32_model = True
                        if convert_to_float16 and breakmodel.primary_device != "cpu" and hascuda and (use_breakmodel or usegpu) and model_dict[key].dtype is torch.float32:
                            model_dict[key] = model_dict[key].to(torch.float16)
                        if breakmodel.primary_device == "cpu" or (not usegpu and not use_breakmodel and model_dict[key].dtype is torch.float16):
                            model_dict[key] = model_dict[key].to(torch.float32)
                        if device == "shared":
                            model_dict[key] = model_dict[key].to("cpu").detach_()
                            if able_to_pin_layers:
                                try:
                                    model_dict[key] = model_dict[key].pin_memory()
                                except:
                                    able_to_pin_layers = False
                        elif device == "disk":
                            accelerate.utils.offload_weight(model_dict[key], get_original_key(key), "accelerate-disk-cache", index=utils.offload_index)
                            model_dict[key] = model_dict[key].to("meta")
                        else:
                            model_dict[key] = model_dict[key].to(device)
                        #print("OK", flush=True)
                        current_offset += nbytes
                        utils.bar.update(1)
                finally:
                    if utils.num_shards is None or utils.current_shard >= utils.num_shards:
                        if utils.offload_index:
                            for name, tensor in utils.named_buffers:
                                if name not in utils.offload_index:
                                    accelerate.utils.offload_weight(tensor, name, "accelerate-disk-cache", index=utils.offload_index)
                            accelerate.utils.save_offload_index(utils.offload_index, "accelerate-disk-cache")
                        utils.bar.close()
                        utils.bar = None
                    lazy_load_callback.nested = False
                    if isinstance(f, zipfile.ZipExtFile):
                        f.close()

        lazy_load_callback.nested = False

        # Since we're using lazy loader, we need to figure out what the model's hidden layers are called
        with torch_lazy_loader.use_lazy_torch_load(dematerialized_modules=True, use_accelerate_init_empty_weights=True):
            try:
                metamodel = AutoModelForCausalLM.from_config(model_config)
            except Exception as e:
                metamodel = GPTNeoForCausalLM.from_config(model_config)
            utils.layers_module_names = utils.get_layers_module_names(metamodel)
            utils.module_names = list(metamodel.state_dict().keys())
            utils.named_buffers = list(metamodel.named_buffers(recurse=True))

        with torch_lazy_loader.use_lazy_torch_load(callback=lazy_load_callback, dematerialized_modules=True):
            if(os.path.isdir(self.data.ckpt_path)):
                try:
                    model     = AutoPromptTuningLM.from_pretrained(self.data.ckpt_path, revision=REVISION, cache_dir="cache")
                except Exception as e:
                    if("out of memory" in traceback.format_exc().lower()):
                        raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                    model     = GPTNeoPromptTuningLM.from_pretrained(self.data.ckpt_path, revision=REVISION, cache_dir="cache")
            elif(os.path.isdir("models/{}".format(self.data.ckpt_path.replace('/', '_')))):
                try:
                    model     = AutoPromptTuningLM.from_pretrained("models/{}".format(self.data.ckpt_path.replace('/', '_')), revision=REVISION, cache_dir="cache")
                except Exception as e:
                    if("out of memory" in traceback.format_exc().lower()):
                        raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                    model     = GPTNeoPromptTuningLM.from_pretrained("models/{}".format(self.data.ckpt_path.replace('/', '_')), revision=REVISION, cache_dir="cache")
            else:
                try:
                    model     = AutoPromptTuningLM.from_pretrained(self.data.ckpt_path, revision=REVISION, cache_dir="cache")
                except Exception as e:
                    if("out of memory" in traceback.format_exc().lower()):
                        raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                    model     = GPTNeoPromptTuningLM.from_pretrained(self.data.ckpt_path, revision=REVISION, cache_dir="cache")

        if(hascuda):
            if(usegpu):
                model = model.half().to(gpu_device)
            elif(use_breakmodel):  # Use both RAM and VRAM (breakmodel)
                move_model_to_devices(model, usegpu, gpu_device)
            elif(__import__("breakmodel").disk_blocks > 0):
                move_model_to_devices(model, usegpu, gpu_device)
            else:
                model = model.to('cpu').float()
        elif(__import__("breakmodel").disk_blocks > 0):
            move_model_to_devices(model, usegpu, gpu_device)
        else:
            model.to('cpu').float()

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
            params=(model.get_soft_params(),),
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

        def save_mkusp(
            loss,
            grad_norm,
        ):
            with open(self.data.save_file, "wb") as f:
                torch.save(
                    {
                        "tensor": soft_embeddings.get_inputs_embeds(),
                        "opt_state": optimizer.state_dict(),
                        "step": step,
                        "loss": loss,
                        "grad_norm": grad_norm,
                    },
                    f,
                )
            self.save_data()
        
        bar1 = tqdm(initial=step + 1, total=steps, desc="CURRENT TRAINING STEP")

        while step < steps:
            step += 1
            model.train()

            total_loss = total_grad = total_grad_norm = 0

            # Get the next sequences from the dataset
            block = torch.tensor(np.int32(self.get_batch(step, self.data.gradient_accumulation_steps))).to(model.transformer.wte.weight.device)

            for sequence in tqdm(block, desc="GRADIENT ACCUMULATION", leave=False):
                # input_ids is the context to the model (without the soft prompt) and labels is what we expect the model to generate (the -100s represent soft prompt tokens for which loss is not calculated)
                input_ids = sequence[:-1].unsqueeze(0).detach()
                labels = torch.cat((torch.full((model.get_soft_params().size(0) - 1,), -100, device=sequence.device), sequence), dim=-1).unsqueeze(0).detach()

                # Give the context to the model and compare the model's output logits with the labels to compute the loss
                logits = model(input_ids=input_ids, labels=input_ids).logits
                loss: torch.Tensor = cross_entropy_loss(logits.view(-1, model.transformer.wte.weight.size(0)), labels.view(-1))
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
            if step == 1 or step % self.data.stparams["save_every"] == 0:
                save_mkusp(mean_loss, mean_grad_norm)

            bar1.set_postfix({"loss": mean_loss, "grad_norm": mean_grad_norm, "learning_rate": lr})
            bar1.update()


class BasicTrainer(TrainerBase):
    class TrainerData(TrainerBase.TrainerData):
        def __init__(self):
            super().__init__()
            self.dataset_file: Optional[str] = None
            self.initial_softprompt: Optional[List[int]] = None

    data: "BasicTrainer.TrainerData"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset: Optional[np.ndarray] = None

    def startup(self, step: int) -> None:
        if self.get_num_sequences() < self.data.gradient_accumulation_steps:
            self.raise_configuration_error(
                "Your dataset is too small!  gradient_accumulation_steps must be less than or equal to the number of sequences.",
                code=101,
            )
        if (
            self.data.prompt_method == "tokens"
            and step < 0
            and self.data.initial_softprompt is None
        ):
            self.raise_configuration_error(
                "You have not set an initial soft prompt string.", code=103
            )
        if self.data.prompt_method == "tokens" and step < 0:
            self.data.soft_in_dim = len(self.data.initial_softprompt)

    def get_batch(self, step: int, size: int) -> np.ndarray:
        return self.dataset[(step - 1) * size : step * size]

    def get_num_sequences(self) -> int:
        if self.dataset is None:
            if self.data.dataset_file is None or not os.path.exists(
                self.data.dataset_file
            ):
                self.raise_configuration_error(
                    f"Dataset file not found at {repr(self.data.dataset_file)}",
                    code=102,
                )
            self.dataset = np.load(self.data.dataset_file, mmap_mode="r")
        assert self.dataset.ndim >= 2
        assert self.dataset.shape[0] >= 2
        return self.dataset.shape[0]

    def get_initial_soft_embeddings(self, model: transformers.PreTrainedModel) -> SoftPrompt:
        if self.data.prompt_method == "vocab_sample":
            rng = np.random.Generator(
                np.random.PCG64(
                    [
                        self.data.prompt_seed,
                        int.from_bytes(hashlib.sha256(model.config.model_type.encode("utf8")).digest()[:4], "little"),
                    ]
                )
            )
            tokenizer = self.get_tokenizer()
            with tokenizer._kai_no_prefix():
                special_tokens = set(
                    itertools.chain.from_iterable(
                        tokenizer.encode(str(v))
                        for v in tokenizer.special_tokens_map_extended.values()
                    )
                )
            sample_space = [
                k for k in range(model.get_input_embeddings().weight.shape[-2]) if k not in special_tokens
            ]
            sample = rng.choice(sample_space, self.data.soft_in_dim, False)
            return SoftPrompt.from_inputs_embeds(model.get_input_embeddings()(torch.tensor(sample, dtype=torch.int32, device=model.get_input_embeddings().weight.device)))
        elif self.data.prompt_method == "tokens":
            return SoftPrompt.from_inputs_embeds(model.get_input_embeddings()(torch.tensor(self.data.initial_softprompt, dtype=torch.int32, device=model.get_input_embeddings().weight.device)))
        self.raise_configuration_error(
            f"Unknown prompt method {repr(self.data.prompt_method)}", code=104
        )

    def tokenize_dataset_callback(
        self, tokenizer: transformers.PreTrainedTokenizerBase, text: str
    ) -> List[int]:
        if self.data.newlinemode == "s":
            text = text.replace("\n", "</s>")
        with tokenizer._kai_no_prefix():
            return tokenizer.encode(text) + self.data.params["eos_token"]
