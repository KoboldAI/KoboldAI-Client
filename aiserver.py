#!/usr/bin/python3
#==================================================================#
# KoboldAI
# Version: 1.19.2
# By: The KoboldAI Community
#==================================================================#

# External packages
from dataclasses import dataclass
from enum import Enum
import random
import shutil
import eventlet

from modeling.inference_model import GenerationMode

eventlet.monkey_patch(all=True, thread=False, os=False)
import os, inspect, contextlib, pickle
os.system("")
__file__ = os.path.dirname(os.path.realpath(__file__))
os.chdir(__file__)
os.environ['EVENTLET_THREADPOOL_SIZE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from eventlet import tpool

import logging
from logger import logger, set_logger_verbosity, quiesce_logger
from ansi2html import Ansi2HTMLConverter

logging.getLogger("urllib3").setLevel(logging.ERROR)

import attention_bias
attention_bias.do_patches()

from os import path, getcwd
import time
import re
import json
import ijson
import datetime
import collections
import zipfile
import packaging.version
import traceback
import markdown
import bleach
import functools
import traceback
import inspect
import warnings
import multiprocessing
import numpy as np
from collections import OrderedDict
from typing import Any, Callable, TypeVar, Tuple, Union, Dict, Set, List, Optional, Type

import requests
import html
import argparse
import sys
import gc
import traceback

import lupa
# Hack to make the new Horde worker understand its imports...
try:
    sys.path.append(os.path.abspath("AI-Horde-Worker"))
except:
    pass

# KoboldAI
import fileops
import gensettings
from utils import debounce
import utils
import koboldai_settings
import torch
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from modeling.ipex import ipex_init
        ipex_init()
except Exception:
    pass
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForTokenClassification
import transformers
import ipaddress
from functools import wraps
from modeling.pickling import RestrictedUnpickler, use_custom_unpickler

# Make settings folder early so we can depend on it anywhere
if not os.path.exists("settings/"):
    os.mkdir("settings")

try:
    from transformers.models.opt.modeling_opt import OPTDecoder
except:
    pass

# Text2img
import base64
from PIL import Image
from io import BytesIO

global tpu_mtj_backend
global allowed_ips
allowed_ips = set()  # empty set
enable_whitelist = False


if lupa.LUA_VERSION[:2] != (5, 4):
    logger.error(f"Please install lupa==1.10. You have lupa {lupa.__version__}.")

patch_causallm_patched = False

# Make sure tqdm progress bars display properly in Colab
from tqdm.auto import tqdm
old_init = tqdm.__init__
def new_init(self, *args, **kwargs):
    old_init(self, *args, **kwargs)
    if 'ncols' in kwargs:
        if(self.ncols == 0 and kwargs.get("ncols") != 0):
            self.ncols = 99
tqdm.__init__ = new_init

# Add _koboldai_header support for some optional tokenizer fixes
# This used to be an OPT tokenizer fix, this has been moved search for "# These are model specific overrides if a model has bad defaults" for the new section
from transformers import PreTrainedTokenizerBase
old_pretrainedtokenizerbase_from_pretrained = PreTrainedTokenizerBase.from_pretrained.__func__
@classmethod
def new_pretrainedtokenizerbase_from_pretrained(cls, *args, **kwargs):
    tokenizer = old_pretrainedtokenizerbase_from_pretrained(cls, *args, **kwargs)
    tokenizer._koboldai_header = []
    return tokenizer
PreTrainedTokenizerBase.from_pretrained = new_pretrainedtokenizerbase_from_pretrained


def is_model_downloaded(model_name: str) -> bool:
    model_stub = model_name.replace("/", "_")
    return os.path.isdir(os.path.join("models", model_stub))

#==================================================================#
# Variables & Storage
#==================================================================#

# Terminal tags for colored text
class colors:
    PURPLE    = '\033[95m'
    BLUE      = '\033[94m'
    CYAN      = '\033[96m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    END       = '\033[0m'
    UNDERLINE = '\033[4m'

class MenuModelType(Enum):
    HUGGINGFACE = 0
    ONLINE_API = 1
    OTHER = 2

class MenuItem:
    def __init__(
        self,
        label: str,
        name: str,
        experimental: bool = False
    ) -> None:
        self.label = label
        self.name = name
        self.experimental = experimental

    def should_show(self) -> bool:
        return koboldai_vars.experimental_features or not self.experimental

class MenuFolder(MenuItem):
    def to_ui1(self) -> list:
        return [
            self.label,
            self.name,
            "",
            True,
        ]
    
    def to_json(self) -> dict:
        return {
            "label": self.label,
            "name": self.name,
            "size": "",
            "isMenu": True,
            "isDownloaded": False,
            "isDirectory":  False
        }

class MenuModel(MenuItem):
    def __init__(
        self,
        label: str,
        name: str,
        vram_requirements: str = "",
        model_type: MenuModelType = MenuModelType.HUGGINGFACE,
        experimental: bool = False,
        model_backend: str = "Huggingface",
    ) -> None:
        super().__init__(label, name, experimental)
        self.model_type = model_type
        self.vram_requirements = vram_requirements
        self.is_downloaded = is_model_downloaded(self.name)
        self.model_backend = model_backend
    
    def to_ui1(self) -> list:
        return [
            self.label,
            self.name,
            self.vram_requirements,
            False,
            self.is_downloaded
        ]

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "name": self.name,
            "size": self.vram_requirements,
            "isMenu": False,
            "isDownloaded": self.is_downloaded,
            "isDirectory": False,
        }

class MenuPath(MenuItem):
    def to_ui1(self) -> list:
        return [
            self.label,
            self.name,
            "",
            True,
        ]
    
    def to_json(self) -> dict:
        return {
            "label": self.label,
            "name": self.name,
            "size": "",
            "isMenu": True,
            "isDownloaded": False,
            "isDirectory": True,
            "path": "./models"
        }

# AI models Menu
# This is a dict of lists where they key is the menu name, and the list is the menu items.
# Each item takes the 4 elements, 1: Text to display, 2: Model Name (koboldai_vars.model) or menu name (Key name for another menu),
# 3: the memory requirement for the model, 4: if the item is a menu or not (True/False)
model_menu = {
    "mainmenu": [
        MenuPath("Load a model from its directory", "NeoCustom"),
        MenuPath("Load an old GPT-2 model (eg CloverEdition)", "GPT2Custom"),
        MenuModel("Load custom Pytorch model from Hugging Face", "customhuggingface", ""),
        MenuModel("Load old GPTQ model from Hugging Face", "customgptq", "", model_backend="GPTQ"),
        MenuFolder("Instruct Models", "instructlist"),
        MenuFolder("Novel Models", "novellist"),
        MenuFolder("Chat Models", "chatlist"),
        MenuFolder("NSFW Models", "nsfwlist"),
        MenuFolder("Adventure Models", "adventurelist"),
        MenuFolder("Untuned OPT", "optlist"),
        MenuFolder("Untuned GPT-Neo/J", "gptneolist"),
        MenuFolder("Untuned Pythia", "pythialist"),
        MenuFolder("Untuned Fairseq Dense", "fsdlist"),
        MenuFolder("Untuned Bloom", "bloomlist"),
        MenuFolder("Untuned XGLM", "xglmlist"),
        MenuFolder("Official RWKV-4", "rwkvlist"),
        MenuFolder("Untuned GPT2", "gpt2list"),
        MenuFolder("Online Services", "apilist"),
        MenuModel("Read Only (No AI)", "ReadOnly", model_type=MenuModelType.OTHER, model_backend="Read Only"),
    ],
    'instructlist': [
        MenuModel("Holomax 13B", "KoboldAI/LLaMA2-13B-Holomax", "12GB*"),        
        MenuModel("Mythomax 13B", "Gryphe/MythoMax-L2-13b", "12GB*"),
        MenuModel("Chronos-Hermes V2 13B", "Austism/chronos-hermes-13b-v2", "12GB*"),
        MenuModel("Legerdemain 13B", "CalderaAI/13B-Legerdemain-L2", "12GB*"),
        MenuModel("Chronos 13b v2", "elinas/chronos-13b-v2", "12GB*"),  
        MenuModel("Huginn 13B", "The-Face-Of-Goonery/Huginn-13b-FP16", "12GB*"),
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'adventurelist': [
        MenuFolder("Instruct models may perform better than the models below (Using Instruct mode)", "instructlist"),
        MenuModel("Skein 20B", "KoboldAI/GPT-NeoX-20B-Skein", "20GB*"),
        MenuModel("Nerys OPT 13B V2 (Hybrid)", "KoboldAI/OPT-13B-Nerys-v2", "12GB"),
        MenuModel("Spring Dragon 13B", "Henk717/spring-dragon", "12GB*"),
        MenuModel("Nerys FSD 13B V2 (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys-v2", "12GB"),
        MenuModel("Nerys FSD 13B (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys", "12GB"),
        MenuModel("Skein 6B", "KoboldAI/GPT-J-6B-Skein", "8GB*"),
        MenuModel("OPT Nerys 6B V2 (Hybrid)", "KoboldAI/OPT-6B-nerys-v2", "8GB"),
        MenuModel("Adventure 6B", "KoboldAI/GPT-J-6B-Adventure", "8GB*"),
        MenuModel("Nerys FSD 2.7B (Hybrid)", "KoboldAI/fairseq-dense-2.7B-Nerys", "6GB"),
        MenuModel("Adventure 2.7B", "KoboldAI/GPT-Neo-2.7B-AID", "6GB"),
        MenuModel("Adventure 1.3B", "KoboldAI/GPT-Neo-1.3B-Adventure", "4GB*"),
        MenuModel("Adventure 125M (Mia)", "Merry/AID-Neo-125M", "2GB"),
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'novellist': [
        MenuModel("Nerys OPT 13B V2 (Hybrid)", "KoboldAI/OPT-13B-Nerys-v2", "32GB"),
        MenuModel("Nerys FSD 13B V2 (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys-v2", "32GB"),
        MenuModel("Janeway FSD 13B", "KoboldAI/fairseq-dense-13B-Janeway", "32GB"),
        MenuModel("Nerys FSD 13B (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys", "32GB"),
        MenuModel("OPT Nerys 6B V2 (Hybrid)", "KoboldAI/OPT-6B-nerys-v2", "16GB"),
        MenuModel("Janeway FSD 6.7B", "KoboldAI/fairseq-dense-6.7B-Janeway", "16GB"),
        MenuModel("Janeway Neo 6B", "KoboldAI/GPT-J-6B-Janeway", "16GB"),
        MenuModel("Qilin Lit 6B (SFW)", "rexwang8/qilin-lit-6b", "16GB"),       
        MenuModel("Janeway Neo 2.7B", "KoboldAI/GPT-Neo-2.7B-Janeway", "8GB"),
        MenuModel("Janeway FSD 2.7B", "KoboldAI/fairseq-dense-2.7B-Janeway", "8GB"),
        MenuModel("Nerys FSD 2.7B (Hybrid)", "KoboldAI/fairseq-dense-2.7B-Nerys", "8GB"),
        MenuModel("Horni-LN 2.7B", "KoboldAI/GPT-Neo-2.7B-Horni-LN", "8GB"),
        MenuModel("Picard 2.7B (Older Janeway)", "KoboldAI/GPT-Neo-2.7B-Picard", "8GB"),
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'nsfwlist': [
        MenuFolder("Looking for NSFW Chat RP? Most chat models give better replies", "chatlist"),
        MenuModel("Green Devil (Novel)", "Pirr/pythia-13b-deduped-green_devil", "14GB"),
        MenuModel("Erebus 20B (Novel)", "KoboldAI/GPT-NeoX-20B-Erebus", "20GB*"),
        MenuModel("Nerybus 13B (Novel)", "KoboldAI/OPT-13B-Nerybus-Mix", "12GB"),
        MenuModel("Erebus 13B (Novel)", "KoboldAI/OPT-13B-Erebus", "12GB"),
        MenuModel("Shinen FSD 13B (Novel)", "KoboldAI/fairseq-dense-13B-Shinen", "12GB"),
        MenuModel("Erebus 6.7B (Novel)", "KoboldAI/OPT-6.7B-Erebus", "8GB"),
        MenuModel("Shinen FSD 6.7B (Novel)", "KoboldAI/fairseq-dense-6.7B-Shinen", "8GB"),
        MenuModel("Lit V2 6B (Novel)", "hakurei/litv2-6B-rev3", "8GB*"),
        MenuModel("Lit 6B (Novel)", "hakurei/lit-6B", "8GB*"),
        MenuModel("Shinen 6B (Novel)", "KoboldAI/GPT-J-6B-Shinen", "6GB"),
        MenuModel("Erebus 2.7B (Novel)", "KoboldAI/OPT-2.7B-Erebus", "6GB"),
        MenuModel("Horni 2.7B (Novel)", "KoboldAI/GPT-Neo-2.7B-Horni", "6GB"),
        MenuModel("Shinen 2.7B (Novel)", "KoboldAI/GPT-Neo-2.7B-Shinen", "6GB"),
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'chatlist': [
        MenuModel("Pygmalion-2 13B", "PygmalionAI/pygmalion-2-13b", "12GB*"),
        MenuModel("Mythalion 13B", "PygmalionAI/mythalion-13b", "12GB*"),
        MenuModel("Mythomax 13B (Instruct)", "Gryphe/MythoMax-L2-13b", "12GB*"),
        MenuModel("Huginn 13B (Instruct)", "The-Face-Of-Goonery/Huginn-13b-FP16", "12GB*"),
        MenuModel("Pygmalion-2 7B", "PygmalionAI/pygmalion-2-7b", "8GB*"),
        MenuModel("Pygmalion 6B", "PygmalionAI/pygmalion-6b", "8GB*"),
        MenuModel("Pygmalion 2.7B", "PygmalionAI/pygmalion-2.7b", "6GB"),
        MenuModel("Pygmalion 1.3B", "PygmalionAI/pygmalion-1.3b", "4GB*"),
        MenuModel("Pygmalion 350M", "PygmalionAI/pygmalion-350m", "2GB"),
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'gptneolist': [
        MenuModel("GPT-NeoX 20B", "EleutherAI/gpt-neox-20b", "64GB"),
        MenuModel("Pythia 13B (NeoX, Same dataset)", "EleutherAI/pythia-13b", "32GB"),
        MenuModel("GPT-J 6B", "EleutherAI/gpt-j-6B", "16GB"),
        MenuModel("GPT-Neo 2.7B", "EleutherAI/gpt-neo-2.7B", "8GB"),
        MenuModel("GPT-Neo 1.3B", "EleutherAI/gpt-neo-1.3B", "6GB"),
        MenuModel("Pythia 800M (NeoX, Same dataset)", "EleutherAI/pythia-800m", "4GB"),
        MenuModel("Pythia 350M (NeoX, Same dataset)", "EleutherAI/pythia-350m", "2GB"),
        MenuModel("GPT-Neo 125M", "EleutherAI/gpt-neo-125M", "2GB"),
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'pythialist': [
        MenuModel("Pythia 13B Deduped", "EleutherAI/pythia-13b-deduped", "32GB"),
        MenuModel("Pythia 13B", "EleutherAI/pythia-13b", "32GB"),
        MenuModel("Pythia 6.7B Deduped", "EleutherAI/pythia-6.7b-deduped", "16GB"),
        MenuModel("Pythia 6.7B", "EleutherAI/pythia-6.7b", "16GB"),
        MenuModel("Pythia 1.3B Deduped", "EleutherAI/pythia-1.3b-deduped", "6GB"),
        MenuModel("Pythia 1.3B", "EleutherAI/pythia-1.3b", "6GB"),
        MenuModel("Pythia 800M", "EleutherAI/pythia-800m", "4GB"),
        MenuModel("Pythia 350M Deduped", "EleutherAI/pythia-350m-deduped", "2GB"),
        MenuModel("Pythia 350M", "EleutherAI/pythia-350m", "2GB"),        
        MenuModel("Pythia 125M Deduped", "EleutherAI/pythia-125m-deduped", "2GB"),
        MenuModel("Pythia 125M", "EleutherAI/pythia-125m", "2GB"),
        MenuModel("Pythia 19M Deduped", "EleutherAI/pythia-19m-deduped", "1GB"),
        MenuModel("Pythia 19M", "EleutherAI/pythia-19m", "1GB"),
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'gpt2list': [
        MenuModel("GPT-2 XL", "gpt2-xl", "6GB"),
        MenuModel("GPT-2 Large", "gpt2-large", "4GB"),
        MenuModel("GPT-2 Med", "gpt2-medium", "2GB"),
        MenuModel("GPT-2", "gpt2", "2GB"),
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'bloomlist': [
        MenuModel("Bloom 176B", "bigscience/bloom"),
        MenuModel("Bloom 7.1B", "bigscience/bloom-7b1"),   
        MenuModel("Bloom 3B", "bigscience/bloom-3b"), 
        MenuModel("Bloom 1.7B", "bigscience/bloom-1b7"), 
        MenuModel("Bloom 560M", "bigscience/bloom-560m"), 
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'optlist': [
        MenuModel("OPT 66B", "facebook/opt-66b", "128GB"),
        MenuModel("OPT 30B", "facebook/opt-30b", "64GB"),
        MenuModel("OPT 13B", "facebook/opt-13b", "32GB"),
        MenuModel("OPT 6.7B", "facebook/opt-6.7b", "16GB"),
        MenuModel("OPT 2.7B", "facebook/opt-2.7b", "8GB"),
        MenuModel("OPT 1.3B", "facebook/opt-1.3b", "4GB"),
        MenuModel("OPT 350M", "facebook/opt-350m", "2GB"),
        MenuModel("OPT 125M", "facebook/opt-125m", "1GB"),
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'fsdlist': [
        MenuModel("Fairseq Dense 13B", "KoboldAI/fairseq-dense-13B", "32GB"),
        MenuModel("Fairseq Dense 6.7B", "KoboldAI/fairseq-dense-6.7B", "16GB"),
        MenuModel("Fairseq Dense 2.7B", "KoboldAI/fairseq-dense-2.7B", "8GB"),
        MenuModel("Fairseq Dense 1.3B", "KoboldAI/fairseq-dense-1.3B", "4GB"),
        MenuModel("Fairseq Dense 355M", "KoboldAI/fairseq-dense-355M", "2GB"),
        MenuModel("Fairseq Dense 125M", "KoboldAI/fairseq-dense-125M", "1GB"),
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'xglmlist': [
        MenuModel("XGLM 4.5B (Larger Dataset)", "facebook/xglm-4.5B", "12GB"),
        MenuModel("XGLM 7.5B", "facebook/xglm-7.5B", "18GB"),
        MenuModel("XGLM 2.9B", "facebook/xglm-2.9B", "10GB"),
        MenuModel("XGLM 1.7B", "facebook/xglm-1.7B", "6GB"),
        MenuModel("XGLM 564M", "facebook/xglm-564M", "4GB"),
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'rwkvlist': [
        MenuModel("RWKV Raven 14B", "RWKV/rwkv-raven-14b", ""),
        MenuModel("RWKV Pile 14B", "RWKV/rwkv-4-14b-pile", ""),
        MenuModel("RWKV Raven 7B", "RWKV/rwkv-raven-7b", ""),        
        MenuModel("RWKV Pile 7B", "RWKV/rwkv-4-7b-pile", ""), 
        MenuModel("RWKV Raven 3B", "RWKV/rwkv-raven-3b", ""), 
        MenuModel("RWKV Pile 3B", "RWKV/rwkv-4-3b-pile", ""), 
        MenuModel("RWKV Raven 1.5B", "RWKV/rwkv-raven-1b5", ""), 
        MenuModel("RWKV Pile 1.5B", "RWKV/rwkv-4-1b5-pile", ""), 
        MenuModel("RWKV Pile 430M", "RWKV/rwkv-4-430m-pile", ""), 
        MenuModel("RWKV Pile 169B", "RWKV/rwkv-4-169m-pile", ""), 
        MenuFolder("Return to Main Menu", "mainmenu"),
        ],
    'apilist': [
        MenuModel("GooseAI API (requires API key)", "GooseAI", model_type=MenuModelType.ONLINE_API, model_backend="GooseAI"),
        MenuModel("OpenAI API (requires API key)", "OAI", model_type=MenuModelType.ONLINE_API, model_backend="OpenAI"),
        MenuModel("KoboldAI API", "API", model_type=MenuModelType.ONLINE_API, model_backend="KoboldAI API"),
        MenuModel("Basic Model API", "Colab", model_type=MenuModelType.ONLINE_API, model_backend="KoboldAI Old Colab Method"),
        MenuModel("KoboldAI Horde", "CLUSTER", model_type=MenuModelType.ONLINE_API, model_backend="Horde"),
        MenuFolder("Return to Main Menu", "mainmenu"),
    ]
}

@dataclass
class ImportBuffer:
    # Singleton!!!
    prompt: Optional[str] = None
    memory: Optional[str] = None
    authors_note: Optional[str] = None
    notes: Optional[str] = None
    world_infos: Optional[dict] = None
    title: Optional[str] = None

    @dataclass
    class PromptPlaceholder:
        id: str
        order: Optional[int] = None
        default: Optional[str] = None
        title: Optional[str] = None
        description: Optional[str] = None
        value: Optional[str] = None

        def to_json(self) -> dict:
            return {key: getattr(self, key) for key in [
                "id",
                "order",
                "default",
                "title",
                "description"
            ]}
    
    def request_client_configuration(self, placeholders: List[PromptPlaceholder]) -> None:
        emit("request_prompt_config", [x.to_json() for x in placeholders], broadcast=False,  room="UI_2")

    def extract_placeholders(self, text: str) -> List[PromptPlaceholder]:
        placeholders = []

        for match in re.finditer(r"\${(.*?)}", text):
            ph_text = match.group(1)

            try:
                ph_order, ph_text = ph_text.split("#")
            except ValueError:
                ph_order = None

            if "[" not in ph_text:
                ph_id = ph_text

               # Already have it!
                if any([x.id == ph_id for x in placeholders]):
                    continue

               # Apparently, none of these characters are supported:
               # "${}[]#:@^|", however I have found some prompts using these,
               # so they will be allowed.
                for char in "${}[]":
                    if char in ph_text:
                        print("[eph] Weird char")
                        print(f"Char: {char}")
                        print(f"Ph_id: {ph_id}")
                        show_error_notification("Error loading prompt", f"Bad character '{char}' in prompt placeholder.")
                        return

                placeholders.append(self.PromptPlaceholder(
                    id=ph_id,
                    order=int(ph_order) if ph_order else None,
                ))
                continue

            ph_id, _ = ph_text.split("[")
            ph_text = ph_text.replace(ph_id, "", 1)

           # Already have it!
            if any([x.id == ph_id for x in placeholders]):
                continue

           # Match won't match it for some reason (???), so we use finditer and next()
            try:
                default_match = next(re.finditer(r"\[(.*?)\]", ph_text))
            except StopIteration:
                print("[eph] Weird brackets")
                show_error_notification("Error loading prompt", f"Unusual bracket structure in prompt.")
                return placeholders

            ph_default = default_match.group(1)
            ph_text = ph_text.replace(default_match.group(0), "")

            try:
                ph_title, ph_desc = ph_text.split(":")
            except ValueError:
                ph_title = ph_text or None
                ph_desc=None

            placeholders.append(self.PromptPlaceholder(
                id=ph_id,
                order=int(ph_order) if ph_order else None,
                default=ph_default,
                title=ph_title,
                description=ph_desc
            ))
        return placeholders

    def _replace_placeholders(self, text: str, ph_ids: dict):
        for ph_id, value in ph_ids.items():
            pattern = "\${(?:\d#)?%s.*?}" % re.escape(ph_id)
            for ph_text in re.findall(pattern, text):
                text = text.replace(ph_text, value)
        return text

    def replace_placeholders(self, ph_ids: dict):
        self.prompt = self._replace_placeholders(self.prompt, ph_ids)
        self.memory = self._replace_placeholders(self.memory, ph_ids)
        self.authors_note = self._replace_placeholders(self.authors_note, ph_ids)

        for i in range(len(self.world_infos)):
            for key in ["content", "comment"]:
                self.world_infos[i][key] = self._replace_placeholders(self.world_infos[i][key])

    def from_club(self, club_id):
        from importers import aetherroom
        import_data: aetherroom.ImportData
        try:
            import_data = aetherroom.import_scenario(club_id)
        except aetherroom.RequestFailed as err:
            status = err.status_code
            print(f"[import] Got {status} on request to club :^(")
            message = f"Club responded with {status}"
            if status == 404:
                message = f"Prompt not found for ID {club_id}"
            show_error_notification("Error loading prompt", message)
            return

        self.prompt = import_data.prompt
        self.memory = import_data.memory
        self.authors_note = import_data.authors_note
        self.notes = import_data.notes
        self.title = import_data.title
        self.world_infos = import_data.world_infos

        placeholders = self.extract_placeholders(self.prompt)
        if not placeholders:
            self.commit()
        else:
            self.request_client_configuration(placeholders)

    def commit(self):
        # Push buffer story to actual story
        exitModes()

        koboldai_vars.create_story("")
        koboldai_vars.gamestarted = True
        koboldai_vars.prompt = self.prompt
        koboldai_vars.memory = self.memory or ""
        koboldai_vars.authornote = self.authors_note or ""
        koboldai_vars.notes = self.notes
        koboldai_vars.story_name = self.title

        for wi in self.world_infos:
            koboldai_vars.worldinfo_v2.add_item(
                wi["key_list"][0],
                wi["key_list"],
                wi.get("keysecondary", []), 
                wi.get("folder", "root"),
                wi.get("constant", False), 
                wi["content"],
                wi.get("comment", "")
            )

        # Reset current save
        koboldai_vars.savedir = getcwd()+"\\stories"
        
        # Refresh game screen
        koboldai_vars.laststory = None
        setgamesaved(False)
        sendwi()
        refresh_story()

import_buffer = ImportBuffer()

# Set logging level to reduce chatter from Flask
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
def UI_2_logger(message):
    conv = Ansi2HTMLConverter(inline=True, dark_bg=True)
    data = json.loads(message)
    data['html'] = [conv.convert(text, full=False) for text in data['text'].split("\n")] 
    if not has_request_context():
        if koboldai_settings.queue is not None:
            koboldai_settings.queue.put(["log_message", data, {"broadcast":True, "room":"UI_2"}])
    else:
        socketio.emit("log_message", data, broadcast=True, room="UI_2")

web_log_history = []
def UI_2_log_history(message):
    conv = Ansi2HTMLConverter(inline=True, dark_bg=True)
    data = json.loads(message)
    data['html'] = [conv.convert(text, full=False) for text in data['text'].split("\n")] 
    if len(web_log_history) >= 100:
        del web_log_history[0]
    web_log_history.append(data)

from flask import Flask, render_template, Response, request, copy_current_request_context, send_from_directory, session, jsonify, abort, redirect, has_request_context, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_socketio import emit as _emit
from flask_session import Session
from flask_compress import Compress
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, NotFound, InternalServerError
import secrets
app = Flask(__name__, root_path=os.getcwd())
app.secret_key = secrets.token_hex()
app.config['SESSION_TYPE'] = 'filesystem'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Hack for socket stuff that needs app context
utils.flask_app = app

Compress(app)
socketio = SocketIO(app, async_method="eventlet", manage_session=False, cors_allowed_origins='*', max_http_buffer_size=10_000_000)
#socketio = SocketIO(app, async_method="eventlet", manage_session=False, cors_allowed_origins='*', max_http_buffer_size=10_000_000, logger=logger, engineio_logger=True)
logger.add(UI_2_log_history, serialize=True, colorize=True, enqueue=True, level="INFO")

#logger.add("log_file_1.log", rotation="500 MB")    # Automatically rotate too big file
koboldai_vars = koboldai_settings.koboldai_vars(socketio)
koboldai_settings.koboldai_vars_main = koboldai_vars
utils.koboldai_vars = koboldai_vars
utils.socketio = socketio

# Weird import position to steal koboldai_vars from utils
from modeling.patches import patch_transformers

#Load all of the model importers
import importlib
model_backend_code = {}
model_backends = {}
model_backend_module_names = {}
model_backend_type_crosswalk = {}

PRIORITIZED_BACKEND_MODULES = {
    "gptq_hf_torch": 2,
    "generic_hf_torch": 1
}

for module in os.listdir("./modeling/inference_models"):
    if module == '__pycache__':
        continue

    module_path = os.path.join("modeling/inference_models", module)
    if not os.path.isdir(module_path):
        # Drop-in modules must be folders
        continue

    if os.listdir(module_path) == ["__pycache__"]:
        # Delete backends which have been deleted upstream. As __pycache__
        # folders aren't tracked, they'll stick around until we zap em'
        assert len(os.listdir(module_path)) == 1
        logger.info(f"Deleting old backend {module}")
        shutil.rmtree(module_path)
        continue

    try:
        backend_code = importlib.import_module('modeling.inference_models.{}.class'.format(module))
        backend_name = backend_code.model_backend_name
        backend_type = backend_code.model_backend_type
        backend_object = backend_code.model_backend()

        if "disable" in vars(backend_object) and backend_object.disable:
            continue

        model_backends[backend_name] = backend_object
        model_backend_code[module] = backend_code

        if backend_name in model_backend_module_names:
            raise RuntimeError(f"{module} cannot make backend '{backend_name}'; it already exists!")
        model_backend_module_names[backend_name] = module

        if backend_type in model_backend_type_crosswalk:
            model_backend_type_crosswalk[backend_type].append(backend_name)
            model_backend_type_crosswalk[backend_type] = list(sorted(
                model_backend_type_crosswalk[backend_type],
                key=lambda name: PRIORITIZED_BACKEND_MODULES.get(
                    [mod for b_name, mod in model_backend_module_names.items() if b_name == name][0],
                    0
                ),
                reverse=True
            ))
        else:
            model_backend_type_crosswalk[backend_type] = [backend_name]

    except Exception:
        logger.error("Model Backend {} failed to load".format(module))
        logger.error(traceback.format_exc())

logger.info("We loaded the following model backends: \n{}".format("\n".join([x for x in model_backends])))
        

old_socketio_on = socketio.on
def new_socketio_on(*a, **k):
    decorator = old_socketio_on(*a, **k)
    def new_decorator(f):
        @functools.wraps(f)
        def g(*a, **k):
            if args.no_ui:
                return
            return f(*a, **k)
        return decorator(g)
    return new_decorator
socketio.on = new_socketio_on

def emit(*args, **kwargs):
    if has_request_context():
        try:
            return _emit(*args, **kwargs)
        except AttributeError:
            return socketio.emit(*args, **kwargs)
    else: #We're trying to send data outside of the http context. This won't work. Try the relay
        if koboldai_settings.queue is not None:
            koboldai_settings.queue.put([args[0], args[1], kwargs])
utils.emit = emit

#replacement for tpool.execute to maintain request contexts
def replacement_tpool_execute(function, *args, **kwargs):
    temp = {}
    socketio.start_background_task(tpool.execute_2, function, temp, *args, **kwargs).join()
    print(temp)
    return temp[1]
    
def replacement_tpool_execute_2(function, temp, *args, **kwargs):
    temp[1] = function(*args, **kwargs)

# marshmallow/apispec setup
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec.ext.marshmallow.field_converter import make_min_max_attributes
from apispec_webframeworks.flask import FlaskPlugin
from marshmallow import Schema, fields, validate, EXCLUDE
from marshmallow.exceptions import ValidationError

class KoboldSchema(Schema):
    pass

def new_make_min_max_attributes(validators, min_attr, max_attr) -> dict:
    # Patched apispec function that creates "exclusiveMinimum"/"exclusiveMaximum" OpenAPI attributes insteaed of "minimum"/"maximum" when using validators.Range or validators.Length with min_inclusive=False or max_inclusive=False
    attributes = {}
    min_list = [validator.min for validator in validators if validator.min is not None]
    max_list = [validator.max for validator in validators if validator.max is not None]
    min_inclusive_list = [getattr(validator, "min_inclusive", True) for validator in validators if validator.min is not None]
    max_inclusive_list = [getattr(validator, "max_inclusive", True) for validator in validators if validator.max is not None]
    if min_list:
        if min_attr == "minimum" and not min_inclusive_list[max(range(len(min_list)), key=min_list.__getitem__)]:
            min_attr = "exclusiveMinimum"
        attributes[min_attr] = max(min_list)
    if max_list:
        if min_attr == "maximum" and not max_inclusive_list[min(range(len(max_list)), key=max_list.__getitem__)]:
            min_attr = "exclusiveMaximum"
        attributes[max_attr] = min(max_list)
    return attributes
make_min_max_attributes.__code__ = new_make_min_max_attributes.__code__

def api_format_docstring(f):
    f.__doc__ = eval('f"""{}"""'.format(f.__doc__.replace("\\", "\\\\")))
    return f

def api_catch_out_of_memory_errors(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            if any (s in traceback.format_exc().lower() for s in ("out of memory", "not enough memory")):
                for line in reversed(traceback.format_exc().split("\n")):
                    if any(s in line.lower() for s in ("out of memory", "not enough memory")) and line.count(":"):
                        line = line.split(":", 1)[1]
                        line = re.sub(r"\[.+?\] +data\.", "", line).strip()
                        raise KoboldOutOfMemoryError("KoboldAI ran out of memory: " + line, type="out_of_memory.gpu.cuda" if "cuda out of memory" in line.lower() else "out_of_memory.gpu.hip" if "hip out of memory" in line.lower() else "out_of_memory.tpu.hbm" if "memory space hbm" in line.lower() else "out_of_memory.cpu.default_memory_allocator" if "defaultmemoryallocator" in line.lower() else "out_of_memory.unknown.unknown")
                raise KoboldOutOfMemoryError(type="out_of_memory.unknown.unknown")
            raise e
    return decorated

def api_schema_wrap(f):
    try:
        input_schema: Type[Schema] = next(iter(inspect.signature(f).parameters.values())).annotation
    except:
        HAS_SCHEMA = False
    else:
        HAS_SCHEMA = inspect.isclass(input_schema) and issubclass(input_schema, Schema)
    f = api_format_docstring(f)
    f = api_catch_out_of_memory_errors(f)
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if HAS_SCHEMA:
            body = request.get_json()
            schema = input_schema.from_dict(input_schema().load(body))
            response = f(schema, *args, **kwargs)
        else:
            response = f(*args, **kwargs)
        if not isinstance(response, Response):
            response = jsonify(response)
        return response
    return decorated

@app.errorhandler(HTTPException)
def handler(e):
    if request.path != "/api" and not request.path.startswith("/api/"):
        return e
    resp = jsonify(detail={"msg": str(e), "type": "generic.error_" + str(e.code)})
    if e.code == 405 and e.valid_methods is not None:
        resp.headers["Allow"] = ", ".join(e.valid_methods)
    return resp, e.code

class KoboldOutOfMemoryError(HTTPException):
    code = 507
    description = "KoboldAI ran out of memory."
    type = "out_of_memory.unknown.unknown"
    def __init__(self, *args, type=None, **kwargs):
        super().__init__(*args, **kwargs)
        if type is not None:
            self.type = type
@app.errorhandler(KoboldOutOfMemoryError)
def handler(e):
    if request.path != "/api" and not request.path.startswith("/api/"):
        return InternalServerError()
    return jsonify(detail={"type": e.type, "msg": e.description}), e.code

@app.errorhandler(ValidationError)
def handler(e):
    if request.path != "/api" and not request.path.startswith("/api/"):
        return InternalServerError()
    return jsonify(detail=e.messages), 422

@app.errorhandler(NotImplementedError)
def handler(e):
    if request.path != "/api" and not request.path.startswith("/api/"):
        return InternalServerError()
    return jsonify(detail={"type": "not_implemented", "msg": str(e).strip()}), 501

api_versions: List[str] = []

class KoboldAPISpec(APISpec):
    class KoboldFlaskPlugin(FlaskPlugin):
        def __init__(self, api: "KoboldAPISpec", *args, **kwargs):
            self._kobold_api_spec = api
            super().__init__(*args, **kwargs)

        def path_helper(self, *args, **kwargs):
            return super().path_helper(*args, **kwargs)[len(self._kobold_api_spec._prefixes[0]):]

    def __init__(self, *args, title: str = "KoboldAI API", openapi_version: str = "3.0.3", version: str = "1.0.0", prefixes: List[str] = None, **kwargs):
        plugins = [KoboldAPISpec.KoboldFlaskPlugin(self), MarshmallowPlugin()]
        self._prefixes = prefixes if prefixes is not None else [""]
        self._kobold_api_spec_version = version
        api_versions.append(version)
        api_versions.sort(key=lambda x: [int(e) for e in x.split(".")])
        super().__init__(*args, title=title, openapi_version=openapi_version, version=version, plugins=plugins, servers=[{"url": self._prefixes[0]}], **kwargs)
        for prefix in self._prefixes:
            app.route(prefix, endpoint="~KoboldAPISpec~" + prefix)(lambda: redirect(request.path + "/docs/"))
            app.route(prefix + "/", endpoint="~KoboldAPISpec~" + prefix + "/")(lambda: redirect("docs/"))
            app.route(prefix + "/docs", endpoint="~KoboldAPISpec~" + prefix + "/docs")(lambda: redirect("docs/"))
            app.route(prefix + "/docs/", endpoint="~KoboldAPISpec~" + prefix + "/docs/")(lambda: render_template("swagger-ui.html", url=self._prefixes[0] + "/openapi.json"))
            app.route(prefix + "/openapi.json", endpoint="~KoboldAPISpec~" + prefix + "/openapi.json")(lambda: jsonify(self.to_dict()))

    def route(self, rule: str, methods=["GET"], **kwargs):
        __F = TypeVar("__F", bound=Callable[..., Any])
        if "strict_slashes" not in kwargs:
            kwargs["strict_slashes"] = False
        def new_decorator(f: __F) -> __F:
            @functools.wraps(f)
            def g(*args, **kwargs):
                global api_version
                api_version = self._kobold_api_spec_version
                try:
                    return f(*args, **kwargs)
                finally:
                    api_version = None
            for prefix in self._prefixes:
                g = app.route(prefix + rule, methods=methods, **kwargs)(g)
            with app.test_request_context():
                self.path(view=g, **kwargs)
            return g
        return new_decorator

    def get(self, rule: str, **kwargs):
        return self.route(rule, methods=["GET"], **kwargs)
    
    def post(self, rule: str, **kwargs):
        return self.route(rule, methods=["POST"], **kwargs)
    
    def put(self, rule: str, **kwargs):
        return self.route(rule, methods=["PUT"], **kwargs)
    
    def patch(self, rule: str, **kwargs):
        return self.route(rule, methods=["PATCH"], **kwargs)
    
    def delete(self, rule: str, **kwargs):
        return self.route(rule, methods=["DELETE"], **kwargs)

tags = [
    {"name": "info", "description": "Metadata about this API"},
    {"name": "generate", "description": "Text generation endpoints"},
    {"name": "model", "description": "Information about the current text generation model"},
    {"name": "story", "description": "Endpoints for managing the story in the KoboldAI GUI"},
    {"name": "world_info", "description": "Endpoints for managing the world info in the KoboldAI GUI"},
    {"name": "config", "description": "Allows you to get/set various setting values"},
]

api_version = None  # This gets set automatically so don't change this value

api_v1 = KoboldAPISpec(
    version="1.2.5",
    prefixes=["/api/v1", "/api/latest"],
    tags=tags,
)

def show_error_notification(title: str, text: str, do_log: bool = False) -> None:
    if do_log:
        logger.error(f"{title}: {text}")

    if has_request_context():
        socketio.emit("show_error_notification", {"title": title, "text": text}, broadcast=True, room="UI_2")
    else:
        koboldai_settings.queue.put(["show_error_notification", {"title": title, "text": text}, {"broadcast":True, "room":'UI_2'}])

# Returns the expected config filename for the current setup.
# If the model_name is specified, it returns what the settings file would be for that model
def get_config_filename(model_name = None):
    if model_name:
        return(f"settings/{model_name.replace('/', '_')}.settings")
    elif args.configname:
        return(f"settings/{args.configname.replace('/', '_')}.settings")
    elif koboldai_vars.configname != '':
        return(f"settings/{koboldai_vars.configname.replace('/', '_')}.settings")
    else:
        logger.warning(f"Empty configfile name sent back. Defaulting to ReadOnly")
        return(f"settings/ReadOnly.settings")

#==================================================================#
# Function to get model selection at startup
#==================================================================#
def sendModelSelection(menu="mainmenu", folder="./models"):
    #If we send one of the manual load options, send back the list of model directories, otherwise send the menu
    if menu in ('NeoCustom', 'GPT2Custom'):
        paths, breadcrumbs = get_folder_path_info(folder)
        # paths = [x for x in paths if "rwkv" not in x[1].lower()]
        if koboldai_vars.host:
            breadcrumbs = []

        menu_list = [[folder, menu, "", False] for folder in paths]
        menu_list_ui_2 = [[folder[0], folder[1], "", False] for folder in paths]
        menu_list.append(["Return to Main Menu", "mainmenu", "", True])
        menu_list_ui_2.append(["Return to Main Menu", "mainmenu", "", True])

        if os.path.abspath("{}/models".format(os.getcwd())) == os.path.abspath(folder):
            showdelete=True
        else:
            showdelete=False
        emit('from_server', {'cmd': 'show_model_menu', 'data': menu_list, 'menu': menu, 'breadcrumbs': breadcrumbs, "showdelete": showdelete}, broadcast=True, room="UI_1")

        p_menu = [{
            "label": m[0],
            "name": m[1],
            "size": m[2],
            "isMenu": m[3],
            "isDownloaded": True,
        } for m in menu_list_ui_2]
        emit('show_model_menu', {'data': p_menu, 'menu': menu, 'breadcrumbs': breadcrumbs, "showdelete": showdelete}, broadcast=False)
    elif menu == "customhuggingface":
        p_menu = [{
            "label": "Return to Main Menu",
            "name": "mainmenu",
            "size": "",
            "isMenu": True,
            "isDownloaded": True,
        }]
        breadcrumbs = []
        showdelete=False
        emit('from_server', {'cmd': 'show_model_menu', 'data': [["Return to Main Menu", "mainmenu", "", True]], 'menu': menu, 'breadcrumbs': breadcrumbs, "showdelete": showdelete}, broadcast=True, room="UI_1")
        emit('show_model_menu', {'data': p_menu, 'menu': menu, 'breadcrumbs': breadcrumbs, "showdelete": showdelete}, broadcast=False)
    else:
        filtered_menu = [item for item in model_menu[menu] if item.should_show()]

        emit(
            "from_server",
            {
                "cmd": "show_model_menu",
                "data": [item.to_ui1() for item in filtered_menu],
                "menu": menu,
                "breadcrumbs": [],
                "showdelete": False
            },
            broadcast=True,
            room="UI_1"
        )

        emit(
            "show_model_menu",
            {
                "data": [item.to_json() for item in filtered_menu],
                "menu": menu,
                "breadcrumbs": [],
                "showdelete": False
            },
            broadcast=False
        )

def get_folder_path_info(base):
    if base is None:
        return [], []
    if base == 'This PC':
        breadcrumbs = [['This PC', 'This PC']]
        paths = [["{}:\\".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]
    else:
        path = os.path.abspath(base)
        if path[-1] == "\\":
            path = path[:-1]
        breadcrumbs = []
        for i in range(len(path.replace("/", "\\").split("\\"))):
            breadcrumbs.append(["\\".join(path.replace("/", "\\").split("\\")[:i+1]),
                                 path.replace("/", "\\").split("\\")[i]])
        if len(breadcrumbs) == 1:
            breadcrumbs = [["{}:\\".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]
        else:
            if len([["{}:\\".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]) > 0:
                breadcrumbs.insert(0, ['This PC', 'This PC'])
        paths = []
        base_path = os.path.abspath(base)
        for item in os.listdir(base_path):
            if os.path.isdir(os.path.join(base_path, item)):
                paths.append([os.path.join(base_path, item), item])
    # Paths/breadcrumbs is a list of lists, where the first element in the sublist is the full path and the second is the folder name
    return (paths, breadcrumbs)


def getModelSelection(modellist):
    print("    #    Model\t\t\t\t\t\tVRAM\n    ========================================================")
    i = 1
    for m in modellist:
        print("    {0} - {1}\t\t\t{2}".format("{:<2}".format(i), m[0].ljust(25), m[2]))
        i += 1
    print(" ");
    modelsel = 0
    koboldai_vars.model = ''
    while(koboldai_vars.model == ''):
        modelsel = input("Model #> ")
        if(modelsel.isnumeric() and int(modelsel) > 0 and int(modelsel) <= len(modellist)):
            koboldai_vars.model = modellist[int(modelsel)-1][1]
        else:
            print("{0}Please enter a valid selection.{1}".format(colors.RED, colors.END))
    
    # Model Lists
    try:
        getModelSelection(eval(koboldai_vars.model))
    except Exception as e:
        if(koboldai_vars.model == "Return"):
            getModelSelection(mainmenu)
                
        # If custom model was selected, get the filesystem location and store it
        if(koboldai_vars.model == "NeoCustom" or koboldai_vars.model == "GPT2Custom"):
            print("{0}Please choose the folder where pytorch_model.bin is located:{1}\n".format(colors.CYAN, colors.END))
            modpath = fileops.getdirpath(getcwd() + "/models", "Select Model Folder")
        
            if(modpath):
                # Save directory to koboldai_vars
                koboldai_vars.custmodpth = modpath
            else:
                # Print error and retry model selection
                print("{0}Model select cancelled!{1}".format(colors.RED, colors.END))
                print("{0}Select an AI model to continue:{1}\n".format(colors.CYAN, colors.END))
                getModelSelection(mainmenu)

def check_if_dir_is_model(path):
    return os.path.exists(os.path.join(path, 'config.json'))
    
#==================================================================#
# Return all keys in tokenizer dictionary containing char
#==================================================================#
#def gettokenids(char):
#    keys = []
#    for key in vocab_keys:
#        if(key.find(char) != -1):
#            keys.append(key)
#    return keys

#==================================================================#
# Return Model Name
#==================================================================#
def getmodelname():
    if(koboldai_vars.online_model != ''):
        return(f"{koboldai_vars.model}/{koboldai_vars.online_model}")
    if(koboldai_vars.model in ("NeoCustom", "GPT2Custom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        modelname = os.path.basename(os.path.normpath(model.path))
        return modelname
    else:
        modelname = koboldai_vars.model if koboldai_vars.model is not None else "Read Only"
        return modelname

#==================================================================#
# Get hidden size from model
#==================================================================#
def get_hidden_size_from_model(model):
    return model.get_input_embeddings().embedding_dim



#==================================================================#
#  Allow the models to override some settings
#==================================================================#
def loadmodelsettings():
    try:
        js   = json.loads(str(model.model_config).partition(' ')[2])
    except Exception as e:
        try:
            try:
                js   = json.load(open(koboldai_vars.custmodpth + "/config.json", "r"))
            except Exception as e:
                js   = json.load(open(koboldai_vars.custmodpth.replace('/', '_') + "/config.json", "r"))
        except Exception as e:
            js   = {}
    koboldai_vars.default_preset = koboldai_settings.default_preset
    if koboldai_vars.model_type == "xglm" or js.get("compat", "j") == "fairseq_lm":
        koboldai_vars.newlinemode = "s"  # Default to </s> newline mode if using XGLM
    if koboldai_vars.model_type == "opt" or koboldai_vars.model_type == "bloom":
        koboldai_vars.newlinemode = "ns"  # Handle </s> but don't convert newlines if using Fairseq models that have newlines trained in them
    koboldai_vars.modelconfig = js
    if("badwordsids" in js):
        koboldai_vars.badwordsids = js["badwordsids"]
    if("nobreakmodel" in js):
        koboldai_vars.nobreakmodel = js["nobreakmodel"]
    if("sampler_order" in js):
        sampler_order = js["sampler_order"]
        if(len(sampler_order) < 7):
            sampler_order = [6] + sampler_order
        koboldai_vars.sampler_order = sampler_order
    if("temp" in js):
        koboldai_vars.temp       = js["temp"]
        koboldai_vars.default_preset['temp'] = js["temp"]
    if("top_p" in js):
        koboldai_vars.top_p      = js["top_p"]
        koboldai_vars.default_preset['top_p'] = js["top_p"]
    if("top_k" in js):
        koboldai_vars.top_k      = js["top_k"]
        koboldai_vars.default_preset['top_k'] = js["top_k"]
    if("tfs" in js):
        koboldai_vars.tfs        = js["tfs"]
        koboldai_vars.default_preset['tfs'] = js["tfs"]
    if("typical" in js):
        koboldai_vars.typical    = js["typical"]
        koboldai_vars.default_preset['typical'] = js["typical"]
    if("top_a" in js):
        koboldai_vars.top_a      = js["top_a"]
        koboldai_vars.default_preset['top_a'] = js["top_a"]
    if("rep_pen" in js):
        koboldai_vars.rep_pen    = js["rep_pen"]
        koboldai_vars.default_preset['rep_pen'] = js["rep_pen"]
    if("rep_pen_slope" in js):
        koboldai_vars.rep_pen_slope = js["rep_pen_slope"]
        koboldai_vars.default_preset['rep_pen_slope'] = js["rep_pen_slope"]
    if("rep_pen_range" in js):
        koboldai_vars.rep_pen_range = js["rep_pen_range"]
        koboldai_vars.default_preset['rep_pen_range'] = js["rep_pen_range"]
    if("adventure" in js):
        koboldai_vars.adventure = js["adventure"]
    if("chatmode" in js):
        koboldai_vars.chatmode = js["chatmode"]
    if("dynamicscan" in js):
        koboldai_vars.dynamicscan = js["dynamicscan"]
    if("formatoptns" in js):
        for setting in ['frmttriminc', 'frmtrmblln', 'frmtrmspch', 'frmtadsnsp', 'singleline']:
            if setting in js["formatoptns"]:
                setattr(koboldai_vars, setting, js["formatoptns"][setting])
    if("welcome" in js):
        koboldai_vars.welcome = kml(js["welcome"]) if js["welcome"] != False else koboldai_vars.welcome_default
    if("newlinemode" in js):
        koboldai_vars.newlinemode = js["newlinemode"]
    if("antemplate" in js):
        koboldai_vars.setauthornotetemplate = js["antemplate"]
        if(not koboldai_vars.gamestarted):
            koboldai_vars.authornotetemplate = koboldai_vars.setauthornotetemplate

#==================================================================#
#  Take settings from koboldai_vars and write them to client settings file
#==================================================================#
def savesettings():
     # Build json to write
    for setting in ['model_settings', 'user_settings', 'system_settings']:
        if setting == "model_settings":
            filename = "settings/{}.v2_settings".format(koboldai_vars.model.replace("/", "_"))
        else:
            filename = "settings/{}.v2_settings".format(setting)
        with open(filename, "w") as settings_file:
            settings_file.write(getattr(koboldai_vars, "_{}".format(setting)).to_json())
    

#==================================================================#
#  Don't save settings unless 2 seconds have passed without modification
#==================================================================#
@debounce(2)
def settingschanged():
    logger.info("Saving settings.")
    savesettings()

#==================================================================#
#  Read settings from client file JSON and send to koboldai_vars
#==================================================================#

def loadsettings():
    if(path.exists("settings/" + getmodelname().replace('/', '_') + ".v2_settings")):
        with open("settings/" + getmodelname().replace('/', '_') + ".v2_settings", "r") as file:
            getattr(koboldai_vars, "_model_settings").from_json(file.read())
        
        
def processsettings(js):
# Copy file contents to vars
    if("apikey" in js):
        # If the model is the HORDE, then previously saved API key in settings
        # Will always override a new key set.
        if koboldai_vars.model != "CLUSTER" or koboldai_vars.apikey == '':
            koboldai_vars.apikey = js["apikey"]
    if("andepth" in js):
        koboldai_vars.andepth = js["andepth"]
    if("sampler_order" in js):
        sampler_order = js["sampler_order"]
        if(len(sampler_order) < 7):
            sampler_order = [6] + sampler_order
        koboldai_vars.sampler_order = sampler_order
    if("temp" in js):
        koboldai_vars.temp = js["temp"]
    if("top_p" in js):
        koboldai_vars.top_p = js["top_p"]
    if("top_k" in js):
        koboldai_vars.top_k = js["top_k"]
    if("tfs" in js):
        koboldai_vars.tfs = js["tfs"]
    if("typical" in js):
        koboldai_vars.typical = js["typical"]
    if("top_a" in js):
        koboldai_vars.top_a = js["top_a"]
    if("rep_pen" in js):
        koboldai_vars.rep_pen = js["rep_pen"]
    if("rep_pen_slope" in js):
        koboldai_vars.rep_pen_slope = js["rep_pen_slope"]
    if("rep_pen_range" in js):
        koboldai_vars.rep_pen_range = js["rep_pen_range"]
    if("genamt" in js):
        koboldai_vars.genamt = js["genamt"]
    if("max_length" in js):
        koboldai_vars.max_length = js["max_length"]
    if("ikgen" in js):
        koboldai_vars.ikgen = js["ikgen"]
    if("formatoptns" in js):
        koboldai_vars.formatoptns = js["formatoptns"]
    if("numseqs" in js):
        koboldai_vars.numseqs = js["numseqs"]
    if("widepth" in js):
        koboldai_vars.widepth = js["widepth"]
    if("useprompt" in js):
        koboldai_vars.useprompt = js["useprompt"]
    if("adventure" in js):
        koboldai_vars.adventure = js["adventure"]
    if("chatmode" in js):
        koboldai_vars.chatmode = js["chatmode"]
    if("chatname" in js):
        koboldai_vars.chatname = js["chatname"]
    if("botname" in js):
        koboldai_vars.botname = js["botname"]
    if("dynamicscan" in js):
        koboldai_vars.dynamicscan = js["dynamicscan"]
    if("nopromptgen" in js):
        koboldai_vars.nopromptgen = js["nopromptgen"]
    if("rngpersist" in js):
        koboldai_vars.rngpersist = js["rngpersist"]
    if("nogenmod" in js):
        koboldai_vars.nogenmod = js["nogenmod"]
    if("fulldeterminism" in js):
        koboldai_vars.full_determinism = js["fulldeterminism"]
    if("stop_sequence" in js):
        koboldai_vars.stop_sequence = js["stop_sequence"]
    if("autosave" in js):
        koboldai_vars.autosave = js["autosave"]
    if("newlinemode" in js):
        koboldai_vars.newlinemode = js["newlinemode"]
    if("welcome" in js):
        koboldai_vars.welcome = js["welcome"]
    if("output_streaming" in js):
        koboldai_vars.output_streaming = js["output_streaming"]
    if("show_probs" in js):
        koboldai_vars.show_probs = js["show_probs"]
    if("show_budget" in js):
        koboldai_vars.show_budget = js["show_budget"]
    
    if("seed" in js):
        koboldai_vars.seed = js["seed"]
        if(koboldai_vars.seed is not None):
            koboldai_vars.seed_specified = True
        else:
            koboldai_vars.seed_specified = False
    else:
        koboldai_vars.seed_specified = False

    if("antemplate" in js):
        koboldai_vars.setauthornotetemplate = js["antemplate"]
        if(not koboldai_vars.gamestarted):
            koboldai_vars.authornotetemplate = koboldai_vars.setauthornotetemplate
    
    if("userscripts" in js):
        koboldai_vars.userscripts = []
        for userscript in js["userscripts"]:
            if type(userscript) is not str:
                continue
            userscript = userscript.strip()
            if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                koboldai_vars.userscripts.append(userscript)

    if("corescript" in js and type(js["corescript"]) is str and all(q not in js["corescript"] for q in ("..", ":")) and all(js["corescript"][0] not in q for q in ("/", "\\"))):
        koboldai_vars.corescript = js["corescript"]
    else:
        koboldai_vars.corescript = "default.lua"

#==================================================================#
#  Load a soft prompt from a file
#==================================================================#

#def check_for_sp_change():
#    while(True):
#        time.sleep(0.05)
#
#        if(koboldai_vars.sp_changed):
#            with app.app_context():
#                emit('from_server', {'cmd': 'spstatitems', 'data': {koboldai_vars.spfilename: koboldai_vars.spmeta} if koboldai_vars.allowsp and len(koboldai_vars.spfilename) else {}}, namespace=None, broadcast=True, room="UI_1")
#            koboldai_vars.sp_changed = False


#socketio.start_background_task(check_for_sp_change)

def spRequest(filename):
    if(not koboldai_vars.allowsp):
        raise RuntimeError("Soft prompts are not supported by your current model/backend")
    
    old_filename = koboldai_vars.spfilename

    koboldai_vars.spfilename = ""
    settingschanged()

    if(len(filename) == 0):
        koboldai_vars.sp = None
        koboldai_vars.sp_length = 0
        if(old_filename != filename):
            koboldai_vars.sp_changed = True
        return

    z, version, shape, fortran_order, dtype = fileops.checksp("./softprompts/"+filename, koboldai_vars.modeldim)
    if not isinstance(z, zipfile.ZipFile):
        raise RuntimeError(f"{repr(filename)} is not a valid soft prompt file")
    with z.open('meta.json') as f:
        koboldai_vars.spmeta = json.load(f)
        koboldai_vars.spname = koboldai_vars.spmeta['name']
    z.close()

    with np.load(fileops.sppath(filename), allow_pickle=False) as f:
        tensor = f['tensor.npy']

    # If the tensor is in bfloat16 format, convert it to float32
    if(tensor.dtype == 'V2'):
        tensor.dtype = np.uint16
        tensor = np.uint32(tensor) << 16
        tensor.dtype = np.float32

    if(tensor.dtype != np.float16):
        tensor = np.float32(tensor)
    assert not np.isinf(tensor).any() and not np.isnan(tensor).any()

    koboldai_vars.sp_length = tensor.shape[-2]
    koboldai_vars.spmeta["n_tokens"] = koboldai_vars.sp_length

    if(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        rows = tensor.shape[0]
        padding_amount = tpu_mtj_backend.params["seq"] - (tpu_mtj_backend.params["seq"] % -tpu_mtj_backend.params["cores_per_replica"]) - rows
        tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
        tensor = tensor.reshape(
            tpu_mtj_backend.params["cores_per_replica"],
            -1,
            tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"]),
        )
        koboldai_vars.sp = tpu_mtj_backend.shard_xmap(np.float32(tensor))
    else:
        koboldai_vars.sp = torch.from_numpy(tensor)

    koboldai_vars.spfilename = filename
    settingschanged()
    if(old_filename != filename):
            koboldai_vars.sp_changed = True

#==================================================================#
# Startup
#==================================================================#
def general_startup(override_args=None):
    global args
    global enable_whitelist
    global allowed_ips
    import configparser
    #Figure out what git we're on if that's available
    config = configparser.ConfigParser()
    if os.path.exists('.git/config'):
        config.read('.git/config')
        koboldai_vars.git_repository = config['remote "origin"']['url']
        for item in config.sections():
            if "branch" in item:
                koboldai_vars.git_branch = item.replace("branch ", "").replace('"', '')
    
        logger.info("Running on Repo: {} Branch: {}".format(koboldai_vars.git_repository, koboldai_vars.git_branch))
    
    # Parsing Parameters
    parser = argparse.ArgumentParser(description="KoboldAI Server")
    parser.add_argument("--remote", action='store_true', help="Optimizes KoboldAI for Remote Play")
    parser.add_argument("--noaimenu", action='store_true', help="Disables the ability to select the AI")
    parser.add_argument("--ngrok", action='store_true', help="Optimizes KoboldAI for Remote Play using Ngrok")
    parser.add_argument("--localtunnel", action='store_true', help="Optimizes KoboldAI for Remote Play using Localtunnel")
    parser.add_argument("--host", type=str, default="Disabled", nargs="?", const="", help="Optimizes KoboldAI for LAN Remote Play without using a proxy service. --host opens to all LAN. Enable IP whitelisting by using a comma separated IP list. Supports individual IPs, ranges, and subnets --host 127.0.0.1,127.0.0.2,127.0.0.3,192.168.1.0-192.168.1.255,10.0.0.0/24,etc")
    parser.add_argument("--port", type=int, help="Specify the port on which the application will be joinable")
    parser.add_argument("--aria2_port", type=int, help="Specify the port on which aria2's RPC interface will be open if aria2 is installed (defaults to 6799)")
    parser.add_argument("--model", help="Specify the Model Type to skip the Menu")
    parser.add_argument("--model_backend", default="Huggingface", help="Specify the model backend you want to use")
    parser.add_argument("--model_parameters", action="store", default="", help="json of id values to use for the input to the model loading process (set to help to get required parameters)")
    parser.add_argument("--path", help="Specify the Path for local models (For model NeoCustom or GPT2Custom)")
    parser.add_argument("--apikey", help="Specify the API key to use for online services")
    parser.add_argument("--sh_apikey", help="Specify the API key to use for txt2img from the Stable Horde. Get a key from https://horde.koboldai.net/register")
    parser.add_argument("--req_model", type=str, action='append', required=False, help="Which models which we allow to generate for us during cluster mode. Can be specified multiple times.")
    parser.add_argument("--revision", help="Specify the model revision for huggingface models (can be a git branch/tag name or a git commit hash)")
    parser.add_argument("--cpu", action='store_true', help="By default unattended launches are on the GPU use this option to force CPU usage.")
    parser.add_argument("--override_delete", action='store_true', help="Deleting stories from inside the browser is disabled if you are using --remote and enabled otherwise. Using this option will instead allow deleting stories if using --remote and prevent deleting stories otherwise.")
    parser.add_argument("--override_rename", action='store_true', help="Renaming stories from inside the browser is disabled if you are using --remote and enabled otherwise. Using this option will instead allow renaming stories if using --remote and prevent renaming stories otherwise.")
    parser.add_argument("--configname", help="Force a fixed configuration name to aid with config management.")
    parser.add_argument("--colab", action='store_true', help="Optimize for Google Colab.")
    parser.add_argument("--nobreakmodel", action='store_true', help="Disables Breakmodel support completely.")
    parser.add_argument("--unblock", action='store_true', default=False, help="Unblocks the KoboldAI port to be accessible from other machines without optimizing for remote play (It is recommended to use --host instead)")
    parser.add_argument("--quiet", action='store_true', default=False, help="If present will suppress any story related text from showing on the console")
    parser.add_argument("--no_aria2", action='store_true', default=False, help="Prevents KoboldAI from using aria2 to download huggingface models more efficiently, in case aria2 is causing you issues")
    parser.add_argument("--lowmem", action='store_true', help="Extra Low Memory loading for the GPU, slower but memory does not peak to twice the usage")
    parser.add_argument("--savemodel", action='store_true', help="Saves the model to the models folder even if --colab is used (Allows you to save models to Google Drive)")
    parser.add_argument("--cacheonly", action='store_true', help="Does not save the model to the models folder when it has been downloaded in the cache")
    parser.add_argument("--customsettings", help="Preloads arguements from json file. You only need to provide the location of the json file. Use customsettings.json template file. It can be renamed if you wish so that you can store multiple configurations. Leave any settings you want as default as null. Any values you wish to set need to be in double quotation marks")
    parser.add_argument("--no_ui", action='store_true', default=False, help="Disables the GUI and Socket.IO server while leaving the API server running.")
    parser.add_argument("--summarizer_model", action='store', default="philschmid/bart-large-cnn-samsum", help="Huggingface model to use for summarization. Defaults to sshleifer/distilbart-cnn-12-6")
    parser.add_argument("--max_summary_length", action='store', default=75, help="Maximum size for summary to send to image generation")
    parser.add_argument("--multi_story", action='store_true', default=False, help="Allow multi-story mode (experimental)")
    parser.add_argument("--peft", type=str, help="Specify the path or HuggingFace ID of a Peft to load it. Not supported on TPU. (Experimental)") 
    parser.add_argument('-f', action='store', help="option for compatability with colab memory profiles")
    parser.add_argument('-v', '--verbosity', action='count', default=0, help="The default logging level is ERROR or higher. This value increases the amount of logging seen in your screen")
    parser.add_argument('-q', '--quiesce', action='count', default=0, help="The default logging level is ERROR or higher. This value decreases the amount of logging seen in your screen")
    parser.add_argument("--panic", action='store_true', help="Disables falling back when loading fails.")

    #args: argparse.Namespace = None
    if "pytest" in sys.modules and override_args is None:
        args = parser.parse_args([])
        return
    if override_args is not None:
        import shlex
        args = parser.parse_args(shlex.split(override_args))
    elif(os.environ.get("KOBOLDAI_ARGS") is not None):
        import shlex
        logger.info("Using environmental variables instead of command arguments: {}".format(os.environ["KOBOLDAI_ARGS"]))
        args = parser.parse_args(shlex.split(os.environ["KOBOLDAI_ARGS"]))
    else:
        args = parser.parse_args()
    
    utils.args = args





    
    #load system and user settings
    for setting in ['user_settings', 'system_settings']:
        if os.path.exists("settings/{}.v2_settings".format(setting)):
            with open("settings/{}.v2_settings".format(setting), "r") as settings_file:
                getattr(koboldai_vars, "_{}".format(setting)).from_json(settings_file.read())
    
    
    temp = [x for x in vars(args)]
    for arg in temp:
        if arg == "path":
            if "model_path" in os.environ:
                logger.info("Setting model path based on enviornmental variable: {}".format(os.environ["model_path"]))
                setattr(args, arg, os.environ["model_path"])
        else:
            if arg in os.environ:
                logger.info("Setting {} based on enviornmental variable: {}".format(arg, os.environ[arg]))
                if isinstance(getattr(args, arg), bool):
                    if os.environ[arg].lower() == "true":
                        setattr(args, arg, True)
                    else:
                        setattr(args, arg, False)
                else:
                    setattr(args, arg, os.environ[arg])
    set_logger_verbosity(args.verbosity)
    quiesce_logger(args.quiesce)
    if args.customsettings:
        f = open (args.customsettings)
        importedsettings = json.load(f)
        for items in importedsettings:
            if importedsettings[items] is not None:
                setattr(args, items, importedsettings[items])            
        f.close()
    
    if args.no_ui:
        def new_emit(*args, **kwargs):
            return
        old_emit = socketio.emit
        socketio.emit = new_emit

    args.max_summary_length = int(args.max_summary_length)

    koboldai_vars.revision = args.revision
    koboldai_settings.multi_story = args.multi_story

    if args.apikey:
        koboldai_vars.apikey = args.apikey
        koboldai_vars.horde_api_key = args.apikey
    if args.sh_apikey:
        koboldai_vars.horde_api_key = args.sh_apikey
    if args.req_model:
        koboldai_vars.cluster_requested_models = args.req_model

    if args.colab:
        args.remote = True;
        args.override_rename = True;
        args.override_delete = True;
        args.quiet = True;
        args.lowmem = True;
        args.noaimenu = True;
        koboldai_vars.colab_arg = True;

    if args.quiet:
        koboldai_vars.quiet = True

    if args.nobreakmodel:
        for model_backend in model_backends:
            model_backends[model_backend].nobreakmodel = True

    if args.remote:
        koboldai_vars.host = True;

    if args.ngrok:
        koboldai_vars.host = True;

    if args.localtunnel:
        koboldai_vars.host = True;

    if args.lowmem:
        model_backends['Huggingface'].low_mem = True

    if args.host != "Disabled":
            # This means --host option was submitted without an argument
            # Enable all LAN IPs (0.0.0.0/0)
        koboldai_vars.host = True
        args.unblock = True
        if args.host != "":
            # Check if --host option was submitted with an argument
            # Parse the supplied IP(s) and add them to the allowed IPs list
            enable_whitelist = True
            for ip_str in args.host.split(","):
                if "/" in ip_str:
                    allowed_ips |= set(str(ip) for ip in ipaddress.IPv4Network(ip_str, strict=False).hosts())
                elif "-" in ip_str:
                    start_ip, end_ip = ip_str.split("-")
                    start_ip_int = int(ipaddress.IPv4Address(start_ip))
                    end_ip_int = int(ipaddress.IPv4Address(end_ip))
                    allowed_ips |= set(str(ipaddress.IPv4Address(ip)) for ip in range(start_ip_int, end_ip_int + 1))
                else:
                    allowed_ips.add(ip_str.strip())
            # Sort and print the allowed IPs list
            allowed_ips = sorted(allowed_ips, key=lambda ip: int(''.join([i.zfill(3) for i in ip.split('.')])))
            print(f"Allowed IPs: {allowed_ips}")

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "None"
        koboldai_vars.use_colab_tpu = False
        koboldai_vars.hascuda = False
        koboldai_vars.usegpu = False
        model_backends['Huggingface'].nobreakmodel = True

    koboldai_vars.smandelete = koboldai_vars.host == args.override_delete
    koboldai_vars.smanrename = koboldai_vars.host == args.override_rename

    koboldai_vars.aria2_port = args.aria2_port or 6799
    
    #Now let's look to see if we are going to force a load of a model from a user selected folder
    if(koboldai_vars.model == "selectfolder"):
        print("{0}Please choose the folder where pytorch_model.bin is located:{1}\n".format(colors.CYAN, colors.END))
        modpath = fileops.getdirpath(getcwd() + "/models", "Select Model Folder")
    
        if(modpath):
            # Save directory to koboldai_vars
            koboldai_vars.model = "NeoCustom"
            args.path = modpath
    elif args.model:
        logger.message(f"Welcome to KoboldAI!")
        logger.message(f"You have selected the following Model: {args.model}")
        if args.path:
            logger.message(f"You have selected the following path for your Model: {args.path}")
            model_backends["KoboldAI Old Colab Method"].colaburl = args.path + "/request"; # Lets just use the same parameter to keep it simple
            
    #setup socketio relay queue
    koboldai_settings.queue = multiprocessing.Queue()
    
    socketio.start_background_task(socket_io_relay, koboldai_settings.queue, socketio)
    
    if koboldai_vars.use_colab_tpu and args.model_backend == "Huggingface":
         args.model_backend = "Huggingface MTJ"
         
    
    if args.model:
        # At this point we have to try to load the model through the selected backend
        if args.model_backend not in model_backends:
            logger.error("Your selected model backend ({}) isn't in the model backends we know about ({})".format(args.model_backend, ", ".join([x for x in model_backends])))
            exit()
        #OK, we've been given a model to load and a backend to load it through. Now we need to get a list of parameters and make sure we get what we need to actually load it
        parameters = model_backends[args.model_backend].get_requested_parameters(args.model, args.path, "")
        ok_to_load = True
        mising_parameters = []
        arg_parameters = json.loads(args.model_parameters.replace("'", "\"")) if args.model_parameters != "" and args.model_parameters.lower() != "help" else {}
        
        #If we're on colab we'll set everything to GPU0
        if args.colab and args.model_backend == 'Huggingface' and koboldai_vars.on_colab:
            arg_parameters['use_gpu'] = True
        
        
        for parameter in parameters:
            if parameter['uitype'] != "Valid Display":
                if parameter['default'] == "" and parameter['id'] not in arg_parameters:
                    mising_parameters.append(parameter['id'])
                    ok_to_load = False
                elif parameter['id'] not in arg_parameters:
                    arg_parameters[parameter['id']] = parameter['default']
        if not ok_to_load:
            logger.error("Your selected backend needs additional parameters to run. Please pass through the parameters as a json like \"{'[ID]': '[Value]', '[ID2]': '[Value]'}\" using --model_parameters (required parameters shown below)")
            logger.error("Parameters (ID: Default Value (Help Text)): {}".format("\n".join(["{}: {} ({})".format(x['id'],x['default'],x['tooltip']) for x in parameters if x['uitype'] != "Valid Display"])))
            logger.error("Missing: {}".format(", ".join(mising_parameters)))
            exit()
        if args.model_parameters.lower() == "help":
            logger.error("Please pass through the parameters as a json like \"{'[ID]': '[Value]', '[ID2]': '[Value]'}\" using --model_parameters (required parameters shown below)")
            logger.error("Parameters (ID: Default Value (Help Text)): {}".format("\n".join(["{}: {} ({})".format(x['id'],x['default'],x['tooltip']) for x in parameters if x['uitype'] != "Valid Display"])))
            exit()
        arg_parameters['id'] = args.model
        arg_parameters['model'] = args.model
        arg_parameters['path'] = args.path
        arg_parameters['menu_path'] = ""
        model_backends[args.model_backend].set_input_parameters(arg_parameters)
        koboldai_vars.model = args.model
        return args.model_backend
    else:
        return "Read Only"
        
    
        
    
def unload_model():
    global model
    global generator
    global model_config
    global tokenizer
    
    #We need to wipe out the existing model and refresh the cuda cache
    model = None
    generator = None
    model_config = None
    koboldai_vars.online_model = ''
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
            for tensor in gc.get_objects():
                try:
                    if torch.is_tensor(tensor):
                        tensor.set_(torch.tensor((), device=tensor.device, dtype=tensor.dtype))
                except:
                    pass
    gc.collect()
    try:
        with torch.no_grad():
            torch.cuda.empty_cache()
    except:
        pass
        
    #Reload our badwords
    koboldai_vars.badwordsids = koboldai_settings.badwordsids_default


def load_model(model_backend, initial_load=False):
    global model
    global tokenizer
    global model_config

    koboldai_vars.aibusy = True
    koboldai_vars.horde_share = False

    koboldai_vars.reset_model()

    koboldai_vars.noai = False
    set_aibusy(True)
    if koboldai_vars.model != 'ReadOnly':
        emit('from_server', {'cmd': 'model_load_status', 'data': "Loading {}".format(model_backends[model_backend].model_name if "model_name" in vars(model_backends[model_backend]) else model_backends[model_backend].id)}, broadcast=True)
        #Have to add a sleep so the server will send the emit for some reason
        time.sleep(0.1)    
    
    # If transformers model was selected & GPU available, ask to use CPU or GPU
    if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ["InferKit", "Colab", "API", "CLUSTER", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        # loadmodelsettings()
        # loadsettings()
        logger.init("GPU support", status="Searching")
        koboldai_vars.bmsupported = ((koboldai_vars.model_type != 'gpt2') or koboldai_vars.model_type in ("gpt_neo", "gptj", "xglm", "opt")) and not koboldai_vars.nobreakmodel
        if(koboldai_vars.hascuda):
            logger.init_ok("GPU support", status="Found")
        else:
            logger.init_warn("GPU support", status="Not Found")
        
        #if koboldai_vars.hascuda:
        #    if(koboldai_vars.bmsupported):
        #        koboldai_vars.usegpu = False
        #        koboldai_vars.breakmodel = True
        #    else:
        #        koboldai_vars.breakmodel = False
        #        koboldai_vars.usegpu = use_gpu
    else:
        koboldai_vars.default_preset = koboldai_settings.default_preset

                    
    
    with use_custom_unpickler(RestrictedUnpickler):
        model = model_backends[model_backend]
        koboldai_vars.supported_gen_modes = [x.value for x in model.get_supported_gen_modes()]
        model.load(initial_load=initial_load, save_model=not (args.colab or args.cacheonly) or args.savemodel)

    koboldai_vars.model = model.model_name if "model_name" in vars(model) else model.id #Should have model_name, but it could be set to id depending on how it's setup
    if koboldai_vars.model in ("NeoCustom", "GPT2Custom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"):
        koboldai_vars.model = os.path.basename(os.path.normpath(model.path))
        logger.info(koboldai_vars.model)
    logger.debug("Model Type: {}".format(koboldai_vars.model_type))
    
    # TODO: Convert everywhere to use model.tokenizer
    if model:
        tokenizer = model.tokenizer

    loadmodelsettings()
    loadsettings()

    lua_startup()
    # Load scripts
    load_lua_scripts()
    
    final_startup()
    #if not initial_load:
    set_aibusy(False)
    socketio.emit('from_server', {'cmd': 'hide_model_name'}, broadcast=True, room="UI_1")
    time.sleep(0.1)
        
    if not koboldai_vars.gamestarted:
        setStartState()
        sendsettings()
        refresh_settings()
    
    #Saving the tokenizer to the KoboldStoryRegister class so we can do token counting on the story data
    if 'tokenizer' in [x for x in globals()]:
        koboldai_vars.tokenizer = tokenizer
    
    #Let's load the presets
    preset_same_model = {}
    preset_same_class_size = {}
    preset_same_class = {}
    preset_others = {}
    model_info_data = model_info()
    
    for file in os.listdir("./presets"):
        if file[-8:] == '.presets':
            with open("./presets/{}".format(file)) as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            for preset in data:
                if preset['Model Name'] == koboldai_vars.model:
                    preset_same_model[preset['preset']] = preset
                    preset_same_model[preset['preset']]['Match'] = "Recommended"
                elif not (preset['preset'] in preset_same_model and preset_same_model[preset['preset']]['Match'] == "Recommended") and model_info_data['Model Type'] == preset['Model Type'] and model_info_data['Model Size'] == preset['Model Size']:
                    preset_same_class_size[preset['preset']] = preset
                    preset_same_class_size[preset['preset']]['Match'] = "Recommended"
                elif not (preset['preset'] in preset_same_model and preset_same_model[preset['preset']]['Match'] == "Recommended") and not ((preset['preset'] in preset_same_class_size and preset_same_class_size[preset['preset']]['Match'] == "Recommended")) and model_info_data['Model Type'] == preset['Model Type']:
                    preset_same_class[preset['preset']] = preset
                    preset_same_class[preset['preset']]['Match'] = "Same Class"
                elif preset['preset'] not in preset_same_model and preset['preset'] not in preset_same_class_size and preset['preset'] not in preset_same_class:
                    preset_others[preset['preset']] = preset
                    preset_others[preset['preset']]['Match'] = "Other"
    
    #Combine it all
    presets = preset_same_model
    for item in preset_same_class_size:
        if item not in presets:
            presets[item] = preset_same_class_size[item]
    for item in preset_same_class:
        if item not in presets:
            presets[item] = preset_same_class[item]
    for item in preset_others:
        if item not in presets:
            presets[item] = preset_others[item]
    
    presets['Default'] = koboldai_vars.default_preset
    
    koboldai_vars.uid_presets = presets
    #We want our data to be a 2 deep dict. Top level is "Recommended", "Same Class", "Model 1", "Model 2", etc
    #Next layer is "Official", "Custom"
    #Then the preset name
    
    to_use = OrderedDict()
    
    to_use["Recommended"] = {
        "Official": [presets[x] for x in presets if presets[x]['Match'] == "Recommended" and presets[x]['Preset Category'] == "Official"], 
        "Custom": [presets[x] for x in presets if presets[x]['Match'] == "Recommended" and presets[x]['Preset Category'] == "Custom"], 
    }
    to_use["Same Class"] = {
        "Official": [presets[x] for x in presets if presets[x]['Match'] == "Same Class" and presets[x]['Preset Category'] == "Official"], 
        "Custom": [presets[x] for x in presets if presets[x]['Match'] == "Same Class" and presets[x]['Preset Category'] == "Custom"], 
    }
    to_use["Other"] = {
    "Official": [presets[x] for x in presets if presets[x]['Match'] == "Other" and presets[x]['Preset Category'] == "Official"], 
    "Custom": [presets[x] for x in presets if presets[x]['Match'] == "Other" and presets[x]['Preset Category'] == "Custom"], 
    }
    koboldai_vars.presets = to_use

    
    koboldai_vars.aibusy = False
    if not os.path.exists("./softprompts"):
        os.mkdir("./softprompts")
    koboldai_vars.splist = [[f, get_softprompt_desc(os.path.join("./softprompts", f),None,True)] for f in os.listdir("./softprompts") if os.path.isfile(os.path.join("./softprompts", f)) and valid_softprompt(os.path.join("./softprompts", f))]
    if initial_load and koboldai_vars.cloudflare_link != "":
        logger.message(f"KoboldAI has finished loading and is available at the following link: {koboldai_vars.cloudflare_link}")
        logger.message(f"KoboldAI has finished loading and is available at the following link for the Classic UI: {koboldai_vars.cloudflare_link}/classic")
        logger.message(f"KoboldAI has finished loading and is available at the following link for KoboldAI Lite: {koboldai_vars.cloudflare_link}/lite")
        logger.message(f"KoboldAI has finished loading and is available at the following link for the API: {koboldai_vars.cloudflare_link}/api")


# Setup IP Whitelisting
# Define a function to check if IP is allowed
def is_allowed_ip():
    global allowed_ips
    client_ip = request.remote_addr
    if request.path != '/genre_data.json':
        print("Connection Attempt: " + request.remote_addr)
        if allowed_ips:
            print("Allowed?: ",  request.remote_addr in allowed_ips)
    return client_ip in allowed_ips



# Define a decorator to enforce IP whitelisting
def require_allowed_ip(func):
    @wraps(func)
    def decorated(*args, **kwargs):
    
        if enable_whitelist and not is_allowed_ip():
            return abort(403)
        return func(*args, **kwargs)
    return decorated




# Set up Flask routes
@app.route('/classic')
@require_allowed_ip
def index():
    if args.no_ui:
        return redirect('/api/latest')
    else:
        return render_template('index.html', hide_ai_menu=args.noaimenu)
@app.route('/api', strict_slashes=False)
@require_allowed_ip
def api():
    return redirect('/api/latest')
@app.route('/favicon.ico')

def favicon():
    return send_from_directory(app.root_path,
                                   'koboldai.ico', mimetype='image/vnd.microsoft.icon')    
@app.route('/download')
@require_allowed_ip
def download():
    if args.no_ui:
        raise NotFound()

    save_format = request.args.get("format", "json").strip().lower()

    if(save_format == "plaintext"):
        txt = koboldai_vars.prompt + "".join(koboldai_vars.actions.values())
        save = Response(txt)
        filename = path.basename(koboldai_vars.savedir)
        if filename[-5:] == ".json":
            filename = filename[:-5]
        save.headers.set('Content-Disposition', 'attachment', filename='%s.txt' % filename)
        return(save)
    
    
    
    save = Response(koboldai_vars.download_story())
    filename = path.basename(koboldai_vars.savedir)
    if filename[-5:] == ".json":
        filename = filename[:-5]
    save.headers.set('Content-Disposition', 'attachment', filename='%s.json' % filename)
    return(save)


#============================ LUA API =============================#
_bridged = {}
F = TypeVar("F", bound=Callable)
def lua_startup():
    global _bridged
    global F
    global bridged
    #if(path.exists("settings/" + getmodelname().replace('/', '_') + ".settings")):
    #    file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
    #    js   = json.load(file)
    #    if("userscripts" in js):
    #        koboldai_vars.userscripts = []
    #        for userscript in js["userscripts"]:
    #            if type(userscript) is not str:
    #               continue
    #            userscript = userscript.strip()
    #            if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
    #                koboldai_vars.userscripts.append(userscript)
    #    if("corescript" in js and type(js["corescript"]) is str and all(q not in js["corescript"] for q in ("..", ":")) and all(js["corescript"][0] not in q for q in ("/", "\\"))):
    #        koboldai_vars.corescript = js["corescript"]
    #    else:
    #        koboldai_vars.corescript = "default.lua"
    #    file.close()
        
    #==================================================================#
    #  Lua runtime startup
    #==================================================================#

    print("", end="", flush=True)
    logger.init("LUA bridge", status="Starting")

    # Set up Lua state
    koboldai_vars.lua_state = lupa.LuaRuntime(unpack_returned_tuples=True)

    # Load bridge.lua
    bridged = {
        "corescript_path": "cores",
        "userscript_path": "userscripts",
        "config_path": "userscripts",
        "lib_paths": koboldai_vars.lua_state.table("lualibs", os.path.join("extern", "lualibs")),
        "koboldai_vars": koboldai_vars,
    }
    for kwarg in _bridged:
        bridged[kwarg] = _bridged[kwarg]
    try:
        koboldai_vars.lua_kobold, koboldai_vars.lua_koboldcore, koboldai_vars.lua_koboldbridge = koboldai_vars.lua_state.globals().dofile("bridge.lua")(
            koboldai_vars.lua_state.globals().python,
            bridged,
        )
    except lupa.LuaError as e:
        print(colors.RED + "ERROR!" + colors.END)
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        logger.error('LUA ERROR: ' + str(e).replace("\033", ""))
        logger.warning("Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.")
        socketio.emit("error", str(e), broadcast=True, room="UI_2")
        exit(1)
    logger.init_ok("LUA bridge", status="OK")


def lua_log_format_name(name):
    return f"[{name}]" if type(name) is str else "CORE"


def bridged_kwarg(name=None):
    def _bridged_kwarg(f: F):
        _bridged[name if name is not None else f.__name__[4:] if f.__name__[:4] == "lua_" else f.__name__] = f
        return f
    return _bridged_kwarg

#==================================================================#
#  Event triggered when a userscript is loaded
#==================================================================#
@bridged_kwarg()
def load_callback(filename, modulename):
    print(colors.GREEN + f"Loading Userscript [{modulename}] <{filename}>" + colors.END)

#==================================================================#
#  Load all Lua scripts
#==================================================================#
def load_lua_scripts():
    logger.init("LUA Scripts", status="Starting")

    filenames = []
    modulenames = []
    descriptions = []

    lst = fileops.getusfiles(long_desc=True)
    filenames_dict = {ob["filename"]: i for i, ob in enumerate(lst)}

    for filename in koboldai_vars.userscripts:
        if filename in filenames_dict:
            i = filenames_dict[filename]
            filenames.append(filename)
            modulenames.append(lst[i]["modulename"])
            descriptions.append(lst[i]["description"])

    koboldai_vars.has_genmod = False

    try:
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        tpool.execute(koboldai_vars.lua_koboldbridge.load_corescript, koboldai_vars.corescript)
        koboldai_vars.has_genmod = tpool.execute(koboldai_vars.lua_koboldbridge.load_userscripts, filenames, modulenames, descriptions)
        koboldai_vars.lua_running = True
    except lupa.LuaError as e:
        try:
            koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        except:
            pass
        koboldai_vars.lua_running = False
        if(koboldai_vars.serverstarted):
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
            sendUSStatItems()
        logger.error('LUA ERROR: ' + str(e).replace("\033", ""))
        logger.warning("Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.")
        socketio.emit("error", str(e), broadcast=True, room="UI_2")
        if(koboldai_vars.serverstarted):
            set_aibusy(0)
    logger.init_ok("LUA Scripts", status="OK")

#==================================================================#
#  Print message that originates from the userscript with the given name
#==================================================================#
@bridged_kwarg()
def lua_print(msg):
    if(koboldai_vars.lua_logname != koboldai_vars.lua_koboldbridge.logging_name):
        koboldai_vars.lua_logname = koboldai_vars.lua_koboldbridge.logging_name
        print(colors.BLUE + lua_log_format_name(koboldai_vars.lua_logname) + ":" + colors.END, file=sys.stderr)
    print(colors.PURPLE + msg.replace("\033", "") + colors.END)

#==================================================================#
#  Print warning that originates from the userscript with the given name
#==================================================================#
@bridged_kwarg()
def lua_warn(msg):
    if(koboldai_vars.lua_logname != koboldai_vars.lua_koboldbridge.logging_name):
        koboldai_vars.lua_logname = koboldai_vars.lua_koboldbridge.logging_name
        print(colors.BLUE + lua_log_format_name(koboldai_vars.lua_logname) + ":" + colors.END, file=sys.stderr)
    print(colors.YELLOW + msg.replace("\033", "") + colors.END)

#==================================================================#
#  Decode tokens into a string using current tokenizer
#==================================================================#
@bridged_kwarg()
def lua_decode(tokens):
    tokens = list(tokens.values())
    assert type(tokens) is list
    if("tokenizer" not in globals()):
        from transformers import GPT2Tokenizer
        global tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
    return utils.decodenewlines(tokenizer.decode(tokens))

#==================================================================#
#  Encode string into list of token IDs using current tokenizer
#==================================================================#
@bridged_kwarg()
def lua_encode(string):
    assert type(string) is str
    if("tokenizer" not in globals()):
        from transformers import GPT2Tokenizer
        global tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
    return tokenizer.encode(utils.encodenewlines(string))

#==================================================================#
#  Computes context given a submission, Lua array of entry UIDs and a Lua array
#  of folder UIDs
#==================================================================#
@bridged_kwarg()
def lua_compute_context(submission, entries, folders, kwargs):
    assert type(submission) is str
    if(kwargs is None):
        kwargs = koboldai_vars.lua_state.table()
    actions = koboldai_vars.actions
    allowed_entries = None
    allowed_folders = None
    if(entries is not None):
        allowed_entries = set()
        i = 1
        while(entries[i] is not None):
            allowed_entries.add(int(entries[i]))
            i += 1
    if(folders is not None):
        allowed_folders = set()
        i = 1
        while(folders[i] is not None):
            allowed_folders.add(int(folders[i]))
            i += 1
    txt, _, _, found_entries = koboldai_vars.calc_ai_text(submitted_text=submission,
                                                allowed_wi_entries=allowed_entries,
                                                allowed_wi_folders=allowed_folders)
    return utils.decodenewlines(tokenizer.decode(txt))

#==================================================================#
#  Get property of a world info entry given its UID and property name
#==================================================================#
@bridged_kwarg()
def lua_get_attr(uid, k):
    assert type(uid) is int and type(k) is str
    if(uid in koboldai_vars.worldinfo_u and k in (
        "key",
        "keysecondary",
        "content",
        "comment",
        "folder",
        "num",
        "selective",
        "constant",
        "uid",
    )):
        return koboldai_vars.worldinfo_u[uid][k]

#==================================================================#
#  Set property of a world info entry given its UID, property name and new value
#==================================================================#
@bridged_kwarg()
def lua_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in koboldai_vars.worldinfo_u and k in (
        "key",
        "keysecondary",
        "content",
        "comment",
        "selective",
        "constant",
    )
    if(type(koboldai_vars.worldinfo_u[uid][k]) is int and type(v) is float):
        v = int(v)
    assert type(koboldai_vars.worldinfo_u[uid][k]) is type(v)
    koboldai_vars.worldinfo_u[uid][k] = v
    print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set {k} of world info entry {uid} to {v}" + colors.END)
    koboldai_vars.sync_worldinfo_v1_to_v2()
    sendwi()

#==================================================================#
#  Get property of a world info folder given its UID and property name
#==================================================================#
@bridged_kwarg()
def lua_folder_get_attr(uid, k):
    assert type(uid) is int and type(k) is str
    if(uid in koboldai_vars.wifolders_d and k in (
        "name",
    )):
        return koboldai_vars.wifolders_d[uid][k]

#==================================================================#
#  Set property of a world info folder given its UID, property name and new value
#==================================================================#
@bridged_kwarg()
def lua_folder_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in koboldai_vars.wifolders_d and k in (
        "name",
    )
    if(type(koboldai_vars.wifolders_d[uid][k]) is int and type(v) is float):
        v = int(v)
    assert type(koboldai_vars.wifolders_d[uid][k]) is type(v)
    koboldai_vars.wifolders_d[uid][k] = v
    print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set {k} of world info folder {uid} to {v}" + colors.END)
    koboldai_vars.sync_worldinfo_v1_to_v2()
    sendwi()

#==================================================================#
#  Get the "Amount to Generate"
#==================================================================#
@bridged_kwarg()
def lua_get_genamt():
    return koboldai_vars.genamt

#==================================================================#
#  Set the "Amount to Generate"
#==================================================================#
@bridged_kwarg()
def lua_set_genamt(genamt):
    assert koboldai_vars.lua_koboldbridge.userstate != "genmod" and type(genamt) in (int, float) and genamt >= 0
    print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set genamt to {int(genamt)}" + colors.END)
    koboldai_vars.genamt = int(genamt)

#==================================================================#
#  Get the "Gens Per Action"
#==================================================================#
@bridged_kwarg()
def lua_get_numseqs():
    return koboldai_vars.numseqs

#==================================================================#
#  Set the "Gens Per Action"
#==================================================================#
@bridged_kwarg()
def lua_set_numseqs(numseqs):
    assert type(numseqs) in (int, float) and numseqs >= 1
    print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set numseqs to {int(numseqs)}" + colors.END)
    koboldai_vars.numseqs = int(numseqs)

#==================================================================#
#  Check if a setting exists with the given name
#==================================================================#
@bridged_kwarg()
def lua_has_setting(setting):
    return setting in (
        "anotedepth",
        "settemp",
        "settopp",
        "settopk",
        "settfs",
        "settypical",
        "settopa",
        "setreppen",
        "setreppenslope",
        "setreppenrange",
        "settknmax",
        "setwidepth",
        "setuseprompt",
        "setadventure",
        "setchatmode",
        "setdynamicscan",
        "setnopromptgen",
        "autosave",
        "setrngpersist",
        "temp",
        "topp",
        "top_p",
        "topk",
        "top_k",
        "tfs",
        "typical",
        "topa",
        "reppen",
        "reppenslope",
        "reppenrange",
        "tknmax",
        "widepth",
        "useprompt",
        "chatmode",
        "chatname",
        "botname",
        "adventure",
        "dynamicscan",
        "nopromptgen",
        "rngpersist",
        "frmttriminc",
        "frmtrmblln",
        "frmtrmspch",
        "frmtadsnsp",
        "frmtsingleline",
        "triminc",
        "rmblln",
        "rmspch",
        "adsnsp",
        "singleline",
        "output_streaming",
        "show_probs"
    )

#==================================================================#
#  Return the setting with the given name if it exists
#==================================================================#
@bridged_kwarg()
def lua_get_setting(setting):
    if(setting in ("settemp", "temp")): return koboldai_vars.temp
    if(setting in ("settopp", "topp", "top_p")): return koboldai_vars.top_p
    if(setting in ("settopk", "topk", "top_k")): return koboldai_vars.top_k
    if(setting in ("settfs", "tfs")): return koboldai_vars.tfs
    if(setting in ("settypical", "typical")): return koboldai_vars.typical
    if(setting in ("settopa", "topa")): return koboldai_vars.top_a
    if(setting in ("setreppen", "reppen")): return koboldai_vars.rep_pen
    if(setting in ("setreppenslope", "reppenslope")): return koboldai_vars.rep_pen_slope
    if(setting in ("setreppenrange", "reppenrange")): return koboldai_vars.rep_pen_range
    if(setting in ("settknmax", "tknmax")): return koboldai_vars.max_length
    if(setting == "anotedepth"): return koboldai_vars.andepth
    if(setting in ("setwidepth", "widepth")): return koboldai_vars.widepth
    if(setting in ("setuseprompt", "useprompt")): return koboldai_vars.useprompt
    if(setting in ("setadventure", "adventure")): return koboldai_vars.adventure
    if(setting in ("setchatmode", "chatmode")): return koboldai_vars.chatmode
    if(setting in ("setdynamicscan", "dynamicscan")): return koboldai_vars.dynamicscan
    if(setting in ("setnopromptgen", "nopromptgen")): return koboldai_vars.nopromptgen
    if(setting in ("autosave", "autosave")): return koboldai_vars.autosave
    if(setting in ("setrngpersist", "rngpersist")): return koboldai_vars.rngpersist
    if(setting in ("frmttriminc", "triminc")): return koboldai_vars.frmttriminc
    if(setting in ("frmtrmblln", "rmblln")): return koboldai_vars.frmttrmblln
    if(setting in ("frmtrmspch", "rmspch")): return koboldai_vars.frmttrmspch
    if(setting in ("frmtadsnsp", "adsnsp")): return koboldai_vars.frmtadsnsp
    if(setting in ("frmtsingleline", "singleline")): return koboldai_vars.singleline
    if(setting == "output_streaming"): return koboldai_vars.output_streaming
    if(setting == "show_probs"): return koboldai_vars.show_probs

#==================================================================#
#  Set the setting with the given name if it exists
#==================================================================#
@bridged_kwarg()
def lua_set_setting(setting, v):
    actual_type = type(lua_get_setting(setting))
    assert v is not None and (actual_type is type(v) or (actual_type is int and type(v) is float))
    v = actual_type(v)
    print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} set {setting} to {v}" + colors.END)
    if(setting in ("setadventure", "adventure") and v):
        koboldai_vars.actionmode = 1
    if(setting in ("settemp", "temp")): koboldai_vars.temp = v
    if(setting in ("settopp", "topp")): koboldai_vars.top_p = v
    if(setting in ("settopk", "topk")): koboldai_vars.top_k = v
    if(setting in ("settfs", "tfs")): koboldai_vars.tfs = v
    if(setting in ("settypical", "typical")): koboldai_vars.typical = v
    if(setting in ("settopa", "topa")): koboldai_vars.top_a = v
    if(setting in ("setreppen", "reppen")): koboldai_vars.rep_pen = v
    if(setting in ("setreppenslope", "reppenslope")): koboldai_vars.rep_pen_slope = v
    if(setting in ("setreppenrange", "reppenrange")): koboldai_vars.rep_pen_range = v
    if(setting in ("settknmax", "tknmax")): koboldai_vars.max_length = v; return True
    if(setting == "anotedepth"): koboldai_vars.andepth = v; return True
    if(setting in ("setwidepth", "widepth")): koboldai_vars.widepth = v; return True
    if(setting in ("setuseprompt", "useprompt")): koboldai_vars.useprompt = v; return True
    if(setting in ("setadventure", "adventure")): koboldai_vars.adventure = v
    if(setting in ("setdynamicscan", "dynamicscan")): koboldai_vars.dynamicscan = v
    if(setting in ("setnopromptgen", "nopromptgen")): koboldai_vars.nopromptgen = v
    if(setting in ("autosave", "noautosave")): koboldai_vars.autosave = v
    if(setting in ("setrngpersist", "rngpersist")): koboldai_vars.rngpersist = v
    if(setting in ("setchatmode", "chatmode")): koboldai_vars.chatmode = v
    if(setting in ("frmttriminc", "triminc")): koboldai_vars.frmttriminc = v
    if(setting in ("frmtrmblln", "rmblln")): koboldai_vars.frmttrmblln = v
    if(setting in ("frmtrmspch", "rmspch")): koboldai_vars.frmttrmspch = v
    if(setting in ("frmtadsnsp", "adsnsp")): koboldai_vars.frmtadsnsp = v
    if(setting in ("frmtsingleline", "singleline")): koboldai_vars.singleline = v
    if(setting == "output_streaming"): koboldai_vars.output_streaming = v
    if(setting == "show_probs"): koboldai_vars.show_probs = v

#==================================================================#
#  Get contents of memory
#==================================================================#
@bridged_kwarg()
def lua_get_memory():
    return koboldai_vars.memory

#==================================================================#
#  Set contents of memory
#==================================================================#
@bridged_kwarg()
def lua_set_memory(m):
    assert type(m) is str
    koboldai_vars.memory = m

#==================================================================#
#  Get contents of author's note
#==================================================================#
@bridged_kwarg()
def lua_get_authorsnote():
    return koboldai_vars.authornote

#==================================================================#
#  Set contents of author's note
#==================================================================#
@bridged_kwarg()
def lua_set_authorsnote(m):
    assert type(m) is str
    koboldai_vars.authornote = m

#==================================================================#
#  Get contents of author's note template
#==================================================================#
@bridged_kwarg()
def lua_get_authorsnotetemplate():
    return koboldai_vars.authornotetemplate

#==================================================================#
#  Set contents of author's note template
#==================================================================#
@bridged_kwarg()
def lua_set_authorsnotetemplate(m):
    assert type(m) is str
    koboldai_vars.authornotetemplate = m

#==================================================================#
#  Save settings and send them to client
#==================================================================#
@bridged_kwarg()
def lua_resend_settings():
    print("lua_resend_settings")
    settingschanged()
    refresh_settings()

#==================================================================#
#  Set story chunk text and delete the chunk if the new chunk is empty
#==================================================================#
@bridged_kwarg()
def lua_set_chunk(k, v):
    assert type(k) in (int, None) and type(v) is str
    assert k >= 0
    assert k != 0 or len(v) != 0
    if(len(v) == 0):
        print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} deleted story chunk {k}" + colors.END)
        chunk = int(k)
        koboldai_vars.actions.delete_action(chunk-1)
        koboldai_vars.lua_deleted.add(chunk)
        send_debug()
    else:
        if(k == 0):
            print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} edited prompt chunk" + colors.END)
        else:
            print(colors.GREEN + f"{lua_log_format_name(koboldai_vars.lua_koboldbridge.logging_name)} edited story chunk {k}" + colors.END)
        chunk = int(k)
        if(chunk == 0):
            if(koboldai_vars.lua_koboldbridge.userstate == "genmod"):
                koboldai_vars._prompt = v
            koboldai_vars.lua_edited.add(chunk)
            koboldai_vars.prompt = v
        else:
            koboldai_vars.lua_edited.add(chunk)
            koboldai_vars.actions[chunk-1] = v
            send_debug()

#==================================================================#
#  Get model type as "gpt-2-xl", "gpt-neo-2.7B", etc.
#==================================================================#
@bridged_kwarg()
def lua_get_modeltype():
    if(koboldai_vars.noai):
        return "readonly"
    if(koboldai_vars.model in ("Colab", "API", "CLUSTER", "OAI", "InferKit")):
        return "api"
    if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and (koboldai_vars.model in ("GPT2Custom", "NeoCustom") or koboldai_vars.model_type in ("gpt2", "gpt_neo", "gptj"))):
        hidden_size = get_hidden_size_from_model(model)
    if(koboldai_vars.model in ("gpt2",) or (koboldai_vars.model_type == "gpt2" and hidden_size == 768)):
        return "gpt2"
    if(koboldai_vars.model in ("gpt2-medium",) or (koboldai_vars.model_type == "gpt2" and hidden_size == 1024)):
        return "gpt2-medium"
    if(koboldai_vars.model in ("gpt2-large",) or (koboldai_vars.model_type == "gpt2" and hidden_size == 1280)):
        return "gpt2-large"
    if(koboldai_vars.model in ("gpt2-xl",) or (koboldai_vars.model_type == "gpt2" and hidden_size == 1600)):
        return "gpt2-xl"
    if(koboldai_vars.model_type == "gpt_neo" and hidden_size == 768):
        return "gpt-neo-125M"
    if(koboldai_vars.model in ("EleutherAI/gpt-neo-1.3B",) or (koboldai_vars.model_type == "gpt_neo" and hidden_size == 2048)):
        return "gpt-neo-1.3B"
    if(koboldai_vars.model in ("EleutherAI/gpt-neo-2.7B",) or (koboldai_vars.model_type == "gpt_neo" and hidden_size == 2560)):
        return "gpt-neo-2.7B"
    if(koboldai_vars.model in ("EleutherAI/gpt-j-6B",) or ((koboldai_vars.use_colab_tpu or koboldai_vars.model == "TPUMeshTransformerGPTJ") and tpu_mtj_backend.params["d_model"] == 4096) or (koboldai_vars.model_type in ("gpt_neo", "gptj") and hidden_size == 4096)):
        return "gpt-j-6B"
    return "unknown"

#==================================================================#
#  Get model backend as "transformers" or "mtj"
#==================================================================#
@bridged_kwarg()
def lua_get_modelbackend():
    if(koboldai_vars.noai):
        return "readonly"
    if(koboldai_vars.model in ("Colab", "API", "CLUSTER", "OAI", "InferKit")):
        return "api"
    if(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        return "mtj"
    return "transformers"

#==================================================================#
#  Check whether model is loaded from a custom path
#==================================================================#
@bridged_kwarg()
def lua_is_custommodel():
    return koboldai_vars.model in ("GPT2Custom", "NeoCustom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")

#==================================================================#
#  Return the filename (as a string) of the current soft prompt, or
#  None if no soft prompt is loaded
#==================================================================#
@bridged_kwarg()
def lua_get_spfilename():
    return koboldai_vars.spfilename.strip() or None

#==================================================================#
#  When called with a string as argument, sets the current soft prompt;
#  when called with None as argument, uses no soft prompt.
#  Returns True if soft prompt changed, False otherwise.
#==================================================================#
@bridged_kwarg()
def lua_set_spfilename(filename: Union[str, None]):
    if(filename is None):
        filename = ""
    filename = str(filename).strip()
    changed = lua_get_spfilename() != filename
    assert all(q not in filename for q in ("/", "\\"))
    spRequest(filename)
    return changed

#==================================================================#
#  
#==================================================================#
def execute_inmod():
    setgamesaved(False)
    koboldai_vars.lua_logname = ...
    koboldai_vars.lua_edited = set()
    koboldai_vars.lua_deleted = set()
    try:
        tpool.execute(koboldai_vars.lua_koboldbridge.execute_inmod)
    except lupa.LuaError as e:
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        koboldai_vars.lua_running = False
        emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
        sendUSStatItems()
        logger.error('LUA ERROR: ' + str(e).replace("\033", ""))
        logger.warning("Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.")
        socketio.emit("error", str(e), broadcast=True, room="UI_2")
        set_aibusy(0)

def execute_genmod():
    koboldai_vars.lua_koboldbridge.execute_genmod()

def execute_outmod():
    setgamesaved(False)
    emit('from_server', {'cmd': 'hidemsg', 'data': ''}, broadcast=True, room="UI_1")
    try:
        tpool.execute(koboldai_vars.lua_koboldbridge.execute_outmod)
    except lupa.LuaError as e:
        koboldai_vars.lua_koboldbridge.obliterate_multiverse()
        koboldai_vars.lua_running = False
        emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
        sendUSStatItems()
        logger.error('LUA ERROR: ' + str(e).replace("\033", ""))
        logger.warning("Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.")
        socketio.emit("error", str(e), broadcast=True, room="UI_2")
        set_aibusy(0)
    if(koboldai_vars.lua_koboldbridge.resend_settings_required):
        koboldai_vars.lua_koboldbridge.resend_settings_required = False
        lua_resend_settings()
    for k in koboldai_vars.lua_edited:
        inlineedit(k, koboldai_vars.actions[k])
    for k in koboldai_vars.lua_deleted:
        inlinedelete(k)




#============================ METHODS =============================#    

#==================================================================#
# Event triggered when browser SocketIO is loaded and connects to server
#==================================================================#
@socketio.on('connect')
def do_connect(_):
    print("Connection Attempt: " + request.remote_addr)
    if allowed_ips:
        print("Allowed?: ",  request.remote_addr in allowed_ips)
    if request.args.get("rely") == "true":
        return
    logger.info("Client connected! UI_{}".format(request.args.get('ui')))
    #If this we have a message to send to the users and they haven't seen it we'll transmit it now
    eventlet.spawn(send_one_time_messages, '', wait_time=1)
    
    join_room("UI_{}".format(request.args.get('ui')))
    if 'story' not in session:
        session['story'] = 'default'
    join_room(session['story'])
    logger.debug("Joining Room UI_{}".format(request.args.get('ui')))
    logger.debug("Session['Story']: {}".format(session['story']))
    if request.args.get("ui") == "2":
        ui2_connect()
        return
    logger.debug("{0}Client connected!{1}".format(colors.GREEN, colors.END))
    emit('from_server', {'cmd': 'setchatname', 'data': koboldai_vars.chatname}, room="UI_1")
    emit('from_server', {'cmd': 'setbotname', 'data': koboldai_vars.botname}, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, room="UI_1")
    emit('from_server', {'cmd': 'connected', 'smandelete': koboldai_vars.smandelete, 'smanrename': koboldai_vars.smanrename, 'modelname': getmodelname()}, room="UI_1")
    if(koboldai_vars.host):
        emit('from_server', {'cmd': 'runs_remotely'}, room="UI_1")
    if(koboldai_vars.flaskwebgui):
        emit('from_server', {'cmd': 'flaskwebgui'}, room="UI_1")
    if(koboldai_vars.allowsp):
        emit('from_server', {'cmd': 'allowsp', 'data': koboldai_vars.allowsp}, room="UI_1")

    sendUSStatItems()
    emit('from_server', {'cmd': 'spstatitems', 'data': {koboldai_vars.spfilename: koboldai_vars.spmeta} if koboldai_vars.allowsp and len(koboldai_vars.spfilename) else {}}, broadcast=True, room="UI_1")

    if(not koboldai_vars.gamestarted):
        setStartState()
        sendsettings()
        refresh_settings()
        koboldai_vars.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, room="UI_1")
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, room="UI_1")
        koboldai_vars.mode = "play"
    else:
        # Game in session, send current game data and ready state to browser
        refresh_story()
        sendsettings()
        refresh_settings()
        emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, room="UI_1")
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, room="UI_1")
        if(koboldai_vars.mode == "play"):
            if(not koboldai_vars.aibusy):
                emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, room="UI_1")
            else:
                emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'}, room="UI_1")
        elif(koboldai_vars.mode == "edit"):
            emit('from_server', {'cmd': 'editmode', 'data': 'true'}, room="UI_1")
        elif(koboldai_vars.mode == "memory"):
            emit('from_server', {'cmd': 'memmode', 'data': 'true'}, room="UI_1")
        elif(koboldai_vars.mode == "wi"):
            emit('from_server', {'cmd': 'wimode', 'data': 'true'}, room="UI_1")

    emit('from_server', {'cmd': 'gamesaved', 'data': koboldai_vars.gamesaved}, broadcast=True, room="UI_1")

#==================================================================#
# Event triggered when browser SocketIO sends data to the server
#==================================================================#
@socketio.on('message')
def get_message(msg):
    if not koboldai_vars.quiet:
        logger.debug(f"Data received: {msg}")
    # Submit action
    if(msg['cmd'] == 'submit'):
        if(koboldai_vars.mode == "play"):
            if(koboldai_vars.aibusy):
                if(msg.get('allowabort', False)):
                    koboldai_vars.abort = True
                return
            koboldai_vars.abort = False
            koboldai_vars.lua_koboldbridge.feedback = None
            if(koboldai_vars.chatmode):
                if(type(msg['chatname']) is not str):
                    raise ValueError("Chatname must be a string")
                koboldai_vars.chatname = msg['chatname']
                koboldai_vars.botname = msg['botname']
                settingschanged()
                emit('from_server', {'cmd': 'setchatname', 'data': koboldai_vars.chatname}, room="UI_1")
                emit('from_server', {'cmd': 'setbotname', 'data': koboldai_vars.botname}, room="UI_1")
            koboldai_vars.recentrng = koboldai_vars.recentrngm = None
            actionsubmit(msg['data'], actionmode=msg['actionmode'])
        elif(koboldai_vars.mode == "edit"):
            editsubmit(msg['data'])
        elif(koboldai_vars.mode == "memory"):
            memsubmit(msg['data'])
    # Retry Action
    elif(msg['cmd'] == 'retry'):
        if(koboldai_vars.aibusy):
            if(msg.get('allowabort', False)):
                koboldai_vars.abort = True
            return
        koboldai_vars.abort = False
        if(koboldai_vars.chatmode):
            if(type(msg['chatname']) is not str):
                raise ValueError("Chatname must be a string")
            koboldai_vars.chatname = msg['chatname']
            koboldai_vars.botname = msg['botname']
            settingschanged()
            emit('from_server', {'cmd': 'setchatname', 'data': koboldai_vars.chatname}, room="UI_1")
            emit('from_server', {'cmd': 'setbotname', 'data': koboldai_vars.botname}, room="UI_1")
        actionretry(msg['data'])
    # Back/Undo Action
    elif(msg['cmd'] == 'back'):
        ignore = actionback()
    # Forward/Redo Action
    elif(msg['cmd'] == 'redo'):
        actionredo()
    # EditMode Action (old)
    elif(msg['cmd'] == 'edit'):
        if(koboldai_vars.mode == "play"):
            koboldai_vars.mode = "edit"
            emit('from_server', {'cmd': 'editmode', 'data': 'true'}, broadcast=True, room="UI_1")
        elif(koboldai_vars.mode == "edit"):
            koboldai_vars.mode = "play"
            emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    # EditLine Action (old)
    elif(msg['cmd'] == 'editline'):
        editrequest(int(msg['data']))
    # Inline edit
    elif(msg['cmd'] == 'inlineedit'):
        inlineedit(msg['chunk'], msg['data'])
    elif(msg['cmd'] == 'inlinedelete'):
        inlinedelete(msg['data'])
    # DeleteLine Action (old)
    elif(msg['cmd'] == 'delete'):
        deleterequest()
    elif(msg['cmd'] == 'memory'):
        togglememorymode()
    #elif(not koboldai_vars.host and msg['cmd'] == 'savetofile'):
    #    savetofile()
    elif(not koboldai_vars.host and msg['cmd'] == 'loadfromfile'):
        loadfromfile()
    elif(msg['cmd'] == 'loadfromstring'):
        loadRequest(json.loads(msg['data']), filename=msg['filename'])
    elif(not koboldai_vars.host and msg['cmd'] == 'import'):
        importRequest()
    elif(msg['cmd'] == 'newgame'):
        newGameRequest()
    elif(msg['cmd'] == 'rndgame'):
        randomGameRequest(msg['data'], memory=msg['memory'])
    elif(msg['cmd'] == 'settemp'):
        koboldai_vars.temp = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltemp', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopp'):
        koboldai_vars.top_p = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopp', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopk'):
        koboldai_vars.top_k = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopk', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settfs'):
        koboldai_vars.tfs = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltfs', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settypical'):
        koboldai_vars.typical = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltypical', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopa'):
        koboldai_vars.top_a = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopa', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppen'):
        koboldai_vars.rep_pen = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppen', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppenslope'):
        koboldai_vars.rep_pen_slope = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppenslope', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppenrange'):
        koboldai_vars.rep_pen_range = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppenrange', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setoutput'):
        koboldai_vars.genamt = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeloutput', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settknmax'):
        koboldai_vars.max_length = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeltknmax', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setikgen'):
        koboldai_vars.ikgen = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelikgen', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    # Author's Note field update
    elif(msg['cmd'] == 'anote'):
        anotesubmit(msg['data'], template=msg['template'])
    # Author's Note depth update
    elif(msg['cmd'] == 'anotedepth'):
        koboldai_vars.andepth = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelanotedepth', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    # Format - Trim incomplete sentences
    elif(msg['cmd'] == 'frmttriminc'):
        koboldai_vars.frmttriminc = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtrmblln'):
        koboldai_vars.frmtrmblln = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtrmspch'):
        koboldai_vars.frmtrmspch = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtadsnsp'):
        koboldai_vars.frmtadsnsp = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'singleline'):
        koboldai_vars.singleline = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'importselect'):
        koboldai_vars.importnum = int(msg["data"].replace("import", ""))
    elif(msg['cmd'] == 'importcancel'):
        emit('from_server', {'cmd': 'popupshow', 'data': False}, room="UI_1")
        koboldai_vars.importjs  = {}
    elif(msg['cmd'] == 'importaccept'):
        emit('from_server', {'cmd': 'popupshow', 'data': False}, room="UI_1")
        importgame()
    elif(msg['cmd'] == 'wi'):
        togglewimode()
    elif(msg['cmd'] == 'wiinit'):
        if(int(msg['data']) < len(koboldai_vars.worldinfo)):
            setgamesaved(False)
            koboldai_vars.worldinfo[msg['data']]["init"] = True
            addwiitem(folder_uid=msg['folder'])
    elif(msg['cmd'] == 'wifolderinit'):
        addwifolder()
    elif(msg['cmd'] == 'wimoveitem'):
        movewiitem(msg['destination'], msg['data'])
    elif(msg['cmd'] == 'wimovefolder'):
        movewifolder(msg['destination'], msg['data'])
    elif(msg['cmd'] == 'widelete'):
        deletewi(msg['data'])
    elif(msg['cmd'] == 'wifolderdelete'):
        deletewifolder(msg['data'])
    elif(msg['cmd'] == 'wiexpand'):
        assert 0 <= int(msg['data']) < len(koboldai_vars.worldinfo)
        setgamesaved(False)
        emit('from_server', {'cmd': 'wiexpand', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiexpandfolder'):
        assert 0 <= int(msg['data']) < len(koboldai_vars.worldinfo)
        setgamesaved(False)
        emit('from_server', {'cmd': 'wiexpandfolder', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wifoldercollapsecontent'):
        setgamesaved(False)
        koboldai_vars.wifolders_d[msg['data']]['collapsed'] = True
        emit('from_server', {'cmd': 'wifoldercollapsecontent', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wifolderexpandcontent'):
        setgamesaved(False)
        koboldai_vars.wifolders_d[msg['data']]['collapsed'] = False
        emit('from_server', {'cmd': 'wifolderexpandcontent', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiupdate'):
        setgamesaved(False)
        num = int(msg['num'])
        fields = ("key", "keysecondary", "content", "comment")
        for field in fields:
            if(field in msg['data'] and type(msg['data'][field]) is str):
                koboldai_vars.worldinfo[num][field] = msg['data'][field]
        emit('from_server', {'cmd': 'wiupdate', 'num': msg['num'], 'data': {field: koboldai_vars.worldinfo[num][field] for field in fields}}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wifolderupdate'):
        setgamesaved(False)
        uid = msg['uid']
        fields = ("name", "collapsed")
        for field in fields:
            if(field in msg['data'] and type(msg['data'][field]) is (str if field != "collapsed" else bool)):
                koboldai_vars.wifolders_d[uid][field] = msg['data'][field]
        emit('from_server', {'cmd': 'wifolderupdate', 'uid': msg['uid'], 'data': {field: koboldai_vars.wifolders_d[uid][field] for field in fields}}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiselon'):
        setgamesaved(False)
        koboldai_vars.worldinfo[msg['data']]["selective"] = True
        emit('from_server', {'cmd': 'wiselon', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiseloff'):
        setgamesaved(False)
        koboldai_vars.worldinfo[msg['data']]["selective"] = False
        emit('from_server', {'cmd': 'wiseloff', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiconstanton'):
        setgamesaved(False)
        koboldai_vars.worldinfo[msg['data']]["constant"] = True
        emit('from_server', {'cmd': 'wiconstanton', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiconstantoff'):
        setgamesaved(False)
        koboldai_vars.worldinfo[msg['data']]["constant"] = False
        emit('from_server', {'cmd': 'wiconstantoff', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'sendwilist'):
        commitwi(msg['data'])
    elif(msg['cmd'] == 'aidgimport'):
        importAidgRequest(msg['data'])
    elif(msg['cmd'] == 'saveasrequest'):
        saveas(msg['data'])
    elif(msg['cmd'] == 'saverequest'):
        save()
    elif(msg['cmd'] == 'loadlistrequest'):
        getloadlist()
    elif(msg['cmd'] == 'splistrequest'):
        getsplist()
    elif(msg['cmd'] == 'uslistrequest'):
        unloaded, loaded = getuslist()
        emit('from_server', {'cmd': 'buildus', 'data': {"unloaded": unloaded, "loaded": loaded}}, room="UI_1")
    elif(msg['cmd'] == 'samplerlistrequest'):
        emit('from_server', {'cmd': 'buildsamplers', 'data': koboldai_vars.sampler_order}, room="UI_1")
    elif(msg['cmd'] == 'usloaded'):
        koboldai_vars.userscripts = []
        for userscript in msg['data']:
            if type(userscript) is not str:
                continue
            userscript = userscript.strip()
            if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                koboldai_vars.userscripts.append(userscript)
        settingschanged()
    elif(msg['cmd'] == 'usload'):
        load_lua_scripts()
        unloaded, loaded = getuslist()
        sendUSStatItems()
    elif(msg['cmd'] == 'samplers'):
        sampler_order = msg["data"]
        sampler_order_min_length = 6
        sampler_order_max_length = 7
        if(not isinstance(sampler_order, list)):
            raise ValueError(f"Sampler order must be a list, but got a {type(sampler_order)}")
        if(not (sampler_order_min_length <= len(sampler_order) <= sampler_order_max_length)):
            raise ValueError(f"Sampler order must be a list of length greater than or equal to {sampler_order_min_length} and less than or equal to {sampler_order_max_length}, but got a list of length {len(sampler_order)}")
        if(not all(isinstance(e, int) for e in sampler_order)):
            raise ValueError(f"Sampler order must be a list of ints, but got a list with at least one non-int element")
        if(min(sampler_order) != 0 or max(sampler_order) != len(sampler_order) - 1 or len(set(sampler_order)) != len(sampler_order)):
            raise ValueError(f"Sampler order list of length {len(sampler_order)} must be a permutation of the first {len(sampler_order)} nonnegative integers")
        koboldai_vars.sampler_order = sampler_order
        settingschanged()
    elif(msg['cmd'] == 'list_model'):
        sendModelSelection(menu=msg['data'])
    elif(msg['cmd'] == 'load_model'):
        logger.debug(f"Selected Model: {koboldai_vars.model_selected}")
        if not os.path.exists("settings/"):
            os.mkdir("settings")
        changed = True
        if os.path.exists("settings/" + koboldai_vars.model_selected.replace('/', '_') + ".breakmodel"):
            with open("settings/" + koboldai_vars.model_selected.replace('/', '_') + ".breakmodel", "r") as file:
                data = file.read().split('\n')[:2]
                if len(data) < 2:
                    data.append("0")
                gpu_layers, disk_layers = data
                if gpu_layers == msg['gpu_layers'] and disk_layers == msg['disk_layers']:
                    changed = False
        if changed:
            if koboldai_vars.model_selected in ["NeoCustom", "GPT2Custom"]:
                filename = "settings/{}.breakmodel".format(os.path.basename(os.path.normpath(koboldai_vars.custmodpth)))
            else:
                filename = "settings/{}.breakmodel".format(koboldai_vars.model_selected.replace('/', '_'))
            f = open(filename, "w")
            f.write(str(msg['gpu_layers']) + '\n' + str(msg['disk_layers']))
            f.close()
        koboldai_vars.colaburl = msg['url'] + "/request"
        koboldai_vars.model = koboldai_vars.model_selected
        if koboldai_vars.model == "CLUSTER":
            if type(msg['online_model']) is not list:
                if msg['online_model'] == '':
                    koboldai_vars.cluster_requested_models = []
                else:
                    koboldai_vars.cluster_requested_models = [msg['online_model']]
            else:
                koboldai_vars.cluster_requested_models = msg['online_model']
        load_model(use_gpu=msg['use_gpu'], gpu_layers=msg['gpu_layers'], disk_layers=msg['disk_layers'], online_model=msg['online_model'])
    elif(msg['cmd'] == 'show_model'):
        logger.info(f"Model Name: {getmodelname()}")
        emit('from_server', {'cmd': 'show_model_name', 'data': getmodelname()}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'selectmodel'):
        # This is run when a model line is selected from the UI (line from the model_menu variable) that is tagged as not a menu
        # otherwise we should be running the msg['cmd'] == 'list_model'
        
        # We have to do a bit of processing though, if we select a custom path, we need to list out the contents of folders
        # But if we select something else, we need to potentially show model layers for each GPU
        # We might also need to show key input. All of that happens here
        
        # The data variable will contain the model name. But our Custom lines need a bit more processing
        # If we're on a custom line that we have selected a model for, the path variable will be in msg
        # so if that's missing we need to run the menu to show the model folders in the models folder
        if msg['data'] in ('NeoCustom', 'GPT2Custom', 'customhuggingface') and 'path' not in msg and 'path_modelname' not in msg:
            if 'folder' not in msg or koboldai_vars.host:
                folder = "./models"
            else:
                folder = msg['folder']
            sendModelSelection(menu=msg['data'], folder=folder)
        elif msg['data'] in ('NeoCustom', 'GPT2Custom', 'customhuggingface') and 'path_modelname' in msg:
            #Here the user entered custom text in the text box. This could be either a model name or a path.
            if check_if_dir_is_model(msg['path_modelname']):
                koboldai_vars.model_selected = msg['data']
                koboldai_vars.custmodpth = msg['path_modelname']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                koboldai_vars.model_selected = msg['path_modelname']
                try:
                    get_model_info(koboldai_vars.model_selected)
                except:
                    emit('from_server', {'cmd': 'errmsg', 'data': "The model entered doesn't exist."}, room="UI_1")
        elif msg['data'] in ('NeoCustom', 'GPT2Custom', 'customhuggingface'):
            if check_if_dir_is_model(msg['path']):
                koboldai_vars.model_selected = msg['data']
                koboldai_vars.custmodpth = msg['path']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                if koboldai_vars.host:
                    sendModelSelection(menu=msg['data'], folder="./models")
                else:
                    sendModelSelection(menu=msg['data'], folder=msg['path'])
        else:
            koboldai_vars.model_selected = msg['data']
            if 'path' in msg:
                koboldai_vars.custmodpth = msg['path']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                get_model_info(koboldai_vars.model_selected)
    elif(msg['cmd'] == 'delete_model'):
        if "{}/models".format(os.getcwd()) in os.path.abspath(msg['data']) or "{}\\models".format(os.getcwd()) in os.path.abspath(msg['data']):
            if check_if_dir_is_model(msg['data']):
                logger.warning(f"Someone deleted {msg['data']}")
                import shutil
                shutil.rmtree(msg['data'])
                sendModelSelection(menu=msg['menu'])
            else:
                logger.error(f"Someone attempted to delete {msg['data']} but this is not a valid model")
        else:
            logger.critical(f"Someone maliciously attempted to delete {msg['data']}. The attempt has been blocked.")
    elif(msg['cmd'] == 'OAI_Key_Update'):
        get_oai_models({'model': koboldai_vars.model, 'key': msg['key']})
    elif(msg['cmd'] == 'Cluster_Key_Update'):
        get_cluster_models({'model': koboldai_vars.model, 'key': msg['key'], 'url': msg['url']})
    elif(msg['cmd'] == 'loadselect'):
        koboldai_vars.loadselect = msg["data"]
    elif(msg['cmd'] == 'spselect'):
        koboldai_vars.spselect = msg["data"]
    elif(msg['cmd'] == 'loadrequest'):
        loadRequest(fileops.storypath(koboldai_vars.loadselect))
    elif(msg['cmd'] == 'sprequest'):
        spRequest(koboldai_vars.spselect)
    elif(msg['cmd'] == 'deletestory'):
        deletesave(msg['data'])
    elif(msg['cmd'] == 'renamestory'):
        renamesave(msg['data'], msg['newname'])
    elif(msg['cmd'] == 'clearoverwrite'):    
        koboldai_vars.svowname = ""
        koboldai_vars.saveow   = False
    elif(msg['cmd'] == 'seqsel'):
        selectsequence(msg['data'])
    elif(msg['cmd'] == 'seqpin'):
        pinsequence(msg['data'])
    elif(msg['cmd'] == 'setnumseq'):
        koboldai_vars.numseqs = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelnumseq', 'data': msg['data']}, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setwidepth'):
        koboldai_vars.widepth = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelwidepth', 'data': msg['data']}, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setuseprompt'):
        koboldai_vars.useprompt = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setadventure'):
        koboldai_vars.adventure = msg['data']
        koboldai_vars.chatmode = False
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'autosave'):
        koboldai_vars.autosave = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setchatmode'):
        koboldai_vars.chatmode = msg['data']
        koboldai_vars.adventure = False
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setdynamicscan'):
        koboldai_vars.dynamicscan = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setnopromptgen'):
        koboldai_vars.nopromptgen = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setrngpersist'):
        koboldai_vars.rngpersist = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setnogenmod'):
        koboldai_vars.nogenmod = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setfulldeterminism'):
        koboldai_vars.full_determinism = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setoutputstreaming'):
        koboldai_vars.output_streaming = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setshowbudget'):
        koboldai_vars.show_budget = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setshowprobs'):
        koboldai_vars.show_probs = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'alttextgen'):
        koboldai_vars.alt_gen = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'alt_multi_gen'):
        koboldai_vars.alt_multi_gen = msg['data']
        settingschanged()
        refresh_settings()
    elif(not koboldai_vars.host and msg['cmd'] == 'importwi'):
        wiimportrequest()
    elif(msg['cmd'] == 'debug'):
        koboldai_vars.debug = msg['data']
        emit('from_server', {'cmd': 'set_debug', 'data': msg['data']}, broadcast=True, room="UI_1")
        if koboldai_vars.debug:
            send_debug()
    elif(msg['cmd'] == 'getfieldbudget'):
        unencoded = msg["data"]["unencoded"]
        field = msg["data"]["field"]

        # Tokenizer may be undefined here when a model has not been chosen.
        if "tokenizer" not in globals():
            # We don't have a tokenizer, just return nulls.
            emit(
                'from_server',
                {'cmd': 'showfieldbudget', 'data': {"length": None, "max": None, "field": field}},
            )
            return

        header_length = len(tokenizer._koboldai_header)
        max_tokens = koboldai_vars.max_length - header_length - koboldai_vars.sp_length - koboldai_vars.genamt

        if not unencoded:
            # Unencoded is empty, just return 0
            emit(
                'from_server',
                {'cmd': 'showfieldbudget', 'data': {"length": 0, "max": max_tokens, "field": field}},
                broadcast=True
            )
        else:
            if field == "anoteinput":
                unencoded = buildauthorsnote(unencoded, msg["data"]["anotetemplate"])
            tokens_length = len(tokenizer.encode(unencoded))

            emit(
                'from_server',
                {'cmd': 'showfieldbudget', 'data': {"length": tokens_length, "max": max_tokens, "field": field}},
                broadcast=True
            )

#==================================================================#
#  Send userscripts list to client
#==================================================================#
def sendUSStatItems():
    _, loaded = getuslist()
    loaded = loaded if koboldai_vars.lua_running else []
    last_userscripts = [e["filename"] for e in loaded]
    emit('from_server', {'cmd': 'usstatitems', 'data': loaded, 'flash': last_userscripts != koboldai_vars.last_userscripts}, broadcast=True, room="UI_1")
    koboldai_vars.last_userscripts = last_userscripts

#==================================================================#
#  KoboldAI Markup Formatting (Mixture of Markdown and sanitized html)
#==================================================================#
def kml(txt):
   txt = txt.replace('>', '&gt;')
   txt = bleach.clean(markdown.markdown(txt), tags = ['p', 'em', 'strong', 'code', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'ul', 'b', 'i', 'a', 'span', 'button'], styles = ['color', 'font-weight'], attributes=['id', 'class', 'style', 'href'])
   return txt

#==================================================================#
#  Send start message and tell Javascript to set UI state
#==================================================================#
def setStartState():
    # Old UI sets welcome to a boolean sometimes
    koboldai_vars.welcome = koboldai_vars.welcome or koboldai_vars.welcome_default

    if koboldai_vars.welcome != koboldai_vars.welcome_default:
        txt = koboldai_vars.welcome + "<br/>"
    else:
        txt = "<span>Welcome to <span class=\"color_cyan\">KoboldAI</span>! You are running <span class=\"color_green\">"+getmodelname()+"</span>.<br/>"
    if(not koboldai_vars.noai and koboldai_vars.welcome == koboldai_vars.welcome_default):
        txt = txt + "Please load a game or enter a prompt below to begin!</span>"
    if(koboldai_vars.noai):
        txt = txt + "Please load or import a story to read. There is no AI in this mode."
    socketio.emit('from_server', {'cmd': 'updatescreen', 'gamestarted': koboldai_vars.gamestarted, 'data': txt}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'setgamestate', 'data': 'start'}, broadcast=True, room="UI_1")

#==================================================================#
#  Transmit applicable settings to SocketIO to build UI sliders/toggles
#==================================================================#
def sendsettings():
    # Send settings for selected AI type
    socketio.emit('from_server', {'cmd': 'reset_menus'}, room="UI_1")
    if(koboldai_vars.model != "InferKit"):
        for set in gensettings.gensettingstf:
            if 'UI_V2_Only' not in set:
                socketio.emit('from_server', {'cmd': 'addsetting', 'data': set}, room="UI_1")
    else:
        for set in gensettings.gensettingsik:
            if 'UI_V2_Only' not in set:
                socketio.emit('from_server', {'cmd': 'addsetting', 'data': set}, room="UI_1")
    
    # Send formatting options
    for frm in gensettings.formatcontrols:
        socketio.emit('from_server', {'cmd': 'addformat', 'data': frm}, room="UI_1")
        # Add format key to vars if it wasn't loaded with client.settings
        if(not hasattr(koboldai_vars, frm["id"])):
            setattr(koboldai_vars, frm["id"], False)

#==================================================================#
#  Set value of gamesaved
#==================================================================#
def setgamesaved(gamesaved):
    assert type(gamesaved) is bool
    if(gamesaved != koboldai_vars.gamesaved):
        socketio.emit('from_server', {'cmd': 'gamesaved', 'data': gamesaved}, broadcast=True, room="UI_1")
    koboldai_vars.gamesaved = gamesaved

#==================================================================#
#  Take input text from SocketIO and decide what to do with it
#==================================================================#

def check_for_backend_compilation():
    if(koboldai_vars.checking):
        return
    koboldai_vars.checking = True
    for _ in range(31):
        time.sleep(0.06276680299820175)
        if(koboldai_vars.compiling):
            emit('from_server', {'cmd': 'warnmsg', 'data': 'Compiling TPU backend&mdash;this usually takes 1&ndash;2 minutes...'}, broadcast=True, room="UI_1")
            break
    koboldai_vars.checking = False

def actionsubmit(
    data,
    actionmode=0,
    force_submit=False,
    force_prompt_gen=False,
    disable_recentrng=False,
    no_generate=False,
    ignore_aibusy=False,
    gen_mode=GenerationMode.STANDARD
):
    # Ignore new submissions if the AI is currently busy
    if koboldai_vars.aibusy and not ignore_aibusy:
        return

    while(True):
        set_aibusy(1)
        koboldai_vars.actions.clear_unused_options()
        if(koboldai_vars.model in ["API","CLUSTER"]):
            global tokenizer
            if koboldai_vars.model == "API":
                tokenizer_id = requests.get(
                    koboldai_vars.colaburl[:-8] + "/api/v1/model",
                ).json()["result"]
            elif len(koboldai_vars.cluster_requested_models) >= 1:
                # If the player has requested one or more models, we use the first one for the tokenizer
                tokenizer_id = koboldai_vars.cluster_requested_models[0]
            # The cluster can return any number of possible models for each gen, but this happens after this step
            # So at this point, this is unknown
            else:
                tokenizer_id = ""
            if tokenizer_id != koboldai_vars.api_tokenizer_id:
                try:
                    if(os.path.isdir(tokenizer_id)):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                        except:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=koboldai_vars.revision, cache_dir="cache")
                    elif(os.path.isdir("models/{}".format(tokenizer_id.replace('/', '_')))):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(tokenizer_id.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                        except:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(tokenizer_id.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
                    else:
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                        except:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=koboldai_vars.revision, cache_dir="cache")
                except:
                    logger.warning(f"Unknown tokenizer {repr(tokenizer_id)}")
                koboldai_vars.api_tokenizer_id = tokenizer_id

        if(disable_recentrng):
            koboldai_vars.recentrng = koboldai_vars.recentrngm = None

        koboldai_vars.recentback = False
        koboldai_vars.recentedit = False
        koboldai_vars.actionmode = actionmode

        # "Action" mode
        if(actionmode == 1):
            data = data.strip().lstrip('>')
            data = re.sub(r'\n+', ' ', data)
            if(len(data)):
                data = f"\n\n> {data}\n"
        
        # "Chat" mode
        if(koboldai_vars.chatmode and koboldai_vars.gamestarted):
            if(koboldai_vars.botname):
                botname = (koboldai_vars.botname + ":")
            else:
                botname = ""
            data = re.sub(r'\n+\Z', '', data)
            if(len(data)):
                data = f"\n{koboldai_vars.chatname}: {data}\n{botname}"
        
        # If we're not continuing, store a copy of the raw input
        if(data != ""):
            koboldai_vars.lastact = data
        
        if(not koboldai_vars.gamestarted):
            koboldai_vars.submission = data
            if(not no_generate):
                execute_inmod()
            koboldai_vars.submission = re.sub(r"[^\S\r\n]*([\r\n]*)$", r"\1", koboldai_vars.submission)  # Remove trailing whitespace, excluding newlines
            data = koboldai_vars.submission
            if koboldai_vars.prompt:
                # This will happen only when loading a non-started save game (i.e. one consisting only of a saved prompt).
                # In this case, prepend the prompt to the sumission. This allows the player to submit a blank initial submission
                # (preserving the prompt), or to add to the prompt by submitting non-empty data.
                data = koboldai_vars.prompt + data
            if(not force_submit and len(data.strip()) == 0):
                set_aibusy(0)
                socketio.emit("error", "No prompt or random story theme entered", broadcast=True, room="UI_2")
                assert False
            # Start the game
            koboldai_vars.gamestarted = True
            if(not koboldai_vars.noai and koboldai_vars.lua_koboldbridge.generating and (not koboldai_vars.nopromptgen or force_prompt_gen)):
                # Save this first action as the prompt
                koboldai_vars.prompt = data
                # Clear the startup text from game screen
                emit('from_server', {'cmd': 'updatescreen', 'gamestarted': False, 'data': 'Please wait, generating story...'}, broadcast=True, room="UI_1")
                calcsubmit("", gen_mode=gen_mode) # Run the first action through the generator
                if(not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None and len(koboldai_vars.genseqs) == 0):
                    data = ""
                    force_submit = True
                    disable_recentrng = True
                    continue
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True, room="UI_1")
                break
            else:
                # Save this first action as the prompt
                koboldai_vars.prompt = data if len(data) > 0 else '"'
                for i in range(koboldai_vars.numseqs):
                    koboldai_vars.lua_koboldbridge.outputs[i+1] = ""
                if(not no_generate):
                    execute_outmod()
                koboldai_vars.lua_koboldbridge.regeneration_required = False
                genout = []
                for i in range(koboldai_vars.numseqs):
                    genout.append({"generated_text": koboldai_vars.lua_koboldbridge.outputs[i+1]})
                    assert type(genout[-1]["generated_text"]) is str
                koboldai_vars.actions.append_options([utils.applyoutputformatting(x["generated_text"]) for x in genout])
                genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
                if(len(genout) == 1):
                    genresult(genout[0]["generated_text"], flash=False)
                    refresh_story()
                    if(len(koboldai_vars.actions) > 0):
                        emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.actions.get_last_key() + 1}, broadcast=True, room="UI_1")
                    if(not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None):
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                else:
                    if(not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None and koboldai_vars.lua_koboldbridge.restart_sequence > 0):
                        genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence-1]["generated_text"], flash=False)
                        refresh_story()
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                    genselect(genout)
                    refresh_story()
                set_aibusy(0)
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True, room="UI_1")
                break
        else:
            # Apply input formatting & scripts before sending to tokenizer
            if(koboldai_vars.actionmode == 0):
                data = applyinputformatting(data)
            koboldai_vars.submission = data
            if(not no_generate):
                execute_inmod()
            koboldai_vars.submission = re.sub(r"[^\S\r\n]*([\r\n]*)$", r"\1", koboldai_vars.submission)  # Remove trailing whitespace, excluding newlines
            data = koboldai_vars.submission
            # Dont append submission if it's a blank/continue action
            if(data != ""):
                # Store the result in the Action log
                if(len(koboldai_vars.prompt.strip()) == 0):
                    koboldai_vars.prompt = data
                else:
                    koboldai_vars.actions.append(data, submission=True)
                update_story_chunk('last')
                send_debug()

            if(not no_generate and not koboldai_vars.noai and koboldai_vars.lua_koboldbridge.generating):
                # Off to the tokenizer!
                calcsubmit("", gen_mode=gen_mode)
                if(not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None and len(koboldai_vars.genseqs) == 0):
                    data = ""
                    force_submit = True
                    disable_recentrng = True
                    continue
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True, room="UI_1")
                break
            else:
                if(not no_generate):
                    for i in range(koboldai_vars.numseqs):
                        koboldai_vars.lua_koboldbridge.outputs[i+1] = ""
                    execute_outmod()
                    koboldai_vars.lua_koboldbridge.regeneration_required = False
                genout = []
                for i in range(koboldai_vars.numseqs):
                    genout.append({"generated_text": koboldai_vars.lua_koboldbridge.outputs[i+1] if not no_generate else ""})
                    assert type(genout[-1]["generated_text"]) is str
                koboldai_vars.actions.append_options([utils.applyoutputformatting(x["generated_text"]) for x in genout])
                genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
                if(len(genout) == 1):
                    genresult(genout[0]["generated_text"])
                    if(not no_generate and not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None):
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                else:
                    if(not no_generate and not koboldai_vars.abort and koboldai_vars.lua_koboldbridge.restart_sequence is not None and koboldai_vars.lua_koboldbridge.restart_sequence > 0):
                        genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                    genselect(genout)
                set_aibusy(0)
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True, room="UI_1")
                break

def apiactionsubmit_generate(txt, minimum, maximum):
    koboldai_vars.generated_tkns = 0

    if not koboldai_vars.quiet:
        logger.debug(f"Prompt Min:{minimum}, Max:{maximum}")
        logger.prompt(utils.decodenewlines(tokenizer.decode(txt)).encode("unicode_escape").decode("utf-8"))

    # Clear CUDA cache if using GPU
    if(koboldai_vars.hascuda and (koboldai_vars.usegpu or koboldai_vars.breakmodel)):
        gc.collect()
        torch.cuda.empty_cache()

    # Submit input text to generator
    _genout, already_generated = tpool.execute(model.core_generate, txt, set())

    genout = [utils.applyoutputformatting(utils.decodenewlines(tokenizer.decode(tokens[-already_generated:]))) for tokens in _genout]

    # Clear CUDA cache again if using GPU
    if(koboldai_vars.hascuda and (koboldai_vars.usegpu or koboldai_vars.breakmodel)):
        del _genout
        gc.collect()
        torch.cuda.empty_cache()

    return genout

def apiactionsubmit_tpumtjgenerate(txt, minimum, maximum):
    koboldai_vars.generated_tkns = 0

    if(koboldai_vars.full_determinism):
        tpu_mtj_backend.set_rng_seed(koboldai_vars.seed)

    if not koboldai_vars.quiet:
        logger.debug(f"Prompt Min:{minimum}, Max:{maximum}")
        logger.prompt(utils.decodenewlines(tokenizer.decode(txt)).encode("unicode_escape").decode("utf-8"))

    koboldai_vars._prompt = koboldai_vars.prompt

    # Submit input text to generator
    soft_tokens = model.get_soft_tokens()
    genout = tpool.execute(
        tpu_mtj_backend.infer_static,
        np.uint32(txt),
        gen_len = maximum-minimum+1,
        temp=koboldai_vars.temp,
        top_p=koboldai_vars.top_p,
        top_k=koboldai_vars.top_k,
        tfs=koboldai_vars.tfs,
        typical=koboldai_vars.typical,
        top_a=koboldai_vars.top_a,
        numseqs=koboldai_vars.numseqs,
        repetition_penalty=koboldai_vars.rep_pen,
        rpslope=koboldai_vars.rep_pen_slope,
        rprange=koboldai_vars.rep_pen_range,
        soft_embeddings=koboldai_vars.sp,
        soft_tokens=soft_tokens,
        sampler_order=koboldai_vars.sampler_order,
    )
    genout = np.array(genout)
    genout = [utils.applyoutputformatting(utils.decodenewlines(tokenizer.decode(txt))) for txt in genout]

    return genout

def apiactionsubmit(data, use_memory=False, use_world_info=False, use_story=False, use_authors_note=False):
    if not model or not model.capabilties.api_host:
        raise NotImplementedError(f"API generation isn't allowed on model '{koboldai_vars.model}'")

    data = applyinputformatting(data)

    if(koboldai_vars.memory != "" and koboldai_vars.memory[-1] != "\n"):
        mem = koboldai_vars.memory + "\n"
    else:
        mem = koboldai_vars.memory

    if(use_authors_note and koboldai_vars.authornote != ""):
        anotetxt  = ("\n" + koboldai_vars.authornotetemplate + "\n").replace("<|>", koboldai_vars.authornote)
    else:
        anotetxt = ""

    MIN_STORY_TOKENS = 8
    story_tokens = []
    mem_tokens = []
    wi_tokens = []

    story_budget = lambda: koboldai_vars.max_length - koboldai_vars.sp_length - koboldai_vars.genamt - len(tokenizer._koboldai_header) - len(story_tokens) - len(mem_tokens) - len(wi_tokens)
    budget = lambda: story_budget() + MIN_STORY_TOKENS
    if budget() < 0:
        abort(Response(json.dumps({"detail": {
            "msg": f"Your Max Tokens setting is too low for your current soft prompt and tokenizer to handle. It needs to be at least {koboldai_vars.max_length - budget()}.",
            "type": "token_overflow",
        }}), mimetype="application/json", status=500))

    if use_memory:
        mem_tokens = tokenizer.encode(utils.encodenewlines(mem))[-budget():]

    if use_world_info:
        #world_info, _ = checkworldinfo(data, force_use_txt=True, scan_story=use_story)
        world_info = koboldai_vars.worldinfo_v2.get_used_wi()
        wi_tokens = tokenizer.encode(utils.encodenewlines(world_info))[-budget():]

    if use_story:
        if koboldai_vars.useprompt:
            story_tokens = tokenizer.encode(utils.encodenewlines(koboldai_vars.prompt))[-budget():]

    story_tokens = tokenizer.encode(utils.encodenewlines(data))[-story_budget():] + story_tokens

    if use_story:
        for i, action in enumerate(reversed(koboldai_vars.actions.values())):
            if story_budget() <= 0:
                assert story_budget() == 0
                break
            story_tokens = tokenizer.encode(utils.encodenewlines(action))[-story_budget():] + story_tokens
            if i == koboldai_vars.andepth - 1:
                story_tokens = tokenizer.encode(utils.encodenewlines(anotetxt))[-story_budget():] + story_tokens
        if not koboldai_vars.useprompt:
            story_tokens = tokenizer.encode(utils.encodenewlines(koboldai_vars.prompt))[-budget():] + story_tokens

    tokens = tokenizer._koboldai_header + mem_tokens + wi_tokens + story_tokens
    assert story_budget() >= 0
    minimum = len(tokens) + 1
    maximum = len(tokens) + koboldai_vars.genamt

    if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ["Colab", "API", "CLUSTER", "OAI", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        genout = apiactionsubmit_generate(tokens, minimum, maximum)
    elif(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        genout = apiactionsubmit_tpumtjgenerate(tokens, minimum, maximum)

    return genout

#==================================================================#
#  
#==================================================================#
def actionretry(data):
    if(koboldai_vars.noai):
        emit('from_server', {'cmd': 'errmsg', 'data': "Retry function unavailable in Read Only mode."}, room="UI_1")
        return
    if(koboldai_vars.recentrng is not None):
        if(not koboldai_vars.aibusy):
            randomGameRequest(koboldai_vars.recentrng, memory=koboldai_vars.recentrngm)
        return
    if actionback():
        actionsubmit("", actionmode=koboldai_vars.actionmode, force_submit=True)
        send_debug()
    elif(not koboldai_vars.useprompt):
        emit('from_server', {'cmd': 'errmsg', 'data': "Please enable \"Always Add Prompt\" to retry with your prompt."}, room="UI_1")

#==================================================================#
#  
#==================================================================#
def actionback():
    if(koboldai_vars.aibusy):
        return
    # Remove last index of actions and refresh game screen
    if(len(koboldai_vars.genseqs) == 0 and len(koboldai_vars.actions) > 0):
        last_key = koboldai_vars.actions.get_last_key()
        koboldai_vars.actions.pop()
        koboldai_vars.recentback = True
        remove_story_chunk(last_key + 1)
        success = True
    elif(len(koboldai_vars.genseqs) == 0):
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."}, room="UI_1")
        success =  False
    else:
        koboldai_vars.genseqs = []
        success = True
    send_debug()
    return success
        
def actionredo():
    genout = [[x['text'], "redo" if x['Previous Selection'] else "pinned" if x['Pinned'] else "normal"] for x in koboldai_vars.actions.get_redo_options()]
    if len(genout) == 0:
        emit('from_server', {'cmd': 'popuperror', 'data': "There's nothing to redo"}, broadcast=True, room="UI_1")
    elif len(genout) == 1:
        genresult(genout[0][0], flash=True, ignore_formatting=True)
    else:
        koboldai_vars.genseqs = [{"generated_text": x[0]} for x in genout]
        emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#  
#==================================================================#
def buildauthorsnote(authorsnote, template):
    # Build Author's Note if set
    if authorsnote == "":
        return ""
    return ("\n" + template + "\n").replace("<|>", authorsnote)

def calcsubmitbudgetheader(txt, **kwargs):
    # Scan for WorldInfo matches
    winfo, found_entries = checkworldinfo(txt, **kwargs)

    # Add a newline to the end of memory
    if(koboldai_vars.memory != "" and koboldai_vars.memory[-1] != "\n"):
        mem = koboldai_vars.memory + "\n"
    else:
        mem = koboldai_vars.memory

    anotetxt = buildauthorsnote(koboldai_vars.authornote, koboldai_vars.authornotetemplate)

    return winfo, mem, anotetxt, found_entries

def calcsubmitbudget(actionlen, winfo, mem, anotetxt, actions, submission=None, budget_deduction=0):
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    anotetkns    = []  # Placeholder for Author's Note tokens
    lnanote      = 0   # Placeholder for Author's Note length

    lnsp = koboldai_vars.sp_length

    if("tokenizer" not in globals()):
        from transformers import GPT2Tokenizer
        global tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")

    lnheader = len(tokenizer._koboldai_header)

    # Calculate token budget
    prompttkns = tokenizer.encode(utils.encodenewlines(koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt)), max_length=int(2e9), truncation=True)
    lnprompt   = len(prompttkns)

    memtokens = tokenizer.encode(utils.encodenewlines(mem), max_length=int(2e9), truncation=True)
    lnmem     = len(memtokens)
    if(lnmem > koboldai_vars.max_length - lnheader - lnsp - koboldai_vars.genamt - budget_deduction):
        raise OverflowError("The memory in your story is too long. Please either write a shorter memory text or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    witokens  = tokenizer.encode(utils.encodenewlines(winfo), max_length=int(2e9), truncation=True)
    lnwi      = len(witokens)
    if(lnmem + lnwi > koboldai_vars.max_length - lnheader - lnsp - koboldai_vars.genamt - budget_deduction):
        raise OverflowError("The current active world info keys take up too many tokens. Please either write shorter world info, decrease World Info Depth or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    if(anotetxt != ""):
        anotetkns = tokenizer.encode(utils.encodenewlines(anotetxt), max_length=int(2e9), truncation=True)
        lnanote   = len(anotetkns)
        if(lnmem + lnwi + lnanote > koboldai_vars.max_length - lnheader - lnsp - koboldai_vars.genamt - budget_deduction):
            raise OverflowError("The author's note in your story is too long. Please either write a shorter author's note or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    if(koboldai_vars.useprompt):
        budget = koboldai_vars.max_length - lnsp - lnprompt - lnmem - lnanote - lnwi - koboldai_vars.genamt - budget_deduction
    else:
        budget = koboldai_vars.max_length - lnheader - lnsp - lnmem - lnanote - lnwi - koboldai_vars.genamt - budget_deduction

    lnsubmission = len(tokenizer.encode(utils.encodenewlines(koboldai_vars.comregex_ai.sub('', submission)), max_length=int(2e9), truncation=True)) if submission is not None else 0
    maybe_lnprompt = lnprompt if koboldai_vars.useprompt and actionlen > 0 else 0

    if(lnmem + lnwi + lnanote + maybe_lnprompt + lnsubmission > koboldai_vars.max_length - lnheader - lnsp - koboldai_vars.genamt - budget_deduction):
        raise OverflowError("Your submission is too long. Please either write a shorter submission or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt. If you are using the Always Add Prompt setting, turning it off may help.")

    assert budget >= 0

    if(actionlen == 0):
        # First/Prompt action
        tokens = (tokenizer._koboldai_header if koboldai_vars.model not in ("Colab", "API", "CLUSTER", "OAI") else []) + memtokens + witokens + anotetkns + prompttkns
        assert len(tokens) <= koboldai_vars.max_length - lnsp - koboldai_vars.genamt - budget_deduction
        ln = len(tokens) + lnsp
        return tokens, ln+1, ln+koboldai_vars.genamt
    else:
        tokens     = []
        
        # Check if we have the action depth to hit our A.N. depth
        if(anotetxt != "" and actionlen < koboldai_vars.andepth):
            forceanote = True
        
        # Get most recent action tokens up to our budget
        n = 0
        for key in reversed(actions):
            chunk = koboldai_vars.comregex_ai.sub('', actions[key])
            
            assert budget >= 0
            if(budget <= 0):
                break
            acttkns = tokenizer.encode(utils.encodenewlines(chunk), max_length=int(2e9), truncation=True)
            tknlen = len(acttkns)
            if(tknlen < budget):
                tokens = acttkns + tokens
                budget -= tknlen
            else:
                count = budget * -1
                truncated_action_tokens = acttkns[count:]
                tokens = truncated_action_tokens + tokens
                budget = 0
                break
            
            # Inject Author's Note if we've reached the desired depth
            if(n == koboldai_vars.andepth-1):
                if(anotetxt != ""):
                    tokens = anotetkns + tokens # A.N. len already taken from bdgt
                    anoteadded = True
            n += 1
        
        # If we're not using the prompt every time and there's still budget left,
        # add some prompt.
        if(not koboldai_vars.useprompt):
            if(budget > 0):
                prompttkns = prompttkns[-budget:]
            else:
                prompttkns = []

        # Did we get to add the A.N.? If not, do it here
        if(anotetxt != ""):
            if((not anoteadded) or forceanote):
                tokens = (tokenizer._koboldai_header if koboldai_vars.model not in ("Colab", "API", "CLUSTER", "OAI") else []) + memtokens + witokens + anotetkns + prompttkns + tokens
            else:
                tokens = (tokenizer._koboldai_header if koboldai_vars.model not in ("Colab", "API", "CLUSTER", "OAI") else []) + memtokens + witokens + prompttkns + tokens
        else:
            # Prepend Memory, WI, and Prompt before action tokens
            tokens = (tokenizer._koboldai_header if koboldai_vars.model not in ("Colab", "API", "CLUSTER", "OAI") else []) + memtokens + witokens + prompttkns + tokens

        # Send completed bundle to generator
        assert len(tokens) <= koboldai_vars.max_length - lnsp - koboldai_vars.genamt - budget_deduction
        ln = len(tokens) + lnsp

        return tokens, ln+1, ln+koboldai_vars.genamt

#==================================================================#
# Take submitted text and build the text to be given to generator
#==================================================================#
def calcsubmit(txt, gen_mode=GenerationMode.STANDARD):
    anotetxt     = ""    # Placeholder for Author's Note text
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    actionlen    = len(koboldai_vars.actions)

    #winfo, mem, anotetxt, found_entries = calcsubmitbudgetheader(txt)
 
    # For all transformers models
    if(koboldai_vars.model != "InferKit"):
        #subtxt, min, max = calcsubmitbudget(actionlen, winfo, mem, anotetxt, koboldai_vars.actions, submission=txt)
        start_time = time.time()
        subtxt, min, max, found_entries  = koboldai_vars.calc_ai_text(submitted_text=txt)
        logger.debug("Submit: get_text time {}s".format(time.time()-start_time))

        start_time = time.time()
        if koboldai_vars.experimental_features and any([c.get("attention_multiplier", 1) != 1 for c in koboldai_vars.context]):
            offset = 0
            applied_biases = []
            for c in koboldai_vars.context:
                length = len(c["tokens"])
                if c.get("attention_multiplier") and c["attention_multiplier"] != 1:
                    applied_biases.append({"start": offset, "end": offset + length, "multiplier": c.get("attention_multiplier", 1)})
                offset += length

            logger.info(f"Applied Biases: {applied_biases}")

            bias = []
            for b in applied_biases:
                for i in range(b["start"], b["end"]):
                    top_index = len(bias) - 1
                    if i > top_index:
                        bias += [1] * (i - top_index)
                    bias[i] = b["multiplier"]

            
            device = model.get_auxilary_device()
            attention_bias.attention_bias = torch.Tensor(bias).to(device)
            logger.info(f"Bias by {koboldai_vars.memory_attn_bias} -- {attention_bias.attention_bias}")
        logger.debug("Submit: experimental_features time {}s".format(time.time()-start_time))
        
        start_time = time.time()
        generate(subtxt, min, max, found_entries, gen_mode=gen_mode)
        logger.debug("Submit: generate time {}s".format(time.time()-start_time))
        attention_bias.attention_bias = None

                    
    # For InferKit web API
    else:
        # Check if we have the action depth to hit our A.N. depth
        if(anotetxt != "" and actionlen < koboldai_vars.andepth):
            forceanote = True
        
        if(koboldai_vars.useprompt):
            budget = koboldai_vars.ikmax - len(koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt)) - len(anotetxt) - len(mem) - len(winfo) - 1
        else:
            budget = koboldai_vars.ikmax - len(anotetxt) - len(mem) - len(winfo) - 1
            
        subtxt = ""
        prompt = koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt)
        n = 0
        for key in reversed(koboldai_vars.actions):
            chunk = koboldai_vars.actions[key]
            
            if(budget <= 0):
                    break
            actlen = len(chunk)
            if(actlen < budget):
                subtxt = chunk + subtxt
                budget -= actlen
            else:
                count = budget * -1
                subtxt = chunk[count:] + subtxt
                budget = 0
                break
            
            # If we're not using the prompt every time and there's still budget left,
            # add some prompt.
            if(not koboldai_vars.useprompt):
                if(budget > 0):
                    prompt = koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt)[-budget:]
                else:
                    prompt = ""
            
            # Inject Author's Note if we've reached the desired depth
            if(n == koboldai_vars.andepth-1):
                if(anotetxt != ""):
                    subtxt = anotetxt + subtxt # A.N. len already taken from bdgt
                    anoteadded = True
            n += 1
        
        # Did we get to add the A.N.? If not, do it here
        if(anotetxt != ""):
            if((not anoteadded) or forceanote):
                subtxt = mem + winfo + anotetxt + prompt + subtxt
            else:
                subtxt = mem + winfo + prompt + subtxt
        else:
            subtxt = mem + winfo + prompt + subtxt
        
        # Send it!
        ikrequest(subtxt)

class HordeException(Exception):
    pass

#==================================================================#
# Send text to generator and deal with output
#==================================================================#

def generate(txt, minimum, maximum, found_entries=None, gen_mode=GenerationMode.STANDARD):
    # Open up token stream
    emit("stream_tokens", True, broadcast=True, room="UI_2")

    # HACK: Show options when streaming more than 1 sequence
    if utils.koboldai_vars.output_streaming:
        koboldai_vars.actions.show_options(koboldai_vars.numseqs > 1, force=True)

    koboldai_vars.generated_tkns = 0

    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(koboldai_vars.numseqs))

    if not koboldai_vars.quiet:
        logger.debug(f"Prompt Min:{minimum}, Max:{maximum}")
        logger.prompt(utils.decodenewlines(model.tokenizer.decode(txt)).encode("unicode_escape").decode("utf-8"))

    # Store context in memory to use it for comparison with generated content
    koboldai_vars.lastctx = utils.decodenewlines(tokenizer.decode(txt))

    # Clear CUDA cache if using GPU
    if(koboldai_vars.hascuda and (koboldai_vars.usegpu or koboldai_vars.breakmodel)):
        gc.collect()
        torch.cuda.empty_cache()

    # Submit input text to generator
    try:
        start_time = time.time()
        genout, already_generated = tpool.execute(model.core_generate, txt, found_entries, gen_mode=gen_mode)
        logger.debug("Generate: core_generate time {}s".format(time.time()-start_time))
    except Exception as e:
        if(issubclass(type(e), lupa.LuaError)):
            koboldai_vars.lua_koboldbridge.obliterate_multiverse()
            koboldai_vars.lua_running = False
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
            sendUSStatItems()
            logger.error('LUA ERROR: ' + str(e).replace("\033", ""))
            logger.warning("Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.")
            socketio.emit("error", str(e), broadcast=True, room="UI_2")
        else:
            emit('from_server', {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'}, broadcast=True, room="UI_1")
            logger.error(traceback.format_exc().replace("\033", ""))
            socketio.emit("error", str(e), broadcast=True, room="UI_2")

        set_aibusy(0)
        # Clean up token stream
        emit("stream_tokens", None, broadcast=True, room="UI_2")
        return

    for i in range(koboldai_vars.numseqs):
        if len(genout[i]) > 0:
            koboldai_vars.lua_koboldbridge.generated[i+1][koboldai_vars.generated_tkns] = int(genout[i, -1].item())
        koboldai_vars.lua_koboldbridge.outputs[i+1] = utils.decodenewlines(tokenizer.decode(genout[i, -already_generated:]))

    execute_outmod()
    if(koboldai_vars.lua_koboldbridge.regeneration_required):
        koboldai_vars.lua_koboldbridge.regeneration_required = False
        genout = []
        for i in range(koboldai_vars.numseqs):
            genout.append({"generated_text": koboldai_vars.lua_koboldbridge.outputs[i+1]})
            assert type(genout[-1]["generated_text"]) is str
    else:
        genout = [{"generated_text": utils.decodenewlines(tokenizer.decode(tokens[-already_generated:]))} for tokens in genout]
    
    if(len(genout) == 1):
        genresult(genout[0]["generated_text"])
    else:
        koboldai_vars.actions.append_options([utils.applyoutputformatting(x["generated_text"]) for x in genout])
        genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
        if(koboldai_vars.lua_koboldbridge.restart_sequence is not None and koboldai_vars.lua_koboldbridge.restart_sequence > 0):
            genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
        else:
            genselect(genout)
    
    # Clear CUDA cache again if using GPU
    if(koboldai_vars.hascuda and (koboldai_vars.usegpu or koboldai_vars.breakmodel)):
        del genout
        gc.collect()
        torch.cuda.empty_cache()

    # Clean up token stream
    emit("stream_tokens", None, broadcast=True, room="UI_2")

    maybe_review_story()

    set_aibusy(0)

#==================================================================#
#  Deal with a single return sequence from generate()
#==================================================================#
def genresult(genout, flash=True, ignore_formatting=False):
    # Format output before continuing
    if not ignore_formatting:
        genout = utils.applyoutputformatting(genout)

    if not koboldai_vars.quiet:
        logger.generation(genout.encode("unicode_escape").decode("utf-8"))

    koboldai_vars.lua_koboldbridge.feedback = genout

    if(len(genout) == 0):
        return
    
    # Add formatted text to Actions array and refresh the game screen
    if(len(koboldai_vars.prompt.strip()) == 0):
        koboldai_vars.prompt = genout
    else:
        koboldai_vars.actions.append(genout)
    update_story_chunk('last')
    if(flash):
        emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.actions.get_last_key() + 1 if len(koboldai_vars.actions) else 0}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#  Send generator sequences to the UI for selection
#==================================================================#
def genselect(genout):
    i = 0
    for result in genout:
        # Apply output formatting rules to sequences
        result["generated_text"] = utils.applyoutputformatting(result["generated_text"])
        if not koboldai_vars.quiet:
            logger.info(f"Generation Result {i}")
            logger.generation(result["generated_text"].encode("unicode_escape").decode("utf-8"))
        i += 1
    
    
    # Store sequences in memory until selection is made
    koboldai_vars.genseqs = genout
    
    
    genout = koboldai_vars.actions.get_current_options_no_edits(ui=1)

    # Send sequences to UI for selection
    emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#  Send selected sequence to action log and refresh UI
#==================================================================#
def selectsequence(n):
    if(len(koboldai_vars.genseqs) == 0):
        return
    koboldai_vars.lua_koboldbridge.feedback = koboldai_vars.genseqs[int(n)]["generated_text"]
    if(len(koboldai_vars.lua_koboldbridge.feedback) != 0):
        koboldai_vars.actions.append(koboldai_vars.lua_koboldbridge.feedback)
        update_story_chunk('last')
        emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.actions.get_last_key() + 1 if len(koboldai_vars.actions) else 0}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True, room="UI_1")
    koboldai_vars.genseqs = []

    if(koboldai_vars.lua_koboldbridge.restart_sequence is not None):
        actionsubmit("", actionmode=koboldai_vars.actionmode, force_submit=True, disable_recentrng=True)
    send_debug()

#==================================================================#
#  Pin/Unpin the selected sequence
#==================================================================#
def pinsequence(n):
    if n.isnumeric():
        koboldai_vars.actions.toggle_pin(koboldai_vars.actions.get_last_key()+1, int(n))
        text = koboldai_vars.genseqs[int(n)]['generated_text']
    send_debug()

#==================================================================#
# Replaces returns and newlines with HTML breaks
#==================================================================#
def formatforhtml(txt):
    return txt.replace("\\r\\n", "<br/>").replace("\\r", "<br/>").replace("\\n", "<br/>").replace("\r\n", "<br/>").replace('\n', '<br/>').replace('\r', '<br/>').replace('&lt;/s&gt;', '<br/>')

#==================================================================#
# Strips submitted text from the text returned by the AI
#==================================================================#
def getnewcontent(txt):
    # If the submitted context was blank, then everything is new
    if(koboldai_vars.lastctx == ""):
        return txt
    
    # Tokenize the last context and the generated content
    ctxtokens = tokenizer.encode(utils.encodenewlines(koboldai_vars.lastctx), max_length=int(2e9), truncation=True)
    txttokens = tokenizer.encode(utils.encodenewlines(txt), max_length=int(2e9), truncation=True)
    dif       = (len(txttokens) - len(ctxtokens)) * -1
    
    # Remove the context from the returned text
    newtokens = txttokens[dif:]
    
    return utils.decodenewlines(tokenizer.decode(newtokens))

#==================================================================#
# Applies chosen formatting options to text submitted to AI
#==================================================================#
def applyinputformatting(txt):
    # Add sentence spacing
    if(koboldai_vars.frmtadsnsp and not koboldai_vars.chatmode):
        txt = utils.addsentencespacing(txt, koboldai_vars)
 
    return txt

#==================================================================#
# Sends the current story content to the Game Screen
#==================================================================#
def refresh_story():
    text_parts = ['<chunk n="0" id="n0" tabindex="-1">', koboldai_vars.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), html.escape(koboldai_vars.prompt)), '</chunk>']
    for idx in koboldai_vars.actions:
        item = koboldai_vars.actions[idx]
        idx += 1
        item = html.escape(item)
        item = koboldai_vars.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), item)  # Add special formatting to comments
        item = koboldai_vars.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions
        text_parts.extend(('<chunk n="', str(idx), '" id="n', str(idx), '" tabindex="-1">', item, '</chunk>'))
    emit('from_server', {'cmd': 'updatescreen', 'gamestarted': koboldai_vars.gamestarted, 'data': formatforhtml(''.join(text_parts))}, broadcast=True, room="UI_1")


#==================================================================#
# Signals the Game Screen to update one of the chunks
#==================================================================#
def update_story_chunk(idx: Union[int, str]):
    if idx == 'last':
        if len(koboldai_vars.actions) <= 1:
            # In this case, we are better off just refreshing the whole thing as the
            # prompt might not have been shown yet (with a "Generating story..."
            # message instead).
            refresh_story()
            setgamesaved(False)
            return

        idx = (koboldai_vars.actions.get_last_key() if len(koboldai_vars.actions) else 0) + 1

    if idx == 0:
        text = koboldai_vars.prompt
    else:
        # Actions are 0 based, but in chunks 0 is the prompt.
        # So the chunk index is one more than the corresponding action index.
        if(idx - 1 not in koboldai_vars.actions):
            return
        text = koboldai_vars.actions[idx - 1]

    item = html.escape(text)
    item = koboldai_vars.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), item)  # Add special formatting to comments
    item = koboldai_vars.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions

    chunk_text = f'<chunk n="{idx}" id="n{idx}" tabindex="-1">{formatforhtml(item)}</chunk>'
    emit('from_server', {'cmd': 'updatechunk', 'data': {'index': idx, 'html': chunk_text}}, broadcast=True, room="UI_1")

    setgamesaved(False)

    

#==================================================================#
# Signals the Game Screen to remove one of the chunks
#==================================================================#
def remove_story_chunk(idx: int):
    emit('from_server', {'cmd': 'removechunk', 'data': idx}, broadcast=True, room="UI_1")
    setgamesaved(False)


#==================================================================#
# Sends the current generator settings to the Game Menu
#==================================================================#
def refresh_settings():
    # Suppress toggle change events while loading state
    socketio.emit('from_server', {'cmd': 'allowtoggle', 'data': False}, broadcast=True, room="UI_1")
    
    if(koboldai_vars.model != "InferKit"):
        socketio.emit('from_server', {'cmd': 'updatetemp', 'data': koboldai_vars.temp}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updatetopp', 'data': koboldai_vars.top_p}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updatetopk', 'data': koboldai_vars.top_k}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updatetfs', 'data': koboldai_vars.tfs}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updatetypical', 'data': koboldai_vars.typical}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updatetopa', 'data': koboldai_vars.top_a}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updatereppen', 'data': koboldai_vars.rep_pen}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updatereppenslope', 'data': koboldai_vars.rep_pen_slope}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updatereppenrange', 'data': koboldai_vars.rep_pen_range}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updateoutlen', 'data': koboldai_vars.genamt}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updatetknmax', 'data': koboldai_vars.max_length}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updatenumseq', 'data': koboldai_vars.numseqs}, broadcast=True, room="UI_1")
    else:
        socketio.emit('from_server', {'cmd': 'updatetemp', 'data': koboldai_vars.temp}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updatetopp', 'data': koboldai_vars.top_p}, broadcast=True, room="UI_1")
        socketio.emit('from_server', {'cmd': 'updateikgen', 'data': koboldai_vars.ikgen}, broadcast=True, room="UI_1")
    
    socketio.emit('from_server', {'cmd': 'updateanotedepth', 'data': koboldai_vars.andepth}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatewidepth', 'data': koboldai_vars.widepth}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updateuseprompt', 'data': koboldai_vars.useprompt}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updateadventure', 'data': koboldai_vars.adventure}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatechatmode', 'data': koboldai_vars.chatmode}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatedynamicscan', 'data': koboldai_vars.dynamicscan}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updateautosave', 'data': koboldai_vars.autosave}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatenopromptgen', 'data': koboldai_vars.nopromptgen}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updaterngpersist', 'data': koboldai_vars.rngpersist}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatenogenmod', 'data': koboldai_vars.nogenmod}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatefulldeterminism', 'data': koboldai_vars.full_determinism}, broadcast=True, room="UI_1")
    
    socketio.emit('from_server', {'cmd': 'updatefrmttriminc', 'data': koboldai_vars.frmttriminc}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatefrmtrmblln', 'data': koboldai_vars.frmtrmblln}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatefrmtrmspch', 'data': koboldai_vars.frmtrmspch}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatefrmtadsnsp', 'data': koboldai_vars.frmtadsnsp}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatesingleline', 'data': koboldai_vars.singleline}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updateoutputstreaming', 'data': koboldai_vars.output_streaming}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updateshowbudget', 'data': koboldai_vars.show_budget}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updateshowprobs', 'data': koboldai_vars.show_probs}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatealt_text_gen', 'data': koboldai_vars.alt_gen}, broadcast=True, room="UI_1")
    socketio.emit('from_server', {'cmd': 'updatealt_multi_gen', 'data': koboldai_vars.alt_multi_gen}, broadcast=True, room="UI_1")
    
    # Allow toggle events again
    socketio.emit('from_server', {'cmd': 'allowtoggle', 'data': True}, broadcast=True, room="UI_1")

#==================================================================#
#  Sets the logical and display states for the AI Busy condition
#==================================================================#
def set_aibusy(state):
    if(koboldai_vars.disable_set_aibusy):
        return
    if(state):
        koboldai_vars.aibusy = True
        emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'}, broadcast=True, room="UI_1")
    else:
        koboldai_vars.aibusy = False
        socketio.emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")

#==================================================================#
# 
#==================================================================#
def editrequest(n):
    if(n == 0):
        txt = koboldai_vars.prompt
    else:
        txt = koboldai_vars.actions[n-1]
    
    koboldai_vars.editln = n
    emit('from_server', {'cmd': 'setinputtext', 'data': txt}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'enablesubmit', 'data': ''}, broadcast=True, room="UI_1")

#==================================================================#
# 
#==================================================================#
def editsubmit(data):
    koboldai_vars.recentedit = True
    if(koboldai_vars.editln == 0):
        koboldai_vars.prompt = data
    else:
        koboldai_vars.actions[koboldai_vars.editln-1] = data
    
    koboldai_vars.mode = "play"
    update_story_chunk(koboldai_vars.editln)
    emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.editln}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'editmode', 'data': 'false'}, room="UI_1")
    send_debug()

#==================================================================#
#  
#==================================================================#
def deleterequest():
    koboldai_vars.recentedit = True
    # Don't delete prompt
    if(koboldai_vars.editln == 0):
        # Send error message
        pass
    else:
        koboldai_vars.actions.delete_action(koboldai_vars.editln-1)
        koboldai_vars.mode = "play"
        remove_story_chunk(koboldai_vars.editln)
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, room="UI_1")
    send_debug()

#==================================================================#
# 
#==================================================================#
def inlineedit(chunk, data):
    koboldai_vars.recentedit = True
    chunk = int(chunk)
    if(chunk == 0):
        if(len(data.strip()) == 0):
            return
        koboldai_vars.prompt = data
    else:
        if(chunk-1 in koboldai_vars.actions):
            koboldai_vars.actions[chunk-1] = data
        else:
            logger.warning(f"Attempted to edit non-existent chunk {chunk}")

    setgamesaved(False)
    update_story_chunk(chunk)
    emit('from_server', {'cmd': 'texteffect', 'data': chunk}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#  
#==================================================================#
def inlinedelete(chunk):
    koboldai_vars.recentedit = True
    chunk = int(chunk)
    # Don't delete prompt
    if(chunk == 0):
        # Send error message
        update_story_chunk(chunk)
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."}, room="UI_1")
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    else:
        if(chunk-1 in koboldai_vars.actions):
            koboldai_vars.actions.delete_action(chunk-1)
        else:
            logger.warning(f"Attempted to delete non-existent chunk {chunk}")
        setgamesaved(False)
        remove_story_chunk(chunk)
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#   Toggles the game mode for memory editing and sends UI commands
#==================================================================#
def togglememorymode():
    if(koboldai_vars.mode == "play"):
        koboldai_vars.mode = "memory"
        emit('from_server', {'cmd': 'memmode', 'data': 'true'}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setinputtext', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")
    elif(koboldai_vars.mode == "memory"):
        koboldai_vars.mode = "play"
        emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True, room="UI_1")

#==================================================================#
#   Toggles the game mode for WI editing and sends UI commands
#==================================================================#
def togglewimode():
    if(koboldai_vars.mode == "play"):
        koboldai_vars.mode = "wi"
        emit('from_server', {'cmd': 'wimode', 'data': 'true'}, broadcast=True, room="UI_1")
    elif(koboldai_vars.mode == "wi"):
        # Commit WI fields first
        requestwi()
        # Then set UI state back to Play
        koboldai_vars.mode = "play"
        emit('from_server', {'cmd': 'wimode', 'data': 'false'}, broadcast=True, room="UI_1")
    sendwi()

#==================================================================#
#   
#==================================================================#
def addwiitem(folder_uid=None):
    assert folder_uid is None or folder_uid in koboldai_vars.wifolders_d
    ob = {"key": "", "keysecondary": "", "content": "", "comment": "", "folder": folder_uid, "num": len(koboldai_vars.worldinfo), "init": False, "selective": False, "constant": False}
    koboldai_vars.worldinfo.append(ob)
    while(True):
        uid = int.from_bytes(os.urandom(4), "little", signed=True)
        if(uid not in koboldai_vars.worldinfo_u):
            break
    koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
    koboldai_vars.worldinfo[-1]["uid"] = uid
    if(folder_uid is not None):
        koboldai_vars.wifolders_u[folder_uid].append(koboldai_vars.worldinfo[-1])
    emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True, room="UI_1")

#==================================================================#
#   Creates a new WI folder with an unused cryptographically secure random UID
#==================================================================#
def addwifolder():
    while(True):
        uid = int.from_bytes(os.urandom(4), "little", signed=True)
        if(uid not in koboldai_vars.wifolders_d):
            break
    ob = {"name": "", "collapsed": False}
    koboldai_vars.wifolders_d[uid] = ob
    koboldai_vars.wifolders_l.append(uid)
    koboldai_vars.wifolders_u[uid] = []
    emit('from_server', {'cmd': 'addwifolder', 'uid': uid, 'data': ob}, broadcast=True, room="UI_1")
    addwiitem(folder_uid=uid)

#==================================================================#
#   Move the WI entry with UID src so that it immediately precedes
#   the WI entry with UID dst
#==================================================================#
def movewiitem(dst, src):
    setgamesaved(False)
    if(koboldai_vars.worldinfo_u[src]["folder"] is not None):
        for i, e in enumerate(koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[src]["folder"]]):
            if(e["uid"] == koboldai_vars.worldinfo_u[src]["uid"]):
                koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[src]["folder"]].pop(i)
                break
    if(koboldai_vars.worldinfo_u[dst]["folder"] is not None):
        koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[dst]["folder"]].append(koboldai_vars.worldinfo_u[src])
    koboldai_vars.worldinfo_u[src]["folder"] = koboldai_vars.worldinfo_u[dst]["folder"]
    for i, e in enumerate(koboldai_vars.worldinfo):
        if(e["uid"] == koboldai_vars.worldinfo_u[src]["uid"]):
            _src = i
        elif(e["uid"] == koboldai_vars.worldinfo_u[dst]["uid"]):
            _dst = i
    koboldai_vars.worldinfo[_src]["folder"] = koboldai_vars.worldinfo[_dst]["folder"]
    koboldai_vars.worldinfo.insert(_dst - (_dst >= _src), koboldai_vars.worldinfo.pop(_src))
    sendwi()

#==================================================================#
#   Move the WI folder with UID src so that it immediately precedes
#   the WI folder with UID dst
#==================================================================#
def movewifolder(dst, src):
    setgamesaved(False)
    koboldai_vars.wifolders_l.remove(src)
    if(dst is None):
        # If dst is None, that means we should move src to be the last folder
        koboldai_vars.wifolders_l.append(src)
    else:
        koboldai_vars.wifolders_l.insert(koboldai_vars.wifolders_l.index(dst), src)
    sendwi()

#==================================================================#
#   
#==================================================================#
def sendwi():
    # Cache len of WI
    ln = len(koboldai_vars.worldinfo)

    # Clear contents of WI container
    emit('from_server', {'cmd': 'wistart', 'wifolders_d': koboldai_vars.wifolders_d, 'wifolders_l': koboldai_vars.wifolders_l, 'data': ''}, broadcast=True, room="UI_1")

    # Stable-sort WI entries in order of folder
    stablesortwi()

    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]

    # If there are no WI entries, send an empty WI object
    if(ln == 0):
        addwiitem()
    else:
        # Send contents of WI array
        last_folder = ...
        for wi in koboldai_vars.worldinfo:
            if(wi["folder"] != last_folder):
                emit('from_server', {'cmd': 'addwifolder', 'uid': wi["folder"], 'data': koboldai_vars.wifolders_d[wi["folder"]] if wi["folder"] is not None else None}, broadcast=True, room="UI_1")
                last_folder = wi["folder"]
            ob = wi
            emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True, room="UI_1")
    
    emit('from_server', {'cmd': 'wifinish', 'data': ''}, broadcast=True, room="UI_1")

#==================================================================#
#  Request current contents of all WI HTML elements
#==================================================================#
def requestwi():
    list = []
    for wi in koboldai_vars.worldinfo:
        list.append(wi["num"])
    emit('from_server', {'cmd': 'requestwiitem', 'data': list}, room="UI_1")

#==================================================================#
#  Stable-sort WI items so that items in the same folder are adjacent,
#  and items in different folders are sorted based on the order of the folders
#==================================================================#
def stablesortwi():
    mapping = {uid: index for index, uid in enumerate(koboldai_vars.wifolders_l)}
    koboldai_vars.worldinfo.sort(key=lambda x: mapping[x["folder"]] if x["folder"] is not None else float("inf"))
    last_folder = ...
    last_wi = None
    for i, wi in enumerate(koboldai_vars.worldinfo):
        wi["num"] = i
        wi["init"] = True
        if(wi["folder"] != last_folder):
            if(last_wi is not None and last_folder is not ...):
                last_wi["init"] = False
            last_folder = wi["folder"]
        last_wi = wi
    if(last_wi is not None):
        last_wi["init"] = False
    for folder in koboldai_vars.wifolders_u:
        koboldai_vars.wifolders_u[folder].sort(key=lambda x: x["num"])

#==================================================================#
#  Extract object from server and send it to WI objects
#==================================================================#
def commitwi(ar):
    for ob in ar:
        koboldai_vars.worldinfo_u[ob["uid"]]["key"]          = ob["key"]
        koboldai_vars.worldinfo_u[ob["uid"]]["keysecondary"] = ob["keysecondary"]
        koboldai_vars.worldinfo_u[ob["uid"]]["content"]      = ob["content"]
        koboldai_vars.worldinfo_u[ob["uid"]]["comment"]      = ob.get("comment", "")
        koboldai_vars.worldinfo_u[ob["uid"]]["folder"]       = ob.get("folder", None)
        koboldai_vars.worldinfo_u[ob["uid"]]["selective"]    = ob["selective"]
        koboldai_vars.worldinfo_u[ob["uid"]]["constant"]     = ob.get("constant", False)
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    koboldai_vars.sync_worldinfo_v1_to_v2()
    sendwi()

#==================================================================#
#  
#==================================================================#
def deletewi(uid):
    if(uid in koboldai_vars.worldinfo_u):
        setgamesaved(False)
        # Store UID of deletion request
        koboldai_vars.deletewi = uid
        if(koboldai_vars.deletewi is not None):
            if(koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["folder"] is not None):
                for i, e in enumerate(koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["folder"]]):
                    if(e["uid"] == koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["uid"]):
                        koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["folder"]].pop(i)
                        break
            for i, e in enumerate(koboldai_vars.worldinfo):
                if(e["uid"] == koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["uid"]):
                    del koboldai_vars.worldinfo[i]
                    break
            del koboldai_vars.worldinfo_u[koboldai_vars.deletewi]
            # Send the new WI array structure
            sendwi()
            # And reset deletewi
            koboldai_vars.deletewi = None

#==================================================================#
#  
#==================================================================#
def deletewifolder(uid):
    del koboldai_vars.wifolders_u[uid]
    del koboldai_vars.wifolders_d[uid]
    del koboldai_vars.wifolders_l[koboldai_vars.wifolders_l.index(uid)]
    setgamesaved(False)
    # Delete uninitialized entries in the folder we're going to delete
    koboldai_vars.worldinfo = [wi for wi in koboldai_vars.worldinfo if wi["folder"] != uid or wi["init"]]
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    # Move WI entries that are inside of the folder we're going to delete
    # so that they're outside of all folders
    for wi in koboldai_vars.worldinfo:
        if(wi["folder"] == uid):
            wi["folder"] = None

    sendwi()

#==================================================================#
#  Look for WI keys in text to generator 
#==================================================================#
def checkworldinfo(txt, allowed_entries=None, allowed_folders=None, force_use_txt=False, scan_story=True, actions=None):
    original_txt = txt

    if(actions is None):
        actions = koboldai_vars.actions

    # Dont go any further if WI is empty
    if(len(koboldai_vars.worldinfo) == 0):
        return "", set()
    
    # Cache actions length
    ln = len(actions)
    
    # Don't bother calculating action history if widepth is 0
    if(koboldai_vars.widepth > 0 and scan_story):
        depth = koboldai_vars.widepth
        # If this is not a continue, add 1 to widepth since submitted
        # text is already in action history @ -1
        if(not force_use_txt and (txt != "" and koboldai_vars.prompt != txt)):
            txt    = ""
            depth += 1
        
        if(ln > 0):
            chunks = actions[-depth:]
            #i = 0
            #for key in reversed(actions):
            #    chunk = actions[key]
            #    chunks.appendleft(chunk)
            #    i += 1
            #    if(i == depth):
            #        break
        if(ln >= depth):
            txt = "".join(chunks)
        elif(ln > 0):
            txt = koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt) + "".join(chunks)
        elif(ln == 0):
            txt = koboldai_vars.comregex_ai.sub('', koboldai_vars.prompt)

    if(force_use_txt):
        txt += original_txt

    # Scan text for matches on WI keys
    wimem = ""
    found_entries = set()
    for wi in koboldai_vars.worldinfo:
        if(allowed_entries is not None and wi["uid"] not in allowed_entries):
            continue
        if(allowed_folders is not None and wi["folder"] not in allowed_folders):
            continue

        if(wi.get("constant", False)):
            wimem = wimem + wi["content"] + "\n"
            found_entries.add(id(wi))
            continue

        if(len(wi["key"].strip()) > 0 and (not wi.get("selective", False) or len(wi.get("keysecondary", "").strip()) > 0)):
            # Split comma-separated keys
            keys = wi["key"].split(",")
            keys_secondary = wi.get("keysecondary", "").split(",")

            for k in keys:
                ky = k
                # Remove leading/trailing spaces if the option is enabled
                if(koboldai_vars.wirmvwhtsp):
                    ky = k.strip()
                if ky.lower() in txt.lower():
                    if wi.get("selective", False) and len(keys_secondary):
                        found = False
                        for ks in keys_secondary:
                            ksy = ks
                            if(koboldai_vars.wirmvwhtsp):
                                ksy = ks.strip()
                            if ksy.lower() in txt.lower():
                                wimem = wimem + wi["content"] + "\n"
                                found_entries.add(id(wi))
                                found = True
                                break
                        if found:
                            break
                    else:
                        wimem = wimem + wi["content"] + "\n"
                        found_entries.add(id(wi))
                        break
    
    return wimem, found_entries
    
#==================================================================#
#  Commit changes to Memory storage
#==================================================================#
def memsubmit(data):
    emit('from_server', {'cmd': 'setinputtext', 'data': data}, broadcast=True, room="UI_1")
    # Maybe check for length at some point
    # For now just send it to storage
    if(data != koboldai_vars.memory):
        setgamesaved(False)
    koboldai_vars.memory = data
    koboldai_vars.mode = "play"
    emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True, room="UI_1")
    
    # Ask for contents of Author's Note field
    emit('from_server', {'cmd': 'getanote', 'data': ''}, room="UI_1")

#==================================================================#
#  Commit changes to Author's Note
#==================================================================#
def anotesubmit(data, template=""):
    assert type(data) is str and type(template) is str
    # Maybe check for length at some point
    # For now just send it to storage
    if(data != koboldai_vars.authornote):
        setgamesaved(False)
    koboldai_vars.authornote = data

    if(koboldai_vars.authornotetemplate != template):
        koboldai_vars.setauthornotetemplate = template
        settingschanged()
    koboldai_vars.authornotetemplate = template

    emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")

#==================================================================#
#  Assembles game data into a request to InferKit API
#==================================================================#
def ikrequest(txt):
    # Log request to console
    if not koboldai_vars.quiet:
        print("{0}Len:{1}, Txt:{2}{3}".format(colors.YELLOW, len(txt), txt, colors.END))
    
    # Build request JSON data
    reqdata = {
        'forceNoEnd': True,
        'length': koboldai_vars.ikgen,
        'prompt': {
            'isContinuation': False,
            'text': txt
        },
        'startFromBeginning': False,
        'streamResponse': False,
        'temperature': koboldai_vars.temp,
        'topP': koboldai_vars.top_p
    }
    
    # Create request
    req = requests.post(
        koboldai_vars.url, 
        json    = reqdata,
        headers = {
            'Authorization': 'Bearer '+koboldai_vars.apikey
            }
        )
    
    # Deal with the response
    if(req.status_code == 200):
        genout = req.json()["data"]["text"]

        koboldai_vars.lua_koboldbridge.outputs[1] = genout

        execute_outmod()
        if(koboldai_vars.lua_koboldbridge.regeneration_required):
            koboldai_vars.lua_koboldbridge.regeneration_required = False
            genout = koboldai_vars.lua_koboldbridge.outputs[1]
            assert genout is str

        if not koboldai_vars.quiet:
            print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
        koboldai_vars.actions.append(genout)
        update_story_chunk('last')
        emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.actions.get_last_key() + 1 if len(koboldai_vars.actions) else 0}, broadcast=True, room="UI_1")
        send_debug()
        set_aibusy(0)
    else:
        # Send error message to web client
        er = req.json()
        if("error" in er):
            code = er["error"]["extensions"]["code"]
        elif("errors" in er):
            code = er["errors"][0]["extensions"]["code"]
            
        errmsg = "InferKit API Error: {0} - {1}".format(req.status_code, code)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True, room="UI_1")
        set_aibusy(0)

#==================================================================#
#  Forces UI to Play mode
#==================================================================#
def exitModes():
    if(koboldai_vars.mode == "edit"):
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    elif(koboldai_vars.mode == "memory"):
        emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True, room="UI_1")
    elif(koboldai_vars.mode == "wi"):
        emit('from_server', {'cmd': 'wimode', 'data': 'false'}, broadcast=True, room="UI_1")
    koboldai_vars.mode = "play"

#==================================================================#
#  Launch in-browser save prompt
#==================================================================#
def saveas(data):
    name = data['name']
    koboldai_vars.story_name = name
    if not data['pins']:
        koboldai_vars.actions.clear_all_options()
    # Check if filename exists already
    save_name = koboldai_vars.story_name if koboldai_vars.story_name != "" else "untitled"
    same_story = True
    if os.path.exists("stories/{}".format(save_name)):
        with open("stories/{}/story.json".format(save_name), "r") as settings_file:
            json_data = json.load(settings_file)
            if 'story_id' in json_data:
                same_story = json_data['story_id'] == koboldai_vars.story_id
            else:
                same_story = False
                
    if same_story:
        # All clear to save
        koboldai_vars.save_story()
        emit('from_server', {'cmd': 'hidesaveas', 'data': ''}, room="UI_1")
    else:
        # File exists, prompt for overwrite
        koboldai_vars.saveow   = True
        koboldai_vars.svowname = name
        emit('from_server', {'cmd': 'askforoverwrite', 'data': ''}, room="UI_1")

#==================================================================#
#  Launch in-browser story-delete prompt
#==================================================================#
def deletesave(name):
    name = utils.cleanfilename(name).replace(".json", "")
    try:
        if os.path.exists("stories/{}".format(name)):
            shutil.rmtree("stories/{}".format(name))
        elif os.path.exists("stories/{}.json".format(name)):
            os.remove("stories/{}.json".format(name))
            
        emit('from_server', {'cmd': 'hidepopupdelete', 'data': ''}, room="UI_1")
        getloadlist()
    except OSError as e:
        print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
        emit('from_server', {'cmd': 'popuperror', 'data': str(e)}, room="UI_1")

#==================================================================#
#  Launch in-browser story-rename prompt
#==================================================================#
def renamesave(name, newname):
    # Check if filename exists already
    name = utils.cleanfilename(name)
    newname = utils.cleanfilename(newname)
    
    if os.path.exists("stories/{}/story.json".format(name)):
        with open("stories/{}/story.json".format(name), "r") as f:
            original_data = json.load(f)
    elif os.path.exists("stories/{}.json".format(name)):
        with open("stories/{}.json".format(name), "r") as f:
            original_data = json.load(f)
    else:
        print("{0}{1}{2}".format(colors.RED, "File doesn't exist to rename", colors.END))
        emit('from_server', {'cmd': 'popuperror', 'data': "File doesn't exist to rename"}, room="UI_1")
        return
    
    story_id = original_data['story_id'] if 'story_id' in original_data else None
    
    #Check if newname already exists:
    same_story = True
    if os.path.exists("stories/{}".format(newname)):
        with open("stories/{}/story.json".format(newname), "r") as settings_file:
            json_data = json.load(settings_file)
            if 'story_id' in json_data:
                same_story = json_data['story_id'] == koboldai_vars.story_id
            else:
                same_story = False

    
    
    
    if same_story or koboldai_vars.saveow:
        if story_id is None:
            os.remove("stories/{}.json".format(newname))
            os.rename("stories/{}.json".format(name), "stories/{}.json".format(newname))
        else:
            shutil.rmtree("stories/{}".format(newname))
            os.rename("stories/{}".format(name), "stories/{}".format(newname))
        emit('from_server', {'cmd': 'hidepopuprename', 'data': ''}, room="UI_1")
        getloadlist()
    else:
        # File exists, prompt for overwrite
        koboldai_vars.saveow   = True
        koboldai_vars.svowname = newname
        emit('from_server', {'cmd': 'askforoverwrite', 'data': ''}, room="UI_1")

#==================================================================#
#  Save the currently running story
#==================================================================#
def save():
    # Check if a file is currently open
    save_name = koboldai_vars.story_name if koboldai_vars.story_name != "" else "untitled"
    same_story = True
    if os.path.exists("stories/{}".format(save_name)):
        with open("stories/{}/story.json".format(save_name), "r", encoding="utf-8") as settings_file:
            json_data = json.load(settings_file)
            if 'story_id' in json_data:
                same_story = json_data['story_id'] == koboldai_vars.story_id
            else:
                same_story = False
    
    if same_story:
        koboldai_vars.save_story()
    else:
        emit('from_server', {'cmd': 'saveas', 'data': ''}, room="UI_1")

#==================================================================#
#  Save the story via file browser (Disabled due to new file format)
#==================================================================#
#def savetofile():
#    savpath = fileops.getsavepath(koboldai_vars.savedir, "Save Story As", [("Json", "*.json")])
#    saveRequest(savpath)

#==================================================================#
#  Save the story to specified path
#==================================================================#
def saveRequest(savpath, savepins=True):    
    if(savpath):
        # Leave Edit/Memory mode before continuing
        exitModes()
        
        # Save path for future saves
        koboldai_vars.savedir = savpath
        txtpath = os.path.splitext(savpath)[0] + ".txt"
        # Build json to write
        js = {}
        js["gamestarted"] = koboldai_vars.gamestarted
        js["prompt"]      = koboldai_vars.prompt
        js["memory"]      = koboldai_vars.memory
        js["authorsnote"] = koboldai_vars.authornote
        js["anotetemplate"] = koboldai_vars.authornotetemplate
        js["actions"]     = tuple(koboldai_vars.actions.values())
        if savepins:
            js["actions_metadata"]     = koboldai_vars.actions.options(ui_version=1)
        js["worldinfo"]   = []
        js["wifolders_d"] = koboldai_vars.wifolders_d
        js["wifolders_l"] = koboldai_vars.wifolders_l
		
        # Extract only the important bits of WI
        for wi in koboldai_vars.worldinfo_i:
            if(True):
                js["worldinfo"].append({
                    "key": wi["key"],
                    "keysecondary": wi["keysecondary"],
                    "content": wi["content"],
                    "comment": wi["comment"],
                    "folder": wi["folder"],
                    "selective": wi["selective"],
                    "constant": wi["constant"]
                })
                
        txt = koboldai_vars.prompt + "".join(koboldai_vars.actions.values())

        # Write it
        try:
            file = open(savpath, "w", encoding="utf-8")
        except Exception as e:
            return e
        try:
            file.write(json.dumps(js, indent=3))
        except Exception as e:
            file.close()
            return e
        file.close()
        
        try:
            file = open(txtpath, "w", encoding="utf-8")
        except Exception as e:
            return e
        try:
            file.write(txt)
        except Exception as e:
            file.close()
            return e
        file.close()

        filename = path.basename(savpath)
        if(filename.endswith('.json')):
            filename = filename[:-5]
        koboldai_vars.laststory = filename
        emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, broadcast=True, room="UI_1")
        setgamesaved(True)
        print("{0}Story saved to {1}!{2}".format(colors.GREEN, path.basename(savpath), colors.END))

#==================================================================#
#  Show list of saved stories
#==================================================================#
def getloadlist(data=None):
    emit('from_server', {'cmd': 'buildload', 'data': fileops.getstoryfiles()}, room="UI_1")

#==================================================================#
#  Show list of soft prompts
#==================================================================#
def getsplist():
    if(koboldai_vars.allowsp):
        emit('from_server', {'cmd': 'buildsp', 'data': fileops.getspfiles(koboldai_vars.modeldim)}, room="UI_1")

#==================================================================#
#  Get list of userscripts
#==================================================================#
def getuslist():
    files = {i: v for i, v in enumerate(fileops.getusfiles())}
    loaded = []
    unloaded = []
    userscripts = set(koboldai_vars.userscripts)
    for i in range(len(files)):
        if files[i]["filename"] not in userscripts:
            unloaded.append(files[i])
    files = {files[k]["filename"]: files[k] for k in files}
    userscripts = set(files.keys())
    for filename in koboldai_vars.userscripts:
        if filename in userscripts:
            loaded.append(files[filename])
    return unloaded, loaded

#==================================================================#
#  Load a saved story via file browser
#==================================================================#
def loadfromfile():
    loadpath = fileops.getloadpath(koboldai_vars.savedir, "Select Story File", [("Json", "*.json")])
    loadRequest(loadpath)

#==================================================================#
#  Load a stored story from a file
#==================================================================#
def loadRequest(loadpath, filename=None):
    logger.debug("Load Request")
    logger.debug("Called from {}".format(inspect.stack()[1].function))

    if not loadpath:
        return
    
    start_time = time.time()
    # Leave Edit/Memory mode before continuing
    exitModes()
    
    # Read file contents into JSON object
    start_time = time.time()
    if(isinstance(loadpath, str)):
		#Original UI only sends the story name and assumes it's always a .json file... here we check to see if it's a directory to load that way
        if not isinstance(loadpath, dict) and not os.path.exists(loadpath):
            if os.path.exists(loadpath.replace(".json", "")):
                loadpath = loadpath.replace(".json", "")

        if not isinstance(loadpath, dict) and os.path.isdir(loadpath):
            if not valid_v3_story(loadpath):
                raise RuntimeError(f"Tried to load {loadpath}, a non-save directory.")
            koboldai_vars.update_story_path_structure(loadpath)
            loadpath = os.path.join(loadpath, "story.json")
            
        with open(loadpath, "r", encoding="utf-8") as file:
            js = json.load(file)
            from_file=loadpath
        if(filename is None):
            filename = path.basename(loadpath)
    else:
        js = loadpath
        if(filename is None):
            filename = "untitled.json"
        from_file=None
    js['v1_loadpath'] = loadpath
    js['v1_filename'] = filename
    logger.debug("Loading JSON data took {}s".format(time.time()-start_time))
    loadJSON(js, from_file=from_file)
    
    #When we load we're not transmitting the data to UI1 anymore. Simplist solution is to refresh the browser so we get current data. 
    #this function does that
    emit('from_server', {'cmd': 'hide_model_name'}, broadcast=True, room="UI_1")

def loadJSON(json_text_or_dict, from_file=None):
    logger.debug("Loading JSON Story")
    logger.debug("Called from {}".format(inspect.stack()[1].function))
    start_time = time.time()
    if isinstance(json_text_or_dict, str):
        json_data = json.loads(json_text_or_dict)
    else:
        json_data = json_text_or_dict
    logger.debug("Loading JSON data took {}s".format(time.time()-start_time))
    if "file_version" in json_data:
        if json_data['file_version'] == 2:
            load_story_v2(json_data, from_file=from_file)
        else:
            load_story_v1(json_data, from_file=from_file)
    else:
        load_story_v1(json_data, from_file=from_file)
    logger.debug("Calcing AI Text from Story Load")
    ignore = koboldai_vars.calc_ai_text()

def load_story_v1(js, from_file=None):
    logger.debug("Loading V1 Story")
    logger.debug("Called from {}".format(inspect.stack()[1].function))
    loadpath = js['v1_loadpath'] if 'v1_loadpath' in js else koboldai_vars.savedir
    filename = js['v1_filename'] if 'v1_filename' in js else 'untitled.json'
    
    
    _filename = filename
    if(filename.endswith('.json')):
        _filename = filename[:-5]
    leave_room(session.get('story', 'default'))
    session['story'] = _filename
    join_room(_filename)
    #create the story
    #koboldai_vars.create_story(session['story'])
    koboldai_vars.create_story(session['story'])
    
    koboldai_vars.laststory = _filename
    #set the story_name
    koboldai_vars.story_name = _filename
    

    # Copy file contents to vars
    koboldai_vars.gamestarted = js["gamestarted"]
    koboldai_vars.prompt      = js["prompt"]
    koboldai_vars.memory      = js["memory"]
    koboldai_vars.worldinfo_v2.reset()
    koboldai_vars.worldinfo   = []
    koboldai_vars.worldinfo_i = []
    koboldai_vars.worldinfo_u = {}
    koboldai_vars.wifolders_d = {int(k): v for k, v in js.get("wifolders_d", {}).items()}
    koboldai_vars.wifolders_l = js.get("wifolders_l", [])
    koboldai_vars.wifolders_u = {uid: [] for uid in koboldai_vars.wifolders_d}
    koboldai_vars.lastact     = ""
    koboldai_vars.submission  = ""
    koboldai_vars.lastctx     = ""
    koboldai_vars.genseqs = []

    actions = collections.deque(js["actions"])
    


    if(len(koboldai_vars.prompt.strip()) == 0):
        while(len(actions)):
            action = actions.popleft()
            if(len(action.strip()) != 0):
                koboldai_vars.prompt = action
                break
        else:
            koboldai_vars.gamestarted = False
    if(koboldai_vars.gamestarted):
        #We set the action count higher so that we don't trigger a scroll in the UI. 
        #Once all but the last is loaded we can bring it back down and do the last one so we scroll to it
        logger.debug("Created temp story class")
        temp_story_class = koboldai_settings.KoboldStoryRegister(None, None, koboldai_vars, tokenizer=None)
        
        for i in range(len(js["actions"])):
            temp_story_class.append(js["actions"][i], recalc=False)
        logger.debug("Added actions to temp story class")
        

        if "actions_metadata" in js:
            if type(js["actions_metadata"]) == dict:
                for key in js["actions_metadata"]:
                    if js["actions_metadata"][key]["Alternative Text"] != []:
                        data = js["actions_metadata"][key]["Alternative Text"]
                        for i in range(len(js["actions_metadata"][key]["Alternative Text"])):
                            data[i]["text"] = data[i].pop("Text")
                        temp_story_class.set_options(data, int(key))
        koboldai_vars.actions.load_json(temp_story_class.to_json())
        logger.debug("Saved temp story class")
        del temp_story_class
    
    # Try not to break older save files
    if("authorsnote" in js):
        koboldai_vars.authornote = js["authorsnote"]
    else:
        koboldai_vars.authornote = ""
    if("anotetemplate" in js):
        koboldai_vars.authornotetemplate = js["anotetemplate"]
    else:
        koboldai_vars.authornotetemplate = "[Author's note: <|>]"
    
    if("worldinfo" in js):
        num = 0
        for wi in js["worldinfo"]:
            if wi.get("folder", "root") == 'root':
                folder = "root" 
            else:
                if 'wifolders_d' in js:
                    if wi['folder'] in js['wifolders_d']:
                        folder = js['wifolders_d'][wi['folder']]['name']
                    else:
                        folder = "root"
                else:
                    folder = "root"
            koboldai_vars.worldinfo_v2.add_item([x.strip() for x in wi["key"].split(",")][0], wi["key"], wi.get("keysecondary", ""), 
                                                folder, wi.get("constant", False), 
                                                wi["content"], wi.get("comment", ""), recalc=False, sync=False, send_to_ui=False)
        koboldai_vars.worldinfo_v2.sync_world_info_to_old_format()
        koboldai_vars.worldinfo_v2.send_to_ui()

    # Save path for save button
    koboldai_vars.savedir = loadpath
    
    # Clear loadselect var
    koboldai_vars.loadselect = ""
    
    # Refresh game screen
    emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, broadcast=True, room="UI_1")
    setgamesaved(True)
    sendwi()
    emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")
    refresh_story()
    emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True, room="UI_1")
    print("{0}Story loaded from {1}!{2}".format(colors.GREEN, filename, colors.END))
    
    send_debug()
    
    if from_file is not None:
        #Save the file so we get a new V2 format, then move the save file into the proper directory
        koboldai_vars.save_story()
        #We're no longer moving the original file. It'll stay in place.
        #shutil.move(from_file, koboldai_vars.save_paths.story.replace("story.json", "v1_file.json"))
    

def load_story_v2(js, from_file=None):
    logger.debug("Loading V2 Story")
    logger.debug("Called from {}".format(inspect.stack()[1].function))

    new_story = js["story_name"]
    # In socket context
    if hasattr(request, "sid"):
        leave_room(session['story'])
        join_room(new_story)
    session['story'] = new_story
    
    koboldai_vars.load_story(session['story'], js)
    
    if from_file is not None and os.path.basename(from_file) != "story.json":
        #Save the file so we get a new V2 format, then move the save file into the proper directory
        koboldai_vars.save_story()
        shutil.move(from_file, koboldai_vars.save_paths.story.replace("story.json", "v2_file.json"))
    


#==================================================================#
# Import an AIDungon game exported with Mimi's tool
#==================================================================#
def importRequest():
    importpath = fileops.getloadpath(koboldai_vars.savedir, "Select AID CAT File", [("Json", "*.json")])
    
    if(importpath):
        # Leave Edit/Memory mode before continuing
        exitModes()
        
        # Read file contents into JSON object
        file = open(importpath, "rb")
        koboldai_vars.importjs = json.load(file)
        
        # If a bundle file is being imported, select just the Adventures object
        if type(koboldai_vars.importjs) is dict and "stories" in koboldai_vars.importjs:
            koboldai_vars.importjs = koboldai_vars.importjs["stories"]
        
        # Clear Popup Contents
        emit('from_server', {'cmd': 'clearpopup', 'data': ''}, broadcast=True, room="UI_1")
        
        # Initialize koboldai_vars
        num = 0
        koboldai_vars.importnum = -1
        
        # Get list of stories
        for story in koboldai_vars.importjs:
            ob = {}
            ob["num"]   = num
            if(story["title"] != "" and story["title"] != None):
                ob["title"] = story["title"]
            else:
                ob["title"] = "(No Title)"
            if(story["description"] != "" and story["description"] != None):
                ob["descr"] = story["description"]
            else:
                ob["descr"] = "(No Description)"
            if("actions" in story):
                ob["acts"]  = len(story["actions"])
            elif("actionWindow" in story):
                ob["acts"]  = len(story["actionWindow"])
            emit('from_server', {'cmd': 'addimportline', 'data': ob}, room="UI_1")
            num += 1
        
        # Show Popup
        emit('from_server', {'cmd': 'popupshow', 'data': True}, room="UI_1")

#==================================================================#
# Import an AIDungon game selected in popup
#==================================================================#
def importgame():
    if(koboldai_vars.importnum >= 0):
        # Cache reference to selected game
        ref = koboldai_vars.importjs[koboldai_vars.importnum]
        
        # Copy game contents to koboldai_vars
        koboldai_vars.gamestarted = True
        
        # Support for different versions of export script
        if("actions" in ref):
            if(len(ref["actions"]) > 0):
                koboldai_vars.prompt = ref["actions"][0]["text"]
            else:
                koboldai_vars.prompt = ""
        elif("actionWindow" in ref):
            if(len(ref["actionWindow"]) > 0):
                koboldai_vars.prompt = ref["actionWindow"][0]["text"]
            else:
                koboldai_vars.prompt = ""
        else:
            koboldai_vars.prompt = ""
        koboldai_vars.memory      = ref["memory"]
        koboldai_vars.authornote  = ref["authorsNote"] if type(ref["authorsNote"]) is str else ""
        koboldai_vars.authornotetemplate = "[Author's note: <|>]"
        koboldai_vars.actions.reset()
        koboldai_vars.actions_metadata = {}
        koboldai_vars.worldinfo   = []
        koboldai_vars.worldinfo_i = []
        koboldai_vars.worldinfo_u = {}
        koboldai_vars.wifolders_d = {}
        koboldai_vars.wifolders_l = []
        koboldai_vars.wifolders_u = {uid: [] for uid in koboldai_vars.wifolders_d}
        koboldai_vars.lastact     = ""
        koboldai_vars.submission  = ""
        koboldai_vars.lastctx     = ""
        
        # Get all actions except for prompt
        if("actions" in ref):
            if(len(ref["actions"]) > 1):
                for act in ref["actions"][1:]:
                    koboldai_vars.actions.append(act["text"])
        elif("actionWindow" in ref):
            if(len(ref["actionWindow"]) > 1):
                for act in ref["actionWindow"][1:]:
                    koboldai_vars.actions.append(act["text"])
        
        # Get just the important parts of world info
        if(ref["worldInfo"] != None):
            if(len(ref["worldInfo"]) > 1):
                num = 0
                for wi in ref["worldInfo"]:
                    koboldai_vars.worldinfo.append({
                        "key": wi["keys"],
                        "keysecondary": wi.get("keysecondary", ""),
                        "content": wi["entry"],
                        "comment": wi.get("comment", ""),
                        "folder": wi.get("folder", None),
                        "num": num,
                        "init": True,
                        "selective": wi.get("selective", False),
                        "constant": wi.get("constant", False),
                        "uid": None,
                    })
                    while(True):
                        uid = int.from_bytes(os.urandom(4), "little", signed=True)
                        if(uid not in koboldai_vars.worldinfo_u):
                            break
                    koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
                    koboldai_vars.worldinfo[-1]["uid"] = uid
                    if(koboldai_vars.worldinfo[-1]["folder"]) is not None:
                        koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
                    num += 1

        for uid in koboldai_vars.wifolders_l + [None]:
            koboldai_vars.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
            while(True):
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if(uid not in koboldai_vars.worldinfo_u):
                    break
            koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
            koboldai_vars.worldinfo[-1]["uid"] = uid
            if(koboldai_vars.worldinfo[-1]["folder"] is not None):
                koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
        stablesortwi()
        koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
        
        # Clear import data
        koboldai_vars.importjs = {}
        
        # Reset current save
        koboldai_vars.savedir = getcwd()+"\\stories"
        
        # Refresh game screen
        koboldai_vars.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, broadcast=True, room="UI_1")
        setgamesaved(False)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True, room="UI_1")

#==================================================================#
# Import an aidg.club prompt and start a new game with it.
#==================================================================#
def importAidgRequest(id):    
    exitModes()
    
    urlformat = "https://aetherroom.club/api/"
    req = requests.get(urlformat+id)
    if(req.status_code == 200):
        js = req.json()
        
        # Import game state
        
        koboldai_vars.create_story("")
        koboldai_vars.gamestarted = True
        koboldai_vars.prompt      = js["promptContent"]
        koboldai_vars.memory      = js["memory"]
        koboldai_vars.authornote  = js["authorsNote"]
        
        
        if not koboldai_vars.memory:
            koboldai_vars.memory = ""
        if not koboldai_vars.authornote:
            koboldai_vars.authornote = ""
        
        num = 0
        for wi in js["worldInfos"]:
            koboldai_vars.worldinfo.append({
                "key": wi["keys"],
                "keysecondary": wi.get("keysecondary", ""),
                "content": wi["entry"],
                "comment": wi.get("comment", ""),
                "folder": wi.get("folder", None),
                "num": num,
                "init": True,
                "selective": wi.get("selective", False),
                "constant": wi.get("constant", False),
                "uid": None,
            })
            
            koboldai_vars.worldinfo_v2.add_item([x.strip() for x in wi["keys"].split(",")][0], wi["keys"], wi.get("keysecondary", ""), 
                                                wi.get("folder", "root"), wi.get("constant", False), 
                                                wi["entry"], wi.get("comment", ""))
                                                
            

        # Reset current save
        koboldai_vars.savedir = getcwd()+"\\stories"
        
        # Refresh game screen
        koboldai_vars.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, broadcast=True, room="UI_1")
        setgamesaved(False)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")

#==================================================================#
#  Import World Info JSON file
#==================================================================#
def wiimportrequest():
    importpath = fileops.getloadpath(koboldai_vars.savedir, "Select World Info File", [("Json", "*.json")])
    if(importpath):
        file = open(importpath, "rb")
        js = json.load(file)
        if(len(js) > 0):
            # If the most recent WI entry is blank, remove it.
            if(not koboldai_vars.worldinfo[-1]["init"]):
                del koboldai_vars.worldinfo[-1]
            # Now grab the new stuff
            num = len(koboldai_vars.worldinfo)
            for wi in js:
                koboldai_vars.worldinfo.append({
                    "key": wi["keys"],
                    "keysecondary": wi.get("keysecondary", ""),
                    "content": wi["entry"],
                    "comment": wi.get("comment", ""),
                    "folder": wi.get("folder", None),
                    "num": num,
                    "init": True,
                    "selective": wi.get("selective", False),
                    "constant": wi.get("constant", False),
                    "uid": None,
                })
                while(True):
                    uid = int.from_bytes(os.urandom(4), "little", signed=True)
                    if(uid not in koboldai_vars.worldinfo_u):
                        break
                koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
                koboldai_vars.worldinfo[-1]["uid"] = uid
                if(koboldai_vars.worldinfo[-1]["folder"]) is not None:
                    koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
                num += 1
            for uid in [None]:
                koboldai_vars.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
                while(True):
                    uid = int.from_bytes(os.urandom(4), "little", signed=True)
                    if(uid not in koboldai_vars.worldinfo_u):
                        break
                koboldai_vars.worldinfo_u[uid] = koboldai_vars.worldinfo[-1]
                koboldai_vars.worldinfo[-1]["uid"] = uid
                if(koboldai_vars.worldinfo[-1]["folder"] is not None):
                    koboldai_vars.wifolders_u[koboldai_vars.worldinfo[-1]["folder"]].append(koboldai_vars.worldinfo[-1])
        
        if not koboldai_vars.quiet:
            print("{0}".format(koboldai_vars.worldinfo[0]))
                
        # Refresh game screen
        setgamesaved(False)
        sendwi()

#==================================================================#
#  Starts a new story
#==================================================================#
def newGameRequest(): 
    # Leave Edit/Memory mode before continuing
    exitModes()
    
    # Clear vars values
    koboldai_vars.gamestarted = False
    koboldai_vars.prompt      = ""
    koboldai_vars.memory      = ""
    koboldai_vars.actions.reset()
    koboldai_vars.actions_metadata = {}
    
    koboldai_vars.authornote  = ""
    koboldai_vars.authornotetemplate = koboldai_vars.setauthornotetemplate
    koboldai_vars.worldinfo   = []
    koboldai_vars.worldinfo_i = []
    koboldai_vars.worldinfo_u = {}
    koboldai_vars.wifolders_d = {}
    koboldai_vars.wifolders_l = []
    koboldai_vars.lastact     = ""
    koboldai_vars.submission  = ""
    koboldai_vars.lastctx     = ""
    
    # Reset current save
    koboldai_vars.savedir = getcwd()+"\\stories"
    
    # Refresh game screen
    koboldai_vars.laststory = None
    emit('from_server', {'cmd': 'setstoryname', 'data': koboldai_vars.laststory}, broadcast=True, room="UI_1")
    setgamesaved(True)
    sendwi()
    emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanote', 'data': koboldai_vars.authornote}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': koboldai_vars.authornotetemplate}, broadcast=True, room="UI_1")
    setStartState()

def randomGameRequest(topic, memory=""): 
    if(koboldai_vars.noai):
        newGameRequest()
        koboldai_vars.memory = memory
        emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")
        return
    koboldai_vars.recentrng = topic
    koboldai_vars.recentrngm = memory
    newGameRequest()
    setgamesaved(False)
    _memory = memory
    if(len(memory) > 0):
        _memory = memory.rstrip() + "\n\n"
    koboldai_vars.memory      = _memory + "You generate the following " + topic + " story concept :"
    koboldai_vars.lua_koboldbridge.feedback = None
    actionsubmit("", force_submit=True, force_prompt_gen=True)
    koboldai_vars.memory      = memory
    emit('from_server', {'cmd': 'setmemory', 'data': koboldai_vars.memory}, broadcast=True, room="UI_1")

def final_startup():
    # Prevent tokenizer from taking extra time the first time it's used
    def __preempt_tokenizer():
        if("tokenizer" not in globals()):
            return
        utils.decodenewlines(tokenizer.decode([25678, 559]))
        tokenizer.encode(utils.encodenewlines("eunoia"))
    tpool.execute(__preempt_tokenizer)

    # Load soft prompt specified by the settings file, if applicable
    if(path.exists("settings/" + getmodelname().replace('/', '_') + ".v2_settings")):
        file = open("settings/" + getmodelname().replace('/', '_') + ".v2_settings", "r")
        js   = json.load(file)
        if(koboldai_vars.allowsp and "softprompt" in js and type(js["softprompt"]) is str and all(q not in js["softprompt"] for q in ("..", ":")) and (len(js["softprompt"]) != 0 and all(js["softprompt"][0] not in q for q in ("/", "\\")))):
            if valid_softprompt("softprompts/"+js["softprompt"]):
                spRequest(js["softprompt"])
        else:
            koboldai_vars.spfilename = ""
        file.close()

    # Precompile TPU backend if required
    if model and model.capabilties.uses_tpu:
        model.raw_generate([23403, 727, 20185], max_new=1)

    # Set the initial RNG seed
    set_seed()

def send_debug():
    if koboldai_vars.debug:
        debug_info = ""
        try:
            debug_info = "{}Seed: {} ({})\n".format(debug_info, repr(__import__("tpu_mtj_backend").get_rng_seed() if koboldai_vars.use_colab_tpu else __import__("torch").initial_seed()), "specified by user in settings file" if koboldai_vars.seed_specified else "randomly generated")
        except:
            pass
        try:
            debug_info = "{}Newline Mode: {}\n".format(debug_info, koboldai_vars.newlinemode)
        except:
            pass
        try:
            debug_info = "{}Action Length: {}\n".format(debug_info, koboldai_vars.actions.get_last_key())
        except:
            pass
        try:
            debug_info = "{}Actions Metadata Length: {}\n".format(debug_info, max(koboldai_vars.actions_metadata) if len(koboldai_vars.actions_metadata) > 0 else 0)
        except:
            pass
        try:
            debug_info = "{}Actions: {}\n".format(debug_info, [k for k in koboldai_vars.actions])
        except:
            pass
        try:
            debug_info = "{}Actions Metadata: {}\n".format(debug_info, [k for k in koboldai_vars.actions_metadata])
        except:
            pass
        try:
            debug_info = "{}Last Action: {}\n".format(debug_info, koboldai_vars.actions[koboldai_vars.actions.get_last_key()])
        except:
            pass
        try:
            debug_info = "{}Last Metadata: {}\n".format(debug_info, koboldai_vars.actions_metadata[max(koboldai_vars.actions_metadata)])
        except:
            pass

        emit('from_server', {'cmd': 'debug_info', 'data': debug_info}, broadcast=True, room="UI_1")


#==================================================================#
# Load file browser for soft prompts
#==================================================================#
@socketio.on('show_folder_soft_prompt')
def show_folder_soft_prompt(data):
    file_popup("Load Softprompt", "./softprompts", "", renameable=True, folder_only=False, editable=False, deleteable=True, jailed=True, item_check=None)

#==================================================================#
# Load file browser for user scripts
#==================================================================#
@socketio.on('show_folder_usersripts')
def show_folder_usersripts(data):
    file_popup("Load Softprompt", "./userscripts", "", renameable=True, folder_only=False, editable=True, deleteable=True, jailed=True, item_check=None)

#==================================================================#
# KoboldAI Lite Server
#==================================================================#
@app.route('/lite')
@require_allowed_ip
@logger.catch
def lite_html():
    return send_from_directory('static', "klite.html")

#==================================================================#
# UI V2 CODE
#==================================================================#
@app.route('/')
@app.route('/new_ui')
@require_allowed_ip
@logger.catch
def new_ui_index():
    if args.no_ui:
        return redirect('/api/latest')
    if 'story' in session:
        if session['story'] not in koboldai_vars.story_list():
            session['story'] = 'default'
    return render_template(
        'index_new.html',
        settings=gensettings.gensettingstf,
        on_colab=koboldai_vars.on_colab,
        hide_ai_menu=args.noaimenu
    )

@logger.catch
def ui2_connect():
    #Send all variables to client
    logger.debug("Sending full data to client for story {}".format(session['story']))
    koboldai_vars.send_to_ui()
    UI_2_load_cookies()
    UI_2_theme_list_refresh(None)
    pass
    
#==================================================================#
# UI V2 CODE Themes
#==================================================================#
@app.route('/themes/<path:path>')
#@require_allowed_ip
@logger.catch
def ui2_serve_themes(path):
    return send_from_directory('themes', path)
    

#==================================================================#
# File Popup options
#==================================================================#
@socketio.on('upload_file')
@logger.catch
def upload_file(data):
    logger.debug("upload_file {}".format(data['filename']))
    if 'upload_no_save' in data and data['upload_no_save']:
        json_data = json.loads(data['data'].decode("utf-8"))
        loadJSON(json_data)
    else:
        if 'current_folder' in session:
            path = os.path.abspath(os.path.join(session['current_folder'], data['filename']).replace("\\", "/")).replace("\\", "/")
            logger.debug("Want to save to {}".format(path))
            if os.path.join(os.getcwd(), "modeling") in path:
                logger.error("Someone tried to upload something to the modeling directory. As the system loads code dynamically from here we cannot allow that!")
                emit("error_popup", "You tried to upload a file to the modeling directory. This is a secuirty concern and cannot be done.", broadcast=False, room="UI_2");
            elif 'popup_jailed_dir' not in session:
                logger.error("Someone is trying to upload a file to your server. Blocked.")
                emit("error_popup", "Someone is trying to upload a file to your server. Blocked.", broadcast=False, room="UI_2");
            elif session['popup_jailed_dir'] is None:
                if os.path.exists(path):
                    emit("error_popup", "The file already exists. Please delete it or rename the file before uploading", broadcast=False, room="UI_2");
                else:
                    with open(path, "wb") as f:
                        f.write(data['data'])
                    get_files_folders(session['current_folder'])
            elif os.path.abspath(session['popup_jailed_dir']) in os.path.abspath(session['current_folder']):
                if os.path.exists(path):
                    emit("error_popup", "The file already exists. Please delete it or rename the file before uploading", broadcast=False,  room="UI_2");
                else:
                    with open(path, "wb") as f:
                        f.write(data['data'])
                    get_files_folders(session['current_folder'])

@app.route("/upload_kai_story/<string:file_name>", methods=["POST"])
@require_allowed_ip
@logger.catch
def UI_2_upload_kai_story(file_name: str):

    assert "/" not in file_name

    raw_folder_name = file_name.replace(".kaistory", "")
    folder_path = path.join("stories", raw_folder_name)
    disambiguator = 0

    while path.exists(folder_path):
        disambiguator += 1
        folder_path = path.join("stories", f"{raw_folder_name} ({disambiguator})")
    
    buffer = BytesIO()
    dat = request.get_data()
    with open("debug.zip", "wb") as file:
        file.write(dat)
    buffer.write(dat)

    with zipfile.ZipFile(buffer, "r") as zipf:
        zipf.extractall(folder_path)

    return ":)"

@socketio.on('popup_change_folder')
@logger.catch
def popup_change_folder(data):
    if koboldai_vars.debug:
        print("Doing popup change folder: {}".format(data))
    if 'popup_jailed_dir' not in session:
        print("Someone is trying to get at files in your server. Blocked.")
        return
    if session['popup_jailed_dir'] is None:
        get_files_folders(data)
    elif session['popup_jailed_dir'] in data:
        get_files_folders(data)
    else:
        print("User is trying to get at files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data))

@socketio.on('popup_rename')
@logger.catch
def popup_rename(data):
    if 'popup_renameable' not in session:
        print("Someone is trying to rename a file in your server. Blocked.")
        return
    if not session['popup_renameable']:
        print("Someone is trying to rename a file in your server. Blocked.")
        return
    
    if session['popup_jailed_dir'] is None:
        new_filename = os.path.join(os.path.dirname(os.path.abspath(data['file'])), data['new_name']+".json")
        os.rename(data['file'], new_filename)
        get_files_folders(os.path.dirname(data['file']))
    elif session['popup_jailed_dir'] in data:
        new_filename = os.path.join(os.path.dirname(os.path.abspath(data['file'])), data['new_name']+".json")
        os.rename(data['file'], new_filename)
        get_files_folders(os.path.dirname(data['file']))
    else:
        print("User is trying to rename files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data['file']))

@socketio.on('popup_rename_story')
@logger.catch
def popup_rename_story(data):
    if 'popup_renameable' not in session:
        logger.warning("Someone is trying to rename a file in your server. Blocked.")
        return
    if not session['popup_renameable']:
        logger.warning("Someone is trying to rename a file in your server. Blocked.")
        return
    if session['popup_jailed_dir'] and session["popup_jailed_dir"] not in data["file"]:
        logger.warning("User is trying to rename files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data['file']))
        return

    path = data["file"]
    new_name = data["new_name"]
    json_path = path
    is_v3 = False

    # Handle directory for v3 save
    if os.path.isdir(path):
        if not valid_v3_story(path):
            return
        is_v3 = True
        json_path = os.path.join(path, "story.json")

    #if we're using a v2 file we can't just rename the file as the story name is in the file
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    if 'story_name' in json_data:
        json_data['story_name'] = new_name

    # For v3 we move the directory, not the json file.
    if is_v3:
        target = os.path.join(os.path.dirname(path), new_name)
        shutil.move(path, target)

        with open(os.path.join(target, "story.json"), "w") as file:
            json.dump(json_data, file)
    else:
        new_filename = os.path.join(os.path.dirname(os.path.abspath(data['file'])), new_name+".json")
        os.remove(data['file'])
        with open(new_filename, "w") as f:
            json.dump(json_data, f)
    get_files_folders(os.path.dirname(path))

@socketio.on('popup_delete')
@logger.catch
def popup_delete(data):
    if 'popup_deletable' not in session:
        print("Someone is trying to delete a file in your server. Blocked.")
        return
    if not session['popup_deletable']:
        print("Someone is trying to delete a file in your server. Blocked.")
        return
    
    if session['popup_jailed_dir'] is None:
        import shutil
        if os.path.isdir(data):
            shutil.rmtree(data)
        else:
            os.remove(data)
        path = os.path.abspath(data).replace("\\", "/")
        if path[-1] == "/":
            path = path[:-1]
        path = "/".join(path.split("/")[:-1])
        get_files_folders(path)
    elif session['popup_jailed_dir'] in data:
        import shutil
        if os.path.isdir(data):
            shutil.rmtree(data)
        else:
            os.remove(data)
        path = os.path.abspath(data).replace("\\", "/")
        if path[-1] == "/":
            path = path[:-1]
        path = "/".join(path.split("/")[:-1])
        get_files_folders(path)
    else:
        print("User is trying to delete files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data))

@socketio.on('popup_edit')
@logger.catch
def popup_edit(data):
    if 'popup_editable' not in session:
        print("Someone is trying to edit a file in your server. Blocked.")
        return
    if not session['popup_editable']:
        print("Someone is trying to edit a file in your server. Blocked.")
        return
    
    if session['popup_jailed_dir'] is None:
        emit("popup_edit_file", {"file": data, "text": open(data, 'r', encoding='utf-8').read()});
    elif session['popup_jailed_dir'] in data:
        emit("popup_edit_file", {"file": data, "text": open(data, 'r', encoding='utf-8').read()});
    else:
        print("User is trying to delete files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data))

@socketio.on('popup_change_file')
@logger.catch
def popup_change_file(data):
    if 'popup_editable' not in session:
        print("Someone is trying to edit a file in your server. Blocked.")
        return
    if not session['popup_editable']:
        print("Someone is trying to edit a file in your server. Blocked.")
        return
    
    if session['popup_jailed_dir'] is None:
        with open(data['file'], 'w') as f:
            f.write(data['data'])
    elif session['popup_jailed_dir'] in data['file']:
        with open(data['file'], 'w') as f:
            f.write(data['data'])
    else:
        print("User is trying to delete files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data))

@logger.catch
def file_popup(popup_title, starting_folder, return_event, upload=True, jailed=True, folder_only=True, renameable=False, deleteable=False, 
                                                           editable=False, show_breadcrumbs=True, item_check=None, show_hidden=False,
                                                           valid_only=False, hide_extention=False, extra_parameter_function=None,
                                                           column_names=['File Name'], show_filename=True, show_folders=True,
                                                           column_widths=["100%"], sort="Modified", advanced_sort=None, desc=False,
                                                           rename_return_emit_name="popup_rename"):
    #starting_folder = The folder we're going to get folders and/or items from
    #return_event = the socketio event that will be emitted when the load button is clicked
    #jailed = if set to true will look for the session variable jailed_folder and prevent navigation outside of that folder
    #folder_only = will only show folders, no files
    #deletable = will show the delete icons/methods.
    #editable = will show the edit icons/methods
    #show_breadcrumbs = will show the breadcrumbs at the top of the screen
    #item_check will call this function to check if the item is valid as a selection if not none. Will pass absolute directory as only argument to function
    #show_hidden = ... really, you have to ask?
    #valid_only = only show valid files
    #hide_extention = hide extensions
    if jailed:
        session['popup_jailed_dir'] = os.path.abspath(starting_folder).replace("\\", "/")
    else:
        session['popup_jailed_dir'] = None
    session['popup_deletable'] = deleteable
    session['popup_renameable'] = renameable
    session['popup_editable'] = editable
    session['popup_show_hidden'] = show_hidden
    session['popup_item_check'] = item_check
    session['extra_parameter_function'] = extra_parameter_function
    session['column_names'] = column_names
    session['popup_folder_only'] = folder_only
    session['popup_show_breadcrumbs'] = show_breadcrumbs
    session['upload'] = upload
    session['valid_only'] = valid_only
    session['hide_extention'] = hide_extention
    session['show_filename'] = show_filename
    session['column_widths'] = column_widths
    session['sort'] = sort
    session['desc'] = desc
    session['show_folders'] = show_folders
    session['advanced_sort'] = advanced_sort
    
    emit("load_popup", {"popup_title": popup_title, "call_back": return_event, "renameable": renameable, "deleteable": deleteable, "editable": editable, 'upload': upload, "rename_return_emit_name": rename_return_emit_name}, broadcast=False)
    emit("load_popup", {"popup_title": popup_title, "call_back": return_event, "renameable": renameable, "deleteable": deleteable, "editable": editable, 'upload': upload, "rename_return_emit_name": rename_return_emit_name}, broadcast=True, room="UI_1")
    
    get_files_folders(starting_folder)

@logger.catch
def get_files_folders(starting_folder):
    import stat
    session['current_folder'] = os.path.abspath(starting_folder).replace("\\", "/")
    item_check = globals()[session['popup_item_check']] if session['popup_item_check'] is not None else None
    extra_parameter_function = globals()[session['extra_parameter_function']] if session['extra_parameter_function'] is not None else None
    show_breadcrumbs = session['popup_show_breadcrumbs']
    show_hidden = session['popup_show_hidden']
    folder_only = session['popup_folder_only']
    valid_only = session['valid_only']
    column_names = session['column_names']
    hide_extention = session['hide_extention']
    show_filename = session['show_filename']
    column_widths = session['column_widths']
    sort = session['sort']
    desc = session['desc']
    show_folders = session['show_folders']
    advanced_sort = globals()[session['advanced_sort']] if session['advanced_sort'] is not None else None
    
    if starting_folder == 'This PC':
        breadcrumbs = [['This PC', 'This PC']]
        items = [["{}:/".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]
    else:
        path = os.path.abspath(starting_folder).replace("\\", "/")
        if path[-1] == "/":
            path = path[:-1]
        breadcrumbs = []
        for i in range(len(path.split("/"))):
            breadcrumbs.append(["/".join(path.split("/")[:i+1]),
                                 path.split("/")[i]])
        if len(breadcrumbs) == 1:
            breadcrumbs = [["{}:/".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]
        else:
            if len([["{}:/".format(chr(i)), "{}:\\".format(chr(i))] for i in range(65, 91) if os.path.exists("{}:".format(chr(i)))]) > 0:
                breadcrumbs.insert(0, ['This PC', 'This PC'])
        
        #if we're jailed, remove the stuff before the jail from the breadcrumbs
        if session['popup_jailed_dir'] is not None:
            
            breadcrumbs = breadcrumbs[len(session['popup_jailed_dir'].split("/")):]
        
        folders = []
        files = []
        base_path = os.path.abspath(starting_folder).replace("\\", "/")

        if advanced_sort is not None:
            files_to_check = advanced_sort(base_path, desc=desc)
        else:
            files_to_check = get_files_sorted(base_path, sort, desc=desc)

        for item in files_to_check:
            item_full_path = os.path.join(base_path, item).replace("\\", "/")
            if hasattr(os.stat(item_full_path), "st_file_attributes"):
                hidden = bool(os.stat(item_full_path).st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)
            else:
                hidden = item[0] == "."
            if item_check is None:
                valid_selection = True
            else:
                valid_selection = item_check(item_full_path)
            if extra_parameter_function is None:
                extra_parameters = []
            else:
                extra_parameters = extra_parameter_function(item_full_path, item, valid_selection)
                
            if (show_hidden and hidden) or not hidden:
                if os.path.isdir(item_full_path):
                    folders.append([
                        # While v3 saves are directories, we should not show them as such.
                        not valid_v3_story(item_full_path),
                        item_full_path,
                        item,
                        valid_selection,
                        extra_parameters
                    ])
                else:
                    if hide_extention:
                        item = ".".join(item.split(".")[:-1])
                    if valid_only:
                        if valid_selection:
                            files.append([False, item_full_path, item,  valid_selection, extra_parameters])
                    else:
                        files.append([False, item_full_path, item,  valid_selection, extra_parameters])
                        
        if show_folders:
            items = folders
        else:
            items = []
        if not folder_only:
            items += files
            
    #items is a list of [Folder True/False, full path, file/folder name, validity of item to load, [list of extra columns]]
    emit("popup_items", {"items": items, "column_names": column_names, "show_filename": show_filename, "column_widths": column_widths}, broadcast=False)
    socketio.emit("popup_items", items, broadcast=True, include_self=True, room="UI_1")
    if show_breadcrumbs:
        emit("popup_breadcrumbs", breadcrumbs, broadcast=False)
        socketio.emit("popup_breadcrumbs", breadcrumbs, broadcast=True, room="UI_1")

@logger.catch
def get_files_sorted(path, sort, desc=False):
    data = {}
    for file in os.scandir(path=path):
        if sort == "Modified":
            data[file.name] = datetime.datetime.fromtimestamp(file.stat().st_mtime)
        elif sort == "Accessed":
            data[file.name] = datetime.datetime.fromtimestamp(file.stat().st_atime)
        elif sort == "Created":
            data[file.name] = datetime.datetime.fromtimestamp(file.stat().st_ctime)
        elif sort == "Name":
            data[file.name] = file.name
            
    return [key[0] for key in sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=desc)]
        

@socketio.on("configure_prompt")
@logger.catch
def UI_2_configure_prompt(data):
    import_buffer.replace_placeholders(data)
    import_buffer.commit()

#==================================================================#
# Event triggered when browser SocketIO detects a variable change
#==================================================================#
@socketio.on('var_change')
@logger.catch
def UI_2_var_change(data):
    if 'value' not in data:
        logger.error("Got a variable change without a value. Data Packet: {}".format(data))
        return
    classname = data['ID'].split("_")[0]
    name = data['ID'][len(classname)+1:]
    classname += "_settings"
    
    #Need to fix the data type of value to match the module
    if type(getattr(koboldai_vars, name)) == int:
        value = int(data['value'])
    elif type(getattr(koboldai_vars, name)) == float:
        value = float(data['value'])
    elif type(getattr(koboldai_vars, name)) == bool:
        value = bool(data['value'])
    elif type(getattr(koboldai_vars, name)) == str:
        value = str(data['value'])
    elif type(getattr(koboldai_vars, name)) == list:
        value = list(data['value'])
    elif type(getattr(koboldai_vars, name)) == dict:
        value = dict(data['value'])
    else:
        raise ValueError("Unknown Type {} = {}".format(name, type(getattr(koboldai_vars, name))))
    
    #print("Setting {} to {} as type {}".format(name, value, type(value)))
    setattr(koboldai_vars, name, value)
    
    #Now let's save except for story changes
    if classname != "story_settings":
        if classname == "model_settings":
            filename = "settings/{}.v2_settings".format(koboldai_vars.model.replace("/", "_"))
        else:
            filename = "settings/{}.v2_settings".format(classname)
        
        if not os.path.exists("settings"):
            os.mkdir("settings")
        with open(filename, "w") as settings_file:
            settings_file.write(getattr(koboldai_vars, "_{}".format(classname)).to_json())
    
    if name in ['seed', 'seed_specified']:
        set_seed()
    
    return {'id': data['ID'], 'status': "Saved"}
    
    
#==================================================================#
# Set the random seed (or constant seed) for generation
#==================================================================#
def set_seed():
    print("Setting Seed")
    if(koboldai_vars.seed is not None):
        if(koboldai_vars.use_colab_tpu):
            if(koboldai_vars.seed_specified):
                __import__("tpu_mtj_backend").set_rng_seed(koboldai_vars.seed)
            else:
                __import__("tpu_mtj_backend").randomize_rng_seed()
        else:
            if(koboldai_vars.seed_specified):
                __import__("torch").manual_seed(koboldai_vars.seed)
            else:
                __import__("torch").seed()
    koboldai_vars.seed = __import__("tpu_mtj_backend").get_rng_seed() if koboldai_vars.use_colab_tpu else __import__("torch").initial_seed()

#==================================================================#
# Saving Story
#==================================================================#
@socketio.on('save_story')
@logger.catch
def UI_2_save_story(data):
    if koboldai_vars.debug:
        print("Saving Story")
    if data is None:
        #We need to check to see if there is a file already and if it's not the same story so we can ask the client if this is OK
        save_name = koboldai_vars.story_name if koboldai_vars.story_name != "" else "untitled"
        same_story = True
        if os.path.exists("stories/{}".format(save_name)):
            with open("stories/{}/story.json".format(save_name), "r") as settings_file:
                json_data = json.load(settings_file)
                if 'story_id' in json_data:
                    same_story = json_data['story_id'] == koboldai_vars.story_id
                else:
                    same_story = False
        
        if same_story:
            koboldai_vars.save_story()
            return "OK"
        else:
            return "overwrite?"
    else:    
        #We have an ack that it's OK to save over the file if one exists
        koboldai_vars.save_story()

def directory_to_zip_data(directory: str, overrides: Optional[dict]) -> bytes:
    overrides = overrides or {}
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w") as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                p = os.path.join(root, file)
                z_path = os.path.join(*p.split(os.path.sep)[2:])

                if z_path in overrides:
                    continue

                zipf.write(p, z_path)

        for path, contents in overrides.items():
            zipf.writestr(path, contents)

    return buffer.getvalue()

#==================================================================#
# Save story to json
#==================================================================#
@app.route("/story_download")
@require_allowed_ip
@logger.catch
def UI_2_download_story():
    if args.no_ui:
        return redirect('/api/latest')
    save_exists = path.exists(koboldai_vars.save_paths.base)
    if koboldai_vars.gamesaved and save_exists:
        # Disk is up to date; download from disk
        data = directory_to_zip_data(koboldai_vars.save_paths.base)
    elif save_exists:
        # We aren't up to date but we are saved; patch what disk gives us
        data = directory_to_zip_data(
            koboldai_vars.save_paths.base,
            {"story.json": koboldai_vars.to_json("story_settings")}
        )
    else:
        # We are not saved; send json in zip from memory
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zipf:
            zipf.writestr("story.json", koboldai_vars.to_json("story_settings"))
        data = buffer.getvalue()

    return Response(
        data,
        mimetype="application/octet-stream",
        headers={"Content-disposition": f"attachment; filename={koboldai_vars.story_name}.kaistory"}
    )
    
    
#==================================================================#
# Event triggered when Selected Text is edited
#==================================================================#
@socketio.on('Set Selected Text')
@logger.catch
def UI_2_Set_Selected_Text(data):
    if not koboldai_vars.quiet:
        logger.info("Updating Selected Text: {}".format(data))
    action_id = int(data["id"])

    if not koboldai_vars.actions.actions[action_id].get("Original Text"):
        koboldai_vars.actions.actions[action_id]["Original Text"] = data["text"]

    koboldai_vars.actions[action_id] = data['text']

#==================================================================#
# Event triggered when Option is Selected
#==================================================================#
@socketio.on('Use Option Text')
@logger.catch
def UI_2_Use_Option_Text(data):
    koboldai_vars.actions.show_options(False)
    if koboldai_vars.prompt == "":
        koboldai_vars.prompt = koboldai_vars.actions.get_current_options()[int(data['option'])]['text']
        koboldai_vars.actions.clear_unused_options()
    else:
        koboldai_vars.actions.use_option(int(data['option']), action_step=int(data['chunk']))

#==================================================================#
# Event triggered when Option is Selected
#==================================================================#
@socketio.on('delete_option')
@logger.catch
def UI_2_delete_option(data):
    koboldai_vars.actions.delete_option(int(data['option']), action_step=int(data['chunk']))

#==================================================================#
# Event triggered when user clicks the submit button
#==================================================================#
@socketio.on('submit')
@logger.catch
def UI_2_submit(data):
    if not koboldai_vars.noai and data['theme']:
        # Random prompt generation
        logger.debug("doing random prompt")
        memory = koboldai_vars.memory
        koboldai_vars.memory = "{}\n\nYou generate the following {} story concept :".format(koboldai_vars.memory, data['theme'])
        koboldai_vars.lua_koboldbridge.feedback = None
        actionsubmit("", force_submit=True, force_prompt_gen=True)
        koboldai_vars.memory = memory
        return

    logger.debug("doing normal input")
    koboldai_vars.actions.clear_unused_options()
    koboldai_vars.lua_koboldbridge.feedback = None
    koboldai_vars.recentrng = koboldai_vars.recentrngm = None

    gen_mode_name = data.get("gen_mode", None) or "standard"
    try:
        gen_mode = GenerationMode(gen_mode_name)
    except ValueError:
        # Invalid enum lookup!
        gen_mode = GenerationMode.STANDARD
        logger.warning(f"Unknown gen_mode '{gen_mode_name}', using STANDARD! Report this!")

    actionsubmit(data['data'], actionmode=koboldai_vars.actionmode, gen_mode=gen_mode)

 #==================================================================#
# Event triggered when user clicks the submit button
#==================================================================#
@socketio.on('abort')
@logger.catch
def UI_2_abort(data):
    if koboldai_vars.debug:
        print("got abort")
    koboldai_vars.abort = True

 
#==================================================================#
# Event triggered when user clicks the pin button
#==================================================================#
@socketio.on('Pinning')
@logger.catch
def UI_2_Pinning(data):
    koboldai_vars.actions.toggle_pin(int(data['chunk']), int(data['option']))
    
#==================================================================#
# Event triggered when user clicks the back button
#==================================================================#
@socketio.on('back')
@logger.catch
def UI_2_back(data):
    if koboldai_vars.aibusy:
        return
    if koboldai_vars.debug:
        print("back")
    koboldai_vars.actions.clear_unused_options()
    ignore = koboldai_vars.actions.pop()
    
#==================================================================#
# Event triggered when user clicks the redo button
#==================================================================#
@socketio.on('redo')
@logger.catch
def UI_2_redo(data):
    if koboldai_vars.aibusy:
        return
    koboldai_vars.actions.go_forward()
    

#==================================================================#
# Event triggered when user clicks the retry button
#==================================================================#
@socketio.on('retry')
@logger.catch
def UI_2_retry(data):
    if koboldai_vars.aibusy:
        return
    if len(koboldai_vars.actions.get_current_options_no_edits()) == 0:
        ignore = koboldai_vars.actions.pop(keep=True)
    koboldai_vars.actions.clear_unused_options()
    koboldai_vars.lua_koboldbridge.feedback = None
    koboldai_vars.recentrng = koboldai_vars.recentrngm = None
    actionsubmit("", actionmode=koboldai_vars.actionmode)
    
#==================================================================#
# Event triggered when user clicks the load model button
#==================================================================#
@socketio.on('load_model_button')
@logger.catch
def UI_2_load_model_button(data):
    emit("open_model_load_menu", {"items": [{**item.to_json(), **{"menu":"mainmenu"}} for item in model_menu['mainmenu'] if item.should_show()]})
    

    
#==================================================================#
# Event triggered when user clicks the a model
#==================================================================#
@socketio.on('select_model')
@logger.catch
def UI_2_select_model(data):
    global model_backend_type_crosswalk #No idea why I have to make this a global where I don't for model_backends...
    logger.debug("Clicked on model entry: {}".format(data))
    if data["name"] in model_menu and data['ismenu'] == "true":
        emit("open_model_load_menu", {"items": [{**item.to_json(), **{"menu":data["name"]}} for item in model_menu[data["name"]] if item.should_show()]})
    else:
        #Get load methods
        if 'ismenu' in data and data['ismenu'] == 'false':
            valid_loaders = {}
            if data['id'] in [item.name for sublist in model_menu for item in model_menu[sublist]]:
                #Here if we have a model id that's in our menu, we explicitly use that backend
                for model_backend_type in set([item.model_backend for sublist in model_menu for item in model_menu[sublist] if item.name == data['id']]):
                    for model_backend in model_backend_type_crosswalk[model_backend_type]:
                        valid_loaders[model_backend] = model_backends[model_backend].get_requested_parameters(data["name"], data["path"] if 'path' in data else None, data["menu"])
                emit("selected_model_info", {"model_backends": valid_loaders})
            else:
                #Here we have a model that's not in our menu structure (either a custom model or a custom path
                #so we'll just go through all the possible loaders
                for model_backend in sorted(
                    model_backends,
                    key=lambda x: PRIORITIZED_BACKEND_MODULES.get(model_backend_module_names[x], 0),
                    reverse=True,
                ):
                    if model_backends[model_backend].is_valid(data["name"], data["path"] if 'path' in data else None, data["menu"]):
                        valid_loaders[model_backend] = model_backends[model_backend].get_requested_parameters(data["name"], data["path"] if 'path' in data else None, data["menu"])
                emit("selected_model_info", {"model_backends": valid_loaders})
        else:
            #Get directories
            paths, breadcrumbs = get_folder_path_info(data['path'])
            output = []
            for path in paths:
                valid=False
                for model_backend in model_backends:
                    if model_backends[model_backend].is_valid(path[1], path[0], "Custom"):
                        logger.debug("{} says valid".format(model_backend))
                        valid=True
                        break
                    else:
                        logger.debug("{} says invalid".format(model_backend))
                    
                output.append({'label': path[1], 'name': path[1], 'size': "", "menu": "Custom", 'path': path[0], 'isMenu': not valid})
            emit("open_model_load_menu", {"items": output+[{'label': 'Return to Main Menu', 'name':'mainmenu', 'size': "", "menu": "Custom", 'isMenu': True}], 'breadcrumbs': breadcrumbs})            
    return




#==================================================================#
# Event triggered when user changes a model parameter and it's set to resubmit
#==================================================================#
@socketio.on('resubmit_model_info')
@logger.catch
def UI_2_resubmit_model_info(data):
    valid_loaders = {}
    for model_backend in data['valid_backends']:
        valid_loaders[model_backend] = model_backends[model_backend].get_requested_parameters(data["name"], data["path"] if 'path' in data else None, data["menu"], parameters=data)
    emit("selected_model_info", {"model_backends": valid_loaders, 'selected_model_backend': data['plugin']})

#==================================================================#
# Event triggered when user loads a model
#==================================================================#
@socketio.on('load_model')
@logger.catch
def UI_2_load_model(data):
    logger.debug("Unloading previous model")
    if 'model' in globals():
        model.unload()
    logger.debug("Loading model with user input of: {}".format(data))
    model_backends[data['plugin']].set_input_parameters(data)
    load_model(data['plugin'])
    #load_model(use_gpu=data['use_gpu'], gpu_layers=data['gpu_layers'], disk_layers=data['disk_layers'], online_model=data['online_model'], url=koboldai_vars.colaburl, use_8_bit=data['use_8_bit'])

#==================================================================#
# Event triggered when load story is clicked
#==================================================================#
@socketio.on('load_story_list')
@logger.catch
def UI_2_load_story_list(data):
    file_popup("Select Story to Load", "./stories", "load_story", upload=True, jailed=True, folder_only=False, renameable=True, 
                                                                  deleteable=True, show_breadcrumbs=True, item_check="valid_story",
                                                                  valid_only=True, hide_extention=True, extra_parameter_function="get_story_listing_data",
                                                                  column_names=['Story Name', 'Action Count', 'Last Loaded'], show_filename=False,
                                                                  column_widths=['minmax(150px, auto)', '140px', '160px'], advanced_sort="story_sort",
                                                                  sort="Modified", desc=True, rename_return_emit_name="popup_rename_story")

@logger.catch
def get_story_listing_data(item_full_path, item, valid_selection):
    title = ""
    action_count = -1
    last_loaded = ""

    if not valid_selection:
        return [title, action_count, last_loaded]
    
    if os.path.isdir(item_full_path):
        if not valid_v3_story(item_full_path):
            return [title, action_count, last_loaded]
        item_full_path = os.path.join(item_full_path, "story.json")

    with open(item_full_path, 'rb') as f:
        parse_event = ijson.parse(f)
        depth=0
        file_version=1
        while True:
            try:
                prefix, event, value = next(parse_event)
            except StopIteration:
                break
            depth+=1
            if depth > 100 or prefix == 'file_version':
                if prefix == 'file_version':
                    file_version=2
                break
                
    with open(item_full_path, 'rb') as f:
        parse_event = ijson.parse(f)
        if file_version == 1:
            title = ".".join(item.split(".")[:-1])
        else:
            for prefix, event, value in parse_event:
                if prefix == 'story_name':
                    title = value
                    break
                
    with open(item_full_path, 'rb') as f:
        parse_event = ijson.parse(f)
        if file_version == 2:
            for prefix, event, value in parse_event:
                if prefix == 'actions.action_count':
                    action_count = value+1
                    break
        else:
            if os.path.getsize(item_full_path)/1024/1024 <= 20:
                action_count=0
                for prefix, event, value in parse_event:
                    if prefix == 'actions.item':
                        action_count+=1
            else:
                action_count = "{}MB".format(round(os.path.getsize(item_full_path)/1024/1024,1))

    if title in koboldai_vars._system_settings.story_loads:
        # UNIX Timestamp
        last_loaded = int(time.mktime(time.strptime(koboldai_vars._system_settings.story_loads[title], "%m/%d/%Y, %H:%M:%S")))
    else:
        last_loaded = os.path.getmtime(item_full_path)

    return [title, action_count, last_loaded]
    
@logger.catch
def valid_story(path: str):
    if os.path.isdir(path):
        return valid_v3_story(path)

    if not path.endswith(".json"):
        return False

    if os.path.exists(path.replace(".json", "/story.json")):
        return False

    try:
        with open(path, 'rb') as file:
            parser = ijson.parse(file)
            for prefix, event, value in parser:
                if prefix == 'memory':
                    return True
    except:
        pass
    return False

@logger.catch
def valid_v3_story(path: str) -> bool:
    if not os.path.exists(path): return False
    if not os.path.isdir(path): return False
    if not os.path.exists(os.path.join(path, "story.json")): return False
    return True

@logger.catch
def story_sort(base_path, desc=False):
    files = {}
    for file in os.scandir(path=base_path):
        if file.is_dir():
            if not valid_v3_story(file.path):
                continue

            story_path = os.path.join(file.path, "story.json")
            story_stat = os.stat(story_path)

            if os.path.getsize(story_path) < 2*1024*1024: #2MB
                with open(story_path, "r") as f:
                    j = json.load(f)
                    if j.get("story_name") in koboldai_vars.story_loads:
                        files[file.name] = datetime.datetime.strptime(koboldai_vars.story_loads[j["story_name"]], "%m/%d/%Y, %H:%M:%S")
                    else:
                        files[file.name] = datetime.datetime.fromtimestamp(story_stat.st_mtime)
            else:
                files[file.name] = datetime.datetime.fromtimestamp(story_stat.st_mtime)
            continue
        
        if not file.name.endswith(".json"):
            continue

        filename = os.path.join(base_path, file.name).replace("\\", "/")
        if os.path.getsize(filename) < 2*1024*1024: #2MB
            with open(filename, "r") as f:
                try:
                    js = json.load(f)
                    if 'story_name' in js and js['story_name'] in koboldai_vars.story_loads:
                        files[file.name] = datetime.datetime.strptime(koboldai_vars.story_loads[js['story_name']], "%m/%d/%Y, %H:%M:%S")
                    else:
                        files[file.name] = datetime.datetime.fromtimestamp(file.stat().st_mtime)
                except:
                    pass
        else:
            files[file.name] = datetime.datetime.fromtimestamp(file.stat().st_mtime)
    return [key[0] for key in sorted(files.items(), key=lambda kv: (kv[1], kv[0]), reverse=desc)]


#==================================================================#
# Event triggered on load story
#==================================================================#
@socketio.on('load_story')
@logger.catch
def UI_2_load_story(file):
    start_time = time.time()
    logger.debug("got a call or loading a story: {}".format(file))
    if koboldai_vars.debug:
        print("loading {}".format(file))
    loadRequest(file)
    logger.debug("Load Story took {}s".format(time.time()-start_time))

#==================================================================#
# Event triggered on new story
#==================================================================#
@socketio.on('new_story')
@logger.catch
def UI_2_new_story(data):
    logger.info("Starting new story")
    koboldai_vars.create_story("")
    
    
#==================================================================#
# Event triggered when user moves world info
#==================================================================#
@socketio.on('move_wi')
@logger.catch
def UI_2_move_wi(data):
    if data['folder'] is None:
        koboldai_vars.worldinfo_v2.reorder(int(data['dragged_id']), int(data['drop_id']))
    else:
        koboldai_vars.worldinfo_v2.add_item_to_folder(int(data['dragged_id']), data['folder'], before=int(data['drop_id']))

#==================================================================#
# Event triggered when user moves world info
#==================================================================#
@socketio.on('wi_set_folder')
@logger.catch
def UI_2_wi_set_folder(data):
    koboldai_vars.worldinfo_v2.add_item_to_folder(int(data['dragged_id']), data['folder'])

#==================================================================#
# Event triggered when user renames world info folder
#==================================================================#
@socketio.on('Rename_World_Info_Folder')
@logger.catch
def UI_2_Rename_World_Info_Folder(data):
    if koboldai_vars.debug:
        print("Rename_World_Info_Folder")
        print(data)
    koboldai_vars.worldinfo_v2.rename_folder(data['old_folder'], data['new_folder'])

#==================================================================#
# Event triggered when user edits world info item
#==================================================================#
@socketio.on('edit_world_info')
@logger.catch
def UI_2_edit_world_info(data):
    if koboldai_vars.debug:
        print("edit_world_info")
        print(data)
    
    if data['uid'] < 0:
        logger.debug("Creating WI: {}".format(data))
        koboldai_vars.worldinfo_v2.add_item(data['title'], data['key'], 
                                             data['keysecondary'], data['folder'], 
                                             data['constant'], data['manual_text'], 
                                             data['comment'], wpp=data['wpp'],
                                             use_wpp=data['use_wpp'], object_type=data["object_type"])
        emit("delete_new_world_info_entry", {})
    else:
        logger.debug("Editting WI: {}".format(data))
        koboldai_vars.worldinfo_v2.edit_item(data['uid'], data['title'], data['key'], 
                                             data['keysecondary'], data['folder'], 
                                             data['constant'], data['manual_text'], 
                                             data['comment'], wi_type=data["type"],
                                             wpp=data['wpp'], use_wpp=data['use_wpp'],
                                             object_type=data["object_type"])


#==================================================================#
# Event triggered when user creates world info folder
#==================================================================#
@socketio.on('create_world_info_folder')
@logger.catch
def UI_2_create_world_info_folder(data):
    koboldai_vars.worldinfo_v2.add_folder("New Folder")

#==================================================================#
# Event triggered when user deletes world info item
#==================================================================#
@socketio.on('delete_world_info')
@logger.catch
def UI_2_delete_world_info(uid):
    koboldai_vars.worldinfo_v2.delete(uid)


#==================================================================#
# Event triggered when user deletes world info folder
#==================================================================#
@socketio.on('delete_wi_folder')
@logger.catch
def UI_2_delete_wi_folder(folder):
    koboldai_vars.worldinfo_v2.delete_folder(folder)


#==================================================================#
# Event triggered when user exports world info folder
#==================================================================#
@app.route('/export_world_info_folder')
@require_allowed_ip
@logger.catch
def UI_2_export_world_info_folder():
    if 'folder' in request.args:
        data = koboldai_vars.worldinfo_v2.to_json(folder=request.args['folder'])
        folder = request.args['folder']
    else:
        data = koboldai_vars.worldinfo_v2.to_json()
        folder = koboldai_vars.story_name
    return Response(
        json.dumps(data, indent="\t"),
        mimetype="application/json",
        headers={"Content-disposition":
                 "attachment; filename={}_world_info.json".format(folder)}
        )

#==================================================================#
# Event triggered when user exports world info folder
#==================================================================#
@socketio.on('upload_world_info_folder')
@logger.catch
def UI_2_upload_world_info_folder(data):
    json_data = json.loads(data['data'])
    koboldai_vars.worldinfo_v2.load_json(json_data, folder=data['folder'])
    logger.debug("Calcing AI Text from WI Upload")
    koboldai_vars.calc_ai_text()

@app.route("/upload_wi", methods=["POST"])
@require_allowed_ip
@logger.catch
def UI_2_import_world_info():
    wi_data = request.get_json()
    uids = {}

    for folder_name, children in wi_data["folders"].items():
        koboldai_vars.worldinfo_v2.add_folder(folder_name)
        for child in children:
            # Child is index
            if child not in uids:
                entry_data = wi_data["entries"][child]
                uids[child] = koboldai_vars.worldinfo_v2.add_item(
                    title=entry_data["title"],
                    key=entry_data["key"],
                    keysecondary=entry_data["keysecondary"],
                    folder=folder_name,
                    constant=entry_data["constant"],
                    manual_text=entry_data["manual_text"],
                    comment=entry_data["comment"],
                    use_wpp=entry_data["use_wpp"],
                    wpp=entry_data["wpp"],
                )
            koboldai_vars.worldinfo_v2.add_item_to_folder(uids[child], folder_name)
    return ":)"

@socketio.on("search_wi")
@logger.catch
def UI_2_search_wi(data):
    query = data["query"].lower()
    full_data = koboldai_vars.worldinfo_v2.to_json()

    results = {"title": [], "key": [], "keysecondary": [], "manual_text": []}

    for entry in full_data["entries"].values():
        # Order matters for what's more important.
        if query in entry["title"].lower():
            results["title"].append(entry)
        elif any([query in k.lower() for k in entry["key"]]):
            results["key"].append(entry)
        elif any([query in k.lower() for k in entry["keysecondary"]]):
            results["keysecondary"].append(entry)
        elif query in entry["content"].lower():
            results["manual_text"].append(entry)
        elif query in entry["manual_text"].lower():
            results["comment"].append(entry)

    emit("wi_results", results, broadcast=True, room="UI_2")

@socketio.on("update_wi_attribute")
@logger.catch
def UI_2_update_wi_attribute(data):
    uid, key, value = data["uid"], data["key"], data["value"]
    koboldai_vars.worldinfo_v2.world_info[uid][key] = value
    socketio.emit("world_info_entry", koboldai_vars.worldinfo_v2.world_info[uid], broadcast=True, room="UI_2")

@socketio.on("update_wi_keys")
@logger.catch
def UI_2_update_wi_keys(data):
    uid, key, is_secondary, operation = data["uid"], data["key"], data["is_secondary"], data["operation"]

    keykey = "key" if not is_secondary else "keysecondary"
    key_exists = key in koboldai_vars.worldinfo_v2.world_info[uid][keykey]

    if operation == "add":
        if not key_exists:
            koboldai_vars.worldinfo_v2.world_info[uid][keykey].append(key)
    elif operation == "remove":
        if key_exists:
            koboldai_vars.worldinfo_v2.world_info[uid][keykey].remove(key)

    if keykey == "keysecondary":
        koboldai_vars.worldinfo_v2.world_info[uid]["selective"] = len(koboldai_vars.worldinfo_v2.world_info[uid]["keysecondary"]) > 0

    # Send to UI
    socketio.emit("world_info_entry", koboldai_vars.worldinfo_v2.world_info[uid], broadcast=True, room="UI_2")

@app.route("/set_wi_image/<int(signed=True):uid>", methods=["POST"])
@require_allowed_ip
@logger.catch
def UI_2_set_wi_image(uid):
    if uid < 0:
        socketio.emit("delete_new_world_info_entry", {})
        uid = koboldai_vars.worldinfo_v2.add_item(
            "New World Info Entry",
            [],
            [],
            None,
            False,
            "",
            "",
        )

    data = base64.b64decode(request.get_data(as_text=True).split(",")[-1])
    path = os.path.join(koboldai_vars.save_paths.wi_images, str(uid))

    if not data:
        # Delete if sent null image
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    else:
        try:
            # Otherwise assign image
            with open(path, "wb") as file:
                file.write(data)
        except FileNotFoundError:
            show_error_notification(
                "Unable to write image",
                "Please save the game before uploading images."
            )
            return ":(", 500
    koboldai_vars.gamesaved = False
    return ":)", 200

@app.route("/get_wi_image/<int(signed=True):uid>", methods=["GET"])
@require_allowed_ip
@logger.catch
def UI_2_get_wi_image(uid):
    if args.no_ui:
        return redirect('/api/latest')
    path = os.path.join(koboldai_vars.save_paths.wi_images, str(uid))
    try:
        return send_file(path)
    except FileNotFoundError:
        return ":( Couldn't find image", 204

@app.route("/set_commentator_picture/<int(signed=True):commentator_id>", methods=["POST"])
@require_allowed_ip
@logger.catch
def UI_2_set_commentator_image(commentator_id):
    data = request.get_data()
    with open(os.path.join(koboldai_vars.save_paths.commentator_pictures, str(commentator_id)), "wb") as file:
        file.write(data)
    return ":)"

@app.route("/image_db.json", methods=["GET"])
@require_allowed_ip
@logger.catch
def UI_2_get_image_db():
    if args.no_ui:
        return redirect('/api/latest')
    try:
        return send_file(os.path.join(koboldai_vars.save_paths.generated_images, "db.json"))
    except FileNotFoundError:
        return jsonify([])

@app.route("/action_composition.json", methods=["GET"])
@require_allowed_ip
@logger.catch
def UI_2_get_action_composition():
    if args.no_ui:
        return redirect('/api/latest')
    try:
        actions = request.args.get("actions").split(",")
        if not actions:
            raise ValueError()
    except (ValueError, AttributeError):
        return "No actions", 400

    try:
        actions = [int(action) for action in actions]
    except TypeError:
        return "Not all actions int", 400

    ret = []
    for action_id in actions:
        try:
            ret.append(koboldai_vars.actions.get_action_composition(action_id))
        except KeyError:
            ret.append([])
    return jsonify(ret)

@app.route("/generated_images/<path:path>")
@require_allowed_ip
def UI_2_send_generated_images(path):
    return send_from_directory(koboldai_vars.save_paths.generated_images, path)

@socketio.on("scratchpad_prompt")
@logger.catch
def UI_2_scratchpad_prompt(data):
    out_text = model.raw_generate(
        data,
        max_new=80,
    ).decoded

    socketio.emit("scratchpad_response", out_text, broadcast=True, room="UI_2")


#==================================================================#
# Event triggered when user edits phrase biases
#==================================================================#
@socketio.on('phrase_bias_update')
@logger.catch
def UI_2_phrase_bias_update(biases):
    koboldai_vars.biases = biases


@socketio.on("substitution_update")
@logger.catch
def UI_2_substitutions_update(substitutions):
    koboldai_vars.substitutions = substitutions


#==================================================================#
# Event triggered to rely a message
#==================================================================#
@logger.catch
def socket_io_relay(queue, socketio):
    while True:
        if not queue.empty():
            while not queue.empty():
                data = queue.get()
                socketio.emit(data[0], data[1], **data[2])
        time.sleep(0.2)
        



#==================================================================#
# Event triggered when Softprompt load menu is clicked
#==================================================================#
@socketio.on('load_softprompt_list')
@logger.catch
def UI_2_load_softprompt_list(data):
    if not koboldai_vars.allowsp:
        socketio.emit("error", "Soft prompts are not supported by your current model/backend", broadcast=True, room="UI_2")
    assert koboldai_vars.allowsp, "Soft prompts are not supported by your current model/backend"
    file_popup("Select Softprompt to Load", "./softprompts", "load_softprompt", upload=True, jailed=True, folder_only=False, renameable=True, 
                                                                  deleteable=True, show_breadcrumbs=True, item_check="valid_softprompt",
                                                                  valid_only=True, hide_extention=True, extra_parameter_function="get_softprompt_desc",
                                                                  column_names=['Softprompt Name', 'Softprompt Description'],
                                                                  show_filename=False,
                                                                  column_widths=['150px', 'auto'])

@logger.catch
def valid_softprompt(file):
    z, version, shape, fortran_order, dtype = fileops.checksp(file, koboldai_vars.modeldim)
    if z in [1, 2, 3, 4]:
        return False
    elif not isinstance(z, zipfile.ZipFile):
        print("not zip")
        return False
    else:
        return True

@logger.catch
def get_softprompt_desc(item_full_path, item, valid_selection):
    if not valid_selection:
        return [None, None]
    z = zipfile.ZipFile(item_full_path)
    with z.open('meta.json') as f:
        ob = json.load(f)
        return [ob['name'], ob['description']]

#==================================================================#
# Event triggered when Softprompt is loaded
#==================================================================#
@socketio.on('load_softprompt')
@logger.catch
def UI_2_load_softprompt(data):
    if koboldai_vars.debug:
        print("Load softprompt: {}".format(data))
    spRequest(data)

#==================================================================#
# Event triggered when load userscripts is clicked
#==================================================================#
@socketio.on('load_userscripts_list')
@logger.catch
def UI_2_load_userscripts_list(data):
    file_popup("Select Userscripts to Load", "./userscripts", "load_userscripts", upload=True, jailed=True, folder_only=False, renameable=True, editable=True, 
                                                                  deleteable=True, show_breadcrumbs=False, item_check="valid_userscripts_to_load",
                                                                  valid_only=True, hide_extention=True, extra_parameter_function="get_userscripts_desc",
                                                                  column_names=['Module Name', 'Description'],
                                                                  show_filename=False, show_folders=False,
                                                                  column_widths=['150px', 'auto'])
                                                                
@logger.catch
def valid_userscripts_to_load(file):
    if koboldai_vars.debug:
        print("{} is valid: {}".format(file, file.endswith(".lua") and os.path.basename(file) not in koboldai_vars.userscripts))
    return file.endswith(".lua") and os.path.basename(file) not in koboldai_vars.userscripts
    
@logger.catch
def valid_userscripts_to_unload(file):
    return file.endswith(".lua") and os.path.basename(file) in koboldai_vars.userscripts

@logger.catch
def get_userscripts_desc(item_full_path, item, valid_selection):
    if not valid_selection:
        return [None, None]
    ob = ["", ""]
    description = []
    multiline = False
    with open(item_full_path) as f:
        ob[0] = f.readline().strip().replace("\033", "")
        if ob[0][:2] != "--":
            ob[0] = file
        else:
            ob[0] = ob[0][2:]
            if ob[0][:2] == "[[":
                ob[0] = ob[0][2:]
                multiline = True
            ob[0] = ob[0].lstrip("-").strip()
            for line in f:
                line = line.strip().replace("\033", "")
                if multiline:
                    index = line.find("]]")
                    if index > -1:
                        description.append(line[:index])
                        if index != len(line) - 2:
                            break
                        multiline = False
                    else:
                        description.append(line)
                else:
                    if line[:2] != "--":
                        break
                    line = line[2:]
                    if line[:2] == "[[":
                        multiline = True
                        line = line[2:]
                    description.append(line.strip())
    ob[1] = "\n".join(description)
    if len(ob[1]) > 250:
        ob[1] = ob[1][:247] + "..."
    return ob

#==================================================================#
# Event triggered when userscript's are loaded
#==================================================================#
@socketio.on('load_userscripts')
@logger.catch
def UI_2_load_userscripts(data):
    if koboldai_vars.debug:
        print("Loading Userscripts: {}".format(os.path.basename(data)))
    koboldai_vars.userscripts = [x for x in koboldai_vars.userscripts if x != os.path.basename(data)]+[os.path.basename(data)]
    load_lua_scripts()
    
#==================================================================#
# Event triggered when userscript's are unloaded
#==================================================================#
@socketio.on('unload_userscripts')
@logger.catch
def UI_2_unload_userscripts(data):
    if koboldai_vars.debug:
        print("Unloading Userscript: {}".format(data))
    koboldai_vars.userscripts = [x for x in koboldai_vars.userscripts if x != data]
    load_lua_scripts()



#==================================================================#
# Event triggered when aidg.club loaded
#==================================================================#
@socketio.on('load_aidg_club')
@logger.catch
def UI_2_load_aidg_club(data):
    if koboldai_vars.debug:
        print("Load aidg.club: {}".format(data))
    import_buffer.from_club(data)
    # importAidgRequest(data) 


#==================================================================#
# Event triggered when Theme Changed
#==================================================================#
@socketio.on('theme_change')
@logger.catch
def UI_2_theme_change(data):
    with open("themes/{}.css".format(data['name']), "w") as f:
        f.write(":root {\n")
        for key, value in data['theme'].items():
            f.write("\t{}: {};\n".format(key, value.replace(";", "").replace("--", "-")))
        f.write("}")
        f.write("--------Special Rules from Original Theme---------\n")
        for rule in data['special_rules']:
            f.write(rule)
            f.write("\n")
    if koboldai_vars.debug:
        print("Theme Saved")


#==================================================================#
# Refresh SP List
#==================================================================#
@socketio.on('sp_list_refresh')
@logger.catch
def UI_2_sp_list_refresh(data):
    koboldai_vars.splist = [[f, get_softprompt_desc(os.path.join("./softprompts", f),None,True)] for f in os.listdir("./softprompts") if os.path.isfile(os.path.join("./softprompts", f)) and valid_softprompt(os.path.join("./softprompts", f))]


#==================================================================#
# Refresh Theme List
#==================================================================#
@socketio.on('theme_list_refresh')
@logger.catch
def UI_2_theme_list_refresh(data):
    koboldai_vars.theme_list = [".".join(f.split(".")[:-1]) for f in os.listdir("./themes") if os.path.isfile(os.path.join("./themes", f))]

#==================================================================#
# Save Tweaks
#==================================================================#
@socketio.on('save_cookies')
@logger.catch
def UI_2_save_cookies(data):
    for key in data:
        #Note this won't sync to the client automatically as we're modifying a variable rather than setting it
        koboldai_vars.cookies[key] = data[key]
    with open("./settings/cookies.settings", "w") as f:
        json.dump(koboldai_vars.cookies, f)

#==================================================================#
# Fewshot WI generation
#==================================================================#
@socketio.on("generate_wi")
@logger.catch
def UI_2_generate_wi(data):
    uid = data["uid"]
    field = data["field"]
    existing = data["existing"]
    gen_amount = data["genAmount"]

    # The template to coax what we want from the model
    extractor_string = ""

    if field == "title":
        for thing in ["type", "desc"]:
            if not existing[thing]:
                continue
            pretty = {"type": "Type", "desc": "Description"}[thing]
            extractor_string += f"{pretty}: {existing[thing]}\n"
        
        pretty = "Title"
        if existing["desc"]:
            # Don't let the model think we're starting a new entry
            pretty = "Alternate Title"

        extractor_string += pretty + ":"
    elif field == "desc":
        # MUST be title and type
        assert existing["title"]
        assert existing["type"]
        extractor_string = f"Title: {existing['title']}\nType: {existing['type']}\nDescription:"
    else:
        assert False, "What"

    with open("data/wi_fewshot.txt", "r") as file:
        fewshot_entries = [x.strip() for x in file.read().split("\n\n") if x]

    # Use user's own WI entries in prompt
    if koboldai_vars.wigen_use_own_wi:
        fewshot_entries += koboldai_vars.worldinfo_v2.to_wi_fewshot_format(excluding_uid=uid)
    
    # We must have this amount or less in our context.
    target = koboldai_vars.max_length - gen_amount - len(tokenizer.encode(extractor_string))

    used = []
    # Walk the entries backwards until we can't cram anymore in
    for entry in reversed(fewshot_entries):
        maybe = [entry] + used
        maybe_str = "\n\n".join(maybe)
        possible_encoded = tokenizer.encode(maybe_str)
        if len(possible_encoded) > target:
            break
        yes_str = maybe_str
        used = maybe
    
    prompt = f"{yes_str}\n\n{extractor_string}"
    
    # logger.info(prompt)
    # TODO: Make single_line mode that stops on newline rather than bans it (for title)
    out_text = tpool.execute(
        model.raw_generate,
        prompt,
        max_new=gen_amount,
        single_line=True,
    ).decoded[0]
    out_text = utils.trimincompletesentence(out_text.strip())

    socketio.emit("generated_wi", {"uid": uid, "field": field, "out": out_text}, room="UI_2")

@app.route("/generate_raw", methods=["GET"])
@require_allowed_ip
def UI_2_generate_raw():
    prompt = request.args.get("prompt")

    if not prompt:
        return Response(json.dumps({"error": "No prompt"}), status=400)

    if not model:
        return Response(json.dumps({"error": "No model"}), status=500)

    try:
        out = model.raw_generate(prompt, max_new=80)
    except NotImplementedError as e:
        return Response(json.dumps({"error": str(e)}), status=500)

    return out.decoded

#==================================================================#
# Load Tweaks
#==================================================================#
@logger.catch
def UI_2_load_cookies():
    if koboldai_vars.on_colab:
        if os.path.exists("./settings/cookies.settings"):
            with open("./settings/cookies.settings", "r") as f:
                data = json.load(f)
                socketio.emit('load_cookies', data, room="UI_2")

#==================================================================#
# Save New Preset
#==================================================================#
@socketio.on('save_new_preset')
@logger.catch
def UI_2_save_new_preset(data):
    preset = model_info()
    #Data to get from current settings
    for item in ["genamt", "rep_pen", "rep_pen_range", "rep_pen_slope", "sampler_order", "temp", "tfs", "top_a", "top_k", "top_p", "typical"]:
        preset[item] = getattr(koboldai_vars, item)
    #Data to get from UI
    for item in ['preset', 'description']:
        preset[item] = data[item]
    preset['Preset Category'] = 'Custom'
    if os.path.exists("./presets/{}.presets".format(data['preset'])):
        with open("./presets/{}.presets".format(data['preset']), "r") as f:
            old_preset = json.load(f)
            if not isinstance(old_preset, list):
                old_preset = [old_preset]
        for i in range(len(old_preset)):
            if old_preset[i]['Model Name'] == preset['Model Name']:
                del old_preset[i]
                break
        old_preset.append(preset)
        preset = old_preset
    else:
        preset = [preset]
    print(preset)
    with open("./presets/{}.presets".format(data['preset']), "w") as f:
        print("Saving to {}".format("./presets/{}.presets".format(data['preset'])))
        json.dump(preset, f, indent="\t")

@logger.catch
def get_model_size(model_name):
    if "30B" in model_name:
        return "30B"
    elif "20B" in model_name:
        return "20B"
    elif "13B" in model_name:
        return "13B"
    elif "6B" in model_name.replace("6.7B", "6B"):
        return "6B"
    elif "2.7B" in model_name:
        return "2.7B"
    elif "1.3B" in model_name:
        return "1.3B"

#==================================================================#
# Save New Preset
#==================================================================#
@socketio.on('save_revision')
@logger.catch
def UI_2_save_revision(data):
    koboldai_vars.save_revision()


#==================================================================#
# Generate Image
#==================================================================#
@socketio.on("generate_image")
@logger.catch
def UI_2_generate_image_from_story(data):
    # Independant of generate_story_image as summarization is rather time consuming
    koboldai_vars.generating_image = True
    eventlet.sleep(0)
    
    art_guide = str(koboldai_vars.img_gen_art_guide)
    
    if 'action_id' in data and (int(data['action_id']) in koboldai_vars.actions.actions or int(data['action_id']) == -1):
        action_id = int(data['action_id'])
    else:
        #get latest action
        if len(koboldai_vars.actions) > 0:
            action = koboldai_vars.actions[-1]
            action_id = len(koboldai_vars.actions) - 1
        else:
            action = koboldai_vars.prompt
            action_id = -1
    
    logger.info("Generating image for action {}".format(action_id))
    
    start_time = time.time()
    if os.path.exists("models/{}".format(args.summarizer_model.replace('/', '_'))):
        koboldai_vars.summary_tokenizer = AutoTokenizer.from_pretrained("models/{}".format(args.summarizer_model.replace('/', '_')), cache_dir="cache")
    else:
        koboldai_vars.summary_tokenizer = AutoTokenizer.from_pretrained(args.summarizer_model, cache_dir="cache")
    #text to summarize (get 1000 tokens worth of text):
    text = []
    text_length = 0
    for item in reversed(koboldai_vars.actions.to_sentences(max_action_id=action_id)):
        if len(koboldai_vars.summary_tokenizer.encode(item[0])) + text_length <= 1000:
            text.append(item[0])
            text_length += len(koboldai_vars.summary_tokenizer.encode(item[0]))
        else:
            break
    text = "".join(text)
    logger.debug("Text to summarizer: {}".format(text))
    
    max_length = args.max_summary_length - len(koboldai_vars.summary_tokenizer.encode(art_guide))
    keys = [summarize(text, max_length=max_length)]
    prompt = ", ".join(keys)
    logger.debug("Text from summarizer: {}".format(prompt))

    if art_guide:
        if '<|>' in art_guide:
            full_prompt = art_guide.replace('<|>', prompt)
        else:
            full_prompt = f"{prompt}, {art_guide}"
    else:
        full_prompt = prompt

    generate_story_image(
        full_prompt,
        file_prefix=f"action_{action_id}",
        display_prompt=full_prompt,
        log_data={"actionId": action_id},
    )

@socketio.on("generate_image_from_prompt")
@logger.catch
def UI_2_generate_image_from_prompt(prompt: str):
    eventlet.sleep(0)
    generate_story_image(prompt, file_prefix="prompt", generation_type="direct_prompt")

def log_image_generation(
    prompt: str,
    display_prompt: str,
    file_name: str,
    generation_type: str,
    other_data: Optional[dict] = None
) -> None:
    # In the future it might be nice to have some UI where you can search past
    # generations or something like that
    db_path = os.path.join(koboldai_vars.save_paths.generated_images, "db.json")

    try:
        with open(db_path, "r") as file:
            j = json.load(file)
    except FileNotFoundError:
        j = []
    
    if not isinstance(j, list):
        logger.warning("Image database is corrupted! Will not add new entry.")
        return

        
    log_data = {
        "prompt": prompt,
        "fileName": file_name,
        "type": generation_type or None,
        "displayPrompt": display_prompt
    }
    log_data.update(other_data or {})
    j.append(log_data)

    with open(db_path, "w") as file:
        json.dump(j, file, indent="\t")

@socketio.on("retry_generated_image")
@logger.catch
def UI2_retry_generated_image():
    eventlet.sleep(0)
    generate_story_image(koboldai_vars.picture_prompt)

def generate_story_image(
    prompt: str,
    file_prefix: str = "image",
    generation_type: str = "",
    display_prompt: Optional[str] = None,
    log_data: Optional[dict] = None
    
) -> None:
    # This function is a wrapper around generate_image() that integrates the
    # result with the story (read: puts it in the corner of the screen).

    log_data = log_data or {}

    if not display_prompt:
        display_prompt = prompt
    koboldai_vars.picture_prompt = display_prompt

    start_time = time.time()
    koboldai_vars.generating_image = True

    image = generate_image(prompt)
    koboldai_vars.generating_image = False

    if not image:
        return
        
    exif = image.getexif()
    exif[0x9286] = prompt
    exif[0x927C] = generation_type if generation_type != "" else "Stable Diffusion from KoboldAI"

    if os.path.exists(koboldai_vars.save_paths.generated_images):
        # Only save image if this is a saved story
        file_name = f"{file_prefix}_{int(time.time())}.jpg"
        image.save(os.path.join(koboldai_vars.save_paths.generated_images, file_name), format="JPEG", exif=exif)
        log_image_generation(prompt, display_prompt, file_name, generation_type, log_data)
        #let's also add this data to the action so we know where the latest picture is at
        logger.info("setting picture filename")
        try:
            koboldai_vars.actions.set_picture(int(log_data['actionId']), file_name, prompt)
        except KeyError:
            pass

    logger.debug("Time to Generate Image {}".format(time.time()-start_time))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    b64_data = base64.b64encode(buffer.getvalue()).decode("ascii")

    koboldai_vars.picture = b64_data

def generate_image(prompt: str) -> Optional[Image.Image]:
    if koboldai_vars.img_gen_priority == 4:
        # Check if stable-diffusion-webui API option selected and use that if found.
        return text2img_api(prompt)
    elif ((not koboldai_vars.hascuda or not os.path.exists("functional_models/stable-diffusion/model_index.json")) and koboldai_vars.img_gen_priority != 0) or koboldai_vars.img_gen_priority == 3:
        # If we don't have a GPU, use horde if we're allowed to
        return text2img_horde(prompt)

    memory = torch.cuda.get_device_properties(0).total_memory

    # We aren't being forced to use horde, so now let's figure out if we should use local
    if memory - torch.cuda.memory_reserved(0) >= 6000000000:
        # We have enough vram, just do it locally
        return text2img_local(prompt)
    elif memory > 6000000000 and koboldai_vars.img_gen_priority <= 1:
        # We could do it locally by swapping the model out
        print("Could do local or online")
        return text2img_horde(prompt)
    elif koboldai_vars.img_gen_priority != 0:
        return text2img_horde(prompt)

    raise RuntimeError("Unable to decide image generation backend. Please report this.")
    

@logger.catch
def text2img_local(prompt: str) -> Optional[Image.Image]:
    start_time = time.time()
    logger.debug("Generating Image")
    from diffusers import StableDiffusionPipeline
    if koboldai_vars.image_pipeline is None:
        if not os.path.exists("functional_models/stable-diffusion/model_index.json"):
            from huggingface_hub import snapshot_download
            snapshot_download("XpucT/Deliberate", local_dir="functional_models/stable-diffusion", local_dir_use_symlinks=False, cache_dir="cache/", ignore_patterns=["*.safetensors"])
        pipe = tpool.execute(StableDiffusionPipeline.from_pretrained, "functional_models/stable-diffusion", safety_checker=None, torch_dtype=torch.float16).to("cuda")
    else:
        pipe = koboldai_vars.image_pipeline.to("cuda")
    logger.debug("time to load: {}".format(time.time() - start_time))
    start_time = time.time()
    
    def get_image(pipe, prompt, num_inference_steps):
        from torch import autocast
        with autocast("cuda"):
            return pipe(prompt, num_inference_steps=num_inference_steps).images[0]
    image = tpool.execute(get_image, pipe, prompt, num_inference_steps=koboldai_vars.img_gen_steps)
    logger.debug("time to generate: {}".format(time.time() - start_time))
    start_time = time.time()
    if koboldai_vars.keep_img_gen_in_memory:
        pipe.to("cpu")
        if koboldai_vars.image_pipeline is None:
            koboldai_vars.image_pipeline = pipe
    else:
        koboldai_vars.image_pipeline = None
        del pipe
    torch.cuda.empty_cache()
    logger.debug("time to unload: {}".format(time.time() - start_time))
    return image

@logger.catch
def text2img_horde(prompt: str) -> Optional[Image.Image]:
    logger.debug("Generating Image using Horde")
    
    final_submit_dict = {
        "prompt": prompt,
        "trusted_workers": False, 
        "models": [
          "stable_diffusion"
        ],
        "params": {
            "n": 1,
            "nsfw": True,
            "sampler_name": "k_euler_a",
            "karras": True,
            "cfg_scale": koboldai_vars.img_gen_cfg_scale,
            "steps": koboldai_vars.img_gen_steps, 
            "width": 512, 
            "height": 512
        }
    }
    client_agent = "KoboldAI:2.0.0:koboldai.org"
    cluster_headers = {
        'apikey': koboldai_vars.horde_api_key,
        "Client-Agent": client_agent
    }    
    id_req = requests.post(f"{koboldai_vars.horde_url}/api/v2/generate/async", json=final_submit_dict, headers=cluster_headers)

    if not id_req.ok:
        if id_req.status_code == 403:
            show_error_notification(
                "Stable Horde failure",
                "Stable Horde is currently not accepting anonymous requuests. " \
                "Try again in a few minutes or register for priority access at https://horde.koboldai.net",
                do_log=True
            )
            return None
        logger.error(f"HTTP {id_req.status_code}, expected OK-ish")
        logger.error(id_req.text)
        logger.error(f"Response headers: {id_req.headers}")
        raise HordeException("Image seeding failed. See console for more details.")
    
    image_id = id_req.json()["id"]

    while True:
        poll_req = requests.get(f"{koboldai_vars.horde_url}/api/v2/generate/check/{image_id}")
        if not poll_req.ok:
            logger.error(f"HTTP {poll_req.status_code}, expected OK-ish")
            logger.error(poll_req.text)
            logger.error(f"Response headers: {poll_req.headers}")
            raise HordeException("Image polling failed. See console for more details.")
        poll_j = poll_req.json()

        if poll_j["finished"] > 0:
            break

        # This should always exist but if it doesn't 2 seems like a safe bet.
        sleepy_time = int(poll_req.headers.get("retry-after", 2))
        time.sleep(sleepy_time)
    
    # Done generating, we can now fetch it.

    gen_req = requests.get(f"{koboldai_vars.horde_url}/api/v2/generate/status/{image_id}")
    if not gen_req.ok:
        logger.error(f"HTTP {gen_req.status_code}, expected OK-ish")
        logger.error(gen_req.text)
        logger.error(f"Response headers: {gen_req.headers}")
        raise HordeException("Image fetching failed. See console for more details.")
    results = gen_req.json()

    if len(results["generations"]) > 1:
        logger.warning(f"Got too many generations, discarding extras. Got {len(results['generations'])}, expected 1.")
    
    imgurl = results["generations"][0]["img"]
    try:
        img_data = requests.get(imgurl, timeout=3).content
        img = Image.open(BytesIO(img_data))
        return img
    except Exception as err:
        logger.error(f"Error retrieving image: {err}")        
        raise HordeException("Image fetching failed. See console for more details.")

@logger.catch
def text2img_api(prompt, art_guide="") -> Image.Image:
    logger.debug("Generating Image using Local SD-WebUI API")
    koboldai_vars.generating_image = True
    #The following list are valid properties with their defaults, to add/modify in final_imgen_params. Will refactor configuring values into UI element in future.
      #"enable_hr": false,
      #"denoising_strength": 0,
      #"firstphase_width": 0,
      #"firstphase_height": 0,
      #"prompt": "",
      #"styles": [
      #  "string"
      #],
      #"seed": -1,
      #"subseed": -1,
      #"subseed_strength": 0,
      #"seed_resize_from_h": -1,
      #"seed_resize_from_w": -1,
      #"batch_size": 1,
      #"n_iter": 1,
      #"steps": 50,
      #"cfg_scale": 7,
      #"width": 512,
      #"height": 512,
      #"restore_faces": false,
      #"tiling": false,
      #"negative_prompt": "string",
      #"eta": 0,
      #"s_churn": 0,
      #"s_tmax": 0,
      #"s_tmin": 0,
      #"s_noise": 1,
      #"override_settings": {},
      #"sampler_index": "Euler"
    final_imgen_params = {
        "prompt": prompt,
        "n_iter": 1,
        "width": 512,
        "height": 512,
        "steps": koboldai_vars.img_gen_steps,
        "cfg_scale": koboldai_vars.img_gen_cfg_scale,
        "negative_prompt": koboldai_vars.img_gen_negative_prompt,
        "sampler_index": "Euler a"
    }
    apiaddress = '{}/sdapi/v1/txt2img'.format(koboldai_vars.img_gen_api_url.rstrip("/"))
    payload_json = json.dumps(final_imgen_params)
    logger.debug(final_imgen_params)

    try:
        logger.info("Gen Image API: Username: {}".format(koboldai_vars.img_gen_api_username))
        if koboldai_vars.img_gen_api_username != "":
            basic = requests.auth.HTTPBasicAuth(koboldai_vars.img_gen_api_username, koboldai_vars.img_gen_api_password)
            submit_req = requests.post(url=apiaddress, data=payload_json, auth=basic)
        else:
            submit_req = requests.post(url=apiaddress, data=payload_json)
    except requests.exceptions.ConnectionError:
        show_error_notification(
            "SD Web API Failure",
            "Unable to connect to SD Web UI. Is it running?",
            do_log=True
        )
        return None
    except Exception as e:
        show_error_notification("SD Web API Failure", "Unknown error in connecting to the SD Web UI. Is it running?")
        logger.error(f"{type(e)}: {e}")
        return None
    finally:
        koboldai_vars.generating_image = False

    if submit_req.status_code == 404:
        show_error_notification(
            "SD Web API Failure",
            f"The SD Web UI was not called with --api. Unable to connect.",
            do_log=True
        )
        return None
    elif not submit_req.ok:
        show_error_notification("SD Web API Failure", f"HTTP Code {submit_req.status_code} -- See console for details")
        logger.error(f"SD Web API Failure: HTTP Code {submit_req.status_code}, Body:\n{submit_req.text}")
        return None

    results = submit_req.json()

    try:
        base64_image = results["images"][0]
    except (IndexError, KeyError):
        show_error_notification("SD Web API Failure", "SD Web API returned no images", do_log=True)
        return None

    return Image.open(BytesIO(base64.b64decode(base64_image)))

@socketio.on("clear_generated_image")
@logger.catch
def UI2_clear_generated_image(data):
    koboldai_vars.picture = ""
    koboldai_vars.picture_prompt = ""

#@logger.catch
def get_items_locations_from_text(text):
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = transformers.pipeline("ner", model=model, tokenizer=tokenizer)
    # input example sentence
    ner_results = nlp(text)
    orgs = []
    last_org_position = -2
    loc = []
    last_loc_position = -2
    per = []
    last_per_position = -2
    for i, result in enumerate(ner_results):
        if result['entity'] in ('B-ORG', 'I-ORG'):
            if result['start']-1 <= last_org_position:
                if result['start'] != last_org_position:
                    orgs[-1] = "{} ".format(orgs[-1])
                orgs[-1] = "{}{}".format(orgs[-1], result['word'].replace("##", ""))
            else:
                orgs.append(result['word'])
            last_org_position = result['end']
        elif result['entity'] in ('B-LOC', 'I-LOC'):
            if result['start']-1 <= last_loc_position:
                if result['start'] != last_loc_position:
                    loc[-1] = "{} ".format(loc[-1])
                loc[-1] = "{}{}".format(loc[-1], result['word'].replace("##", ""))
            else:
                loc.append(result['word'])
            last_loc_position = result['end']
        elif result['entity'] in ('B-PER', 'I-PER'):
            if result['start']-1 <= last_per_position:
                if result['start'] != last_per_position:
                    per[-1] = "{} ".format(per[-1])
                per[-1] = "{}{}".format(per[-1], result['word'].replace("##", ""))
            else:
                per.append(result['word'])
            last_per_position = result['end']

    print("Orgs: {}".format(orgs))
    print("Locations: {}".format(loc))
    print("People: {}".format(per))

#==================================================================#
# summarizer
#==================================================================#
def summarize(text, max_length=100, min_length=30, unload=True):
    from transformers import pipeline as summary_pipeline
    start_time = time.time()
    if koboldai_vars.summarizer is None:
        if os.path.exists("functional_models/{}".format(args.summarizer_model.replace('/', '_'))):
            koboldai_vars.summary_tokenizer = AutoTokenizer.from_pretrained("functional_models/{}".format(args.summarizer_model.replace('/', '_')), cache_dir="cache")
            koboldai_vars.summarizer = AutoModelForSeq2SeqLM.from_pretrained("functional_models/{}".format(args.summarizer_model.replace('/', '_')), cache_dir="cache")
        else:
            koboldai_vars.summary_tokenizer = AutoTokenizer.from_pretrained(args.summarizer_model, cache_dir="cache")
            koboldai_vars.summarizer = AutoModelForSeq2SeqLM.from_pretrained(args.summarizer_model, cache_dir="cache")
            koboldai_vars.summary_tokenizer.save_pretrained("functional_models/{}".format(args.summarizer_model.replace('/', '_')), max_shard_size="500MiB")
            koboldai_vars.summarizer.save_pretrained("functional_models/{}".format(args.summarizer_model.replace('/', '_')), max_shard_size="500MiB")

    #Try GPU accel
    if koboldai_vars.hascuda and torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0) >= 1645778560:
        koboldai_vars.summarizer.to(0)
        device=0
    else:
        device=-1
    summarizer = tpool.execute(summary_pipeline, task="summarization", model=koboldai_vars.summarizer, tokenizer=koboldai_vars.summary_tokenizer, device=device)
    logger.debug("Time to load summarizer: {}".format(time.time()-start_time))
    
    #Actual sumarization
    start_time = time.time()
    #make sure text is less than 1024 tokens, otherwise we'll crash
    if len(koboldai_vars.summary_tokenizer.encode(text)) > 1000:
        text = koboldai_vars.summary_tokenizer.decode(koboldai_vars.summary_tokenizer.encode(text)[:1000])
    output = tpool.execute(summarizer, text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    logger.debug("Time to summarize: {}".format(time.time()-start_time))
    #move model back to CPU to save precious vram
    torch.cuda.empty_cache()
    logger.debug("VRAM used by summarization: {}".format(torch.cuda.memory_reserved(0)))
    if unload:
        koboldai_vars.summarizer.to("cpu")
    torch.cuda.empty_cache()
    
    #logger.debug("Original Text: {}".format(text))
    #logger.debug("Summarized Text: {}".format(output))
    
    return output

#==================================================================#
# Auto-memory function
#==================================================================#
@socketio.on("refresh_auto_memory")
@logger.catch
def UI_2_refresh_auto_memory(data):
    koboldai_vars.auto_memory = "Generating..."
    if koboldai_vars.summary_tokenizer is None:
        if os.path.exists("models/{}".format(args.summarizer_model.replace('/', '_'))):
            koboldai_vars.summary_tokenizer = AutoTokenizer.from_pretrained("models/{}".format(args.summarizer_model.replace('/', '_')), cache_dir="cache")
        else:
            koboldai_vars.summary_tokenizer = AutoTokenizer.from_pretrained(args.summarizer_model, cache_dir="cache")
    #first, let's get all of our game text and split it into sentences
    sentences = [x[0] for x in koboldai_vars.actions.to_sentences()]
    sentences_lengths = [len(koboldai_vars.summary_tokenizer.encode(x)) for x in sentences]
    
    pass_number = 1
    while len(koboldai_vars.summary_tokenizer.encode("".join(sentences))) > 1000:
        #Now let's split them into 1000 token chunks
        summary_chunks = [""]
        summary_chunk_lengths = [0]
        for i in range(len(sentences)):
            if summary_chunk_lengths[-1] + sentences_lengths[i] <= 1000:
                summary_chunks[-1] += sentences[i]
                summary_chunk_lengths[-1] += sentences_lengths[i]
            else:
                summary_chunks.append(sentences[i])
                summary_chunk_lengths.append(sentences_lengths[i])
        new_sentences = []
        i=0
        for summary_chunk in summary_chunks:
            logger.debug("summarizing chunk {}".format(i))
            new_sentences.extend(re.split("(?<=[.!?])\s+", summarize(summary_chunk, unload=False)))
            i+=1
        logger.debug("Pass {}:\nSummarized to {} sentencees from {}".format(pass_number, len(new_sentences), len(sentences)))
        sentences = new_sentences
        koboldai_vars.auto_memory += "Pass {}:\n{}\n\n".format(pass_number, "\n".join(sentences))
        pass_number+=1
    logger.debug("OK, doing final summarization")
    output = summarize(" ".join(sentences))
    koboldai_vars.auto_memory += "\n\n Final Result:\n" + output


#==================================================================#
# Story review zero-shot
#==================================================================#
def maybe_review_story() -> None:
    commentary_characters = koboldai_vars.worldinfo_v2.get_commentators()
    if not (
        commentary_characters
        and koboldai_vars.commentary_chance
        and koboldai_vars.commentary_enabled
    ):
        return

    if random.randrange(100) > koboldai_vars.commentary_chance:
        return

    char = random.choice(commentary_characters)
    speaker_uid = char["uid"]
    speaker_name = char["title"]

    allowed_wi_uids = [speaker_uid]
    for uid, wi in koboldai_vars.worldinfo_v2.world_info.items():
        if wi["type"] == "commentator":
            continue
        allowed_wi_uids.append(uid)

    prompt = f"\n\n{speaker_name}'s thoughts on what just happened in this story: \""

    context = koboldai_vars.calc_ai_text(
        prompt,
        return_text=True,
        send_context=False,
        allowed_wi_entries=allowed_wi_uids
    )


    out_text = tpool.execute(
        model.raw_generate,
        context,
        max_new=30
    ).decoded[0]

    out_text = re.sub(r"[\s\(\)]", " ", out_text)

    while "  " in out_text:
        out_text = out_text.replace("  ", " ")

    if '"' in out_text:
        out_text = out_text.split('"')[0]

    out_text = out_text.strip()
    out_text = utils.trimincompletesentence(out_text)
    emit("show_story_review", {"who": speaker_name, "review": out_text, "uid": speaker_uid})

#==================================================================#
# Get next 100 actions for infinate scroll
#==================================================================#
@socketio.on("get_next_100_actions")
@logger.catch
def UI_2_get_next_100_actions(data):
    logger.debug("Sending an additional 100 actions, starting at action {}".format(data-1))
    sent = 0
    data_to_send = []
    for i in reversed(list(koboldai_vars.actions.actions)):
        if i < data:
            if sent >= 100:
                break
            data_to_send.append({"id": i, "action": koboldai_vars.actions.actions[i]})
            sent += 1
    logger.debug("data_to_send length: {}".format(len(data_to_send)))
    emit("var_changed", {"classname": "story", "name": "actions", "old_value": None, "value":data_to_send})

#==================================================================#
# Get context tokens
#==================================================================#
@socketio.on("update_tokens")
@logger.catch
def UI_2_update_tokens(data):
    ignore = koboldai_vars.calc_ai_text(submitted_text=data)

#==================================================================#
# Enable/Disable Privacy Mode
#==================================================================#
@socketio.on("privacy_mode")
@logger.catch
def UI_2_privacy_mode(data):
    if data['enabled']:
        koboldai_vars.privacy_mode = True
        return

    if data['password'] == koboldai_vars.privacy_password:
        koboldai_vars.privacy_mode = False
    else:
        logger.warning("Watch out! Someone tried to unlock your instance with an incorrect password! Stay on your toes...")
        show_error_notification(
            title="Invalid password",
            text="The password you provided was incorrect. Please try again."
        )

#==================================================================#
# Genres
#==================================================================#
@app.route("/genre_data.json", methods=["GET"])
@require_allowed_ip
def UI_2_get_applicable_genres():
    with open("data/genres.json", "r") as file:
        genre_list = json.load(file)
    return Response(json.dumps({
        "list": genre_list,
        "init": koboldai_vars.genres,
    }))
#==================================================================#
# Soft Prompt Tuning
#==================================================================#
@socketio.on("create_new_softprompt")
@logger.catch
def UI_2_create_new_softprompt(data):
    import breakmodel
    logger.info("Soft Prompt Dataset: {}".format(data))
    from prompt_tuner import BasicTrainer
    trainer = BasicTrainer(None, quiet=koboldai_vars.quiet)
    trainer.data.ckpt_path = koboldai_vars.model
    trainer.get_hf_checkpoint_metadata()
    trainer.data.save_file = "{}.mtjsp".format("".join(x for x in data['sp_title'] if x.isalnum() or x in [" ", "-", "_"]))
    trainer.data.prompt_method = "tokens"
    tokenizer = trainer.get_tokenizer()
    if trainer.data.newlinemode == "s":  # Handle fairseq-style newlines if required
        initial_softprompt = data['sp_prompt'].replace("\n", "</s>")
    trainer.data.initial_softprompt = tokenizer.encode(
        data['sp_prompt'], max_length=int(2e9), truncation=True
    )
    trainer.tokenize_dataset(dataset_path=data['sp_dataset'], 
                             output_file="softprompts/{}.npy".format("".join(x for x in data['sp_title'] if x.isalnum() or x in [" ", "-", "_"])), 
                             batch_size=2048 if 'batch_size' not in data else data['batch_size'], 
                             epochs=1 if 'epochs' not in data else data['epochs'])
    trainer.data.dataset_file = "softprompts/{}.npy".format("".join(x for x in data['sp_title'] if x.isalnum() or x in [" ", "-", "_"]))
    trainer.data.gradient_accumulation_steps = 16  if 'gradient_accumulation_steps' not in data else data['gradient_accumulation_steps']
    
    trainer.data.stparams = {
        "lr": 3e-5,
        "max_grad_norm": 10.0,
        "weight_decay": 0.1,
        "warmup": 0.1,
        "end_lr_multiplier": 0.1,
        "save_every": 50,
    }
    
    unload_model()
    trainer.train(breakmodel_primary_device=breakmodel.primary_device,
                    breakmodel_gpulayers=breakmodel.gpu_blocks,
                    breakmodel_disklayers=breakmodel.disk_blocks)
    
    output_file = "softprompts/{}.zip".format("".join(x for x in data['sp_title'] if x.isalnum() or x in [" ", "-", "_"]))
    name = data['sp_title']
    author = data['sp_author']
    supported = koboldai_vars.model
    description = data['sp_description']
    trainer.export_to_kobold(output_file, name, author, supported, description)
    output_file = "softprompts/{}.json".format("".join(x for x in data['sp_title'] if x.isalnum() or x in [" ", "-", "_"]))
    trainer.export_to_mkultra(output_file, name, description)
    

#==================================================================#
# Test
#==================================================================#
@socketio.on("get_log")
def UI_2_get_log(data):
    emit("log_message", web_log_history)
    
@app.route("/get_log")
@require_allowed_ip
def UI_2_get_log_get():
    if args.no_ui:
        return redirect('/api/latest')
    return {'aiserver_log': web_log_history}

@app.route("/test_match")
@require_allowed_ip
@logger.catch
def UI_2_test_match():
    koboldai_vars.assign_world_info_to_actions()
    return show_vars()

#==================================================================#
# Download of the audio file
#==================================================================#
@app.route("/audio")
@require_allowed_ip
@logger.catch
def UI_2_audio():
    if args.no_ui:
        return redirect('/api/latest')
    action_id = int(request.args['id']) if 'id' in request.args else koboldai_vars.actions.action_count
    filename = os.path.join(koboldai_vars.save_paths.generated_audio, f"{action_id}.ogg")
    filename_slow = os.path.join(koboldai_vars.save_paths.generated_audio, f"{action_id}_slow.ogg")
    
    if os.path.exists(filename_slow):
        return send_file(
                 filename_slow, 
                 mimetype="audio/ogg")
    if not os.path.exists(filename):
        koboldai_vars.actions.gen_audio(action_id)
        start_time = time.time()
        while not os.path.exists(filename) and time.time()-start_time < 60: #Waiting up to 60 seconds for the file to be generated
            time.sleep(0.1)
    return send_file(
             filename, 
             mimetype="audio/ogg")


#==================================================================#
# Download of the image for an action
#==================================================================#
@app.route("/action_image")
@require_allowed_ip
@logger.catch
def UI_2_action_image():
    if args.no_ui:
        return redirect('/api/latest')
    action_id = int(request.args['id']) if 'id' in request.args else koboldai_vars.actions.action_count
    filename, prompt = koboldai_vars.actions.get_picture(action_id)
    koboldai_vars.picture_prompt = prompt
    if filename is not None:
        return send_file(
                 filename, 
                 mimetype="image/jpeg")
    else:
        return send_file(
                 "static/blank.png", 
                 mimetype="image/png")

#==================================================================#
# display messages if they have never been sent before on this install
#==================================================================#
with open("data/one_time_messages.json", "r") as f:
    messages = json.load(f)
    messages = {int(x): messages[x] for x in messages}
@logger.catch
@socketio.on("check_messages")
def send_one_time_messages(data, wait_time=0):
    time.sleep(wait_time) #Need to wait a bit for the web page to load as the connect event is very eary
    if data != '':
        if int(data) not in koboldai_vars.seen_messages:
            koboldai_vars.seen_messages.append(int(data))
            #Now let's save
            filename = "settings/system_settings.v2_settings"
            if not os.path.exists("settings"):
                os.mkdir("settings")
            with open(filename, "w") as settings_file:
                settings_file.write(getattr(koboldai_vars, "_system_settings").to_json())
    for message in messages:
        if message not in koboldai_vars.seen_messages:
            socketio.emit("message", messages[message])
            break

#==================================================================#
# Test
#==================================================================#
def model_info():
    global model_config
    if 'model_config' in globals() and model_config is not None:
        if isinstance(model_config, dict):
            if 'model_type' in model_config:
                model_type = str(model_config['model_type'])
            elif koboldai_vars.mode[:4] == 'gpt2':
                model_type = 'gpt2'
            else:
                model_type = "Unknown"
        else:
            model_type = str(model_config.model_type)
        return {"Model Type": model_type, "Model Size": get_model_size(koboldai_vars.model), "Model Name": koboldai_vars.model.replace("_", "/")}
    else:
        return {"Model Type": "Read Only", "Model Size": "0", "Model Name": koboldai_vars.model.replace("_", "/")}

@app.route("/vars")
@require_allowed_ip
@logger.catch
def show_vars():
    if args.no_ui or koboldai_vars.host:
        return redirect('/api/latest')
    json_data = {}
    json_data['story_settings'] = json.loads(koboldai_vars.to_json("story_settings"))
    json_data['model_settings'] = json.loads(koboldai_vars.to_json("model_settings"))
    json_data['user_settings'] = json.loads(koboldai_vars.to_json("user_settings"))
    json_data['system_settings'] = json.loads(koboldai_vars.to_json("system_settings"))
    return json_data

@socketio.on("trigger_error")
@logger.catch
def trigger_error(data):
    temp = this_var_doesnt_exist

#==================================================================#
class EmptySchema(KoboldSchema):
    pass

class BasicTextResultInnerSchema(KoboldSchema):
    text: str = fields.String(required=True)

class BasicTextResultSchema(KoboldSchema):
    result: BasicTextResultInnerSchema = fields.Nested(BasicTextResultInnerSchema)

class BasicResultInnerSchema(KoboldSchema):
    result: str = fields.String(required=True)

class BasicResultSchema(KoboldSchema):
    result: BasicResultInnerSchema = fields.Nested(BasicResultInnerSchema, required=True)

class BasicResultsSchema(KoboldSchema):
    results: BasicResultInnerSchema = fields.List(fields.Nested(BasicResultInnerSchema), required=True)

class BasicStringSchema(KoboldSchema):
    value: str = fields.String(required=True)

class BasicBooleanSchema(KoboldSchema):
    value: bool = fields.Boolean(required=True)

class BasicUIDSchema(KoboldSchema):
    uid: str = fields.Integer(required=True, validate=validate.Range(min=-2147483648, max=2147483647), metadata={"description": "32-bit signed integer unique to this world info entry/folder."})

class BasicErrorSchema(KoboldSchema):
    msg: str = fields.String(required=True)
    type: str = fields.String(required=True)

class StoryEmptyErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)

class StoryTooShortErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)

class OutOfMemoryErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)

class NotFoundErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)

api_out_of_memory_response = """507:
          description: Out of memory
          content:
            application/json:
              schema: OutOfMemoryErrorSchema
              examples:
                gpu.cuda:
                  value:
                    detail:
                      msg: "KoboldAI ran out of memory: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 4.00 GiB total capacity; 2.97 GiB already allocated; 0 bytes free; 2.99 GiB reserved in total by PyTorch)"
                      type: out_of_memory.gpu.cuda
                gpu.hip:
                  value:
                    detail:
                      msg: "KoboldAI ran out of memory: HIP out of memory. Tried to allocate 20.00 MiB (GPU 0; 4.00 GiB total capacity; 2.97 GiB already allocated; 0 bytes free; 2.99 GiB reserved in total by PyTorch)"
                      type: out_of_memory.gpu.hip
                tpu.hbm:
                  value:
                    detail:
                      msg: "KoboldAI ran out of memory: Compilation failed: Compilation failure: Ran out of memory in memory space hbm. Used 8.83G of 8.00G hbm. Exceeded hbm capacity by 848.88M."
                      type: out_of_memory.tpu.hbm
                cpu.default_cpu_allocator:
                  value:
                    detail:
                      msg: "KoboldAI ran out of memory: DefaultCPUAllocator: not enough memory: you tried to allocate 209715200 bytes."
                      type: out_of_memory.cpu.default_cpu_allocator
                unknown.unknown:
                  value:
                    detail:
                      msg: "KoboldAI ran out of memory."
                      type: out_of_memory.unknown.unknown"""

class ValidationErrorSchema(KoboldSchema):
    detail: Dict[str, List[str]] = fields.Dict(keys=fields.String(), values=fields.List(fields.String(), validate=validate.Length(min=1)), required=True)

api_validation_error_response = """422:
          description: Validation error
          content:
            application/json:
              schema: ValidationErrorSchema"""

class ServerBusyErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)

api_server_busy_response = """503:
          description: Server is busy
          content:
            application/json:
              schema: ServerBusyErrorSchema
              example:
                detail:
                  msg: Server is busy; please try again later.
                  type: service_unavailable"""

class NotImplementedErrorSchema(KoboldSchema):
    detail: BasicErrorSchema = fields.Nested(BasicErrorSchema, required=True)

api_not_implemented_response = """501:
          description: Not implemented
          content:
            application/json:
              schema: NotImplementedErrorSchema
              example:
                detail:
                  msg: API generation is not supported in read-only mode; please load a model and then try again.
                  type: not_implemented"""

class SamplerSettingsSchema(KoboldSchema):
    rep_pen: Optional[float] = fields.Float(validate=validate.Range(min=1), metadata={"description": "Base repetition penalty value."})
    rep_pen_range: Optional[int] = fields.Integer(validate=validate.Range(min=0), metadata={"description": "Repetition penalty range."})
    rep_pen_slope: Optional[float] = fields.Float(validate=validate.Range(min=0), metadata={"description": "Repetition penalty slope."})
    top_k: Optional[int] = fields.Integer(validate=validate.Range(min=0), metadata={"description": "Top-k sampling value."})
    top_a: Optional[float] = fields.Float(validate=validate.Range(min=0), metadata={"description": "Top-a sampling value."})
    top_p: Optional[float] = fields.Float(validate=validate.Range(min=0, max=1), metadata={"description": "Top-p sampling value."})
    tfs: Optional[float] = fields.Float(validate=validate.Range(min=0, max=1), metadata={"description": "Tail free sampling value."})
    typical: Optional[float] = fields.Float(validate=validate.Range(min=0, max=1), metadata={"description": "Typical sampling value."})
    temperature: Optional[float] = fields.Float(validate=validate.Range(min=0, min_inclusive=False), metadata={"description": "Temperature value."})

def soft_prompt_validator(soft_prompt: str):
    if len(soft_prompt.strip()) == 0:
        return
    if not koboldai_vars.allowsp:
        raise ValidationError("Cannot use soft prompts with current backend.")
    if any(q in soft_prompt for q in ("/", "\\")):
        return
    z, _, _, _, _ = fileops.checksp("./softprompts/"+soft_prompt.strip(), koboldai_vars.modeldim)
    if isinstance(z, int):
        raise ValidationError("Must be a valid soft prompt name.")
    z.close()
    return True

def story_load_validator(name: str):
    if any(q in name for q in ("/", "\\")):
        return
    if len(name.strip()) == 0 or not os.path.isfile(fileops.storypath(name)):
        raise ValidationError("Must be a valid story name.")
    return True

def permutation_validator(lst: list):
    if any(not isinstance(e, int) for e in lst):
        return
    if min(lst) != 0 or max(lst) != len(lst) - 1 or len(set(lst)) != len(lst):
        raise ValidationError("Must be a permutation of the first N non-negative integers, where N is the length of this array")
    return True

class GenerationInputSchema(SamplerSettingsSchema):
    class Meta:
        unknown = EXCLUDE # Doing it on this level is not a deliberate design choice on our part, it doesn't work nested... - Henk
    prompt: str = fields.String(required=True, metadata={"description": "This is the submission."})
    use_memory: bool = fields.Boolean(load_default=False, metadata={"description": "Whether or not to use the memory from the KoboldAI GUI when generating text."})
    use_story: bool = fields.Boolean(load_default=False, metadata={"description": "Whether or not to use the story from the KoboldAI GUI when generating text."})
    use_authors_note: bool = fields.Boolean(load_default=False, metadata={"description": "Whether or not to use the author's note from the KoboldAI GUI when generating text. This has no effect unless `use_story` is also enabled."})
    use_world_info: bool = fields.Boolean(load_default=False, metadata={"description": "Whether or not to use the world info from the KoboldAI GUI when generating text."})
    use_userscripts: bool = fields.Boolean(load_default=False, metadata={"description": "Whether or not to use the userscripts from the KoboldAI GUI when generating text."})
    soft_prompt: Optional[str] = fields.String(metadata={"description": "Soft prompt to use when generating. If set to the empty string or any other string containing no non-whitespace characters, uses no soft prompt."}, validate=[soft_prompt_validator, validate.Regexp(r"^[^/\\]*$")])
    max_length: int = fields.Integer(validate=validate.Range(min=1), metadata={"description": "Number of tokens to generate."})
    max_context_length: int = fields.Integer(validate=validate.Range(min=1), metadata={"description": "Maximum number of tokens to send to the model."})
    n: int = fields.Integer(validate=validate.Range(min=1, max=5), metadata={"description": "Number of outputs to generate."})
    disable_output_formatting: bool = fields.Boolean(load_default=True, metadata={"description": "When enabled, all output formatting options default to `false` instead of the value in the KoboldAI GUI."})
    frmttriminc: Optional[bool] = fields.Boolean(metadata={"description": "Output formatting option. When enabled, removes some characters from the end of the output such that the output doesn't end in the middle of a sentence. If the output is less than one sentence long, does nothing.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."})
    frmtrmblln: Optional[bool] = fields.Boolean(metadata={"description": "Output formatting option. When enabled, replaces all occurrences of two or more consecutive newlines in the output with one newline.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."})
    frmtrmspch: Optional[bool] = fields.Boolean(metadata={"description": "Output formatting option. When enabled, removes `#/@%{}+=~|\^<>` from the output.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."})
    singleline: Optional[bool] = fields.Boolean(metadata={"description": "Output formatting option. When enabled, removes everything after the first line of the output, including the newline.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."})
    use_default_badwordsids: bool = fields.Boolean(load_default=True, metadata={"description": "Ban tokens that commonly worsen the writing experience for continuous story writing"})
    disable_input_formatting: bool = fields.Boolean(load_default=True, metadata={"description": "When enabled, all input formatting options default to `false` instead of the value in the KoboldAI GUI"})
    frmtadsnsp: Optional[bool] = fields.Boolean(metadata={"description": "Input formatting option. When enabled, adds a leading space to your input if there is no trailing whitespace at the end of the previous action.\n\nIf `disable_input_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."})
    quiet: Optional[bool] = fields.Boolean(metadata={"description": "When enabled, Generated output will not be displayed in the console."})
    sampler_order: Optional[List[int]] = fields.List(fields.Integer(), validate=[validate.Length(min=6), permutation_validator], metadata={"description": "Sampler order to be used. If N is the length of this array, then N must be greater than or equal to 6 and the array must be a permutation of the first N non-negative integers."})
    sampler_seed: Optional[int] = fields.Integer(validate=validate.Range(min=0, max=2**64 - 1), metadata={"description": "RNG seed to use for sampling. If not specified, the global RNG will be used."})
    sampler_full_determinism: Optional[bool] = fields.Boolean(metadata={"description": "If enabled, the generated text will always be the same as long as you use the same RNG seed, input and settings. If disabled, only the *sequence* of generated texts that you get when repeatedly generating text will be the same given the same RNG seed, input and settings."})
    stop_sequence: Optional[List[str]] = fields.List(fields.String(),metadata={"description": "An array of string sequences where the API will stop generating further tokens. The returned text WILL contain the stop sequence."})


class GenerationResultSchema(KoboldSchema):
    text: str = fields.String(required=True, metadata={"description": "Generated output as plain text."})

class GenerationOutputSchema(KoboldSchema):
    results: List[GenerationResultSchema] = fields.List(fields.Nested(GenerationResultSchema), required=True, metadata={"description": "Array of generated outputs."})

class StoryNumsChunkSchema(KoboldSchema):
    num: int = fields.Integer(required=True, metadata={"description": "Guaranteed to not equal the `num` of any other active story chunk. Equals 0 iff this is the first action of the story (the prompt)."})

class StoryChunkSchema(StoryNumsChunkSchema, KoboldSchema):
    text: str = fields.String(required=True, metadata={"description": "The text inside this story chunk."})

class StorySchema(KoboldSchema):
    results: List[StoryChunkSchema] = fields.List(fields.Nested(StoryChunkSchema), required=True, metadata={"description": "Array of story actions. The array is sorted such that actions closer to the end of this array are closer to the end of the story."})

class BasicBooleanResultSchema(KoboldSchema):
    result: bool = fields.Boolean(required=True)

class StoryNumsSchema(KoboldSchema):
    results: List[int] = fields.List(fields.Integer(), required=True, metadata={"description": "Array of story action nums. The array is sorted such that actions closer to the end of this array are closer to the end of the story."})

class StoryChunkResultSchema(KoboldSchema):
    result: StoryChunkSchema = fields.Nested(StoryChunkSchema, required=True)

class StoryChunkNumSchema(KoboldSchema):
    value: int = fields.Integer(required=True)

class StoryChunkTextSchema(KoboldSchema):
    value: str = fields.String(required=True)

class StoryChunkSetTextSchema(KoboldSchema):
    value: str = fields.String(required=True, validate=validate.Regexp(r"^(.|\n)*\S$"))

class StoryLoadSchema(KoboldSchema):
    name: str = fields.String(required=True, validate=[story_load_validator, validate.Regexp(r"^[^/\\]*$")])

class StorySaveSchema(KoboldSchema):
    name: str = fields.String(required=True, validate=validate.Regexp(r"^(?=.*\S)(?!.*[/\\]).*$"))

class WorldInfoEntrySchema(KoboldSchema):
    uid: int = fields.Integer(required=True, validate=validate.Range(min=-2147483648, max=2147483647), metadata={"description": "32-bit signed integer unique to this world info entry."})
    content: str = fields.String(required=True, metadata={"description": "The \"What To Remember\" for this entry."})
    key: str = fields.String(required=True, metadata={"description": "Comma-separated list of keys, or of primary keys if selective mode is enabled."})
    keysecondary: str = fields.String(metadata={"description": "Comma-separated list of secondary keys if selective mode is enabled."})
    selective: bool = fields.Boolean(required=True, metadata={"description": "Whether or not selective mode is enabled for this world info entry."})
    constant: bool = fields.Boolean(required=True, metadata={"description": "Whether or not constant mode is enabled for this world info entry."})
    comment: bool = fields.String(required=True, metadata={"description": "The comment/description/title for this world info entry."})

class WorldInfoEntryResultSchema(KoboldSchema):
    result: WorldInfoEntrySchema = fields.Nested(WorldInfoEntrySchema, required=True)

class WorldInfoFolderBasicSchema(KoboldSchema):
    uid: int = fields.Integer(required=True, validate=validate.Range(min=-2147483648, max=2147483647), metadata={"description": "32-bit signed integer unique to this world info folder."})
    name: str = fields.String(required=True, metadata={"description": "Name of this world info folder."})

class WorldInfoFolderSchema(WorldInfoFolderBasicSchema):
    entries: List[WorldInfoEntrySchema] = fields.List(fields.Nested(WorldInfoEntrySchema), required=True)

class WorldInfoFolderUIDsSchema(KoboldSchema):
    uid: int = fields.Integer(required=True, validate=validate.Range(min=-2147483648, max=2147483647), metadata={"description": "32-bit signed integer unique to this world info folder."})
    entries: List[int] = fields.List(fields.Integer(required=True, validate=validate.Range(min=-2147483648, max=2147483647), metadata={"description": "32-bit signed integer unique to this world info entry."}), required=True)

class WorldInfoEntriesSchema(KoboldSchema):
    entries: List[WorldInfoEntrySchema] = fields.List(fields.Nested(WorldInfoEntrySchema), required=True)

class WorldInfoFoldersSchema(KoboldSchema):
    folders: List[WorldInfoFolderBasicSchema] = fields.List(fields.Nested(WorldInfoFolderBasicSchema), required=True)

class WorldInfoSchema(WorldInfoEntriesSchema):
    folders: List[WorldInfoFolderSchema] = fields.List(fields.Nested(WorldInfoFolderSchema), required=True)

class WorldInfoEntriesUIDsSchema(KoboldSchema):
    entries: List[int] = fields.List(fields.Integer(required=True, validate=validate.Range(min=-2147483648, max=2147483647), metadata={"description": "32-bit signed integer unique to this world info entry."}), required=True)

class WorldInfoFoldersUIDsSchema(KoboldSchema):
    folders: List[int] = fields.List(fields.Integer(required=True, validate=validate.Range(min=-2147483648, max=2147483647), metadata={"description": "32-bit signed integer unique to this world info folder."}), required=True)

class WorldInfoUIDsSchema(WorldInfoEntriesUIDsSchema):
    folders: List[WorldInfoFolderSchema] = fields.List(fields.Nested(WorldInfoFolderUIDsSchema), required=True)

class ModelSelectionSchema(KoboldSchema):
    model: str = fields.String(required=True, validate=validate.Regexp(r"^(?!\s*NeoCustom)(?!\s*GPT2Custom)(?!\s*TPUMeshTransformerGPTJ)(?!\s*TPUMeshTransformerGPTNeoX)(?!\s*GooseAI)(?!\s*OAI)(?!\s*InferKit)(?!\s*Colab)(?!\s*API).*$"), metadata={"description": 'Hugging Face model ID, the path to a model folder (relative to the "models" folder in the KoboldAI root folder) or "ReadOnly" for no model'})
    backend: Optional[str] = fields.String(required=False, validate=validate.OneOf(model_backends.keys()))

def _generate_text(body: GenerationInputSchema):
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
    if koboldai_vars.use_colab_tpu:
        import tpu_mtj_backend
        tpu_mtj_backend.socketio = socketio
    if hasattr(body, "sampler_seed"):
        # If a seed was specified, we need to save the global RNG state so we
        # can restore it later
        old_seed = koboldai_vars.seed
        old_rng_state = tpu_mtj_backend.get_rng_state() if koboldai_vars.use_colab_tpu else torch.get_rng_state()
        koboldai_vars.seed = body.sampler_seed
        # We should try to use a previously saved RNG state with the same seed
        if body.sampler_seed in koboldai_vars.rng_states:
            if koboldai_vars.use_colab_tpu:
                tpu_mtj_backend.set_rng_state(koboldai_vars.rng_states[body.sampler_seed])
            else:
                torch.set_rng_state(koboldai_vars.rng_states[body.sampler_seed])
        else:
            if koboldai_vars.use_colab_tpu:
                tpu_mtj_backend.set_rng_state(tpu_mtj_backend.new_rng_state(body.sampler_seed))
            else:
                torch.manual_seed(body.sampler_seed)
        koboldai_vars.rng_states[body.sampler_seed] = tpu_mtj_backend.get_rng_state() if koboldai_vars.use_colab_tpu else torch.get_rng_state()
    if hasattr(body, "sampler_order"):
        if len(body.sampler_order) < 7:
            body.sampler_order = [6] + body.sampler_order
    # This maps each property of the setting to use when sending the generate idempotently
    # To the object which typically contains it's value
    # This allows to set the property only for the API generation, and then revert the setting
    # To what it was before.
    mapping = {
        "disable_input_formatting": ("koboldai_vars", "disable_input_formatting", None),
        "disable_output_formatting": ("koboldai_vars", "disable_output_formatting", None),
        "rep_pen": ("koboldai_vars", "rep_pen", None),
        "rep_pen_range": ("koboldai_vars", "rep_pen_range", None),
        "rep_pen_slope": ("koboldai_vars", "rep_pen_slope", None),
        "top_k": ("koboldai_vars", "top_k", None),
        "top_a": ("koboldai_vars", "top_a", None),
        "top_p": ("koboldai_vars", "top_p", None),
        "tfs": ("koboldai_vars", "tfs", None),
        "typical": ("koboldai_vars", "typical", None),
        "temperature": ("koboldai_vars", "temp", None),
        "frmtadsnsp": ("koboldai_vars", "frmtadsnsp", "input"),
        "frmttriminc": ("koboldai_vars", "frmttriminc", "output"),
        "frmtrmblln": ("koboldai_vars", "frmtrmblln", "output"),
        "frmtrmspch": ("koboldai_vars", "frmtrmspch", "output"),
        "singleline": ("koboldai_vars", "singleline", "output"),
        "max_length": ("koboldai_vars", "genamt", None),
        "max_context_length": ("koboldai_vars", "max_length", None),
        "n": ("koboldai_vars", "numseqs", None),
        "quiet": ("koboldai_vars", "quiet", None),
        "sampler_order": ("koboldai_vars", "sampler_order", None),
        "sampler_full_determinism": ("koboldai_vars", "full_determinism", None),
        "stop_sequence": ("koboldai_vars", "stop_sequence", None),
        "use_default_badwordsids": ("koboldai_vars", "use_default_badwordsids", None),
    }
    saved_settings = {}
    set_aibusy(1)
    disable_set_aibusy = koboldai_vars.disable_set_aibusy
    koboldai_vars.disable_set_aibusy = True
    _standalone = koboldai_vars.standalone
    koboldai_vars.standalone = True
    show_probs = koboldai_vars.show_probs
    koboldai_vars.show_probs = False
    output_streaming = koboldai_vars.output_streaming
    koboldai_vars.output_streaming = False
    for key, entry in mapping.items():
        obj = {"koboldai_vars": koboldai_vars}[entry[0]]
        if entry[2] == "input" and koboldai_vars.disable_input_formatting and not hasattr(body, key):
            setattr(body, key, False)
        if entry[2] == "output" and koboldai_vars.disable_output_formatting and not hasattr(body, key):
            setattr(body, key, False)
        if getattr(body, key, None) is not None:
            if entry[1].startswith("@"):
                saved_settings[key] = obj[entry[1][1:]]
                obj[entry[1][1:]] = getattr(body, key)
            else:
                saved_settings[key] = getattr(obj, entry[1])
                setattr(obj, entry[1], getattr(body, key))
    try:
        if koboldai_vars.allowsp and getattr(body, "soft_prompt", None) is not None:
            if any(q in body.soft_prompt for q in ("/", "\\")):
                raise RuntimeError
            old_spfilename = koboldai_vars.spfilename
            spRequest(body.soft_prompt.strip())
        genout = apiactionsubmit(body.prompt, use_memory=body.use_memory, use_story=body.use_story, use_world_info=body.use_world_info, use_authors_note=body.use_authors_note)
        output = {"results": [{"text": txt} for txt in genout]}
    finally:
        for key in saved_settings:
            entry = mapping[key]
            obj = {"koboldai_vars": koboldai_vars}[entry[0]]
            if getattr(body, key, None) is not None:
                if entry[1].startswith("@"):
                    if obj[entry[1][1:]] == getattr(body, key):
                        obj[entry[1][1:]] = saved_settings[key]
                else:
                    if getattr(obj, entry[1]) == getattr(body, key):
                        setattr(obj, entry[1], saved_settings[key])
        koboldai_vars.disable_set_aibusy = disable_set_aibusy
        koboldai_vars.standalone = _standalone
        koboldai_vars.show_probs = show_probs
        koboldai_vars.output_streaming = output_streaming
        if koboldai_vars.allowsp and getattr(body, "soft_prompt", None) is not None:
            spRequest(old_spfilename)
        if hasattr(body, "sampler_seed"):
            koboldai_vars.seed = old_seed
            if koboldai_vars.use_colab_tpu:
                tpu_mtj_backend.set_rng_state(old_rng_state)
            else:
                torch.set_rng_state(old_rng_state)
        set_aibusy(0)
    return output


@api_v1.get("/info/version")
@api_schema_wrap
def get_version():
    """---
    get:
      summary: Current API version
      tags:
        - info
      description: |-2
        Returns the version of the API that you are currently using.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicResultSchema
              example:
                result: 1.0.0
    """
    return {"result": api_version}


@api_v1.get("/info/version/latest")
@api_schema_wrap
def get_version_latest():
    """---
    get:
      summary: Latest API version
      tags:
        - info
      description: |-2
        Returns the latest API version available.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicResultSchema
              example:
                result: 1.0.0
    """
    return {"result": api_versions[-1]}


@api_v1.get("/info/version/list")
@api_schema_wrap
def get_version_list():
    """---
    get:
      summary: List API versions
      tags:
        - info
      description: |-2
        Returns a list of available API versions sorted in ascending order.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicResultsSchema
              example:
                results:
                  - 1.0.0
    """
    return {"results": api_versions}


@api_v1.post("/generate")
@api_schema_wrap
def post_generate(body: GenerationInputSchema):
    """---
    post:
      summary: Generate text
      tags:
        - generate
      description: |-2
        Generates text given a submission, sampler settings, soft prompt and number of return sequences.

        By default, the story, userscripts, memory, author's note and world info are disabled.

        Unless otherwise specified, optional values default to the values in the KoboldAI GUI.
      requestBody:
        required: true
        content:
          application/json:
            schema: GenerationInputSchema
            example:
              prompt: |-2
                Niko the kobold stalked carefully down the alley, his small scaly figure obscured by a dusky cloak that fluttered lightly in the cold winter breeze.
              top_p: 0.9
              temperature: 0.5
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: GenerationOutputSchema
              example:
                results:
                  - text: |-2
                       Holding up his tail to keep it from dragging in the dirty snow that covered the cobblestone, he waited patiently for the butcher to turn his attention from his stall so that he could pilfer his next meal: a tender-looking chicken.
        {api_validation_error_response}
        {api_not_implemented_response}
        {api_server_busy_response}
        {api_out_of_memory_response}
    """
    return _generate_text(body)


@api_v1.get("/model")
@api_schema_wrap
def get_model():
    """---
    get:
      summary: Retrieve the current model string
      description: |-2
        Gets the current model string, which is shown in the title of the KoboldAI GUI in parentheses, e.g. "KoboldAI Client (KoboldAI/fairseq-dense-13B-Nerys-v2)".
      tags:
        - model
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicResultSchema
              example:
                result: KoboldAI/fairseq-dense-13B-Nerys-v2
    """
    return {"result": koboldai_vars.model}


@api_v1.put("/model")
@api_schema_wrap
def put_model(body: ModelSelectionSchema):
    """---
    put:
      summary: Load a model
      description: |-2
        Loads a model given its Hugging Face model ID, the path to a model folder (relative to the "models" folder in the KoboldAI root folder) or "ReadOnly" for no model.
        Optionally, a backend parameter can be passed in to dictate which backend loads the model.
      tags:
        - model
      requestBody:
        required: true
        content:
          application/json:
            schema: ModelSelectionSchema
            example:
              model: ReadOnly
              backend: Read Only
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
        {api_server_busy_response}
    """
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
    set_aibusy(1)
    old_model = koboldai_vars.model
    koboldai_vars.model = body.model.strip()

    backend = getattr(body, "backend", None)
    if not backend:
        # Backend is optional for backwards compatibility; it should probably be
        # required on the next major API version.
        if body.model == "ReadOnly":
            backend = "Read Only"
        else:
            backend = "Huggingface"

    try:
        if 'model' in globals():
            model.unload()
        load_model(backend)
    except Exception as e:
        koboldai_vars.model = old_model
        raise e
    set_aibusy(0)
    return {}


def prompt_validator(prompt: str):
    if len(prompt.strip()) == 0:
        raise ValidationError("String does not match expected pattern.")

class SubmissionInputSchema(KoboldSchema):
    prompt: str = fields.String(required=True, validate=prompt_validator, metadata={"pattern": r"^[\S\s]*\S[\S\s]*$", "description": "This is the submission."})
    disable_input_formatting: bool = fields.Boolean(load_default=True, metadata={"description": "When enabled, disables all input formatting options, overriding their individual enabled/disabled states."})
    frmtadsnsp: Optional[bool] = fields.Boolean(metadata={"description": "Input formatting option. When enabled, adds a leading space to your input if there is no trailing whitespace at the end of the previous action."})

@api_v1.post("/story/end")
@api_schema_wrap
def post_story_end(body: SubmissionInputSchema):
    """---
    post:
      summary: Add an action to the end of the story
      tags:
        - story
      description: |-2
        Inserts a single action at the end of the story in the KoboldAI GUI without generating text.
      requestBody:
        required: true
        content:
          application/json:
            schema: SubmissionInputSchema
            example:
              prompt: |-2
                 This is some text to put at the end of the story.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
        {api_server_busy_response}
    """
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
    set_aibusy(1)
    disable_set_aibusy = koboldai_vars.disable_set_aibusy
    koboldai_vars.disable_set_aibusy = True
    _standalone = koboldai_vars.standalone
    koboldai_vars.standalone = True
    numseqs = koboldai_vars.numseqs
    koboldai_vars.numseqs = 1
    try:
        actionsubmit(body.prompt, force_submit=True, no_generate=True, ignore_aibusy=True)
    finally:
        koboldai_vars.disable_set_aibusy = disable_set_aibusy
        koboldai_vars.standalone = _standalone
        koboldai_vars.numseqs = numseqs
    set_aibusy(0)
    return {}


@api_v1.get("/story/end")
@api_schema_wrap
def get_story_end():
    """---
    get:
      summary: Retrieve the last action of the story
      tags:
        - story
      description: |-2
        Returns the last action of the story in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StoryChunkResultSchema
        510:
          description: Story is empty
          content:
            application/json:
              schema: StoryEmptyErrorSchema
              example:
                detail:
                  msg: Could not retrieve the last action of the story because the story is empty.
                  type: story_empty
    """
    if not koboldai_vars.gamestarted:
        abort(Response(json.dumps({"detail": {
            "msg": "Could not retrieve the last action of the story because the story is empty.",
            "type": "story_empty",
        }}), mimetype="application/json", status=510))
    if len(koboldai_vars.actions) == 0:
        return {"result": {"text": koboldai_vars.prompt, "num": 0}}
    return {"result": {"text": koboldai_vars.actions[koboldai_vars.actions.get_last_key()], "num": koboldai_vars.actions.get_last_key() + 1}}


@api_v1.get("/story/end/num")
@api_schema_wrap
def get_story_end_num():
    """---
    get:
      summary: Retrieve the num of the last action of the story
      tags:
        - story
      description: |-2
        Returns the `num` of the last action of the story in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StoryChunkNumSchema
        510:
          description: Story is empty
          content:
            application/json:
              schema: StoryEmptyErrorSchema
              example:
                detail:
                  msg: Could not retrieve the last action of the story because the story is empty.
                  type: story_empty
    """
    if not koboldai_vars.gamestarted:
        abort(Response(json.dumps({"detail": {
            "msg": "Could not retrieve the last action of the story because the story is empty.",
            "type": "story_empty",
        }}), mimetype="application/json", status=510))
    if len(koboldai_vars.actions) == 0:
        return {"result": {"text": 0}}
    return {"result": {"text": koboldai_vars.actions.get_last_key() + 1}}


@api_v1.get("/story/end/text")
@api_schema_wrap
def get_story_end_text():
    """---
    get:
      summary: Retrieve the text of the last action of the story
      tags:
        - story
      description: |-2
        Returns the text of the last action of the story in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StoryChunkTextSchema
        510:
          description: Story is empty
          content:
            application/json:
              schema: StoryEmptyErrorSchema
              example:
                detail:
                  msg: Could not retrieve the last action of the story because the story is empty.
                  type: story_empty
    """
    if not koboldai_vars.gamestarted:
        abort(Response(json.dumps({"detail": {
            "msg": "Could not retrieve the last action of the story because the story is empty.",
            "type": "story_empty",
        }}), mimetype="application/json", status=510))
    if len(koboldai_vars.actions) == 0:
        return {"result": {"text": koboldai_vars.prompt}}
    return {"result": {"text": koboldai_vars.actions[koboldai_vars.actions.get_last_key()]}}


@api_v1.put("/story/end/text")
@api_schema_wrap
def put_story_end_text(body: StoryChunkSetTextSchema):
    """---
    put:
      summary: Set the text of the last action of the story
      tags:
        - story
      description: |-2
        Sets the text of the last action of the story in the KoboldAI GUI to the desired value.
      requestBody:
        required: true
        content:
          application/json:
            schema: StoryChunkSetTextSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        510:
          description: Story is empty
          content:
            application/json:
              schema: StoryEmptyErrorSchema
              example:
                detail:
                  msg: Could not retrieve the last action of the story because the story is empty.
                  type: story_empty
        {api_validation_error_response}
    """
    if not koboldai_vars.gamestarted:
        abort(Response(json.dumps({"detail": {
            "msg": "Could not retrieve the last action of the story because the story is empty.",
            "type": "story_empty",
        }}), mimetype="application/json", status=510))
    value = body.value.rstrip()
    if len(koboldai_vars.actions) == 0:
        inlineedit(0, value)
    else:
        inlineedit(koboldai_vars.actions.get_last_key() + 1, value)
    return {}


@api_v1.post("/story/end/delete")
@api_schema_wrap
def post_story_end_delete(body: EmptySchema):
    """---
    post:
      summary: Remove the last action of the story
      tags:
        - story
      description: |-2
        Removes the last action of the story in the KoboldAI GUI.
      requestBody:
        required: true
        content:
          application/json:
            schema: EmptySchema
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        510:
          description: Story too short
          content:
            application/json:
              schema: StoryTooShortErrorSchema
              example:
                detail:
                  msg: Could not delete the last action of the story because the number of actions in the story is less than or equal to 1.
                  type: story_too_short
        {api_validation_error_response}
        {api_server_busy_response}
    """
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
    if not koboldai_vars.gamestarted or not len(koboldai_vars.actions):
        abort(Response(json.dumps({"detail": {
            "msg": "Could not delete the last action of the story because the number of actions in the story is less than or equal to 1.",
            "type": "story_too_short",
        }}), mimetype="application/json", status=510))
    actionback()
    return {}


@api_v1.get("/story")
@api_schema_wrap
def get_story():
    """---
    get:
      summary: Retrieve the entire story
      tags:
        - story
      description: |-2
        Returns the entire story currently shown in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StorySchema
    """
    chunks = []
    if koboldai_vars.gamestarted:
        chunks.append({"num": 0, "text": koboldai_vars.prompt})

    last_action_num = list(koboldai_vars.actions.actions.keys())[-1]
    for num, action in koboldai_vars.actions.actions.items():
        text = action["Selected Text"]
        # The last action seems to always be empty
        if not text and num == last_action_num:
            continue
        chunks.append({"num": num + 1, "text": text})
    return {"results": chunks}


@api_v1.get("/story/nums")
@api_schema_wrap
def get_story_nums():
    """---
    get:
      summary: Retrieve a list of the nums of the chunks in the current story
      tags:
        - story
      description: |-2
        Returns the `num`s of the story chunks currently shown in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StorySchema
    """
    chunks = []
    if koboldai_vars.gamestarted:
        chunks.append(0)
    for num in koboldai_vars.actions.actions.keys():
        chunks.append(num + 1)
    return {"results": chunks}


@api_v1.get("/story/nums/<int(signed=True):num>")
@api_schema_wrap
def get_story_nums_num(num: int):
    """---
    get:
      summary: Determine whether or not there is a story chunk with the given num
      tags:
        - story
      parameters:
        - name: num
          in: path
          description: |-2
            `num` of the desired story chunk.
          schema:
            type: integer
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
    """
    if num == 0:
        return {"result": koboldai_vars.gamestarted}
    return {"result": num - 1 in koboldai_vars.actions}


@api_v1.get("/story/<int(signed=True):num>")
@api_schema_wrap
def get_story_num(num: int):
    """---
    get:
      summary: Retrieve a story chunk
      tags:
        - story
      description: |-2
        Returns information about a story chunk given its `num`.
      parameters:
        - name: num
          in: path
          description: |-2
            `num` of the desired story chunk.
          schema:
            type: integer
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StoryChunkResultSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No chunk with the given num exists.
                  type: key_error
    """
    if num == 0:
        if not koboldai_vars.gamestarted:
            abort(Response(json.dumps({"detail": {
                "msg": "No chunk with the given num exists.",
                "type": "key_error",
            }}), mimetype="application/json", status=404))
        return {"result": {"text": koboldai_vars.prompt, "num": num}}
    if num - 1 not in koboldai_vars.actions:
        abort(Response(json.dumps({"detail": {
            "msg": "No chunk with the given num exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"result": {"text": koboldai_vars.actions[num - 1], "num": num}}


@api_v1.get("/story/<int(signed=True):num>/text")
@api_schema_wrap
def get_story_num_text(num: int):
    """---
    get:
      summary: Retrieve the text of a story chunk
      tags:
        - story
      description: |-2
        Returns the text inside a story chunk given its `num`.
      parameters:
        - name: num
          in: path
          description: |-2
            `num` of the desired story chunk.
          schema:
            type: integer
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: StoryChunkTextSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No chunk with the given num exists.
                  type: key_error
    """
    if num == 0:
        if not koboldai_vars.gamestarted:
            abort(Response(json.dumps({"detail": {
                "msg": "No chunk with the given num exists.",
                "type": "key_error",
            }}), mimetype="application/json", status=404))
        return {"value": koboldai_vars.prompt}
    if num - 1 not in koboldai_vars.actions:
        abort(Response(json.dumps({"detail": {
            "msg": "No chunk with the given num exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": koboldai_vars.actions[num - 1]}


@api_v1.put("/story/<int(signed=True):num>/text")
@api_schema_wrap
def put_story_num_text(body: StoryChunkSetTextSchema, num: int):
    """---
    put:
      summary: Set the text of a story chunk
      tags:
        - story
      description: |-2
        Sets the text inside a story chunk given its `num`.
      parameters:
        - name: num
          in: path
          description: |-2
            `num` of the desired story chunk.
          schema:
            type: integer
      requestBody:
        required: true
        content:
          application/json:
            schema: StoryChunkSetTextSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No chunk with the given num exists.
                  type: key_error
        {api_validation_error_response}
    """
    if num == 0:
        if not koboldai_vars.gamestarted:
            abort(Response(json.dumps({"detail": {
                "msg": "No chunk with the given num exists.",
                "type": "key_error",
            }}), mimetype="application/json", status=404))
        inlineedit(0, body.value.rstrip())
        return {}
    if num - 1 not in koboldai_vars.actions:
        abort(Response(json.dumps({"detail": {
            "msg": "No chunk with the given num exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    inlineedit(num, body.value.rstrip())
    return {}


@api_v1.delete("/story/<int(signed=True):num>")
@api_schema_wrap
def post_story_num_delete(num: int):
    """---
    delete:
      summary: Remove a story chunk
      tags:
        - story
      description: |-2
        Removes a story chunk from the story in the KoboldAI GUI given its `num`. Cannot be used to delete the first action (the prompt).
      parameters:
        - name: num
          in: path
          description: |-2
            `num` of the desired story chunk. Must be larger than or equal to 1.
          schema:
            type: integer
            minimum: 1
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No chunk with the given num exists.
                  type: key_error
        {api_server_busy_response}
    """
    if num < 1:
        abort(Response(json.dumps({"detail": {
            "num": ["Must be greater than or equal to 1."],
        }}), mimetype="application/json", status=422))
    if num - 1 not in koboldai_vars.actions:
        abort(Response(json.dumps({"detail": {
            "msg": "No chunk with the given num exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
    inlinedelete(num)
    return {}


@api_v1.delete("/story")
@api_schema_wrap
def delete_story():
    """---
    delete:
      summary: Clear the story
      tags:
        - story
      description: |-2
        Starts a new blank story.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_server_busy_response}
    """
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
    newGameRequest()
    return {}


@api_v1.put("/story/load")
@api_schema_wrap
def put_story_load(body: StoryLoadSchema):
    """---
    put:
      summary: Load a story
      tags:
        - story
      description: |-2
        Loads a story given its filename (without the .json).
      requestBody:
        required: true
        content:
          application/json:
            schema: StoryLoadSchema
            example:
              name: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
        {api_server_busy_response}
    """
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
    loadRequest(fileops.storypath(body.name.strip()))
    return {}


@api_v1.put("/story/save")
@api_schema_wrap
def put_story_save(body: StorySaveSchema):
    """---
    put:
      summary: Save the current story
      tags:
        - story
      description: |-2
        Saves the current story given its destination filename (without the .json).
      requestBody:
        required: true
        content:
          application/json:
            schema: StorySaveSchema
            example:
              name: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
    """
    saveRequest(fileops.storypath(body.name.strip()))
    return {}


@api_v1.get("/world_info")
@api_schema_wrap
def get_world_info():
    """---
    get:
      summary: Retrieve all world info entries
      tags:
        - world_info
      description: |-2
        Returns all world info entries currently shown in the KoboldAI GUI.

        The `folders` are sorted in the same order as they are in the GUI and the `entries` within the folders and within the parent `result` object are all sorted in the same order as they are in their respective parts of the GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoSchema
    """
    folders = []
    entries = []
    ln = len(koboldai_vars.worldinfo)
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    folder: Optional[list] = None
    if ln:
        last_folder = ...
        for wi in koboldai_vars.worldinfo_i:
            if wi["folder"] != last_folder:
                folder = []
                if wi["folder"] is not None:
                    folders.append({"uid": wi["folder"], "name": koboldai_vars.wifolders_d[wi["folder"]]["name"], "entries": folder})
                last_folder = wi["folder"]
            (folder if wi["folder"] is not None else entries).append({k: v for k, v in wi.items() if k not in ("init", "folder", "num") and (wi["selective"] or k != "keysecondary")})
    return {"folders": folders, "entries": entries}

@api_v1.get("/world_info/uids")
@api_schema_wrap
def get_world_info_uids():
    """---
    get:
      summary: Retrieve the UIDs of all world info entries
      tags:
        - world_info
      description: |-2
        Returns in a similar format as GET /world_info except only the `uid`s are returned.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoUIDsSchema
    """
    folders = []
    entries = []
    ln = len(koboldai_vars.worldinfo)
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    folder: Optional[list] = None
    if ln:
        last_folder = ...
        for wi in koboldai_vars.worldinfo_i:
            if wi["folder"] != last_folder:
                folder = []
                if wi["folder"] is not None:
                    folders.append({"uid": wi["folder"], "entries": folder})
                last_folder = wi["folder"]
            (folder if wi["folder"] is not None else entries).append(wi["uid"])
    return {"folders": folders, "entries": entries}


@api_v1.get("/world_info/uids/<int(signed=True):uid>")
@api_schema_wrap
def get_world_info_uids_uid(uid: int):
    """---
    get:
      summary: Determine whether or not there is a world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
    """
    return {"result": uid in koboldai_vars.worldinfo_u and koboldai_vars.worldinfo_u[uid]["init"]}


@api_v1.get("/world_info/folders")
@api_schema_wrap
def get_world_info_folders():
    """---
    get:
      summary: Retrieve all world info folders
      tags:
        - world_info
      description: |-2
        Returns details about all world info folders currently shown in the KoboldAI GUI.

        The `folders` are sorted in the same order as they are in the GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoFoldersSchema
    """
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    return {"folders": [{"uid": folder, **{k: v for k, v in koboldai_vars.wifolders_d[folder].items() if k != "collapsed"}} for folder in koboldai_vars.wifolders_l]}


@api_v1.get("/world_info/folders/uids")
@api_schema_wrap
def get_world_info_folders_uids():
    """---
    get:
      summary: Retrieve the UIDs all world info folders
      tags:
        - world_info
      description: |-2
        Returns the `uid`s of all world info folders currently shown in the KoboldAI GUI.

        The `folders` are sorted in the same order as they are in the GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoFoldersUIDsSchema
    """
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    return {"folders": koboldai_vars.wifolders_l}


@api_v1.get("/world_info/folders/none")
@api_schema_wrap
def get_world_info_folders_none():
    """---
    get:
      summary: Retrieve all world info entries not in a folder
      tags:
        - world_info
      description: |-2
        Returns all world info entries that are not in a world info folder.

        The `entries` are sorted in the same order as they are in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoEntriesSchema
    """
    entries = []
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    for wi in reversed(koboldai_vars.worldinfo_i):
        if wi["folder"] is not None:
            break
        entries.append({k: v for k, v in wi.items() if k not in ("init", "folder", "num") and (wi["selective"] or k != "keysecondary")})
    return {"entries": list(reversed(entries))}


@api_v1.get("/world_info/folders/none/uids")
@api_schema_wrap
def get_world_info_folders_none_uids():
    """---
    get:
      summary: Retrieve the UIDs of all world info entries not in a folder
      tags:
        - world_info
      description: |-2
        Returns the `uid`s of all world info entries that are not in a world info folder.

        The `entries` are sorted in the same order as they are in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoEntriesUIDsSchema
    """
    entries = []
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    for wi in reversed(koboldai_vars.worldinfo_i):
        if wi["folder"] is not None:
            break
        entries.append(wi["uid"])
    return {"entries": list(reversed(entries))}


@api_v1.get("/world_info/folders/none/uids/<int(signed=True):uid>")
@api_schema_wrap
def get_world_info_folders_none_uids_uid(uid: int):
    """---
    get:
      summary: Determine whether or not there is a world info entry with the given UID that is not in a world info folder
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
    """
    return {"result": uid in koboldai_vars.worldinfo_u and koboldai_vars.worldinfo_u[uid]["folder"] is None and koboldai_vars.worldinfo_u[uid]["init"]}


@api_v1.get("/world_info/folders/<int(signed=True):uid>")
@api_schema_wrap
def get_world_info_folders_uid(uid: int):
    """---
    get:
      summary: Retrieve all world info entries in the given folder
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      description: |-2
        Returns all world info entries that are in the world info folder with the given `uid`.

        The `entries` are sorted in the same order as they are in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoEntriesSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folder with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info folder with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    entries = []
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    for wi in koboldai_vars.wifolders_u[uid]:
        if wi["init"]:
            entries.append({k: v for k, v in wi.items() if k not in ("init", "folder", "num") and (wi["selective"] or k != "keysecondary")})
    return {"entries": entries}


@api_v1.get("/world_info/folders/<int(signed=True):uid>/uids")
@api_schema_wrap
def get_world_info_folders_uid_uids(uid: int):
    """---
    get:
      summary: Retrieve the UIDs of all world info entries in the given folder
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      description: |-2
        Returns the `uid`s of all world info entries that are in the world info folder with the given `uid`.

        The `entries` are sorted in the same order as they are in the KoboldAI GUI.
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoEntriesUIDsSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folder with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info folder with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    entries = []
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    for wi in koboldai_vars.wifolders_u[uid]:
        if wi["init"]:
            entries.append(wi["uid"])
    return {"entries": entries}


@api_v1.get("/world_info/folders/<int(signed=True):folder_uid>/uids/<int(signed=True):entry_uid>")
@api_schema_wrap
def get_world_info_folders_folder_uid_uids_entry_uid(folder_uid: int, entry_uid: int):
    """---
    get:
      summary: Determine whether or not there is a world info entry with the given UID in the world info folder with the given UID
      tags:
        - world_info
      parameters:
        - name: folder_uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
        - name: entry_uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
    """
    return {"result": entry_uid in koboldai_vars.worldinfo_u and koboldai_vars.worldinfo_u[entry_uid]["folder"] == folder_uid and koboldai_vars.worldinfo_u[entry_uid]["init"]}


@api_v1.get("/world_info/folders/<int(signed=True):uid>/name")
@api_schema_wrap
def get_world_info_folders_uid_name(uid: int):
    """---
    get:
      summary: Retrieve the name of the world info folder with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicStringSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folder with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info folder with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": koboldai_vars.wifolders_d[uid]["name"]}


@api_v1.put("/world_info/folders/<int(signed=True):uid>/name")
@api_schema_wrap
def put_world_info_folders_uid_name(body: BasicStringSchema, uid: int):
    """---
    put:
      summary: Set the name of the world info folder with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicStringSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folder with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info folder with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    koboldai_vars.wifolders_d[uid]["name"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>")
@api_schema_wrap
def get_world_info_uid(uid: int):
    """---
    get:
      summary: Retrieve information about the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: WorldInfoEntrySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    wi = koboldai_vars.worldinfo_u[uid]
    return {k: v for k, v in wi.items() if k not in ("init", "folder", "num") and (wi["selective"] or k != "keysecondary")}


@api_v1.get("/world_info/<int(signed=True):uid>/comment")
@api_schema_wrap
def get_world_info_uid_comment(uid: int):
    """---
    get:
      summary: Retrieve the comment of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicStringSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": koboldai_vars.worldinfo_u[uid]["comment"]}


@api_v1.put("/world_info/<int(signed=True):uid>/comment")
@api_schema_wrap
def put_world_info_uid_comment(body: BasicStringSchema, uid: int):
    """---
    put:
      summary: Set the comment of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicStringSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    koboldai_vars.worldinfo_u[uid]["comment"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>/content")
@api_schema_wrap
def get_world_info_uid_content(uid: int):
    """---
    get:
      summary: Retrieve the content of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicStringSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": koboldai_vars.worldinfo_u[uid]["content"]}


@api_v1.put("/world_info/<int(signed=True):uid>/content")
@api_schema_wrap
def put_world_info_uid_content(body: BasicStringSchema, uid: int):
    """---
    put:
      summary: Set the content of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicStringSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    koboldai_vars.worldinfo_u[uid]["content"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>/key")
@api_schema_wrap
def get_world_info_uid_key(uid: int):
    """---
    get:
      summary: Retrieve the keys or primary keys of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicStringSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": koboldai_vars.worldinfo_u[uid]["key"]}


@api_v1.put("/world_info/<int(signed=True):uid>/key")
@api_schema_wrap
def put_world_info_uid_key(body: BasicStringSchema, uid: int):
    """---
    put:
      summary: Set the keys or primary keys of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicStringSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    koboldai_vars.worldinfo_u[uid]["key"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>/keysecondary")
@api_schema_wrap
def get_world_info_uid_keysecondary(uid: int):
    """---
    get:
      summary: Retrieve the secondary keys of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicStringSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": koboldai_vars.worldinfo_u[uid]["keysecondary"]}


@api_v1.put("/world_info/<int(signed=True):uid>/keysecondary")
@api_schema_wrap
def put_world_info_uid_keysecondary(body: BasicStringSchema, uid: int):
    """---
    put:
      summary: Set the secondary keys of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicStringSchema
            example:
              value: string
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    koboldai_vars.worldinfo_u[uid]["keysecondary"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>/selective")
@api_schema_wrap
def get_world_info_uid_selective(uid: int):
    """---
    get:
      summary: Retrieve the selective mode state of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": koboldai_vars.worldinfo_u[uid]["selective"]}


@api_v1.put("/world_info/<int(signed=True):uid>/selective")
@api_schema_wrap
def put_world_info_uid_selective(body: BasicBooleanSchema, uid: int):
    """---
    put:
      summary: Set the selective mode state of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicBooleanSchema
            example:
              value: true
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    koboldai_vars.worldinfo_u[uid]["selective"] = body.value
    setgamesaved(False)
    return {}


@api_v1.get("/world_info/<int(signed=True):uid>/constant")
@api_schema_wrap
def get_world_info_uid_constant(uid: int):
    """---
    get:
      summary: Retrieve the constant mode state of the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicBooleanResultSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": koboldai_vars.worldinfo_u[uid]["constant"]}


@api_v1.put("/world_info/<int(signed=True):uid>/constant")
@api_schema_wrap
def put_world_info_uid_constant(body: BasicBooleanSchema, uid: int):
    """---
    put:
      summary: Set the constant mode state of the world info entry with the given UID to the specified value
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: BasicBooleanSchema
            example:
              value: true
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    koboldai_vars.worldinfo_u[uid]["constant"] = body.value
    setgamesaved(False)
    return {}


@api_v1.post("/world_info/folders/none")
@api_schema_wrap
def post_world_info_folders_none(body: EmptySchema):
    """---
    post:
      summary: Create a new world info entry outside of a world info folder, at the end of the world info
      tags:
        - world_info
      requestBody:
        required: true
        content:
          application/json:
            schema: EmptySchema
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicUIDSchema
        {api_validation_error_response}
    """
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    setgamesaved(False)
    emit('from_server', {'cmd': 'wiexpand', 'data': koboldai_vars.worldinfo[-1]["num"]}, broadcast=True)
    koboldai_vars.worldinfo[-1]["init"] = True
    addwiitem(folder_uid=None)
    return {"uid": koboldai_vars.worldinfo[-2]["uid"]}


@api_v1.post("/world_info/folders/<int(signed=True):uid>")
@api_schema_wrap
def post_world_info_folders_uid(body: EmptySchema, uid: int):
    """---
    post:
      summary: Create a new world info entry at the end of the world info folder with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      requestBody:
        required: true
        content:
          application/json:
            schema: EmptySchema
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicUIDSchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folder with the given uid exists.
                  type: key_error
        {api_validation_error_response}
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info folder with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]
    setgamesaved(False)
    emit('from_server', {'cmd': 'wiexpand', 'data': koboldai_vars.wifolders_u[uid][-1]["num"]}, broadcast=True)
    koboldai_vars.wifolders_u[uid][-1]["init"] = True
    addwiitem(folder_uid=uid)
    return {"uid": koboldai_vars.wifolders_u[uid][-2]["uid"]}


@api_v1.delete("/world_info/<int(signed=True):uid>")
@api_schema_wrap
def delete_world_info_uid(uid: int):
    """---
    delete:
      summary: Delete the world info entry with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info entry.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info entry with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    deletewi(uid)
    return {}


@api_v1.post("/world_info/folders")
@api_schema_wrap
def post_world_info_folders(body: EmptySchema):
    """---
    post:
      summary: Create a new world info folder at the end of the world info
      tags:
        - world_info
      requestBody:
        required: true
        content:
          application/json:
            schema: EmptySchema
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: BasicUIDSchema
        {api_validation_error_response}
    """
    addwifolder()
    return {"uid": koboldai_vars.wifolders_l[-1]}


@api_v1.delete("/world_info/folders/<int(signed=True):uid>")
@api_schema_wrap
def delete_world_info_folders_uid(uid: int):
    """---
    delete:
      summary: Delete the world info folder with the given UID
      tags:
        - world_info
      parameters:
        - name: uid
          in: path
          description: |-2
            `uid` of the desired world info folder.
          schema:
            type: integer
            minimum: -2147483648
            maximum: 2147483647
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        404:
          description: Not found
          content:
            application/json:
              schema: NotFoundErrorSchema
              example:
                detail:
                  msg: No world info folders with the given uid exists.
                  type: key_error
    """
    if uid not in koboldai_vars.wifolders_d:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info folder with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    deletewifolder(uid)
    return {}


def _make_f_get(obj, _var_name, _name, _schema, _example_yaml_value):
    def f_get():
        """---
    get:
      summary: Retrieve the current {} setting value
      tags:
        - config
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: {}
              example:
                value: {}
        """
        _obj = {"koboldai_vars": koboldai_vars}[obj]
        if _var_name.startswith("@"):
            return {"value": _obj[_var_name[1:]]}
        else:
            return {"value": getattr(_obj, _var_name)}
    f_get.__doc__ = f_get.__doc__.format(_name, _schema, _example_yaml_value)
    return f_get

def _make_f_put(schema_class: Type[KoboldSchema], obj, _var_name, _name, _schema, _example_yaml_value):
    def f_put(body: schema_class):
        """---
    put:
      summary: Set {} setting to specified value
      tags:
        - config
      requestBody:
        required: true
        content:
          application/json:
            schema: {}
            example:
              value: {}
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
        """
        _obj = {"koboldai_vars": koboldai_vars}[obj]
        if _var_name.startswith("@"):
            _obj[_var_name[1:]] = body.value
        else:
            setattr(_obj, _var_name, body.value)
        settingschanged()
        refresh_settings()
        return {}
    f_put.__doc__ = f_put.__doc__.format(_name, _schema, _example_yaml_value, api_validation_error_response=api_validation_error_response)
    return f_put

def create_config_endpoint(method="GET", schema="MemorySchema"):
    _name = globals()[schema].KoboldMeta.name
    _var_name = globals()[schema].KoboldMeta.var_name
    _route_name = globals()[schema].KoboldMeta.route_name
    _obj = globals()[schema].KoboldMeta.obj
    _example_yaml_value = globals()[schema].KoboldMeta.example_yaml_value
    _schema = schema
    f = _make_f_get(_obj, _var_name, _name, _schema, _example_yaml_value) if method == "GET" else _make_f_put(globals()[schema], _obj, _var_name, _name, _schema, _example_yaml_value)
    f.__name__ = f"{method.lower()}_config_{_name}"
    f = api_schema_wrap(f)
    for api in (api_v1,):
        f = api.route(f"/config/{_route_name}", methods=[method])(f)

class SoftPromptSettingSchema(KoboldSchema):
    value: str = fields.String(required=True, validate=[soft_prompt_validator, validate.Regexp(r"^[^/\\]*$")], metadata={"description": "Soft prompt name, or a string containing only whitespace for no soft prompt. If using the GET method and no soft prompt is loaded, this will always be the empty string."})

@api_v1.get("/config/soft_prompt")
@api_schema_wrap
def get_config_soft_prompt():
    """---
    get:
      summary: Retrieve the current soft prompt name
      tags:
        - config
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: SoftPromptSettingSchema
              example:
                value: ""
    """
    return {"value": koboldai_vars.spfilename.strip()}

class SoftPromptsListSchema(KoboldSchema):
    values: List[SoftPromptSettingSchema] = fields.List(fields.Nested(SoftPromptSettingSchema), required=True, metadata={"description": "Array of available softprompts."})

@api_v1.get("/config/soft_prompts_list")
@api_schema_wrap
def get_config_soft_prompts_list():
    """---
    get:
      summary: Retrieve all available softprompt filenames
      tags:
        - config
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: SoftPromptsListSchema
              example:
                values: []
    """
    splist = []
    for sp in fileops.getspfiles(koboldai_vars.modeldim):

        splist.append({"value":sp["filename"]})
    return {"values": splist}

@api_v1.put("/config/soft_prompt")
@api_schema_wrap
def put_config_soft_prompt(body: SoftPromptSettingSchema):
    """---
    put:
      summary: Set soft prompt by name
      tags:
        - config
      requestBody:
        required: true
        content:
          application/json:
            schema: SoftPromptSettingSchema
            example:
              value: ""
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
    """
    if koboldai_vars.allowsp:
        spRequest(body.value)
        settingschanged()
    return {}

class SamplerSeedSettingSchema(KoboldSchema):
    value: int = fields.Integer(validate=validate.Range(min=0, max=2**64 - 1), required=True)

@api_v1.get("/config/sampler_seed")
@api_schema_wrap
def get_config_sampler_seed():
    """---
    get:
      summary: Retrieve the current global sampler seed value
      tags:
        - config
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: SamplerSeedSettingSchema
              example:
                value: 3475097509890965500
    """
    return {"value": __import__("tpu_mtj_backend").get_rng_seed() if koboldai_vars.use_colab_tpu else __import__("torch").initial_seed()}

@api_v1.put("/config/sampler_seed")
@api_schema_wrap
def put_config_sampler_seed(body: SamplerSeedSettingSchema):
    """---
    put:
      summary: Set the global sampler seed value
      tags:
        - config
      requestBody:
        required: true
        content:
          application/json:
            schema: SamplerSeedSettingSchema
            example:
              value: 3475097509890965500
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
    """
    if koboldai_vars.use_colab_tpu:
        import tpu_mtj_backend
        tpu_mtj_backend.socketio = socketio
        tpu_mtj_backend.set_rng_seed(body.value)
    else:
        import torch
        torch.manual_seed(body.value)
    koboldai_vars.seed = body.value
    return {}

config_endpoint_schemas: List[Type[KoboldSchema]] = []

def config_endpoint_schema(c: Type[KoboldSchema]):
    config_endpoint_schemas.append(c)
    return c


@config_endpoint_schema
class MemorySettingSchema(KoboldSchema):
    value = fields.String(required=True)
    class KoboldMeta:
        route_name = "memory"
        obj = "koboldai_vars"
        var_name = "memory"
        name = "memory"
        example_yaml_value = "Memory"

@config_endpoint_schema
class AuthorsNoteSettingSchema(KoboldSchema):
    value = fields.String(required=True)
    class KoboldMeta:
        route_name = "authors_note"
        obj = "koboldai_vars"
        var_name = "authornote"
        name = "author's note"
        example_yaml_value = "''"

@config_endpoint_schema
class AuthorsNoteTemplateSettingSchema(KoboldSchema):
    value = fields.String(required=True)
    class KoboldMeta:
        route_name = "authors_note_template"
        obj = "koboldai_vars"
        var_name = "authornotetemplate"
        name = "author's note template"
        example_yaml_value = "\"[Author's note: <|>]\""

@config_endpoint_schema
class TopKSamplingSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=0), required=True)
    class KoboldMeta:
        route_name = "top_k"
        obj = "koboldai_vars"
        var_name = "top_k"
        name = "top-k sampling"
        example_yaml_value = "0"

@config_endpoint_schema
class TopASamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0), required=True)
    class KoboldMeta:
        route_name = "top_a"
        obj = "koboldai_vars"
        var_name = "top_a"
        name = "top-a sampling"
        example_yaml_value = "0.0"

@config_endpoint_schema
class TopPSamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0, max=1), required=True)
    class KoboldMeta:
        route_name = "top_p"
        obj = "koboldai_vars"
        var_name = "top_p"
        name = "top-p sampling"
        example_yaml_value = "0.9"

@config_endpoint_schema
class TailFreeSamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0, max=1), required=True)
    class KoboldMeta:
        route_name = "tfs"
        obj = "koboldai_vars"
        var_name = "tfs"
        name = "tail free sampling"
        example_yaml_value = "1.0"

@config_endpoint_schema
class TypicalSamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0, max=1), required=True)
    class KoboldMeta:
        route_name = "typical"
        obj = "koboldai_vars"
        var_name = "typical"
        name = "typical sampling"
        example_yaml_value = "1.0"

@config_endpoint_schema
class TemperatureSamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0, min_inclusive=False), required=True)
    class KoboldMeta:
        route_name = "temperature"
        obj = "koboldai_vars"
        var_name = "temp"
        name = "temperature"
        example_yaml_value = "0.5"

@config_endpoint_schema
class GensPerActionSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=0, max=5), required=True)
    class KoboldMeta:
        route_name = "n"
        obj = "koboldai_vars"
        var_name = "numseqs"
        name = "Gens Per Action"
        example_yaml_value = "1"

@config_endpoint_schema
class MaxLengthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=1, max=512), required=True)
    class KoboldMeta:
        route_name = "max_length"
        obj = "koboldai_vars"
        var_name = "genamt"
        name = "max length"
        example_yaml_value = "80"

@config_endpoint_schema
class WorldInfoDepthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=1, max=5), required=True)
    class KoboldMeta:
        route_name = "world_info_depth"
        obj = "koboldai_vars"
        var_name = "widepth"
        name = "world info depth"
        example_yaml_value = "3"

@config_endpoint_schema
class AuthorsNoteDepthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=1, max=5), required=True)
    class KoboldMeta:
        route_name = "authors_note_depth"
        obj = "koboldai_vars"
        var_name = "andepth"
        name = "author's note depth"
        example_yaml_value = "3"

@config_endpoint_schema
class MaxContextLengthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=512, max=2048), required=True)
    class KoboldMeta:
        route_name = "max_context_length"
        obj = "koboldai_vars"
        var_name = "max_length"
        name = "max context length"
        example_yaml_value = "2048"

@config_endpoint_schema
class TrimIncompleteSentencesSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "frmttriminc"
        obj = "koboldai_vars"
        var_name = "frmttriminc"
        name = "trim incomplete sentences (output formatting)"
        example_yaml_value = "false"

@config_endpoint_schema
class RemoveBlankLinesSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "frmtrmblln"
        obj = "koboldai_vars"
        var_name = "frmtrmblln"
        name = "remove blank lines (output formatting)"
        example_yaml_value = "false"

@config_endpoint_schema
class RemoveSpecialCharactersSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "frmtrmspch"
        obj = "koboldai_vars"
        var_name = "frmtrmspch"
        name = "remove special characters (output formatting)"
        example_yaml_value = "false"

@config_endpoint_schema
class SingleLineSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "singleline"
        obj = "koboldai_vars"
        var_name = "singleline"
        name = "single line (output formatting)"
        example_yaml_value = "false"

@config_endpoint_schema
class AddSentenceSpacingSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "frmtadsnsp"
        obj = "koboldai_vars"
        var_name = "frmtadsnsp"
        name = "add sentence spacing (input formatting)"
        example_yaml_value = "false"

@config_endpoint_schema
class SamplerOrderSettingSchema(KoboldSchema):
    value = fields.List(fields.Integer(), validate=[validate.Length(min=6), permutation_validator], required=True)
    class KoboldMeta:
        route_name = "sampler_order"
        obj = "koboldai_vars"
        var_name = "sampler_order"
        name = "sampler order"
        example_yaml_value = "[6, 0, 1, 2, 3, 4, 5]"

@config_endpoint_schema
class SamplerFullDeterminismSettingSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "sampler_full_determinism"
        obj = "koboldai_vars"
        var_name = "full_determinism"
        name = "sampler full determinism"
        example_yaml_value = "false"


for schema in config_endpoint_schemas:
    create_config_endpoint(schema=schema.__name__, method="GET")
    create_config_endpoint(schema=schema.__name__, method="PUT")


#==================================================================#
#  Final startup commands to launch Flask app
#==================================================================#
def startup(command_line_backend):
    socketio.start_background_task(load_model, *(command_line_backend,), **{'initial_load':True})
            
print("", end="", flush=True)

@logger.catch
def run():
    global app
    global tpu_mtj_backend

    command_line_backend = general_startup()
    # Start flask & SocketIO
    logger.init("Flask", status="Starting")
    if koboldai_vars.host:
        CORS(app)
    Session(app)
    logger.init_ok("Flask", status="OK")
    logger.init("Webserver", status="Starting")
    patch_transformers(use_tpu=koboldai_vars.use_colab_tpu)
    
    # Start Flask/SocketIO (Blocking, so this must be last method!)
    port = args.port if "port" in args and args.port is not None else 5000
    koboldai_vars.port = port

    # TODO: Top-level tpu_mtj_backend will be removed in modularity PR
    if koboldai_vars.use_colab_tpu:
        import tpu_mtj_backend
        tpu_mtj_backend.socketio = socketio
    
    if(koboldai_vars.host):
        if(args.localtunnel):
            public_ip = requests.get("https://ipv4.icanhazip.com/")
            logger.message(f"The Public IP of this machine is : {public_ip.text}")
            import subprocess, shutil
            localtunnel = subprocess.Popen([shutil.which('lt'), '-p', str(port), 'http'], stdout=subprocess.PIPE)
            attempts = 0
            while attempts < 10:
                try:
                    cloudflare = str(localtunnel.stdout.readline())
                    cloudflare = (re.search("(?P<url>https?:\/\/[^\s]+loca.lt)", cloudflare).group("url"))
                    koboldai_vars.cloudflare_link = cloudflare
                    break
                except:
                    attempts += 1
                    time.sleep(3)
                    continue
            if attempts == 10:
                print("LocalTunnel could not be created, falling back to cloudflare...")
                from flask_cloudflared import _run_cloudflared
                cloudflare = _run_cloudflared(port)
                koboldai_vars.cloudflare_link = cloudflare
        elif(args.ngrok):
            from flask_ngrok import _run_ngrok
            cloudflare = _run_ngrok()
            koboldai_vars.cloudflare_link = cloudflare
        elif(args.remote):
           from flask_cloudflared import _run_cloudflared
           cloudflare = _run_cloudflared(port)
           koboldai_vars.cloudflare_link = cloudflare
           
        startup(command_line_backend)
       
        if(args.localtunnel or args.ngrok or args.remote):
            with open('cloudflare.log', 'w') as cloudflarelog:
                cloudflarelog.write("KoboldAI is available at the following link : " + cloudflare)
                logger.init_ok("Webserver", status="OK")
                if not koboldai_vars.use_colab_tpu and args.model:
                    # If we're using a TPU our UI will freeze during the connection to the TPU. To prevent this from showing to the user we 
                    # delay the display of this message until after that step
                    logger.message(f"KoboldAI is still loading your model but available at the following link: {cloudflare}")
                    logger.message(f"KoboldAI is still loading your model but available at the following link for the Classic UI: {cloudflare}/classic")
                    logger.message(f"KoboldAI is still loading your model but available at the following link for KoboldAI Lite: {cloudflare}/lite")
                    logger.message(f"KoboldAI is still loading your model but available at the following link for the API: [Loading Model...]")
                    logger.message(f"While the model loads you can use the above links to begin setting up your session, for generations you must wait until after its done loading.")
        else:
            logger.init_ok("Webserver", status="OK")
            logger.message(f"Webserver has started, you can now connect to this machine at port: {port}")
        koboldai_vars.serverstarted = True
        if args.unblock:
            socketio.run(app, port=port, host='0.0.0.0')
        else:
            socketio.run(app, port=port)
    else:
        startup(command_line_backend)
        if args.unblock:
            if not args.no_ui:
                try:
                    import webbrowser
                    webbrowser.open_new('http://localhost:{0}'.format(port))
                except:
                    pass
            logger.init_ok("Webserver", status="OK")
            logger.message(f"Webserver started! You may now connect with a browser at http://127.0.0.1:{port}")
            koboldai_vars.serverstarted = True
            socketio.run(app, port=port, host='0.0.0.0')
        else:
            if not args.no_ui:
                try:
                    import webbrowser
                    webbrowser.open_new('http://localhost:{0}'.format(port))
                except:
                    pass
            logger.init_ok("Webserver", status="OK")
            logger.message(f"Webserver started! You may now connect with a browser at http://127.0.0.1:{port}")
            koboldai_vars.serverstarted = True
            socketio.run(app, port=port)
    logger.init("Webserver", status="Closed")
    
if __name__ == "__main__":
    run()
else:
    command_line_backend = general_startup()
    # Start flask & SocketIO
    logger.init("Flask", status="Starting")
    Session(app)
    logger.init_ok("Flask", status="OK")
    patch_transformers(use_tpu=koboldai_vars.use_colab_tpu)
    startup(command_line_backend)
    koboldai_settings.port = args.port if "port" in args and args.port is not None else 5000
    print("{0}\nServer started in WSGI mode!{1}".format(colors.GREEN, colors.END), flush=True)
    
