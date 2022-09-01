#!/usr/bin/python3
#==================================================================#
# KoboldAI
# Version: 1.18.1
# By: KoboldAIDev and the KoboldAI Community
#==================================================================#

# External packages
import eventlet
eventlet.monkey_patch(all=True, thread=False, os=False)
import os
os.system("")
__file__ = os.path.dirname(os.path.realpath(__file__))
os.chdir(__file__)
os.environ['EVENTLET_THREADPOOL_SIZE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from eventlet import tpool

import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)

from os import path, getcwd
import time
import re
import json
import datetime
import collections
import zipfile
import packaging
import packaging.version
import contextlib
import traceback
import threading
import markdown
import bleach
import itertools
import bisect
import functools
import traceback
import inspect
import warnings
import multiprocessing
import copy
from collections.abc import Iterable
from collections import OrderedDict
from typing import Any, Callable, TypeVar, Tuple, Union, Dict, Set, List, Optional, Type

import requests
import html
import argparse
import sys
import gc

import lupa
import importlib

# KoboldAI
import fileops
import gensettings
from utils import debounce
import utils
import koboldai_settings
import torch
from transformers import StoppingCriteria, GPT2TokenizerFast, GPT2LMHeadModel, GPTNeoForCausalLM, GPTNeoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, modeling_utils
from transformers import __version__ as transformers_version
import transformers
try:
    from transformers.models.opt.modeling_opt import OPTDecoder
except:
    pass
import transformers.generation_utils
global tpu_mtj_backend


if lupa.LUA_VERSION[:2] != (5, 4):
    print(f"Please install lupa==1.10. You have lupa {lupa.__version__}.", file=sys.stderr)

patch_causallm_patched = False

# Make sure tqdm progress bars display properly in Colab
from tqdm.auto import tqdm
old_init = tqdm.__init__
def new_init(self, *args, **kwargs):
    old_init(self, *args, **kwargs)
    if(self.ncols == 0 and kwargs.get("ncols") != 0):
        self.ncols = 99
tqdm.__init__ = new_init

# Fix some issues with the OPT tokenizer
from transformers import PreTrainedTokenizerBase
old_pretrainedtokenizerbase_from_pretrained = PreTrainedTokenizerBase.from_pretrained.__func__
@classmethod
def new_pretrainedtokenizerbase_from_pretrained(cls, *args, **kwargs):
    tokenizer = old_pretrainedtokenizerbase_from_pretrained(cls, *args, **kwargs)
    tokenizer._koboldai_header = tokenizer.encode("")
    tokenizer.add_bos_token = False
    tokenizer.add_prefix_space = False
    return tokenizer
PreTrainedTokenizerBase.from_pretrained = new_pretrainedtokenizerbase_from_pretrained

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

# AI models Menu
# This is a dict of lists where they key is the menu name, and the list is the menu items.
# Each item takes the 4 elements, 1: Text to display, 2: Model Name (koboldai_vars.model) or menu name (Key name for another menu),
# 3: the memory requirement for the model, 4: if the item is a menu or not (True/False)
model_menu = {
    'mainmenu': [
        ["Load a model from its directory", "NeoCustom", "", False],
        ["Load an old GPT-2 model (eg CloverEdition)", "GPT2Custom", "", False],
        ["Adventure Models", "adventurelist", "", True],
        ["Novel Models", "novellist", "", True],
        ["NSFW Models", "nsfwlist", "", True],
        ["Untuned OPT", "optlist", "", True],
        ["Untuned GPT-Neo/J", "gptneolist", "", True],
        ["Untuned Fairseq Dense", "fsdlist", "", True],
        ["Untuned XGLM", "xglmlist", "", True],
        ["Untuned GPT2", "gpt2list", "", True],
        ["Online Services", "apilist", "", True],
        ["Read Only (No AI)", "ReadOnly", "", False]
        ],
    'adventurelist': [
        ["Nerys FSD 13B V2 (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys-v2", "32GB", False],
        ["Nerys FSD 13B (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys", "32GB", False],
        ["Skein 6B", "KoboldAI/GPT-J-6B-Skein", "16GB", False],
        ["OPT Nerys 6B V2", "KoboldAI/OPT-6B-nerys-v2", "16GB", False],
        ["Adventure 6B", "KoboldAI/GPT-J-6B-Adventure", "16GB", False],
        ["Nerys FSD 2.7B (Hybrid)", "KoboldAI/fairseq-dense-2.7B-Nerys", "8GB", False],
        ["Adventure 2.7B", "KoboldAI/GPT-Neo-2.7B-AID", "8GB", False],
        ["Adventure 1.3B", "KoboldAI/GPT-Neo-1.3B-Adventure", "6GB", False],
        ["Adventure 125M (Mia)", "Merry/AID-Neo-125M", "2GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'novellist': [
        ["Nerys FSD 13B V2 (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys-v2", "32GB", False],
        ["Janeway FSD 13B", "KoboldAI/fairseq-dense-13B-Janeway", "32GB", False],
        ["Nerys FSD 13B (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys", "32GB", False],
        ["OPT Nerys 6B V2", "KoboldAI/OPT-6B-nerys-v2", "16GB", False],
        ["Janeway FSD 6.7B", "KoboldAI/fairseq-dense-6.7B-Janeway", "16GB", False],
        ["Janeway Neo 6B", "KoboldAI/GPT-J-6B-Janeway", "16GB", False],
        ["Janeway Neo 2.7B", "KoboldAI/GPT-Neo-2.7B-Janeway", "8GB", False],
        ["Janeway FSD 2.7B", "KoboldAI/fairseq-dense-2.7B-Janeway", "8GB", False],
        ["Nerys FSD 2.7B (Hybrid)", "KoboldAI/fairseq-dense-2.7B-Nerys", "8GB", False],
        ["Horni-LN 2.7B", "KoboldAI/GPT-Neo-2.7B-Horni-LN", "8GB", False],
        ["Picard 2.7B (Older Janeway)", "KoboldAI/GPT-Neo-2.7B-Picard", "8GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'nsfwlist': [
        ["Shinen FSD 13B (NSFW)", "KoboldAI/fairseq-dense-13B-Shinen", "32GB", False],
        ["Shinen FSD 6.7B (NSFW)", "KoboldAI/fairseq-dense-6.7B-Shinen", "16GB", False],
        ["Lit 6B (NSFW)", "hakurei/lit-6B", "16GB", False],
        ["Shinen 6B (NSFW)", "KoboldAI/GPT-J-6B-Shinen", "16GB", False],
        ["Horni 2.7B (NSFW)", "KoboldAI/GPT-Neo-2.7B-Horni", "8GB", False],
        ["Shinen 2.7B (NSFW)", "KoboldAI/GPT-Neo-2.7B-Shinen", "8GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'chatlist': [
        ["Convo 6B (Chatbot)", "hitomi-team/convo-6B", "16GB", False],
        ["C1 6B (Chatbot)", "hakurei/c1-6B", "16GB", False],
        ["C1 1.3B (Chatbot)", "iokru/c1-1.3B", "6GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'gptneolist': [
        ["GPT-J 6B", "EleutherAI/gpt-j-6B", "16GB", False],
        ["GPT-Neo 2.7B", "EleutherAI/gpt-neo-2.7B", "8GB", False],
        ["GPT-Neo 1.3B", "EleutherAI/gpt-neo-1.3B", "6GB", False],
        ["GPT-Neo 125M", "EleutherAI/gpt-neo-125M", "2GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'gpt2list': [
        ["GPT-2 XL", "gpt2-xl", "6GB", False],
        ["GPT-2 Large", "gpt2-large", "4GB", False],
        ["GPT-2 Med", "gpt2-medium", "2GB", False],
        ["GPT-2", "gpt2", "2GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'optlist': [
        ["OPT 66B", "facebook/opt-66b", "128GB", False],
        ["OPT 30B", "facebook/opt-30b", "64GB", False],
        ["OPT 13B", "facebook/opt-13b", "32GB", False],
        ["OPT 6.7B", "facebook/opt-6.7b", "16GB", False],
        ["OPT 2.7B", "facebook/opt-2.7b", "8GB", False],
        ["OPT 1.3B", "facebook/opt-1.3b", "4GB", False],
        ["OPT 350M", "facebook/opt-350m", "2GB", False],
        ["OPT 125M", "facebook/opt-125m", "1GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'fsdlist': [
        ["Fairseq Dense 13B", "KoboldAI/fairseq-dense-13B", "32GB", False],
        ["Fairseq Dense 6.7B", "KoboldAI/fairseq-dense-6.7B", "16GB", False],
        ["Fairseq Dense 2.7B", "KoboldAI/fairseq-dense-2.7B", "8GB", False],
        ["Fairseq Dense 1.3B", "KoboldAI/fairseq-dense-1.3B", "4GB", False],
        ["Fairseq Dense 355M", "KoboldAI/fairseq-dense-355M", "2GB", False],
        ["Fairseq Dense 125M", "KoboldAI/fairseq-dense-125M", "1GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'xglmlist': [
        ["XGLM 4.5B (Larger Dataset)", "facebook/xglm-4.5B", "12GB", False],
        ["XGLM 7.5B", "facebook/xglm-7.5B", "18GB", False],
        ["XGLM 2.9B", "facebook/xglm-2.9B", "10GB", False],
        ["XGLM 1.7B", "facebook/xglm-1.7B", "6GB", False],
        ["XGLM 564M", "facebook/xglm-564M", "4GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'apilist': [
        ["GooseAI API (requires API key)", "GooseAI", "", False],
        ["OpenAI API (requires API key)", "OAI", "", False],
        ["InferKit API (requires API key)", "InferKit", "", False],
        # ["KoboldAI Server API (Old Google Colab)", "Colab", "", False],
        ["KoboldAI API", "API", "", False],
        ["Return to Main Menu", "mainmenu", "", True],
    ]
    }


class Send_to_socketio(object):
    def write(self, bar):
        print(bar, end="")
        time.sleep(0.01)
        try:
            emit('from_server', {'cmd': 'model_load_status', 'data': bar.replace(" ", "&nbsp;")}, broadcast=True, room="UI_1")
        except:
            pass
        
    def flush(self):
        pass
                                
# Set logging level to reduce chatter from Flask
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Start flask & SocketIO
print("{0}Initializing Flask... {1}".format(colors.PURPLE, colors.END), end="")
from flask import Flask, render_template, Response, request, copy_current_request_context, send_from_directory, session, jsonify, abort, redirect, has_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_socketio import emit as _emit
from flask_session import Session
import secrets
from werkzeug.exceptions import HTTPException, NotFound, InternalServerError
import secrets
app = Flask(__name__, root_path=os.getcwd())
app.secret_key = secrets.token_hex()
app.config['SESSION_TYPE'] = 'filesystem'
app.config['TEMPLATES_AUTO_RELOAD'] = True
Session(app)
socketio = SocketIO(app, async_method="eventlet", manage_session=False, cors_allowed_origins='*')
#socketio = SocketIO(app, async_method="eventlet", logger=True, engineio_logger=True, manage_session=False)
koboldai_vars = koboldai_settings.koboldai_vars(session, socketio)

utils.koboldai_vars = koboldai_vars

print("{0}OK!{1}".format(colors.GREEN, colors.END))

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
    try:
        return _emit(*args, **kwargs)
    except AttributeError:
        return socketio.emit(*args, **kwargs)

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
    version="1.1.4",
    prefixes=["/api/v1", "/api/latest"],
    tags=tags,
)

#==================================================================#
# Function to get model selection at startup
#==================================================================#
def sendModelSelection(menu="mainmenu", folder="./models"):
    #If we send one of the manual load options, send back the list of model directories, otherwise send the menu
    if menu in ('NeoCustom', 'GPT2Custom'):
        (paths, breadcrumbs) = get_folder_path_info(folder)
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
        emit('show_model_menu', {'data': menu_list_ui_2, 'menu': menu, 'breadcrumbs': breadcrumbs, "showdelete": showdelete}, broadcast=False, room="UI_2")
    else:
        emit('from_server', {'cmd': 'show_model_menu', 'data': model_menu[menu], 'menu': menu, 'breadcrumbs': [], "showdelete": False}, broadcast=True, room="UI_1")
        emit('show_model_menu', {'data': model_menu[menu], 'menu': menu, 'breadcrumbs': [], "showdelete": False}, broadcast=False, room="UI_2")

def get_folder_path_info(base):
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
    if os.path.exists(path):
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(path)
        except:
            return False
        return True
    else:
        return False
    
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
    if(args.configname):
       modelname = args.configname
       return modelname
    if(koboldai_vars.model in ("NeoCustom", "GPT2Custom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        modelname = os.path.basename(os.path.normpath(koboldai_vars.custmodpth))
        return modelname
    else:
        modelname = koboldai_vars.model
        return modelname

#==================================================================#
# Breakmodel configuration functions
#==================================================================#
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
    if(utils.HAS_ACCELERATE):
        print(f"{row_color}{colors.YELLOW + '->' + row_color if -1 == selected else '  '} {' '*9} N/A  {sep_color}|{row_color}     {breakmodel.disk_blocks:3}  {sep_color}|{row_color}  (Disk cache){colors.END}")
    print(f"{row_color}   {' '*9} N/A  {sep_color}|{row_color}     {n_layers:3}  {sep_color}|{row_color}  (CPU){colors.END}")

def device_config(config):
    global breakmodel, generator
    import breakmodel
    n_layers = utils.num_layers(config)
    if args.cpu:
        breakmodel.gpu_blocks = [0]*n_layers
        return
    elif(args.breakmodel_gpulayers is not None or (utils.HAS_ACCELERATE and args.breakmodel_disklayers is not None)):
        try:
            if(not args.breakmodel_gpulayers):
                breakmodel.gpu_blocks = []
            else:
                breakmodel.gpu_blocks = list(map(int, args.breakmodel_gpulayers.split(',')))
            assert len(breakmodel.gpu_blocks) <= torch.cuda.device_count()
            s = n_layers
            for i in range(len(breakmodel.gpu_blocks)):
                if(breakmodel.gpu_blocks[i] <= -1):
                    breakmodel.gpu_blocks[i] = s
                    break
                else:
                    s -= breakmodel.gpu_blocks[i]
            assert sum(breakmodel.gpu_blocks) <= n_layers
            n_layers -= sum(breakmodel.gpu_blocks)
            if(args.breakmodel_disklayers is not None):
                assert args.breakmodel_disklayers <= n_layers
                breakmodel.disk_blocks = args.breakmodel_disklayers
                n_layers -= args.breakmodel_disklayers
        except:
            print("WARNING: --breakmodel_gpulayers is malformatted. Please use the --help option to see correct usage of --breakmodel_gpulayers. Defaulting to all layers on device 0.", file=sys.stderr)
            breakmodel.gpu_blocks = [n_layers]
            n_layers = 0
    elif(args.breakmodel_layers is not None):
        breakmodel.gpu_blocks = [n_layers - max(0, min(n_layers, args.breakmodel_layers))]
        n_layers -= sum(breakmodel.gpu_blocks)
    elif(args.model is not None):
        print("Breakmodel not specified, assuming GPU 0")
        breakmodel.gpu_blocks = [n_layers]
        n_layers = 0
    else:
        device_count = torch.cuda.device_count()
        if(device_count > 1):
            print(colors.CYAN + "\nPlease select one of your GPUs to be your primary GPU.")
            print("VRAM usage in your primary GPU will be higher than for your other ones.")
            print("It is recommended you make your fastest GPU your primary GPU.")
            device_list(n_layers)
            while(True):
                primaryselect = input("device ID> ")
                if(primaryselect.isnumeric() and 0 <= int(primaryselect) < device_count):
                    breakmodel.primary_device = int(primaryselect)
                    break
                else:
                    print(f"{colors.RED}Please enter an integer between 0 and {device_count-1}.{colors.END}")
        else:
            breakmodel.primary_device = 0

        print(colors.PURPLE + "\nIf you don't have enough VRAM to run the model on a single GPU")
        print("you can split the model between your CPU and your GPU(s), or between")
        print("multiple GPUs if you have more than one.")
        print("By putting more 'layers' on a GPU or CPU, more computations will be")
        print("done on that device and more VRAM or RAM will be required on that device")
        print("(roughly proportional to number of layers).")
        print("It should be noted that GPUs are orders of magnitude faster than the CPU.")
        print(f"This model has{colors.YELLOW} {n_layers} {colors.PURPLE}layers.{colors.END}\n")

        for i in range(device_count):
            device_list(n_layers, primary=breakmodel.primary_device, selected=i)
            print(f"{colors.CYAN}\nHow many of the remaining{colors.YELLOW} {n_layers} {colors.CYAN}layers would you like to put into device {i}?\nYou can also enter -1 to allocate all remaining layers to this device.{colors.END}\n")
            while(True):
                layerselect = input("# of layers> ")
                if((layerselect.isnumeric() or layerselect.strip() == '-1') and -1 <= int(layerselect) <= n_layers):
                    layerselect = int(layerselect)
                    layerselect = n_layers if layerselect == -1 else layerselect
                    breakmodel.gpu_blocks.append(layerselect)
                    n_layers -= layerselect
                    break
                else:
                    print(f"{colors.RED}Please enter an integer between -1 and {n_layers}.{colors.END}")
            if(n_layers == 0):
                break

        if(utils.HAS_ACCELERATE and n_layers > 0):
            device_list(n_layers, primary=breakmodel.primary_device, selected=-1)
            print(f"{colors.CYAN}\nHow many of the remaining{colors.YELLOW} {n_layers} {colors.CYAN}layers would you like to put into the disk cache?\nYou can also enter -1 to allocate all remaining layers to this device.{colors.END}\n")
            while(True):
                layerselect = input("# of layers> ")
                if((layerselect.isnumeric() or layerselect.strip() == '-1') and -1 <= int(layerselect) <= n_layers):
                    layerselect = int(layerselect)
                    layerselect = n_layers if layerselect == -1 else layerselect
                    breakmodel.disk_blocks = layerselect
                    n_layers -= layerselect
                    break
                else:
                    print(f"{colors.RED}Please enter an integer between -1 and {n_layers}.{colors.END}")

    print(colors.PURPLE + "\nFinal device configuration:")
    device_list(n_layers)

    # If all layers are on the same device, use the old GPU generation mode
    while(len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] == 0):
        breakmodel.gpu_blocks.pop()
    if(len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] in (-1, utils.num_layers(config))):
        koboldai_vars.breakmodel = False
        koboldai_vars.usegpu = True
        koboldai_vars.gpu_device = len(breakmodel.gpu_blocks)-1
        return

    if(not breakmodel.gpu_blocks):
        print("Nothing assigned to a GPU, reverting to CPU only mode")
        import breakmodel
        breakmodel.primary_device = "cpu"
        koboldai_vars.breakmodel = False
        koboldai_vars.usegpu = False
        return

def move_model_to_devices(model):
    global generator

    if(not utils.HAS_ACCELERATE and not koboldai_vars.breakmodel):
        if(koboldai_vars.usegpu):
            model = model.half().to(koboldai_vars.gpu_device)
        else:
            model = model.to('cpu').float()
        generator = model.generate
        return

    import breakmodel

    if(utils.HAS_ACCELERATE):
        import accelerate.utils
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

    model.half()
    gc.collect()

    if(hasattr(model, "transformer")):
        model.transformer.wte.to(breakmodel.primary_device)
        model.transformer.ln_f.to(breakmodel.primary_device)
        if(hasattr(model, 'lm_head')):
            model.lm_head.to(breakmodel.primary_device)
        if(hasattr(model.transformer, 'wpe')):
            model.transformer.wpe.to(breakmodel.primary_device)
    elif(not hasattr(model.model, "decoder")):
        model.model.embed_tokens.to(breakmodel.primary_device)
        model.model.layer_norm.to(breakmodel.primary_device)
        model.lm_head.to(breakmodel.primary_device)
        model.model.embed_positions.to(breakmodel.primary_device)
    else:
        model.model.decoder.embed_tokens.to(breakmodel.primary_device)
        if(model.model.decoder.project_in is not None):
            model.model.decoder.project_in.to(breakmodel.primary_device)
        if(model.model.decoder.project_out is not None):
            model.model.decoder.project_out.to(breakmodel.primary_device)
        model.model.decoder.embed_positions.to(breakmodel.primary_device)
    gc.collect()
    GPTNeoModel.forward = breakmodel.new_forward_neo
    if("GPTJModel" in globals()):
        GPTJModel.forward = breakmodel.new_forward_neo # type: ignore
    if("XGLMModel" in globals()):
        XGLMModel.forward = breakmodel.new_forward_xglm # type: ignore
    if("OPTDecoder" in globals()):
        OPTDecoder.forward = breakmodel.new_forward_opt # type: ignore
    generator = model.generate
    if(hasattr(model, "transformer")):
        breakmodel.move_hidden_layers(model.transformer)
    elif(not hasattr(model.model, "decoder")):
        breakmodel.move_hidden_layers(model.model, model.model.layers)
    else:
        breakmodel.move_hidden_layers(model.model.decoder, model.model.decoder.layers)

#==================================================================#
#  Allow the models to override some settings
#==================================================================#
def loadmodelsettings():
    try:
        js   = json.loads(str(model_config).partition(' ')[2])
    except Exception as e:
        try:
            try:
                js   = json.load(open(koboldai_vars.custmodpth + "/config.json", "r"))
            except Exception as e:
                js   = json.load(open(koboldai_vars.custmodpth.replace('/', '_') + "/config.json", "r"))            
        except Exception as e:
            js   = {}
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
        sampler_order = koboldai_vars.sampler_order
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
        koboldai_vars.welcome = js["welcome"]
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
    js = {}
    js["apikey"]      = koboldai_vars.apikey
    js["andepth"]     = koboldai_vars.andepth
    js["sampler_order"] = koboldai_vars.sampler_order
    js["temp"]        = koboldai_vars.temp
    js["top_p"]       = koboldai_vars.top_p
    js["top_k"]       = koboldai_vars.top_k
    js["tfs"]         = koboldai_vars.tfs
    js["typical"]     = koboldai_vars.typical
    js["top_a"]       = koboldai_vars.top_a
    js["rep_pen"]     = koboldai_vars.rep_pen
    js["rep_pen_slope"] = koboldai_vars.rep_pen_slope
    js["rep_pen_range"] = koboldai_vars.rep_pen_range
    js["genamt"]      = koboldai_vars.genamt
    js["max_length"]  = koboldai_vars.max_length
    js["ikgen"]       = koboldai_vars.ikgen
    js["formatoptns"] = {'frmttriminc': koboldai_vars.frmttriminc, 'frmtrmblln': koboldai_vars.frmtrmblln, 
                         'frmtrmspch': koboldai_vars.frmtrmspch, 'frmtadsnsp': koboldai_vars.frmtadsnsp, 'singleline': koboldai_vars.singleline}
    js["numseqs"]     = koboldai_vars.numseqs
    js["widepth"]     = koboldai_vars.widepth
    js["useprompt"]   = koboldai_vars.useprompt
    js["adventure"]   = koboldai_vars.adventure
    js["chatmode"]    = koboldai_vars.chatmode
    js["chatname"]    = koboldai_vars.chatname
    js["dynamicscan"] = koboldai_vars.dynamicscan
    js["nopromptgen"] = koboldai_vars.nopromptgen
    js["rngpersist"]  = koboldai_vars.rngpersist
    js["nogenmod"]    = koboldai_vars.nogenmod
    js["fulldeterminism"] = koboldai_vars.full_determinism
    js["autosave"]    = koboldai_vars.autosave
    js["welcome"]     = koboldai_vars.welcome
    js["output_streaming"] = koboldai_vars.output_streaming
    js["show_probs"] = koboldai_vars.show_probs

    if(koboldai_vars.seed_specified):
        js["seed"]    = koboldai_vars.seed
    else:
        js["seed"]    = None

    js["newlinemode"] = koboldai_vars.newlinemode

    js["antemplate"]  = koboldai_vars.setauthornotetemplate

    js["userscripts"] = koboldai_vars.userscripts
    js["corescript"]  = koboldai_vars.corescript
    js["softprompt"]  = koboldai_vars.spfilename

    # Write it
    if not os.path.exists('settings'):
        os.mkdir('settings')
    file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
    try:
        file.write(json.dumps(js, indent=3))
    finally:
        file.close()

#==================================================================#
#  Don't save settings unless 2 seconds have passed without modification
#==================================================================#
@debounce(2)
def settingschanged():
    print("{0}Saving settings!{1}".format(colors.GREEN, colors.END))
    savesettings()

#==================================================================#
#  Read settings from client file JSON and send to koboldai_vars
#==================================================================#

def loadsettings():
    if(path.exists("defaults/" + getmodelname().replace('/', '_') + ".settings")):
        # Read file contents into JSON object
        file = open("defaults/" + getmodelname().replace('/', '_') + ".settings", "r")
        js   = json.load(file)
        
        processsettings(js)
        file.close()
    if(path.exists("settings/" + getmodelname().replace('/', '_') + ".settings")):
        # Read file contents into JSON object
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        js   = json.load(file)
        
        processsettings(js)
        file.close()
        
def processsettings(js):
# Copy file contents to koboldai_vars
    if("apikey" in js):
        koboldai_vars.apikey     = js["apikey"]
    if("andepth" in js):
        koboldai_vars.andepth    = js["andepth"]
    if("sampler_order" in js):
        sampler_order = koboldai_vars.sampler_order
        if(len(sampler_order) < 7):
            sampler_order = [6] + sampler_order
        koboldai_vars.sampler_order = sampler_order
    if("temp" in js):
        koboldai_vars.temp       = js["temp"]
    if("top_p" in js):
        koboldai_vars.top_p      = js["top_p"]
    if("top_k" in js):
        koboldai_vars.top_k      = js["top_k"]
    if("tfs" in js):
        koboldai_vars.tfs        = js["tfs"]
    if("typical" in js):
        koboldai_vars.typical    = js["typical"]
    if("top_a" in js):
        koboldai_vars.top_a      = js["top_a"]
    if("rep_pen" in js):
        koboldai_vars.rep_pen    = js["rep_pen"]
    if("rep_pen_slope" in js):
        koboldai_vars.rep_pen_slope = js["rep_pen_slope"]
    if("rep_pen_range" in js):
        koboldai_vars.rep_pen_range = js["rep_pen_range"]
    if("genamt" in js):
        koboldai_vars.genamt     = js["genamt"]
    if("max_length" in js):
        koboldai_vars.max_length = js["max_length"]
    if("ikgen" in js):
        koboldai_vars.ikgen      = js["ikgen"]
    if("formatoptns" in js):
        for setting in ['frmttriminc', 'frmtrmblln', 'frmtrmspch', 'frmtadsnsp', 'singleline']:
            if setting in js["formatoptns"]:
                setattr(koboldai_vars, setting, js["formatoptns"][setting])
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

def check_for_sp_change():
    while(True):
        time.sleep(0.05)

        if(koboldai_vars.sp_changed):
            with app.app_context():
                emit('from_server', {'cmd': 'spstatitems', 'data': {koboldai_vars.spfilename: koboldai_vars.spmeta} if koboldai_vars.allowsp and len(koboldai_vars.spfilename) else {}}, namespace=None, broadcast=True, room="UI_1")
            koboldai_vars.sp_changed = False


socketio.start_background_task(check_for_sp_change)

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

    global np
    if 'np' not in globals():
        import numpy as np

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
    # Parsing Parameters
    parser = argparse.ArgumentParser(description="KoboldAI Server")
    parser.add_argument("--remote", action='store_true', help="Optimizes KoboldAI for Remote Play")
    parser.add_argument("--noaimenu", action='store_true', help="Disables the ability to select the AI")
    parser.add_argument("--ngrok", action='store_true', help="Optimizes KoboldAI for Remote Play using Ngrok")
    parser.add_argument("--localtunnel", action='store_true', help="Optimizes KoboldAI for Remote Play using Localtunnel")
    parser.add_argument("--host", action='store_true', help="Optimizes KoboldAI for Remote Play without using a proxy service")
    parser.add_argument("--port", type=int, help="Specify the port on which the application will be joinable")
    parser.add_argument("--aria2_port", type=int, help="Specify the port on which aria2's RPC interface will be open if aria2 is installed (defaults to 6799)")
    parser.add_argument("--model", help="Specify the Model Type to skip the Menu")
    parser.add_argument("--path", help="Specify the Path for local models (For model NeoCustom or GPT2Custom)")
    parser.add_argument("--revision", help="Specify the model revision for huggingface models (can be a git branch/tag name or a git commit hash)")
    parser.add_argument("--cpu", action='store_true', help="By default unattended launches are on the GPU use this option to force CPU usage.")
    parser.add_argument("--breakmodel", action='store_true', help=argparse.SUPPRESS)
    parser.add_argument("--breakmodel_layers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--breakmodel_gpulayers", type=str, help="If using a model that supports hybrid generation, this is a comma-separated list that specifies how many layers to put on each GPU device. For example to put 8 layers on device 0, 9 layers on device 1 and 11 layers on device 2, use --beakmodel_gpulayers 8,9,11")
    parser.add_argument("--breakmodel_disklayers", type=int, help="If using a model that supports hybrid generation, this is the number of layers to put in disk cache.")
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
    parser.add_argument("--customsettings", help="Preloads arguements from json file. You only need to provide the location of the json file. Use customsettings.json template file. It can be renamed if you wish so that you can store multiple configurations. Leave any settings you want as default as null. Any values you wish to set need to be in double quotation marks")
    parser.add_argument("--no_ui", action='store_true', default=False, help="Disables the GUI and Socket.IO server while leaving the API server running.")
    #args: argparse.Namespace = None
    if "pytest" in sys.modules and override_args is None:
        args = parser.parse_args([])
        return
    if override_args is not None:
        import shlex
        args = parser.parse_args(shlex.split(override_args))
    elif(os.environ.get("KOBOLDAI_ARGS") is not None):
        import shlex
        args = parser.parse_args(shlex.split(os.environ["KOBOLDAI_ARGS"]))
    else:
        args = parser.parse_args()

    temp = [x for x in vars(args)]
    for arg in temp:
        if arg == "path":
            if "model_path" in os.environ:
                setattr(args, arg, os.environ["model_path"])
        else:
            if arg in os.environ:
                if isinstance(getattr(args, arg), bool):
                    if os.environ[arg].lower() == "true":
                        setattr(args, arg, True)
                    else:
                        setattr(args, arg, False)
                else:
                    setattr(args, arg, os.environ[arg])
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

    koboldai_vars.model = args.model;
    koboldai_vars.revision = args.revision

    if args.colab:
        args.remote = True;
        args.override_rename = True;
        args.override_delete = True;
        args.nobreakmodel = True;
        args.quiet = True;
        args.lowmem = True;
        args.noaimenu = True;

    if args.quiet:
        koboldai_vars.quiet = True

    if args.nobreakmodel:
        koboldai_vars.nobreakmodel = True;

    if args.remote:
        koboldai_vars.host = True;

    if args.ngrok:
        koboldai_vars.host = True;

    if args.localtunnel:
        koboldai_vars.host = True;

    if args.host:
        koboldai_vars.host = True;

    if args.cpu:
        koboldai_vars.use_colab_tpu = False

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
            koboldai_vars.custmodpth = modpath
    elif args.model:
        print("Welcome to KoboldAI!\nYou have selected the following Model:", koboldai_vars.model)
        if args.path:
            print("You have selected the following path for your Model :", args.path)
            koboldai_vars.custmodpth = args.path;
            koboldai_vars.colaburl = args.path + "/request"; # Lets just use the same parameter to keep it simple
            
    #setup socketio relay queue
    koboldai_settings.queue = multiprocessing.Queue()
    socketio.start_background_task(socket_io_relay, koboldai_settings.queue, socketio)
        
#==================================================================#
# Load Model
#==================================================================# 

def tpumtjgetsofttokens():
    soft_tokens = None
    if(koboldai_vars.sp is None):
        global np
        if 'np' not in globals():
            import numpy as np
        tensor = np.zeros((1, tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"])), dtype=np.float32)
        rows = tensor.shape[0]
        padding_amount = tpu_mtj_backend.params["seq"] - (tpu_mtj_backend.params["seq"] % -tpu_mtj_backend.params["cores_per_replica"]) - rows
        tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
        tensor = tensor.reshape(
            tpu_mtj_backend.params["cores_per_replica"],
            -1,
            tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"]),
        )
        koboldai_vars.sp = tpu_mtj_backend.shard_xmap(tensor)
    soft_tokens = np.arange(
        tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"],
        tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"] + koboldai_vars.sp_length,
        dtype=np.uint32
    )
    return soft_tokens
 
def get_model_info(model, directory=""):
    # if the model is in the api list
    disk_blocks = 0
    key = False
    breakmodel = False
    gpu = False
    layer_count = None
    key_value = ""
    break_values = []
    url = False
    gpu_count = torch.cuda.device_count()
    gpu_names = []
    for i in range(gpu_count):
        gpu_names.append(torch.cuda.get_device_name(i))
    if model in ['Colab', 'API']:
        url = True
    elif model in [x[1] for x in model_menu['apilist']]:
        if path.exists("settings/{}.settings".format(model)):
            with open("settings/{}.settings".format(model), "r") as file:
                # Check if API key exists
                js = json.load(file)
                if("apikey" in js and js["apikey"] != ""):
                    # API key exists, grab it and close the file
                    key_value = js["apikey"]
                elif 'oaiapikey' in js and js['oaiapikey'] != "":
                    key_value = js["oaiapikey"]
                if model in ('GooseAI', 'OAI'): 
                    get_oai_models({'model': model, 'key': key_value})
        key = True
    elif model == 'ReadOnly':
        pass
    elif not utils.HAS_ACCELERATE and not torch.cuda.is_available():
        pass
    elif args.cpu:
        pass
    else:
        layer_count = get_layer_count(model, directory=directory)
        if layer_count is None:
            breakmodel = False
        else:
            breakmodel = True
            if model in ["NeoCustom", "GPT2Custom"]:
                filename = "settings/{}.breakmodel".format(os.path.basename(os.path.normpath(directory)))
            else:
                filename = "settings/{}.breakmodel".format(model.replace("/", "_"))
            if path.exists(filename):
                with open(filename, "r") as file:
                    data = file.read().split("\n")[:2]
                    if len(data) < 2:
                        data.append("0")
                    break_values, disk_blocks = data
                    break_values = break_values.split(",")
            else:
                break_values = [layer_count]
            if break_values == ['']:
                break_values = []
            break_values = [int(x) for x in break_values]
            break_values += [0] * (gpu_count - len(break_values))
    emit('from_server', {'cmd': 'selected_model_info', 'key_value': key_value, 'key':key, 
                         'gpu':gpu, 'layer_count':layer_count, 'breakmodel':breakmodel, 
                         'disk_break_value': disk_blocks, 'accelerate': utils.HAS_ACCELERATE,
                         'break_values': break_values, 'gpu_count': gpu_count,
                         'url': url, 'gpu_names': gpu_names}, broadcast=True, room="UI_1")
    emit('selected_model_info', {'key_value': key_value, 'key':key, 
                         'gpu':gpu, 'layer_count':layer_count, 'breakmodel':breakmodel, 
                         'disk_break_value': disk_blocks, 'disk_break': utils.HAS_ACCELERATE,
                         'break_values': break_values, 'gpu_count': gpu_count,
                         'url': url, 'gpu_names': gpu_names}, broadcast=False, room="UI_2")
    
    

def get_layer_count(model, directory=""):
    if(model not in ["InferKit", "Colab", "API", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ"]):
        if(model == "GPT2Custom"):
            with open(os.path.join(directory, "config.json"), "r") as f:
                model_config = json.load(f)
        # Get the model_type from the config or assume a model type if it isn't present
        else:
            if(directory):
                model = directory
            from transformers import AutoConfig
            if(os.path.isdir(model.replace('/', '_'))):
                model_config = AutoConfig.from_pretrained(model.replace('/', '_'), revision=koboldai_vars.revision, cache_dir="cache")
            elif(os.path.isdir("models/{}".format(model.replace('/', '_')))):
                model_config = AutoConfig.from_pretrained("models/{}".format(model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
            elif(os.path.isdir(directory)):
                model_config = AutoConfig.from_pretrained(directory, revision=koboldai_vars.revision, cache_dir="cache")
            elif(os.path.isdir(koboldai_vars.custmodpth.replace('/', '_'))):
                model_config = AutoConfig.from_pretrained(koboldai_vars.custmodpth.replace('/', '_'), revision=koboldai_vars.revision, cache_dir="cache")
            else:
                model_config = AutoConfig.from_pretrained(model, revision=koboldai_vars.revision, cache_dir="cache")
        return utils.num_layers(model_config)
    else:
        return None

@socketio.on('OAI_Key_Update')
def get_oai_models(data):
    key = data['key']
    model = data['model']
    koboldai_vars.oaiapikey = key
    if model == 'OAI':
        url = "https://api.openai.com/v1/engines"
    elif model == 'GooseAI':
        url = "https://api.goose.ai/v1/engines"
    else:
        return
        
    # Get list of models from OAI
    print("{0}Retrieving engine list...{1}".format(colors.PURPLE, colors.END), end="")
    req = requests.get(
        url, 
        headers = {
            'Authorization': 'Bearer '+key
            }
        )
    if(req.status_code == 200):
        engines = req.json()["data"]
        try:
            engines = [[en["id"], "{} ({})".format(en['id'], "Ready" if en["ready"] == True else "Not Ready")] for en in engines]
        except:
            print(engines)
            raise
        
        online_model = ""
        changed=False
        
        #Save the key
        if not path.exists("settings"):
            # If the client settings file doesn't exist, create it
            # Write API key to file
            os.makedirs('settings', exist_ok=True)
        if path.exists("settings/{}.settings".format(model)):
            with open("settings/{}.settings".format(model), "r") as file:
                js = json.load(file)
                if 'online_model' in js:
                    online_model = js['online_model']
                if "apikey" in js:
                    if js['apikey'] != key:
                        changed=True
        if changed:
            with open("settings/{}.settings".format(model), "w") as file:
                js["apikey"] = key
                file.write(json.dumps(js, indent=3))
            
        emit('from_server', {'cmd': 'oai_engines', 'data': engines, 'online_model': online_model}, broadcast=True, room="UI_1")
        emit('oai_engines', {'data': engines, 'online_model': online_model}, broadcast=False, room="UI_2")
    else:
        # Something went wrong, print the message and quit since we can't initialize an engine
        print("{0}ERROR!{1}".format(colors.RED, colors.END), room="UI_1")
        print(req.json())
        emit('from_server', {'cmd': 'errmsg', 'data': req.json()})


# Function to patch transformers to use our soft prompt
def patch_causallm(model):
    from torch.nn import Embedding
    if(getattr(Embedding, "_koboldai_patch_causallm_model", None)):
        Embedding._koboldai_patch_causallm_model = model
        return model
    old_embedding_call = Embedding.__call__
    def new_embedding_call(self, input_ids, *args, **kwargs):
        if(Embedding._koboldai_patch_causallm_model.get_input_embeddings() is not self):
            return old_embedding_call(self, input_ids, *args, **kwargs)
        assert input_ids is not None
        if(koboldai_vars.sp is not None):
            shifted_input_ids = input_ids - model.config.vocab_size
        input_ids.clamp_(max=model.config.vocab_size-1)
        inputs_embeds = old_embedding_call(self, input_ids, *args, **kwargs)
        if(koboldai_vars.sp is not None):
            koboldai_vars.sp = koboldai_vars.sp.to(inputs_embeds.dtype).to(inputs_embeds.device)
            inputs_embeds = torch.where(
                (shifted_input_ids >= 0)[..., None],
                koboldai_vars.sp[shifted_input_ids.clamp(min=0)],
                inputs_embeds,
            )
        return inputs_embeds
    Embedding.__call__ = new_embedding_call
    Embedding._koboldai_patch_causallm_model = model
    return model

def patch_transformers_download():
    global transformers
    import copy, requests, tqdm, time
    class Send_to_socketio(object):
        def write(self, bar):
            bar = bar.replace("\r", "").replace("\n", "")
            
            if bar != "":
                try:
                    print(bar, end="\r")
                    emit('from_server', {'cmd': 'model_load_status', 'data': bar.replace(" ", "&nbsp;")}, broadcast=True, room="UI_1")
                    eventlet.sleep(seconds=0)
                except:
                    pass
        def flush(self):
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
        koboldai_vars.fp32_model = False
        utils.num_shards = None
        utils.current_shard = 0
        utils.from_pretrained_model_name = pretrained_model_name_or_path
        utils.from_pretrained_index_filename = None
        utils.from_pretrained_kwargs = kwargs
        utils.bar = None
        if not args.no_aria2:
            utils.aria2_hook(pretrained_model_name_or_path, **kwargs)
        return old_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    PreTrainedModel.from_pretrained = new_from_pretrained
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


    # Patch transformers to use our custom logit warpers
    from transformers import LogitsProcessorList, LogitsWarper, LogitsProcessor, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor
    from warpers import AdvancedRepetitionPenaltyLogitsProcessor, TailFreeLogitsWarper, TypicalLogitsWarper, TopALogitsWarper

    def dynamic_processor_wrap(cls, field_name, var_name, cond=None):
        old_call = cls.__call__
        def new_call(self, *args, **kwargs):
            if(not isinstance(field_name, str) and isinstance(field_name, Iterable)):
                conds = []
                for f, v in zip(field_name, var_name):
                    conds.append(getattr(koboldai_vars, v))
                    setattr(self, f, conds[-1])
            else:
                conds = getattr(koboldai_vars, var_name)
                setattr(self, field_name, conds)
            assert len(args) == 2
            if(cond is None or cond(conds)):
                return old_call(self, *args, **kwargs)
            return args[1]
        cls.__call__ = new_call
    dynamic_processor_wrap(AdvancedRepetitionPenaltyLogitsProcessor, ("penalty", "penalty_slope", "penalty_range"), ("rep_pen", "rep_pen_slope", "rep_pen_range"), cond=lambda x: x[0] != 1.0)
    dynamic_processor_wrap(TopKLogitsWarper, "top_k", "top_k", cond=lambda x: x > 0)
    dynamic_processor_wrap(TopALogitsWarper, "top_a", "top_a", cond=lambda x: x > 0.0)
    dynamic_processor_wrap(TopPLogitsWarper, "top_p", "top_p", cond=lambda x: x < 1.0)
    dynamic_processor_wrap(TailFreeLogitsWarper, "tfs", "tfs", cond=lambda x: x < 1.0)
    dynamic_processor_wrap(TypicalLogitsWarper, "typical", "typical", cond=lambda x: x < 1.0)
    dynamic_processor_wrap(TemperatureLogitsWarper, "temperature", "temp", cond=lambda x: x != 1.0)

    class PhraseBiasLogitsProcessor(LogitsProcessor):
        def __init__(self):
            pass

        def _rindex(self, lst: List, target) -> Optional[int]:
            for index, item in enumerate(reversed(lst)):
                if item == target:
                    return len(lst) - index - 1
            return None

        def _find_intersection(self, big: List, small: List) -> int:
            # Find the intersection of the end of "big" and the beginning of
            # "small". A headache to think about, personally. Returns the index
            # into "small" where the two stop intersecting.
            start = self._rindex(big, small[0])

            # No progress into the token sequence, bias the first one.
            if not start:
                return 0

            for i in range(len(small)):
                try:
                    big_i = big[start + i]
                except IndexError:
                    return i

            # It's completed :^)
            return 0

        def _get_biased_tokens(self, input_ids: List) -> Dict:
            # TODO: Different "bias slopes"?

            ret = {}
            for phrase, _bias in koboldai_vars.biases.items():
                bias_score, completion_threshold = _bias
                # TODO: Cache these tokens, invalidate when model or bias is
                # changed.
                token_seq = tokenizer.encode(phrase)
                bias_index = self._find_intersection(input_ids, token_seq)

                # Ensure completion after completion_threshold tokens
                if bias_index + 1 > completion_threshold:
                    bias_score = 999

                token_to_bias = token_seq[bias_index]
                ret[token_to_bias] = bias_score
            return ret

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            assert scores.ndim == 2
            assert input_ids.ndim == 2

            scores_shape = scores.shape

            for batch in range(scores_shape[0]):
                for token, bias in self._get_biased_tokens(input_ids[batch]).items():
                    scores[batch][token] += bias

            return scores


    class LuaLogitsProcessor(LogitsProcessor):

        def __init__(self):
            pass

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            assert scores.ndim == 2
            assert input_ids.ndim == 2
            self.regeneration_required = False
            self.halt = False

            if(koboldai_vars.standalone):
                return scores

            scores_shape = scores.shape
            scores_list = scores.tolist()
            koboldai_vars.lua_koboldbridge.logits = koboldai_vars.lua_state.table()
            for r, row in enumerate(scores_list):
                koboldai_vars.lua_koboldbridge.logits[r+1] = koboldai_vars.lua_state.table(*row)
            koboldai_vars.lua_koboldbridge.vocab_size = scores_shape[-1]

            execute_genmod()

            scores = torch.tensor(
                tuple(tuple(row.values()) for row in koboldai_vars.lua_koboldbridge.logits.values()),
                device=scores.device,
                dtype=scores.dtype,
            )
            assert scores.shape == scores_shape

            return scores

    from torch.nn import functional as F

    class ProbabilityVisualizerLogitsProcessor(LogitsProcessor):
        def __init__(self):
            pass

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            assert scores.ndim == 2
            assert input_ids.ndim == 2

            if not koboldai_vars.show_probs:
                return scores

            for batch_index, batch in enumerate(scores):
                probs = F.softmax(batch, dim = -1).cpu().numpy()

                token_prob_info = []
                for token_id, score in sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:8]:
                    token_prob_info.append({
                        "tokenId": token_id,
                        "decoded": utils.decodenewlines(tokenizer.decode(token_id)),
                        "score": float(score),
                    })


                if len(scores) == 1:
                    koboldai_vars.actions.set_probabilities(token_prob_info)
                else:
                    koboldai_vars.actions.set_option_probabilities(token_prob_info, batch_index)

            return scores
    
    def new_get_logits_processor(*args, **kwargs) -> LogitsProcessorList:
        processors = new_get_logits_processor.old_get_logits_processor(*args, **kwargs)
        processors.insert(0, LuaLogitsProcessor())
        processors.append(ProbabilityVisualizerLogitsProcessor())
        return processors
    new_get_logits_processor.old_get_logits_processor = transformers.generation_utils.GenerationMixin._get_logits_processor
    transformers.generation_utils.GenerationMixin._get_logits_processor = new_get_logits_processor

    class KoboldLogitsWarperList(LogitsProcessorList):
        def __init__(self, beams: int = 1, **kwargs):
            self.__warper_list: List[LogitsWarper] = []
            self.__warper_list.append(TopKLogitsWarper(top_k=1, min_tokens_to_keep=1 + (beams > 1)))
            self.__warper_list.append(TopALogitsWarper(top_a=0.5, min_tokens_to_keep=1 + (beams > 1)))
            self.__warper_list.append(TopPLogitsWarper(top_p=0.5, min_tokens_to_keep=1 + (beams > 1)))
            self.__warper_list.append(TailFreeLogitsWarper(tfs=0.5, min_tokens_to_keep=1 + (beams > 1)))
            self.__warper_list.append(TypicalLogitsWarper(typical=0.5, min_tokens_to_keep=1 + (beams > 1)))
            self.__warper_list.append(TemperatureLogitsWarper(temperature=0.5))
            self.__warper_list.append(AdvancedRepetitionPenaltyLogitsProcessor())

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, *args, **kwargs):
            sampler_order = koboldai_vars.sampler_order[:]
            if len(sampler_order) < 7:  # Add repetition penalty at beginning if it's not present
                sampler_order = [6] + sampler_order
            for k in sampler_order:
                scores = self.__warper_list[k](input_ids, scores, *args, **kwargs)
            return scores

    def new_get_logits_warper(beams: int = 1,) -> LogitsProcessorList:
        return KoboldLogitsWarperList(beams=beams)
    
    def new_sample(self, *args, **kwargs):
        assert kwargs.pop("logits_warper", None) is not None
        kwargs["logits_warper"] = new_get_logits_warper(
            beams=1,
        )
        if(koboldai_vars.newlinemode == "s") or (koboldai_vars.newlinemode == "ns"):
            kwargs["eos_token_id"] = -1
            kwargs.setdefault("pad_token_id", 2)
        return new_sample.old_sample(self, *args, **kwargs)
    new_sample.old_sample = transformers.generation_utils.GenerationMixin.sample
    transformers.generation_utils.GenerationMixin.sample = new_sample


    # Allow bad words filter to ban <|endoftext|> token
    import transformers.generation_logits_process
    def new_init(self, bad_words_ids: List[List[int]], eos_token_id: int):
        return new_init.old_init(self, bad_words_ids, -1)
    new_init.old_init = transformers.generation_logits_process.NoBadWordsLogitsProcessor.__init__
    transformers.generation_logits_process.NoBadWordsLogitsProcessor.__init__ = new_init

    class TokenStreamer(StoppingCriteria):
        # A StoppingCriteria is used here because it seems to run after
        # everything has been evaluated score-wise. 
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs,
        ) -> bool:
            if (not koboldai_vars.output_streaming):
                return False

            #for batch, ids in enumerate(input_ids):
                #tokenizer_text = utils.decodenewlines(tokenizer.decode(ids[-1]))
                #koboldai_vars.actions.stream_token(tokenizer_text, batch=batch)
               
            if koboldai_vars.output_streaming:
                koboldai_vars.actions.stream_tokens([utils.decodenewlines(tokenizer.decode(x[-1])) for x in input_ids])
            #if len(input_ids) > 1:
            #    koboldai_vars.actions.clear_unused_options()
            #    koboldai_vars.actions.append_options([utils.decodenewlines(tokenizer.decode(x[-1])) for x in input_ids])
            #else:
            #    koboldai_vars.actions[koboldai_vars.actions.action_count+1] = utils.decodenewlines(tokenizer.decode(input_ids[0, -1]))
                
            return False

    # Sets up dynamic world info scanner
    class DynamicWorldInfoScanCriteria(StoppingCriteria):
        def __init__(
            self,
            tokenizer,
            excluded_world_info: List[Set],
        ):
            self.regeneration_required = False
            self.halt = False
            self.tokenizer = tokenizer
            self.excluded_world_info = excluded_world_info
        def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs,
        ) -> bool:
            koboldai_vars.generated_tkns += 1
            if(not koboldai_vars.standalone and koboldai_vars.lua_koboldbridge.generated_cols and koboldai_vars.generated_tkns != koboldai_vars.lua_koboldbridge.generated_cols):
                raise RuntimeError(f"Inconsistency detected between KoboldAI Python and Lua backends ({koboldai_vars.generated_tkns} != {koboldai_vars.lua_koboldbridge.generated_cols})")
            if(koboldai_vars.abort or koboldai_vars.generated_tkns >= koboldai_vars.genamt):
                self.regeneration_required = False
                self.halt = False
                return True
            if(koboldai_vars.standalone):
                return False

            assert input_ids.ndim == 2
            assert len(self.excluded_world_info) == input_ids.shape[0]
            self.regeneration_required = koboldai_vars.lua_koboldbridge.regeneration_required
            self.halt = not koboldai_vars.lua_koboldbridge.generating
            koboldai_vars.lua_koboldbridge.regeneration_required = False

            for i in range(koboldai_vars.numseqs):
                koboldai_vars.lua_koboldbridge.generated[i+1][koboldai_vars.generated_tkns] = int(input_ids[i, -1].item())

            if(not koboldai_vars.dynamicscan):
                return self.regeneration_required or self.halt
            tail = input_ids[..., -koboldai_vars.generated_tkns:]
            for i, t in enumerate(tail):
                decoded = utils.decodenewlines(tokenizer.decode(t))
                _, found = checkworldinfo(decoded, force_use_txt=True, actions=koboldai_vars._actions)
                found -= self.excluded_world_info[i]
                if(len(found) != 0):
                    self.regeneration_required = True
                    break
            return self.regeneration_required or self.halt
    old_get_stopping_criteria = transformers.generation_utils.GenerationMixin._get_stopping_criteria
    def new_get_stopping_criteria(self, *args, **kwargs):
        stopping_criteria = old_get_stopping_criteria(self, *args, **kwargs)
        global tokenizer
        self.kai_scanner = DynamicWorldInfoScanCriteria(
            tokenizer=tokenizer,
            excluded_world_info=self.kai_scanner_excluded_world_info,
        )
        token_streamer = TokenStreamer(tokenizer=tokenizer)

        stopping_criteria.insert(0, self.kai_scanner)
        token_streamer = TokenStreamer(tokenizer=tokenizer)
        stopping_criteria.insert(0, token_streamer)
        return stopping_criteria
    transformers.generation_utils.GenerationMixin._get_stopping_criteria = new_get_stopping_criteria

def reset_model_settings():
    koboldai_vars.socketio = socketio
    koboldai_vars.max_length  = 1024    # Maximum number of tokens to submit per action
    koboldai_vars.ikmax       = 3000    # Maximum number of characters to submit to InferKit
    koboldai_vars.genamt      = 80      # Amount of text for each action to generate
    koboldai_vars.ikgen       = 200     # Number of characters for InferKit to generate
    koboldai_vars.rep_pen     = 1.1     # Default generator repetition_penalty
    koboldai_vars.rep_pen_slope = 0.7   # Default generator repetition penalty slope
    koboldai_vars.rep_pen_range = 1024  # Default generator repetition penalty range
    koboldai_vars.temp        = 0.5     # Default generator temperature
    koboldai_vars.top_p       = 0.9     # Default generator top_p
    koboldai_vars.top_k       = 0       # Default generator top_k
    koboldai_vars.top_a       = 0.0     # Default generator top-a
    koboldai_vars.tfs         = 1.0     # Default generator tfs (tail-free sampling)
    koboldai_vars.typical     = 1.0     # Default generator typical sampling threshold
    koboldai_vars.numseqs     = 1       # Number of sequences to ask the generator to create
    koboldai_vars.generated_tkns = 0    # If using a backend that supports Lua generation modifiers, how many tokens have already been generated, otherwise 0
    koboldai_vars.badwordsids = []
    koboldai_vars.fp32_model  = False  # Whether or not the most recently loaded HF model was in fp32 format
    koboldai_vars.modeldim    = -1     # Embedding dimension of your model (e.g. it's 4096 for GPT-J-6B and 2560 for GPT-Neo-2.7B)
    koboldai_vars.sampler_order = [0, 1, 2, 3, 4, 5]
    koboldai_vars.newlinemode = "n"
    koboldai_vars.revision    = None

def load_model(use_gpu=True, gpu_layers=None, disk_layers=None, initial_load=False, online_model="", use_breakmodel_args=False, breakmodel_args_default_to_cpu=False):
    global model
    global generator
    global torch
    global model_config
    global GPT2TokenizerFast
    global tokenizer
    koboldai_vars.aibusy = True
    if(initial_load):
        use_breakmodel_args = True
    reset_model_settings()
    if not utils.HAS_ACCELERATE:
        disk_layers = None
    koboldai_vars.reset_model()
    koboldai_vars.noai = False
    if not use_breakmodel_args:
        set_aibusy(True)
        if koboldai_vars.model != 'ReadOnly':
            emit('from_server', {'cmd': 'model_load_status', 'data': "Loading {}".format(koboldai_vars.model)}, broadcast=True)
            #Have to add a sleep so the server will send the emit for some reason
            time.sleep(0.1)
    if gpu_layers is not None:
        args.breakmodel_gpulayers = gpu_layers
    elif use_breakmodel_args:
        gpu_layers = args.breakmodel_gpulayers
    if breakmodel_args_default_to_cpu and gpu_layers is None:
        gpu_layers = args.breakmodel_gpulayers = []
    if disk_layers is not None:
        args.breakmodel_disklayers = int(disk_layers)
    elif use_breakmodel_args:
        disk_layers = args.breakmodel_disklayers
    if breakmodel_args_default_to_cpu and disk_layers is None:
        disk_layers = args.breakmodel_disklayers = 0
    
    #We need to wipe out the existing model and refresh the cuda cache
    model = None
    generator = None
    model_config = None
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
    
    #Let's set the GooseAI or OpenAI server URLs if that's applicable
    if online_model != "":
        if path.exists("settings/{}.settings".format(koboldai_vars.model)):
            changed=False
            with open("settings/{}.settings".format(koboldai_vars.model), "r") as file:
                # Check if API key exists
                js = json.load(file)
                if 'online_model' in js:
                    if js['online_model'] != online_model:
                        changed=True
                        js['online_model'] = online_model
                else:
                    changed=True
                    js['online_model'] = online_model
            if changed:
                with open("settings/{}.settings".format(koboldai_vars.model), "w") as file:
                    file.write(json.dumps(js, indent=3))
        # Swap OAI Server if GooseAI was selected
        if(koboldai_vars.model == "GooseAI"):
            koboldai_vars.oaiengines = "https://api.goose.ai/v1/engines"
            koboldai_vars.model = "OAI"
            args.configname = "GooseAI" + "/" + online_model
        else:
            args.configname = koboldai_vars.model + "/" + online_model
        koboldai_vars.oaiurl = koboldai_vars.oaiengines + "/{0}/completions".format(online_model)
    
    
    # If transformers model was selected & GPU available, ask to use CPU or GPU
    if(koboldai_vars.model not in ["InferKit", "Colab", "API", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        koboldai_vars.allowsp = True
        # Test for GPU support
        
        # Make model path the same as the model name to make this consistent with the other loading method if it isn't a known model type
        # This code is not just a workaround for below, it is also used to make the behavior consistent with other loading methods - Henk717
        if(not koboldai_vars.model in ["NeoCustom", "GPT2Custom"]):
            koboldai_vars.custmodpth = koboldai_vars.model
        elif(koboldai_vars.model == "NeoCustom"):
            koboldai_vars.model = os.path.basename(os.path.normpath(koboldai_vars.custmodpth))

        # Get the model_type from the config or assume a model type if it isn't present
        from transformers import AutoConfig
        if(os.path.isdir(koboldai_vars.custmodpth.replace('/', '_'))):
            try:
                model_config = AutoConfig.from_pretrained(koboldai_vars.custmodpth.replace('/', '_'), revision=koboldai_vars.revision, cache_dir="cache")
                koboldai_vars.model_type = model_config.model_type
            except ValueError as e:
                koboldai_vars.model_type = "not_found"
        elif(os.path.isdir("models/{}".format(koboldai_vars.custmodpth.replace('/', '_')))):
            try:
                model_config = AutoConfig.from_pretrained("models/{}".format(koboldai_vars.custmodpth.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
                koboldai_vars.model_type = model_config.model_type
            except ValueError as e:
                koboldai_vars.model_type = "not_found"
        else:
            try:
                model_config = AutoConfig.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                koboldai_vars.model_type = model_config.model_type
            except ValueError as e:
                koboldai_vars.model_type = "not_found"
        if(koboldai_vars.model_type == "not_found" and koboldai_vars.model == "NeoCustom"):
            koboldai_vars.model_type = "gpt_neo"
        elif(koboldai_vars.model_type == "not_found" and koboldai_vars.model == "GPT2Custom"):
            koboldai_vars.model_type = "gpt2"
        elif(koboldai_vars.model_type == "not_found"):
            print("WARNING: No model type detected, assuming Neo (If this is a GPT2 model use the other menu option or --model GPT2Custom)")
            koboldai_vars.model_type = "gpt_neo"

    if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ["InferKit", "Colab", "API", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        loadmodelsettings()
        loadsettings()
        print(2)
        print("{0}Looking for GPU support...{1}".format(colors.PURPLE, colors.END), end="")
        koboldai_vars.hascuda = torch.cuda.is_available()
        koboldai_vars.bmsupported = (utils.HAS_ACCELERATE or koboldai_vars.model_type in ("gpt_neo", "gptj", "xglm", "opt")) and not koboldai_vars.nobreakmodel
        if(args.breakmodel is not None and args.breakmodel):
            print("WARNING: --breakmodel is no longer supported. Breakmodel mode is now automatically enabled when --breakmodel_gpulayers is used (see --help for details).", file=sys.stderr)
        if(args.breakmodel_layers is not None):
            print("WARNING: --breakmodel_layers is deprecated. Use --breakmodel_gpulayers instead (see --help for details).", file=sys.stderr)
        if(args.model and koboldai_vars.bmsupported and not args.breakmodel_gpulayers and not args.breakmodel_layers and (not utils.HAS_ACCELERATE or not args.breakmodel_disklayers)):
            print("WARNING: Model launched without the --breakmodel_gpulayers argument, defaulting to GPU only mode.", file=sys.stderr)
            koboldai_vars.bmsupported = False
        if(not koboldai_vars.bmsupported and (args.breakmodel_gpulayers is not None or args.breakmodel_layers is not None or args.breakmodel_disklayers is not None)):
            print("WARNING: This model does not support hybrid generation. --breakmodel_gpulayers will be ignored.", file=sys.stderr)
        if(koboldai_vars.hascuda):
            print("{0}FOUND!{1}".format(colors.GREEN, colors.END))
        else:
            print("{0}NOT FOUND!{1}".format(colors.YELLOW, colors.END))
        
        if args.cpu:
            koboldai_vars.usegpu = False
            gpu_layers = None
            disk_layers = None
            koboldai_vars.breakmodel = False
        elif koboldai_vars.hascuda:
            if(koboldai_vars.bmsupported):
                koboldai_vars.usegpu = False
                koboldai_vars.breakmodel = True
            else:
                koboldai_vars.breakmodel = False
                koboldai_vars.usegpu = True


    # Ask for API key if InferKit was selected
    if(koboldai_vars.model == "InferKit"):
        koboldai_vars.apikey = koboldai_vars.oaiapikey
                    
    # Swap OAI Server if GooseAI was selected
    if(koboldai_vars.model == "GooseAI"):
        koboldai_vars.oaiengines = "https://api.goose.ai/v1/engines"
        koboldai_vars.model = "OAI"
        args.configname = "GooseAI"

    # Ask for API key if OpenAI was selected
    if(koboldai_vars.model == "OAI"):
        if not args.configname:
            args.configname = "OAI"
        
    if(koboldai_vars.model == "ReadOnly"):
        koboldai_vars.noai = True

    # Start transformers and create pipeline
    if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ["InferKit", "Colab", "API", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        if(not koboldai_vars.noai):
            print("{0}Initializing transformers, please wait...{1}".format(colors.PURPLE, colors.END))
            for m in ("GPTJModel", "XGLMModel"):
                try:
                    globals()[m] = getattr(__import__("transformers"), m)
                except:
                    pass

            # Lazy loader
            import torch_lazy_loader
            def get_lazy_load_callback(n_layers, convert_to_float16=True):
                if not koboldai_vars.lazy_load:
                    return

                from tqdm.auto import tqdm

                global breakmodel
                import breakmodel

                if utils.HAS_ACCELERATE:
                    import accelerate.utils

                if args.breakmodel_disklayers is not None:
                    breakmodel.disk_blocks = args.breakmodel_disklayers

                disk_blocks = breakmodel.disk_blocks
                gpu_blocks = breakmodel.gpu_blocks
                ram_blocks = ram_blocks = n_layers - sum(gpu_blocks)
                cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))

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
                            device_map[key] = koboldai_vars.gpu_device if koboldai_vars.hascuda and koboldai_vars.usegpu else "cpu" if not koboldai_vars.hascuda or not koboldai_vars.breakmodel else breakmodel.primary_device
                        else:
                            layer = int(max((n for n in utils.layers_module_names if original_key.startswith(n)), key=len).rsplit(".", 1)[1])
                            device = koboldai_vars.gpu_device if koboldai_vars.hascuda and koboldai_vars.usegpu else "disk" if layer < disk_blocks and layer < ram_blocks else "cpu" if not koboldai_vars.hascuda or not koboldai_vars.breakmodel else "shared" if layer < ram_blocks else bisect.bisect_right(cumulative_gpu_blocks, layer - ram_blocks)
                            device_map[key] = device

                    if utils.num_shards is None or utils.current_shard == 0:
                        utils.offload_index = {}
                        if utils.HAS_ACCELERATE:
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
                        koboldai_vars.total_layers = num_tensors
                        koboldai_vars.loaded_layers = 0
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
                                if model_dict[key].dtype is torch.float32:
                                    koboldai_vars.fp32_model = True
                                if convert_to_float16 and breakmodel.primary_device != "cpu" and koboldai_vars.hascuda and (koboldai_vars.breakmodel or koboldai_vars.usegpu) and model_dict[key].dtype is torch.float32:
                                    model_dict[key] = model_dict[key].to(torch.float16)
                                if breakmodel.primary_device == "cpu" or (not koboldai_vars.usegpu and not koboldai_vars.breakmodel and model_dict[key].dtype is torch.float16):
                                    model_dict[key] = model_dict[key].to(torch.float32)
                                if device == "shared":
                                    model_dict[key] = model_dict[key].to("cpu").detach_()
                                    if able_to_pin_layers and utils.HAS_ACCELERATE:
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
                                koboldai_vars.loaded_layers += 1
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
                return lazy_load_callback


            def get_hidden_size_from_model(model):
                try:
                    return int(model.model.decoder.project_in.in_features)
                except:
                    try:
                        return int(model.model.decoder.embed_tokens.out_features)
                    except:
                        try:
                            return int(model.transformer.hidden_size)
                        except:
                            try:
                                return int(model.transformer.embed_dim)
                            except:
                                return int(model.lm_head.in_features)
            
            def maybe_low_cpu_mem_usage() -> Dict[str, Any]:
                if(packaging.version.parse(transformers_version) < packaging.version.parse("4.11.0")):
                    print(f"\nWARNING:  Please upgrade to transformers 4.11.0 for lower RAM usage.  You have transformers {transformers_version}.", file=sys.stderr)
                    return {}
                return {"low_cpu_mem_usage": True}
            
            @contextlib.contextmanager
            def maybe_use_float16(always_use=False):
                if(always_use or (koboldai_vars.hascuda and args.lowmem and (koboldai_vars.usegpu or koboldai_vars.breakmodel))):
                    original_dtype = torch.get_default_dtype()
                    torch.set_default_dtype(torch.float16)
                    yield True
                    torch.set_default_dtype(original_dtype)
                else:
                    yield False

            # If custom GPT2 model was chosen
            if(koboldai_vars.model == "GPT2Custom"):
                koboldai_vars.lazy_load = False
                model_config = open(koboldai_vars.custmodpth + "/config.json", "r")
                js   = json.load(model_config)
                with(maybe_use_float16()):
                    try:
                        model = GPT2LMHeadModel.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                    except Exception as e:
                        if("out of memory" in traceback.format_exc().lower()):
                            raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                        raise e
                tokenizer = GPT2TokenizerFast.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                koboldai_vars.modeldim = get_hidden_size_from_model(model)
                # Is CUDA available? If so, use GPU, otherwise fall back to CPU
                if(koboldai_vars.hascuda and koboldai_vars.usegpu):
                    model = model.half().to(koboldai_vars.gpu_device)
                    generator = model.generate
                else:
                    model = model.to('cpu').float()
                    generator = model.generate
                patch_causallm(model)
            # Use the Generic implementation
            else:
                lowmem = maybe_low_cpu_mem_usage()
                # We must disable low_cpu_mem_usage (by setting lowmem to {}) if
                # using a GPT-2 model because GPT-2 is not compatible with this
                # feature yet
                if(koboldai_vars.model_type == "gpt2"):
                    lowmem = {}
                    koboldai_vars.lazy_load = False  # Also, lazy loader doesn't support GPT-2 models
                
                # If we're using torch_lazy_loader, we need to get breakmodel config
                # early so that it knows where to load the individual model tensors
                if (utils.HAS_ACCELERATE or koboldai_vars.lazy_load and koboldai_vars.hascuda and koboldai_vars.breakmodel) and not koboldai_vars.nobreakmodel:
                    device_config(model_config)

                # Download model from Huggingface if it does not exist, otherwise load locally
                
                #If we specify a model and it's in the root directory, we need to move it to the models directory (legacy folder structure to new)
                if os.path.isdir(koboldai_vars.model.replace('/', '_')):
                    import shutil
                    shutil.move(koboldai_vars.model.replace('/', '_'), "models/{}".format(koboldai_vars.model.replace('/', '_')))
                print("\n", flush=True)
                if(koboldai_vars.lazy_load):  # If we're using lazy loader, we need to figure out what the model's hidden layers are called
                    with torch_lazy_loader.use_lazy_torch_load(dematerialized_modules=True, use_accelerate_init_empty_weights=True):
                        try:
                            metamodel = AutoModelForCausalLM.from_config(model_config)
                        except Exception as e:
                            metamodel = GPTNeoForCausalLM.from_config(model_config)
                        utils.layers_module_names = utils.get_layers_module_names(metamodel)
                        utils.module_names = list(metamodel.state_dict().keys())
                        utils.named_buffers = list(metamodel.named_buffers(recurse=True))
                with maybe_use_float16(), torch_lazy_loader.use_lazy_torch_load(enable=koboldai_vars.lazy_load, callback=get_lazy_load_callback(utils.num_layers(model_config)) if koboldai_vars.lazy_load else None, dematerialized_modules=True):
                    if(koboldai_vars.lazy_load):  # torch_lazy_loader.py and low_cpu_mem_usage can't be used at the same time
                        lowmem = {}
                    if(os.path.isdir(koboldai_vars.custmodpth)):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                        except Exception as e:
                            pass
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = GPT2TokenizerFast.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache")
                            except Exception as e:
                                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained(koboldai_vars.custmodpth, revision=koboldai_vars.revision, cache_dir="cache", **lowmem)
                    elif(os.path.isdir("models/{}".format(koboldai_vars.model.replace('/', '_')))):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
                        except Exception as e:
                            pass
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = GPT2TokenizerFast.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
                            except Exception as e:
                                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache", **lowmem)
                    else:
                        old_rebuild_tensor = torch._utils._rebuild_tensor
                        def new_rebuild_tensor(storage: Union[torch_lazy_loader.LazyTensor, torch.Storage], storage_offset, shape, stride):
                            if(not isinstance(storage, torch_lazy_loader.LazyTensor)):
                                dtype = storage.dtype
                            else:
                                dtype = storage.storage_type.dtype
                                if(not isinstance(dtype, torch.dtype)):
                                    dtype = storage.storage_type(0).dtype
                            if(dtype is torch.float32 and len(shape) >= 2):
                                koboldai_vars.fp32_model = True
                            return old_rebuild_tensor(storage, storage_offset, shape, stride)
                        torch._utils._rebuild_tensor = new_rebuild_tensor

                        try:
                            tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache")
                        except Exception as e:
                            pass
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = GPT2TokenizerFast.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache")
                            except Exception as e:
                                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained(koboldai_vars.model, revision=koboldai_vars.revision, cache_dir="cache", **lowmem)

                        torch._utils._rebuild_tensor = old_rebuild_tensor

                        if not args.colab or args.savemodel:
                            import shutil
                            tokenizer.save_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')))
                            if(koboldai_vars.fp32_model):  # Use save_pretrained to convert fp32 models to fp16
                                model = model.half()
                                model.save_pretrained("models/{}".format(koboldai_vars.model.replace('/', '_')), max_shard_size="500MiB")
                            else:  # For fp16 models, we can just copy the model files directly
                                import transformers.configuration_utils
                                import transformers.modeling_utils
                                import transformers.file_utils
                                # Save the config.json
                                shutil.move(transformers.file_utils.get_from_cache(transformers.file_utils.hf_bucket_url(koboldai_vars.model, transformers.configuration_utils.CONFIG_NAME, revision=koboldai_vars.revision), cache_dir="cache", local_files_only=True), os.path.join("models/{}".format(koboldai_vars.model.replace('/', '_')), transformers.configuration_utils.CONFIG_NAME))
                                if(utils.num_shards is None):
                                    # Save the pytorch_model.bin of an unsharded model
                                    shutil.move(transformers.file_utils.get_from_cache(transformers.file_utils.hf_bucket_url(koboldai_vars.model, transformers.modeling_utils.WEIGHTS_NAME, revision=koboldai_vars.revision), cache_dir="cache", local_files_only=True), os.path.join("models/{}".format(koboldai_vars.model.replace('/', '_')), transformers.modeling_utils.WEIGHTS_NAME))
                                else:
                                    with open(utils.from_pretrained_index_filename) as f:
                                        map_data = json.load(f)
                                    filenames = set(map_data["weight_map"].values())
                                    # Save the pytorch_model.bin.index.json of a sharded model
                                    shutil.move(utils.from_pretrained_index_filename, os.path.join("models/{}".format(koboldai_vars.model.replace('/', '_')), transformers.modeling_utils.WEIGHTS_INDEX_NAME))
                                    # Then save the pytorch_model-#####-of-#####.bin files
                                    for filename in filenames:
                                        shutil.move(transformers.file_utils.get_from_cache(transformers.file_utils.hf_bucket_url(koboldai_vars.model, filename, revision=koboldai_vars.revision), cache_dir="cache", local_files_only=True), os.path.join("models/{}".format(koboldai_vars.model.replace('/', '_')), filename))
                            shutil.rmtree("cache/")

                if(koboldai_vars.badwordsids is koboldai_settings.badwordsids_default and koboldai_vars.model_type not in ("gpt2", "gpt_neo", "gptj")):
                    koboldai_vars.badwordsids = [[v] for k, v in tokenizer.get_vocab().items() if any(c in str(k) for c in "<>[]") if koboldai_vars.newlinemode != "s" or str(k) != "</s>"]

                patch_causallm(model)

                if(koboldai_vars.hascuda):
                    if(koboldai_vars.usegpu):
                        koboldai_vars.modeldim = get_hidden_size_from_model(model)
                        model = model.half().to(koboldai_vars.gpu_device)
                        generator = model.generate
                    elif(koboldai_vars.breakmodel):  # Use both RAM and VRAM (breakmodel)
                        koboldai_vars.modeldim = get_hidden_size_from_model(model)
                        if(not koboldai_vars.lazy_load):
                            device_config(model.config)
                        move_model_to_devices(model)
                    elif(utils.HAS_ACCELERATE and __import__("breakmodel").disk_blocks > 0):
                        move_model_to_devices(model)
                        koboldai_vars.modeldim = get_hidden_size_from_model(model)
                        generator = model.generate
                    else:
                        model = model.to('cpu').float()
                        koboldai_vars.modeldim = get_hidden_size_from_model(model)
                        generator = model.generate
                elif(utils.HAS_ACCELERATE and __import__("breakmodel").disk_blocks > 0):
                    move_model_to_devices(model)
                    koboldai_vars.modeldim = get_hidden_size_from_model(model)
                    generator = model.generate
                else:
                    model.to('cpu').float()
                    koboldai_vars.modeldim = get_hidden_size_from_model(model)
                    generator = model.generate
            
            # Suppress Author's Note by flagging square brackets (Old implementation)
            #vocab         = tokenizer.get_vocab()
            #vocab_keys    = vocab.keys()
            #koboldai_vars.badwords = gettokenids("[")
            #for key in koboldai_vars.badwords:
            #    koboldai_vars.badwordsids.append([vocab[key]])
            
            print("{0}OK! {1} pipeline created!{2}".format(colors.GREEN, koboldai_vars.model, colors.END))
        
        else:
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
    else:
        from transformers import PreTrainedModel
        from transformers import modeling_utils
        old_from_pretrained = PreTrainedModel.from_pretrained.__func__
        @classmethod
        def new_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            koboldai_vars.fp32_model = False
            utils.num_shards = None
            utils.current_shard = 0
            utils.from_pretrained_model_name = pretrained_model_name_or_path
            utils.from_pretrained_index_filename = None
            utils.from_pretrained_kwargs = kwargs
            utils.bar = None
            if not args.no_aria2:
                utils.aria2_hook(pretrained_model_name_or_path, **kwargs)
            return old_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
        PreTrainedModel.from_pretrained = new_from_pretrained
        if(hasattr(modeling_utils, "get_checkpoint_shard_files")):
            old_get_checkpoint_shard_files = modeling_utils.get_checkpoint_shard_files
            def new_get_checkpoint_shard_files(pretrained_model_name_or_path, index_filename, *args, **kwargs):
                utils.num_shards = utils.get_num_shards(index_filename)
                utils.from_pretrained_index_filename = index_filename
                return old_get_checkpoint_shard_files(pretrained_model_name_or_path, index_filename, *args, **kwargs)
            modeling_utils.get_checkpoint_shard_files = new_get_checkpoint_shard_files


        def tpumtjgenerate_warper_callback(scores) -> "np.array":
            scores_shape = scores.shape
            scores_list = scores.tolist()
            koboldai_vars.lua_koboldbridge.logits = koboldai_vars.lua_state.table()
            for r, row in enumerate(scores_list):
                koboldai_vars.lua_koboldbridge.logits[r+1] = koboldai_vars.lua_state.table(*row)
            koboldai_vars.lua_koboldbridge.vocab_size = scores_shape[-1]

            execute_genmod()

            scores = np.array(
                tuple(tuple(row.values()) for row in koboldai_vars.lua_koboldbridge.logits.values()),
                dtype=scores.dtype,
            )
            assert scores.shape == scores_shape

            return scores
        
        def tpumtjgenerate_stopping_callback(generated, n_generated, excluded_world_info) -> Tuple[List[set], bool, bool]:
            koboldai_vars.generated_tkns += 1

            assert len(excluded_world_info) == len(generated)
            regeneration_required = koboldai_vars.lua_koboldbridge.regeneration_required
            halt = koboldai_vars.abort or not koboldai_vars.lua_koboldbridge.generating or koboldai_vars.generated_tkns >= koboldai_vars.genamt
            koboldai_vars.lua_koboldbridge.regeneration_required = False

            global past

            for i in range(koboldai_vars.numseqs):
                koboldai_vars.lua_koboldbridge.generated[i+1][koboldai_vars.generated_tkns] = int(generated[i, tpu_mtj_backend.params["seq"] + n_generated - 1].item())

            if(not koboldai_vars.dynamicscan or halt):
                return excluded_world_info, regeneration_required, halt

            for i, t in enumerate(generated):
                decoded = utils.decodenewlines(tokenizer.decode(past[i])) + utils.decodenewlines(tokenizer.decode(t[tpu_mtj_backend.params["seq"] : tpu_mtj_backend.params["seq"] + n_generated]))
                _, found = checkworldinfo(decoded, force_use_txt=True, actions=koboldai_vars._actions)
                found -= excluded_world_info[i]
                if(len(found) != 0):
                    regeneration_required = True
                    break
            return excluded_world_info, regeneration_required, halt

        def tpumtjgenerate_compiling_callback() -> None:
            print(colors.GREEN + "TPU backend compilation triggered" + colors.END)
            koboldai_vars.compiling = True

        def tpumtjgenerate_stopped_compiling_callback() -> None:
            print(colors.GREEN + "TPU backend compilation stopped" + colors.END)
            koboldai_vars.compiling = False
        
        def tpumtjgenerate_settings_callback() -> dict:
            sampler_order = vars.sampler_order[:]
            if len(sampler_order) < 7:  # Add repetition penalty at beginning if it's not present
                sampler_order = [6] + sampler_order
            return {
                "sampler_order": koboldai_vars.sampler_order,
                "top_p": float(koboldai_vars.top_p),
                "temp": float(koboldai_vars.temp),
                "top_k": int(koboldai_vars.top_k),
                "tfs": float(koboldai_vars.tfs),
                "typical": float(koboldai_vars.typical),
                "top_a": float(koboldai_vars.top_a),
                "repetition_penalty": float(koboldai_vars.rep_pen),
                "rpslope": float(koboldai_vars.rep_pen_slope),
                "rprange": int(koboldai_vars.rep_pen_range),
            }

        # If we're running Colab or OAI, we still need a tokenizer.
        if(koboldai_vars.model in ("Colab", "API")):
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-2.7B", revision=koboldai_vars.revision, cache_dir="cache")
            loadsettings()
        elif(koboldai_vars.model == "OAI"):
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
            loadsettings()
        # Load the TPU backend if requested
        elif(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
            global tpu_mtj_backend
            import tpu_mtj_backend
            if(koboldai_vars.model == "TPUMeshTransformerGPTNeoX"):
                koboldai_vars.badwordsids = koboldai_vars.badwordsids_neox
            print("{0}Initializing Mesh Transformer JAX, please wait...{1}".format(colors.PURPLE, colors.END))
            if koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and (not koboldai_vars.custmodpth or not os.path.isdir(koboldai_vars.custmodpth)):
                raise FileNotFoundError(f"The specified model path {repr(koboldai_vars.custmodpth)} is not the path to a valid folder")
            import tpu_mtj_backend
            if(koboldai_vars.model == "TPUMeshTransformerGPTNeoX"):
                tpu_mtj_backend.pad_token_id = 2
            tpu_mtj_backend.koboldai_vars = koboldai_vars
            tpu_mtj_backend.warper_callback = tpumtjgenerate_warper_callback
            tpu_mtj_backend.stopping_callback = tpumtjgenerate_stopping_callback
            tpu_mtj_backend.compiling_callback = tpumtjgenerate_compiling_callback
            tpu_mtj_backend.stopped_compiling_callback = tpumtjgenerate_stopped_compiling_callback
            tpu_mtj_backend.settings_callback = tpumtjgenerate_settings_callback
            koboldai_vars.allowsp = True
            loadmodelsettings()
            loadsettings()
            tpu_mtj_backend.load_model(koboldai_vars.custmodpth, hf_checkpoint=koboldai_vars.model not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and koboldai_vars.use_colab_tpu, **koboldai_vars.modelconfig)
            koboldai_vars.modeldim = int(tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"]))
            tokenizer = tpu_mtj_backend.tokenizer
            if(koboldai_vars.badwordsids is koboldai_settings.badwordsids_default and koboldai_vars.model_type not in ("gpt2", "gpt_neo", "gptj")):
                koboldai_vars.badwordsids = [[v] for k, v in tokenizer.get_vocab().items() if any(c in str(k) for c in "<>[]") if koboldai_vars.newlinemode != "s" or str(k) != "</s>"]
        else:
            loadsettings()
    
    lua_startup()
    # Load scripts
    load_lua_scripts()
    
    final_startup()
    if not initial_load:
        set_aibusy(False)
        emit('from_server', {'cmd': 'hide_model_name'}, broadcast=True, room="UI_1")
        time.sleep(0.1)
        
        if not koboldai_vars.gamestarted:
            setStartState()
            sendsettings()
            refresh_settings()
    
    #Saving the tokenizer to the KoboldStoryRegister class so we can do token counting on the story data
    if 'tokenizer' in [x for x in globals()]:
        koboldai_vars.tokenizer = tokenizer
    
    #Let's load the presets
    with open('official.presets') as f:
        presets = json.load(f)
        koboldai_vars.uid_presets = {x['uid']: x for x in presets}
        #We want our data to be a 2 deep dict. Top level is "Recommended", "Same Class", "Model 1", "Model 2", etc
        #Next layer is "Official", "Custom"
        #Then the preset name
        
        to_use = OrderedDict()
        
        to_use["Recommended"] = {"Official": [], "Custom": []}
        to_use["Same Class"] = {"Official": [], "Custom": []}
        to_use["Other"] = {"Official": [], "Custom": []}
        used_ids = []
        #Build recommended first:
        for preset in presets:
            if preset['Model Type'] == koboldai_vars.model and preset['uid'] not in used_ids:
                if preset['Model Category'] == 'Custom':
                    to_use['Recommended']['Custom'].append(preset)
                else:
                    to_use['Recommended']['Official'].append(preset)
                used_ids.append(preset['uid'])
        #Build Same Class
        for preset in presets:
            if preset['Model Size'] in koboldai_vars.model.replace("6.7B", "6B") and preset['uid'] not in used_ids:
                if preset['Model Category'] == 'Custom':
                    to_use['Same Class']['Custom'].append(preset)
                else:
                    to_use['Same Class']['Official'].append(preset)
                used_ids.append(preset['uid'])
        #Build the rest of the stuff
        for preset in presets:
            if preset['uid'] not in used_ids:
                used_ids.append(preset['uid'])
                if preset['Model Category'] == 'Custom':
                    to_use["Other"]['Custom'].append(preset)
                else:
                    to_use["Other"]['Official'].append(preset)
        
        koboldai_vars.presets = to_use
    koboldai_vars.aibusy = False
    koboldai_vars.splist = [[f, get_softprompt_desc(os.path.join("./softprompts", f),None,True)] for f in os.listdir("./softprompts") if os.path.isfile(os.path.join("./softprompts", f)) and valid_softprompt(os.path.join("./softprompts", f))]

# Set up Flask routes
@app.route('/')
@app.route('/index')
def index():
    if args.no_ui:
        return redirect('/api/latest')
    if 'new_ui' in request.args:
        return render_template('index_new.html', hide_ai_menu=args.noaimenu)
    else:
        return render_template('index.html', hide_ai_menu=args.noaimenu, flaskwebgui=koboldai_vars.flaskwebgui)
@app.route('/api', strict_slashes=False)
def api():
    return redirect('/api/latest')
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path,
                                   'koboldai.ico', mimetype='image/vnd.microsoft.icon')    
@app.route('/download')
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

    # Build json to write
    js = {}
    js["gamestarted"] = koboldai_vars.gamestarted
    js["prompt"]      = koboldai_vars.prompt
    js["memory"]      = koboldai_vars.memory
    js["authorsnote"] = koboldai_vars.authornote
    js["anotetemplate"] = koboldai_vars.authornotetemplate
    js["actions"]     = koboldai_vars.actions.to_json()
    js["worldinfo"]   = []
        
    # Extract only the important bits of WI
    for wi in koboldai_vars.worldinfo:
        if(wi["constant"] or wi["key"] != ""):
            js["worldinfo"].append({
                "key": wi["key"],
                "keysecondary": wi["keysecondary"],
                "content": wi["content"],
                "comment": wi["comment"],
                "folder": wi["folder"],
                "selective": wi["selective"],
                "constant": wi["constant"]
            })
    
    save = Response(json.dumps(js, indent=3))
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
    if(path.exists("settings/" + getmodelname().replace('/', '_') + ".settings")):
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        js   = json.load(file)
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
        file.close()
        
    #==================================================================#
    #  Lua runtime startup
    #==================================================================#

    print("", end="", flush=True)
    print(colors.PURPLE + "Initializing Lua Bridge... " + colors.END, end="", flush=True)

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
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        socketio.emit("error", str(e), broadcast=True, room="UI_2")
        exit(1)
    print(colors.GREEN + "OK!" + colors.END)


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
    print(colors.GREEN + "Loading Core Script" + colors.END)

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
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        socketio.emit("error", str(e), broadcast=True, room="UI_2")
        if(koboldai_vars.serverstarted):
            set_aibusy(0)

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
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
    return utils.decodenewlines(tokenizer.decode(tokens))

#==================================================================#
#  Encode string into list of token IDs using current tokenizer
#==================================================================#
@bridged_kwarg()
def lua_encode(string):
    assert type(string) is str
    if("tokenizer" not in globals()):
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")
    return tokenizer.encode(utils.encodenewlines(string), max_length=int(4e9), truncation=True)

#==================================================================#
#  Computes context given a submission, Lua array of entry UIDs and a Lua array
#  of folder UIDs
#==================================================================#
@bridged_kwarg()
def lua_compute_context(submission, entries, folders, kwargs):
    assert type(submission) is str
    if(kwargs is None):
        kwargs = koboldai_vars.lua_state.table()
    actions = koboldai_vars._actions if koboldai_vars.lua_koboldbridge.userstate == "genmod" else koboldai_vars.actions
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
    winfo, mem, anotetxt, _ = calcsubmitbudgetheader(
        submission,
        allowed_entries=allowed_entries,
        allowed_folders=allowed_folders,
        force_use_txt=True,
        scan_story=kwargs["scan_story"] if kwargs["scan_story"] != None else True,
    )
    if koboldai_vars.alt_gen:
       txt, _, _ = koboldai_vars.calc_ai_text()
       print("Using Alt Gen: {}".format(tokenizer.decode(txt)))
    else:
        txt, _, _ = calcsubmitbudget(
            len(actions),
            winfo,
            mem,
            anotetxt,
            actions,
        )
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
        if(koboldai_vars.lua_koboldbridge.userstate == "genmod"):
            del koboldai_vars._actions[chunk-1]
        koboldai_vars.lua_deleted.add(chunk)
        if(not hasattr(koboldai_vars, "_actions") or koboldai_vars._actions is not koboldai_vars.actions):
            #Instead of deleting we'll blank out the text. This way our actions and actions_metadata stay in sync and we can restore the chunk on an undo
            koboldai_vars.actions[chunk-1] = ""
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
            if(koboldai_vars.lua_koboldbridge.userstate == "genmod"):
                koboldai_vars._actions[chunk-1] = v
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
    if(koboldai_vars.model in ("Colab", "API", "OAI", "InferKit")):
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
    if(koboldai_vars.model in ("Colab", "API", "OAI", "InferKit")):
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
    spRequest("softprompts/"+filename)
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
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
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
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
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
def do_connect():
    if request.args.get("rely") == "true":
        return
    join_room("UI_{}".format(request.args.get('ui')))
    print("Joining Room UI_{}".format(request.args.get('ui')))
    if request.args.get("ui") == "2":
        ui2_connect()
        return
    print("{0}Client connected!{1}".format(colors.GREEN, colors.END))
    emit('from_server', {'cmd': 'setchatname', 'data': koboldai_vars.chatname}, room="UI_1")
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
        print("{0}Data received:{1}{2}".format(colors.GREEN, msg, colors.END))
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
                settingschanged()
                emit('from_server', {'cmd': 'setchatname', 'data': koboldai_vars.chatname}, room="UI_1")
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
            settingschanged()
            emit('from_server', {'cmd': 'setchatname', 'data': koboldai_vars.chatname}, room="UI_1")
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
    elif(not koboldai_vars.host and msg['cmd'] == 'savetofile'):
        savetofile()
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
        uid = int(msg['uid'])
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
        if not os.path.exists("settings/"):
            os.mkdir("settings")
        changed = True
        if not utils.HAS_ACCELERATE:
            msg['disk_layers'] = "0"
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
        vars.model = vars.model_selected
        load_model(use_gpu=msg['use_gpu'], gpu_layers=msg['gpu_layers'], disk_layers=msg['disk_layers'], online_model=msg['online_model'])
    elif(msg['cmd'] == 'show_model'):
        print("Model Name: {}".format(getmodelname()))
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
        if msg['data'] in ('NeoCustom', 'GPT2Custom') and 'path' not in msg and 'path_modelname' not in msg:
            if 'folder' not in msg or koboldai_vars.host:
                folder = "./models"
            else:
                folder = msg['folder']
            sendModelSelection(menu=msg['data'], folder=folder)
        elif msg['data'] in ('NeoCustom', 'GPT2Custom') and 'path_modelname' in msg:
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
        elif msg['data'] in ('NeoCustom', 'GPT2Custom'):
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
                print(colors.YELLOW + "WARNING: Someone deleted " + msg['data'])
                import shutil
                shutil.rmtree(msg['data'])
                sendModelSelection(menu=msg['menu'])
            else:
                print(colors.RED + "ERROR: Someone attempted to delete " + msg['data'] + " but this is not a valid model")
        else:
            print(colors.RED + "WARNING!!: Someone maliciously attempted to delete " + msg['data'] + " the attempt has been blocked.")
    elif(msg['cmd'] == 'OAI_Key_Update'):
        get_oai_models({'model': koboldai_vars.model, 'key': msg['key']})
    elif(msg['cmd'] == 'loadselect'):
        koboldai_vars.loadselect = msg["data"]
    elif(msg['cmd'] == 'spselect'):
        koboldai_vars.spselect = msg["data"]
    elif(msg['cmd'] == 'loadrequest'):
        loadRequest(fileops.storypath(koboldai_vars.loadselect))
    elif(msg['cmd'] == 'sprequest'):
        spRequest("softprompts/"+koboldai_vars.spselect)
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
    elif(msg['cmd'] == 'setshowprobs'):
        koboldai_vars.show_probs = msg['data']
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
    if(koboldai_vars.welcome):
        txt = kml(koboldai_vars.welcome) + "<br/>"
    else:
        txt = "<span>Welcome to <span class=\"color_cyan\">KoboldAI</span>! You are running <span class=\"color_green\">"+getmodelname()+"</span>.<br/>"
    if(not koboldai_vars.noai and not koboldai_vars.welcome):
        txt = txt + "Please load a game or enter a prompt below to begin!</span>"
    if(koboldai_vars.noai):
        txt = txt + "Please load or import a story to read. There is no AI in this mode."
    emit('from_server', {'cmd': 'updatescreen', 'gamestarted': koboldai_vars.gamestarted, 'data': txt}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setgamestate', 'data': 'start'}, broadcast=True, room="UI_1")

#==================================================================#
#  Transmit applicable settings to SocketIO to build UI sliders/toggles
#==================================================================#
def sendsettings():
    # Send settings for selected AI type
    emit('from_server', {'cmd': 'reset_menus'}, room="UI_1")
    if(koboldai_vars.model != "InferKit"):
        for set in gensettings.gensettingstf:
            if 'UI_V2_Only' not in set:
                emit('from_server', {'cmd': 'addsetting', 'data': set}, room="UI_1")
    else:
        for set in gensettings.gensettingsik:
            if 'UI_V2_Only' not in set:
                emit('from_server', {'cmd': 'addsetting', 'data': set}, room="UI_1")
    
    # Send formatting options
    for frm in gensettings.formatcontrols:
        emit('from_server', {'cmd': 'addformat', 'data': frm}, room="UI_1")
        # Add format key to vars if it wasn't loaded with client.settings
        if(not hasattr(koboldai_vars, frm["id"])):
            setattr(koboldai_vars, frm["id"], False)

#==================================================================#
#  Set value of gamesaved
#==================================================================#
def setgamesaved(gamesaved):
    assert type(gamesaved) is bool
    if(gamesaved != koboldai_vars.gamesaved):
        emit('from_server', {'cmd': 'gamesaved', 'data': gamesaved}, broadcast=True, room="UI_1")
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

def actionsubmit(data, actionmode=0, force_submit=False, force_prompt_gen=False, disable_recentrng=False, no_generate=False):
    # Ignore new submissions if the AI is currently busy
    if(koboldai_vars.aibusy):
        return
    
    while(True):
        set_aibusy(1)
        koboldai_vars.actions.clear_unused_options()
        if(koboldai_vars.model == "API"):
            global tokenizer
            tokenizer_id = requests.get(
                koboldai_vars.colaburl[:-8] + "/api/v1/model",
            ).json()["result"]
            if tokenizer_id != koboldai_vars.api_tokenizer_id:
                try:
                    if(os.path.isdir(tokenizer_id)):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=koboldai_vars.revision, cache_dir="cache")
                        except:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                    elif(os.path.isdir("models/{}".format(tokenizer_id.replace('/', '_')))):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(tokenizer_id.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache")
                        except:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(tokenizer_id.replace('/', '_')), revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                    else:
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=koboldai_vars.revision, cache_dir="cache")
                        except:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=koboldai_vars.revision, cache_dir="cache", use_fast=False)
                except:
                    print(f"WARNING:  Unknown tokenizer {repr(tokenizer_id)}")
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
            data = re.sub(r'\n+', ' ', data)
            if(len(data)):
                data = f"\n{koboldai_vars.chatname}: {data}\n"
        
        # If we're not continuing, store a copy of the raw input
        if(data != ""):
            koboldai_vars.lastact = data
        
        if(not koboldai_vars.gamestarted):
            koboldai_vars.submission = data
            if(not no_generate):
                execute_inmod()
            koboldai_vars.submission = re.sub(r"[^\S\r\n]*([\r\n]*)$", r"\1", koboldai_vars.submission)  # Remove trailing whitespace, excluding newlines
            data = koboldai_vars.submission
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
                calcsubmit(data) # Run the first action through the generator
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
                koboldai_vars.actions.append_options([applyoutputformatting(x["generated_text"]) for x in genout])
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
                    koboldai_vars.actions.append(data)
                update_story_chunk('last')
                send_debug()

            if(not no_generate and not koboldai_vars.noai and koboldai_vars.lua_koboldbridge.generating):
                # Off to the tokenizer!
                calcsubmit(data)
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
                koboldai_vars.actions.append_options([applyoutputformatting(x["generated_text"]) for x in genout])
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
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, utils.decodenewlines(tokenizer.decode(txt)), colors.END))

    # Clear CUDA cache if using GPU
    if(koboldai_vars.hascuda and (koboldai_vars.usegpu or koboldai_vars.breakmodel)):
        gc.collect()
        torch.cuda.empty_cache()

    # Submit input text to generator
    _genout, already_generated = tpool.execute(_generate, txt, minimum, maximum, set())

    genout = [applyoutputformatting(utils.decodenewlines(tokenizer.decode(tokens[-already_generated:]))) for tokens in _genout]

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
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, utils.decodenewlines(tokenizer.decode(txt)), colors.END))

    koboldai_vars._actions = koboldai_vars.actions
    koboldai_vars._prompt = koboldai_vars.prompt
    if(koboldai_vars.dynamicscan):
        koboldai_vars._actions = koboldai_vars._actions.copy()

    # Submit input text to generator
    soft_tokens = tpumtjgetsofttokens()
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
    genout = [applyoutputformatting(utils.decodenewlines(tokenizer.decode(txt))) for txt in genout]

    return genout

def apiactionsubmit(data, use_memory=False, use_world_info=False, use_story=False, use_authors_note=False):
    if(koboldai_vars.model == "Colab"):
        raise NotImplementedError("API generation is not supported in old Colab API mode.")
    elif(koboldai_vars.model == "API"):
        raise NotImplementedError("API generation is not supported in API mode.")
    elif(koboldai_vars.model == "OAI"):
        raise NotImplementedError("API generation is not supported in OpenAI/GooseAI mode.")
    elif(koboldai_vars.model == "ReadOnly"):
        raise NotImplementedError("API generation is not supported in read-only mode; please load a model and then try again.")

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
        world_info, _ = checkworldinfo(data, force_use_txt=True, scan_story=use_story)
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

    if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ["Colab", "API", "OAI", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
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
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=koboldai_vars.revision, cache_dir="cache")

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
        tokens = (tokenizer._koboldai_header if koboldai_vars.model not in ("Colab", "API", "OAI") else []) + memtokens + witokens + anotetkns + prompttkns
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
                # header, mem, wi, anote, prompt, actions
                tokens = (tokenizer._koboldai_header if koboldai_vars.model not in ("Colab", "API", "OAI") else []) + memtokens + witokens + anotetkns + prompttkns + tokens
            else:
                tokens = (tokenizer._koboldai_header if koboldai_vars.model not in ("Colab", "API", "OAI") else []) + memtokens + witokens + prompttkns + tokens
        else:
            # Prepend Memory, WI, and Prompt before action tokens
            tokens = (tokenizer._koboldai_header if koboldai_vars.model not in ("Colab", "API", "OAI") else []) + memtokens + witokens + prompttkns + tokens

        # Send completed bundle to generator
        assert len(tokens) <= koboldai_vars.max_length - lnsp - koboldai_vars.genamt - budget_deduction
        ln = len(tokens) + lnsp

        return tokens, ln+1, ln+koboldai_vars.genamt

#==================================================================#
# Take submitted text and build the text to be given to generator
#==================================================================#
def calcsubmit(txt):
    anotetxt     = ""    # Placeholder for Author's Note text
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    actionlen    = len(koboldai_vars.actions)

    winfo, mem, anotetxt, found_entries = calcsubmitbudgetheader(txt)
 
    # For all transformers models
    if(koboldai_vars.model != "InferKit"):
        if koboldai_vars.alt_gen:
            subtxt, min, max  = koboldai_vars.calc_ai_text(submitted_text=txt)
            print("Using Alt Gen: {}".format(tokenizer.decode(subtxt)))
        else:
            subtxt, min, max = calcsubmitbudget(actionlen, winfo, mem, anotetxt, koboldai_vars.actions, submission=txt)
        if(actionlen == 0):
            if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ["Colab", "API", "OAI", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(koboldai_vars.model == "Colab"):
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(koboldai_vars.model == "API"):
                sendtoapi(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(koboldai_vars.model == "OAI"):
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
                tpumtjgenerate(subtxt, min, max, found_entries=found_entries)
        else:
            if(not koboldai_vars.use_colab_tpu and koboldai_vars.model not in ["Colab", "API", "OAI", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(koboldai_vars.model == "Colab"):
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(koboldai_vars.model == "API"):
                sendtoapi(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(koboldai_vars.model == "OAI"):
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
                tpumtjgenerate(subtxt, min, max, found_entries=found_entries)
                    
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

#==================================================================#
# Send text to generator and deal with output
#==================================================================#

def _generate(txt, minimum, maximum, found_entries):
    if(koboldai_vars.full_determinism):
        torch.manual_seed(koboldai_vars.seed)

    gen_in = torch.tensor(txt, dtype=torch.long)[None]
    if(koboldai_vars.sp is not None):
        soft_tokens = torch.arange(
            model.config.vocab_size,
            model.config.vocab_size + koboldai_vars.sp.shape[0],
        )
        gen_in = torch.cat((soft_tokens[None], gen_in), dim=-1)
    assert gen_in.shape[-1] + koboldai_vars.genamt <= koboldai_vars.max_length

    if(koboldai_vars.hascuda and koboldai_vars.usegpu):
        gen_in = gen_in.to(koboldai_vars.gpu_device)
    elif(koboldai_vars.hascuda and koboldai_vars.breakmodel):
        gen_in = gen_in.to(breakmodel.primary_device)
    else:
        gen_in = gen_in.to('cpu')

    model.kai_scanner_excluded_world_info = found_entries

    koboldai_vars._actions = koboldai_vars.actions
    koboldai_vars._prompt = koboldai_vars.prompt
    if(koboldai_vars.dynamicscan):
        koboldai_vars._actions = [x for x in koboldai_vars.actions]

    with torch.no_grad():
        already_generated = 0
        numseqs = koboldai_vars.numseqs
        while True:
            genout = generator(
                gen_in, 
                do_sample=True, 
                max_length=int(2e9),
                repetition_penalty=1.0,
                bad_words_ids=koboldai_vars.badwordsids,
                use_cache=True,
                num_return_sequences=numseqs
                )
            already_generated += len(genout[0]) - len(gen_in[0])
            assert already_generated <= koboldai_vars.genamt
            if(model.kai_scanner.halt or not model.kai_scanner.regeneration_required):
                break
            assert genout.ndim >= 2
            assert genout.shape[0] == koboldai_vars.numseqs
            if(koboldai_vars.lua_koboldbridge.generated_cols and koboldai_vars.generated_tkns != koboldai_vars.lua_koboldbridge.generated_cols):
                raise RuntimeError("Inconsistency detected between KoboldAI Python and Lua backends")
            if(already_generated != koboldai_vars.generated_tkns):
                raise RuntimeError("WI scanning error")
            for r in range(koboldai_vars.numseqs):
                for c in range(already_generated):
                    assert koboldai_vars.lua_koboldbridge.generated[r+1][c+1] is not None
                    genout[r][genout.shape[-1] - already_generated + c] = koboldai_vars.lua_koboldbridge.generated[r+1][c+1]
            encoded = []
            for i in range(koboldai_vars.numseqs):
                txt = utils.decodenewlines(tokenizer.decode(genout[i, -already_generated:]))
                winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True, actions=koboldai_vars._actions)
                found_entries[i].update(_found_entries)
                if koboldai_vars.alt_gen:
                   txt, _, _ = koboldai_vars.calc_ai_text(submitted_text=txt)
                   print("Using Alt Gen: {}".format(tokenizer.decode(txt)))
                else:
                    txt, _, _ = calcsubmitbudget(len(koboldai_vars._actions), winfo, mem, anotetxt, koboldai_vars._actions, submission=txt)
                encoded.append(torch.tensor(txt, dtype=torch.long, device=genout.device))
            max_length = len(max(encoded, key=len))
            encoded = torch.stack(tuple(torch.nn.functional.pad(e, (max_length - len(e), 0), value=model.config.pad_token_id or model.config.eos_token_id) for e in encoded))
            genout = torch.cat(
                (
                    encoded,
                    genout[..., -already_generated:],
                ),
                dim=-1
            )
            if(koboldai_vars.sp is not None):
                soft_tokens = torch.arange(
                    model.config.vocab_size,
                    model.config.vocab_size + koboldai_vars.sp.shape[0],
                    device=genout.device,
                )
                genout = torch.cat((soft_tokens.tile(koboldai_vars.numseqs, 1), genout), dim=-1)
            assert genout.shape[-1] + koboldai_vars.genamt - already_generated <= koboldai_vars.max_length
            diff = genout.shape[-1] - gen_in.shape[-1]
            minimum += diff
            maximum += diff
            gen_in = genout
            numseqs = 1
    
    return genout, already_generated
    

def generate(txt, minimum, maximum, found_entries=None):    
    koboldai_vars.generated_tkns = 0

    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(koboldai_vars.numseqs))

    if not koboldai_vars.quiet:
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, utils.decodenewlines(tokenizer.decode(txt)), colors.END))

    # Store context in memory to use it for comparison with generated content
    koboldai_vars.lastctx = utils.decodenewlines(tokenizer.decode(txt))

    # Clear CUDA cache if using GPU
    if(koboldai_vars.hascuda and (koboldai_vars.usegpu or koboldai_vars.breakmodel)):
        gc.collect()
        torch.cuda.empty_cache()

    # Submit input text to generator
    try:
        genout, already_generated = tpool.execute(_generate, txt, minimum, maximum, found_entries)
    except Exception as e:
        if(issubclass(type(e), lupa.LuaError)):
            koboldai_vars.lua_koboldbridge.obliterate_multiverse()
            koboldai_vars.lua_running = False
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
            sendUSStatItems()
            print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
            print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
            print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
            socketio.emit("error", str(e), broadcast=True, room="UI_2")
        else:
            emit('from_server', {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'}, broadcast=True, room="UI_1")
            print("{0}{1}{2}".format(colors.RED, traceback.format_exc().replace("\033", ""), colors.END), file=sys.stderr)
            socketio.emit("error", str(e), broadcast=True, room="UI_2")
        set_aibusy(0)
        return

    for i in range(koboldai_vars.numseqs):
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
    print([applyoutputformatting(x["generated_text"]) for x in genout])
    
    if(len(genout) == 1):
        genresult(genout[0]["generated_text"])
        #koboldai_vars.actions.append(applyoutputformatting(genout[0]["generated_text"]))
    else:
        koboldai_vars.actions.append_options([applyoutputformatting(x["generated_text"]) for x in genout])
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
    
    set_aibusy(0)

#==================================================================#
#  Deal with a single return sequence from generate()
#==================================================================#
def genresult(genout, flash=True, ignore_formatting=False):
    if not koboldai_vars.quiet:
        print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
    
    # Format output before continuing
    if not ignore_formatting:
        genout = applyoutputformatting(genout)

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
        result["generated_text"] = applyoutputformatting(result["generated_text"])
        if not koboldai_vars.quiet:
            print("{0}[Result {1}]\n{2}{3}".format(colors.CYAN, i, result["generated_text"], colors.END))
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
#  Send transformers-style request to ngrok/colab host
#==================================================================#
def sendtocolab(txt, min, max):
    # Log request to console
    if not koboldai_vars.quiet:
        print("{0}Tokens:{1}, Txt:{2}{3}".format(colors.YELLOW, min-1, txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    koboldai_vars.lastctx = txt
    
    # Build request JSON data
    reqdata = {
        'text': txt,
        'min': min,
        'max': max,
        'rep_pen': koboldai_vars.rep_pen,
        'rep_pen_slope': koboldai_vars.rep_pen_slope,
        'rep_pen_range': koboldai_vars.rep_pen_range,
        'temperature': koboldai_vars.temp,
        'top_p': koboldai_vars.top_p,
        'top_k': koboldai_vars.top_k,
        'tfs': koboldai_vars.tfs,
        'typical': koboldai_vars.typical,
        'topa': koboldai_vars.top_a,
        'numseqs': koboldai_vars.numseqs,
        'retfultxt': False
    }
    
    # Create request
    req = requests.post(
        koboldai_vars.colaburl, 
        json = reqdata
        )
    
    # Deal with the response
    if(req.status_code == 200):
        js = req.json()["data"]
        
        # Try to be backwards compatible with outdated colab
        if("text" in js):
            genout = [getnewcontent(js["text"])]
        else:
            genout = js["seqs"]
        
        for i in range(koboldai_vars.numseqs):
            koboldai_vars.lua_koboldbridge.outputs[i+1] = genout[i]

        execute_outmod()
        if(koboldai_vars.lua_koboldbridge.regeneration_required):
            koboldai_vars.lua_koboldbridge.regeneration_required = False
            genout = []
            for i in range(koboldai_vars.numseqs):
                genout.append(koboldai_vars.lua_koboldbridge.outputs[i+1])
                assert type(genout[-1]) is str

        koboldai_vars.actions.clear_unused_options()
        koboldai_vars.actions.append_options([applyoutputformatting(x["generated_text"]) for x in genout])
        genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
        if(len(genout) == 1):
            
            genresult(genout[0])
        else:
            # Convert torch output format to transformers
            seqs = []
            for seq in genout:
                seqs.append({"generated_text": seq})
            if(koboldai_vars.lua_koboldbridge.restart_sequence is not None and koboldai_vars.lua_koboldbridge.restart_sequence > 0):
                genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
            else:
                genselect(genout)
        
        # Format output before continuing
        #genout = applyoutputformatting(getnewcontent(genout))
        
        # Add formatted text to Actions array and refresh the game screen
        #koboldai_vars.actions.append(genout)
        #refresh_story()
        #emit('from_server', {'cmd': 'texteffect', 'data': koboldai_vars.actions.get_last_key() + 1 if len(koboldai_vars.actions) else 0})
        
        set_aibusy(0)
    else:
        errmsg = "Colab API Error: Failed to get a reply from the server. Please check the colab console."
        print("{0}{1}{2}".format(colors.RED, errmsg, colors.END))
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True, room="UI_1")
        set_aibusy(0)


#==================================================================#
#  Send transformers-style request to KoboldAI API
#==================================================================#
def sendtoapi(txt, min, max):
    # Log request to console
    if not koboldai_vars.quiet:
        print("{0}Tokens:{1}, Txt:{2}{3}".format(colors.YELLOW, min-1, txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    koboldai_vars.lastctx = txt
    
    # Build request JSON data
    reqdata = {
        'prompt': txt,
        'max_length': max - min + 1,
        'max_context_length': koboldai_vars.max_length,
        'rep_pen': koboldai_vars.rep_pen,
        'rep_pen_slope': koboldai_vars.rep_pen_slope,
        'rep_pen_range': koboldai_vars.rep_pen_range,
        'temperature': koboldai_vars.temp,
        'top_p': koboldai_vars.top_p,
        'top_k': koboldai_vars.top_k,
        'top_a': koboldai_vars.top_a,
        'tfs': koboldai_vars.tfs,
        'typical': koboldai_vars.typical,
        'n': koboldai_vars.numseqs,
    }
    
    # Create request
    while True:
        req = requests.post(
            koboldai_vars.colaburl[:-8] + "/api/v1/generate",
            json=reqdata,
        )
        if(req.status_code == 503):  # Server is currently generating something else so poll until it's our turn
            time.sleep(1)
            continue
        js = req.json()
        if(req.status_code != 200):
            errmsg = "KoboldAI API Error: Failed to get a reply from the server. Please check the console."
            print("{0}{1}{2}".format(colors.RED, json.dumps(js, indent=2), colors.END))
            emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
            set_aibusy(0)
            return

        genout = [obj["text"] for obj in js["results"]]

        for i in range(koboldai_vars.numseqs):
            koboldai_vars.lua_koboldbridge.outputs[i+1] = genout[i]

        execute_outmod()
        if(koboldai_vars.lua_koboldbridge.regeneration_required):
            koboldai_vars.lua_koboldbridge.regeneration_required = False
            genout = []
            for i in range(koboldai_vars.numseqs):
                genout.append(koboldai_vars.lua_koboldbridge.outputs[i+1])
                assert type(genout[-1]) is str

        if(len(genout) == 1):
            genresult(genout[0])
        else:
            # Convert torch output format to transformers
            seqs = []
            for seq in genout:
                seqs.append({"generated_text": seq})
            if(koboldai_vars.lua_koboldbridge.restart_sequence is not None and koboldai_vars.lua_koboldbridge.restart_sequence > 0):
                genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
            else:
                genselect(genout)

        set_aibusy(0)
        return


#==================================================================#
#  Send text to TPU mesh transformer backend
#==================================================================#
def tpumtjgenerate(txt, minimum, maximum, found_entries=None):
    if(koboldai_vars.full_determinism):
        tpu_mtj_backend.set_rng_seed(koboldai_vars.seed)

    koboldai_vars.generated_tkns = 0

    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(koboldai_vars.numseqs))

    if not koboldai_vars.quiet:
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, utils.decodenewlines(tokenizer.decode(txt)), colors.END))

    koboldai_vars._actions = koboldai_vars.actions
    koboldai_vars._prompt = koboldai_vars.prompt
    if(koboldai_vars.dynamicscan):
        koboldai_vars._actions = koboldai_vars._actions.copy()

    # Submit input text to generator
    try:
        soft_tokens = tpumtjgetsofttokens()

        global past

        socketio.start_background_task(copy_current_request_context(check_for_backend_compilation))

        if(koboldai_vars.dynamicscan or (not koboldai_vars.nogenmod and koboldai_vars.has_genmod)):

            context = np.tile(np.uint32(txt), (koboldai_vars.numseqs, 1))
            past = np.empty((koboldai_vars.numseqs, 0), dtype=np.uint32)

            while(True):
                genout, n_generated, regeneration_required, halt = tpool.execute(
                    tpu_mtj_backend.infer_dynamic,
                    context,
                    gen_len = maximum-minimum+1,
                    numseqs=koboldai_vars.numseqs,
                    soft_embeddings=koboldai_vars.sp,
                    soft_tokens=soft_tokens,
                    excluded_world_info=found_entries,
                )

                past = np.pad(past, ((0, 0), (0, n_generated)))
                for r in range(koboldai_vars.numseqs):
                    for c in range(koboldai_vars.lua_koboldbridge.generated_cols):
                        assert koboldai_vars.lua_koboldbridge.generated[r+1][c+1] is not None
                        past[r, c] = koboldai_vars.lua_koboldbridge.generated[r+1][c+1]

                if(koboldai_vars.abort or halt or not regeneration_required):
                    break
                print("(regeneration triggered)")

                encoded = []
                for i in range(koboldai_vars.numseqs):
                    txt = utils.decodenewlines(tokenizer.decode(past[i]))
                    winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True, actions=koboldai_vars._actions)
                    found_entries[i].update(_found_entries)
                    if koboldai_vars.alt_gen:
                       txt, _, _ = koboldai_vars.calc_ai_text(submitted_text=txt)
                       print("Using Alt Gen: {}".format(tokenizer.decode(txt)))
                    else:
                        txt, _, _ = calcsubmitbudget(len(koboldai_vars._actions), winfo, mem, anotetxt, koboldai_vars._actions, submission=txt)
                    encoded.append(np.array(txt, dtype=np.uint32))
                max_length = len(max(encoded, key=len))
                encoded = np.stack(tuple(np.pad(e, (max_length - len(e), 0), constant_values=tpu_mtj_backend.pad_token_id) for e in encoded))
                context = np.concatenate(
                    (
                        encoded,
                        past,
                    ),
                    axis=-1,
                )

        else:
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
            past = genout
            for i in range(koboldai_vars.numseqs):
                koboldai_vars.lua_koboldbridge.generated[i+1] = koboldai_vars.lua_state.table(*genout[i].tolist())
            koboldai_vars.lua_koboldbridge.generated_cols = koboldai_vars.generated_tkns = genout[0].shape[-1]

    except Exception as e:
        if(issubclass(type(e), lupa.LuaError)):
            koboldai_vars.lua_koboldbridge.obliterate_multiverse()
            koboldai_vars.lua_running = False
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
            sendUSStatItems()
            print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
            print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
            print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
            socketio.emit("error", str(e), broadcast=True, room="UI_2")
        else:
            emit('from_server', {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'}, broadcast=True, room="UI_1")
            print("{0}{1}{2}".format(colors.RED, traceback.format_exc().replace("\033", ""), colors.END), file=sys.stderr)
            socketio.emit("error", str(e), broadcast=True, room="UI_2")
        set_aibusy(0)
        return

    for i in range(koboldai_vars.numseqs):
        koboldai_vars.lua_koboldbridge.outputs[i+1] = utils.decodenewlines(tokenizer.decode(past[i]))
    genout = past

    execute_outmod()
    if(koboldai_vars.lua_koboldbridge.regeneration_required):
        koboldai_vars.lua_koboldbridge.regeneration_required = False
        genout = []
        for i in range(koboldai_vars.numseqs):
            genout.append({"generated_text": koboldai_vars.lua_koboldbridge.outputs[i+1]})
            assert type(genout[-1]["generated_text"]) is str
    else:
        genout = [{"generated_text": utils.decodenewlines(tokenizer.decode(txt))} for txt in genout]

    koboldai_vars.actions.append_options([applyoutputformatting(x["generated_text"]) for x in genout])
    genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
    if(len(koboldai_vars.actions.get_current_options()) == 1):
        genresult(koboldai_vars.actions.get_current_options()[0]['text'])
    else:
        if(koboldai_vars.lua_koboldbridge.restart_sequence is not None and koboldai_vars.lua_koboldbridge.restart_sequence > 0):
            genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
        else:
            genselect([{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()])

    set_aibusy(0)


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
    if(koboldai_vars.frmtadsnsp):
        txt = utils.addsentencespacing(txt, koboldai_vars)
 
    return txt

#==================================================================#
# Applies chosen formatting options to text returned from AI
#==================================================================#
def applyoutputformatting(txt):
    # Use standard quotes and apostrophes
    txt = utils.fixquotes(txt)

    # Adventure mode clipping of all characters after '>'
    if(koboldai_vars.adventure):
        txt = koboldai_vars.acregex_ai.sub('', txt)
    
    # Trim incomplete sentences
    if(koboldai_vars.frmttriminc and not koboldai_vars.chatmode):
        txt = utils.trimincompletesentence(txt)
    # Replace blank lines
    if(koboldai_vars.frmtrmblln or koboldai_vars.chatmode):
        txt = utils.replaceblanklines(txt)
    # Remove special characters
    if(koboldai_vars.frmtrmspch):
        txt = utils.removespecialchars(txt, koboldai_vars)
	# Single Line Mode
    if(koboldai_vars.singleline or koboldai_vars.chatmode):
        txt = utils.singlelineprocessing(txt, koboldai_vars)
    
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

    #If we've set the auto save flag, we'll now save the file
    if koboldai_vars.autosave and (".json" in koboldai_vars.savedir):
        save()


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
    emit('from_server', {'cmd': 'allowtoggle', 'data': False}, broadcast=True, room="UI_1")
    
    if(koboldai_vars.model != "InferKit"):
        emit('from_server', {'cmd': 'updatetemp', 'data': koboldai_vars.temp}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopp', 'data': koboldai_vars.top_p}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopk', 'data': koboldai_vars.top_k}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetfs', 'data': koboldai_vars.tfs}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetypical', 'data': koboldai_vars.typical}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopa', 'data': koboldai_vars.top_a}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatereppen', 'data': koboldai_vars.rep_pen}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatereppenslope', 'data': koboldai_vars.rep_pen_slope}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatereppenrange', 'data': koboldai_vars.rep_pen_range}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updateoutlen', 'data': koboldai_vars.genamt}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetknmax', 'data': koboldai_vars.max_length}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatenumseq', 'data': koboldai_vars.numseqs}, broadcast=True, room="UI_1")
    else:
        emit('from_server', {'cmd': 'updatetemp', 'data': koboldai_vars.temp}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopp', 'data': koboldai_vars.top_p}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updateikgen', 'data': koboldai_vars.ikgen}, broadcast=True, room="UI_1")
    
    emit('from_server', {'cmd': 'updateanotedepth', 'data': koboldai_vars.andepth}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatewidepth', 'data': koboldai_vars.widepth}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateuseprompt', 'data': koboldai_vars.useprompt}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateadventure', 'data': koboldai_vars.adventure}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatechatmode', 'data': koboldai_vars.chatmode}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatedynamicscan', 'data': koboldai_vars.dynamicscan}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateautosave', 'data': koboldai_vars.autosave}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatenopromptgen', 'data': koboldai_vars.nopromptgen}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updaterngpersist', 'data': koboldai_vars.rngpersist}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatenogenmod', 'data': koboldai_vars.nogenmod}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatefulldeterminism', 'data': koboldai_vars.full_determinism}, broadcast=True, room="UI_1")
    
    emit('from_server', {'cmd': 'updatefrmttriminc', 'data': koboldai_vars.frmttriminc}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatefrmtrmblln', 'data': koboldai_vars.frmtrmblln}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatefrmtrmspch', 'data': koboldai_vars.frmtrmspch}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatefrmtadsnsp', 'data': koboldai_vars.frmtadsnsp}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatesingleline', 'data': koboldai_vars.singleline}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateoutputstreaming', 'data': koboldai_vars.output_streaming}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateshowprobs', 'data': koboldai_vars.show_probs}, broadcast=True, room="UI_1")
    
    # Allow toggle events again
    emit('from_server', {'cmd': 'allowtoggle', 'data': True}, broadcast=True, room="UI_1")

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
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")

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
            print(f"WARNING: Attempted to edit non-existent chunk {chunk}")

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
            print(f"WARNING: Attempted to delete non-existent chunk {chunk}")
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
            if(e is koboldai_vars.worldinfo_u[src]):
                koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[src]["folder"]].pop(i)
                break
    if(koboldai_vars.worldinfo_u[dst]["folder"] is not None):
        koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[dst]["folder"]].append(koboldai_vars.worldinfo_u[src])
    koboldai_vars.worldinfo_u[src]["folder"] = koboldai_vars.worldinfo_u[dst]["folder"]
    for i, e in enumerate(koboldai_vars.worldinfo):
        if(e is koboldai_vars.worldinfo_u[src]):
            _src = i
        elif(e is koboldai_vars.worldinfo_u[dst]):
            _dst = i
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
        ob["uid"] = int(ob["uid"])
        koboldai_vars.worldinfo_u[ob["uid"]]["key"]          = ob["key"]
        koboldai_vars.worldinfo_u[ob["uid"]]["keysecondary"] = ob["keysecondary"]
        koboldai_vars.worldinfo_u[ob["uid"]]["content"]      = ob["content"]
        koboldai_vars.worldinfo_u[ob["uid"]]["comment"]      = ob.get("comment", "")
        koboldai_vars.worldinfo_u[ob["uid"]]["folder"]       = ob.get("folder", None)
        koboldai_vars.worldinfo_u[ob["uid"]]["selective"]    = ob["selective"]
        koboldai_vars.worldinfo_u[ob["uid"]]["constant"]     = ob.get("constant", False)
    stablesortwi()
    koboldai_vars.worldinfo_i = [wi for wi in koboldai_vars.worldinfo if wi["init"]]

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
                    if(e is koboldai_vars.worldinfo_u[koboldai_vars.deletewi]):
                        koboldai_vars.wifolders_u[koboldai_vars.worldinfo_u[koboldai_vars.deletewi]["folder"]].pop(i)
            for i, e in enumerate(koboldai_vars.worldinfo):
                if(e is koboldai_vars.worldinfo_u[koboldai_vars.deletewi]):
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
    uid = int(uid)
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
            chunks = collections.deque()
            i = 0
            for key in reversed(actions):
                chunk = actions[key]
                chunks.appendleft(chunk)
                i += 1
                if(i == depth):
                    break
        
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
                if ky in txt:
                    if wi.get("selective", False) and len(keys_secondary):
                        found = False
                        for ks in keys_secondary:
                            ksy = ks
                            if(koboldai_vars.wirmvwhtsp):
                                ksy = ks.strip()
                            if ksy in txt:
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
        print("anotesubmit")
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
#  Assembles game data into a request to OpenAI API
#==================================================================#
def oairequest(txt, min, max):
    # Log request to console
    if not koboldai_vars.quiet:
        print("{0}Len:{1}, Txt:{2}{3}".format(colors.YELLOW, len(txt), txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    koboldai_vars.lastctx = txt
    
    # Build request JSON data
    if 'GooseAI' in args.configname:
        reqdata = {
            'prompt': txt,
            'max_tokens': koboldai_vars.genamt,
            'temperature': koboldai_vars.temp,
            'top_a': koboldai_vars.top_a,
            'top_p': koboldai_vars.top_p,
            'top_k': koboldai_vars.top_k,
            'tfs': koboldai_vars.tfs,
            'typical_p': koboldai_vars.typical,
            'repetition_penalty': koboldai_vars.rep_pen,
            'repetition_penalty_slope': koboldai_vars.rep_pen_slope,
            'repetition_penalty_range': koboldai_vars.rep_pen_range,
            'n': koboldai_vars.numseqs,
            'stream': False
        }
    else:
        reqdata = {
            'prompt': txt,
            'max_tokens': koboldai_vars.genamt,
            'temperature': koboldai_vars.temp,
            'top_p': koboldai_vars.top_p,
            'n': koboldai_vars.numseqs,
            'stream': False
        }
    
    req = requests.post(
        koboldai_vars.oaiurl, 
        json    = reqdata,
        headers = {
            'Authorization': 'Bearer '+koboldai_vars.oaiapikey,
            'Content-Type': 'application/json'
            }
        )
    
    # Deal with the response
    if(req.status_code == 200):
        outputs = [out["text"] for out in req.json()["choices"]]

        for idx in range(len(outputs)):
            koboldai_vars.lua_koboldbridge.outputs[idx+1] = outputs[idx]

        execute_outmod()
        if (koboldai_vars.lua_koboldbridge.regeneration_required):
            koboldai_vars.lua_koboldbridge.regeneration_required = False
            genout = []
            for i in range(len(outputs)):
                genout.append(
                    {"generated_text": koboldai_vars.lua_koboldbridge.outputs[i + 1]})
                assert type(genout[-1]["generated_text"]) is str
        else:
            genout = [
                {"generated_text": utils.decodenewlines(txt)}
                for txt in outputs]

        koboldai_vars.actions.append_options([applyoutputformatting(x["generated_text"]) for x in genout])
        genout = [{"generated_text": x['text']} for x in koboldai_vars.actions.get_current_options()]
        if (len(genout) == 1):
            genresult(genout[0]["generated_text"])
        else:
            if (koboldai_vars.lua_koboldbridge.restart_sequence is not None and
                    koboldai_vars.lua_koboldbridge.restart_sequence > 0):
                genresult(genout[koboldai_vars.lua_koboldbridge.restart_sequence - 1][
                              "generated_text"])
            else:
                genselect(genout)

        if not koboldai_vars.quiet:
            print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))

        set_aibusy(0)
    else:
        # Send error message to web client            
        er = req.json()
        if("error" in er):
            type    = er["error"]["type"]
            message = er["error"]["message"]
            
        errmsg = "OpenAI API Error: {0} - {1}".format(type, message)
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
    savepins = data['pins']
    # Check if filename exists already
    name = utils.cleanfilename(name)
    if(not fileops.saveexists(name) or (koboldai_vars.saveow and koboldai_vars.svowname == name)):
        # All clear to save
        e = saveRequest(fileops.storypath(name), savepins=savepins)
        koboldai_vars.saveow = False
        koboldai_vars.svowname = ""
        if(e is None):
            emit('from_server', {'cmd': 'hidesaveas', 'data': ''}, room="UI_1")
        else:
            print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
            emit('from_server', {'cmd': 'popuperror', 'data': str(e)}, room="UI_1")
    else:
        # File exists, prompt for overwrite
        koboldai_vars.saveow   = True
        koboldai_vars.svowname = name
        emit('from_server', {'cmd': 'askforoverwrite', 'data': ''}, room="UI_1")

#==================================================================#
#  Launch in-browser story-delete prompt
#==================================================================#
def deletesave(name):
    name = utils.cleanfilename(name)
    e = fileops.deletesave(name)
    if(e is None):
        if(koboldai_vars.smandelete):
            emit('from_server', {'cmd': 'hidepopupdelete', 'data': ''}, room="UI_1")
            getloadlist()
        else:
            emit('from_server', {'cmd': 'popuperror', 'data': "The server denied your request to delete this story"}, room="UI_1")
    else:
        print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
        emit('from_server', {'cmd': 'popuperror', 'data': str(e)}, room="UI_1")

#==================================================================#
#  Launch in-browser story-rename prompt
#==================================================================#
def renamesave(name, newname):
    # Check if filename exists already
    name = utils.cleanfilename(name)
    newname = utils.cleanfilename(newname)
    if(not fileops.saveexists(newname) or name == newname or (koboldai_vars.saveow and koboldai_vars.svowname == newname)):
        e = fileops.renamesave(name, newname)
        koboldai_vars.saveow = False
        koboldai_vars.svowname = ""
        if(e is None):
            if(koboldai_vars.smanrename):
                emit('from_server', {'cmd': 'hidepopuprename', 'data': ''}, room="UI_1")
                getloadlist()
            else:
                emit('from_server', {'cmd': 'popuperror', 'data': "The server denied your request to rename this story"}, room="UI_1")
        else:
            print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
            emit('from_server', {'cmd': 'popuperror', 'data': str(e)}, room="UI_1")
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
    if(".json" in koboldai_vars.savedir):
        saveRequest(koboldai_vars.savedir)
    else:
        emit('from_server', {'cmd': 'saveas', 'data': ''}, room="UI_1")

#==================================================================#
#  Save the story via file browser
#==================================================================#
def savetofile():
    savpath = fileops.getsavepath(koboldai_vars.savedir, "Save Story As", [("Json", "*.json")])
    saveRequest(savpath)

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
            file = open(savpath, "w")
        except Exception as e:
            return e
        try:
            file.write(json.dumps(js, indent=3))
        except Exception as e:
            file.close()
            return e
        file.close()
        
        try:
            file = open(txtpath, "w")
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
    if(loadpath):
        # Leave Edit/Memory mode before continuing
        exitModes()
        
        
        # Read file contents into JSON object
        if(isinstance(loadpath, str)):
            with open(loadpath, "r") as file:
                js = json.load(file)
            if(filename is None):
                filename = path.basename(loadpath)
        else:
            js = loadpath
            if(filename is None):
                filename = "untitled.json"
        js['v1_loadpath'] = loadpath
        js['v1_filename'] = filename
        loadJSON(js)

def loadJSON(json_text_or_dict):
    if isinstance(json_text_or_dict, str):
        json_data = json.loads(json_text_or_dict)
    else:
        json_data = json_text_or_dict
    if "file_version" in json_data:
        if json_data['file_version'] == 2:
            load_story_v2(json_data)
        else:
            load_story_v1(json_data)
    else:
        load_story_v1(json_data)

def load_story_v1(js):
    loadpath = js['v1_loadpath']
    filename = js['v1_filename']
    
    _filename = filename
    if(filename.endswith('.json')):
        _filename = filename[:-5]
    session['story'] = _filename
    #create the story
    #koboldai_vars.create_story(session['story'])
    koboldai_vars.create_story('default')
    
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
        for s in actions:
            koboldai_vars.actions.append(s)

    if "actions_metadata" in js:
        if type(js["actions_metadata"]) == dict:
            for key in js["actions_metadata"]:
                if js["actions_metadata"][key]["Alternative Text"] != []:
                    data = js["actions_metadata"][key]["Alternative Text"]
                    for i in range(len(js["actions_metadata"][key]["Alternative Text"])):
                        data[i]["text"] = data[i].pop("Text")
                    koboldai_vars.actions.set_options(data, key)
    
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
            koboldai_vars.worldinfo_v2.add_item([x.strip() for x in wi["key"].split(",")][0], wi["key"], wi.get("keysecondary", ""), 
                                                wi.get("folder", "root"), wi.get("constant", False), 
                                                wi["content"], wi.get("comment", ""))

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

def load_story_v2(js):
    session['story'] = js['story_name']
    koboldai_vars.load_story(session['story'], js)
    

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
    threading.Thread(target=__preempt_tokenizer).start()

    # Load soft prompt specified by the settings file, if applicable
    if(path.exists("settings/" + getmodelname().replace('/', '_') + ".settings")):
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        js   = json.load(file)
        if(koboldai_vars.allowsp and "softprompt" in js and type(js["softprompt"]) is str and all(q not in js["softprompt"] for q in ("..", ":")) and (len(js["softprompt"]) != 0 and all(js["softprompt"][0] not in q for q in ("/", "\\")))):
            if valid_softprompt("softprompts/"+js["softprompt"]):
                spRequest("softprompts/"+js["softprompt"])
        else:
            koboldai_vars.spfilename = ""
        file.close()

    # Precompile TPU backend if required
    if(koboldai_vars.use_colab_tpu or koboldai_vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        soft_tokens = tpumtjgetsofttokens()
        if(koboldai_vars.dynamicscan or (not koboldai_vars.nogenmod and koboldai_vars.has_genmod)):
            threading.Thread(
                target=tpu_mtj_backend.infer_dynamic,
                args=(np.tile(np.uint32((23403, 727, 20185)), (koboldai_vars.numseqs, 1)),),
                kwargs={
                    "soft_embeddings": koboldai_vars.sp,
                    "soft_tokens": soft_tokens,
                    "gen_len": 1,
                    "use_callback": False,
                    "numseqs": koboldai_vars.numseqs,
                    "excluded_world_info": list(set() for _ in range(koboldai_vars.numseqs)),
                },
            ).start()
        else:
            threading.Thread(
                target=tpu_mtj_backend.infer_static,
                args=(np.uint32((23403, 727, 20185)),),
                kwargs={
                    "soft_embeddings": koboldai_vars.sp,
                    "soft_tokens": soft_tokens,
                    "gen_len": 1,
                    "numseqs": koboldai_vars.numseqs,
                },
            ).start()

    # Set the initial RNG seed
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
# UI V2 CODE
#==================================================================#
@app.route('/ai_text')
def ai_text():
    start_time = time.time()
    text = koboldai_vars.calc_ai_text()
    print("Generating Game Text took {} seconds".format(time.time()-start_time))
    return text
    
    
    

#==================================================================#
# UI V2 CODE
#==================================================================#
@app.route('/new_ui')
def new_ui_index():
    if 'story' in session:
        if session['story'] not in koboldai_vars.story_list():
            session['story'] = 'default'
    return render_template('index_new.html', settings=gensettings.gensettingstf if koboldai_vars.model != "InferKit" else gensettings.gensettingsik )

def ui2_connect():
    #Send all variables to client
    koboldai_vars.send_to_ui()
    
    pass
    
#==================================================================#
# UI V2 CODE Themes
#==================================================================#
@app.route('/themes/<path:path>')
def ui2_serve_themes(path):
    return send_from_directory('themes', path)
    

#==================================================================#
# File Popup options
#==================================================================#
@socketio.on('upload_file')
def upload_file(data):
    print("upload_file {}".format(data['filename']))
    if 'current_folder' in session:
        path = os.path.abspath(os.path.join(session['current_folder'], data['filename']).replace("\\", "/")).replace("\\", "/")
        print("Want to save to {}".format(path))
        if 'popup_jailed_dir' not in session:
            print("Someone is trying to upload a file to your server. Blocked.")
        elif session['popup_jailed_dir'] is None:
            if os.path.exists(path):
                emit("error_popup", "The file already exists. Please delete it or rename the file before uploading", broadcast=False, room="UI_2");
            else:
                with open(path, "wb") as f:
                    f.write(data['data'])
                get_files_folders(session['current_folder'])
        elif session['popup_jailed_dir'] in session['current_folder']:
            if os.path.exists(path):
                emit("error_popup", "The file already exists. Please delete it or rename the file before uploading", broadcast=False,  room="UI_2");
            else:
                with open(path, "wb") as f:
                    f.write(data['data'])
                get_files_folders(session['current_folder'])

@socketio.on('popup_change_folder')
def popup_change_folder(data):
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
def popup_rename(data):
    if 'popup_renameable' not in session:
        print("Someone is trying to rename a file in your server. Blocked.")
        return
    if not session['popup_renameable']:
        print("Someone is trying to rename a file in your server. Blocked.")
        return
    
    if session['popup_jailed_dir'] is None:
        os.rename(data['file'], data['new_name'])
        get_files_folders(os.path.dirname(data['file']))
    elif session['popup_jailed_dir'] in data:
        os.rename(data['file'], data['new_name'])
        get_files_folders(os.path.dirname(data['file']))
    else:
        print("User is trying to rename files in your server outside the jail. Blocked. Jailed Dir: {}  Requested Dir: {}".format(session['popup_jailed_dir'], data['file']))

@socketio.on('popup_delete')
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

def file_popup(popup_title, starting_folder, return_event, upload=True, jailed=True, folder_only=True, renameable=False, deleteable=False, 
                                                           editable=False, show_breadcrumbs=True, item_check=None, show_hidden=False,
                                                           valid_only=False, hide_extention=False, extra_parameter_function=None,
                                                           column_names=['File Name'], show_filename=True, show_folders=True,
                                                           column_widths=["100%"],
                                                           sort="Modified", desc=False):
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
    
    socketio.emit("load_popup", {"popup_title": popup_title, "call_back": return_event, "renameable": renameable, "deleteable": deleteable, "editable": editable, 'upload': upload}, broadcast=False, room="UI_2")
    socketio.emit("load_popup", {"popup_title": popup_title, "call_back": return_event, "renameable": renameable, "deleteable": deleteable, "editable": editable, 'upload': upload}, broadcast=True, room="UI_1")
    
    get_files_folders(starting_folder)
    
def get_files_folders(starting_folder):
    import stat
    session['current_folder'] = os.path.abspath(starting_folder).replace("\\", "/")
    item_check = session['popup_item_check']
    extra_parameter_function = session['extra_parameter_function']
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
        for item in get_files_sorted(base_path, sort, desc=desc):
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
                if os.path.isdir(os.path.join(base_path, item)):
                    folders.append([True, item_full_path, item,  valid_selection, extra_parameters])
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
    socketio.emit("popup_items", {"items": items, "column_names": column_names, "show_filename": show_filename, "column_widths": column_widths}, broadcast=False, include_self=True, room="UI_2")
    socketio.emit("popup_items", items, broadcast=True, include_self=True, room="UI_1")
    if show_breadcrumbs:
        socketio.emit("popup_breadcrumbs", breadcrumbs, broadcast=False, room="UI_2")
        socketio.emit("popup_breadcrumbs", breadcrumbs, broadcast=True, room="UI_1")

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
        

#==================================================================#
# Event triggered when browser SocketIO detects a variable change
#==================================================================#
@socketio.on('var_change')
def UI_2_var_change(data):
    if 'value' not in data:
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
    else:
        print("Unknown Type {} = {}".format(name, type(getattr(koboldai_vars, name))))
    
    #print("Setting {} to {} as type {}".format(name, value, type(value)))
    setattr(koboldai_vars, name, value)
    
    #Now let's save except for story changes
    if classname != "story_settings":
        if classname == "model_settings":
            filename = "settings/{}.v2_settings".format(koboldai_vars.model.replace("/", "_"))
        else:
            filename = "settings/{}.v2_settings".format(classname)
        
        with open(filename, "w") as settings_file:
            settings_file.write(getattr(koboldai_vars, "_{}".format(classname)).to_json())
    
    return {'id': data['ID'], 'status': "Saved"}
    
#==================================================================#
# Saving Story
#==================================================================#
@socketio.on('save_story')
def UI_2_save_story(data):
    print("Saving Story")
    if data is None:
        #We need to check to see if there is a file already and if it's not the same story so we can ask the client if this is OK
        save_name = koboldai_vars.story_name if koboldai_vars.story_name != "" else "untitled"
        same_story = True
        if os.path.exists("stories/{}_v2.json".format(save_name)):
            with open("stories/{}_v2.json".format(save_name), "r") as settings_file:
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
    
#==================================================================#
# Save story to json
#==================================================================#
@app.route("/json")
def UI_2_save_to_json():
    return Response(
        koboldai_vars.to_json('story_settings'),
        mimetype="application/json",
        headers={"Content-disposition":
                 "attachment; filename={}_v2.json".format(koboldai_vars.story_name)})
    
    
#==================================================================#
# Event triggered when Selected Text is edited
#==================================================================#
@socketio.on('Set Selected Text')
def UI_2_Set_Selected_Text(data):
    print("Updating Selected Text: {}".format(data))
    koboldai_vars.actions[int(data['id'])] = data['text']

#==================================================================#
# Event triggered when Option is Selected
#==================================================================#
@socketio.on('Use Option Text')
def UI_2_Use_Option_Text(data):
    if koboldai_vars.prompt == "":
        koboldai_vars.prompt = koboldai_vars.actions.get_current_options()[int(data['option'])]['text']
        koboldai_vars.actions.clear_unused_options()
    else:
        koboldai_vars.actions.use_option(int(data['option']), action_step=int(data['chunk']))


#==================================================================#
# Event triggered when user clicks the submit button
#==================================================================#
@socketio.on('submit')
def UI_2_submit(data):
    print(data)
    if not koboldai_vars.noai and data['theme'] != "":
        print("doing random prompt")
        memory = koboldai_vars.memory
        koboldai_vars.memory = "{}\n\nYou generate the following {} story concept :".format(koboldai_vars.memory, data['theme'])
        koboldai_vars.lua_koboldbridge.feedback = None
        actionsubmit("", force_submit=True, force_prompt_gen=True)
        koboldai_vars.memory = memory
    else:
        print("doing normal input")
        koboldai_vars.actions.clear_unused_options()
        koboldai_vars.lua_koboldbridge.feedback = None
        koboldai_vars.recentrng = koboldai_vars.recentrngm = None
        if koboldai_vars.actions.action_count == -1:
            actionsubmit(data['data'], actionmode=0)
        else:
            actionsubmit(data['data'], actionmode=koboldai_vars.actionmode)
 
 #==================================================================#
# Event triggered when user clicks the submit button
#==================================================================#
@socketio.on('abort')
def UI_2_abort(data):
    print("got abort")
    koboldai_vars.abort = True
    print(koboldai_vars.abort)

 
#==================================================================#
# Event triggered when user clicks the pin button
#==================================================================#
@socketio.on('Pinning')
def UI_2_Pinning(data):
    koboldai_vars.actions.toggle_pin(int(data['chunk']), int(data['option']))
    
#==================================================================#
# Event triggered when user clicks the back button
#==================================================================#
@socketio.on('back')
def UI_2_back(data):
    print("back")
    koboldai_vars.actions.clear_unused_options()
    ignore = koboldai_vars.actions.pop()
    
#==================================================================#
# Event triggered when user clicks the redo button
#==================================================================#
@socketio.on('redo')
def UI_2_redo(data):
    if len(koboldai_vars.actions.get_current_options()) == 1:
        koboldai_vars.actions.use_option(0)
    

#==================================================================#
# Event triggered when user clicks the retry button
#==================================================================#
@socketio.on('retry')
def UI_2_retry(data):
    
    if len(koboldai_vars.actions.get_current_options_no_edits()) == 0:
        UI_2_back(None)
    koboldai_vars.actions.clear_unused_options()
    koboldai_vars.lua_koboldbridge.feedback = None
    koboldai_vars.recentrng = koboldai_vars.recentrngm = None
    actionsubmit("", actionmode=koboldai_vars.actionmode)
    
#==================================================================#
# Event triggered when user clicks the load model button
#==================================================================#
@socketio.on('load_model_button')
def UI_2_load_model_button(data):
    sendModelSelection()
    
#==================================================================#
# Event triggered when user clicks the a model
#==================================================================#
@socketio.on('select_model')
def UI_2_select_model(data):
    print(data)
    
    #We've selected a menu
    if data['model'] in model_menu:
        sendModelSelection(menu=data['model'])
    #We've selected a custom line
    elif data['menu'] in ("NeoCustom", "GPT2Custom"):
        get_model_info(data['menu'], directory=data['display_name'])
    #We've selected a custom menu folder
    elif data['model'] in ("NeoCustom", "GPT2Custom") and 'path' in data:
        sendModelSelection(menu=data['model'], folder=data['path'])
    #We've selected a custom menu
    elif data['model'] in ("NeoCustom", "GPT2Custom"):
        sendModelSelection(menu=data['model'], folder="./models")
    else:
        #We now have some model we want to potentially load.
        #First we need to send the client the model parameters (layers, etc)
        get_model_info(data['model'])

#==================================================================#
# Event triggered when user loads a model
#==================================================================#
@socketio.on('load_model')
def UI_2_load_model(data):
    print(data)
    if not os.path.exists("settings/"):
        os.mkdir("settings")
    changed = True
    if not utils.HAS_ACCELERATE:
        data['disk_layers'] = "0"
    if os.path.exists("settings/" + data['model'].replace('/', '_') + ".breakmodel"):
        with open("settings/" + data['model'].replace('/', '_') + ".breakmodel", "r") as file:
            file_data = file.read().split('\n')[:2]
            if len(file_data) < 2:
                file_data.append("0")
            gpu_layers, disk_layers = file_data
            if gpu_layers == data['gpu_layers'] and disk_layers == data['disk_layers']:
                changed = False
    if changed:
        f = open("settings/" + data['model'].replace('/', '_') + ".breakmodel", "w")
        f.write("{}\n{}".format(data['gpu_layers'], data['disk_layers']))
        f.close()
    koboldai_vars.colaburl = data['url'] + "/request"
    koboldai_vars.model = data['model']
    koboldai_vars.custmodpth = data['path']
    print("loading Model")
    load_model(use_gpu=data['use_gpu'], gpu_layers=data['gpu_layers'], disk_layers=data['disk_layers'], online_model=data['online_model'])

#==================================================================#
# Event triggered when load story is clicked
#==================================================================#
@socketio.on('load_story_list')
def UI_2_load_story_list(data):
    file_popup("Select Story to Load", "./stories", "load_story", upload=True, jailed=True, folder_only=False, renameable=True, 
                                                                  deleteable=True, show_breadcrumbs=True, item_check=valid_story,
                                                                  valid_only=True, hide_extention=True, extra_parameter_function=get_story_length,
                                                                  column_names=['Story Name', 'Action Count'], show_filename=False,
                                                                  column_widths=['auto', '100px'],
                                                                  sort="Modified", desc=True)
                                                                  
def get_story_length(item_full_path, item, valid_selection):
    if not valid_selection:
        return ["", ""]
    with open(item_full_path, "r") as f:
        js = json.load(f)
        title = js['story_name'] if 'story_name' in js else ".".join(item.split(".")[:-1])
        if 'file_version' not in js:
            return [title, len(js['actions'])]
        if js['file_version'] == 1:
            return [title, len(js['actions'])]
        return [title, 0 if js['actions']['action_count'] == -1 else js['actions']['action_count'] ]
    

def valid_story(file):
        if file.endswith(".json"):
            with open(file, "r") as f:
                try:
                    js = json.load(f)
                except:
                    pass
                    return False
                
                return 'actions' in js

#==================================================================#
# Event triggered on load story
#==================================================================#
@socketio.on('load_story')
def UI_2_load_story(file):
    print("loading {}".format(file))
    loadRequest(file)

#==================================================================#
# Event triggered on load story
#==================================================================#
@socketio.on('new_story')
def UI_2_new_story(data):
    koboldai_vars.create_story("")
    
    
#==================================================================#
# Event triggered when user moves world info
#==================================================================#
@socketio.on('move_wi')
def UI_2_move_wi(data):
    print(data)
    if data['folder'] is None:
        koboldai_vars.worldinfo_v2.reorder(int(data['dragged_id']), int(data['drop_id']))
    else:
        koboldai_vars.worldinfo_v2.add_item_to_folder(int(data['dragged_id']), data['folder'], before=int(data['drop_id']))

#==================================================================#
# Event triggered when user moves world info
#==================================================================#
@socketio.on('wi_set_folder')
def UI_2_wi_set_folder(data):
    print(data)
    koboldai_vars.worldinfo_v2.add_item_to_folder(int(data['dragged_id']), data['folder'])

#==================================================================#
# Event triggered when user renames world info folder
#==================================================================#
@socketio.on('Rename_World_Info_Folder')
def UI_2_Rename_World_Info_Folder(data):
    print("Rename_World_Info_Folder")
    print(data)
    koboldai_vars.worldinfo_v2.rename_folder(data['old_folder'], data['new_folder'])

#==================================================================#
# Event triggered when user edits world info item
#==================================================================#
@socketio.on('edit_world_info')
def UI_2_edit_world_info(data):
    print("edit_world_info")
    print(data)
    
    if data['uid'] < 0:
        koboldai_vars.worldinfo_v2.add_item(data['title'], data['key'], 
                                             data['keysecondary'], data['folder'], 
                                             data['constant'], data['manual_text'], 
                                             data['comment'], wpp=data['wpp'], use_wpp=data['use_wpp'])
        emit("delete_new_world_info_entry", {})
    else:
        koboldai_vars.worldinfo_v2.edit_item(data['uid'], data['title'], data['key'], 
                                             data['keysecondary'], data['folder'], 
                                             data['constant'], data['manual_text'], 
                                             data['comment'], wpp=data['wpp'], use_wpp=data['use_wpp'])


#==================================================================#
# Event triggered when user edits world info item
#==================================================================#
@socketio.on('create_world_info_folder')
def UI_2_create_world_info_folder(data):
    koboldai_vars.worldinfo_v2.add_folder("New Folder")

#==================================================================#
# Event triggered when user edits world info item
#==================================================================#
@socketio.on('delete_world_info')
def UI_2_delete_world_info(uid):
    koboldai_vars.worldinfo_v2.delete(int(uid))

#==================================================================#
# Event triggered when user exports world info folder
#==================================================================#
@app.route('/export_world_info_folder')
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
def UI_2_upload_world_info_folder(data):
    json_data = json.loads(data['data'])
    koboldai_vars.worldinfo_v2.load_json(json_data, folder=data['folder'])

@socketio.on('import_world_info')
def UI_2_import_world_info(data):
    wi_data = data["data"]
    uids = {}

    for folder_name, children in wi_data["folders"].items():
        koboldai_vars.worldinfo_v2.add_folder(folder_name)
        for child in children:
            # Child is index
            if child not in uids:
                entry_data = wi_data["entries"][str(child)]
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


#==================================================================#
# Event triggered when user edits phrase biases
#==================================================================#
@socketio.on('phrase_bias_update')
def UI_2_phrase_bias_update(biases):
    print(biases)
    koboldai_vars.biases = biases



#==================================================================#
# Event triggered to rely a message
#==================================================================#
def socket_io_relay(queue, socketio):
    while True:
        if not queue.empty():
            while not queue.empty():
                data = queue.get()
                socketio.emit(data[0], data[1], **data[2])
                #socketio.emit(data[0], data[1], broadcast=True, room="UI_2")
        time.sleep(0.2)
        

#==================================================================#
# Event triggered when program errors out
#==================================================================#
def my_except_hook(exctype, value, traceback):
    print("sending error to clients")
    socketio.emit("error", "{}: {}".format(exctype, value), broadcast=True, room="UI_2")
    sys.__excepthook__(exctype, value, traceback)
sys.excepthook = my_except_hook

from werkzeug.exceptions import HTTPException

@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e

    # now you're handling non-HTTP exceptions only
    print("sending error to clients")
    socketio.emit("error", "{}: {}".format(e.message, e.args), broadcast=True, room="UI_2")
    return render_template("500_generic.html", e=e), 500


#==================================================================#
# Event triggered when Softprompt load menu is clicked
#==================================================================#
@socketio.on('load_softprompt_list')
def UI_2_load_softprompt_list(data):
    if not koboldai_vars.allowsp:
        socketio.emit("error", "Soft prompts are not supported by your current model/backend", broadcast=True, room="UI_2")
    assert koboldai_vars.allowsp, "Soft prompts are not supported by your current model/backend"
    file_popup("Select Softprompt to Load", "./softprompts", "load_softprompt", upload=True, jailed=True, folder_only=False, renameable=True, 
                                                                  deleteable=True, show_breadcrumbs=True, item_check=valid_softprompt,
                                                                  valid_only=True, hide_extention=True, extra_parameter_function=get_softprompt_desc,
                                                                  column_names=['Softprompt Name', 'Softprompt Description'],
                                                                  show_filename=False,
                                                                  column_widths=['150px', 'auto'])
                                                                
def valid_softprompt(file):
    z, version, shape, fortran_order, dtype = fileops.checksp(file, koboldai_vars.modeldim)
    if z in [1, 2, 3, 4]:
        return False
    elif not isinstance(z, zipfile.ZipFile):
        print("not zip")
        return False
    else:
        return True

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
def UI_2_load_softprompt(data):
    print("Load softprompt: {}".format(data))
    spRequest(data)

#==================================================================#
# Event triggered when load userscripts is clicked
#==================================================================#
@socketio.on('load_userscripts_list')
def UI_2_load_userscripts_list(data):
    file_popup("Select Userscripts to Load", "./userscripts", "load_userscripts", upload=True, jailed=True, folder_only=False, renameable=True, editable=True, 
                                                                  deleteable=True, show_breadcrumbs=False, item_check=valid_userscripts_to_load,
                                                                  valid_only=True, hide_extention=True, extra_parameter_function=get_userscripts_desc,
                                                                  column_names=['Module Name', 'Description'],
                                                                  show_filename=True, show_folders=False,
                                                                  column_widths=['200px', '150px', 'auto'])
                                                                
def valid_userscripts_to_load(file):
    print("{} is valid: {}".format(file, file.endswith(".lua") and os.path.basename(file) not in koboldai_vars.userscripts))
    return file.endswith(".lua") and os.path.basename(file) not in koboldai_vars.userscripts
    
def valid_userscripts_to_unload(file):
    return file.endswith(".lua") and os.path.basename(file) in koboldai_vars.userscripts

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
def UI_2_load_userscripts(data):
    print("Loading Userscripts: {}".format(os.path.basename(data)))
    koboldai_vars.userscripts = [x for x in koboldai_vars.userscripts if x != os.path.basename(data)]+[os.path.basename(data)]
    load_lua_scripts()
    
#==================================================================#
# Event triggered when userscript's are unloaded
#==================================================================#
@socketio.on('unload_userscripts')
def UI_2_unload_userscripts(data):
    print("Unloading Userscript: {}".format(data))
    koboldai_vars.userscripts = [x for x in koboldai_vars.userscripts if x != data]
    load_lua_scripts()



#==================================================================#
# Event triggered when aidg.club loaded
#==================================================================#
@socketio.on('load_aidg_club')
def UI_2_load_aidg_club(data):
    print("Load aidg.club: {}".format(data))
    importAidgRequest(data) 


#==================================================================#
# Event triggered when Theme Changed
#==================================================================#
@socketio.on('theme_change')
def UI_2_theme_change(data):
    with open("themes/{}.css".format(data['name']), "w") as f:
        f.write(":root {\n")
        for key, value in data['theme'].items():
            f.write("\t{}: {};\n".format(key, value.replace(";", "")))
        f.write("}")
    print("Theme Saved")


#==================================================================#
# Refresh SP List
#==================================================================#
@socketio.on('sp_list_refresh')
def UI_2_sp_list_refresh(data):
    koboldai_vars.splist = [[f, get_softprompt_desc(os.path.join("./softprompts", f),None,True)] for f in os.listdir("./softprompts") if os.path.isfile(os.path.join("./softprompts", f)) and valid_softprompt(os.path.join("./softprompts", f))]


#==================================================================#
# Refresh Theme List
#==================================================================#
@socketio.on('theme_list_refresh')
def UI_2_theme_list_refresh(data):
    koboldai_vars.theme_list = [".".join(f.split(".")[:-1]) for f in os.listdir("./themes") if os.path.isfile(os.path.join("./themes", f))]

#==================================================================#
# Test
#==================================================================#
@app.route("/actions")
def show_actions():
    return koboldai_vars.actions.actions
    
@app.route("/vars")
def show_vars():
    json_data = {}
    json_data['story_settings'] = json.loads(koboldai_vars.to_json("story_settings"))
    json_data['model_settings'] = json.loads(koboldai_vars.to_json("model_settings"))
    json_data['user_settings'] = json.loads(koboldai_vars.to_json("user_settings"))
    json_data['system_settings'] = json.loads(koboldai_vars.to_json("system_settings"))
    return json_data

    

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
    z, _, _, _, _ = fileops.checksp(soft_prompt.strip(), koboldai_vars.modeldim)
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

class GenerationInputSchema(SamplerSettingsSchema):
    prompt: str = fields.String(required=True, metadata={"description": "This is the submission."})
    use_memory: bool = fields.Boolean(load_default=False, metadata={"description": "Whether or not to use the memory from the KoboldAI GUI when generating text."})
    use_story: bool = fields.Boolean(load_default=False, metadata={"description": "Whether or not to use the story from the KoboldAI GUI when generating text."})
    use_authors_note: bool = fields.Boolean(load_default=False, metadata={"description": "Whether or not to use the author's note from the KoboldAI GUI when generating text. This has no effect unless `use_story` is also enabled."})
    use_world_info: bool = fields.Boolean(load_default=False, metadata={"description": "Whether or not to use the world info from the KoboldAI GUI when generating text."})
    use_userscripts: bool = fields.Boolean(load_default=False, metadata={"description": "Whether or not to use the userscripts from the KoboldAI GUI when generating text."})
    soft_prompt: Optional[str] = fields.String(metadata={"description": "Soft prompt to use when generating. If set to the empty string or any other string containing no non-whitespace characters, uses no soft prompt."}, validate=[soft_prompt_validator, validate.Regexp(r"^[^/\\]*$")])
    max_length: int = fields.Integer(validate=validate.Range(min=1, max=512), metadata={"description": "Number of tokens to generate."})
    max_context_length: int = fields.Integer(validate=validate.Range(min=512, max=2048), metadata={"description": "Maximum number of tokens to send to the model."})
    n: int = fields.Integer(validate=validate.Range(min=1, max=5), metadata={"description": "Number of outputs to generate."})
    disable_output_formatting: bool = fields.Boolean(load_default=True, metadata={"description": "When enabled, all output formatting options default to `false` instead of the value in the KoboldAI GUI."})
    frmttriminc: Optional[bool] = fields.Boolean(metadata={"description": "Output formatting option. When enabled, removes some characters from the end of the output such that the output doesn't end in the middle of a sentence. If the output is less than one sentence long, does nothing.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."})
    frmtrmblln: Optional[bool] = fields.Boolean(metadata={"description": "Output formatting option. When enabled, replaces all occurrences of two or more consecutive newlines in the output with one newline.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."})
    frmtrmspch: Optional[bool] = fields.Boolean(metadata={"description": "Output formatting option. When enabled, removes `#/@%{}+=~|\^<>` from the output.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."})
    singleline: Optional[bool] = fields.Boolean(metadata={"description": "Output formatting option. When enabled, removes everything after the first line of the output, including the newline.\n\nIf `disable_output_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."})
    disable_input_formatting: bool = fields.Boolean(load_default=True, metadata={"description": "When enabled, all input formatting options default to `false` instead of the value in the KoboldAI GUI"})
    frmtadsnsp: Optional[bool] = fields.Boolean(metadata={"description": "Input formatting option. When enabled, adds a leading space to your input if there is no trailing whitespace at the end of the previous action.\n\nIf `disable_input_formatting` is `true`, this defaults to `false` instead of the value in the KoboldAI GUI."})

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

class BasicBooleanSchema(KoboldSchema):
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

def _generate_text(body: GenerationInputSchema):
    if koboldai_vars.aibusy or koboldai_vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
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
                Explosions of suspicious origin occur at AMNAT satellite-receiver stations from Turkey to Labrador as three high-level Canadian defense ministers vanish and then a couple of days later are photographed at a Volgograd bistro hoisting shots of Stolichnaya with Slavic bimbos on their knee.
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
                       It is later established that all of the cabinet members have died of old age.
                      MEGAMATRIX becomes involved in the growing number of mass abductions and kidnappings. Many disappearances occur along highways in western Canada, usually when traffic has come to a standstill because of a stalled truck or snowstorm. One or two abducted individuals will be released within a day or so but never
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
      tags:
        - model
      requestBody:
        required: true
        content:
          application/json:
            schema: ModelSelectionSchema
            example:
              model: ReadOnly
      responses:
        200:
          description: Successful request
          content:
            application/json:
              schema: EmptySchema
        {api_validation_error_response}
        {api_server_busy_response}
    """
    if vars.aibusy or vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
    set_aibusy(1)
    old_model = vars.model
    vars.model = body.model.strip()
    try:
        load_model(use_breakmodel_args=True, breakmodel_args_default_to_cpu=True)
    except Exception as e:
        vars.model = old_model
        raise e
    set_aibusy(0)
    return {}


def prompt_validator(prompt: str):
    if len(prompt.strip()) == 0:
        raise ValidationError("String does not match expected pattern.")

class SubmissionInputSchema(KoboldSchema):
    prompt: str = fields.String(required=True, validate=prompt_validator, metadata={"pattern": r"^.*\S.*$", "description": "This is the submission."})
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
        actionsubmit(body.prompt, force_submit=True, no_generate=True)
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
    for num, action in koboldai_vars.actions.items():
        chunks.append({"num": num + 1, "text": action})
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
    for num in koboldai_vars.actions.keys():
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
              schema: BasicBooleanSchema
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
              schema: BasicBooleanSchema
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
              schema: BasicBooleanSchema
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
              schema: BasicBooleanSchema
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
              schema: BasicBooleanSchema
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
              schema: BasicBooleanSchema
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



for schema in config_endpoint_schemas:
    create_config_endpoint(schema=schema.__name__, method="GET")
    create_config_endpoint(schema=schema.__name__, method="PUT")


#==================================================================#
#  Final startup commands to launch Flask app
#==================================================================#
@app.before_first_request
def startup():
    if koboldai_vars.model == "" or koboldai_vars.model is None:
        koboldai_vars.model = "ReadOnly"
    socketio.start_background_task(load_model, **{'initial_load':True})
            
print("", end="", flush=True)
if __name__ == "__main__":
    print("{0}\nStarting webserver...{1}".format(colors.GREEN, colors.END), flush=True)

    general_startup()
    patch_transformers()
    # Start Flask/SocketIO (Blocking, so this must be last method!)
    port = args.port if "port" in args and args.port is not None else 5000
    koboldai_settings.port = port
    
    if(koboldai_vars.host):
        if(args.localtunnel):
            import subprocess, shutil
            localtunnel = subprocess.Popen([shutil.which('lt'), '-p', str(port), 'http'], stdout=subprocess.PIPE)
            attempts = 0
            while attempts < 10:
                try:
                    cloudflare = str(localtunnel.stdout.readline())
                    cloudflare = (re.search("(?P<url>https?:\/\/[^\s]+loca.lt)", cloudflare).group("url"))
                    break
                except:
                    attempts += 1
                    time.sleep(3)
                    continue
            if attempts == 10:
                print("LocalTunnel could not be created, falling back to cloudflare...")
                from flask_cloudflared import _run_cloudflared
                cloudflare = _run_cloudflared(port)
        elif(args.ngrok):
            from flask_ngrok import _run_ngrok
            cloudflare = _run_ngrok()
        elif(args.remote):
           from flask_cloudflared import _run_cloudflared
           cloudflare = _run_cloudflared(port)
        if(args.localtunnel or args.ngrok or args.remote):
            with open('cloudflare.log', 'w') as cloudflarelog:
                cloudflarelog.write("KoboldAI has finished loading and is available at the following link : " + cloudflare)
                print(format(colors.GREEN) + "KoboldAI has finished loading and is available at the following link for UI 1: " + cloudflare + format(colors.END))
                print(format(colors.GREEN) + "KoboldAI has finished loading and is available at the following link for UI 2: " + cloudflare + "/new_ui" + format(colors.END))
        else:
            print("{0}Webserver has started, you can now connect to this machine at port {1}{2}"
                  .format(colors.GREEN, port, colors.END))
        koboldai_vars.serverstarted = True
        
        socketio.run(app, host='0.0.0.0', port=port)
    else:
        if args.unblock:
            if not args.no_ui:
                import webbrowser
                webbrowser.open_new('http://localhost:{0}'.format(port))
            print("{0}Server started!\nYou may now connect with a browser at http://127.0.0.1:{1}/{2}"
                  .format(colors.GREEN, port, colors.END))
            koboldai_vars.serverstarted = True
            socketio.run(app, port=port, host='0.0.0.0')
        else:
            try:
                from flaskwebgui import FlaskUI
                koboldai_vars.serverstarted = True
                koboldai_vars.flaskwebgui = True
                FlaskUI(app, socketio=socketio, start_server="flask-socketio", maximized=True, close_server_on_exit=True).run()
            except:
                if not args.no_ui:
                    import webbrowser
                    webbrowser.open_new('http://localhost:{0}'.format(port))
                print("{0}Server started!\nYou may now connect with a browser at http://127.0.0.1:{1}/{2}"
                        .format(colors.GREEN, port, colors.END))
                koboldai_vars.serverstarted = True
                socketio.run(app, port=port)

else:
    general_startup()
    patch_transformers()
    startup()
    koboldai_settings.port = args.port if "port" in args and args.port is not None else 5000
    print("{0}\nServer started in WSGI mode!{1}".format(colors.GREEN, colors.END), flush=True)
    
