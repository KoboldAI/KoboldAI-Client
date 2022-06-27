#!/usr/bin/python3
#==================================================================#
# KoboldAI
# Version: 1.18.1
# By: KoboldAIDev and the KoboldAI Community
#==================================================================#

# External packages
import eventlet
from eventlet import tpool
eventlet.monkey_patch(all=True, thread=False)
#eventlet.monkey_patch(os=True, select=True, socket=True, thread=True, time=True, psycopg=True)
import os
os.system("")
__file__ = os.path.dirname(os.path.realpath(__file__))
os.chdir(__file__)
os.environ['EVENTLET_THREADPOOL_SIZE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)

from os import path, getcwd
import time
import re
import json
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
from collections.abc import Iterable
from typing import Any, Callable, TypeVar, Tuple, Union, Dict, Set, List

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
# Each item takes the 4 elements, 1: Text to display, 2: Model Name (model_settings.model) or menu name (Key name for another menu),
# 3: the memory requirement for the model, 4: if the item is a menu or not (True/False)
model_menu = {
    'mainmenu': [
        ["Load a model from its directory", "NeoCustom", "", False],
        ["Load an old GPT-2 model (eg CloverEdition)", "GPT2Custom", "", False],
        ["Adventure Models", "adventurelist", "", True],
        ["Novel Models", "novellist", "", True],
        ["NSFW Models", "nsfwlist", "", True],
        ["Untuned GPT-Neo/J", "gptneolist", "", True],
        ["Untuned Fairseq Dense", "fsdlist", "", True],
        ["Untuned OPT", "optlist", "", True],
        ["Untuned XGLM", "xglmlist", "", True],
        ["Untuned GPT2", "gpt2list", "", True],
        ["Online Services", "apilist", "", True],
        ["Read Only (No AI)", "ReadOnly", "", False]
        ],
    'adventurelist': [
        ["Nerys FSD 13B (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys", "32GB", False],
        ["Skein 6B", "KoboldAI/GPT-J-6B-Skein", "16GB", False],
        ["Adventure 6B", "KoboldAI/GPT-J-6B-Adventure", "16GB", False],
        ["Nerys FSD 2.7B (Hybrid)", "KoboldAI/fairseq-dense-2.7B-Nerys", "8GB", False],
        ["Adventure 2.7B", "KoboldAI/GPT-Neo-2.7B-AID", "8GB", False],
        ["Adventure 1.3B", "KoboldAI/GPT-Neo-1.3B-Adventure", "6GB", False],
        ["Adventure 125M (Mia)", "Merry/AID-Neo-125M", "2GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'novellist': [
        ["Nerys FSD 13B (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys", "32GB", False],
        ["Janeway FSD 13B", "KoboldAI/fairseq-dense-13B-Janeway", "32GB", False],
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
        ["KoboldAI Server API (Old Google Colab)", "Colab", "", False],
        ["Return to Main Menu", "mainmenu", "", True],
    ]
    }

model_settings = koboldai_settings.model_settings()
story_settings = koboldai_settings.story_settings()
user_settings = koboldai_settings.user_settings()
system_settings = koboldai_settings.system_settings()

utils.model_settings = model_settings
utils.story_settings = story_settings
utils.user_settings = user_settings
utils.system_settings = system_settings

class Send_to_socketio(object):
    def write(self, bar):
        print(bar, end="")
        time.sleep(0.01)
        try:
            emit('from_server', {'cmd': 'model_load_status', 'data': bar.replace(" ", "&nbsp;")}, broadcast=True, room="UI_1")
        except:
            pass
                                
# Set logging level to reduce chatter from Flask
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Start flask & SocketIO
print("{0}Initializing Flask... {1}".format(colors.PURPLE, colors.END), end="")
from flask import Flask, render_template, Response, request, copy_current_request_context, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
app = Flask(__name__, root_path=os.getcwd())
app.config['SECRET KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app, async_method="eventlet")
#socketio = SocketIO(app, async_method="eventlet", logger=True, engineio_logger=True)
koboldai_settings.socketio = socketio
print("{0}OK!{1}".format(colors.GREEN, colors.END))

#==================================================================#
# Function to get model selection at startup
#==================================================================#
def sendModelSelection(menu="mainmenu", folder="./models"):
    #If we send one of the manual load options, send back the list of model directories, otherwise send the menu
    if menu in ('NeoCustom', 'GPT2Custom'):
        (paths, breadcrumbs) = get_folder_path_info(folder)
        if system_settings.host:
            breadcrumbs = []
        menu_list = [[folder, menu, "", False] for folder in paths]
        menu_list.append(["Return to Main Menu", "mainmenu", "", True])
        if os.path.abspath("{}/models".format(os.getcwd())) == os.path.abspath(folder):
            showdelete=True
        else:
            showdelete=False
        emit('from_server', {'cmd': 'show_model_menu', 'data': menu_list, 'menu': menu, 'breadcrumbs': breadcrumbs, "showdelete": showdelete}, broadcast=True, room="UI_1")
    else:
        emit('from_server', {'cmd': 'show_model_menu', 'data': model_menu[menu], 'menu': menu, 'breadcrumbs': [], "showdelete": False}, broadcast=True, room="UI_1")

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
    model_settings.model = ''
    while(model_settings.model == ''):
        modelsel = input("Model #> ")
        if(modelsel.isnumeric() and int(modelsel) > 0 and int(modelsel) <= len(modellist)):
            model_settings.model = modellist[int(modelsel)-1][1]
        else:
            print("{0}Please enter a valid selection.{1}".format(colors.RED, colors.END))
    
    # Model Lists
    try:
        getModelSelection(eval(model_settings.model))
    except Exception as e:
        if(model_settings.model == "Return"):
            getModelSelection(mainmenu)
                
        # If custom model was selected, get the filesystem location and store it
        if(model_settings.model == "NeoCustom" or model_settings.model == "GPT2Custom"):
            print("{0}Please choose the folder where pytorch_model.bin is located:{1}\n".format(colors.CYAN, colors.END))
            modpath = fileops.getdirpath(getcwd() + "/models", "Select Model Folder")
        
            if(modpath):
                # Save directory to vars
                model_settings.custmodpth = modpath
            else:
                # Print error and retry model selection
                print("{0}Model select cancelled!{1}".format(colors.RED, colors.END))
                print("{0}Select an AI model to continue:{1}\n".format(colors.CYAN, colors.END))
                getModelSelection(mainmenu)

def check_if_dir_is_model(path):
    if os.path.exists(path):
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(path, revision=model_settings.revision, cache_dir="cache")
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
    if(model_settings.model in ("NeoCustom", "GPT2Custom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        modelname = os.path.basename(os.path.normpath(model_settings.custmodpth))
        return modelname
    else:
        modelname = model_settings.model
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
    if(args.breakmodel_gpulayers is not None or (utils.HAS_ACCELERATE and args.breakmodel_disklayers is not None)):
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
        system_settings.breakmodel = False
        system_settings.usegpu = True
        system_settings.gpu_device = len(breakmodel.gpu_blocks)-1
        return

    if(not breakmodel.gpu_blocks):
        print("Nothing assigned to a GPU, reverting to CPU only mode")
        import breakmodel
        breakmodel.primary_device = "cpu"
        system_settings.breakmodel = False
        system_settings.usegpu = False
        return

def move_model_to_devices(model):
    global generator

    if(not utils.HAS_ACCELERATE and not system_settings.breakmodel):
        if(system_settings.usegpu):
            model = model.half().to(system_settings.gpu_device)
        else:
            model = model.to('cpu').float()
        generator = model.generate
        return

    import breakmodel

    if(utils.HAS_ACCELERATE):
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
                js   = json.load(open(model_settings.custmodpth + "/config.json", "r"))
            except Exception as e:
                js   = json.load(open(model_settings.custmodpth.replace('/', '_') + "/config.json", "r"))            
        except Exception as e:
            js   = {}
    if model_settings.model_type == "xglm" or js.get("compat", "j") == "fairseq_lm":
        model_settings.newlinemode = "s"  # Default to </s> newline mode if using XGLM
    if model_settings.model_type == "opt" or model_settings.model_type == "bloom":
        model_settings.newlinemode = "ns"  # Handle </s> but don't convert newlines if using Fairseq models that have newlines trained in them
    model_settings.modelconfig = js
    if("badwordsids" in js):
        model_settings.badwordsids = js["badwordsids"]
    if("nobreakmodel" in js):
        system_settings.nobreakmodel = js["nobreakmodel"]
    if("sampler_order" in js):
        model_settings.sampler_order = js["sampler_order"]
    if("temp" in js):
        model_settings.temp       = js["temp"]
    if("top_p" in js):
        model_settings.top_p      = js["top_p"]
    if("top_k" in js):
        model_settings.top_k      = js["top_k"]
    if("tfs" in js):
        model_settings.tfs        = js["tfs"]
    if("typical" in js):
        model_settings.typical    = js["typical"]
    if("top_a" in js):
        model_settings.top_a      = js["top_a"]
    if("rep_pen" in js):
        model_settings.rep_pen    = js["rep_pen"]
    if("rep_pen_slope" in js):
        model_settings.rep_pen_slope = js["rep_pen_slope"]
    if("rep_pen_range" in js):
        model_settings.rep_pen_range = js["rep_pen_range"]
    if("adventure" in js):
        story_settings.adventure = js["adventure"]
    if("chatmode" in js):
        story_settings.chatmode = js["chatmode"]
    if("dynamicscan" in js):
        story_settings.dynamicscan = js["dynamicscan"]
    if("formatoptns" in js):
        user_settings.formatoptns = js["formatoptns"]
    if("welcome" in js):
        system_settings.welcome = js["welcome"]
    if("newlinemode" in js):
        model_settings.newlinemode = js["newlinemode"]
    if("antemplate" in js):
        story_settings.setauthornotetemplate = js["antemplate"]
        if(not story_settings.gamestarted):
            story_settings.authornotetemplate = story_settings.setauthornotetemplate

#==================================================================#
#  Take settings from vars and write them to client settings file
#==================================================================#
def savesettings():
     # Build json to write
    js = {}
    js["apikey"]      = model_settings.apikey
    js["andepth"]     = story_settings.andepth
    js["sampler_order"] = model_settings.sampler_order
    js["temp"]        = model_settings.temp
    js["top_p"]       = model_settings.top_p
    js["top_k"]       = model_settings.top_k
    js["tfs"]         = model_settings.tfs
    js["typical"]     = model_settings.typical
    js["top_a"]       = model_settings.top_a
    js["rep_pen"]     = model_settings.rep_pen
    js["rep_pen_slope"] = model_settings.rep_pen_slope
    js["rep_pen_range"] = model_settings.rep_pen_range
    js["genamt"]      = model_settings.genamt
    js["max_length"]  = model_settings.max_length
    js["ikgen"]       = model_settings.ikgen
    js["formatoptns"] = user_settings.formatoptns
    js["numseqs"]     = model_settings.numseqs
    js["widepth"]     = user_settings.widepth
    js["useprompt"]   = story_settings.useprompt
    js["adventure"]   = story_settings.adventure
    js["chatmode"]    = story_settings.chatmode
    js["chatname"]    = story_settings.chatname
    js["dynamicscan"] = story_settings.dynamicscan
    js["nopromptgen"] = user_settings.nopromptgen
    js["rngpersist"]  = user_settings.rngpersist
    js["nogenmod"]    = user_settings.nogenmod
    js["autosave"]    = user_settings.autosave
    js["welcome"]     = system_settings.welcome
    js["newlinemode"] = model_settings.newlinemode

    js["antemplate"]  = story_settings.setauthornotetemplate

    js["userscripts"] = system_settings.userscripts
    js["corescript"]  = system_settings.corescript
    js["softprompt"]  = system_settings.spfilename

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
#  Read settings from client file JSON and send to vars
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
# Copy file contents to vars
    if("apikey" in js):
        model_settings.apikey     = js["apikey"]
    if("andepth" in js):
        story_settings.andepth    = js["andepth"]
    if("sampler_order" in js):
        model_settings.sampler_order = js["sampler_order"]
    if("temp" in js):
        model_settings.temp       = js["temp"]
    if("top_p" in js):
        model_settings.top_p      = js["top_p"]
    if("top_k" in js):
        model_settings.top_k      = js["top_k"]
    if("tfs" in js):
        model_settings.tfs        = js["tfs"]
    if("typical" in js):
        model_settings.typical    = js["typical"]
    if("top_a" in js):
        model_settings.top_a      = js["top_a"]
    if("rep_pen" in js):
        model_settings.rep_pen    = js["rep_pen"]
    if("rep_pen_slope" in js):
        model_settings.rep_pen_slope = js["rep_pen_slope"]
    if("rep_pen_range" in js):
        model_settings.rep_pen_range = js["rep_pen_range"]
    if("genamt" in js):
        model_settings.genamt     = js["genamt"]
    if("max_length" in js):
        model_settings.max_length = js["max_length"]
    if("ikgen" in js):
        model_settings.ikgen      = js["ikgen"]
    if("formatoptns" in js):
        user_settings.formatoptns = js["formatoptns"]
    if("numseqs" in js):
        model_settings.numseqs = js["numseqs"]
    if("widepth" in js):
        user_settings.widepth = js["widepth"]
    if("useprompt" in js):
        story_settings.useprompt = js["useprompt"]
    if("adventure" in js):
        story_settings.adventure = js["adventure"]
    if("chatmode" in js):
        story_settings.chatmode = js["chatmode"]
    if("chatname" in js):
        story_settings.chatname = js["chatname"]
    if("dynamicscan" in js):
        story_settings.dynamicscan = js["dynamicscan"]
    if("nopromptgen" in js):
        user_settings.nopromptgen = js["nopromptgen"]
    if("rngpersist" in js):
        user_settings.rngpersist = js["rngpersist"]
    if("nogenmod" in js):
        user_settings.nogenmod = js["nogenmod"]
    if("autosave" in js):
        user_settings.autosave = js["autosave"]
    if("newlinemode" in js):
        model_settings.newlinemode = js["newlinemode"]
    if("welcome" in js):
        system_settings.welcome = js["welcome"]

    if("antemplate" in js):
        story_settings.setauthornotetemplate = js["antemplate"]
        if(not story_settings.gamestarted):
            story_settings.authornotetemplate = story_settings.setauthornotetemplate
    
    if("userscripts" in js):
        system_settings.userscripts = []
        for userscript in js["userscripts"]:
            if type(userscript) is not str:
                continue
            userscript = userscript.strip()
            if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                system_settings.userscripts.append(userscript)

    if("corescript" in js and type(js["corescript"]) is str and all(q not in js["corescript"] for q in ("..", ":")) and all(js["corescript"][0] not in q for q in ("/", "\\"))):
        system_settings.corescript = js["corescript"]
    else:
        system_settings.corescript = "default.lua"

#==================================================================#
#  Load a soft prompt from a file
#==================================================================#

def check_for_sp_change():
    while(True):
        time.sleep(0.1)
        if(system_settings.sp_changed):
            with app.app_context():
                emit('from_server', {'cmd': 'spstatitems', 'data': {system_settings.spfilename: system_settings.spmeta} if system_settings.allowsp and len(system_settings.spfilename) else {}}, namespace=None, broadcast=True, room="UI_1")
            system_settings.sp_changed = False

socketio.start_background_task(check_for_sp_change)

def spRequest(filename):
    if(not system_settings.allowsp):
        raise RuntimeError("Soft prompts are not supported by your current model/backend")
    
    old_filename = system_settings.spfilename

    system_settings.spfilename = ""
    settingschanged()

    if(len(filename) == 0):
        system_settings.sp = None
        system_settings.sp_length = 0
        if(old_filename != filename):
            system_settings.sp_changed = True
        return

    global np
    if 'np' not in globals():
        import numpy as np

    z, version, shape, fortran_order, dtype = fileops.checksp(filename, model_settings.modeldim)
    if not isinstance(z, zipfile.ZipFile):
        raise RuntimeError(f"{repr(filename)} is not a valid soft prompt file")
    with z.open('meta.json') as f:
        system_settings.spmeta = json.load(f)
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

    system_settings.sp_length = tensor.shape[-2]
    system_settings.spmeta["n_tokens"] = system_settings.sp_length

    if(system_settings.use_colab_tpu or model_settings.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        rows = tensor.shape[0]
        padding_amount = tpu_mtj_backend.params["seq"] - (tpu_mtj_backend.params["seq"] % -tpu_mtj_backend.params["cores_per_replica"]) - rows
        tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
        tensor = tensor.reshape(
            tpu_mtj_backend.params["cores_per_replica"],
            -1,
            tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"]),
        )
        system_settings.sp = tpu_mtj_backend.shard_xmap(np.float32(tensor))
    else:
        system_settings.sp = torch.from_numpy(tensor)

    system_settings.spfilename = filename
    settingschanged()
    if(old_filename != filename):
            system_settings.sp_changed = True

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

    model_settings.model = args.model;
    model_settings.revision = args.revision

    if args.colab:
        args.remote = True;
        args.override_rename = True;
        args.override_delete = True;
        args.nobreakmodel = True;
        args.quiet = True;
        args.lowmem = True;
        args.noaimenu = True;

    if args.quiet:
        system_settings.quiet = True

    if args.nobreakmodel:
        system_settings.nobreakmodel = True;

    if args.remote:
        system_settings.host = True;

    if args.ngrok:
        system_settings.host = True;

    if args.localtunnel:
        system_settings.host = True;

    if args.host:
        system_settings.host = True;

    if args.cpu:
        system_settings.use_colab_tpu = False

    system_settings.smandelete = system_settings.host == args.override_delete
    system_settings.smanrename = system_settings.host == args.override_rename

    system_settings.aria2_port = args.aria2_port or 6799
    
    #Now let's look to see if we are going to force a load of a model from a user selected folder
    if(model_settings.model == "selectfolder"):
        print("{0}Please choose the folder where pytorch_model.bin is located:{1}\n".format(colors.CYAN, colors.END))
        modpath = fileops.getdirpath(getcwd() + "/models", "Select Model Folder")
    
        if(modpath):
            # Save directory to vars
            model_settings.model = "NeoCustom"
            model_settings.custmodpth = modpath
    elif args.model:
        print("Welcome to KoboldAI!\nYou have selected the following Model:", model_settings.model)
        if args.path:
            print("You have selected the following path for your Model :", args.path)
            model_settings.custmodpth = args.path;
            model_settings.colaburl = args.path + "/request"; # Lets just use the same parameter to keep it simple
#==================================================================#
# Load Model
#==================================================================# 

def tpumtjgetsofttokens():
    soft_tokens = None
    if(system_settings.sp is None):
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
        system_settings.sp = tpu_mtj_backend.shard_xmap(tensor)
    soft_tokens = np.arange(
        tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"],
        tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"] + system_settings.sp_length,
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
    if model in [x[1] for x in model_menu['apilist']]:
        if path.exists("settings/{}.settings".format(model)):
            with open("settings/{}.settings".format(model), "r") as file:
                # Check if API key exists
                js = json.load(file)
                if("apikey" in js and js["apikey"] != ""):
                    # API key exists, grab it and close the file
                    key_value = js["apikey"]
                elif 'oaiapikey' in js and js['oaiapikey'] != "":
                    key_value = js["oaiapikey"]
        key = True
    elif model == 'ReadOnly':
        pass
    elif model == 'Colab':
        url = True
    elif not utils.HAS_ACCELERATE and not torch.cuda.is_available():
        pass
    else:
        layer_count = get_layer_count(model, directory=directory)
        if layer_count is None:
            breakmodel = False
        else:
            breakmodel = True
            if model in ["NeoCustom", "GPT2Custom"]:
                filename = os.path.basename(os.path.normpath(directory))
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
            break_values += [0] * (gpu_count - len(break_values))
    emit('from_server', {'cmd': 'selected_model_info', 'key_value': key_value, 'key':key, 
                         'gpu':gpu, 'layer_count':layer_count, 'breakmodel':breakmodel, 
                         'disk_break_value': disk_blocks, 'accelerate': utils.HAS_ACCELERATE,
                         'break_values': break_values, 'gpu_count': gpu_count,
                         'url': url, 'gpu_names': gpu_names}, broadcast=True, room="UI_1")
    if key_value != "":
        get_oai_models(key_value)
    

def get_layer_count(model, directory=""):
    if(model not in ["InferKit", "Colab", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ"]):
        if(model_settings.model == "GPT2Custom"):
            model_config = open(model_settings.custmodpth + "/config.json", "r")
        # Get the model_type from the config or assume a model type if it isn't present
        else:
            from transformers import AutoConfig
            if directory == "":
                model_config = AutoConfig.from_pretrained(model_settings.model, revision=model_settings.revision, cache_dir="cache")
            elif(os.path.isdir(model_settings.custmodpth.replace('/', '_'))):
                model_config = AutoConfig.from_pretrained(model_settings.custmodpth.replace('/', '_'), revision=model_settings.revision, cache_dir="cache")
            elif(os.path.isdir(directory)):
                model_config = AutoConfig.from_pretrained(directory, revision=model_settings.revision, cache_dir="cache")
            else:
                model_config = AutoConfig.from_pretrained(model_settings.custmodpth, revision=model_settings.revision, cache_dir="cache")
        
        
        
        return utils.num_layers(model_config)
    else:
        return None


def get_oai_models(key):
    model_settings.oaiapikey = key
    if model_settings.model == 'OAI':
        url = "https://api.openai.com/v1/engines"
    elif model_settings.model == 'GooseAI':
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
        if path.exists("settings/{}.settings".format(model_settings.model)):
            with open("settings/{}.settings".format(model_settings.model), "r") as file:
                js = json.load(file)
                if 'online_model' in js:
                    online_model = js['online_model']
                if "apikey" in js:
                    if js['apikey'] != key:
                        changed=True
        if changed:
            with open("settings/{}.settings".format(model_settings.model), "w") as file:
                js["apikey"] = key
                file.write(json.dumps(js, indent=3), room="UI_1")
            
        emit('from_server', {'cmd': 'oai_engines', 'data': engines, 'online_model': online_model}, broadcast=True)
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
        if(system_settings.sp is not None):
            shifted_input_ids = input_ids - model.config.vocab_size
        input_ids.clamp_(max=model.config.vocab_size-1)
        inputs_embeds = old_embedding_call(self, input_ids, *args, **kwargs)
        if(system_settings.sp is not None):
            system_settings.sp = system_settings.sp.to(inputs_embeds.dtype).to(inputs_embeds.device)
            inputs_embeds = torch.where(
                (shifted_input_ids >= 0)[..., None],
                system_settings.sp[shifted_input_ids.clamp(min=0)],
                inputs_embeds,
            )
        return inputs_embeds
    Embedding.__call__ = new_embedding_call
    Embedding._koboldai_patch_causallm_model = model
    return model


def patch_transformers():
    global transformers
    old_from_pretrained = PreTrainedModel.from_pretrained.__func__
    @classmethod
    def new_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_settings.fp32_model = False
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
                    conds.append(getattr(model_settings, v))
                    setattr(self, f, conds[-1])
            else:
                conds = getattr(model_settings, var_name)
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
    RepetitionPenaltyLogitsProcessor.__init__ = AdvancedRepetitionPenaltyLogitsProcessor.__init__
    RepetitionPenaltyLogitsProcessor.__call__ = AdvancedRepetitionPenaltyLogitsProcessor.__call__

    class LuaLogitsProcessor(LogitsProcessor):

        def __init__(self):
            pass

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            assert scores.ndim == 2
            assert input_ids.ndim == 2
            self.regeneration_required = False
            self.halt = False

            scores_shape = scores.shape
            scores_list = scores.tolist()
            system_settings.lua_koboldbridge.logits = system_settings.lua_state.table()
            for r, row in enumerate(scores_list):
                system_settings.lua_koboldbridge.logits[r+1] = system_settings.lua_state.table(*row)
            system_settings.lua_koboldbridge.vocab_size = scores_shape[-1]

            execute_genmod()

            scores = torch.tensor(
                tuple(tuple(row.values()) for row in system_settings.lua_koboldbridge.logits.values()),
                device=scores.device,
                dtype=scores.dtype,
            )
            assert scores.shape == scores_shape

            return scores
    
    def new_get_logits_processor(*args, **kwargs) -> LogitsProcessorList:
        processors = new_get_logits_processor.old_get_logits_processor(*args, **kwargs)
        processors.insert(0, LuaLogitsProcessor())
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

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, *args, **kwargs):
            for k in model_settings.sampler_order:
                scores = self.__warper_list[k](input_ids, scores, *args, **kwargs)
            return scores

    def new_get_logits_warper(beams: int = 1,) -> LogitsProcessorList:
        return KoboldLogitsWarperList(beams=beams)
    
    def new_sample(self, *args, **kwargs):
        assert kwargs.pop("logits_warper", None) is not None
        kwargs["logits_warper"] = new_get_logits_warper(
            beams=1,
        )
        if(model_settings.newlinemode == "s") or (model_settings.newlinemode == "ns"):
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
            model_settings.generated_tkns += 1
            if(system_settings.lua_koboldbridge.generated_cols and model_settings.generated_tkns != system_settings.lua_koboldbridge.generated_cols):
                raise RuntimeError(f"Inconsistency detected between KoboldAI Python and Lua backends ({model_settings.generated_tkns} != {system_settings.lua_koboldbridge.generated_cols})")
            if(system_settings.abort or model_settings.generated_tkns >= model_settings.genamt):
                self.regeneration_required = False
                self.halt = False
                return True

            assert input_ids.ndim == 2
            assert len(self.excluded_world_info) == input_ids.shape[0]
            self.regeneration_required = system_settings.lua_koboldbridge.regeneration_required
            self.halt = not system_settings.lua_koboldbridge.generating
            system_settings.lua_koboldbridge.regeneration_required = False

            for i in range(model_settings.numseqs):
                system_settings.lua_koboldbridge.generated[i+1][model_settings.generated_tkns] = int(input_ids[i, -1].item())

            if(not story_settings.dynamicscan):
                return self.regeneration_required or self.halt
            tail = input_ids[..., -model_settings.generated_tkns:]
            for i, t in enumerate(tail):
                decoded = utils.decodenewlines(tokenizer.decode(t))
                _, found = checkworldinfo(decoded, force_use_txt=True, actions=story_settings._actions)
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
        stopping_criteria.insert(0, self.kai_scanner)
        return stopping_criteria
    transformers.generation_utils.GenerationMixin._get_stopping_criteria = new_get_stopping_criteria

def load_model(use_gpu=True, gpu_layers=None, disk_layers=None, initial_load=False, online_model=""):
    global model
    global generator
    global torch
    global model_config
    global GPT2TokenizerFast
    global tokenizer
    if not utils.HAS_ACCELERATE:
        disk_layers = None
    system_settings.noai = False
    if not initial_load:
        set_aibusy(True)
        if model_settings.model != 'ReadOnly':
            emit('from_server', {'cmd': 'model_load_status', 'data': "Loading {}".format(model_settings.model)}, broadcast=True, room="UI_1")
            #Have to add a sleep so the server will send the emit for some reason
            time.sleep(0.1)
    if gpu_layers is not None:
        args.breakmodel_gpulayers = gpu_layers
    if disk_layers is not None:
        args.breakmodel_disklayers = int(disk_layers)
    
    #We need to wipe out the existing model and refresh the cuda cache
    model = None
    generator = None
    model_config = None
    for tensor in gc.get_objects():
        try:
            if torch.is_tensor(tensor):
                with torch.no_grad():
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
    model_settings.badwordsids = koboldai_settings.badwordsids_default
    
    #Let's set the GooseAI or OpenAI server URLs if that's applicable
    if online_model != "":
        if path.exists("settings/{}.settings".format(model_settings.model)):
            changed=False
            with open("settings/{}.settings".format(model_settings.model), "r") as file:
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
                with open("settings/{}.settings".format(model_settings.model), "w") as file:
                    file.write(json.dumps(js, indent=3))
        # Swap OAI Server if GooseAI was selected
        if(model_settings.model == "GooseAI"):
            model_settings.oaiengines = "https://api.goose.ai/v1/engines"
            model_settings.model = "OAI"
            args.configname = "GooseAI" + "/" + online_model
        else:
            args.configname = model_settings.model + "/" + online_model
        model_settings.oaiurl = model_settings.oaiengines + "/{0}/completions".format(online_model)
    
    
    # If transformers model was selected & GPU available, ask to use CPU or GPU
    if(model_settings.model not in ["InferKit", "Colab", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        system_settings.allowsp = True
        # Test for GPU support
        
        # Make model path the same as the model name to make this consistent with the other loading method if it isn't a known model type
        # This code is not just a workaround for below, it is also used to make the behavior consistent with other loading methods - Henk717
        if(not model_settings.model in ["NeoCustom", "GPT2Custom"]):
            model_settings.custmodpth = model_settings.model
        elif(model_settings.model == "NeoCustom"):
            model_settings.model = os.path.basename(os.path.normpath(model_settings.custmodpth))

        # Get the model_type from the config or assume a model type if it isn't present
        from transformers import AutoConfig
        if(os.path.isdir(model_settings.custmodpth.replace('/', '_'))):
            try:
                model_config = AutoConfig.from_pretrained(model_settings.custmodpth.replace('/', '_'), revision=model_settings.revision, cache_dir="cache")
                model_settings.model_type = model_config.model_type
            except ValueError as e:
                model_settings.model_type = "not_found"
        elif(os.path.isdir("models/{}".format(model_settings.custmodpth.replace('/', '_')))):
            try:
                model_config = AutoConfig.from_pretrained("models/{}".format(model_settings.custmodpth.replace('/', '_')), revision=model_settings.revision, cache_dir="cache")
                model_settings.model_type = model_config.model_type
            except ValueError as e:
                model_settings.model_type = "not_found"
        else:
            try:
                model_config = AutoConfig.from_pretrained(model_settings.custmodpth, revision=model_settings.revision, cache_dir="cache")
                model_settings.model_type = model_config.model_type
            except ValueError as e:
                model_settings.model_type = "not_found"
        if(model_settings.model_type == "not_found" and model_settings.model == "NeoCustom"):
            model_settings.model_type = "gpt_neo"
        elif(model_settings.model_type == "not_found" and model_settings.model == "GPT2Custom"):
            model_settings.model_type = "gpt2"
        elif(model_settings.model_type == "not_found"):
            print("WARNING: No model type detected, assuming Neo (If this is a GPT2 model use the other menu option or --model GPT2Custom)")
            model_settings.model_type = "gpt_neo"

    if(not system_settings.use_colab_tpu and model_settings.model not in ["InferKit", "Colab", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        loadmodelsettings()
        loadsettings()
        print("{0}Looking for GPU support...{1}".format(colors.PURPLE, colors.END), end="")
        system_settings.hascuda = torch.cuda.is_available()
        system_settings.bmsupported = (utils.HAS_ACCELERATE or model_settings.model_type in ("gpt_neo", "gptj", "xglm", "opt")) and not system_settings.nobreakmodel
        if(args.breakmodel is not None and args.breakmodel):
            print("WARNING: --breakmodel is no longer supported. Breakmodel mode is now automatically enabled when --breakmodel_gpulayers is used (see --help for details).", file=sys.stderr)
        if(args.breakmodel_layers is not None):
            print("WARNING: --breakmodel_layers is deprecated. Use --breakmodel_gpulayers instead (see --help for details).", file=sys.stderr)
        if(args.model and system_settings.bmsupported and not args.breakmodel_gpulayers and not args.breakmodel_layers and (not utils.HAS_ACCELERATE or not args.breakmodel_disklayers)):
            print("WARNING: Model launched without the --breakmodel_gpulayers argument, defaulting to GPU only mode.", file=sys.stderr)
            system_settings.bmsupported = False
        if(not system_settings.bmsupported and (args.breakmodel_gpulayers is not None or args.breakmodel_layers is not None or args.breakmodel_disklayers is not None)):
            print("WARNING: This model does not support hybrid generation. --breakmodel_gpulayers will be ignored.", file=sys.stderr)
        if(system_settings.hascuda):
            print("{0}FOUND!{1}".format(colors.GREEN, colors.END))
        else:
            print("{0}NOT FOUND!{1}".format(colors.YELLOW, colors.END))
        
        if args.model:
            if(system_settings.hascuda):
                genselected = True
                system_settings.usegpu = True
                system_settings.breakmodel = utils.HAS_ACCELERATE
            if(system_settings.bmsupported):
                system_settings.usegpu = False
                system_settings.breakmodel = True
            if(args.cpu):
                system_settings.usegpu = False
                system_settings.breakmodel = utils.HAS_ACCELERATE
        elif(system_settings.hascuda):    
            if(system_settings.bmsupported):
                genselected = True
                system_settings.usegpu = False
                system_settings.breakmodel = True
            else:
                genselected = False
        else:
            genselected = False

        if(system_settings.hascuda):
            if(use_gpu):
                if(system_settings.bmsupported):
                    system_settings.breakmodel = True
                    system_settings.usegpu = False
                    genselected = True
                else:
                    system_settings.breakmodel = False
                    system_settings.usegpu = True
                    genselected = True
            else:
                system_settings.breakmodel = utils.HAS_ACCELERATE
                system_settings.usegpu = False
                genselected = True

    # Ask for API key if InferKit was selected
    if(model_settings.model == "InferKit"):
        model_settings.apikey = model_settings.oaiapikey
                    
    # Swap OAI Server if GooseAI was selected
    if(model_settings.model == "GooseAI"):
        model_settings.oaiengines = "https://api.goose.ai/v1/engines"
        model_settings.model = "OAI"
        args.configname = "GooseAI"

    # Ask for API key if OpenAI was selected
    if(model_settings.model == "OAI"):
        if not args.configname:
            args.configname = "OAI"
        
    if(model_settings.model == "ReadOnly"):
        system_settings.noai = True

    # Start transformers and create pipeline
    if(not system_settings.use_colab_tpu and model_settings.model not in ["InferKit", "Colab", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        if(not system_settings.noai):
            print("{0}Initializing transformers, please wait...{1}".format(colors.PURPLE, colors.END))
            for m in ("GPTJModel", "XGLMModel"):
                try:
                    globals()[m] = getattr(__import__("transformers"), m)
                except:
                    pass

            # Lazy loader
            import torch_lazy_loader
            def get_lazy_load_callback(n_layers, convert_to_float16=True):
                if not model_settings.lazy_load:
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
                            device_map[key] = system_settings.gpu_device if system_settings.hascuda and system_settings.usegpu else "cpu" if not system_settings.hascuda or not system_settings.breakmodel else breakmodel.primary_device
                        else:
                            layer = int(max((n for n in utils.layers_module_names if original_key.startswith(n)), key=len).rsplit(".", 1)[1])
                            device = system_settings.gpu_device if system_settings.hascuda and system_settings.usegpu else "disk" if layer < disk_blocks and layer < ram_blocks else "cpu" if not system_settings.hascuda or not system_settings.breakmodel else "shared" if layer < ram_blocks else bisect.bisect_right(cumulative_gpu_blocks, layer - ram_blocks)
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
                                    model_settings.fp32_model = True
                                if convert_to_float16 and breakmodel.primary_device != "cpu" and system_settings.hascuda and (system_settings.breakmodel or system_settings.usegpu) and model_dict[key].dtype is torch.float32:
                                    model_dict[key] = model_dict[key].to(torch.float16)
                                if breakmodel.primary_device == "cpu" or (not system_settings.usegpu and not system_settings.breakmodel and model_dict[key].dtype is torch.float16):
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
                if(always_use or (system_settings.hascuda and args.lowmem and (system_settings.usegpu or system_settings.breakmodel))):
                    original_dtype = torch.get_default_dtype()
                    torch.set_default_dtype(torch.float16)
                    yield True
                    torch.set_default_dtype(original_dtype)
                else:
                    yield False

            # If custom GPT2 model was chosen
            if(model_settings.model == "GPT2Custom"):
                model_settings.lazy_load = False
                model_config = open(model_settings.custmodpth + "/config.json", "r")
                js   = json.load(model_config)
                with(maybe_use_float16()):
                    try:
                        model = GPT2LMHeadModel.from_pretrained(model_settings.custmodpth, revision=model_settings.revision, cache_dir="cache")
                    except Exception as e:
                        if("out of memory" in traceback.format_exc().lower()):
                            raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                        raise e
                tokenizer = GPT2TokenizerFast.from_pretrained(model_settings.custmodpth, revision=model_settings.revision, cache_dir="cache")
                model_settings.modeldim = get_hidden_size_from_model(model)
                # Is CUDA available? If so, use GPU, otherwise fall back to CPU
                if(system_settings.hascuda and system_settings.usegpu):
                    model = model.half().to(system_settings.gpu_device)
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
                if(model_settings.model_type == "gpt2"):
                    lowmem = {}
                    model_settings.lazy_load = False  # Also, lazy loader doesn't support GPT-2 models
                
                # If we're using torch_lazy_loader, we need to get breakmodel config
                # early so that it knows where to load the individual model tensors
                if(utils.HAS_ACCELERATE or model_settings.lazy_load and system_settings.hascuda and system_settings.breakmodel):
                    device_config(model_config)

                # Download model from Huggingface if it does not exist, otherwise load locally
                
                #If we specify a model and it's in the root directory, we need to move it to the models directory (legacy folder structure to new)
                if os.path.isdir(model_settings.model.replace('/', '_')):
                    import shutil
                    shutil.move(model_settings.model.replace('/', '_'), "models/{}".format(model_settings.model.replace('/', '_')))
                print("\n", flush=True)
                if(model_settings.lazy_load):  # If we're using lazy loader, we need to figure out what the model's hidden layers are called
                    with torch_lazy_loader.use_lazy_torch_load(dematerialized_modules=True, use_accelerate_init_empty_weights=True):
                        try:
                            metamodel = AutoModelForCausalLM.from_config(model_config)
                        except Exception as e:
                            metamodel = GPTNeoForCausalLM.from_config(model_config)
                        utils.layers_module_names = utils.get_layers_module_names(metamodel)
                        utils.module_names = list(metamodel.state_dict().keys())
                        utils.named_buffers = list(metamodel.named_buffers(recurse=True))
                with maybe_use_float16(), torch_lazy_loader.use_lazy_torch_load(enable=model_settings.lazy_load, callback=get_lazy_load_callback(utils.num_layers(model_config)) if model_settings.lazy_load else None, dematerialized_modules=True):
                    if(model_settings.lazy_load):  # torch_lazy_loader.py and low_cpu_mem_usage can't be used at the same time
                        lowmem = {}
                    if(os.path.isdir(model_settings.custmodpth)):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(model_settings.custmodpth, revision=model_settings.revision, cache_dir="cache")
                        except Exception as e:
                            pass
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(model_settings.custmodpth, revision=model_settings.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = GPT2TokenizerFast.from_pretrained(model_settings.custmodpth, revision=model_settings.revision, cache_dir="cache")
                            except Exception as e:
                                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=model_settings.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained(model_settings.custmodpth, revision=model_settings.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained(model_settings.custmodpth, revision=model_settings.revision, cache_dir="cache", **lowmem)
                    elif(os.path.isdir("models/{}".format(model_settings.model.replace('/', '_')))):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(model_settings.model.replace('/', '_')), revision=model_settings.revision, cache_dir="cache")
                        except Exception as e:
                            pass
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(model_settings.model.replace('/', '_')), revision=model_settings.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = GPT2TokenizerFast.from_pretrained("models/{}".format(model_settings.model.replace('/', '_')), revision=model_settings.revision, cache_dir="cache")
                            except Exception as e:
                                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=model_settings.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained("models/{}".format(model_settings.model.replace('/', '_')), revision=model_settings.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained("models/{}".format(model_settings.model.replace('/', '_')), revision=model_settings.revision, cache_dir="cache", **lowmem)
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
                                model_settings.fp32_model = True
                            return old_rebuild_tensor(storage, storage_offset, shape, stride)
                        torch._utils._rebuild_tensor = new_rebuild_tensor

                        try:
                            tokenizer = AutoTokenizer.from_pretrained(model_settings.model, revision=model_settings.revision, cache_dir="cache")
                        except Exception as e:
                            pass
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(model_settings.model, revision=model_settings.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = GPT2TokenizerFast.from_pretrained(model_settings.model, revision=model_settings.revision, cache_dir="cache")
                            except Exception as e:
                                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=model_settings.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained(model_settings.model, revision=model_settings.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained(model_settings.model, revision=model_settings.revision, cache_dir="cache", **lowmem)

                        torch._utils._rebuild_tensor = old_rebuild_tensor

                        if not args.colab or args.savemodel:
                            import shutil
                            tokenizer.save_pretrained("models/{}".format(model_settings.model.replace('/', '_')))
                            if(model_settings.fp32_model):  # Use save_pretrained to convert fp32 models to fp16
                                model = model.half()
                                model.save_pretrained("models/{}".format(model_settings.model.replace('/', '_')), max_shard_size="500MiB")
                            else:  # For fp16 models, we can just copy the model files directly
                                import transformers.configuration_utils
                                import transformers.modeling_utils
                                import transformers.file_utils
                                # Save the config.json
                                shutil.move(transformers.file_utils.get_from_cache(transformers.file_utils.hf_bucket_url(model_settings.model, transformers.configuration_utils.CONFIG_NAME, revision=model_settings.revision), cache_dir="cache", local_files_only=True), os.path.join("models/{}".format(model_settings.model.replace('/', '_')), transformers.configuration_utils.CONFIG_NAME))
                                if(utils.num_shards is None):
                                    # Save the pytorch_model.bin of an unsharded model
                                    shutil.move(transformers.file_utils.get_from_cache(transformers.file_utils.hf_bucket_url(model_settings.model, transformers.modeling_utils.WEIGHTS_NAME, revision=model_settings.revision), cache_dir="cache", local_files_only=True), os.path.join("models/{}".format(model_settings.model.replace('/', '_')), transformers.modeling_utils.WEIGHTS_NAME))
                                else:
                                    with open(utils.from_pretrained_index_filename) as f:
                                        map_data = json.load(f)
                                    filenames = set(map_data["weight_map"].values())
                                    # Save the pytorch_model.bin.index.json of a sharded model
                                    shutil.move(utils.from_pretrained_index_filename, os.path.join("models/{}".format(model_settings.model.replace('/', '_')), transformers.modeling_utils.WEIGHTS_INDEX_NAME))
                                    # Then save the pytorch_model-#####-of-#####.bin files
                                    for filename in filenames:
                                        shutil.move(transformers.file_utils.get_from_cache(transformers.file_utils.hf_bucket_url(model_settings.model, filename, revision=model_settings.revision), cache_dir="cache", local_files_only=True), os.path.join("models/{}".format(model_settings.model.replace('/', '_')), filename))
                            shutil.rmtree("cache/")

                if(model_settings.badwordsids is koboldai_settings.badwordsids_default and model_settings.model_type not in ("gpt2", "gpt_neo", "gptj")):
                    model_settings.badwordsids = [[v] for k, v in tokenizer.get_vocab().items() if any(c in str(k) for c in "<>[]") if model_settings.newlinemode != "s" or str(k) != "</s>"]

                patch_causallm(model)

                if(system_settings.hascuda):
                    if(system_settings.usegpu):
                        model_settings.modeldim = get_hidden_size_from_model(model)
                        model = model.half().to(system_settings.gpu_device)
                        generator = model.generate
                    elif(system_settings.breakmodel):  # Use both RAM and VRAM (breakmodel)
                        model_settings.modeldim = get_hidden_size_from_model(model)
                        if(not model_settings.lazy_load):
                            device_config(model.config)
                        move_model_to_devices(model)
                    elif(utils.HAS_ACCELERATE and __import__("breakmodel").disk_blocks > 0):
                        move_model_to_devices(model)
                        model_settings.modeldim = get_hidden_size_from_model(model)
                        generator = model.generate
                    else:
                        model = model.to('cpu').float()
                        model_settings.modeldim = get_hidden_size_from_model(model)
                        generator = model.generate
                elif(utils.HAS_ACCELERATE and __import__("breakmodel").disk_blocks > 0):
                    move_model_to_devices(model)
                    model_settings.modeldim = get_hidden_size_from_model(model)
                    generator = model.generate
                else:
                    model.to('cpu').float()
                    model_settings.modeldim = get_hidden_size_from_model(model)
                    generator = model.generate
            
            # Suppress Author's Note by flagging square brackets (Old implementation)
            #vocab         = tokenizer.get_vocab()
            #vocab_keys    = vocab.keys()
            #vars.badwords = gettokenids("[")
            #for key in vars.badwords:
            #    model_settings.badwordsids.append([vocab[key]])
            
            print("{0}OK! {1} pipeline created!{2}".format(colors.GREEN, model_settings.model, colors.END))
        
        else:
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=model_settings.revision, cache_dir="cache")
    else:
        from transformers import PreTrainedModel
        from transformers import modeling_utils
        old_from_pretrained = PreTrainedModel.from_pretrained.__func__
        @classmethod
        def new_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            model_settings.fp32_model = False
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
            system_settings.lua_koboldbridge.logits = system_settings.lua_state.table()
            for r, row in enumerate(scores_list):
                system_settings.lua_koboldbridge.logits[r+1] = system_settings.lua_state.table(*row)
            system_settings.lua_koboldbridge.vocab_size = scores_shape[-1]

            execute_genmod()

            scores = np.array(
                tuple(tuple(row.values()) for row in system_settings.lua_koboldbridge.logits.values()),
                dtype=scores.dtype,
            )
            assert scores.shape == scores_shape

            return scores
        
        def tpumtjgenerate_stopping_callback(generated, n_generated, excluded_world_info) -> Tuple[List[set], bool, bool]:
            model_settings.generated_tkns += 1

            assert len(excluded_world_info) == len(generated)
            regeneration_required = system_settings.lua_koboldbridge.regeneration_required
            halt = system_settings.abort or not system_settings.lua_koboldbridge.generating or model_settings.generated_tkns >= model_settings.genamt
            system_settings.lua_koboldbridge.regeneration_required = False

            global past

            for i in range(model_settings.numseqs):
                system_settings.lua_koboldbridge.generated[i+1][model_settings.generated_tkns] = int(generated[i, tpu_mtj_backend.params["seq"] + n_generated - 1].item())

            if(not story_settings.dynamicscan or halt):
                return excluded_world_info, regeneration_required, halt

            for i, t in enumerate(generated):
                decoded = utils.decodenewlines(tokenizer.decode(past[i])) + utils.decodenewlines(tokenizer.decode(t[tpu_mtj_backend.params["seq"] : tpu_mtj_backend.params["seq"] + n_generated]))
                _, found = checkworldinfo(decoded, force_use_txt=True, actions=story_settings._actions)
                found -= excluded_world_info[i]
                if(len(found) != 0):
                    regeneration_required = True
                    break
            return excluded_world_info, regeneration_required, halt

        def tpumtjgenerate_compiling_callback() -> None:
            print(colors.GREEN + "TPU backend compilation triggered" + colors.END)
            system_settings.compiling = True

        def tpumtjgenerate_stopped_compiling_callback() -> None:
            system_settings.compiling = False
        
        def tpumtjgenerate_settings_callback() -> dict:
            return {
                "sampler_order": model_settings.sampler_order,
                "top_p": float(model_settings.top_p),
                "temp": float(model_settings.temp),
                "top_k": int(model_settings.top_k),
                "tfs": float(model_settings.tfs),
                "typical": float(model_settings.typical),
                "top_a": float(model_settings.top_a),
                "repetition_penalty": float(model_settings.rep_pen),
                "rpslope": float(model_settings.rep_pen_slope),
                "rprange": int(model_settings.rep_pen_range),
            }

        # If we're running Colab or OAI, we still need a tokenizer.
        if(model_settings.model == "Colab"):
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-2.7B", revision=model_settings.revision, cache_dir="cache")
            loadsettings()
        elif(model_settings.model == "OAI"):
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=model_settings.revision, cache_dir="cache")
            loadsettings()
        # Load the TPU backend if requested
        elif(system_settings.use_colab_tpu or model_settings.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
            global tpu_mtj_backend
            import tpu_mtj_backend
            if(model_settings.model == "TPUMeshTransformerGPTNeoX"):
                model_settings.badwordsids = koboldai_settings.badwordsids_neox
            print("{0}Initializing Mesh Transformer JAX, please wait...{1}".format(colors.PURPLE, colors.END))
            if model_settings.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and (not model_settings.custmodpth or not os.path.isdir(model_settings.custmodpth)):
                raise FileNotFoundError(f"The specified model path {repr(model_settings.custmodpth)} is not the path to a valid folder")
            import tpu_mtj_backend
            if(model_settings.model == "TPUMeshTransformerGPTNeoX"):
                tpu_mtj_backend.pad_token_id = 2
            tpu_mtj_backend.model_settings = model_settings
            tpu_mtj_backend.warper_callback = tpumtjgenerate_warper_callback
            tpu_mtj_backend.stopping_callback = tpumtjgenerate_stopping_callback
            tpu_mtj_backend.compiling_callback = tpumtjgenerate_compiling_callback
            tpu_mtj_backend.stopped_compiling_callback = tpumtjgenerate_stopped_compiling_callback
            tpu_mtj_backend.settings_callback = tpumtjgenerate_settings_callback
            system_settings.allowsp = True
            loadmodelsettings()
            loadsettings()
            tpu_mtj_backend.load_model(model_settings.custmodpth, hf_checkpoint=model_settings.model not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and system_settings.use_colab_tpu, **model_settings.modelconfig)
            model_settings.modeldim = int(tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"]))
            tokenizer = tpu_mtj_backend.tokenizer
            if(model_settings.badwordsids is koboldai_settings.badwordsids_default and model_settings.model_type not in ("gpt2", "gpt_neo", "gptj")):
                model_settings.badwordsids = [[v] for k, v in tokenizer.get_vocab().items() if any(c in str(k) for c in "<>[]") if model_settings.newlinemode != "s" or str(k) != "</s>"]
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
        
        if not story_settings.gamestarted:
            setStartState()
            sendsettings()
            refresh_settings()
    
    #Let's load the presets
    with open('settings/preset/official.presets') as f:
        presets = json.load(f)
        if model_settings.model in presets:
            model_settings.presets = presets[model_settings.model]
        elif model_settings.model.replace("/", "_") in presets:
            model_settings.presets = presets[model_settings.model.replace("/", "_")]
        else:
            model_settings.presets = {}

# Set up Flask routes
@app.route('/')
@app.route('/index')
def index():
    if 'new_ui' in request.args:
        return render_template('index_new.html', hide_ai_menu=args.noaimenu)
    else:
        return render_template('index.html', hide_ai_menu=args.noaimenu, flaskwebgui=system_settings.flaskwebgui)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path,
                                   'koboldai.ico', mimetype='image/vnd.microsoft.icon')    
@app.route('/download')
def download():
    save_format = request.args.get("format", "json").strip().lower()

    if(save_format == "plaintext"):
        txt = story_settings.prompt + "".join(story_settings.actions.values())
        save = Response(txt)
        filename = path.basename(system_settings.savedir)
        if filename[-5:] == ".json":
            filename = filename[:-5]
        save.headers.set('Content-Disposition', 'attachment', filename='%s.txt' % filename)
        return(save)

    # Build json to write
    js = {}
    js["gamestarted"] = story_settings.gamestarted
    js["prompt"]      = story_settings.prompt
    js["memory"]      = story_settings.memory
    js["authorsnote"] = story_settings.authornote
    js["anotetemplate"] = story_settings.authornotetemplate
    js["actions"]     = story_settings.actions.to_json()
    js["worldinfo"]   = []
        
    # Extract only the important bits of WI
    for wi in story_settings.worldinfo:
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
    filename = path.basename(system_settings.savedir)
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
            system_settings.userscripts = []
            for userscript in js["userscripts"]:
                if type(userscript) is not str:
                    continue
                userscript = userscript.strip()
                if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                    system_settings.userscripts.append(userscript)
        if("corescript" in js and type(js["corescript"]) is str and all(q not in js["corescript"] for q in ("..", ":")) and all(js["corescript"][0] not in q for q in ("/", "\\"))):
            system_settings.corescript = js["corescript"]
        else:
            system_settings.corescript = "default.lua"
        file.close()
        
    #==================================================================#
    #  Lua runtime startup
    #==================================================================#

    print("", end="", flush=True)
    print(colors.PURPLE + "Initializing Lua Bridge... " + colors.END, end="", flush=True)

    # Set up Lua state
    system_settings.lua_state = lupa.LuaRuntime(unpack_returned_tuples=True)

    # Load bridge.lua
    bridged = {
        "corescript_path": "cores",
        "userscript_path": "userscripts",
        "config_path": "userscripts",
        "lib_paths": system_settings.lua_state.table("lualibs", os.path.join("extern", "lualibs")),
        "model_settings": model_settings,
        "story_settings": story_settings,
        "user_settings": user_settings,
        "system_settings": system_settings,
    }
    for kwarg in _bridged:
        bridged[kwarg] = _bridged[kwarg]
    try:
        system_settings.lua_kobold, system_settings.lua_koboldcore, system_settings.lua_koboldbridge = system_settings.lua_state.globals().dofile("bridge.lua")(
            system_settings.lua_state.globals().python,
            bridged,
        )
    except lupa.LuaError as e:
        print(colors.RED + "ERROR!" + colors.END)
        system_settings.lua_koboldbridge.obliterate_multiverse()
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
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

    for filename in system_settings.userscripts:
        if filename in filenames_dict:
            i = filenames_dict[filename]
            filenames.append(filename)
            modulenames.append(lst[i]["modulename"])
            descriptions.append(lst[i]["description"])

    system_settings.has_genmod = False

    try:
        system_settings.lua_koboldbridge.obliterate_multiverse()
        tpool.execute(system_settings.lua_koboldbridge.load_corescript, system_settings.corescript)
        system_settings.has_genmod = tpool.execute(system_settings.lua_koboldbridge.load_userscripts, filenames, modulenames, descriptions)
        system_settings.lua_running = True
    except lupa.LuaError as e:
        try:
            system_settings.lua_koboldbridge.obliterate_multiverse()
        except:
            pass
        system_settings.lua_running = False
        if(system_settings.serverstarted):
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
            sendUSStatItems()
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        if(system_settings.serverstarted):
            set_aibusy(0)

#==================================================================#
#  Print message that originates from the userscript with the given name
#==================================================================#
@bridged_kwarg()
def lua_print(msg):
    if(system_settings.lua_logname != system_settings.lua_koboldbridge.logging_name):
        system_settings.lua_logname = system_settings.lua_koboldbridge.logging_name
        print(colors.BLUE + lua_log_format_name(system_settings.lua_logname) + ":" + colors.END, file=sys.stderr)
    print(colors.PURPLE + msg.replace("\033", "") + colors.END)

#==================================================================#
#  Print warning that originates from the userscript with the given name
#==================================================================#
@bridged_kwarg()
def lua_warn(msg):
    if(system_settings.lua_logname != system_settings.lua_koboldbridge.logging_name):
        system_settings.lua_logname = system_settings.lua_koboldbridge.logging_name
        print(colors.BLUE + lua_log_format_name(system_settings.lua_logname) + ":" + colors.END, file=sys.stderr)
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
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=model_settings.revision, cache_dir="cache")
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
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=model_settings.revision, cache_dir="cache")
    return tokenizer.encode(utils.encodenewlines(string), max_length=int(4e9), truncation=True)

#==================================================================#
#  Computes context given a submission, Lua array of entry UIDs and a Lua array
#  of folder UIDs
#==================================================================#
@bridged_kwarg()
def lua_compute_context(submission, entries, folders, kwargs):
    assert type(submission) is str
    if(kwargs is None):
        kwargs = system_settings.lua_state.table()
    actions = story_settings._actions if system_settings.lua_koboldbridge.userstate == "genmod" else story_settings.actions
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
    if(uid in story_settings.worldinfo_u and k in (
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
        return story_settings.worldinfo_u[uid][k]

#==================================================================#
#  Set property of a world info entry given its UID, property name and new value
#==================================================================#
@bridged_kwarg()
def lua_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in story_settings.worldinfo_u and k in (
        "key",
        "keysecondary",
        "content",
        "comment",
        "selective",
        "constant",
    )
    if(type(story_settings.worldinfo_u[uid][k]) is int and type(v) is float):
        v = int(v)
    assert type(story_settings.worldinfo_u[uid][k]) is type(v)
    story_settings.worldinfo_u[uid][k] = v
    print(colors.GREEN + f"{lua_log_format_name(system_settings.lua_koboldbridge.logging_name)} set {k} of world info entry {uid} to {v}" + colors.END)

#==================================================================#
#  Get property of a world info folder given its UID and property name
#==================================================================#
@bridged_kwarg()
def lua_folder_get_attr(uid, k):
    assert type(uid) is int and type(k) is str
    if(uid in story_settings.wifolders_d and k in (
        "name",
    )):
        return story_settings.wifolders_d[uid][k]

#==================================================================#
#  Set property of a world info folder given its UID, property name and new value
#==================================================================#
@bridged_kwarg()
def lua_folder_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in story_settings.wifolders_d and k in (
        "name",
    )
    if(type(story_settings.wifolders_d[uid][k]) is int and type(v) is float):
        v = int(v)
    assert type(story_settings.wifolders_d[uid][k]) is type(v)
    story_settings.wifolders_d[uid][k] = v
    print(colors.GREEN + f"{lua_log_format_name(system_settings.lua_koboldbridge.logging_name)} set {k} of world info folder {uid} to {v}" + colors.END)

#==================================================================#
#  Get the "Amount to Generate"
#==================================================================#
@bridged_kwarg()
def lua_get_genamt():
    return model_settings.genamt

#==================================================================#
#  Set the "Amount to Generate"
#==================================================================#
@bridged_kwarg()
def lua_set_genamt(genamt):
    assert system_settings.lua_koboldbridge.userstate != "genmod" and type(genamt) in (int, float) and genamt >= 0
    print(colors.GREEN + f"{lua_log_format_name(system_settings.lua_koboldbridge.logging_name)} set genamt to {int(genamt)}" + colors.END)
    model_settings.genamt = int(genamt)

#==================================================================#
#  Get the "Gens Per Action"
#==================================================================#
@bridged_kwarg()
def lua_get_numseqs():
    return model_settings.numseqs

#==================================================================#
#  Set the "Gens Per Action"
#==================================================================#
@bridged_kwarg()
def lua_set_numseqs(numseqs):
    assert type(numseqs) in (int, float) and numseqs >= 1
    print(colors.GREEN + f"{lua_log_format_name(system_settings.lua_koboldbridge.logging_name)} set numseqs to {int(numseqs)}" + colors.END)
    model_settings.numseqs = int(numseqs)

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
    )

#==================================================================#
#  Return the setting with the given name if it exists
#==================================================================#
@bridged_kwarg()
def lua_get_setting(setting):
    if(setting in ("settemp", "temp")): return model_settings.temp
    if(setting in ("settopp", "topp", "top_p")): return model_settings.top_p
    if(setting in ("settopk", "topk", "top_k")): return model_settings.top_k
    if(setting in ("settfs", "tfs")): return model_settings.tfs
    if(setting in ("settypical", "typical")): return model_settings.typical
    if(setting in ("settopa", "topa")): return model_settings.top_a
    if(setting in ("setreppen", "reppen")): return model_settings.rep_pen
    if(setting in ("setreppenslope", "reppenslope")): return model_settings.rep_pen_slope
    if(setting in ("setreppenrange", "reppenrange")): return model_settings.rep_pen_range
    if(setting in ("settknmax", "tknmax")): return model_settings.max_length
    if(setting == "anotedepth"): return story_settings.andepth
    if(setting in ("setwidepth", "widepth")): return user_settings.widepth
    if(setting in ("setuseprompt", "useprompt")): return story_settings.useprompt
    if(setting in ("setadventure", "adventure")): return story_settings.adventure
    if(setting in ("setchatmode", "chatmode")): return story_settings.chatmode
    if(setting in ("setdynamicscan", "dynamicscan")): return story_settings.dynamicscan
    if(setting in ("setnopromptgen", "nopromptgen")): return user_settings.nopromptgen
    if(setting in ("autosave", "autosave")): return user_settings.autosave
    if(setting in ("setrngpersist", "rngpersist")): return user_settings.rngpersist
    if(setting in ("frmttriminc", "triminc")): return user_settings.formatoptns["frmttriminc"]
    if(setting in ("frmtrmblln", "rmblln")): return user_settings.formatoptns["frmttrmblln"]
    if(setting in ("frmtrmspch", "rmspch")): return user_settings.formatoptns["frmttrmspch"]
    if(setting in ("frmtadsnsp", "adsnsp")): return user_settings.formatoptns["frmtadsnsp"]
    if(setting in ("frmtsingleline", "singleline")): return user_settings.formatoptns["singleline"]

#==================================================================#
#  Set the setting with the given name if it exists
#==================================================================#
@bridged_kwarg()
def lua_set_setting(setting, v):
    actual_type = type(lua_get_setting(setting))
    assert v is not None and (actual_type is type(v) or (actual_type is int and type(v) is float))
    v = actual_type(v)
    print(colors.GREEN + f"{lua_log_format_name(system_settings.lua_koboldbridge.logging_name)} set {setting} to {v}" + colors.END)
    if(setting in ("setadventure", "adventure") and v):
        story_settings.actionmode = 1
    if(setting in ("settemp", "temp")): model_settings.temp = v
    if(setting in ("settopp", "topp")): model_settings.top_p = v
    if(setting in ("settopk", "topk")): model_settings.top_k = v
    if(setting in ("settfs", "tfs")): model_settings.tfs = v
    if(setting in ("settypical", "typical")): model_settings.typical = v
    if(setting in ("settopa", "topa")): model_settings.top_a = v
    if(setting in ("setreppen", "reppen")): model_settings.rep_pen = v
    if(setting in ("setreppenslope", "reppenslope")): model_settings.rep_pen_slope = v
    if(setting in ("setreppenrange", "reppenrange")): model_settings.rep_pen_range = v
    if(setting in ("settknmax", "tknmax")): model_settings.max_length = v; return True
    if(setting == "anotedepth"): story_settings.andepth = v; return True
    if(setting in ("setwidepth", "widepth")): user_settings.widepth = v; return True
    if(setting in ("setuseprompt", "useprompt")): story_settings.useprompt = v; return True
    if(setting in ("setadventure", "adventure")): story_settings.adventure = v
    if(setting in ("setdynamicscan", "dynamicscan")): story_settings.dynamicscan = v
    if(setting in ("setnopromptgen", "nopromptgen")): user_settings.nopromptgen = v
    if(setting in ("autosave", "noautosave")): user_settings.autosave = v
    if(setting in ("setrngpersist", "rngpersist")): user_settings.rngpersist = v
    if(setting in ("setchatmode", "chatmode")): story_settings.chatmode = v
    if(setting in ("frmttriminc", "triminc")): user_settings.formatoptns["frmttriminc"] = v
    if(setting in ("frmtrmblln", "rmblln")): user_settings.formatoptns["frmttrmblln"] = v
    if(setting in ("frmtrmspch", "rmspch")): user_settings.formatoptns["frmttrmspch"] = v
    if(setting in ("frmtadsnsp", "adsnsp")): user_settings.formatoptns["frmtadsnsp"] = v
    if(setting in ("frmtsingleline", "singleline")): user_settings.formatoptns["singleline"] = v

#==================================================================#
#  Get contents of memory
#==================================================================#
@bridged_kwarg()
def lua_get_memory():
    return story_settings.memory

#==================================================================#
#  Set contents of memory
#==================================================================#
@bridged_kwarg()
def lua_set_memory(m):
    assert type(m) is str
    story_settings.memory = m

#==================================================================#
#  Get contents of author's note
#==================================================================#
@bridged_kwarg()
def lua_get_authorsnote():
    return story_settings.authornote

#==================================================================#
#  Set contents of author's note
#==================================================================#
@bridged_kwarg()
def lua_set_authorsnote(m):
    assert type(m) is str
    story_settings.authornote = m

#==================================================================#
#  Get contents of author's note template
#==================================================================#
@bridged_kwarg()
def lua_get_authorsnotetemplate():
    return story_settings.authornotetemplate

#==================================================================#
#  Set contents of author's note template
#==================================================================#
@bridged_kwarg()
def lua_set_authorsnotetemplate(m):
    assert type(m) is str
    story_settings.authornotetemplate = m

#==================================================================#
#  Save settings and send them to client
#==================================================================#
@bridged_kwarg()
def lua_resend_settings():
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
        print(colors.GREEN + f"{lua_log_format_name(system_settings.lua_koboldbridge.logging_name)} deleted story chunk {k}" + colors.END)
        chunk = int(k)
        if(system_settings.lua_koboldbridge.userstate == "genmod"):
            del story_settings._actions[chunk-1]
        story_settings.lua_deleted.add(chunk)
        if(not hasattr(story_settings, "_actions") or story_settings._actions is not story_settings.actions):
            #Instead of deleting we'll blank out the text. This way our actions and actions_metadata stay in sync and we can restore the chunk on an undo
            story_settings.actions[chunk-1] = ""
            send_debug()
    else:
        if(k == 0):
            print(colors.GREEN + f"{lua_log_format_name(system_settings.lua_koboldbridge.logging_name)} edited prompt chunk" + colors.END)
        else:
            print(colors.GREEN + f"{lua_log_format_name(system_settings.lua_koboldbridge.logging_name)} edited story chunk {k}" + colors.END)
        chunk = int(k)
        if(chunk == 0):
            if(system_settings.lua_koboldbridge.userstate == "genmod"):
                story_settings._prompt = v
            story_settings.lua_edited.add(chunk)
            story_settings.prompt = v
        else:
            if(system_settings.lua_koboldbridge.userstate == "genmod"):
                story_settings._actions[chunk-1] = v
            story_settings.lua_edited.add(chunk)
            story_settings.actions[chunk-1] = v
            send_debug()

#==================================================================#
#  Get model type as "gpt-2-xl", "gpt-neo-2.7B", etc.
#==================================================================#
@bridged_kwarg()
def lua_get_modeltype():
    if(system_settings.noai):
        return "readonly"
    if(model_settings.model in ("Colab", "OAI", "InferKit")):
        return "api"
    if(not system_settings.use_colab_tpu and model_settings.model not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and (model_settings.model in ("GPT2Custom", "NeoCustom") or model_settings.model_type in ("gpt2", "gpt_neo", "gptj"))):
        hidden_size = get_hidden_size_from_model(model)
    if(model_settings.model in ("gpt2",) or (model_settings.model_type == "gpt2" and hidden_size == 768)):
        return "gpt2"
    if(model_settings.model in ("gpt2-medium",) or (model_settings.model_type == "gpt2" and hidden_size == 1024)):
        return "gpt2-medium"
    if(model_settings.model in ("gpt2-large",) or (model_settings.model_type == "gpt2" and hidden_size == 1280)):
        return "gpt2-large"
    if(model_settings.model in ("gpt2-xl",) or (model_settings.model_type == "gpt2" and hidden_size == 1600)):
        return "gpt2-xl"
    if(model_settings.model_type == "gpt_neo" and hidden_size == 768):
        return "gpt-neo-125M"
    if(model_settings.model in ("EleutherAI/gpt-neo-1.3B",) or (model_settings.model_type == "gpt_neo" and hidden_size == 2048)):
        return "gpt-neo-1.3B"
    if(model_settings.model in ("EleutherAI/gpt-neo-2.7B",) or (model_settings.model_type == "gpt_neo" and hidden_size == 2560)):
        return "gpt-neo-2.7B"
    if(model_settings.model in ("EleutherAI/gpt-j-6B",) or ((system_settings.use_colab_tpu or model_settings.model == "TPUMeshTransformerGPTJ") and tpu_mtj_backend.params["d_model"] == 4096) or (model_settings.model_type in ("gpt_neo", "gptj") and hidden_size == 4096)):
        return "gpt-j-6B"
    return "unknown"

#==================================================================#
#  Get model backend as "transformers" or "mtj"
#==================================================================#
@bridged_kwarg()
def lua_get_modelbackend():
    if(system_settings.noai):
        return "readonly"
    if(model_settings.model in ("Colab", "OAI", "InferKit")):
        return "api"
    if(system_settings.use_colab_tpu or model_settings.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        return "mtj"
    return "transformers"

#==================================================================#
#  Check whether model is loaded from a custom path
#==================================================================#
@bridged_kwarg()
def lua_is_custommodel():
    return model_settings.model in ("GPT2Custom", "NeoCustom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")

#==================================================================#
#  Return the filename (as a string) of the current soft prompt, or
#  None if no soft prompt is loaded
#==================================================================#
@bridged_kwarg()
def lua_get_spfilename():
    return system_settings.spfilename.strip() or None

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
    system_settings.lua_logname = ...
    story_settings.lua_edited = set()
    story_settings.lua_deleted = set()
    try:
        tpool.execute(system_settings.lua_koboldbridge.execute_inmod)
    except lupa.LuaError as e:
        system_settings.lua_koboldbridge.obliterate_multiverse()
        system_settings.lua_running = False
        emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
        sendUSStatItems()
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        set_aibusy(0)

def execute_genmod():
    system_settings.lua_koboldbridge.execute_genmod()

def execute_outmod():
    setgamesaved(False)
    emit('from_server', {'cmd': 'hidemsg', 'data': ''}, broadcast=True, room="UI_1")
    try:
        tpool.execute(system_settings.lua_koboldbridge.execute_outmod)
    except lupa.LuaError as e:
        system_settings.lua_koboldbridge.obliterate_multiverse()
        system_settings.lua_running = False
        emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
        sendUSStatItems()
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        set_aibusy(0)
    if(system_settings.lua_koboldbridge.resend_settings_required):
        system_settings.lua_koboldbridge.resend_settings_required = False
        lua_resend_settings()
    for k in story_settings.lua_edited:
        inlineedit(k, story_settings.actions[k])
    for k in story_settings.lua_deleted:
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
    if request.args.get("ui") == 2:
        ui2_connect()
        return
    #Send all variables to client
    model_settings.send_to_ui()
    story_settings.send_to_ui()
    user_settings.send_to_ui()
    system_settings.send_to_ui()
    print("{0}Client connected!{1}".format(colors.GREEN, colors.END))
    emit('from_server', {'cmd': 'setchatname', 'data': story_settings.chatname}, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': story_settings.authornotetemplate}, room="UI_1")
    emit('from_server', {'cmd': 'connected', 'smandelete': system_settings.smandelete, 'smanrename': system_settings.smanrename, 'modelname': getmodelname()}, room="UI_1")
    if(system_settings.host):
        emit('from_server', {'cmd': 'runs_remotely'}, room="UI_1")
    if(system_settings.flaskwebgui):
        emit('from_server', {'cmd': 'flaskwebgui'}, room="UI_1")
    if(system_settings.allowsp):
        emit('from_server', {'cmd': 'allowsp', 'data': system_settings.allowsp}, room="UI_1")

    sendUSStatItems()
    emit('from_server', {'cmd': 'spstatitems', 'data': {system_settings.spfilename: system_settings.spmeta} if system_settings.allowsp and len(system_settings.spfilename) else {}}, broadcast=True, room="UI_1")

    if(not story_settings.gamestarted):
        setStartState()
        sendsettings()
        refresh_settings()
        user_settings.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': user_settings.laststory}, room="UI_1")
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': story_settings.memory}, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': story_settings.authornote}, room="UI_1")
        story_settings.mode = "play"
    else:
        # Game in session, send current game data and ready state to browser
        refresh_story()
        sendsettings()
        refresh_settings()
        emit('from_server', {'cmd': 'setstoryname', 'data': user_settings.laststory}, room="UI_1")
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': story_settings.memory}, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': story_settings.authornote}, room="UI_1")
        if(story_settings.mode == "play"):
            if(not system_settings.aibusy):
                emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, room="UI_1")
            else:
                emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'}, room="UI_1")
        elif(story_settings.mode == "edit"):
            emit('from_server', {'cmd': 'editmode', 'data': 'true'}, room="UI_1")
        elif(story_settings.mode == "memory"):
            emit('from_server', {'cmd': 'memmode', 'data': 'true'}, room="UI_1")
        elif(story_settings.mode == "wi"):
            emit('from_server', {'cmd': 'wimode', 'data': 'true'}, room="UI_1")

    emit('from_server', {'cmd': 'gamesaved', 'data': story_settings.gamesaved}, broadcast=True, room="UI_1")

#==================================================================#
# Event triggered when browser SocketIO sends data to the server
#==================================================================#
@socketio.on('message')
def get_message(msg):
    if not system_settings.quiet:
        print("{0}Data received:{1}{2}".format(colors.GREEN, msg, colors.END))
    # Submit action
    if(msg['cmd'] == 'submit'):
        if(story_settings.mode == "play"):
            if(system_settings.aibusy):
                if(msg.get('allowabort', False)):
                    system_settings.abort = True
                return
            system_settings.abort = False
            system_settings.lua_koboldbridge.feedback = None
            if(story_settings.chatmode):
                if(type(msg['chatname']) is not str):
                    raise ValueError("Chatname must be a string")
                story_settings.chatname = msg['chatname']
                settingschanged()
                emit('from_server', {'cmd': 'setchatname', 'data': story_settings.chatname}, room="UI_1")
            story_settings.recentrng = story_settings.recentrngm = None
            actionsubmit(msg['data'], actionmode=msg['actionmode'])
        elif(story_settings.mode == "edit"):
            editsubmit(msg['data'])
        elif(story_settings.mode == "memory"):
            memsubmit(msg['data'])
    # Retry Action
    elif(msg['cmd'] == 'retry'):
        if(system_settings.aibusy):
            if(msg.get('allowabort', False)):
                system_settings.abort = True
            return
        system_settings.abort = False
        if(story_settings.chatmode):
            if(type(msg['chatname']) is not str):
                raise ValueError("Chatname must be a string")
            story_settings.chatname = msg['chatname']
            settingschanged()
            emit('from_server', {'cmd': 'setchatname', 'data': story_settings.chatname}, room="UI_1")
        actionretry(msg['data'])
    # Back/Undo Action
    elif(msg['cmd'] == 'back'):
        ignore = actionback()
    # Forward/Redo Action
    elif(msg['cmd'] == 'redo'):
        actionredo()
    # EditMode Action (old)
    elif(msg['cmd'] == 'edit'):
        if(story_settings.mode == "play"):
            story_settings.mode = "edit"
            emit('from_server', {'cmd': 'editmode', 'data': 'true'}, broadcast=True, room="UI_1")
        elif(story_settings.mode == "edit"):
            story_settings.mode = "play"
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
    elif(not system_settings.host and msg['cmd'] == 'savetofile'):
        savetofile()
    elif(not system_settings.host and msg['cmd'] == 'loadfromfile'):
        loadfromfile()
    elif(msg['cmd'] == 'loadfromstring'):
        loadRequest(json.loads(msg['data']), filename=msg['filename'])
    elif(not system_settings.host and msg['cmd'] == 'import'):
        importRequest()
    elif(msg['cmd'] == 'newgame'):
        newGameRequest()
    elif(msg['cmd'] == 'rndgame'):
        randomGameRequest(msg['data'], memory=msg['memory'])
    elif(msg['cmd'] == 'settemp'):
        model_settings.temp = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltemp', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopp'):
        model_settings.top_p = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopp', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopk'):
        model_settings.top_k = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopk', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settfs'):
        model_settings.tfs = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltfs', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settypical'):
        model_settings.typical = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltypical', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopa'):
        model_settings.top_a = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopa', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppen'):
        model_settings.rep_pen = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppen', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppenslope'):
        model_settings.rep_pen_slope = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppenslope', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppenrange'):
        model_settings.rep_pen_range = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppenrange', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setoutput'):
        model_settings.genamt = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeloutput', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settknmax'):
        model_settings.max_length = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeltknmax', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setikgen'):
        model_settings.ikgen = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelikgen', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    # Author's Note field update
    elif(msg['cmd'] == 'anote'):
        anotesubmit(msg['data'], template=msg['template'])
    # Author's Note depth update
    elif(msg['cmd'] == 'anotedepth'):
        story_settings.andepth = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelanotedepth', 'data': msg['data']}, broadcast=True, room="UI_1")
        settingschanged()
        refresh_settings()
    # Format - Trim incomplete sentences
    elif(msg['cmd'] == 'frmttriminc'):
        if('frmttriminc' in user_settings.formatoptns):
            user_settings.formatoptns["frmttriminc"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtrmblln'):
        if('frmtrmblln' in user_settings.formatoptns):
            user_settings.formatoptns["frmtrmblln"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtrmspch'):
        if('frmtrmspch' in user_settings.formatoptns):
            user_settings.formatoptns["frmtrmspch"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtadsnsp'):
        if('frmtadsnsp' in user_settings.formatoptns):
            user_settings.formatoptns["frmtadsnsp"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'singleline'):
        if('singleline' in user_settings.formatoptns):
            user_settings.formatoptns["singleline"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'importselect'):
        user_settings.importnum = int(msg["data"].replace("import", ""))
    elif(msg['cmd'] == 'importcancel'):
        emit('from_server', {'cmd': 'popupshow', 'data': False}, room="UI_1")
        user_settings.importjs  = {}
    elif(msg['cmd'] == 'importaccept'):
        emit('from_server', {'cmd': 'popupshow', 'data': False}, room="UI_1")
        importgame()
    elif(msg['cmd'] == 'wi'):
        togglewimode()
    elif(msg['cmd'] == 'wiinit'):
        if(int(msg['data']) < len(story_settings.worldinfo)):
            setgamesaved(False)
            story_settings.worldinfo[msg['data']]["init"] = True
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
        assert 0 <= int(msg['data']) < len(story_settings.worldinfo)
        setgamesaved(False)
        emit('from_server', {'cmd': 'wiexpand', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiexpandfolder'):
        assert 0 <= int(msg['data']) < len(story_settings.worldinfo)
        setgamesaved(False)
        emit('from_server', {'cmd': 'wiexpandfolder', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wifoldercollapsecontent'):
        setgamesaved(False)
        story_settings.wifolders_d[msg['data']]['collapsed'] = True
        emit('from_server', {'cmd': 'wifoldercollapsecontent', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wifolderexpandcontent'):
        setgamesaved(False)
        story_settings.wifolders_d[msg['data']]['collapsed'] = False
        emit('from_server', {'cmd': 'wifolderexpandcontent', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiupdate'):
        setgamesaved(False)
        num = int(msg['num'])
        fields = ("key", "keysecondary", "content", "comment")
        for field in fields:
            if(field in msg['data'] and type(msg['data'][field]) is str):
                story_settings.worldinfo[num][field] = msg['data'][field]
        emit('from_server', {'cmd': 'wiupdate', 'num': msg['num'], 'data': {field: story_settings.worldinfo[num][field] for field in fields}}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wifolderupdate'):
        setgamesaved(False)
        uid = int(msg['uid'])
        fields = ("name", "collapsed")
        for field in fields:
            if(field in msg['data'] and type(msg['data'][field]) is (str if field != "collapsed" else bool)):
                story_settings.wifolders_d[uid][field] = msg['data'][field]
        emit('from_server', {'cmd': 'wifolderupdate', 'uid': msg['uid'], 'data': {field: story_settings.wifolders_d[uid][field] for field in fields}}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiselon'):
        setgamesaved(False)
        story_settings.worldinfo[msg['data']]["selective"] = True
        emit('from_server', {'cmd': 'wiselon', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiseloff'):
        setgamesaved(False)
        story_settings.worldinfo[msg['data']]["selective"] = False
        emit('from_server', {'cmd': 'wiseloff', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiconstanton'):
        setgamesaved(False)
        story_settings.worldinfo[msg['data']]["constant"] = True
        emit('from_server', {'cmd': 'wiconstanton', 'data': msg['data']}, broadcast=True, room="UI_1")
    elif(msg['cmd'] == 'wiconstantoff'):
        setgamesaved(False)
        story_settings.worldinfo[msg['data']]["constant"] = False
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
        emit('from_server', {'cmd': 'buildsamplers', 'data': model_settings.sampler_order}, room="UI_1")
    elif(msg['cmd'] == 'usloaded'):
        system_settings.userscripts = []
        for userscript in msg['data']:
            if type(userscript) is not str:
                continue
            userscript = userscript.strip()
            if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                system_settings.userscripts.append(userscript)
        settingschanged()
    elif(msg['cmd'] == 'usload'):
        load_lua_scripts()
        unloaded, loaded = getuslist()
        sendUSStatItems()
    elif(msg['cmd'] == 'samplers'):
        sampler_order = msg["data"]
        if(not isinstance(sampler_order, list)):
            raise ValueError(f"Sampler order must be a list, but got a {type(sampler_order)}")
        if(len(sampler_order) != len(model_settings.sampler_order)):
            raise ValueError(f"Sampler order must be a list of length {len(model_settings.sampler_order)}, but got a list of length {len(sampler_order)}")
        if(not all(isinstance(e, int) for e in sampler_order)):
            raise ValueError(f"Sampler order must be a list of ints, but got a list with at least one non-int element")
        model_settings.sampler_order = sampler_order
        settingschanged()
    elif(msg['cmd'] == 'list_model'):
        sendModelSelection(menu=msg['data'])
    elif(msg['cmd'] == 'load_model'):
        if not os.path.exists("settings/"):
            os.mkdir("settings")
        changed = True
        if not utils.HAS_ACCELERATE:
            msg['disk_layers'] = "0"
        if os.path.exists("settings/" + model_settings.model.replace('/', '_') + ".breakmodel"):
            with open("settings/" + model_settings.model.replace('/', '_') + ".breakmodel", "r") as file:
                data = file.read().split('\n')[:2]
                if len(data) < 2:
                    data.append("0")
                gpu_layers, disk_layers = data
                if gpu_layers == msg['gpu_layers'] and disk_layers == msg['disk_layers']:
                    changed = False
        if changed:
            f = open("settings/" + model_settings.model.replace('/', '_') + ".breakmodel", "w")
            f.write(msg['gpu_layers'] + '\n' + msg['disk_layers'])
            f.close()
        model_settings.colaburl = msg['url'] + "/request"
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
            if 'folder' not in msg or system_settings.host:
                folder = "./models"
            else:
                folder = msg['folder']
            sendModelSelection(menu=msg['data'], folder=folder)
        elif msg['data'] in ('NeoCustom', 'GPT2Custom') and 'path_modelname' in msg:
            #Here the user entered custom text in the text box. This could be either a model name or a path.
            if check_if_dir_is_model(msg['path_modelname']):
                model_settings.model = msg['data']
                model_settings.custmodpth = msg['path_modelname']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                model_settings.model = msg['path_modelname']
                try:
                    get_model_info(model_settings.model)
                except:
                    emit('from_server', {'cmd': 'errmsg', 'data': "The model entered doesn't exist."}, room="UI_1")
        elif msg['data'] in ('NeoCustom', 'GPT2Custom'):
            if check_if_dir_is_model(msg['path']):
                model_settings.model = msg['data']
                model_settings.custmodpth = msg['path']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                if system_settings.host:
                    sendModelSelection(menu=msg['data'], folder="./models")
                else:
                    sendModelSelection(menu=msg['data'], folder=msg['path'])
        else:
            model_settings.model = msg['data']
            if 'path' in msg:
                model_settings.custmodpth = msg['path']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                get_model_info(model_settings.model)
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
        get_oai_models(msg['key'])
    elif(msg['cmd'] == 'loadselect'):
        user_settings.loadselect = msg["data"]
    elif(msg['cmd'] == 'spselect'):
        user_settings.spselect = msg["data"]
    elif(msg['cmd'] == 'loadrequest'):
        loadRequest(fileops.storypath(user_settings.loadselect))
    elif(msg['cmd'] == 'sprequest'):
        spRequest(user_settings.spselect)
    elif(msg['cmd'] == 'deletestory'):
        deletesave(msg['data'])
    elif(msg['cmd'] == 'renamestory'):
        renamesave(msg['data'], msg['newname'])
    elif(msg['cmd'] == 'clearoverwrite'):    
        user_settings.svowname = ""
        user_settings.saveow   = False
    elif(msg['cmd'] == 'seqsel'):
        selectsequence(msg['data'])
    elif(msg['cmd'] == 'seqpin'):
        pinsequence(msg['data'])
    elif(msg['cmd'] == 'setnumseq'):
        model_settings.numseqs = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelnumseq', 'data': msg['data']}, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setwidepth'):
        user_settings.widepth = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelwidepth', 'data': msg['data']}, room="UI_1")
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setuseprompt'):
        story_settings.useprompt = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setadventure'):
        story_settings.adventure = msg['data']
        story_settings.chatmode = False
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'autosave'):
        user_settings.autosave = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setchatmode'):
        story_settings.chatmode = msg['data']
        story_settings.adventure = False
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setdynamicscan'):
        story_settings.dynamicscan = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setnopromptgen'):
        user_settings.nopromptgen = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setrngpersist'):
        user_settings.rngpersist = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setnogenmod'):
        user_settings.nogenmod = msg['data']
        settingschanged()
        refresh_settings()
    elif(not system_settings.host and msg['cmd'] == 'importwi'):
        wiimportrequest()
    elif(msg['cmd'] == 'debug'):
        user_settings.debug = msg['data']
        emit('from_server', {'cmd': 'set_debug', 'data': msg['data']}, broadcast=True, room="UI_1")
        if user_settings.debug:
            send_debug()

#==================================================================#
#  Send userscripts list to client
#==================================================================#
def sendUSStatItems():
    _, loaded = getuslist()
    loaded = loaded if system_settings.lua_running else []
    last_userscripts = [e["filename"] for e in loaded]
    emit('from_server', {'cmd': 'usstatitems', 'data': loaded, 'flash': last_userscripts != system_settings.last_userscripts}, broadcast=True, room="UI_1")
    system_settings.last_userscripts = last_userscripts

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
    if(system_settings.welcome):
        txt = kml(system_settings.welcome) + "<br/>"
    else:
        txt = "<span>Welcome to <span class=\"color_cyan\">KoboldAI</span>! You are running <span class=\"color_green\">"+getmodelname()+"</span>.<br/>"
    if(not system_settings.noai and not system_settings.welcome):
        txt = txt + "Please load a game or enter a prompt below to begin!</span>"
    if(system_settings.noai):
        txt = txt + "Please load or import a story to read. There is no AI in this mode."
    emit('from_server', {'cmd': 'updatescreen', 'gamestarted': story_settings.gamestarted, 'data': txt}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setgamestate', 'data': 'start'}, broadcast=True, room="UI_1")

#==================================================================#
#  Transmit applicable settings to SocketIO to build UI sliders/toggles
#==================================================================#
def sendsettings():
    # Send settings for selected AI type
    emit('from_server', {'cmd': 'reset_menus'}, room="UI_1")
    if(model_settings.model != "InferKit"):
        for set in gensettings.gensettingstf:
            emit('from_server', {'cmd': 'addsetting', 'data': set}, room="UI_1")
    else:
        for set in gensettings.gensettingsik:
            emit('from_server', {'cmd': 'addsetting', 'data': set}, room="UI_1")
    
    # Send formatting options
    for frm in gensettings.formatcontrols:
        emit('from_server', {'cmd': 'addformat', 'data': frm}, room="UI_1")
        # Add format key to vars if it wasn't loaded with client.settings
        if(not frm["id"] in user_settings.formatoptns):
            user_settings.formatoptns[frm["id"]] = False;

#==================================================================#
#  Set value of gamesaved
#==================================================================#
def setgamesaved(gamesaved):
    assert type(gamesaved) is bool
    if(gamesaved != story_settings.gamesaved):
        emit('from_server', {'cmd': 'gamesaved', 'data': gamesaved}, broadcast=True, room="UI_1")
    story_settings.gamesaved = gamesaved

#==================================================================#
#  Take input text from SocketIO and decide what to do with it
#==================================================================#

def check_for_backend_compilation():
    if(system_settings.checking):
        return
    system_settings.checking = True
    for _ in range(31):
        time.sleep(0.06276680299820175)
        if(system_settings.compiling):
            emit('from_server', {'cmd': 'warnmsg', 'data': 'Compiling TPU backend&mdash;this usually takes 1&ndash;2 minutes...'}, broadcast=True, room="UI_1")
            break
    system_settings.checking = False

def actionsubmit(data, actionmode=0, force_submit=False, force_prompt_gen=False, disable_recentrng=False):
    # Ignore new submissions if the AI is currently busy
    if(system_settings.aibusy):
        return
    
    while(True):
        set_aibusy(1)

        if(disable_recentrng):
            story_settings.recentrng = story_settings.recentrngm = None

        story_settings.recentback = False
        story_settings.recentedit = False
        story_settings.actionmode = actionmode

        # "Action" mode
        if(actionmode == 1):
            data = data.strip().lstrip('>')
            data = re.sub(r'\n+', ' ', data)
            if(len(data)):
                data = f"\n\n> {data}\n"
        
        # "Chat" mode
        if(story_settings.chatmode and story_settings.gamestarted):
            data = re.sub(r'\n+', ' ', data)
            if(len(data)):
                data = f"\n{story_settings.chatname}: {data}\n"
        
        # If we're not continuing, store a copy of the raw input
        if(data != ""):
            story_settings.lastact = data
        
        if(not story_settings.gamestarted):
            story_settings.submission = data
            execute_inmod()
            data = story_settings.submission
            if(not force_submit and len(data.strip()) == 0):
                assert False
            # Start the game
            story_settings.gamestarted = True
            if(not system_settings.noai and system_settings.lua_koboldbridge.generating and (not user_settings.nopromptgen or force_prompt_gen)):
                # Save this first action as the prompt
                story_settings.prompt = data
                # Clear the startup text from game screen
                emit('from_server', {'cmd': 'updatescreen', 'gamestarted': False, 'data': 'Please wait, generating story...'}, broadcast=True, room="UI_1")
                calcsubmit(data) # Run the first action through the generator
                if(not system_settings.abort and system_settings.lua_koboldbridge.restart_sequence is not None and len(story_settings.genseqs) == 0):
                    data = ""
                    force_submit = True
                    disable_recentrng = True
                    continue
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True, room="UI_1")
                break
            else:
                # Save this first action as the prompt
                story_settings.prompt = data if len(data) > 0 else '"'
                for i in range(model_settings.numseqs):
                    system_settings.lua_koboldbridge.outputs[i+1] = ""
                execute_outmod()
                system_settings.lua_koboldbridge.regeneration_required = False
                genout = []
                for i in range(model_settings.numseqs):
                    genout.append({"generated_text": system_settings.lua_koboldbridge.outputs[i+1]})
                    assert type(genout[-1]["generated_text"]) is str
                story_settings.actions.clear_unused_options()
                story_settings.actions.append_options([x["generated_text"] for x in genout])
                genout = [{"generated_text": x['text']} for x in story_settings.actions.get_current_options()]
                if(len(genout) == 1):
                    genresult(genout[0]["generated_text"], flash=False)
                    refresh_story()
                    if(len(story_settings.actions) > 0):
                        emit('from_server', {'cmd': 'texteffect', 'data': story_settings.actions.get_last_key() + 1}, broadcast=True, room="UI_1")
                    if(not system_settings.abort and system_settings.lua_koboldbridge.restart_sequence is not None):
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                else:
                    if(not system_settings.abort and system_settings.lua_koboldbridge.restart_sequence is not None and system_settings.lua_koboldbridge.restart_sequence > 0):
                        genresult(genout[system_settings.lua_koboldbridge.restart_sequence-1]["generated_text"], flash=False)
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
            if(story_settings.actionmode == 0):
                data = applyinputformatting(data)
            story_settings.submission = data
            execute_inmod()
            data = story_settings.submission
            # Dont append submission if it's a blank/continue action
            if(data != ""):
                # Store the result in the Action log
                if(len(story_settings.prompt.strip()) == 0):
                    story_settings.prompt = data
                else:
                    story_settings.actions.append(data)
                update_story_chunk('last')
                send_debug()

            if(not system_settings.noai and system_settings.lua_koboldbridge.generating):
                # Off to the tokenizer!
                calcsubmit(data)
                if(not system_settings.abort and system_settings.lua_koboldbridge.restart_sequence is not None and len(story_settings.genseqs) == 0):
                    data = ""
                    force_submit = True
                    disable_recentrng = True
                    continue
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True, room="UI_1")
                break
            else:
                for i in range(model_settings.numseqs):
                    system_settings.lua_koboldbridge.outputs[i+1] = ""
                execute_outmod()
                system_settings.lua_koboldbridge.regeneration_required = False
                genout = []
                for i in range(model_settings.numseqs):
                    genout.append({"generated_text": system_settings.lua_koboldbridge.outputs[i+1]})
                    assert type(genout[-1]["generated_text"]) is str
                story_settings.actions.clear_unused_options()
                story_settings.actions.append_options([x["generated_text"] for x in genout])
                genout = [{"generated_text": x['text']} for x in story_settings.actions.get_current_options()]
                if(len(genout) == 1):
                    genresult(genout[0]["generated_text"])
                    if(not system_settings.abort and system_settings.lua_koboldbridge.restart_sequence is not None):
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                else:
                    if(not system_settings.abort and system_settings.lua_koboldbridge.restart_sequence is not None and system_settings.lua_koboldbridge.restart_sequence > 0):
                        genresult(genout[system_settings.lua_koboldbridge.restart_sequence-1]["generated_text"])
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                    genselect(genout)
                set_aibusy(0)
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True, room="UI_1")
                break

#==================================================================#
#  
#==================================================================#
def actionretry(data):
    if(system_settings.noai):
        emit('from_server', {'cmd': 'errmsg', 'data': "Retry function unavailable in Read Only mode."}, room="UI_1")
        return
    if(story_settings.recentrng is not None):
        if(not system_settings.aibusy):
            randomGameRequest(story_settings.recentrng, memory=story_settings.recentrngm)
        return
    if actionback():
        actionsubmit("", actionmode=story_settings.actionmode, force_submit=True)
        send_debug()
    elif(not story_settings.useprompt):
        emit('from_server', {'cmd': 'errmsg', 'data': "Please enable \"Always Add Prompt\" to retry with your prompt."}, room="UI_1")

#==================================================================#
#  
#==================================================================#
def actionback():
    if(system_settings.aibusy):
        return
    # Remove last index of actions and refresh game screen
    if(len(story_settings.genseqs) == 0 and len(story_settings.actions) > 0):
        last_key = story_settings.actions.get_last_key()
        story_settings.actions.pop()
        story_settings.recentback = True
        remove_story_chunk(last_key + 1)
        success = True
    elif(len(story_settings.genseqs) == 0):
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."}, room="UI_1")
        success =  False
    else:
        story_settings.genseqs = []
        success = True
    send_debug()
    return success
        
def actionredo():
    genout = [[x['text'], "redo" if x['Previous Selection'] else "pinned" if x['Pinned'] else "normal"] for x in story_settings.actions.get_redo_options()]
    if len(genout) == 0:
        emit('from_server', {'cmd': 'popuperror', 'data': "There's nothing to redo"}, broadcast=True, room="UI_1")
    elif len(genout) == 1:
        genresult(genout[0][0], flash=True, ignore_formatting=True)
    else:
        story_settings.genseqs = [{"generated_text": x[0]} for x in genout]
        emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#  
#==================================================================#
def calcsubmitbudgetheader(txt, **kwargs):
    # Scan for WorldInfo matches
    winfo, found_entries = checkworldinfo(txt, **kwargs)

    # Add a newline to the end of memory
    if(story_settings.memory != "" and story_settings.memory[-1] != "\n"):
        mem = story_settings.memory + "\n"
    else:
        mem = story_settings.memory

    # Build Author's Note if set
    if(story_settings.authornote != ""):
        anotetxt  = ("\n" + story_settings.authornotetemplate + "\n").replace("<|>", story_settings.authornote)
    else:
        anotetxt = ""

    return winfo, mem, anotetxt, found_entries

def calcsubmitbudget(actionlen, winfo, mem, anotetxt, actions, submission=None, budget_deduction=0):
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    anotetkns    = []  # Placeholder for Author's Note tokens
    lnanote      = 0   # Placeholder for Author's Note length

    lnsp = system_settings.sp_length

    if("tokenizer" not in globals()):
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", revision=model_settings.revision, cache_dir="cache")

    lnheader = len(tokenizer._koboldai_header)

    # Calculate token budget
    prompttkns = tokenizer.encode(utils.encodenewlines(system_settings.comregex_ai.sub('', story_settings.prompt)), max_length=int(2e9), truncation=True)
    lnprompt   = len(prompttkns)

    memtokens = tokenizer.encode(utils.encodenewlines(mem), max_length=int(2e9), truncation=True)
    lnmem     = len(memtokens)
    if(lnmem > model_settings.max_length - lnheader - lnsp - model_settings.genamt - budget_deduction):
        raise OverflowError("The memory in your story is too long. Please either write a shorter memory text or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    witokens  = tokenizer.encode(utils.encodenewlines(winfo), max_length=int(2e9), truncation=True)
    lnwi      = len(witokens)
    if(lnmem + lnwi > model_settings.max_length - lnheader - lnsp - model_settings.genamt - budget_deduction):
        raise OverflowError("The current active world info keys take up too many tokens. Please either write shorter world info, decrease World Info Depth or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    if(anotetxt != ""):
        anotetkns = tokenizer.encode(utils.encodenewlines(anotetxt), max_length=int(2e9), truncation=True)
        lnanote   = len(anotetkns)
        if(lnmem + lnwi + lnanote > model_settings.max_length - lnheader - lnsp - model_settings.genamt - budget_deduction):
            raise OverflowError("The author's note in your story is too long. Please either write a shorter author's note or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    if(story_settings.useprompt):
        budget = model_settings.max_length - lnsp - lnprompt - lnmem - lnanote - lnwi - model_settings.genamt - budget_deduction
    else:
        budget = model_settings.max_length - lnsp - lnmem - lnanote - lnwi - model_settings.genamt - budget_deduction

    lnsubmission = len(tokenizer.encode(utils.encodenewlines(system_settings.comregex_ai.sub('', submission)), max_length=int(2e9), truncation=True)) if submission is not None else 0
    maybe_lnprompt = lnprompt if story_settings.useprompt and actionlen > 0 else 0

    if(lnmem + lnwi + lnanote + maybe_lnprompt + lnsubmission > model_settings.max_length - lnheader - lnsp - model_settings.genamt - budget_deduction):
        raise OverflowError("Your submission is too long. Please either write a shorter submission or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt. If you are using the Always Add Prompt setting, turning it off may help.")

    assert budget >= 0

    if(actionlen == 0):
        # First/Prompt action
        tokens = tokenizer._koboldai_header + memtokens + witokens + anotetkns + prompttkns
        assert len(tokens) <= model_settings.max_length - lnsp - model_settings.genamt - budget_deduction
        ln = len(tokens) + lnsp
        return tokens, ln+1, ln+model_settings.genamt
    else:
        tokens     = []
        
        # Check if we have the action depth to hit our A.N. depth
        if(anotetxt != "" and actionlen < story_settings.andepth):
            forceanote = True
        
        # Get most recent action tokens up to our budget
        n = 0
        for key in reversed(actions):
            chunk = system_settings.comregex_ai.sub('', actions[key])
            
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
                tokens = acttkns[count:] + tokens
                budget = 0
                break
            
            # Inject Author's Note if we've reached the desired depth
            if(n == story_settings.andepth-1):
                if(anotetxt != ""):
                    tokens = anotetkns + tokens # A.N. len already taken from bdgt
                    anoteadded = True
            n += 1
        
        # If we're not using the prompt every time and there's still budget left,
        # add some prompt.
        if(not story_settings.useprompt):
            if(budget > 0):
                prompttkns = prompttkns[-budget:]
            else:
                prompttkns = []

        # Did we get to add the A.N.? If not, do it here
        if(anotetxt != ""):
            if((not anoteadded) or forceanote):
                tokens = tokenizer._koboldai_header + memtokens + witokens + anotetkns + prompttkns + tokens
            else:
                tokens = tokenizer._koboldai_header + memtokens + witokens + prompttkns + tokens
        else:
            # Prepend Memory, WI, and Prompt before action tokens
            tokens = tokenizer._koboldai_header + memtokens + witokens + prompttkns + tokens

        # Send completed bundle to generator
        assert len(tokens) <= model_settings.max_length - lnsp - model_settings.genamt - budget_deduction
        ln = len(tokens) + lnsp
        return tokens, ln+1, ln+model_settings.genamt

#==================================================================#
# Take submitted text and build the text to be given to generator
#==================================================================#
def calcsubmit(txt):
    anotetxt     = ""    # Placeholder for Author's Note text
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    actionlen    = len(story_settings.actions)

    winfo, mem, anotetxt, found_entries = calcsubmitbudgetheader(txt)
 
    # For all transformers models
    if(model_settings.model != "InferKit"):
        subtxt, min, max = calcsubmitbudget(actionlen, winfo, mem, anotetxt, story_settings.actions, submission=txt)
        if(actionlen == 0):
            if(not system_settings.use_colab_tpu and model_settings.model not in ["Colab", "OAI", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(model_settings.model == "Colab"):
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(model_settings.model == "OAI"):
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(system_settings.use_colab_tpu or model_settings.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
                tpumtjgenerate(subtxt, min, max, found_entries=found_entries)
        else:
            if(not system_settings.use_colab_tpu and model_settings.model not in ["Colab", "OAI", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(model_settings.model == "Colab"):
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(model_settings.model == "OAI"):
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(system_settings.use_colab_tpu or model_settings.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
                tpumtjgenerate(subtxt, min, max, found_entries=found_entries)
                    
    # For InferKit web API
    else:
        # Check if we have the action depth to hit our A.N. depth
        if(anotetxt != "" and actionlen < story_settings.andepth):
            forceanote = True
        
        if(story_settings.useprompt):
            budget = model_settings.ikmax - len(system_settings.comregex_ai.sub('', story_settings.prompt)) - len(anotetxt) - len(mem) - len(winfo) - 1
        else:
            budget = model_settings.ikmax - len(anotetxt) - len(mem) - len(winfo) - 1
            
        subtxt = ""
        prompt = system_settings.comregex_ai.sub('', story_settings.prompt)
        n = 0
        for key in reversed(story_settings.actions):
            chunk = story_settings.actions[key]
            
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
            if(not story_settings.useprompt):
                if(budget > 0):
                    prompt = system_settings.comregex_ai.sub('', story_settings.prompt)[-budget:]
                else:
                    prompt = ""
            
            # Inject Author's Note if we've reached the desired depth
            if(n == story_settings.andepth-1):
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
    gen_in = torch.tensor(txt, dtype=torch.long)[None]
    if(system_settings.sp is not None):
        soft_tokens = torch.arange(
            model.config.vocab_size,
            model.config.vocab_size + system_settings.sp.shape[0],
        )
        gen_in = torch.cat((soft_tokens[None], gen_in), dim=-1)
    assert gen_in.shape[-1] + model_settings.genamt <= model_settings.max_length

    if(system_settings.hascuda and system_settings.usegpu):
        gen_in = gen_in.to(system_settings.gpu_device)
    elif(system_settings.hascuda and system_settings.breakmodel):
        gen_in = gen_in.to(breakmodel.primary_device)
    else:
        gen_in = gen_in.to('cpu')

    model.kai_scanner_excluded_world_info = found_entries

    story_settings._actions = story_settings.actions
    story_settings._prompt = story_settings.prompt
    if(story_settings.dynamicscan):
        story_settings._actions = story_settings._actions.copy()

    with torch.no_grad():
        already_generated = 0
        numseqs = model_settings.numseqs
        while True:
            genout = generator(
                gen_in, 
                do_sample=True, 
                max_length=int(2e9),
                repetition_penalty=1.1,
                bad_words_ids=model_settings.badwordsids,
                use_cache=True,
                num_return_sequences=numseqs
                )
            already_generated += len(genout[0]) - len(gen_in[0])
            assert already_generated <= model_settings.genamt
            if(model.kai_scanner.halt or not model.kai_scanner.regeneration_required):
                break
            assert genout.ndim >= 2
            assert genout.shape[0] == model_settings.numseqs
            if(system_settings.lua_koboldbridge.generated_cols and model_settings.generated_tkns != system_settings.lua_koboldbridge.generated_cols):
                raise RuntimeError("Inconsistency detected between KoboldAI Python and Lua backends")
            if(already_generated != model_settings.generated_tkns):
                raise RuntimeError("WI scanning error")
            for r in range(model_settings.numseqs):
                for c in range(already_generated):
                    assert system_settings.lua_koboldbridge.generated[r+1][c+1] is not None
                    genout[r][genout.shape[-1] - already_generated + c] = system_settings.lua_koboldbridge.generated[r+1][c+1]
            encoded = []
            for i in range(model_settings.numseqs):
                txt = utils.decodenewlines(tokenizer.decode(genout[i, -already_generated:]))
                winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True, actions=story_settings._actions)
                found_entries[i].update(_found_entries)
                txt, _, _ = calcsubmitbudget(len(story_settings._actions), winfo, mem, anotetxt, story_settings._actions, submission=txt)
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
            if(system_settings.sp is not None):
                soft_tokens = torch.arange(
                    model.config.vocab_size,
                    model.config.vocab_size + system_settings.sp.shape[0],
                    device=genout.device,
                )
                genout = torch.cat((soft_tokens.tile(model_settings.numseqs, 1), genout), dim=-1)
            assert genout.shape[-1] + model_settings.genamt - already_generated <= model_settings.max_length
            diff = genout.shape[-1] - gen_in.shape[-1]
            minimum += diff
            maximum += diff
            gen_in = genout
            numseqs = 1
    
    return genout, already_generated
    

def generate(txt, minimum, maximum, found_entries=None):    
    model_settings.generated_tkns = 0

    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(model_settings.numseqs))

    if not system_settings.quiet:
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, utils.decodenewlines(tokenizer.decode(txt)), colors.END))

    # Store context in memory to use it for comparison with generated content
    story_settings.lastctx = utils.decodenewlines(tokenizer.decode(txt))

    # Clear CUDA cache if using GPU
    if(system_settings.hascuda and (system_settings.usegpu or system_settings.breakmodel)):
        gc.collect()
        torch.cuda.empty_cache()

    # Submit input text to generator
    try:
        genout, already_generated = tpool.execute(_generate, txt, minimum, maximum, found_entries)
    except Exception as e:
        if(issubclass(type(e), lupa.LuaError)):
            system_settings.lua_koboldbridge.obliterate_multiverse()
            system_settings.lua_running = False
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
            sendUSStatItems()
            print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
            print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
            print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        else:
            emit('from_server', {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'}, broadcast=True, room="UI_1")
            print("{0}{1}{2}".format(colors.RED, traceback.format_exc().replace("\033", ""), colors.END), file=sys.stderr)
        set_aibusy(0)
        return

    for i in range(model_settings.numseqs):
        system_settings.lua_koboldbridge.generated[i+1][model_settings.generated_tkns] = int(genout[i, -1].item())
        system_settings.lua_koboldbridge.outputs[i+1] = utils.decodenewlines(tokenizer.decode(genout[i, -already_generated:]))

    execute_outmod()
    if(system_settings.lua_koboldbridge.regeneration_required):
        system_settings.lua_koboldbridge.regeneration_required = False
        genout = []
        for i in range(model_settings.numseqs):
            genout.append({"generated_text": system_settings.lua_koboldbridge.outputs[i+1]})
            assert type(genout[-1]["generated_text"]) is str
    else:
        genout = [{"generated_text": utils.decodenewlines(tokenizer.decode(tokens[-already_generated:]))} for tokens in genout]
    
    story_settings.actions.clear_unused_options()
    story_settings.actions.append_options([x["generated_text"] for x in genout])
    genout = [{"generated_text": x['text']} for x in story_settings.actions.get_current_options()]
    if(len(genout) == 1):
        genresult(genout[0]["generated_text"])
    else:
        if(system_settings.lua_koboldbridge.restart_sequence is not None and system_settings.lua_koboldbridge.restart_sequence > 0):
            genresult(genout[system_settings.lua_koboldbridge.restart_sequence-1]["generated_text"])
        else:
            genselect(genout)
    
    # Clear CUDA cache again if using GPU
    if(system_settings.hascuda and (system_settings.usegpu or system_settings.breakmodel)):
        del genout
        gc.collect()
        torch.cuda.empty_cache()
    
    set_aibusy(0)

#==================================================================#
#  Deal with a single return sequence from generate()
#==================================================================#
def genresult(genout, flash=True, ignore_formatting=False):
    if not system_settings.quiet:
        print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
    
    # Format output before continuing
    if not ignore_formatting:
        genout = applyoutputformatting(genout)

    system_settings.lua_koboldbridge.feedback = genout

    if(len(genout) == 0):
        return
    
    # Add formatted text to Actions array and refresh the game screen
    if(len(story_settings.prompt.strip()) == 0):
        story_settings.prompt = genout
    else:
        story_settings.actions.append(genout)
    update_story_chunk('last')
    if(flash):
        emit('from_server', {'cmd': 'texteffect', 'data': story_settings.actions.get_last_key() + 1 if len(story_settings.actions) else 0}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#  Send generator sequences to the UI for selection
#==================================================================#
def genselect(genout):
    i = 0
    for result in genout:
        # Apply output formatting rules to sequences
        result["generated_text"] = applyoutputformatting(result["generated_text"])
        if not system_settings.quiet:
            print("{0}[Result {1}]\n{2}{3}".format(colors.CYAN, i, result["generated_text"], colors.END))
        i += 1
    
    
    # Store sequences in memory until selection is made
    story_settings.genseqs = genout
    
    
    genout = story_settings.actions.get_current_options_no_edits(ui=1)

    # Send sequences to UI for selection
    emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True, room="UI_1")
    send_debug()

#==================================================================#
#  Send selected sequence to action log and refresh UI
#==================================================================#
def selectsequence(n):
    if(len(story_settings.genseqs) == 0):
        return
    system_settings.lua_koboldbridge.feedback = story_settings.genseqs[int(n)]["generated_text"]
    if(len(system_settings.lua_koboldbridge.feedback) != 0):
        story_settings.actions.append(system_settings.lua_koboldbridge.feedback)
        update_story_chunk('last')
        emit('from_server', {'cmd': 'texteffect', 'data': story_settings.actions.get_last_key() + 1 if len(story_settings.actions) else 0}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True, room="UI_1")
    story_settings.genseqs = []

    if(system_settings.lua_koboldbridge.restart_sequence is not None):
        actionsubmit("", actionmode=story_settings.actionmode, force_submit=True, disable_recentrng=True)
    send_debug()

#==================================================================#
#  Pin/Unpin the selected sequence
#==================================================================#
def pinsequence(n):
    if n.isnumeric():
        story_settings.actions.toggle_pin(story_settings.actions.get_last_key()+1, int(n))
        text = story_settings.genseqs[int(n)]['generated_text']
    send_debug()


#==================================================================#
#  Send transformers-style request to ngrok/colab host
#==================================================================#
def sendtocolab(txt, min, max):
    # Log request to console
    if not system_settings.quiet:
        print("{0}Tokens:{1}, Txt:{2}{3}".format(colors.YELLOW, min-1, txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    story_settings.lastctx = txt
    
    # Build request JSON data
    reqdata = {
        'text': txt,
        'min': min,
        'max': max,
        'rep_pen': model_settings.rep_pen,
        'rep_pen_slope': model_settings.rep_pen_slope,
        'rep_pen_range': model_settings.rep_pen_range,
        'temperature': model_settings.temp,
        'top_p': model_settings.top_p,
        'top_k': model_settings.top_k,
        'tfs': model_settings.tfs,
        'typical': model_settings.typical,
        'topa': model_settings.top_a,
        'numseqs': model_settings.numseqs,
        'retfultxt': False
    }
    
    # Create request
    req = requests.post(
        model_settings.colaburl, 
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
        
        for i in range(model_settings.numseqs):
            system_settings.lua_koboldbridge.outputs[i+1] = genout[i]

        execute_outmod()
        if(system_settings.lua_koboldbridge.regeneration_required):
            system_settings.lua_koboldbridge.regeneration_required = False
            genout = []
            for i in range(model_settings.numseqs):
                genout.append(system_settings.lua_koboldbridge.outputs[i+1])
                assert type(genout[-1]) is str

        story_settings.actions.clear_unused_options()
        story_settings.actions.append_options([x["generated_text"] for x in genout])
        genout = [{"generated_text": x['text']} for x in story_settings.actions.get_current_options()]
        if(len(genout) == 1):
            
            genresult(genout[0])
        else:
            # Convert torch output format to transformers
            seqs = []
            for seq in genout:
                seqs.append({"generated_text": seq})
            if(system_settings.lua_koboldbridge.restart_sequence is not None and system_settings.lua_koboldbridge.restart_sequence > 0):
                genresult(genout[system_settings.lua_koboldbridge.restart_sequence-1]["generated_text"])
            else:
                genselect(genout)
        
        # Format output before continuing
        #genout = applyoutputformatting(getnewcontent(genout))
        
        # Add formatted text to Actions array and refresh the game screen
        #story_settings.actions.append(genout)
        #refresh_story()
        #emit('from_server', {'cmd': 'texteffect', 'data': story_settings.actions.get_last_key() + 1 if len(story_settings.actions) else 0})
        
        set_aibusy(0)
    else:
        errmsg = "Colab API Error: Failed to get a reply from the server. Please check the colab console."
        print("{0}{1}{2}".format(colors.RED, errmsg, colors.END))
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True, room="UI_1")
        set_aibusy(0)

#==================================================================#
#  Send text to TPU mesh transformer backend
#==================================================================#
def tpumtjgenerate(txt, minimum, maximum, found_entries=None):
    model_settings.generated_tkns = 0

    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(model_settings.numseqs))

    if not system_settings.quiet:
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, utils.decodenewlines(tokenizer.decode(txt)), colors.END))

    story_settings._actions = story_settings.actions
    story_settings._prompt = story_settings.prompt
    if(story_settings.dynamicscan):
        story_settings._actions = story_settings._actions.copy()

    # Submit input text to generator
    try:
        soft_tokens = tpumtjgetsofttokens()

        global past

        socketio.start_background_task(copy_current_request_context(check_for_backend_compilation))

        if(story_settings.dynamicscan or (not user_settings.nogenmod and system_settings.has_genmod)):

            context = np.tile(np.uint32(txt), (model_settings.numseqs, 1))
            past = np.empty((model_settings.numseqs, 0), dtype=np.uint32)

            while(True):
                genout, n_generated, regeneration_required, halt = tpool.execute(
                    tpu_mtj_backend.infer_dynamic,
                    context,
                    gen_len = maximum-minimum+1,
                    numseqs=model_settings.numseqs,
                    soft_embeddings=system_settings.sp,
                    soft_tokens=soft_tokens,
                    excluded_world_info=found_entries,
                )

                past = np.pad(past, ((0, 0), (0, n_generated)))
                for r in range(model_settings.numseqs):
                    for c in range(system_settings.lua_koboldbridge.generated_cols):
                        assert system_settings.lua_koboldbridge.generated[r+1][c+1] is not None
                        past[r, c] = system_settings.lua_koboldbridge.generated[r+1][c+1]

                if(system_settings.abort or halt or not regeneration_required):
                    break
                print("(regeneration triggered)")

                encoded = []
                for i in range(model_settings.numseqs):
                    txt = utils.decodenewlines(tokenizer.decode(past[i]))
                    winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True, actions=story_settings._actions)
                    found_entries[i].update(_found_entries)
                    txt, _, _ = calcsubmitbudget(len(story_settings._actions), winfo, mem, anotetxt, story_settings._actions, submission=txt)
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
                temp=model_settings.temp,
                top_p=model_settings.top_p,
                top_k=model_settings.top_k,
                tfs=model_settings.tfs,
                typical=model_settings.typical,
                top_a=model_settings.top_a,
                numseqs=model_settings.numseqs,
                repetition_penalty=model_settings.rep_pen,
                rpslope=model_settings.rep_pen_slope,
                rprange=model_settings.rep_pen_range,
                soft_embeddings=system_settings.sp,
                soft_tokens=soft_tokens,
                sampler_order=model_settings.sampler_order,
            )
            past = genout
            for i in range(model_settings.numseqs):
                system_settings.lua_koboldbridge.generated[i+1] = system_settings.lua_state.table(*genout[i].tolist())
            system_settings.lua_koboldbridge.generated_cols = model_settings.generated_tkns = genout[0].shape[-1]

    except Exception as e:
        if(issubclass(type(e), lupa.LuaError)):
            system_settings.lua_koboldbridge.obliterate_multiverse()
            system_settings.lua_running = False
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True, room="UI_1")
            sendUSStatItems()
            print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
            print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
            print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        else:
            emit('from_server', {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'}, broadcast=True, room="UI_1")
            print("{0}{1}{2}".format(colors.RED, traceback.format_exc().replace("\033", ""), colors.END), file=sys.stderr)
        set_aibusy(0)
        return

    for i in range(model_settings.numseqs):
        system_settings.lua_koboldbridge.outputs[i+1] = utils.decodenewlines(tokenizer.decode(past[i]))
    genout = past

    execute_outmod()
    if(system_settings.lua_koboldbridge.regeneration_required):
        system_settings.lua_koboldbridge.regeneration_required = False
        genout = []
        for i in range(model_settings.numseqs):
            genout.append({"generated_text": system_settings.lua_koboldbridge.outputs[i+1]})
            assert type(genout[-1]["generated_text"]) is str
    else:
        genout = [{"generated_text": utils.decodenewlines(tokenizer.decode(txt))} for txt in genout]

    story_settings.actions.clear_unused_options()
    story_settings.actions.append_options([x["generated_text"] for x in genout])
    genout = [{"generated_text": x['text']} for x in story_settings.actions.get_current_options()]
    if(len(story_settings.actions.get_current_options()) == 1):
        genresult(story_settings.actions.get_current_options()[0])
    else:
        if(system_settings.lua_koboldbridge.restart_sequence is not None and system_settings.lua_koboldbridge.restart_sequence > 0):
            genresult(genout[system_settings.lua_koboldbridge.restart_sequence-1]["generated_text"])
        else:
            genselect([{"generated_text": x} for x in story_settings.actions.get_current_options()])

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
    if(story_settings.lastctx == ""):
        return txt
    
    # Tokenize the last context and the generated content
    ctxtokens = tokenizer.encode(utils.encodenewlines(story_settings.lastctx), max_length=int(2e9), truncation=True)
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
    if(user_settings.formatoptns["frmtadsnsp"]):
        txt = utils.addsentencespacing(txt, story_settings)
 
    return txt

#==================================================================#
# Applies chosen formatting options to text returned from AI
#==================================================================#
def applyoutputformatting(txt):
    # Use standard quotes and apostrophes
    txt = utils.fixquotes(txt)

    # Adventure mode clipping of all characters after '>'
    if(story_settings.adventure):
        txt = system_settings.acregex_ai.sub('', txt)
    
    # Trim incomplete sentences
    if(user_settings.formatoptns["frmttriminc"] and not story_settings.chatmode):
        txt = utils.trimincompletesentence(txt)
    # Replace blank lines
    if(user_settings.formatoptns["frmtrmblln"] or story_settings.chatmode):
        txt = utils.replaceblanklines(txt)
    # Remove special characters
    if(user_settings.formatoptns["frmtrmspch"]):
        txt = utils.removespecialchars(txt, story_settings)
	# Single Line Mode
    if(user_settings.formatoptns["singleline"] or story_settings.chatmode):
        txt = utils.singlelineprocessing(txt, story_settings)
    
    return txt

#==================================================================#
# Sends the current story content to the Game Screen
#==================================================================#
def refresh_story():
    text_parts = ['<chunk n="0" id="n0" tabindex="-1">', system_settings.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), html.escape(story_settings.prompt)), '</chunk>']
    for idx in story_settings.actions:
        item = story_settings.actions[idx]
        idx += 1
        item = html.escape(item)
        item = system_settings.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), item)  # Add special formatting to comments
        item = system_settings.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions
        text_parts.extend(('<chunk n="', str(idx), '" id="n', str(idx), '" tabindex="-1">', item, '</chunk>'))
    emit('from_server', {'cmd': 'updatescreen', 'gamestarted': story_settings.gamestarted, 'data': formatforhtml(''.join(text_parts))}, broadcast=True, room="UI_1")


#==================================================================#
# Signals the Game Screen to update one of the chunks
#==================================================================#
def update_story_chunk(idx: Union[int, str]):
    if idx == 'last':
        if len(story_settings.actions) <= 1:
            # In this case, we are better off just refreshing the whole thing as the
            # prompt might not have been shown yet (with a "Generating story..."
            # message instead).
            refresh_story()
            setgamesaved(False)
            return

        idx = (story_settings.actions.get_last_key() if len(story_settings.actions) else 0) + 1

    if idx == 0:
        text = story_settings.prompt
    else:
        # Actions are 0 based, but in chunks 0 is the prompt.
        # So the chunk index is one more than the corresponding action index.
        if(idx - 1 not in story_settings.actions):
            return
        text = story_settings.actions[idx - 1]

    item = html.escape(text)
    item = system_settings.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), item)  # Add special formatting to comments
    item = system_settings.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions

    chunk_text = f'<chunk n="{idx}" id="n{idx}" tabindex="-1">{formatforhtml(item)}</chunk>'
    emit('from_server', {'cmd': 'updatechunk', 'data': {'index': idx, 'html': chunk_text}}, broadcast=True, room="UI_1")

    setgamesaved(False)

    #If we've set the auto save flag, we'll now save the file
    if user_settings.autosave and (".json" in system_settings.savedir):
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
    
    if(model_settings.model != "InferKit"):
        emit('from_server', {'cmd': 'updatetemp', 'data': model_settings.temp}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopp', 'data': model_settings.top_p}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopk', 'data': model_settings.top_k}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetfs', 'data': model_settings.tfs}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetypical', 'data': model_settings.typical}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopa', 'data': model_settings.top_a}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatereppen', 'data': model_settings.rep_pen}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatereppenslope', 'data': model_settings.rep_pen_slope}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatereppenrange', 'data': model_settings.rep_pen_range}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updateoutlen', 'data': model_settings.genamt}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetknmax', 'data': model_settings.max_length}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatenumseq', 'data': model_settings.numseqs}, broadcast=True, room="UI_1")
    else:
        emit('from_server', {'cmd': 'updatetemp', 'data': model_settings.temp}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updatetopp', 'data': model_settings.top_p}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'updateikgen', 'data': model_settings.ikgen}, broadcast=True, room="UI_1")
    
    emit('from_server', {'cmd': 'updateanotedepth', 'data': story_settings.andepth}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatewidepth', 'data': user_settings.widepth}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateuseprompt', 'data': story_settings.useprompt}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateadventure', 'data': story_settings.adventure}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatechatmode', 'data': story_settings.chatmode}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatedynamicscan', 'data': story_settings.dynamicscan}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updateautosave', 'data': user_settings.autosave}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatenopromptgen', 'data': user_settings.nopromptgen}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updaterngpersist', 'data': user_settings.rngpersist}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatenogenmod', 'data': user_settings.nogenmod}, broadcast=True, room="UI_1")
    
    emit('from_server', {'cmd': 'updatefrmttriminc', 'data': user_settings.formatoptns["frmttriminc"]}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatefrmtrmblln', 'data': user_settings.formatoptns["frmtrmblln"]}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatefrmtrmspch', 'data': user_settings.formatoptns["frmtrmspch"]}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatefrmtadsnsp', 'data': user_settings.formatoptns["frmtadsnsp"]}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'updatesingleline', 'data': user_settings.formatoptns["singleline"]}, broadcast=True, room="UI_1")
    
    # Allow toggle events again
    emit('from_server', {'cmd': 'allowtoggle', 'data': True}, broadcast=True, room="UI_1")

#==================================================================#
#  Sets the logical and display states for the AI Busy condition
#==================================================================#
def set_aibusy(state):
    if(state):
        system_settings.aibusy = True
        emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'}, broadcast=True, room="UI_1")
    else:
        system_settings.aibusy = False
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")

#==================================================================#
# 
#==================================================================#
def editrequest(n):
    if(n == 0):
        txt = story_settings.prompt
    else:
        txt = story_settings.actions[n-1]
    
    story_settings.editln = n
    emit('from_server', {'cmd': 'setinputtext', 'data': txt}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'enablesubmit', 'data': ''}, broadcast=True, room="UI_1")

#==================================================================#
# 
#==================================================================#
def editsubmit(data):
    story_settings.recentedit = True
    if(story_settings.editln == 0):
        story_settings.prompt = data
    else:
        story_settings.actions[story_settings.editln-1] = data
    
    story_settings.mode = "play"
    update_story_chunk(story_settings.editln)
    emit('from_server', {'cmd': 'texteffect', 'data': story_settings.editln}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'editmode', 'data': 'false'}, room="UI_1")
    send_debug()

#==================================================================#
#  
#==================================================================#
def deleterequest():
    story_settings.recentedit = True
    # Don't delete prompt
    if(story_settings.editln == 0):
        # Send error message
        pass
    else:
        story_settings.actions.delete_action(story_settings.editln-1)
        story_settings.mode = "play"
        remove_story_chunk(story_settings.editln)
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, room="UI_1")
    send_debug()

#==================================================================#
# 
#==================================================================#
def inlineedit(chunk, data):
    story_settings.recentedit = True
    chunk = int(chunk)
    if(chunk == 0):
        if(len(data.strip()) == 0):
            return
        story_settings.prompt = data
    else:
        if(chunk-1 in story_settings.actions):
            story_settings.actions[chunk-1] = data
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
    story_settings.recentedit = True
    chunk = int(chunk)
    # Don't delete prompt
    if(chunk == 0):
        # Send error message
        update_story_chunk(chunk)
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."}, room="UI_1")
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    else:
        if(chunk-1 in story_settings.actions):
            story_settings.actions.delete_action(chunk-1)
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
    if(story_settings.mode == "play"):
        story_settings.mode = "memory"
        emit('from_server', {'cmd': 'memmode', 'data': 'true'}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setinputtext', 'data': story_settings.memory}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': story_settings.authornote}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanotetemplate', 'data': story_settings.authornotetemplate}, broadcast=True, room="UI_1")
    elif(story_settings.mode == "memory"):
        story_settings.mode = "play"
        emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True, room="UI_1")

#==================================================================#
#   Toggles the game mode for WI editing and sends UI commands
#==================================================================#
def togglewimode():
    if(story_settings.mode == "play"):
        story_settings.mode = "wi"
        emit('from_server', {'cmd': 'wimode', 'data': 'true'}, broadcast=True, room="UI_1")
    elif(story_settings.mode == "wi"):
        # Commit WI fields first
        requestwi()
        # Then set UI state back to Play
        story_settings.mode = "play"
        emit('from_server', {'cmd': 'wimode', 'data': 'false'}, broadcast=True, room="UI_1")
    sendwi()

#==================================================================#
#   
#==================================================================#
def addwiitem(folder_uid=None):
    assert folder_uid is None or folder_uid in story_settings.wifolders_d
    ob = {"key": "", "keysecondary": "", "content": "", "comment": "", "folder": folder_uid, "num": len(story_settings.worldinfo), "init": False, "selective": False, "constant": False}
    story_settings.worldinfo.append(ob)
    while(True):
        uid = int.from_bytes(os.urandom(4), "little", signed=True)
        if(uid not in story_settings.worldinfo_u):
            break
    story_settings.worldinfo_u[uid] = story_settings.worldinfo[-1]
    story_settings.worldinfo[-1]["uid"] = uid
    if(folder_uid is not None):
        story_settings.wifolders_u[folder_uid].append(story_settings.worldinfo[-1])
    emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True, room="UI_1")

#==================================================================#
#   Creates a new WI folder with an unused cryptographically secure random UID
#==================================================================#
def addwifolder():
    while(True):
        uid = int.from_bytes(os.urandom(4), "little", signed=True)
        if(uid not in story_settings.wifolders_d):
            break
    ob = {"name": "", "collapsed": False}
    story_settings.wifolders_d[uid] = ob
    story_settings.wifolders_l.append(uid)
    story_settings.wifolders_u[uid] = []
    emit('from_server', {'cmd': 'addwifolder', 'uid': uid, 'data': ob}, broadcast=True, room="UI_1")
    addwiitem(folder_uid=uid)

#==================================================================#
#   Move the WI entry with UID src so that it immediately precedes
#   the WI entry with UID dst
#==================================================================#
def movewiitem(dst, src):
    setgamesaved(False)
    if(story_settings.worldinfo_u[src]["folder"] is not None):
        for i, e in enumerate(story_settings.wifolders_u[story_settings.worldinfo_u[src]["folder"]]):
            if(e is story_settings.worldinfo_u[src]):
                story_settings.wifolders_u[story_settings.worldinfo_u[src]["folder"]].pop(i)
                break
    if(story_settings.worldinfo_u[dst]["folder"] is not None):
        story_settings.wifolders_u[story_settings.worldinfo_u[dst]["folder"]].append(story_settings.worldinfo_u[src])
    story_settings.worldinfo_u[src]["folder"] = story_settings.worldinfo_u[dst]["folder"]
    for i, e in enumerate(story_settings.worldinfo):
        if(e is story_settings.worldinfo_u[src]):
            _src = i
        elif(e is story_settings.worldinfo_u[dst]):
            _dst = i
    story_settings.worldinfo.insert(_dst - (_dst >= _src), story_settings.worldinfo.pop(_src))
    sendwi()

#==================================================================#
#   Move the WI folder with UID src so that it immediately precedes
#   the WI folder with UID dst
#==================================================================#
def movewifolder(dst, src):
    setgamesaved(False)
    story_settings.wifolders_l.remove(src)
    if(dst is None):
        # If dst is None, that means we should move src to be the last folder
        story_settings.wifolders_l.append(src)
    else:
        story_settings.wifolders_l.insert(story_settings.wifolders_l.index(dst), src)
    sendwi()

#==================================================================#
#   
#==================================================================#
def sendwi():
    # Cache len of WI
    ln = len(story_settings.worldinfo)

    # Clear contents of WI container
    emit('from_server', {'cmd': 'wistart', 'wifolders_d': story_settings.wifolders_d, 'wifolders_l': story_settings.wifolders_l, 'data': ''}, broadcast=True, room="UI_1")

    # Stable-sort WI entries in order of folder
    stablesortwi()

    story_settings.worldinfo_i = [wi for wi in story_settings.worldinfo if wi["init"]]

    # If there are no WI entries, send an empty WI object
    if(ln == 0):
        addwiitem()
    else:
        # Send contents of WI array
        last_folder = ...
        for wi in story_settings.worldinfo:
            if(wi["folder"] != last_folder):
                emit('from_server', {'cmd': 'addwifolder', 'uid': wi["folder"], 'data': story_settings.wifolders_d[wi["folder"]] if wi["folder"] is not None else None}, broadcast=True, room="UI_1")
                last_folder = wi["folder"]
            ob = wi
            emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True, room="UI_1")
    
    emit('from_server', {'cmd': 'wifinish', 'data': ''}, broadcast=True, room="UI_1")

#==================================================================#
#  Request current contents of all WI HTML elements
#==================================================================#
def requestwi():
    list = []
    for wi in story_settings.worldinfo:
        list.append(wi["num"])
    emit('from_server', {'cmd': 'requestwiitem', 'data': list}, room="UI_1")

#==================================================================#
#  Stable-sort WI items so that items in the same folder are adjacent,
#  and items in different folders are sorted based on the order of the folders
#==================================================================#
def stablesortwi():
    mapping = {uid: index for index, uid in enumerate(story_settings.wifolders_l)}
    story_settings.worldinfo.sort(key=lambda x: mapping[x["folder"]] if x["folder"] is not None else float("inf"))
    last_folder = ...
    last_wi = None
    for i, wi in enumerate(story_settings.worldinfo):
        wi["num"] = i
        wi["init"] = True
        if(wi["folder"] != last_folder):
            if(last_wi is not None and last_folder is not ...):
                last_wi["init"] = False
            last_folder = wi["folder"]
        last_wi = wi
    if(last_wi is not None):
        last_wi["init"] = False
    for folder in story_settings.wifolders_u:
        story_settings.wifolders_u[folder].sort(key=lambda x: x["num"])

#==================================================================#
#  Extract object from server and send it to WI objects
#==================================================================#
def commitwi(ar):
    for ob in ar:
        ob["uid"] = int(ob["uid"])
        story_settings.worldinfo_u[ob["uid"]]["key"]          = ob["key"]
        story_settings.worldinfo_u[ob["uid"]]["keysecondary"] = ob["keysecondary"]
        story_settings.worldinfo_u[ob["uid"]]["content"]      = ob["content"]
        story_settings.worldinfo_u[ob["uid"]]["comment"]      = ob.get("comment", "")
        story_settings.worldinfo_u[ob["uid"]]["folder"]       = ob.get("folder", None)
        story_settings.worldinfo_u[ob["uid"]]["selective"]    = ob["selective"]
        story_settings.worldinfo_u[ob["uid"]]["constant"]     = ob.get("constant", False)
    stablesortwi()
    story_settings.worldinfo_i = [wi for wi in story_settings.worldinfo if wi["init"]]

#==================================================================#
#  
#==================================================================#
def deletewi(uid):
    if(uid in story_settings.worldinfo_u):
        setgamesaved(False)
        # Store UID of deletion request
        story_settings.deletewi = uid
        if(story_settings.deletewi is not None):
            if(story_settings.worldinfo_u[story_settings.deletewi]["folder"] is not None):
                for i, e in enumerate(story_settings.wifolders_u[story_settings.worldinfo_u[story_settings.deletewi]["folder"]]):
                    if(e is story_settings.worldinfo_u[story_settings.deletewi]):
                        story_settings.wifolders_u[story_settings.worldinfo_u[story_settings.deletewi]["folder"]].pop(i)
            for i, e in enumerate(story_settings.worldinfo):
                if(e is story_settings.worldinfo_u[story_settings.deletewi]):
                    del story_settings.worldinfo[i]
                    break
            del story_settings.worldinfo_u[story_settings.deletewi]
            # Send the new WI array structure
            sendwi()
            # And reset deletewi
            story_settings.deletewi = None

#==================================================================#
#  
#==================================================================#
def deletewifolder(uid):
    uid = int(uid)
    del story_settings.wifolders_u[uid]
    del story_settings.wifolders_d[uid]
    del story_settings.wifolders_l[story_settings.wifolders_l.index(uid)]
    setgamesaved(False)
    # Delete uninitialized entries in the folder we're going to delete
    story_settings.worldinfo = [wi for wi in story_settings.worldinfo if wi["folder"] != uid or wi["init"]]
    story_settings.worldinfo_i = [wi for wi in story_settings.worldinfo if wi["init"]]
    # Move WI entries that are inside of the folder we're going to delete
    # so that they're outside of all folders
    for wi in story_settings.worldinfo:
        if(wi["folder"] == uid):
            wi["folder"] = None

    sendwi()

#==================================================================#
#  Look for WI keys in text to generator 
#==================================================================#
def checkworldinfo(txt, allowed_entries=None, allowed_folders=None, force_use_txt=False, scan_story=True, actions=None):
    original_txt = txt

    if(actions is None):
        actions = story_settings.actions

    # Dont go any further if WI is empty
    if(len(story_settings.worldinfo) == 0):
        return "", set()
    
    # Cache actions length
    ln = len(actions)
    
    # Don't bother calculating action history if widepth is 0
    if(user_settings.widepth > 0 and scan_story):
        depth = user_settings.widepth
        # If this is not a continue, add 1 to widepth since submitted
        # text is already in action history @ -1
        if(not force_use_txt and (txt != "" and story_settings.prompt != txt)):
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
            txt = system_settings.comregex_ai.sub('', story_settings.prompt) + "".join(chunks)
        elif(ln == 0):
            txt = system_settings.comregex_ai.sub('', story_settings.prompt)

    if(force_use_txt):
        txt += original_txt

    # Scan text for matches on WI keys
    wimem = ""
    found_entries = set()
    for wi in story_settings.worldinfo:
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
                if(user_settings.wirmvwhtsp):
                    ky = k.strip()
                if ky in txt:
                    if wi.get("selective", False) and len(keys_secondary):
                        found = False
                        for ks in keys_secondary:
                            ksy = ks
                            if(user_settings.wirmvwhtsp):
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
    if(data != story_settings.memory):
        setgamesaved(False)
    story_settings.memory = data
    story_settings.mode = "play"
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
    if(data != story_settings.authornote):
        setgamesaved(False)
    story_settings.authornote = data

    if(story_settings.authornotetemplate != template):
        story_settings.setauthornotetemplate = template
        settingschanged()
    story_settings.authornotetemplate = template

    emit('from_server', {'cmd': 'setanote', 'data': story_settings.authornote}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': story_settings.authornotetemplate}, broadcast=True, room="UI_1")

#==================================================================#
#  Assembles game data into a request to InferKit API
#==================================================================#
def ikrequest(txt):
    # Log request to console
    if not system_settings.quiet:
        print("{0}Len:{1}, Txt:{2}{3}".format(colors.YELLOW, len(txt), txt, colors.END))
    
    # Build request JSON data
    reqdata = {
        'forceNoEnd': True,
        'length': model_settings.ikgen,
        'prompt': {
            'isContinuation': False,
            'text': txt
        },
        'startFromBeginning': False,
        'streamResponse': False,
        'temperature': model_settings.temp,
        'topP': model_settings.top_p
    }
    
    # Create request
    req = requests.post(
        model_settings.url, 
        json    = reqdata,
        headers = {
            'Authorization': 'Bearer '+model_settings.apikey
            }
        )
    
    # Deal with the response
    if(req.status_code == 200):
        genout = req.json()["data"]["text"]

        system_settings.lua_koboldbridge.outputs[1] = genout

        execute_outmod()
        if(system_settings.lua_koboldbridge.regeneration_required):
            system_settings.lua_koboldbridge.regeneration_required = False
            genout = system_settings.lua_koboldbridge.outputs[1]
            assert genout is str

        if not system_settings.quiet:
            print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
        story_settings.actions.append(genout)
        update_story_chunk('last')
        emit('from_server', {'cmd': 'texteffect', 'data': story_settings.actions.get_last_key() + 1 if len(story_settings.actions) else 0}, broadcast=True, room="UI_1")
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
    if not system_settings.quiet:
        print("{0}Len:{1}, Txt:{2}{3}".format(colors.YELLOW, len(txt), txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    story_settings.lastctx = txt
    
    # Build request JSON data
    if 'GooseAI' in args.configname:
        reqdata = {
            'prompt': txt,
            'max_tokens': model_settings.genamt,
            'temperature': model_settings.temp,
            'top_a': model_settings.top_a,
            'top_p': model_settings.top_p,
            'top_k': model_settings.top_k,
            'tfs': model_settings.tfs,
            'typical_p': model_settings.typical,
            'repetition_penalty': model_settings.rep_pen,
            'repetition_penalty_slope': model_settings.rep_pen_slope,
            'repetition_penalty_range': model_settings.rep_pen_range,
            'n': model_settings.numseqs,
            'stream': False
        }
    else:
        reqdata = {
            'prompt': txt,
            'max_tokens': model_settings.genamt,
            'temperature': model_settings.temp,
            'top_p': model_settings.top_p,
            'n': model_settings.numseqs,
            'stream': False
        }
    
    req = requests.post(
        model_settings.oaiurl, 
        json    = reqdata,
        headers = {
            'Authorization': 'Bearer '+model_settings.oaiapikey,
            'Content-Type': 'application/json'
            }
        )
    
    # Deal with the response
    if(req.status_code == 200):
        outputs = [out["text"] for out in req.json()["choices"]]

        for idx in range(len(outputs)):
            system_settings.lua_koboldbridge.outputs[idx+1] = outputs[idx]

        execute_outmod()
        if (system_settings.lua_koboldbridge.regeneration_required):
            system_settings.lua_koboldbridge.regeneration_required = False
            genout = []
            for i in range(len(outputs)):
                genout.append(
                    {"generated_text": system_settings.lua_koboldbridge.outputs[i + 1]})
                assert type(genout[-1]["generated_text"]) is str
        else:
            genout = [
                {"generated_text": utils.decodenewlines(txt)}
                for txt in outputs]

        story_settings.actions.clear_unused_options()
        story_settings.actions.append_options([x["generated_text"] for x in genout])
        genout = [{"generated_text": x['text']} for x in story_settings.actions.get_current_options()]
        if (len(genout) == 1):
            genresult(genout[0]["generated_text"])
        else:
            if (system_settings.lua_koboldbridge.restart_sequence is not None and
                    system_settings.lua_koboldbridge.restart_sequence > 0):
                genresult(genout[system_settings.lua_koboldbridge.restart_sequence - 1][
                              "generated_text"])
            else:
                genselect(genout)

        if not system_settings.quiet:
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
    if(story_settings.mode == "edit"):
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True, room="UI_1")
    elif(story_settings.mode == "memory"):
        emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True, room="UI_1")
    elif(story_settings.mode == "wi"):
        emit('from_server', {'cmd': 'wimode', 'data': 'false'}, broadcast=True, room="UI_1")
    story_settings.mode = "play"

#==================================================================#
#  Launch in-browser save prompt
#==================================================================#
def saveas(data):
    
    name = data['name']
    savepins = data['pins']
    # Check if filename exists already
    name = utils.cleanfilename(name)
    if(not fileops.saveexists(name) or (user_settings.saveow and user_settings.svowname == name)):
        # All clear to save
        e = saveRequest(fileops.storypath(name), savepins=savepins)
        user_settings.saveow = False
        user_settings.svowname = ""
        if(e is None):
            emit('from_server', {'cmd': 'hidesaveas', 'data': ''}, room="UI_1")
        else:
            print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
            emit('from_server', {'cmd': 'popuperror', 'data': str(e)}, room="UI_1")
    else:
        # File exists, prompt for overwrite
        user_settings.saveow   = True
        user_settings.svowname = name
        emit('from_server', {'cmd': 'askforoverwrite', 'data': ''}, room="UI_1")

#==================================================================#
#  Launch in-browser story-delete prompt
#==================================================================#
def deletesave(name):
    name = utils.cleanfilename(name)
    e = fileops.deletesave(name)
    if(e is None):
        if(system_settings.smandelete):
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
    if(not fileops.saveexists(newname) or name == newname or (user_settings.saveow and user_settings.svowname == newname)):
        e = fileops.renamesave(name, newname)
        user_settings.saveow = False
        user_settings.svowname = ""
        if(e is None):
            if(system_settings.smanrename):
                emit('from_server', {'cmd': 'hidepopuprename', 'data': ''}, room="UI_1")
                getloadlist()
            else:
                emit('from_server', {'cmd': 'popuperror', 'data': "The server denied your request to rename this story"}, room="UI_1")
        else:
            print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
            emit('from_server', {'cmd': 'popuperror', 'data': str(e)}, room="UI_1")
    else:
        # File exists, prompt for overwrite
        user_settings.saveow   = True
        user_settings.svowname = newname
        emit('from_server', {'cmd': 'askforoverwrite', 'data': ''}, room="UI_1")

#==================================================================#
#  Save the currently running story
#==================================================================#
def save():
    # Check if a file is currently open
    if(".json" in system_settings.savedir):
        saveRequest(system_settings.savedir)
    else:
        emit('from_server', {'cmd': 'saveas', 'data': ''}, room="UI_1")

#==================================================================#
#  Save the story via file browser
#==================================================================#
def savetofile():
    savpath = fileops.getsavepath(system_settings.savedir, "Save Story As", [("Json", "*.json")])
    saveRequest(savpath)

#==================================================================#
#  Save the story to specified path
#==================================================================#
def saveRequest(savpath, savepins=True):    
    if(savpath):
        # Leave Edit/Memory mode before continuing
        exitModes()
        
        # Save path for future saves
        system_settings.savedir = savpath
        txtpath = os.path.splitext(savpath)[0] + ".txt"
        # Build json to write
        js = {}
        js["gamestarted"] = story_settings.gamestarted
        js["prompt"]      = story_settings.prompt
        js["memory"]      = story_settings.memory
        js["authorsnote"] = story_settings.authornote
        js["anotetemplate"] = story_settings.authornotetemplate
        js["actions"]     = tuple(story_settings.actions.values())
        if savepins:
            js["actions_metadata"]     = story_settings.actions_metadata
        js["worldinfo"]   = []
        js["wifolders_d"] = story_settings.wifolders_d
        js["wifolders_l"] = story_settings.wifolders_l
		
        # Extract only the important bits of WI
        for wi in story_settings.worldinfo_i:
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
                
        txt = story_settings.prompt + "".join(story_settings.actions.values())

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
        user_settings.laststory = filename
        emit('from_server', {'cmd': 'setstoryname', 'data': user_settings.laststory}, broadcast=True, room="UI_1")
        setgamesaved(True)
        print("{0}Story saved to {1}!{2}".format(colors.GREEN, path.basename(savpath), colors.END))

#==================================================================#
#  Show list of saved stories
#==================================================================#
def getloadlist():
    emit('from_server', {'cmd': 'buildload', 'data': fileops.getstoryfiles()}, room="UI_1")

#==================================================================#
#  Show list of soft prompts
#==================================================================#
def getsplist():
    if(system_settings.allowsp):
        emit('from_server', {'cmd': 'buildsp', 'data': fileops.getspfiles(model_settings.modeldim)}, room="UI_1")

#==================================================================#
#  Get list of userscripts
#==================================================================#
def getuslist():
    files = {i: v for i, v in enumerate(fileops.getusfiles())}
    loaded = []
    unloaded = []
    userscripts = set(system_settings.userscripts)
    for i in range(len(files)):
        if files[i]["filename"] not in userscripts:
            unloaded.append(files[i])
    files = {files[k]["filename"]: files[k] for k in files}
    userscripts = set(files.keys())
    for filename in system_settings.userscripts:
        if filename in userscripts:
            loaded.append(files[filename])
    return unloaded, loaded

#==================================================================#
#  Load a saved story via file browser
#==================================================================#
def loadfromfile():
    loadpath = fileops.getloadpath(system_settings.savedir, "Select Story File", [("Json", "*.json")])
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

    # Copy file contents to vars
    story_settings.gamestarted = js["gamestarted"]
    story_settings.prompt      = js["prompt"]
    story_settings.memory      = js["memory"]
    story_settings.worldinfo   = []
    story_settings.worldinfo   = []
    story_settings.worldinfo_u = {}
    story_settings.wifolders_d = {int(k): v for k, v in js.get("wifolders_d", {}).items()}
    story_settings.wifolders_l = js.get("wifolders_l", [])
    story_settings.wifolders_u = {uid: [] for uid in story_settings.wifolders_d}
    story_settings.lastact     = ""
    story_settings.submission  = ""
    story_settings.lastctx     = ""
    story_settings.genseqs = []

    del story_settings.actions
    story_settings.actions = koboldai_settings.KoboldStoryRegister()
    actions = collections.deque(js["actions"])
    

    if "actions_metadata" in js:
        
        if type(js["actions_metadata"]) == dict:
            temp = js["actions_metadata"]
            story_settings.actions_metadata = {}
            #we need to redo the numbering of the actions_metadata since the actions list doesn't preserve it's number on saving
            if len(temp) > 0:
                counter = 0
                temp = {int(k):v for k,v in temp.items()}
                for i in range(max(temp)+1):
                    if i in temp:
                        story_settings.actions_metadata[counter] = temp[i]
                        counter += 1
            del temp
        else:
            #fix if we're using the old metadata format
            story_settings.actions_metadata = {}
            i = 0
            
            for text in js['actions']:
                story_settings.actions_metadata[i] = {'Selected Text': text, 'Alternative Text': []}
                i+=1
    else:
        story_settings.actions_metadata = {}
        i = 0
        
        for text in js['actions']:
            story_settings.actions_metadata[i] = {'Selected Text': text, 'Alternative Text': []}
            i+=1
            

    if(len(story_settings.prompt.strip()) == 0):
        while(len(actions)):
            action = actions.popleft()
            if(len(action.strip()) != 0):
                story_settings.prompt = action
                break
        else:
            story_settings.gamestarted = False
    if(story_settings.gamestarted):
        for s in actions:
            story_settings.actions.append(s)
    
    # Try not to break older save files
    if("authorsnote" in js):
        story_settings.authornote = js["authorsnote"]
    else:
        story_settings.authornote = ""
    if("anotetemplate" in js):
        story_settings.authornotetemplate = js["anotetemplate"]
    else:
        story_settings.authornotetemplate = "[Author's note: <|>]"
    
    if("worldinfo" in js):
        num = 0
        for wi in js["worldinfo"]:
            story_settings.worldinfo.append({
                "key": wi["key"],
                "keysecondary": wi.get("keysecondary", ""),
                "content": wi["content"],
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
                if(uid not in story_settings.worldinfo_u):
                    break
            story_settings.worldinfo_u[uid] = story_settings.worldinfo[-1]
            story_settings.worldinfo[-1]["uid"] = uid
            if(story_settings.worldinfo[-1]["folder"] is not None):
                story_settings.wifolders_u[story_settings.worldinfo[-1]["folder"]].append(story_settings.worldinfo[-1])
            num += 1

    for uid in story_settings.wifolders_l + [None]:
        story_settings.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
        while(True):
            uid = int.from_bytes(os.urandom(4), "little", signed=True)
            if(uid not in story_settings.worldinfo_u):
                break
        story_settings.worldinfo_u[uid] = story_settings.worldinfo[-1]
        story_settings.worldinfo[-1]["uid"] = uid
        if(story_settings.worldinfo[-1]["folder"] is not None):
            story_settings.wifolders_u[story_settings.worldinfo[-1]["folder"]].append(story_settings.worldinfo[-1])
    stablesortwi()
    story_settings.worldinfo_i = [wi for wi in story_settings.worldinfo if wi["init"]]

    # Save path for save button
    system_settings.savedir = loadpath
    
    # Clear loadselect var
    user_settings.loadselect = ""
    
    # Refresh game screen
    _filename = filename
    if(filename.endswith('.json')):
        _filename = filename[:-5]
    user_settings.laststory = _filename
    #set the story_name
    story_settings.story_name = _filename
    emit('from_server', {'cmd': 'setstoryname', 'data': user_settings.laststory}, broadcast=True, room="UI_1")
    setgamesaved(True)
    sendwi()
    emit('from_server', {'cmd': 'setmemory', 'data': story_settings.memory}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanote', 'data': story_settings.authornote}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': story_settings.authornotetemplate}, broadcast=True, room="UI_1")
    refresh_story()
    emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True, room="UI_1")
    print("{0}Story loaded from {1}!{2}".format(colors.GREEN, filename, colors.END))
    
    send_debug()

def load_story_v2(js):
    story_settings.from_json(js)

#==================================================================#
# Import an AIDungon game exported with Mimi's tool
#==================================================================#
def importRequest():
    importpath = fileops.getloadpath(system_settings.savedir, "Select AID CAT File", [("Json", "*.json")])
    
    if(importpath):
        # Leave Edit/Memory mode before continuing
        exitModes()
        
        # Read file contents into JSON object
        file = open(importpath, "rb")
        user_settings.importjs = json.load(file)
        
        # If a bundle file is being imported, select just the Adventures object
        if type(user_settings.importjs) is dict and "stories" in user_settings.importjs:
            user_settings.importjs = user_settings.importjs["stories"]
        
        # Clear Popup Contents
        emit('from_server', {'cmd': 'clearpopup', 'data': ''}, broadcast=True, room="UI_1")
        
        # Initialize vars
        num = 0
        user_settings.importnum = -1
        
        # Get list of stories
        for story in user_settings.importjs:
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
    if(user_settings.importnum >= 0):
        # Cache reference to selected game
        ref = user_settings.importjs[user_settings.importnum]
        
        # Copy game contents to vars
        story_settings.gamestarted = True
        
        # Support for different versions of export script
        if("actions" in ref):
            if(len(ref["actions"]) > 0):
                story_settings.prompt = ref["actions"][0]["text"]
            else:
                story_settings.prompt = ""
        elif("actionWindow" in ref):
            if(len(ref["actionWindow"]) > 0):
                story_settings.prompt = ref["actionWindow"][0]["text"]
            else:
                story_settings.prompt = ""
        else:
            story_settings.prompt = ""
        story_settings.memory      = ref["memory"]
        story_settings.authornote  = ref["authorsNote"] if type(ref["authorsNote"]) is str else ""
        story_settings.authornotetemplate = "[Author's note: <|>]"
        story_settings.actions     = koboldai_settings.KoboldStoryRegister()
        story_settings.actions_metadata = {}
        story_settings.worldinfo   = []
        story_settings.worldinfo_i = []
        story_settings.worldinfo_u = {}
        story_settings.wifolders_d = {}
        story_settings.wifolders_l = []
        story_settings.wifolders_u = {uid: [] for uid in story_settings.wifolders_d}
        story_settings.lastact     = ""
        story_settings.submission  = ""
        story_settings.lastctx     = ""
        
        # Get all actions except for prompt
        if("actions" in ref):
            if(len(ref["actions"]) > 1):
                for act in ref["actions"][1:]:
                    story_settings.actions.append(act["text"])
        elif("actionWindow" in ref):
            if(len(ref["actionWindow"]) > 1):
                for act in ref["actionWindow"][1:]:
                    story_settings.actions.append(act["text"])
        
        # Get just the important parts of world info
        if(ref["worldInfo"] != None):
            if(len(ref["worldInfo"]) > 1):
                num = 0
                for wi in ref["worldInfo"]:
                    story_settings.worldinfo.append({
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
                        if(uid not in story_settings.worldinfo_u):
                            break
                    story_settings.worldinfo_u[uid] = story_settings.worldinfo[-1]
                    story_settings.worldinfo[-1]["uid"] = uid
                    if(story_settings.worldinfo[-1]["folder"]) is not None:
                        story_settings.wifolders_u[story_settings.worldinfo[-1]["folder"]].append(story_settings.worldinfo[-1])
                    num += 1

        for uid in story_settings.wifolders_l + [None]:
            story_settings.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
            while(True):
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if(uid not in story_settings.worldinfo_u):
                    break
            story_settings.worldinfo_u[uid] = story_settings.worldinfo[-1]
            story_settings.worldinfo[-1]["uid"] = uid
            if(story_settings.worldinfo[-1]["folder"] is not None):
                story_settings.wifolders_u[story_settings.worldinfo[-1]["folder"]].append(story_settings.worldinfo[-1])
        stablesortwi()
        story_settings.worldinfo_i = [wi for wi in story_settings.worldinfo if wi["init"]]
        
        # Clear import data
        user_settings.importjs = {}
        
        # Reset current save
        system_settings.savedir = getcwd()+"\\stories"
        
        # Refresh game screen
        user_settings.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': user_settings.laststory}, broadcast=True, room="UI_1")
        setgamesaved(False)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': story_settings.memory}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': story_settings.authornote}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanotetemplate', 'data': story_settings.authornotetemplate}, broadcast=True, room="UI_1")
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True, room="UI_1")

#==================================================================#
# Import an aidg.club prompt and start a new game with it.
#==================================================================#
def importAidgRequest(id):    
    exitModes()
    
    urlformat = "https://prompts.aidg.club/api/"
    req = requests.get(urlformat+id)

    if(req.status_code == 200):
        js = req.json()
        
        # Import game state
        story_settings.gamestarted = True
        story_settings.prompt      = js["promptContent"]
        story_settings.memory      = js["memory"]
        story_settings.authornote  = js["authorsNote"]
        story_settings.authornotetemplate = "[Author's note: <|>]"
        story_settings.actions     = koboldai_settings.KoboldStoryRegister()
        story_settings.actions_metadata = {}
        story_settings.worldinfo   = []
        story_settings.worldinfo_i = []
        story_settings.worldinfo_u = {}
        story_settings.wifolders_d = {}
        story_settings.wifolders_l = []
        story_settings.wifolders_u = {uid: [] for uid in story_settings.wifolders_d}
        story_settings.lastact     = ""
        story_settings.submission  = ""
        story_settings.lastctx     = ""
        
        num = 0
        for wi in js["worldInfos"]:
            story_settings.worldinfo.append({
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
                if(uid not in story_settings.worldinfo_u):
                    break
            story_settings.worldinfo_u[uid] = story_settings.worldinfo[-1]
            story_settings.worldinfo[-1]["uid"] = uid
            if(story_settings.worldinfo[-1]["folder"]) is not None:
                story_settings.wifolders_u[story_settings.worldinfo[-1]["folder"]].append(story_settings.worldinfo[-1])
            num += 1

        for uid in story_settings.wifolders_l + [None]:
            story_settings.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
            while(True):
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if(uid not in story_settings.worldinfo_u):
                    break
            story_settings.worldinfo_u[uid] = story_settings.worldinfo[-1]
            story_settings.worldinfo[-1]["uid"] = uid
            if(story_settings.worldinfo[-1]["folder"] is not None):
                story_settings.wifolders_u[story_settings.worldinfo[-1]["folder"]].append(story_settings.worldinfo[-1])
        stablesortwi()
        story_settings.worldinfo_i = [wi for wi in story_settings.worldinfo if wi["init"]]

        # Reset current save
        system_settings.savedir = getcwd()+"\\stories"
        
        # Refresh game screen
        user_settings.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': user_settings.laststory}, broadcast=True, room="UI_1")
        setgamesaved(False)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': story_settings.memory}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanote', 'data': story_settings.authornote}, broadcast=True, room="UI_1")
        emit('from_server', {'cmd': 'setanotetemplate', 'data': story_settings.authornotetemplate}, broadcast=True, room="UI_1")
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True, room="UI_1")

#==================================================================#
#  Import World Info JSON file
#==================================================================#
def wiimportrequest():
    importpath = fileops.getloadpath(system_settings.savedir, "Select World Info File", [("Json", "*.json")])
    if(importpath):
        file = open(importpath, "rb")
        js = json.load(file)
        if(len(js) > 0):
            # If the most recent WI entry is blank, remove it.
            if(not story_settings.worldinfo[-1]["init"]):
                del story_settings.worldinfo[-1]
            # Now grab the new stuff
            num = len(story_settings.worldinfo)
            for wi in js:
                story_settings.worldinfo.append({
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
                    if(uid not in story_settings.worldinfo_u):
                        break
                story_settings.worldinfo_u[uid] = story_settings.worldinfo[-1]
                story_settings.worldinfo[-1]["uid"] = uid
                if(story_settings.worldinfo[-1]["folder"]) is not None:
                    story_settings.wifolders_u[story_settings.worldinfo[-1]["folder"]].append(story_settings.worldinfo[-1])
                num += 1
            for uid in [None]:
                story_settings.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
                while(True):
                    uid = int.from_bytes(os.urandom(4), "little", signed=True)
                    if(uid not in story_settings.worldinfo_u):
                        break
                story_settings.worldinfo_u[uid] = story_settings.worldinfo[-1]
                story_settings.worldinfo[-1]["uid"] = uid
                if(story_settings.worldinfo[-1]["folder"] is not None):
                    story_settings.wifolders_u[story_settings.worldinfo[-1]["folder"]].append(story_settings.worldinfo[-1])
        
        if not system_settings.quiet:
            print("{0}".format(story_settings.worldinfo[0]))
                
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
    story_settings.gamestarted = False
    story_settings.prompt      = ""
    story_settings.memory      = ""
    story_settings.actions     = koboldai_settings.KoboldStoryRegister()
    story_settings.actions_metadata = {}
    
    story_settings.authornote  = ""
    story_settings.authornotetemplate = story_settings.setauthornotetemplate
    story_settings.worldinfo   = []
    story_settings.worldinfo_i = []
    story_settings.worldinfo_u = {}
    story_settings.wifolders_d = {}
    story_settings.wifolders_l = []
    story_settings.lastact     = ""
    story_settings.submission  = ""
    story_settings.lastctx     = ""
    
    # Reset current save
    system_settings.savedir = getcwd()+"\\stories"
    
    # Refresh game screen
    user_settings.laststory = None
    emit('from_server', {'cmd': 'setstoryname', 'data': user_settings.laststory}, broadcast=True, room="UI_1")
    setgamesaved(True)
    sendwi()
    emit('from_server', {'cmd': 'setmemory', 'data': story_settings.memory}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanote', 'data': story_settings.authornote}, broadcast=True, room="UI_1")
    emit('from_server', {'cmd': 'setanotetemplate', 'data': story_settings.authornotetemplate}, broadcast=True, room="UI_1")
    setStartState()

def randomGameRequest(topic, memory=""): 
    if(system_settings.noai):
        newGameRequest()
        story_settings.memory = memory
        emit('from_server', {'cmd': 'setmemory', 'data': story_settings.memory}, broadcast=True, room="UI_1")
        return
    story_settings.recentrng = topic
    story_settings.recentrngm = memory
    newGameRequest()
    setgamesaved(False)
    _memory = memory
    if(len(memory) > 0):
        _memory = memory.rstrip() + "\n\n"
    story_settings.memory      = _memory + "You generate the following " + topic + " story concept :"
    system_settings.lua_koboldbridge.feedback = None
    actionsubmit("", force_submit=True, force_prompt_gen=True)
    story_settings.memory      = memory
    emit('from_server', {'cmd': 'setmemory', 'data': story_settings.memory}, broadcast=True, room="UI_1")

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
        if(system_settings.allowsp and "softprompt" in js and type(js["softprompt"]) is str and all(q not in js["softprompt"] for q in ("..", ":")) and (len(js["softprompt"]) == 0 or all(js["softprompt"][0] not in q for q in ("/", "\\")))):
            spRequest(js["softprompt"])
        else:
            system_settings.spfilename = ""
        file.close()

    # Precompile TPU backend if required
    if(system_settings.use_colab_tpu or model_settings.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        soft_tokens = tpumtjgetsofttokens()
        if(story_settings.dynamicscan or (not user_settings.nogenmod and system_settings.has_genmod)):
            threading.Thread(
                target=tpu_mtj_backend.infer_dynamic,
                args=(np.tile(np.uint32((23403, 727, 20185)), (model_settings.numseqs, 1)),),
                kwargs={
                    "soft_embeddings": system_settings.sp,
                    "soft_tokens": soft_tokens,
                    "gen_len": 1,
                    "use_callback": False,
                    "numseqs": model_settings.numseqs,
                    "excluded_world_info": list(set() for _ in range(model_settings.numseqs)),
                },
            ).start()
        else:
            threading.Thread(
                target=tpu_mtj_backend.infer_static,
                args=(np.uint32((23403, 727, 20185)),),
                kwargs={
                    "soft_embeddings": system_settings.sp,
                    "soft_tokens": soft_tokens,
                    "gen_len": 1,
                    "numseqs": model_settings.numseqs,
                },
            ).start()

def send_debug():
    if user_settings.debug:
        debug_info = ""
        try:
            debug_info = "{}Newline Mode: {}\n".format(debug_info, model_settings.newlinemode)
        except:
            pass
        try:
            debug_info = "{}Action Length: {}\n".format(debug_info, story_settings.actions.get_last_key())
        except:
            pass
        try:
            debug_info = "{}Actions Metadata Length: {}\n".format(debug_info, max(story_settings.actions_metadata) if len(story_settings.actions_metadata) > 0 else 0)
        except:
            pass
        try:
            debug_info = "{}Actions: {}\n".format(debug_info, [k for k in story_settings.actions])
        except:
            pass
        try:
            debug_info = "{}Actions Metadata: {}\n".format(debug_info, [k for k in story_settings.actions_metadata])
        except:
            pass
        try:
            debug_info = "{}Last Action: {}\n".format(debug_info, story_settings.actions[story_settings.actions.get_last_key()])
        except:
            pass
        try:
            debug_info = "{}Last Metadata: {}\n".format(debug_info, story_settings.actions_metadata[max(story_settings.actions_metadata)])
        except:
            pass

        emit('from_server', {'cmd': 'debug_info', 'data': debug_info}, broadcast=True, room="UI_1")


#==================================================================#
# UI V2 CODE
#==================================================================#
@app.route('/new_ui')
def new_ui_index():
    return render_template('index_new.html', settings=gensettings.gensettingstf if model_settings.model != "InferKit" else gensettings.gensettingsik )

def ui2_connect():
    pass

#==================================================================#
# Event triggered when browser SocketIO detects a variable change
#==================================================================#
@socketio.on('var_change')
def UI_2_var_change(data):
    print(data)
    classname = data['ID'].split("_")[0]
    name = data['ID'][len(classname)+1:]
    classname += "_settings"
    
    #Need to fix the data type of value to match the module
    if type(getattr(globals()[classname], name)) == int:
        value = int(data['value'])
    elif type(getattr(globals()[classname], name)) == float:
        value = float(data['value'])
    elif type(getattr(globals()[classname], name)) == bool:
        value = bool(data['value'])
    elif type(getattr(globals()[classname], name)) == str:
        value = str(data['value'])
    else:
        print("Unknown Type {} = {}".format(name, type(getattr(globals()[classname], name))))
    
    print("{} {} = {}".format(classname, name, value))
    
    setattr(globals()[classname], name, value)
    
    #Now let's save except for story changes
    if classname != "story_settings":
        with open("settings/{}.v2_settings".format(classname), "w") as settings_file:
            settings_file.write(globals()[classname].to_json())
    
#==================================================================#
# Saving Story
#==================================================================#
@socketio.on('save_story')
def UI_2_save_story(data):
    json_data = story_settings.to_json()
    save_name = story_settings.story_name if story_settings.story_name is not None else "untitled"
    with open("stories/{}_v2.json".format(save_name), "w") as settings_file:
        settings_file.write(story_settings.to_json())
    story_settings.gamesaved = True
    
    
#==================================================================#
# Event triggered when Selected Text is edited, Option is Selected, etc
#==================================================================#
@socketio.on('Set Selected Text')
def UI_2_Set_Selected_Text(data):
    print("Updating Selected Text: {}".format(data))
    story_settings.actions.use_option(int(data['option']), action_step=int(data['chunk']))

#==================================================================#
# Event triggered when user clicks the submit button
#==================================================================#
@socketio.on('submit')
def UI_2_submit(data):
    system_settings.lua_koboldbridge.feedback = None
    story_settings.recentrng = story_settings.recentrngm = None
    actionsubmit(data['data'], actionmode=story_settings.actionmode)
    
#==================================================================#
# Event triggered when user clicks the pin button
#==================================================================#
@socketio.on('Pinning')
def UI_2_Pinning(data):
    story_settings.actions.toggle_pin(int(data['chunk']), int(data['option']))
    
#==================================================================#
# Event triggered when user clicks the back button
#==================================================================#
@socketio.on('back')
def UI_2_back(data):
    print("back")
    ignore = story_settings.actions.pop()
    
#==================================================================#
# Event triggered when user clicks the redo button
#==================================================================#
@socketio.on('redo')
def UI_2_redo(data):
    if len(story_settings.actions.get_current_options()) == 1:
        story_settings.actions.use_option(0)

#==================================================================#
# Event triggered when user clicks the redo button
#==================================================================#
@socketio.on('retry')
def UI_2_retry(data):
    story_settings.actions.clear_unused_options()
    system_settings.lua_koboldbridge.feedback = None
    story_settings.recentrng = story_settings.recentrngm = None
    actionsubmit("", actionmode=story_settings.actionmode)
    
#==================================================================#
# Event triggered to rely a message
#==================================================================#
@socketio.on('relay')
def UI_2_relay(data):
    socketio.emit(data[0], data[1], **data[2])


#==================================================================#
# Test
#==================================================================#
@app.route("/actions")
def show_actions():
    return story_settings.actions.actions
    
@app.route("/story")
def show_story():
    return story_settings.to_json()
    
    

#==================================================================#
#  Final startup commands to launch Flask app
#==================================================================#
print("", end="", flush=True)
if __name__ == "__main__":
    print("{0}\nStarting webserver...{1}".format(colors.GREEN, colors.END), flush=True)

    general_startup()
    patch_transformers()
    #show_select_model_list()
    if model_settings.model == "" or model_settings.model is None:
        model_settings.model = "ReadOnly"
    load_model(initial_load=True)

    # Start Flask/SocketIO (Blocking, so this must be last method!)
    port = args.port if "port" in args and args.port is not None else 5000
    
    if(system_settings.host):
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
                print(format(colors.GREEN) + "KoboldAI has finished loading and is available at the following link : " + cloudflare + format(colors.END))
        else:
            print("{0}Webserver has started, you can now connect to this machine at port {1}{2}"
                  .format(colors.GREEN, port, colors.END))
        system_settings.serverstarted = True
        socketio.run(app, host='0.0.0.0', port=port)
    else:
        if args.unblock:
            import webbrowser
            webbrowser.open_new('http://localhost:{0}'.format(port))
            print("{0}Server started!\nYou may now connect with a browser at http://127.0.0.1:{1}/{2}"
                  .format(colors.GREEN, port, colors.END))
            system_settings.serverstarted = True
            socketio.run(app, port=port, host='0.0.0.0')
        else:
            try:
                from flaskwebgui import FlaskUI
                system_settings.serverstarted = True
                system_settings.flaskwebgui = True
                FlaskUI(app, socketio=socketio, start_server="flask-socketio", maximized=True, close_server_on_exit=True).run()
            except:
                import webbrowser
                webbrowser.open_new('http://localhost:{0}'.format(port))
                print("{0}Server started!\nYou may now connect with a browser at http://127.0.0.1:{1}/{2}"
                        .format(colors.GREEN, port, colors.END))
                system_settings.serverstarted = True
                socketio.run(app, port=port)

else:
    general_startup()
    patch_transformers()
    #show_select_model_list()
    if model_settings.model == "" or model_settings.model is None:
        model_settings.model = "ReadOnly"
    load_model(initial_load=True)
    print("{0}\nServer started in WSGI mode!{1}".format(colors.GREEN, colors.END), flush=True)
    
