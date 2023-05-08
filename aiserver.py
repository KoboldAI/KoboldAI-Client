#!/usr/bin/python3
#==================================================================#
# KoboldAI
# Version: 1.19.2
# By: The KoboldAI Community
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
from logger import logger, set_logger_verbosity, quiesce_logger

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
import inspect
import warnings
from collections.abc import Iterable
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
import structures
import torch
from transformers import StoppingCriteria, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, GPTNeoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, modeling_utils
from transformers import __version__ as transformers_version
import transformers
try:
    from transformers.models.opt.modeling_opt import OPTDecoder
except:
    pass
import transformers.generation_utils

global tpu_mtj_backend


if lupa.LUA_VERSION[:2] != (5, 4):
    logger.error(f"Please install lupa==1.10. You have lupa {lupa.__version__}.")

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
# Each item takes the 4 elements, 1: Text to display, 2: Model Name (var.model) or menu name (Key name for another menu),
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
        ["Untuned Pythia", "pythialist", "", True],
        ["Untuned Fairseq Dense", "fsdlist", "", True],
        ["Untuned Bloom", "bloomlist", "", True],
        ["Untuned XGLM", "xglmlist", "", True],
        ["Untuned GPT2", "gpt2list", "", True],
        ["Online Services", "apilist", "", True],
        ["Read Only (No AI)", "ReadOnly", "", False]
        ],
    'adventurelist': [
        ["Skein 20B", "KoboldAI/GPT-NeoX-20B-Skein", "64GB", False],
        ["Nerys OPT 13B V2 (Hybrid)", "KoboldAI/OPT-13B-Nerys-v2", "32GB", False],
        ["Nerys FSD 13B V2 (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys-v2", "32GB", False],
        ["Nerys FSD 13B (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys", "32GB", False],
        ["Skein 6B", "KoboldAI/GPT-J-6B-Skein", "16GB", False],
        ["OPT Nerys 6B V2 (Hybrid)", "KoboldAI/OPT-6B-nerys-v2", "16GB", False],
        ["Adventure 6B", "KoboldAI/GPT-J-6B-Adventure", "16GB", False],
        ["Nerys FSD 2.7B (Hybrid)", "KoboldAI/fairseq-dense-2.7B-Nerys", "8GB", False],
        ["Adventure 2.7B", "KoboldAI/GPT-Neo-2.7B-AID", "8GB", False],
        ["Adventure 1.3B", "KoboldAI/GPT-Neo-1.3B-Adventure", "6GB", False],
        ["Adventure 125M (Mia)", "Merry/AID-Neo-125M", "2GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'novellist': [
        ["Nerys OPT 13B V2 (Hybrid)", "KoboldAI/OPT-13B-Nerys-v2", "32GB", False],
        ["Nerys FSD 13B V2 (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys-v2", "32GB", False],
        ["Janeway FSD 13B", "KoboldAI/fairseq-dense-13B-Janeway", "32GB", False],
        ["Nerys FSD 13B (Hybrid)", "KoboldAI/fairseq-dense-13B-Nerys", "32GB", False],
        ["OPT Nerys 6B V2 (Hybrid)", "KoboldAI/OPT-6B-nerys-v2", "16GB", False],
        ["Janeway FSD 6.7B", "KoboldAI/fairseq-dense-6.7B-Janeway", "16GB", False],
        ["Janeway Neo 6B", "KoboldAI/GPT-J-6B-Janeway", "16GB", False],
        ["Qilin Lit 6B (SFW)", "rexwang8/qilin-lit-6b", "16GB", False],       
        ["Janeway Neo 2.7B", "KoboldAI/GPT-Neo-2.7B-Janeway", "8GB", False],
        ["Janeway FSD 2.7B", "KoboldAI/fairseq-dense-2.7B-Janeway", "8GB", False],
        ["Nerys FSD 2.7B (Hybrid)", "KoboldAI/fairseq-dense-2.7B-Nerys", "8GB", False],
        ["Horni-LN 2.7B", "KoboldAI/GPT-Neo-2.7B-Horni-LN", "8GB", False],
        ["Picard 2.7B (Older Janeway)", "KoboldAI/GPT-Neo-2.7B-Picard", "8GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'nsfwlist': [
        ["Erebus 20B (NSFW)", "KoboldAI/GPT-NeoX-20B-Erebus", "64GB", False],
        ["Nerybus 13B (NSFW)", "KoboldAI/OPT-13B-Nerybus-Mix", "32GB", False],
        ["Erebus 13B (NSFW)", "KoboldAI/OPT-13B-Erebus", "32GB", False],
        ["Shinen FSD 13B (NSFW)", "KoboldAI/fairseq-dense-13B-Shinen", "32GB", False],
        ["Nerybus 6.7B (NSFW)", "KoboldAI/OPT-6.7B-Nerybus-Mix", "16GB", False],
        ["Erebus 6.7B (NSFW)", "KoboldAI/OPT-6.7B-Erebus", "16GB", False],
        ["Shinen FSD 6.7B (NSFW)", "KoboldAI/fairseq-dense-6.7B-Shinen", "16GB", False],
        ["Lit V2 6B (NSFW)", "hakurei/litv2-6B-rev3", "16GB", False],
        ["Lit 6B (NSFW)", "hakurei/lit-6B", "16GB", False],
        ["Shinen 6B (NSFW)", "KoboldAI/GPT-J-6B-Shinen", "16GB", False],
        ["Nerybus 2.7B (NSFW)", "KoboldAI/OPT-2.7B-Nerybus-Mix", "8GB", False],
        ["Erebus 2.7B (NSFW)", "KoboldAI/OPT-2.7B-Erebus", "8GB", False],
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
        ["GPT-NeoX 20B", "EleutherAI/gpt-neox-20b", "64GB", False],
        ["Pythia 13B (NeoX, Same dataset)", "EleutherAI/pythia-13b", "32GB", False],
        ["GPT-J 6B", "EleutherAI/gpt-j-6B", "16GB", False],
        ["GPT-Neo 2.7B", "EleutherAI/gpt-neo-2.7B", "8GB", False],
        ["GPT-Neo 1.3B", "EleutherAI/gpt-neo-1.3B", "6GB", False],
        ["Pythia 800M (NeoX, Same dataset)", "EleutherAI/pythia-800m", "4GB", False],
        ["Pythia 350M (NeoX, Same dataset)", "EleutherAI/pythia-350m", "2GB", False],
        ["GPT-Neo 125M", "EleutherAI/gpt-neo-125M", "2GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'pythialist': [
        ["Pythia 13B Deduped", "EleutherAI/pythia-13b-deduped", "32GB", False],
        ["Pythia 13B", "EleutherAI/pythia-13b", "32GB", False],
        ["Pythia 6.7B Deduped", "EleutherAI/pythia-6.7b-deduped", "16GB", False],
        ["Pythia 6.7B", "EleutherAI/pythia-6.7b", "16GB", False],
        ["Pythia 1.3B Deduped", "EleutherAI/pythia-1.3b-deduped", "6GB", False],
        ["Pythia 1.3B", "EleutherAI/pythia-1.3b", "6GB", False],
        ["Pythia 800M", "EleutherAI/pythia-800m", "4GB", False],
        ["Pythia 350M Deduped", "EleutherAI/pythia-350m-deduped", "2GB", False],
        ["Pythia 350M", "EleutherAI/pythia-350m", "2GB", False],        
        ["Pythia 125M Deduped", "EleutherAI/pythia-125m-deduped", "2GB", False],
        ["Pythia 125M", "EleutherAI/pythia-125m", "2GB", False],
        ["Pythia 19M Deduped", "EleutherAI/pythia-19m-deduped", "1GB", False],
        ["Pythia 19M", "EleutherAI/pythia-19m", "1GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'gpt2list': [
        ["GPT-2 XL", "gpt2-xl", "6GB", False],
        ["GPT-2 Large", "gpt2-large", "4GB", False],
        ["GPT-2 Med", "gpt2-medium", "2GB", False],
        ["GPT-2", "gpt2", "2GB", False],
        ["Return to Main Menu", "mainmenu", "", True],
        ],
    'bloomlist': [
        ["Bloom 176B", "bigscience/bloom", "", False],
        ["Bloom 7.1B", "bigscience/bloom-7b1", "", False],   
        ["Bloom 3B", "bigscience/bloom-3b", "", False], 
        ["Bloom 1.7B", "bigscience/bloom-1b7", "", False], 
        ["Bloom 560M", "bigscience/bloom-560m", "", False], 
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
        ["KoboldAI Horde", "CLUSTER", "", False],
        ["Return to Main Menu", "mainmenu", "", True],
    ]
    }

class TokenStreamQueue:
    def __init__(self):
        self.probability_buffer = None
        self.queue = []

    def add_text(self, text):
        self.queue.append({
            "decoded": text,
            "probabilities": self.probability_buffer
        })
        self.probability_buffer = None

# Variables
class vars:
    lastact     = ""     # The last action received from the user
    submission  = ""     # Same as above, but after applying input formatting
    lastctx     = ""     # The last context submitted to the generator
    model       = "ReadOnly"     # Model ID string chosen at startup
    online_model = ""     # Used when Model ID is an online service, and there is a secondary option for the actual model name
    model_selected = ""  #selected model in UI
    model_type  = ""     # Model Type (Automatically taken from the model config)
    noai        = False  # Runs the script without starting up the transformers pipeline
    aibusy      = False  # Stops submissions while the AI is working
    max_length  = 1024    # Maximum number of tokens to submit per action
    ikmax       = 3000   # Maximum number of characters to submit to InferKit
    genamt      = 80     # Amount of text for each action to generate
    ikgen       = 200    # Number of characters for InferKit to generate
    rep_pen     = 1.1    # Default generator repetition_penalty
    rep_pen_slope = 0.7  # Default generator repetition penalty slope
    rep_pen_range = 1024 # Default generator repetition penalty range
    temp        = 0.5    # Default generator temperature
    top_p       = 0.9    # Default generator top_p
    top_k       = 0      # Default generator top_k
    top_a       = 0.0    # Default generator top-a
    tfs         = 1.0    # Default generator tfs (tail-free sampling)
    typical     = 1.0    # Default generator typical sampling threshold
    numseqs     = 1     # Number of sequences to ask the generator to create
    full_determinism = False  # Whether or not full determinism is enabled
    seed_specified = False  # Whether or not the current RNG seed was specified by the user (in their settings file)
    seed        = None   # The current RNG seed (as an int), or None if unknown
    gamestarted = False  # Whether the game has started (disables UI elements)
    gamesaved   = True   # Whether or not current game is saved
    serverstarted = False  # Whether or not the Flask server has started
    prompt      = ""     # Prompt
    memory      = ""     # Text submitted to memory field
    authornote  = ""     # Text submitted to Author's Note field
    authornotetemplate = "[Author's note: <|>]"  # Author's note template
    setauthornotetemplate = authornotetemplate  # Saved author's note template in settings
    andepth     = 3      # How far back in history to append author's note
    actions     = structures.KoboldStoryRegister()  # Actions submitted by user and AI
    actions_metadata = {} # List of dictonaries, one dictonary for every action that contains information about the action like alternative options.
                          # Contains at least the same number of items as actions. Back action will remove an item from actions, but not actions_metadata
                          # Dictonary keys are:
                          # Selected Text: (text the user had selected. None when this is a newly generated action)
                          # Alternative Generated Text: {Text, Pinned, Previous Selection, Edited}
                          # 
    worldinfo   = []     # List of World Info key/value objects
    worldinfo_i = []     # List of World Info key/value objects sans uninitialized entries
    worldinfo_u = {}     # Dictionary of World Info UID - key/value pairs
    wifolders_d = {}     # Dictionary of World Info folder UID-info pairs
    wifolders_l = []     # List of World Info folder UIDs
    wifolders_u = {}     # Dictionary of pairs of folder UID - list of WI UID
    modelconfig = {}     # Raw contents of the model's config.json, or empty dictionary if none found
    lua_state   = None   # Lua state of the Lua scripting system
    lua_koboldbridge = None  # `koboldbridge` from bridge.lua
    lua_kobold  = None   # `kobold` from` bridge.lua
    lua_koboldcore = None  # `koboldcore` from bridge.lua
    lua_logname = ...    # Name of previous userscript that logged to terminal
    lua_running = False  # Whether or not Lua is running (i.e. wasn't stopped due to an error)
    lua_edited  = set()  # Set of chunk numbers that were edited from a Lua generation modifier
    lua_deleted = set()  # Set of chunk numbers that were deleted from a Lua generation modifier
    generated_tkns = 0   # If using a backend that supports Lua generation modifiers, how many tokens have already been generated, otherwise 0
    abort       = False  # Whether or not generation was aborted by clicking on the submit button during generation
    compiling   = False  # If using a TPU Colab, this will be set to True when the TPU backend starts compiling and then set to False again
    checking    = False  # Whether or not we are actively checking to see if TPU backend is compiling or not
    sp_changed  = False  # This gets set to True whenever a userscript changes the soft prompt so that check_for_sp_change() can alert the browser that the soft prompt has changed
    spfilename  = ""     # Filename of soft prompt to load, or an empty string if not using a soft prompt
    userscripts = []     # List of userscripts to load
    last_userscripts = []  # List of previous userscript filenames from the previous time userscripts were send via usstatitems
    corescript  = "default.lua"  # Filename of corescript to load
    # badwords    = []     # Array of str/chr values that should be removed from output
    badwordsids = []
    badwordsids_default = [[13460], [6880], [50256], [42496], [4613], [17414], [22039], [16410], [27], [29], [38430], [37922], [15913], [24618], [28725], [58], [47175], [36937], [26700], [12878], [16471], [37981], [5218], [29795], [13412], [45160], [3693], [49778], [4211], [20598], [36475], [33409], [44167], [32406], [29847], [29342], [42669], [685], [25787], [7359], [3784], [5320], [33994], [33490], [34516], [43734], [17635], [24293], [9959], [23785], [21737], [28401], [18161], [26358], [32509], [1279], [38155], [18189], [26894], [6927], [14610], [23834], [11037], [14631], [26933], [46904], [22330], [25915], [47934], [38214], [1875], [14692], [41832], [13163], [25970], [29565], [44926], [19841], [37250], [49029], [9609], [44438], [16791], [17816], [30109], [41888], [47527], [42924], [23984], [49074], [33717], [31161], [49082], [30138], [31175], [12240], [14804], [7131], [26076], [33250], [3556], [38381], [36338], [32756], [46581], [17912], [49146]] # Tokenized array of badwords used to prevent AI artifacting
    badwordsids_neox = [[0], [1], [44162], [9502], [12520], [31841], [36320], [49824], [34417], [6038], [34494], [24815], [26635], [24345], [3455], [28905], [44270], [17278], [32666], [46880], [7086], [43189], [37322], [17778], [20879], [49821], [3138], [14490], [4681], [21391], [26786], [43134], [9336], [683], [48074], [41256], [19181], [29650], [28532], [36487], [45114], [46275], [16445], [15104], [11337], [1168], [5647], [29], [27482], [44965], [43782], [31011], [42944], [47389], [6334], [17548], [38329], [32044], [35487], [2239], [34761], [7444], [1084], [12399], [18990], [17636], [39083], [1184], [35830], [28365], [16731], [43467], [47744], [1138], [16079], [40116], [45564], [18297], [42368], [5456], [18022], [42696], [34476], [23505], [23741], [39334], [37944], [45382], [38709], [33440], [26077], [43600], [34418], [36033], [6660], [48167], [48471], [15775], [19884], [41533], [1008], [31053], [36692], [46576], [20095], [20629], [31759], [46410], [41000], [13488], [30952], [39258], [16160], [27655], [22367], [42767], [43736], [49694], [13811], [12004], [46768], [6257], [37471], [5264], [44153], [33805], [20977], [21083], [25416], [14277], [31096], [42041], [18331], [33376], [22372], [46294], [28379], [38475], [1656], [5204], [27075], [50001], [16616], [11396], [7748], [48744], [35402], [28120], [41512], [4207], [43144], [14767], [15640], [16595], [41305], [44479], [38958], [18474], [22734], [30522], [46267], [60], [13976], [31830], [48701], [39822], [9014], [21966], [31422], [28052], [34607], [2479], [3851], [32214], [44082], [45507], [3001], [34368], [34758], [13380], [38363], [4299], [46802], [30996], [12630], [49236], [7082], [8795], [5218], [44740], [9686], [9983], [45301], [27114], [40125], [1570], [26997], [544], [5290], [49193], [23781], [14193], [40000], [2947], [43781], [9102], [48064], [42274], [18772], [49384], [9884], [45635], [43521], [31258], [32056], [47686], [21760], [13143], [10148], [26119], [44308], [31379], [36399], [23983], [46694], [36134], [8562], [12977], [35117], [28591], [49021], [47093], [28653], [29013], [46468], [8605], [7254], [25896], [5032], [8168], [36893], [38270], [20499], [27501], [34419], [29547], [28571], [36586], [20871], [30537], [26842], [21375], [31148], [27618], [33094], [3291], [31789], [28391], [870], [9793], [41361], [47916], [27468], [43856], [8850], [35237], [15707], [47552], [2730], [41449], [45488], [3073], [49806], [21938], [24430], [22747], [20924], [46145], [20481], [20197], [8239], [28231], [17987], [42804], [47269], [29972], [49884], [21382], [46295], [36676], [34616], [3921], [26991], [27720], [46265], [654], [9855], [40354], [5291], [34904], [44342], [2470], [14598], [880], [19282], [2498], [24237], [21431], [16369], [8994], [44524], [45662], [13663], [37077], [1447], [37786], [30863], [42854], [1019], [20322], [4398], [12159], [44072], [48664], [31547], [18736], [9259], [31], [16354], [21810], [4357], [37982], [5064], [2033], [32871], [47446], [62], [22158], [37387], [8743], [47007], [17981], [11049], [4622], [37916], [36786], [35138], [29925], [14157], [18095], [27829], [1181], [22226], [5709], [4725], [30189], [37014], [1254], [11380], [42989], [696], [24576], [39487], [30119], [1092], [8088], [2194], [9899], [14412], [21828], [3725], [13544], [5180], [44679], [34398], [3891], [28739], [14219], [37594], [49550], [11326], [6904], [17266], [5749], [10174], [23405], [9955], [38271], [41018], [13011], [48392], [36784], [24254], [21687], [23734], [5413], [41447], [45472], [10122], [17555], [15830], [47384], [12084], [31350], [47940], [11661], [27988], [45443], [905], [49651], [16614], [34993], [6781], [30803], [35869], [8001], [41604], [28118], [46462], [46762], [16262], [17281], [5774], [10943], [5013], [18257], [6750], [4713], [3951], [11899], [38791], [16943], [37596], [9318], [18413], [40473], [13208], [16375]]
    badwordsids_opt = [[44717], [46613], [48513], [49923], [50185], [48755], [8488], [43303], [49659], [48601], [49817], [45405], [48742], [49925], [47720], [11227], [48937], [48784], [50017], [42248], [49310], [48082], [49895], [50025], [49092], [49007], [8061], [44226], [0], [742], [28578], [15698], [49784], [46679], [39365], [49281], [49609], [48081], [48906], [46161], [48554], [49670], [48677], [49721], [49632], [48610], [48462], [47457], [10975], [46077], [28696], [48709], [43839], [49798], [49154], [48203], [49625], [48395], [50155], [47161], [49095], [48833], [49420], [49666], [48443], [22176], [49242], [48651], [49138], [49750], [40389], [48021], [21838], [49070], [45333], [40862], [1], [49915], [33525], [49858], [50254], [44403], [48992], [48872], [46117], [49853], [47567], [50206], [41552], [50068], [48999], [49703], [49940], [49329], [47620], [49868], [49962], [2], [44082], [50236], [31274], [50260], [47052], [42645], [49177], [17523], [48691], [49900], [49069], [49358], [48794], [47529], [46479], [48457], [646], [49910], [48077], [48935], [46386], [48902], [49151], [48759], [49803], [45587], [48392], [47789], [48654], [49836], [49230], [48188], [50264], [46844], [44690], [48505], [50161], [27779], [49995], [41833], [50154], [49097], [48520], [50018], [8174], [50084], [49366], [49526], [50193], [7479], [49982], [3]]
    fp32_model  = False  # Whether or not the most recently loaded HF model was in fp32 format
    deletewi    = None   # Temporary storage for UID to delete
    wirmvwhtsp  = True  # Whether to remove leading whitespace from WI entries
    widepth     = 3      # How many historical actions to scan for WI hits
    mode        = "play" # Whether the interface is in play, memory, or edit mode
    editln      = 0      # Which line was last selected in Edit Mode
    gpu_device  = 0      # Which PyTorch device to use when using pure GPU generation
    url         = "https://api.inferkit.com/v1/models/standard/generate" # InferKit API URL
    oaiurl      = "" # OpenAI API URL
    oaiengines  = "https://api.openai.com/v1/engines"
    colaburl    = ""     # Ngrok url for Google Colab mode
    apikey      = ""     # API key to use for InferKit API calls
    oaiapikey   = ""     # API key to use for OpenAI API calls
    cluster_requested_models = [] # The models which we allow to generate during cluster mode
    savedir     = getcwd()+"\\stories"
    hascuda     = False  # Whether torch has detected CUDA on the system
    usegpu      = False  # Whether to launch pipeline with GPU support
    custmodpth  = ""     # Filesystem location of custom model to run
    formatoptns = {'frmttriminc': True, 'frmtrmblln': False, 'frmtrmspch': False, 'frmtadsnsp': True, 'singleline': False}     # Container for state of formatting options
    importnum   = -1     # Selection on import popup list
    importjs    = {}     # Temporary storage for import data
    loadselect  = ""     # Temporary storage for story filename to load
    spselect    = ""     # Temporary storage for soft prompt filename to load
    spmeta      = None   # Metadata of current soft prompt, or None if not using a soft prompt
    sp          = None   # Current soft prompt tensor (as a NumPy array)
    sp_length   = 0      # Length of current soft prompt in tokens, or 0 if not using a soft prompt
    has_genmod  = False  # Whether or not at least one loaded Lua userscript has a generation modifier
    svowname    = ""     # Filename that was flagged for overwrite confirm
    saveow      = False  # Whether or not overwrite confirm has been displayed
    autosave    = False  # Whether or not to automatically save after each action
    genseqs     = []     # Temporary storage for generated sequences
    recentback  = False  # Whether Back button was recently used without Submitting or Retrying after
    recentrng   = None   # If a new random game was recently generated without Submitting after, this is the topic used (as a string), otherwise this is None
    recentrngm  = None   # If a new random game was recently generated without Submitting after, this is the memory used (as a string), otherwise this is None
    useprompt   = False   # Whether to send the full prompt with every submit action
    breakmodel  = False  # For GPU users, whether to use both system RAM and VRAM to conserve VRAM while offering speedup compared to CPU-only
    bmsupported = False  # Whether the breakmodel option is supported (GPT-Neo/GPT-J/XGLM/OPT only, currently)
    nobreakmodel = False  # Something specifically requested Breakmodel to be disabled (For example a models config)
    smandelete  = False  # Whether stories can be deleted from inside the browser
    smanrename  = False  # Whether stories can be renamed from inside the browser
    allowsp     = False  # Whether we are allowed to use soft prompts (by default enabled if we're using GPT-2, GPT-Neo or GPT-J)
    modeldim    = -1     # Embedding dimension of your model (e.g. it's 4096 for GPT-J-6B and 2560 for GPT-Neo-2.7B)
    laststory   = None   # Filename (without extension) of most recent story JSON file we loaded
    regex_sl    = re.compile(r'\n*(?<=.) *\n(.|\n)*')  # Pattern for limiting the output to a single line
    acregex_ai  = re.compile(r'\n* *>(.|\n)*')  # Pattern for matching adventure actions from the AI so we can remove them
    acregex_ui  = re.compile(r'^ *(&gt;.*)$', re.MULTILINE)    # Pattern for matching actions in the HTML-escaped story so we can apply colouring, etc (make sure to encase part to format in parentheses)
    comregex_ai = re.compile(r'(?:\n<\|(?:.|\n)*?\|>(?=\n|$))|(?:<\|(?:.|\n)*?\|>\n?)')  # Pattern for matching comments to remove them before sending them to the AI
    comregex_ui = re.compile(r'(&lt;\|(?:.|\n)*?\|&gt;)')  # Pattern for matching comments in the editor
    sampler_order = utils.default_sampler_order.copy()
    rng_states  = {}  # Used by the POST /generate endpoint to store sampler RNG states
    chatmode    = False
    chatname    = "You"
    adventure   = False
    actionmode  = 1
    dynamicscan = False
    host        = False
    nopromptgen = False
    rngpersist  = False
    nogenmod    = False
    welcome     = False  # Custom Welcome Text (False is default)
    newlinemode = "ns"
    quiet       = False # If set will suppress any story text from being printed to the console (will only be seen on the client web page)
    debug       = False # If set to true, will send debug information to the client for display
    lazy_load   = True  # Whether or not to use torch_lazy_loader.py for transformers models in order to reduce CPU memory usage
    use_colab_tpu = os.environ.get("COLAB_TPU_ADDR", "") != "" or os.environ.get("TPU_NAME", "") != ""  # Whether or not we're in a Colab TPU instance or Kaggle TPU instance and are going to use the TPU rather than the CPU
    revision    = None
    standalone = False
    api_tokenizer_id = None
    disable_set_aibusy = False
    disable_input_formatting = False
    disable_output_formatting = False
    output_streaming = True
    token_stream_queue = TokenStreamQueue() # Queue for the token streaming
    show_probs = False # Whether or not to show token probabilities
    show_budget = False # Whether or not to show token probabilities
    configname = None

utils.vars = vars

class Send_to_socketio(object):
    def write(self, bar):
        print(bar, end="")
        time.sleep(0.01)
        try:
            gui_msg = bar.replace(f"{colors.PURPLE}INIT{colors.END}       | ","").replace(" ", "&nbsp;")
            emit('from_server', {'cmd': 'model_load_status', 'data': gui_msg}, broadcast=True)
        except:
            pass
                                
# Set logging level to reduce chatter from Flask
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from flask import Flask, render_template, Response, request, copy_current_request_context, send_from_directory, session, jsonify, abort, redirect
from flask_socketio import SocketIO
from flask_socketio import emit as _emit
from flask_session import Session
from werkzeug.exceptions import HTTPException, NotFound, InternalServerError
import secrets
app = Flask(__name__, root_path=os.getcwd())
app.secret_key = secrets.token_hex()
app.config['SESSION_TYPE'] = 'filesystem'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app, async_method="eventlet")

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
utils.emit = emit

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
    version="1.2.1",
    prefixes=["/api/v1", "/api/latest"],
    tags=tags,
)

# Returns the expected config filename for the current setup.
# If the model_name is specified, it returns what the settings file would be for that model
def get_config_filename(model_name = None):
    if model_name:
        return(f"settings/{model_name.replace('/', '_')}.settings")
    elif args.configname:
        return(f"settings/{args.configname.replace('/', '_')}.settings")
    elif vars.configname != '':
        return(f"settings/{vars.configname.replace('/', '_')}.settings")
    else:
        logger.warning(f"Empty configfile name sent back. Defaulting to ReadOnly")
        return(f"settings/ReadOnly.settings")
#==================================================================#
# Function to get model selection at startup
#==================================================================#
def sendModelSelection(menu="mainmenu", folder="./models"):
    #If we send one of the manual load options, send back the list of model directories, otherwise send the menu
    if menu in ('NeoCustom', 'GPT2Custom'):
        (paths, breadcrumbs) = get_folder_path_info(folder)
        if vars.host:
            breadcrumbs = []
        menu_list = [[folder, menu, "", False] for folder in paths]
        menu_list.append(["Return to Main Menu", "mainmenu", "", True])
        if os.path.abspath("{}/models".format(os.getcwd())) == os.path.abspath(folder):
            showdelete=True
        else:
            showdelete=False
        emit('from_server', {'cmd': 'show_model_menu', 'data': menu_list, 'menu': menu, 'breadcrumbs': breadcrumbs, "showdelete": showdelete}, broadcast=True)
    else:
        emit('from_server', {'cmd': 'show_model_menu', 'data': model_menu[menu], 'menu': menu, 'breadcrumbs': [], "showdelete": False}, broadcast=True)

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
    vars.model = ''
    while(vars.model == ''):
        modelsel = input("Model #> ")
        if(modelsel.isnumeric() and int(modelsel) > 0 and int(modelsel) <= len(modellist)):
            vars.model = modellist[int(modelsel)-1][1]
        else:
            print("{0}Please enter a valid selection.{1}".format(colors.RED, colors.END))
    
    # Model Lists
    try:
        getModelSelection(eval(vars.model))
    except Exception as e:
        if(vars.model == "Return"):
            getModelSelection(mainmenu)
                
        # If custom model was selected, get the filesystem location and store it
        if(vars.model == "NeoCustom" or vars.model == "GPT2Custom"):
            print("{0}Please choose the folder where pytorch_model.bin is located:{1}\n".format(colors.CYAN, colors.END))
            modpath = fileops.getdirpath(getcwd() + "/models", "Select Model Folder")
        
            if(modpath):
                # Save directory to vars
                vars.custmodpth = modpath
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
    if(vars.online_model != ''):
       return(f"{vars.model}/{vars.online_model}")
    if(vars.model in ("NeoCustom", "GPT2Custom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        modelname = os.path.basename(os.path.normpath(vars.custmodpth))
        return modelname
    else:
        modelname = vars.model
        return modelname

#==================================================================#
# Get hidden size from model
#==================================================================#
def get_hidden_size_from_model(model):
    return model.get_input_embeddings().embedding_dim

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
            logger.warning("--breakmodel_gpulayers is malformatted. Please use the --help option to see correct usage of --breakmodel_gpulayers. Defaulting to all layers on device 0.")
            breakmodel.gpu_blocks = [n_layers]
            n_layers = 0
    elif(args.breakmodel_layers is not None):
        breakmodel.gpu_blocks = [n_layers - max(0, min(n_layers, args.breakmodel_layers))]
        n_layers -= sum(breakmodel.gpu_blocks)
    elif(args.model is not None):
        logger.info("Breakmodel not specified, assuming GPU 0")
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

    logger.init_ok("Final device configuration:", status="Info")
    device_list(n_layers, primary=breakmodel.primary_device)

    # If all layers are on the same device, use the old GPU generation mode
    while(len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] == 0):
        breakmodel.gpu_blocks.pop()
    if(len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] in (-1, utils.num_layers(config))):
        vars.breakmodel = False
        vars.usegpu = True
        vars.gpu_device = len(breakmodel.gpu_blocks)-1
        return

    if(not breakmodel.gpu_blocks):
        logger.warning("Nothing assigned to a GPU, reverting to CPU only mode")
        import breakmodel
        breakmodel.primary_device = "cpu"
        vars.breakmodel = False
        vars.usegpu = False
        return

def move_model_to_devices(model):
    global generator

    if(not utils.HAS_ACCELERATE and not vars.breakmodel):
        if(vars.usegpu):
            model = model.half().to(vars.gpu_device)
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
                js   = json.load(open(vars.custmodpth + "/config.json", "r"))
            except Exception as e:
                js   = json.load(open(vars.custmodpth.replace('/', '_') + "/config.json", "r"))            
        except Exception as e:
            js   = {}
    if vars.model_type == "xglm" or js.get("compat", "j") == "fairseq_lm":
        vars.newlinemode = "s"  # Default to </s> newline mode if using XGLM
    if vars.model_type == "opt" or vars.model_type == "bloom":
        vars.newlinemode = "ns"  # Handle </s> but don't convert newlines if using Fairseq models that have newlines trained in them
    vars.modelconfig = js
    if("badwordsids" in js):
        vars.badwordsids = js["badwordsids"]
    if("nobreakmodel" in js):
        vars.nobreakmodel = js["nobreakmodel"]
    if("sampler_order" in js):
        sampler_order = js["sampler_order"]
        if(len(sampler_order) < 7):
            sampler_order = [6] + sampler_order
        vars.sampler_order = sampler_order
    if("temp" in js):
        vars.temp       = js["temp"]
    if("top_p" in js):
        vars.top_p      = js["top_p"]
    if("top_k" in js):
        vars.top_k      = js["top_k"]
    if("tfs" in js):
        vars.tfs        = js["tfs"]
    if("typical" in js):
        vars.typical    = js["typical"]
    if("top_a" in js):
        vars.top_a      = js["top_a"]
    if("rep_pen" in js):
        vars.rep_pen    = js["rep_pen"]
    if("rep_pen_slope" in js):
        vars.rep_pen_slope = js["rep_pen_slope"]
    if("rep_pen_range" in js):
        vars.rep_pen_range = js["rep_pen_range"]
    if("adventure" in js):
        vars.adventure = js["adventure"]
    if("chatmode" in js):
        vars.chatmode = js["chatmode"]
    if("dynamicscan" in js):
        vars.dynamicscan = js["dynamicscan"]
    if("formatoptns" in js):
        vars.formatoptns = js["formatoptns"]
    if("welcome" in js):
        vars.welcome = js["welcome"]
    if("newlinemode" in js):
        vars.newlinemode = js["newlinemode"]
    if("antemplate" in js):
        vars.setauthornotetemplate = js["antemplate"]
        if(not vars.gamestarted):
            vars.authornotetemplate = vars.setauthornotetemplate

#==================================================================#
#  Take settings from vars and write them to client settings file
#==================================================================#
def savesettings():
     # Build json to write
    js = {}
    js["apikey"]      = vars.apikey
    js["andepth"]     = vars.andepth
    js["sampler_order"] = vars.sampler_order
    js["temp"]        = vars.temp
    js["top_p"]       = vars.top_p
    js["top_k"]       = vars.top_k
    js["tfs"]         = vars.tfs
    js["typical"]     = vars.typical
    js["top_a"]       = vars.top_a
    js["rep_pen"]     = vars.rep_pen
    js["rep_pen_slope"] = vars.rep_pen_slope
    js["rep_pen_range"] = vars.rep_pen_range
    js["genamt"]      = vars.genamt
    js["max_length"]  = vars.max_length
    js["ikgen"]       = vars.ikgen
    js["formatoptns"] = vars.formatoptns
    js["numseqs"]     = vars.numseqs
    js["widepth"]     = vars.widepth
    js["useprompt"]   = vars.useprompt
    js["adventure"]   = vars.adventure
    js["chatmode"]    = vars.chatmode
    js["chatname"]    = vars.chatname
    js["dynamicscan"] = vars.dynamicscan
    js["nopromptgen"] = vars.nopromptgen
    js["rngpersist"]  = vars.rngpersist
    js["nogenmod"]    = vars.nogenmod
    js["fulldeterminism"] = vars.full_determinism
    js["autosave"]    = vars.autosave
    js["welcome"]     = vars.welcome
    js["output_streaming"] = vars.output_streaming
    js["show_probs"] = vars.show_probs
    js["show_budget"] = vars.show_budget

    if(vars.seed_specified):
        js["seed"]    = vars.seed
    else:
        js["seed"]    = None

    js["newlinemode"] = vars.newlinemode

    js["antemplate"]  = vars.setauthornotetemplate

    js["userscripts"] = vars.userscripts
    js["corescript"]  = vars.corescript
    js["softprompt"]  = vars.spfilename

    # Write it
    if not os.path.exists('settings'):
        os.mkdir('settings')
    file = open(get_config_filename(), "w")
    try:
        file.write(json.dumps(js, indent=3))
    finally:
        file.close()

#==================================================================#
#  Don't save settings unless 2 seconds have passed without modification
#==================================================================#
@debounce(2)
def settingschanged():
    logger.info("Saving settings.")
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
    if(path.exists(get_config_filename())):
        # Read file contents into JSON object
        file = open(get_config_filename(), "r")
        js   = json.load(file)
        
        processsettings(js)
        file.close()
        
def processsettings(js):
# Copy file contents to vars
    if("apikey" in js):
        # If the model is the HORDE, then previously saved API key in settings
        # Will always override a new key set.
        if vars.model != "CLUSTER" or vars.apikey == '':
            vars.apikey = js["apikey"]
    if("andepth" in js):
        vars.andepth = js["andepth"]
    if("sampler_order" in js):
        sampler_order = js["sampler_order"]
        if(len(sampler_order) < 7):
            sampler_order = [6] + sampler_order
        vars.sampler_order = sampler_order
    if("temp" in js):
        vars.temp = js["temp"]
    if("top_p" in js):
        vars.top_p = js["top_p"]
    if("top_k" in js):
        vars.top_k = js["top_k"]
    if("tfs" in js):
        vars.tfs = js["tfs"]
    if("typical" in js):
        vars.typical = js["typical"]
    if("top_a" in js):
        vars.top_a = js["top_a"]
    if("rep_pen" in js):
        vars.rep_pen = js["rep_pen"]
    if("rep_pen_slope" in js):
        vars.rep_pen_slope = js["rep_pen_slope"]
    if("rep_pen_range" in js):
        vars.rep_pen_range = js["rep_pen_range"]
    if("genamt" in js):
        vars.genamt = js["genamt"]
    if("max_length" in js):
        vars.max_length = js["max_length"]
    if("ikgen" in js):
        vars.ikgen = js["ikgen"]
    if("formatoptns" in js):
        vars.formatoptns = js["formatoptns"]
    if("numseqs" in js):
        vars.numseqs = js["numseqs"]
    if("widepth" in js):
        vars.widepth = js["widepth"]
    if("useprompt" in js):
        vars.useprompt = js["useprompt"]
    if("adventure" in js):
        vars.adventure = js["adventure"]
    if("chatmode" in js):
        vars.chatmode = js["chatmode"]
    if("chatname" in js):
        vars.chatname = js["chatname"]
    if("dynamicscan" in js):
        vars.dynamicscan = js["dynamicscan"]
    if("nopromptgen" in js):
        vars.nopromptgen = js["nopromptgen"]
    if("rngpersist" in js):
        vars.rngpersist = js["rngpersist"]
    if("nogenmod" in js):
        vars.nogenmod = js["nogenmod"]
    if("fulldeterminism" in js):
        vars.full_determinism = js["fulldeterminism"]
    if("autosave" in js):
        vars.autosave = js["autosave"]
    if("newlinemode" in js):
        vars.newlinemode = js["newlinemode"]
    if("welcome" in js):
        vars.welcome = js["welcome"]
    if("output_streaming" in js):
        vars.output_streaming = js["output_streaming"]
    if("show_probs" in js):
        vars.show_probs = js["show_probs"]
    if("show_budget" in js):
        vars.show_budget = js["show_budget"]
    
    if("seed" in js):
        vars.seed = js["seed"]
        if(vars.seed is not None):
            vars.seed_specified = True
        else:
            vars.seed_specified = False
    else:
        vars.seed_specified = False

    if("antemplate" in js):
        vars.setauthornotetemplate = js["antemplate"]
        if(not vars.gamestarted):
            vars.authornotetemplate = vars.setauthornotetemplate
    
    if("userscripts" in js):
        vars.userscripts = []
        for userscript in js["userscripts"]:
            if type(userscript) is not str:
                continue
            userscript = userscript.strip()
            if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                vars.userscripts.append(userscript)

    if("corescript" in js and type(js["corescript"]) is str and all(q not in js["corescript"] for q in ("..", ":")) and all(js["corescript"][0] not in q for q in ("/", "\\"))):
        vars.corescript = js["corescript"]
    else:
        vars.corescript = "default.lua"

#==================================================================#
#  Load a soft prompt from a file
#==================================================================#

def check_for_sp_change():
    while(True):
        time.sleep(0.05)

        if(vars.sp_changed):
            with app.app_context():
                emit('from_server', {'cmd': 'spstatitems', 'data': {vars.spfilename: vars.spmeta} if vars.allowsp and len(vars.spfilename) else {}}, namespace=None, broadcast=True)
            vars.sp_changed = False

        if(vars.token_stream_queue.queue):
            # If emit blocks, waiting for it to complete before clearing could
            # introduce a race condition that drops tokens.
            queued_tokens = list(vars.token_stream_queue.queue)
            vars.token_stream_queue.queue.clear()
            socketio.emit("from_server", {"cmd": "streamtoken", "data": queued_tokens}, namespace=None, broadcast=True)

socketio.start_background_task(check_for_sp_change)

def spRequest(filename):
    if(not vars.allowsp):
        raise RuntimeError("Soft prompts are not supported by your current model/backend")
    
    old_filename = vars.spfilename

    vars.spfilename = ""
    settingschanged()

    if(len(filename) == 0):
        vars.sp = None
        vars.sp_length = 0
        if(old_filename != filename):
            vars.sp_changed = True
        return

    global np
    if 'np' not in globals():
        import numpy as np

    z, version, shape, fortran_order, dtype = fileops.checksp(filename, vars.modeldim)
    if not isinstance(z, zipfile.ZipFile):
        raise RuntimeError(f"{repr(filename)} is not a valid soft prompt file")
    with z.open('meta.json') as f:
        vars.spmeta = json.load(f)
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

    vars.sp_length = tensor.shape[-2]
    vars.spmeta["n_tokens"] = vars.sp_length

    if(vars.use_colab_tpu or vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        rows = tensor.shape[0]
        padding_amount = tpu_mtj_backend.params["seq"] - (tpu_mtj_backend.params["seq"] % -tpu_mtj_backend.params["cores_per_replica"]) - rows
        tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
        tensor = tensor.reshape(
            tpu_mtj_backend.params["cores_per_replica"],
            -1,
            tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"]),
        )
        vars.sp = tpu_mtj_backend.shard_xmap(np.float32(tensor))
    else:
        vars.sp = torch.from_numpy(tensor)

    vars.spfilename = filename
    settingschanged()
    if(old_filename != filename):
            vars.sp_changed = True

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
    parser.add_argument("--apikey", help="Specify the API key to use for online services")
    parser.add_argument("--req_model", type=str, action='append', required=False, help="Which models which we allow to generate for us during cluster mode. Can be specified multiple times.")
    parser.add_argument("--revision", help="Specify the model revision for huggingface models (can be a git branch/tag name or a git commit hash)")
    parser.add_argument("--cpu", action='store_true', help="By default unattended launches are on the GPU use this option to force CPU usage.")
    parser.add_argument("--breakmodel", action='store_true', help=argparse.SUPPRESS)
    parser.add_argument("--breakmodel_layers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--breakmodel_gpulayers", type=str, help="If using a model that supports hybrid generation, this is a comma-separated list that specifies how many layers to put on each GPU device. For example to put 8 layers on device 0, 9 layers on device 1 and 11 layers on device 2, use --breakmodel_gpulayers 8,9,11")
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
    parser.add_argument('-v', '--verbosity', action='count', default=0, help="The default logging level is ERROR or higher. This value increases the amount of logging seen in your screen")
    parser.add_argument('-q', '--quiesce', action='count', default=0, help="The default logging level is ERROR or higher. This value decreases the amount of logging seen in your screen")

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
    
    utils.args = args

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

    vars.model = args.model;
    vars.revision = args.revision

    if args.apikey:
        vars.apikey = args.apikey
    if args.req_model:
        vars.cluster_requested_models = args.req_model

    if args.colab:
        args.remote = True;
        args.override_rename = True;
        args.override_delete = True;
        args.nobreakmodel = True;
        args.quiet = True;
        args.lowmem = True;
        args.noaimenu = True;

    if args.quiet:
        vars.quiet = True

    if args.nobreakmodel:
        vars.nobreakmodel = True;

    if args.remote:
        vars.host = True;

    if args.ngrok:
        vars.host = True;

    if args.localtunnel:
        vars.host = True;

    if args.host:
        vars.host = True;

    if args.cpu:
        vars.use_colab_tpu = False

    vars.smandelete = vars.host == args.override_delete
    vars.smanrename = vars.host == args.override_rename

    vars.aria2_port = args.aria2_port or 6799
    
    #Now let's look to see if we are going to force a load of a model from a user selected folder
    if(vars.model == "selectfolder"):
        print("{0}Please choose the folder where pytorch_model.bin is located:{1}\n".format(colors.CYAN, colors.END))
        modpath = fileops.getdirpath(getcwd() + "/models", "Select Model Folder")
    
        if(modpath):
            # Save directory to vars
            vars.model = "NeoCustom"
            vars.custmodpth = modpath
    elif args.model:
        logger.message(f"Welcome to KoboldAI!")
        logger.message(f"You have selected the following Model: {vars.model}")
        if args.path:
            logger.message(f"You have selected the following path for your Model: {args.path}")
            vars.custmodpth = args.path;
            vars.colaburl = args.path + "/request"; # Lets just use the same parameter to keep it simple
#==================================================================#
# Load Model
#==================================================================# 

def tpumtjgetsofttokens():
    soft_tokens = None
    if(vars.sp is None):
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
        vars.sp = tpu_mtj_backend.shard_xmap(tensor)
    soft_tokens = np.arange(
        tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"],
        tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"] + vars.sp_length,
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
    default_url = None
    models_on_url = False
    multi_online_models = False
    gpu_count = torch.cuda.device_count()
    gpu_names = []
    send_horde_models = False
    for i in range(gpu_count):
        gpu_names.append(torch.cuda.get_device_name(i))
    if model in ['Colab', 'API']:
        url = True
    elif model == 'CLUSTER':
        models_on_url = True
        url = True
        key = True
        default_url = 'https://horde.koboldai.net'
        multi_online_models = True
        if path.exists(get_config_filename(model)):
            with open(get_config_filename(model), "r") as file:
                # Check if API key exists
                js = json.load(file)
                if("apikey" in js and js["apikey"] != ""):
                    # API key exists, grab it and close the file
                    key_value = js["apikey"]
                elif 'oaiapikey' in js and js['oaiapikey'] != "":
                    key_value = js["oaiapikey"]
                if 'url' in js and js['url'] != "":
                    url = js['url']
            if key_value != "":
                send_horde_models = True
    elif model in [x[1] for x in model_menu['apilist']]:
        if path.exists(get_config_filename(model)):
            with open(get_config_filename(model), "r") as file:
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
    elif not utils.HAS_ACCELERATE and not torch.cuda.is_available():
        pass
    elif args.cpu:
        pass
    else:
        layer_count = get_layer_count(model, directory=directory)
        if layer_count is None:
            breakmodel = False
            gpu = True
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
            break_values += [0] * (gpu_count - len(break_values))
    #print("Model_info: {}".format({'cmd': 'selected_model_info', 'key_value': key_value, 'key':key, 
    #                     'gpu':gpu, 'layer_count':layer_count, 'breakmodel':breakmodel, 
    #                     'break_values': break_values, 'gpu_count': gpu_count,
    #                     'url': url, 'gpu_names': gpu_names}))
    emit('from_server', {'cmd': 'selected_model_info', 'key_value': key_value, 'key':key, 
                         'gpu':gpu, 'layer_count':layer_count, 'breakmodel':breakmodel, 
                         'disk_break_value': disk_blocks, 'accelerate': utils.HAS_ACCELERATE,
                         'break_values': break_values, 'gpu_count': gpu_count, 'multi_online_models': multi_online_models,
                         'url': url, 'default_url': default_url, 'gpu_names': gpu_names, 'models_on_url': models_on_url}, broadcast=True)
    if send_horde_models:
        get_cluster_models({'key': key_value, 'url': default_url})
    elif key_value != "" and model in [x[1] for x in model_menu['apilist']] and model != 'CLUSTER':
        get_oai_models(key_value)
    

def get_layer_count(model, directory=""):
    if(model not in ["InferKit", "Colab", "API", "CLUSTER", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ"]):
        if(model == "GPT2Custom"):
            with open(os.path.join(directory, "config.json"), "r") as f:
                model_config = json.load(f)
        # Get the model_type from the config or assume a model type if it isn't present
        else:
            if(directory):
                model = directory
            from transformers import AutoConfig
            if(os.path.isdir(model.replace('/', '_'))):
                model_config = AutoConfig.from_pretrained(model.replace('/', '_'), revision=args.revision, cache_dir="cache")
            elif(os.path.isdir("models/{}".format(model.replace('/', '_')))):
                model_config = AutoConfig.from_pretrained("models/{}".format(model.replace('/', '_')), revision=args.revision, cache_dir="cache")
            elif(os.path.isdir(directory)):
                model_config = AutoConfig.from_pretrained(directory, revision=args.revision, cache_dir="cache")
            else:
                model_config = AutoConfig.from_pretrained(model, revision=args.revision, cache_dir="cache")
        try:
            if ((utils.HAS_ACCELERATE and model_config.model_type != 'gpt2') or model_config.model_type in ("gpt_neo", "gptj", "xglm", "opt")) and not vars.nobreakmodel:
                return utils.num_layers(model_config)
            else:
                return None
        except:
            return None
    else:
        return None

def get_oai_models(key):
    vars.oaiapikey = key
    if vars.model_selected == 'OAI':
        url = "https://api.openai.com/v1/engines"
    elif vars.model_selected == 'GooseAI':
        url = "https://api.goose.ai/v1/engines"
    else:
        return
        
    # Get list of models from OAI
    logger.init("OAI Engines", status="Retrieving")
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
            logger.error(engines)
            raise
        
        online_model = ""
        changed=False
        
        #Save the key
        if not path.exists("settings"):
            # If the client settings file doesn't exist, create it
            # Write API key to file
            os.makedirs('settings', exist_ok=True)
        if path.exists(get_config_filename(vars.model_selected)):
            with open(get_config_filename(vars.model_selected), "r") as file:
                js = json.load(file)
                if 'online_model' in js:
                    online_model = js['online_model']
                if "apikey" in js:
                    if js['apikey'] != key:
                        changed=True
        else:
            changed=True
        if changed:
            js={}
            with open(get_config_filename(vars.model_selected), "w") as file:
                js["apikey"] = key
                file.write(json.dumps(js, indent=3))
            
        logger.init_ok("OAI Engines", status="OK")
        emit('from_server', {'cmd': 'oai_engines', 'data': engines, 'online_model': online_model}, broadcast=True)
    else:
        # Something went wrong, print the message and quit since we can't initialize an engine
        logger.init_err("OAI Engines", status="Failed")
        logger.error(req.json())
        emit('from_server', {'cmd': 'errmsg', 'data': req.json()})

def get_cluster_models(msg):
    vars.oaiapikey = msg['key']
    vars.apikey = vars.oaiapikey
    url = msg['url']
    # Get list of models from public cluster
    logger.init("KAI Horde Models", status="Retrieving")
    try:
        req = requests.get(f"{url}/api/v2/status/models?type=text")
    except requests.exceptions.ConnectionError:
        logger.init_err("KAI Horde Models", status="Failed")
        logger.error("Provided KoboldAI Horde URL unreachable")
        emit('from_server', {'cmd': 'errmsg', 'data': "Provided KoboldAI Horde URL unreachable"})
        return
    if(not req.ok):
        # Something went wrong, print the message and quit since we can't initialize an engine
        logger.init_err("KAI Horde Models", status="Failed")
        logger.error(req.json())
        emit('from_server', {'cmd': 'errmsg', 'data': req.json()})
        return

    engines = req.json()
    logger.debug(engines)
    try:
        engines = [[en["name"], en["name"]] for en in engines]
    except:
        logger.error(engines)
        raise
    logger.debug(engines)
    
    online_model = ""
    changed=False
    
    #Save the key
    if not path.exists("settings"):
        # If the client settings file doesn't exist, create it
        # Write API key to file
        os.makedirs('settings', exist_ok=True)
    if path.exists(get_config_filename(vars.model_selected)):
        with open(get_config_filename(vars.model_selected), "r") as file:
            js = json.load(file)
            if 'online_model' in js:
                online_model = js['online_model']
            if "apikey" in js:
                if js['apikey'] != vars.oaiapikey:
                    changed=True
    else:
        changed=True
    if changed:
        js={}
        with open(get_config_filename(vars.model_selected), "w") as file:
            js["apikey"] = vars.oaiapikey
            js["url"] = url
            file.write(json.dumps(js, indent=3))
        
    logger.init_ok("KAI Horde Models", status="OK")
    emit('from_server', {'cmd': 'oai_engines', 'data': engines, 'online_model': online_model}, broadcast=True)


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
        if(vars.sp is not None):
            shifted_input_ids = input_ids - model.config.vocab_size
        input_ids.clamp_(max=model.config.vocab_size-1)
        inputs_embeds = old_embedding_call(self, input_ids, *args, **kwargs)
        if(vars.sp is not None):
            vars.sp = vars.sp.to(inputs_embeds.dtype).to(inputs_embeds.device)
            inputs_embeds = torch.where(
                (shifted_input_ids >= 0)[..., None],
                vars.sp[shifted_input_ids.clamp(min=0)],
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
                    emit('from_server', {'cmd': 'model_load_status', 'data': bar.replace(" ", "&nbsp;")}, broadcast=True)
                    eventlet.sleep(seconds=0)
                except:
                    pass
    def http_get(
        url: str,
        temp_file,
        proxies=None,
        resume_size=0,
        headers=None,
        file_name=None,
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
        vars.fp32_model = False
        utils.num_shards = None
        utils.current_shard = 0
        utils.from_pretrained_model_name = pretrained_model_name_or_path
        utils.from_pretrained_index_filename = None
        utils.from_pretrained_kwargs = kwargs
        utils.bar = None
        if not args.no_aria2:
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


    # Patch transformers to use our custom logit warpers
    from transformers import LogitsProcessorList, LogitsWarper, LogitsProcessor, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor
    from warpers import AdvancedRepetitionPenaltyLogitsProcessor, TailFreeLogitsWarper, TypicalLogitsWarper, TopALogitsWarper

    def dynamic_processor_wrap(cls, field_name, var_name, cond=None):
        old_call = cls.__call__
        def new_call(self, *args, **kwargs):
            if(not isinstance(field_name, str) and isinstance(field_name, Iterable)):
                conds = []
                for f, v in zip(field_name, var_name):
                    conds.append(getattr(vars, v))
                    setattr(self, f, conds[-1])
            else:
                conds = getattr(vars, var_name)
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

    class LuaLogitsProcessor(LogitsProcessor):

        def __init__(self):
            pass

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            assert scores.ndim == 2
            assert input_ids.ndim == 2
            self.regeneration_required = False
            self.halt = False

            if(vars.standalone):
                return scores

            scores_shape = scores.shape
            scores_list = scores.tolist()
            vars.lua_koboldbridge.logits = vars.lua_state.table()
            for r, row in enumerate(scores_list):
                vars.lua_koboldbridge.logits[r+1] = vars.lua_state.table(*row)
            vars.lua_koboldbridge.vocab_size = scores_shape[-1]

            execute_genmod()

            scores = torch.tensor(
                tuple(tuple(row.values()) for row in vars.lua_koboldbridge.logits.values()),
                device=scores.device,
                dtype=scores.dtype,
            )
            assert scores.shape == scores_shape

            return scores

    from torch.nn import functional as F

    def visualize_probabilities(scores: torch.FloatTensor) -> None:
        assert scores.ndim == 2

        if vars.numseqs > 1 or not vars.show_probs:
            return

        probs = F.softmax(scores, dim = -1).cpu().numpy()[0]
        token_prob_info = []
        for token_id, score in sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:8]:
            token_prob_info.append({
                "tokenId": token_id,
                "decoded": utils.decodenewlines(tokenizer.decode(token_id)),
                "score": float(score),
            })

        vars.token_stream_queue.probability_buffer = token_prob_info
    
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
            self.__warper_list.append(AdvancedRepetitionPenaltyLogitsProcessor())

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, *args, **kwargs):
            sampler_order = vars.sampler_order[:]
            if len(sampler_order) < 7:  # Add repetition penalty at beginning if it's not present
                sampler_order = [6] + sampler_order
            for k in sampler_order:
                scores = self.__warper_list[k](input_ids, scores, *args, **kwargs)
            visualize_probabilities(scores)
            return scores

    def new_get_logits_warper(beams: int = 1,) -> LogitsProcessorList:
        return KoboldLogitsWarperList(beams=beams)
    
    def new_sample(self, *args, **kwargs):
        assert kwargs.pop("logits_warper", None) is not None
        kwargs["logits_warper"] = new_get_logits_warper(
            beams=1,
        )
        if(vars.newlinemode == "s") or (vars.newlinemode == "ns"):
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
            # Do not intermingle multiple generations' outputs!
            if vars.numseqs > 1:
                return False

            if not (vars.show_probs or vars.output_streaming):
                return False

            if vars.chatmode:
                return False
            tokenizer_text = utils.decodenewlines(tokenizer.decode(input_ids[0, -1]))
            vars.token_stream_queue.add_text(tokenizer_text)
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
            vars.generated_tkns += 1
            if(not vars.standalone and vars.lua_koboldbridge.generated_cols and vars.generated_tkns != vars.lua_koboldbridge.generated_cols):
                raise RuntimeError(f"Inconsistency detected between KoboldAI Python and Lua backends ({vars.generated_tkns} != {vars.lua_koboldbridge.generated_cols})")
            if(vars.abort or vars.generated_tkns >= vars.genamt):
                self.regeneration_required = False
                self.halt = False
                return True
            if(vars.standalone):
                return False

            assert input_ids.ndim == 2
            assert len(self.excluded_world_info) == input_ids.shape[0]
            self.regeneration_required = vars.lua_koboldbridge.regeneration_required
            self.halt = not vars.lua_koboldbridge.generating
            vars.lua_koboldbridge.regeneration_required = False

            for i in range(vars.numseqs):
                vars.lua_koboldbridge.generated[i+1][vars.generated_tkns] = int(input_ids[i, -1].item())

            if(not vars.dynamicscan):
                return self.regeneration_required or self.halt
            tail = input_ids[..., -vars.generated_tkns:]
            for i, t in enumerate(tail):
                decoded = utils.decodenewlines(tokenizer.decode(t))
                _, found = checkworldinfo(decoded, force_use_txt=True, actions=vars._actions)
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
        stopping_criteria.insert(0, token_streamer)
        return stopping_criteria
    transformers.generation_utils.GenerationMixin._get_stopping_criteria = new_get_stopping_criteria

def reset_model_settings():
    vars.socketio = socketio
    vars.max_length  = 1024    # Maximum number of tokens to submit per action
    vars.ikmax       = 3000    # Maximum number of characters to submit to InferKit
    vars.genamt      = 80      # Amount of text for each action to generate
    vars.ikgen       = 200     # Number of characters for InferKit to generate
    vars.rep_pen     = 1.1     # Default generator repetition_penalty
    vars.rep_pen_slope = 0.7   # Default generator repetition penalty slope
    vars.rep_pen_range = 1024  # Default generator repetition penalty range
    vars.temp        = 0.5     # Default generator temperature
    vars.top_p       = 0.9     # Default generator top_p
    vars.top_k       = 0       # Default generator top_k
    vars.top_a       = 0.0     # Default generator top-a
    vars.tfs         = 1.0     # Default generator tfs (tail-free sampling)
    vars.typical     = 1.0     # Default generator typical sampling threshold
    vars.numseqs     = 1       # Number of sequences to ask the generator to create
    vars.generated_tkns = 0    # If using a backend that supports Lua generation modifiers, how many tokens have already been generated, otherwise 0
    vars.badwordsids = []
    vars.fp32_model  = False  # Whether or not the most recently loaded HF model was in fp32 format
    vars.modeldim    = -1     # Embedding dimension of your model (e.g. it's 4096 for GPT-J-6B and 2560 for GPT-Neo-2.7B)
    vars.sampler_order = [6, 0, 1, 2, 3, 4, 5]
    vars.newlinemode = "n"
    vars.revision    = None
    vars.lazy_load = True
    

def load_model(use_gpu=True, gpu_layers=None, disk_layers=None, initial_load=False, online_model="", use_breakmodel_args=False, breakmodel_args_default_to_cpu=False):
    global model
    global generator
    global torch
    global model_config
    global GPT2Tokenizer
    global tokenizer
    if(initial_load):
        use_breakmodel_args = True
    reset_model_settings()
    if not utils.HAS_ACCELERATE:
        disk_layers = None
    vars.noai = False
    if not use_breakmodel_args:
        set_aibusy(True)
        if vars.model != 'ReadOnly':
            emit('from_server', {'cmd': 'model_load_status', 'data': "Loading {}".format(vars.model)}, broadcast=True)
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
    vars.online_model = ''
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
        torch.cuda.empty_cache()
    except:
        pass
        
    #Reload our badwords
    vars.badwordsids = vars.badwordsids_default
    
    if online_model == "":
        vars.configname = getmodelname()
    #Let's set the GooseAI or OpenAI server URLs if that's applicable
    else:
        vars.online_model = online_model
        # Swap OAI Server if GooseAI was selected
        if(vars.model == "GooseAI"):
            vars.oaiengines = "https://api.goose.ai/v1/engines"
            vars.model = "OAI"
            vars.configname = f"GooseAI_{online_model.replace('/', '_')}"
        elif(vars.model == "CLUSTER") and type(online_model) is list:
                if len(online_model) != 1:
                    vars.configname = vars.model
                else:
                    vars.configname = f"{vars.model}_{online_model[0].replace('/', '_')}"
        else:
            vars.configname = f"{vars.model}_{online_model.replace('/', '_')}"
        if path.exists(get_config_filename()):
            changed=False
            with open(get_config_filename(), "r") as file:
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
                with open(get_config_filename(), "w") as file:
                    file.write(json.dumps(js, indent=3))

        # Swap OAI Server if GooseAI was selected
        if(vars.model == "GooseAI"):
            vars.oaiengines = "https://api.goose.ai/v1/engines"
            vars.model = "OAI"
            args.configname = "GooseAI" + "/" + online_model
        elif vars.model != "CLUSTER":
            args.configname = vars.model + "/" + online_model
        vars.oaiurl = vars.oaiengines + "/{0}/completions".format(online_model)
    
    
    # If transformers model was selected & GPU available, ask to use CPU or GPU
    if(vars.model not in ["InferKit", "Colab", "API", "CLUSTER", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        vars.allowsp = True
        # Test for GPU support
        
        # Make model path the same as the model name to make this consistent with the other loading method if it isn't a known model type
        # This code is not just a workaround for below, it is also used to make the behavior consistent with other loading methods - Henk717
        if(not vars.model in ["NeoCustom", "GPT2Custom"]):
            vars.custmodpth = vars.model
        elif(vars.model == "NeoCustom"):
            vars.model = os.path.basename(os.path.normpath(vars.custmodpth))

        # Get the model_type from the config or assume a model type if it isn't present
        from transformers import AutoConfig
        if(os.path.isdir(vars.custmodpth.replace('/', '_'))):
            try:
                model_config = AutoConfig.from_pretrained(vars.custmodpth.replace('/', '_'), revision=args.revision, cache_dir="cache")
                vars.model_type = model_config.model_type
            except ValueError as e:
                vars.model_type = "not_found"
        elif(os.path.isdir("models/{}".format(vars.custmodpth.replace('/', '_')))):
            try:
                model_config = AutoConfig.from_pretrained("models/{}".format(vars.custmodpth.replace('/', '_')), revision=args.revision, cache_dir="cache")
                vars.model_type = model_config.model_type
            except ValueError as e:
                vars.model_type = "not_found"
        else:
            try:
                model_config = AutoConfig.from_pretrained(vars.custmodpth, revision=args.revision, cache_dir="cache")
                vars.model_type = model_config.model_type
            except ValueError as e:
                vars.model_type = "not_found"
        if(vars.model_type == "not_found" and vars.model == "NeoCustom"):
            vars.model_type = "gpt_neo"
        elif(vars.model_type == "not_found" and vars.model == "GPT2Custom"):
            vars.model_type = "gpt2"
        elif(vars.model_type == "not_found"):
            logger.warning("No model type detected, assuming Neo (If this is a GPT2 model use the other menu option or --model GPT2Custom)")
            vars.model_type = "gpt_neo"

    if(not vars.use_colab_tpu and vars.model not in ["InferKit", "Colab", "API", "CLUSTER", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        loadmodelsettings()
        loadsettings()
        logger.init("GPU support", status="Searching")
        vars.hascuda = torch.cuda.is_available() and not args.cpu
        vars.bmsupported = ((utils.HAS_ACCELERATE and vars.model_type != 'gpt2') or vars.model_type in ("gpt_neo", "gptj", "xglm", "opt")) and not vars.nobreakmodel
        if(args.breakmodel is not None and args.breakmodel):
            logger.warning("--breakmodel is no longer supported. Breakmodel mode is now automatically enabled when --breakmodel_gpulayers is used (see --help for details).")
        if(args.breakmodel_layers is not None):
            logger.warning("--breakmodel_layers is deprecated. Use --breakmodel_gpulayers instead (see --help for details).")
        if(args.model and vars.bmsupported and not args.breakmodel_gpulayers and not args.breakmodel_layers and (not utils.HAS_ACCELERATE or not args.breakmodel_disklayers)):
            logger.warning("Model launched without the --breakmodel_gpulayers argument, defaulting to GPU only mode.")
            vars.bmsupported = False
        if(not vars.bmsupported and (args.breakmodel_gpulayers is not None or args.breakmodel_layers is not None or args.breakmodel_disklayers is not None)):
            logger.warning("This model does not support hybrid generation. --breakmodel_gpulayers will be ignored.")
        if(vars.hascuda):
            logger.init_ok("GPU support", status="Found")
        else:
            logger.init_warn("GPU support", status="Not Found")
        
        if args.cpu:
            vars.usegpu = False
            gpu_layers = None
            disk_layers = None
            vars.breakmodel = False
        elif vars.hascuda:
            if(vars.bmsupported):
                vars.usegpu = False
                vars.breakmodel = True
            else:
                vars.breakmodel = False
                vars.usegpu = use_gpu


    # Ask for API key if InferKit was selected
    if(vars.model == "InferKit"):
        vars.apikey = vars.oaiapikey
                    
    # Swap OAI Server if GooseAI was selected
    if(vars.model == "GooseAI"):
        vars.oaiengines = "https://api.goose.ai/v1/engines"
        vars.model = "OAI"
        vars.configname = "GooseAI"

    # Ask for API key if OpenAI was selected
    if(vars.model == "OAI"):
        if not vars.configname:
            vars.configname = "OAI"
        
    if(vars.model == "ReadOnly"):
        vars.noai = True

    # Start transformers and create pipeline
    if(not vars.use_colab_tpu and vars.model not in ["InferKit", "Colab", "API", "CLUSTER", "OAI", "GooseAI" , "ReadOnly", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        if(not vars.noai):
            logger.init("Transformers", status='Starting')
            for m in ("GPTJModel", "XGLMModel"):
                try:
                    globals()[m] = getattr(__import__("transformers"), m)
                except:
                    pass

            # Lazy loader
            import torch_lazy_loader
            def get_lazy_load_callback(n_layers, convert_to_float16=True):
                if not vars.lazy_load:
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
                            device_map[key] = vars.gpu_device if vars.hascuda and vars.usegpu else "cpu" if not vars.hascuda or not vars.breakmodel else breakmodel.primary_device
                        else:
                            layer = int(max((n for n in utils.layers_module_names if original_key.startswith(n)), key=len).rsplit(".", 1)[1])
                            device = vars.gpu_device if vars.hascuda and vars.usegpu else "disk" if layer < disk_blocks and layer < ram_blocks else "cpu" if not vars.hascuda or not vars.breakmodel else "shared" if layer < ram_blocks else bisect.bisect_right(cumulative_gpu_blocks, layer - ram_blocks)
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
                        utils.bar = tqdm(total=num_tensors, desc=f"{colors.PURPLE}INIT{colors.END}       | Loading model tensors", file=Send_to_socketio())

                    with zipfile.ZipFile(f, "r") as z:
                        try:
                            last_storage_key = None
                            zipfolder = os.path.basename(os.path.normpath(f)).split('.')[0]
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
                                    try:
                                        f = z.open(f"archive/data/{storage_key}")
                                    except:
                                        f = z.open(f"{zipfolder}/data/{storage_key}")
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
                                    vars.fp32_model = True
                                if convert_to_float16 and breakmodel.primary_device != "cpu" and vars.hascuda and (vars.breakmodel or vars.usegpu) and model_dict[key].dtype is torch.float32:
                                    model_dict[key] = model_dict[key].to(torch.float16)
                                if breakmodel.primary_device == "cpu" or (not vars.usegpu and not vars.breakmodel and model_dict[key].dtype is torch.float16):
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
                                        dtype = tensor.dtype
                                        if convert_to_float16 and breakmodel.primary_device != "cpu" and vars.hascuda and (vars.breakmodel or vars.usegpu):
                                            dtype = torch.float16
                                        if breakmodel.primary_device == "cpu" or (not vars.usegpu and not vars.breakmodel):
                                            dtype = torch.float32
                                        if name in model_dict and model_dict[name].dtype is not dtype:
                                            model_dict[name] = model_dict[name].to(dtype)
                                        if tensor.dtype is not dtype:
                                            tensor = tensor.to(dtype)
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


            def maybe_low_cpu_mem_usage() -> Dict[str, Any]:
                if(packaging.version.parse(transformers_version) < packaging.version.parse("4.11.0")):
                    logger.warning(f"Please upgrade to transformers 4.11.0 for lower RAM usage. You have transformers {transformers_version}.")
                    return {}
                return {"low_cpu_mem_usage": True}
            
            @contextlib.contextmanager
            def maybe_use_float16(always_use=False):
                if(always_use or (vars.hascuda and args.lowmem and (vars.usegpu or vars.breakmodel))):
                    original_dtype = torch.get_default_dtype()
                    torch.set_default_dtype(torch.float16)
                    yield True
                    torch.set_default_dtype(original_dtype)
                else:
                    yield False

            # If custom GPT2 model was chosen
            if(vars.model_type == "gpt2"):
                vars.lazy_load = False
                if os.path.exists(vars.custmodpth):
                    model_config = open(vars.custmodpth + "/config.json", "r")
                elif os.path.exists(os.path.join("models/", vars.custmodpth)):
                    config_path = os.path.join("models/", vars.custmodpth)
                    config_path = os.path.join(config_path, "config.json").replace("\\", "//")
                    model_config = open(config_path, "r")
                #js   = json.load(model_config)
                with(maybe_use_float16()):
                    try:
                        if os.path.exists(vars.custmodpth):
                            model = GPT2LMHeadModel.from_pretrained(vars.custmodpth, revision=args.revision, cache_dir="cache")
                            tokenizer = GPT2Tokenizer.from_pretrained(vars.custmodpth, revision=args.revision, cache_dir="cache")
                        elif os.path.exists(os.path.join("models/", vars.custmodpth)):
                            model = GPT2LMHeadModel.from_pretrained(os.path.join("models/", vars.custmodpth), revision=args.revision, cache_dir="cache")
                            tokenizer = GPT2Tokenizer.from_pretrained(os.path.join("models/", vars.custmodpth), revision=args.revision, cache_dir="cache")
                        else:
                            model = GPT2LMHeadModel.from_pretrained(vars.custmodpth, revision=args.revision, cache_dir="cache")
                            tokenizer = GPT2Tokenizer.from_pretrained(vars.custmodpth, revision=args.revision, cache_dir="cache")
                    except Exception as e:
                        if("out of memory" in traceback.format_exc().lower()):
                            raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                        raise e
                tokenizer = GPT2Tokenizer.from_pretrained(vars.custmodpth, revision=args.revision, cache_dir="cache")
                model.save_pretrained("models/{}".format(vars.model.replace('/', '_')), max_shard_size="500MiB")
                tokenizer.save_pretrained("models/{}".format(vars.model.replace('/', '_')))
                vars.modeldim = get_hidden_size_from_model(model)
                # Is CUDA available? If so, use GPU, otherwise fall back to CPU
                if(vars.hascuda and vars.usegpu):
                    model = model.half().to(vars.gpu_device)
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
                if(vars.model_type == "gpt2"):
                    lowmem = {}
                    vars.lazy_load = False  # Also, lazy loader doesn't support GPT-2 models
                
                # If we're using torch_lazy_loader, we need to get breakmodel config
                # early so that it knows where to load the individual model tensors
                if (utils.HAS_ACCELERATE or vars.lazy_load and vars.hascuda and vars.breakmodel) and not vars.nobreakmodel:
                    device_config(model_config)

                # Download model from Huggingface if it does not exist, otherwise load locally
                
                #If we specify a model and it's in the root directory, we need to move it to the models directory (legacy folder structure to new)
                if os.path.isdir(vars.model.replace('/', '_')):
                    import shutil
                    shutil.move(vars.model.replace('/', '_'), "models/{}".format(vars.model.replace('/', '_')))
                if(vars.lazy_load):  # If we're using lazy loader, we need to figure out what the model's hidden layers are called
                    with torch_lazy_loader.use_lazy_torch_load(dematerialized_modules=True, use_accelerate_init_empty_weights=True):
                        try:
                            metamodel = AutoModelForCausalLM.from_config(model_config)
                        except Exception as e:
                            metamodel = GPTNeoForCausalLM.from_config(model_config)
                        utils.layers_module_names = utils.get_layers_module_names(metamodel)
                        utils.module_names = list(metamodel.state_dict().keys())
                        utils.named_buffers = list(metamodel.named_buffers(recurse=True))
                with maybe_use_float16(), torch_lazy_loader.use_lazy_torch_load(enable=vars.lazy_load, callback=get_lazy_load_callback(utils.num_layers(model_config)) if vars.lazy_load else None, dematerialized_modules=True):
                    if(vars.lazy_load):  # torch_lazy_loader.py and low_cpu_mem_usage can't be used at the same time
                        lowmem = {}
                    if(os.path.isdir(vars.custmodpth)):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(vars.custmodpth, revision=args.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = AutoTokenizer.from_pretrained(vars.custmodpth, revision=args.revision, cache_dir="cache")
                            except Exception as e:
                                try:
                                    tokenizer = GPT2Tokenizer.from_pretrained(vars.custmodpth, revision=args.revision, cache_dir="cache")
                                except Exception as e:
                                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=args.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained(vars.custmodpth, revision=args.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained(vars.custmodpth, revision=args.revision, cache_dir="cache", **lowmem)
                    elif(os.path.isdir("models/{}".format(vars.model.replace('/', '_')))):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(vars.model.replace('/', '_')), revision=args.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = AutoTokenizer.from_pretrained("models/{}".format(vars.model.replace('/', '_')), revision=args.revision, cache_dir="cache")
                            except Exception as e:
                                try:
                                    tokenizer = GPT2Tokenizer.from_pretrained("models/{}".format(vars.model.replace('/', '_')), revision=args.revision, cache_dir="cache")
                                except Exception as e:
                                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=args.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained("models/{}".format(vars.model.replace('/', '_')), revision=args.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained("models/{}".format(vars.model.replace('/', '_')), revision=args.revision, cache_dir="cache", **lowmem)
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
                                vars.fp32_model = True
                            return old_rebuild_tensor(storage, storage_offset, shape, stride)
                        torch._utils._rebuild_tensor = new_rebuild_tensor

                        try:
                            tokenizer = AutoTokenizer.from_pretrained(vars.model, revision=args.revision, cache_dir="cache", use_fast=False)
                        except Exception as e:
                            try:
                                tokenizer = AutoTokenizer.from_pretrained(vars.model, revision=args.revision, cache_dir="cache")
                            except Exception as e:
                                try:
                                    tokenizer = GPT2Tokenizer.from_pretrained(vars.model, revision=args.revision, cache_dir="cache")
                                except Exception as e:
                                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=args.revision, cache_dir="cache")
                        try:
                            model     = AutoModelForCausalLM.from_pretrained(vars.model, revision=args.revision, cache_dir="cache", **lowmem)
                        except Exception as e:
                            if("out of memory" in traceback.format_exc().lower()):
                                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                            model     = GPTNeoForCausalLM.from_pretrained(vars.model, revision=args.revision, cache_dir="cache", **lowmem)

                        torch._utils._rebuild_tensor = old_rebuild_tensor

                        if not args.colab or args.savemodel:
                            import shutil
                            tokenizer.save_pretrained("models/{}".format(vars.model.replace('/', '_')))
                            if(vars.fp32_model and ("breakmodel" not in globals() or not breakmodel.disk_blocks)):  # Use save_pretrained to convert fp32 models to fp16, unless we are using disk cache because save_pretrained is not supported in that case
                                model = model.half()
                                model.save_pretrained("models/{}".format(vars.model.replace('/', '_')), max_shard_size="500MiB")
                            else:  # For fp16 models, we can just copy the model files directly
                                import transformers.configuration_utils
                                import transformers.modeling_utils
                                import transformers.file_utils
                                import huggingface_hub
                                legacy = packaging.version.parse(transformers_version) < packaging.version.parse("4.22.0.dev0")
                                # Save the config.json
                                shutil.move(os.path.realpath(huggingface_hub.hf_hub_download(vars.model, transformers.configuration_utils.CONFIG_NAME, revision=args.revision, cache_dir="cache", local_files_only=True, legacy_cache_layout=legacy)), os.path.join("models/{}".format(vars.model.replace('/', '_')), transformers.configuration_utils.CONFIG_NAME))
                                if(utils.num_shards is None):
                                    # Save the pytorch_model.bin of an unsharded model
                                    shutil.move(os.path.realpath(huggingface_hub.hf_hub_download(vars.model, transformers.modeling_utils.WEIGHTS_NAME, revision=args.revision, cache_dir="cache", local_files_only=True, legacy_cache_layout=legacy)), os.path.join("models/{}".format(vars.model.replace('/', '_')), transformers.modeling_utils.WEIGHTS_NAME))
                                else:
                                    with open(utils.from_pretrained_index_filename) as f:
                                        map_data = json.load(f)
                                    filenames = set(map_data["weight_map"].values())
                                    # Save the pytorch_model.bin.index.json of a sharded model
                                    shutil.move(os.path.realpath(utils.from_pretrained_index_filename), os.path.join("models/{}".format(vars.model.replace('/', '_')), transformers.modeling_utils.WEIGHTS_INDEX_NAME))
                                    # Then save the pytorch_model-#####-of-#####.bin files
                                    for filename in filenames:
                                        shutil.move(os.path.realpath(huggingface_hub.hf_hub_download(vars.model, filename, revision=args.revision, cache_dir="cache", local_files_only=True, legacy_cache_layout=legacy)), os.path.join("models/{}".format(vars.model.replace('/', '_')), filename))
                            shutil.rmtree("cache/")

                if(vars.badwordsids is vars.badwordsids_default and vars.model_type not in ("gpt2", "gpt_neo", "gptj")):
                    vars.badwordsids = [[v] for k, v in tokenizer.get_vocab().items() if any(c in str(k) for c in "<>[]") if vars.newlinemode != "s" or str(k) != "</s>"]

                patch_causallm(model)

                if(vars.hascuda):
                    if(vars.usegpu):
                        vars.modeldim = get_hidden_size_from_model(model)
                        model = model.half().to(vars.gpu_device)
                        generator = model.generate
                    elif(vars.breakmodel):  # Use both RAM and VRAM (breakmodel)
                        vars.modeldim = get_hidden_size_from_model(model)
                        if(not vars.lazy_load):
                            device_config(model.config)
                        move_model_to_devices(model)
                    elif(utils.HAS_ACCELERATE and __import__("breakmodel").disk_blocks > 0):
                        move_model_to_devices(model)
                        vars.modeldim = get_hidden_size_from_model(model)
                        generator = model.generate
                    else:
                        model = model.to('cpu').float()
                        vars.modeldim = get_hidden_size_from_model(model)
                        generator = model.generate
                elif(utils.HAS_ACCELERATE and __import__("breakmodel").disk_blocks > 0):
                    move_model_to_devices(model)
                    vars.modeldim = get_hidden_size_from_model(model)
                    generator = model.generate
                else:
                    model.to('cpu').float()
                    vars.modeldim = get_hidden_size_from_model(model)
                    generator = model.generate
            
            # Suppress Author's Note by flagging square brackets (Old implementation)
            #vocab         = tokenizer.get_vocab()
            #vocab_keys    = vocab.keys()
            #vars.badwords = gettokenids("[")
            #for key in vars.badwords:
            #    vars.badwordsids.append([vocab[key]])
            
            logger.info(f"Pipeline created: {vars.model}")
        
        else:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=args.revision, cache_dir="cache")
    else:
        from transformers import PreTrainedModel
        from transformers import modeling_utils
        old_from_pretrained = PreTrainedModel.from_pretrained.__func__
        @classmethod
        def new_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            vars.fp32_model = False
            utils.num_shards = None
            utils.current_shard = 0
            utils.from_pretrained_model_name = pretrained_model_name_or_path
            utils.from_pretrained_index_filename = None
            utils.from_pretrained_kwargs = kwargs
            utils.bar = None
            if not args.no_aria2:
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


        def tpumtjgenerate_warper_callback(scores) -> "np.array":
            scores_shape = scores.shape
            scores_list = scores.tolist()
            vars.lua_koboldbridge.logits = vars.lua_state.table()
            for r, row in enumerate(scores_list):
                vars.lua_koboldbridge.logits[r+1] = vars.lua_state.table(*row)
            vars.lua_koboldbridge.vocab_size = scores_shape[-1]

            execute_genmod()

            scores = np.array(
                tuple(tuple(row.values()) for row in vars.lua_koboldbridge.logits.values()),
                dtype=scores.dtype,
            )
            assert scores.shape == scores_shape

            return scores
        
        def tpumtjgenerate_stopping_callback(generated, n_generated, excluded_world_info) -> Tuple[List[set], bool, bool]:
            vars.generated_tkns += 1

            assert len(excluded_world_info) == len(generated)
            regeneration_required = vars.lua_koboldbridge.regeneration_required
            halt = vars.abort or not vars.lua_koboldbridge.generating or vars.generated_tkns >= vars.genamt
            vars.lua_koboldbridge.regeneration_required = False

            global past

            for i in range(vars.numseqs):
                vars.lua_koboldbridge.generated[i+1][vars.generated_tkns] = int(generated[i, tpu_mtj_backend.params["seq"] + n_generated - 1].item())

            if(not vars.dynamicscan or halt):
                return excluded_world_info, regeneration_required, halt

            for i, t in enumerate(generated):
                decoded = utils.decodenewlines(tokenizer.decode(past[i])) + utils.decodenewlines(tokenizer.decode(t[tpu_mtj_backend.params["seq"] : tpu_mtj_backend.params["seq"] + n_generated]))
                _, found = checkworldinfo(decoded, force_use_txt=True, actions=vars._actions)
                found -= excluded_world_info[i]
                if(len(found) != 0):
                    regeneration_required = True
                    break
            return excluded_world_info, regeneration_required, halt

        def tpumtjgenerate_compiling_callback() -> None:
            print(colors.GREEN + "TPU backend compilation triggered" + colors.END)
            vars.compiling = True

        def tpumtjgenerate_stopped_compiling_callback() -> None:
            vars.compiling = False
        
        def tpumtjgenerate_settings_callback() -> dict:
            sampler_order = vars.sampler_order[:]
            if len(sampler_order) < 7:  # Add repetition penalty at beginning if it's not present
                sampler_order = [6] + sampler_order
            return {
                "sampler_order": sampler_order,
                "top_p": float(vars.top_p),
                "temp": float(vars.temp),
                "top_k": int(vars.top_k),
                "tfs": float(vars.tfs),
                "typical": float(vars.typical),
                "top_a": float(vars.top_a),
                "repetition_penalty": float(vars.rep_pen),
                "rpslope": float(vars.rep_pen_slope),
                "rprange": int(vars.rep_pen_range),
            }

        # If we're running Colab or OAI, we still need a tokenizer.
        if(vars.model in ("Colab", "API", "CLUSTER")):
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", revision=args.revision, cache_dir="cache")
            loadsettings()
        elif(vars.model == "OAI"):
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=args.revision, cache_dir="cache")
            loadsettings()
        # Load the TPU backend if requested
        elif(vars.use_colab_tpu or vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
            global tpu_mtj_backend
            import tpu_mtj_backend
            if(vars.model == "TPUMeshTransformerGPTNeoX"):
                vars.badwordsids = vars.badwordsids_neox
            print("{0}Initializing Mesh Transformer JAX, please wait...{1}".format(colors.PURPLE, colors.END))
            if vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and (not vars.custmodpth or not os.path.isdir(vars.custmodpth)):
                raise FileNotFoundError(f"The specified model path {repr(vars.custmodpth)} is not the path to a valid folder")
            import tpu_mtj_backend
            if(vars.model == "TPUMeshTransformerGPTNeoX"):
                tpu_mtj_backend.pad_token_id = 2
            tpu_mtj_backend.vars = vars
            tpu_mtj_backend.warper_callback = tpumtjgenerate_warper_callback
            tpu_mtj_backend.stopping_callback = tpumtjgenerate_stopping_callback
            tpu_mtj_backend.compiling_callback = tpumtjgenerate_compiling_callback
            tpu_mtj_backend.stopped_compiling_callback = tpumtjgenerate_stopped_compiling_callback
            tpu_mtj_backend.settings_callback = tpumtjgenerate_settings_callback
            vars.allowsp = True
            loadmodelsettings()
            loadsettings()
            tpu_mtj_backend.load_model(vars.custmodpth, hf_checkpoint=vars.model not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and vars.use_colab_tpu, **vars.modelconfig)
            vars.modeldim = int(tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"]))
            tokenizer = tpu_mtj_backend.tokenizer
            if(vars.badwordsids is vars.badwordsids_default and vars.model_type not in ("gpt2", "gpt_neo", "gptj")):
                vars.badwordsids = [[v] for k, v in tokenizer.get_vocab().items() if any(c in str(k) for c in "<>[]") if vars.newlinemode != "s" or str(k) != "</s>"]
        else:
            loadsettings()
    
    lua_startup()
    # Load scripts
    load_lua_scripts()
    
    final_startup()
    if not initial_load:
        set_aibusy(False)
        emit('from_server', {'cmd': 'hide_model_name'}, broadcast=True)
        time.sleep(0.1)
        
        if not vars.gamestarted:
            setStartState()
            sendsettings()
            refresh_settings()


# Set up Flask routes
@app.route('/')
@app.route('/index')
def index():
    if args.no_ui:
        return redirect('/api/latest')
    else:
        return render_template('index.html', hide_ai_menu=args.noaimenu)
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
        txt = vars.prompt + "".join(vars.actions.values())
        save = Response(txt)
        filename = path.basename(vars.savedir)
        if filename[-5:] == ".json":
            filename = filename[:-5]
        save.headers.set('Content-Disposition', 'attachment', filename='%s.txt' % filename)
        return(save)

    # Build json to write
    js = {}
    js["gamestarted"] = vars.gamestarted
    js["prompt"]      = vars.prompt
    js["memory"]      = vars.memory
    js["authorsnote"] = vars.authornote
    js["anotetemplate"] = vars.authornotetemplate
    js["actions"]     = tuple(vars.actions.values())
    js["actions_metadata"] = vars.actions_metadata
    js["worldinfo"]   = []
        
    # Extract only the important bits of WI
    for wi in vars.worldinfo:
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
    filename = path.basename(vars.savedir)
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
    if(path.exists(get_config_filename())):
        file = open(get_config_filename(), "r")
        js   = json.load(file)
        if("userscripts" in js):
            vars.userscripts = []
            for userscript in js["userscripts"]:
                if type(userscript) is not str:
                    continue
                userscript = userscript.strip()
                if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                    vars.userscripts.append(userscript)
        if("corescript" in js and type(js["corescript"]) is str and all(q not in js["corescript"] for q in ("..", ":")) and all(js["corescript"][0] not in q for q in ("/", "\\"))):
            vars.corescript = js["corescript"]
        else:
            vars.corescript = "default.lua"
        file.close()
        
    #==================================================================#
    #  Lua runtime startup
    #==================================================================#

    print("", end="", flush=True)
    logger.init("LUA bridge", status="Starting")

    # Set up Lua state
    vars.lua_state = lupa.LuaRuntime(unpack_returned_tuples=True)

    # Load bridge.lua
    bridged = {
        "corescript_path": "cores",
        "userscript_path": "userscripts",
        "config_path": "userscripts",
        "lib_paths": vars.lua_state.table("lualibs", os.path.join("extern", "lualibs")),
        "vars": vars,
    }
    for kwarg in _bridged:
        bridged[kwarg] = _bridged[kwarg]
    try:
        vars.lua_kobold, vars.lua_koboldcore, vars.lua_koboldbridge = vars.lua_state.globals().dofile("bridge.lua")(
            vars.lua_state.globals().python,
            bridged,
        )
    except lupa.LuaError as e:
        print(colors.RED + "ERROR!" + colors.END)
        vars.lua_koboldbridge.obliterate_multiverse()
        logger.error('LUA ERROR: ' + str(e).replace("\033", ""))
        logger.warning("Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.")
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

    for filename in vars.userscripts:
        if filename in filenames_dict:
            i = filenames_dict[filename]
            filenames.append(filename)
            modulenames.append(lst[i]["modulename"])
            descriptions.append(lst[i]["description"])

    vars.has_genmod = False

    try:
        vars.lua_koboldbridge.obliterate_multiverse()
        tpool.execute(vars.lua_koboldbridge.load_corescript, vars.corescript)
        vars.has_genmod = tpool.execute(vars.lua_koboldbridge.load_userscripts, filenames, modulenames, descriptions)
        vars.lua_running = True
    except lupa.LuaError as e:
        try:
            vars.lua_koboldbridge.obliterate_multiverse()
        except:
            pass
        vars.lua_running = False
        if(vars.serverstarted):
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True)
            sendUSStatItems()
        logger.error('LUA ERROR: ' + str(e).replace("\033", ""))
        logger.warning("Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.")
        if(vars.serverstarted):
            set_aibusy(0)
    logger.init_ok("LUA Scripts", status="OK")

#==================================================================#
#  Print message that originates from the userscript with the given name
#==================================================================#
@bridged_kwarg()
def lua_print(msg):
    if(vars.lua_logname != vars.lua_koboldbridge.logging_name):
        vars.lua_logname = vars.lua_koboldbridge.logging_name
        print(colors.BLUE + lua_log_format_name(vars.lua_logname) + ":" + colors.END, file=sys.stderr)
    print(colors.PURPLE + msg.replace("\033", "") + colors.END)

#==================================================================#
#  Print warning that originates from the userscript with the given name
#==================================================================#
@bridged_kwarg()
def lua_warn(msg):
    if(vars.lua_logname != vars.lua_koboldbridge.logging_name):
        vars.lua_logname = vars.lua_koboldbridge.logging_name
        print(colors.BLUE + lua_log_format_name(vars.lua_logname) + ":" + colors.END, file=sys.stderr)
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
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=args.revision, cache_dir="cache")
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
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=args.revision, cache_dir="cache")
    return tokenizer.encode(utils.encodenewlines(string), max_length=int(4e9), truncation=True)

#==================================================================#
#  Computes context given a submission, Lua array of entry UIDs and a Lua array
#  of folder UIDs
#==================================================================#
@bridged_kwarg()
def lua_compute_context(submission, entries, folders, kwargs):
    assert type(submission) is str
    if(kwargs is None):
        kwargs = vars.lua_state.table()
    actions = vars._actions if vars.lua_koboldbridge.userstate == "genmod" else vars.actions
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
    if kwargs["include_anote"] is not None and not kwargs["include_anote"]:
        anotetxt = ""
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
    if(uid in vars.worldinfo_u and k in (
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
        return vars.worldinfo_u[uid][k]

#==================================================================#
#  Set property of a world info entry given its UID, property name and new value
#==================================================================#
@bridged_kwarg()
def lua_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in vars.worldinfo_u and k in (
        "key",
        "keysecondary",
        "content",
        "comment",
        "selective",
        "constant",
    )
    if(type(vars.worldinfo_u[uid][k]) is int and type(v) is float):
        v = int(v)
    assert type(vars.worldinfo_u[uid][k]) is type(v)
    vars.worldinfo_u[uid][k] = v
    print(colors.GREEN + f"{lua_log_format_name(vars.lua_koboldbridge.logging_name)} set {k} of world info entry {uid} to {v}" + colors.END)

#==================================================================#
#  Get property of a world info folder given its UID and property name
#==================================================================#
@bridged_kwarg()
def lua_folder_get_attr(uid, k):
    assert type(uid) is int and type(k) is str
    if(uid in vars.wifolders_d and k in (
        "name",
    )):
        return vars.wifolders_d[uid][k]

#==================================================================#
#  Set property of a world info folder given its UID, property name and new value
#==================================================================#
@bridged_kwarg()
def lua_folder_set_attr(uid, k, v):
    assert type(uid) is int and type(k) is str
    assert uid in vars.wifolders_d and k in (
        "name",
    )
    if(type(vars.wifolders_d[uid][k]) is int and type(v) is float):
        v = int(v)
    assert type(vars.wifolders_d[uid][k]) is type(v)
    vars.wifolders_d[uid][k] = v
    print(colors.GREEN + f"{lua_log_format_name(vars.lua_koboldbridge.logging_name)} set {k} of world info folder {uid} to {v}" + colors.END)

#==================================================================#
#  Get the "Amount to Generate"
#==================================================================#
@bridged_kwarg()
def lua_get_genamt():
    return vars.genamt

#==================================================================#
#  Set the "Amount to Generate"
#==================================================================#
@bridged_kwarg()
def lua_set_genamt(genamt):
    assert vars.lua_koboldbridge.userstate != "genmod" and type(genamt) in (int, float) and genamt >= 0
    print(colors.GREEN + f"{lua_log_format_name(vars.lua_koboldbridge.logging_name)} set genamt to {int(genamt)}" + colors.END)
    vars.genamt = int(genamt)

#==================================================================#
#  Get the "Gens Per Action"
#==================================================================#
@bridged_kwarg()
def lua_get_numseqs():
    return vars.numseqs

#==================================================================#
#  Set the "Gens Per Action"
#==================================================================#
@bridged_kwarg()
def lua_set_numseqs(numseqs):
    assert type(numseqs) in (int, float) and numseqs >= 1
    print(colors.GREEN + f"{lua_log_format_name(vars.lua_koboldbridge.logging_name)} set numseqs to {int(numseqs)}" + colors.END)
    vars.numseqs = int(numseqs)

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
    if(setting in ("settemp", "temp")): return vars.temp
    if(setting in ("settopp", "topp", "top_p")): return vars.top_p
    if(setting in ("settopk", "topk", "top_k")): return vars.top_k
    if(setting in ("settfs", "tfs")): return vars.tfs
    if(setting in ("settypical", "typical")): return vars.typical
    if(setting in ("settopa", "topa")): return vars.top_a
    if(setting in ("setreppen", "reppen")): return vars.rep_pen
    if(setting in ("setreppenslope", "reppenslope")): return vars.rep_pen_slope
    if(setting in ("setreppenrange", "reppenrange")): return vars.rep_pen_range
    if(setting in ("settknmax", "tknmax")): return vars.max_length
    if(setting == "anotedepth"): return vars.andepth
    if(setting in ("setwidepth", "widepth")): return vars.widepth
    if(setting in ("setuseprompt", "useprompt")): return vars.useprompt
    if(setting in ("setadventure", "adventure")): return vars.adventure
    if(setting in ("setchatmode", "chatmode")): return vars.chatmode
    if(setting in ("setdynamicscan", "dynamicscan")): return vars.dynamicscan
    if(setting in ("setnopromptgen", "nopromptgen")): return vars.nopromptgen
    if(setting in ("autosave", "autosave")): return vars.autosave
    if(setting in ("setrngpersist", "rngpersist")): return vars.rngpersist
    if(setting in ("frmttriminc", "triminc")): return vars.formatoptns["frmttriminc"]
    if(setting in ("frmtrmblln", "rmblln")): return vars.formatoptns["frmttrmblln"]
    if(setting in ("frmtrmspch", "rmspch")): return vars.formatoptns["frmttrmspch"]
    if(setting in ("frmtadsnsp", "adsnsp")): return vars.formatoptns["frmtadsnsp"]
    if(setting in ("frmtsingleline", "singleline")): return vars.formatoptns["singleline"]
    if(setting == "output_streaming"): return vars.output_streaming
    if(setting == "show_probs"): return vars.show_probs

#==================================================================#
#  Set the setting with the given name if it exists
#==================================================================#
@bridged_kwarg()
def lua_set_setting(setting, v):
    actual_type = type(lua_get_setting(setting))
    assert v is not None and (actual_type is type(v) or (actual_type is int and type(v) is float))
    v = actual_type(v)
    print(colors.GREEN + f"{lua_log_format_name(vars.lua_koboldbridge.logging_name)} set {setting} to {v}" + colors.END)
    if(setting in ("setadventure", "adventure") and v):
        vars.actionmode = 1
    if(setting in ("settemp", "temp")): vars.temp = v
    if(setting in ("settopp", "topp")): vars.top_p = v
    if(setting in ("settopk", "topk")): vars.top_k = v
    if(setting in ("settfs", "tfs")): vars.tfs = v
    if(setting in ("settypical", "typical")): vars.typical = v
    if(setting in ("settopa", "topa")): vars.top_a = v
    if(setting in ("setreppen", "reppen")): vars.rep_pen = v
    if(setting in ("setreppenslope", "reppenslope")): vars.rep_pen_slope = v
    if(setting in ("setreppenrange", "reppenrange")): vars.rep_pen_range = v
    if(setting in ("settknmax", "tknmax")): vars.max_length = v; return True
    if(setting == "anotedepth"): vars.andepth = v; return True
    if(setting in ("setwidepth", "widepth")): vars.widepth = v; return True
    if(setting in ("setuseprompt", "useprompt")): vars.useprompt = v; return True
    if(setting in ("setadventure", "adventure")): vars.adventure = v
    if(setting in ("setdynamicscan", "dynamicscan")): vars.dynamicscan = v
    if(setting in ("setnopromptgen", "nopromptgen")): vars.nopromptgen = v
    if(setting in ("autosave", "noautosave")): vars.autosave = v
    if(setting in ("setrngpersist", "rngpersist")): vars.rngpersist = v
    if(setting in ("setchatmode", "chatmode")): vars.chatmode = v
    if(setting in ("frmttriminc", "triminc")): vars.formatoptns["frmttriminc"] = v
    if(setting in ("frmtrmblln", "rmblln")): vars.formatoptns["frmttrmblln"] = v
    if(setting in ("frmtrmspch", "rmspch")): vars.formatoptns["frmttrmspch"] = v
    if(setting in ("frmtadsnsp", "adsnsp")): vars.formatoptns["frmtadsnsp"] = v
    if(setting in ("frmtsingleline", "singleline")): vars.formatoptns["singleline"] = v
    if(setting == "output_streaming"): vars.output_streaming = v
    if(setting == "show_probs"): vars.show_probs = v

#==================================================================#
#  Get contents of memory
#==================================================================#
@bridged_kwarg()
def lua_get_memory():
    return vars.memory

#==================================================================#
#  Set contents of memory
#==================================================================#
@bridged_kwarg()
def lua_set_memory(m):
    assert type(m) is str
    vars.memory = m

#==================================================================#
#  Get contents of author's note
#==================================================================#
@bridged_kwarg()
def lua_get_authorsnote():
    return vars.authornote

#==================================================================#
#  Set contents of author's note
#==================================================================#
@bridged_kwarg()
def lua_set_authorsnote(m):
    assert type(m) is str
    vars.authornote = m

#==================================================================#
#  Get contents of author's note template
#==================================================================#
@bridged_kwarg()
def lua_get_authorsnotetemplate():
    return vars.authornotetemplate

#==================================================================#
#  Set contents of author's note template
#==================================================================#
@bridged_kwarg()
def lua_set_authorsnotetemplate(m):
    assert type(m) is str
    vars.authornotetemplate = m

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
        print(colors.GREEN + f"{lua_log_format_name(vars.lua_koboldbridge.logging_name)} deleted story chunk {k}" + colors.END)
        chunk = int(k)
        if(vars.lua_koboldbridge.userstate == "genmod"):
            del vars._actions[chunk-1]
        vars.lua_deleted.add(chunk)
        if(not hasattr(vars, "_actions") or vars._actions is not vars.actions):
            #Instead of deleting we'll blank out the text. This way our actions and actions_metadata stay in sync and we can restore the chunk on an undo
            vars.actions[chunk-1] = ""
            vars.actions_metadata[chunk-1]['Alternative Text'] = [{"Text": vars.actions_metadata[chunk-1]['Selected Text'], "Pinned": False, "Editted": True}] + vars.actions_metadata[chunk-1]['Alternative Text']
            vars.actions_metadata[chunk-1]['Selected Text'] = ''
            send_debug()
    else:
        if(k == 0):
            print(colors.GREEN + f"{lua_log_format_name(vars.lua_koboldbridge.logging_name)} edited prompt chunk" + colors.END)
        else:
            print(colors.GREEN + f"{lua_log_format_name(vars.lua_koboldbridge.logging_name)} edited story chunk {k}" + colors.END)
        chunk = int(k)
        if(chunk == 0):
            if(vars.lua_koboldbridge.userstate == "genmod"):
                vars._prompt = v
            vars.lua_edited.add(chunk)
            vars.prompt = v
        else:
            if(vars.lua_koboldbridge.userstate == "genmod"):
                vars._actions[chunk-1] = v
            vars.lua_edited.add(chunk)
            vars.actions[chunk-1] = v
            vars.actions_metadata[chunk-1]['Alternative Text'] = [{"Text": vars.actions_metadata[chunk-1]['Selected Text'], "Pinned": False, "Editted": True}] + vars.actions_metadata[chunk-1]['Alternative Text']
            vars.actions_metadata[chunk-1]['Selected Text'] = v
            send_debug()

#==================================================================#
#  Get model type as "gpt-2-xl", "gpt-neo-2.7B", etc.
#==================================================================#
@bridged_kwarg()
def lua_get_modeltype():
    if(vars.noai):
        return "readonly"
    if(vars.model in ("Colab", "API", "CLUSTER", "OAI", "InferKit")):
        return "api"
    if(not vars.use_colab_tpu and vars.model not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX") and (vars.model in ("GPT2Custom", "NeoCustom") or vars.model_type in ("gpt2", "gpt_neo", "gptj"))):
        hidden_size = get_hidden_size_from_model(model)
    if(vars.model in ("gpt2",) or (vars.model_type == "gpt2" and hidden_size == 768)):
        return "gpt2"
    if(vars.model in ("gpt2-medium",) or (vars.model_type == "gpt2" and hidden_size == 1024)):
        return "gpt2-medium"
    if(vars.model in ("gpt2-large",) or (vars.model_type == "gpt2" and hidden_size == 1280)):
        return "gpt2-large"
    if(vars.model in ("gpt2-xl",) or (vars.model_type == "gpt2" and hidden_size == 1600)):
        return "gpt2-xl"
    if(vars.model_type == "gpt_neo" and hidden_size == 768):
        return "gpt-neo-125M"
    if(vars.model in ("EleutherAI/gpt-neo-1.3B",) or (vars.model_type == "gpt_neo" and hidden_size == 2048)):
        return "gpt-neo-1.3B"
    if(vars.model in ("EleutherAI/gpt-neo-2.7B",) or (vars.model_type == "gpt_neo" and hidden_size == 2560)):
        return "gpt-neo-2.7B"
    if(vars.model in ("EleutherAI/gpt-j-6B",) or ((vars.use_colab_tpu or vars.model == "TPUMeshTransformerGPTJ") and tpu_mtj_backend.params["d_model"] == 4096) or (vars.model_type in ("gpt_neo", "gptj") and hidden_size == 4096)):
        return "gpt-j-6B"
    return "unknown"

#==================================================================#
#  Get model backend as "transformers" or "mtj"
#==================================================================#
@bridged_kwarg()
def lua_get_modelbackend():
    if(vars.noai):
        return "readonly"
    if(vars.model in ("Colab", "API", "CLUSTER", "OAI", "InferKit")):
        return "api"
    if(vars.use_colab_tpu or vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        return "mtj"
    return "transformers"

#==================================================================#
#  Check whether model is loaded from a custom path
#==================================================================#
@bridged_kwarg()
def lua_is_custommodel():
    return vars.model in ("GPT2Custom", "NeoCustom", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")

#==================================================================#
#  Return the filename (as a string) of the current soft prompt, or
#  None if no soft prompt is loaded
#==================================================================#
@bridged_kwarg()
def lua_get_spfilename():
    return vars.spfilename.strip() or None

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
    vars.lua_logname = ...
    vars.lua_edited = set()
    vars.lua_deleted = set()
    try:
        tpool.execute(vars.lua_koboldbridge.execute_inmod)
    except lupa.LuaError as e:
        vars.lua_koboldbridge.obliterate_multiverse()
        vars.lua_running = False
        emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True)
        sendUSStatItems()
        logger.error('LUA ERROR: ' + str(e).replace("\033", ""))
        logger.warning("Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.")
        set_aibusy(0)

def execute_genmod():
    vars.lua_koboldbridge.execute_genmod()

def execute_outmod():
    setgamesaved(False)
    emit('from_server', {'cmd': 'hidemsg', 'data': ''}, broadcast=True)
    try:
        tpool.execute(vars.lua_koboldbridge.execute_outmod)
    except lupa.LuaError as e:
        vars.lua_koboldbridge.obliterate_multiverse()
        vars.lua_running = False
        emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True)
        sendUSStatItems()
        logger.error('LUA ERROR: ' + str(e).replace("\033", ""))
        logger.warning("Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.")
        set_aibusy(0)
    if(vars.lua_koboldbridge.resend_settings_required):
        vars.lua_koboldbridge.resend_settings_required = False
        lua_resend_settings()
    for k in vars.lua_edited:
        inlineedit(k, vars.actions[k])
    for k in vars.lua_deleted:
        inlinedelete(k)




#============================ METHODS =============================#    

#==================================================================#
# Event triggered when browser SocketIO is loaded and connects to server
#==================================================================#
@socketio.on('connect')
def do_connect():
    logger.info("Client connected!")
    emit('from_server', {'cmd': 'setchatname', 'data': vars.chatname})
    emit('from_server', {'cmd': 'setanotetemplate', 'data': vars.authornotetemplate})
    emit('from_server', {'cmd': 'connected', 'smandelete': vars.smandelete, 'smanrename': vars.smanrename, 'modelname': getmodelname()})
    if(vars.host):
        emit('from_server', {'cmd': 'runs_remotely'})
    if(vars.allowsp):
        emit('from_server', {'cmd': 'allowsp', 'data': vars.allowsp})

    sendUSStatItems()
    emit('from_server', {'cmd': 'spstatitems', 'data': {vars.spfilename: vars.spmeta} if vars.allowsp and len(vars.spfilename) else {}}, broadcast=True)

    if(not vars.gamestarted):
        setStartState()
        sendsettings()
        refresh_settings()
        vars.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': vars.laststory})
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': vars.memory})
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote})
        vars.mode = "play"
    else:
        # Game in session, send current game data and ready state to browser
        refresh_story()
        sendsettings()
        refresh_settings()
        emit('from_server', {'cmd': 'setstoryname', 'data': vars.laststory})
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': vars.memory})
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote})
        if(vars.mode == "play"):
            if(not vars.aibusy):
                emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'})
            else:
                emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'})
        elif(vars.mode == "edit"):
            emit('from_server', {'cmd': 'editmode', 'data': 'true'})
        elif(vars.mode == "memory"):
            emit('from_server', {'cmd': 'memmode', 'data': 'true'})
        elif(vars.mode == "wi"):
            emit('from_server', {'cmd': 'wimode', 'data': 'true'})

    emit('from_server', {'cmd': 'gamesaved', 'data': vars.gamesaved}, broadcast=True)

#==================================================================#
# Event triggered when browser SocketIO sends data to the server
#==================================================================#
@socketio.on('message')
def get_message(msg):
    if not vars.quiet:
        logger.debug(f"Data received: {msg}")
    # Submit action
    if(msg['cmd'] == 'submit'):
        if(vars.mode == "play"):
            if(vars.aibusy):
                if(msg.get('allowabort', False)):
                    vars.abort = True
                return
            vars.abort = False
            vars.lua_koboldbridge.feedback = None
            if(vars.chatmode):
                if(type(msg['chatname']) is not str):
                    raise ValueError("Chatname must be a string")
                vars.chatname = msg['chatname']
                settingschanged()
                emit('from_server', {'cmd': 'setchatname', 'data': vars.chatname})
            vars.recentrng = vars.recentrngm = None
            actionsubmit(msg['data'], actionmode=msg['actionmode'])
        elif(vars.mode == "edit"):
            editsubmit(msg['data'])
        elif(vars.mode == "memory"):
            memsubmit(msg['data'])
    # Retry Action
    elif(msg['cmd'] == 'retry'):
        if(vars.aibusy):
            if(msg.get('allowabort', False)):
                vars.abort = True
            return
        vars.abort = False
        if(vars.chatmode):
            if(type(msg['chatname']) is not str):
                raise ValueError("Chatname must be a string")
            vars.chatname = msg['chatname']
            settingschanged()
            emit('from_server', {'cmd': 'setchatname', 'data': vars.chatname})
        actionretry(msg['data'])
    # Back/Undo Action
    elif(msg['cmd'] == 'back'):
        ignore = actionback()
    # Forward/Redo Action
    elif(msg['cmd'] == 'redo'):
        actionredo()
    # EditMode Action (old)
    elif(msg['cmd'] == 'edit'):
        if(vars.mode == "play"):
            vars.mode = "edit"
            emit('from_server', {'cmd': 'editmode', 'data': 'true'}, broadcast=True)
        elif(vars.mode == "edit"):
            vars.mode = "play"
            emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)
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
    elif(not vars.host and msg['cmd'] == 'savetofile'):
        savetofile()
    elif(not vars.host and msg['cmd'] == 'loadfromfile'):
        loadfromfile()
    elif(msg['cmd'] == 'loadfromstring'):
        loadRequest(json.loads(msg['data']), filename=msg['filename'])
    elif(not vars.host and msg['cmd'] == 'import'):
        importRequest()
    elif(msg['cmd'] == 'newgame'):
        newGameRequest()
    elif(msg['cmd'] == 'rndgame'):
        randomGameRequest(msg['data'], memory=msg['memory'])
    elif(msg['cmd'] == 'settemp'):
        vars.temp = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltemp', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopp'):
        vars.top_p = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopp', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopk'):
        vars.top_k = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopk', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settfs'):
        vars.tfs = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltfs', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settypical'):
        vars.typical = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltypical', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settopa'):
        vars.top_a = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopa', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppen'):
        vars.rep_pen = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppen', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppenslope'):
        vars.rep_pen_slope = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppenslope', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setreppenrange'):
        vars.rep_pen_range = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppenrange', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setoutput'):
        vars.genamt = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeloutput', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'settknmax'):
        vars.max_length = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeltknmax', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setikgen'):
        vars.ikgen = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelikgen', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    # Author's Note field update
    elif(msg['cmd'] == 'anote'):
        anotesubmit(msg['data'], template=msg['template'])
    # Author's Note depth update
    elif(msg['cmd'] == 'anotedepth'):
        vars.andepth = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelanotedepth', 'data': msg['data']}, broadcast=True)
        settingschanged()
        refresh_settings()
    # Format - Trim incomplete sentences
    elif(msg['cmd'] == 'frmttriminc'):
        if('frmttriminc' in vars.formatoptns):
            vars.formatoptns["frmttriminc"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtrmblln'):
        if('frmtrmblln' in vars.formatoptns):
            vars.formatoptns["frmtrmblln"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtrmspch'):
        if('frmtrmspch' in vars.formatoptns):
            vars.formatoptns["frmtrmspch"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'frmtadsnsp'):
        if('frmtadsnsp' in vars.formatoptns):
            vars.formatoptns["frmtadsnsp"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'singleline'):
        if('singleline' in vars.formatoptns):
            vars.formatoptns["singleline"] = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'importselect'):
        vars.importnum = int(msg["data"].replace("import", ""))
    elif(msg['cmd'] == 'importcancel'):
        emit('from_server', {'cmd': 'popupshow', 'data': False})
        vars.importjs  = {}
    elif(msg['cmd'] == 'importaccept'):
        emit('from_server', {'cmd': 'popupshow', 'data': False})
        importgame()
    elif(msg['cmd'] == 'wi'):
        togglewimode()
    elif(msg['cmd'] == 'wiinit'):
        if(int(msg['data']) < len(vars.worldinfo)):
            setgamesaved(False)
            vars.worldinfo[msg['data']]["init"] = True
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
        assert 0 <= int(msg['data']) < len(vars.worldinfo)
        setgamesaved(False)
        emit('from_server', {'cmd': 'wiexpand', 'data': msg['data']}, broadcast=True)
    elif(msg['cmd'] == 'wiexpandfolder'):
        assert 0 <= int(msg['data']) < len(vars.worldinfo)
        setgamesaved(False)
        emit('from_server', {'cmd': 'wiexpandfolder', 'data': msg['data']}, broadcast=True)
    elif(msg['cmd'] == 'wifoldercollapsecontent'):
        setgamesaved(False)
        vars.wifolders_d[msg['data']]['collapsed'] = True
        emit('from_server', {'cmd': 'wifoldercollapsecontent', 'data': msg['data']}, broadcast=True)
    elif(msg['cmd'] == 'wifolderexpandcontent'):
        setgamesaved(False)
        vars.wifolders_d[msg['data']]['collapsed'] = False
        emit('from_server', {'cmd': 'wifolderexpandcontent', 'data': msg['data']}, broadcast=True)
    elif(msg['cmd'] == 'wiupdate'):
        setgamesaved(False)
        num = int(msg['num'])
        fields = ("key", "keysecondary", "content", "comment")
        for field in fields:
            if(field in msg['data'] and type(msg['data'][field]) is str):
                vars.worldinfo[num][field] = msg['data'][field]
        emit('from_server', {'cmd': 'wiupdate', 'num': msg['num'], 'data': {field: vars.worldinfo[num][field] for field in fields}}, broadcast=True)
    elif(msg['cmd'] == 'wifolderupdate'):
        setgamesaved(False)
        uid = int(msg['uid'])
        fields = ("name", "collapsed")
        for field in fields:
            if(field in msg['data'] and type(msg['data'][field]) is (str if field != "collapsed" else bool)):
                vars.wifolders_d[uid][field] = msg['data'][field]
        emit('from_server', {'cmd': 'wifolderupdate', 'uid': msg['uid'], 'data': {field: vars.wifolders_d[uid][field] for field in fields}}, broadcast=True)
    elif(msg['cmd'] == 'wiselon'):
        setgamesaved(False)
        vars.worldinfo[msg['data']]["selective"] = True
        emit('from_server', {'cmd': 'wiselon', 'data': msg['data']}, broadcast=True)
    elif(msg['cmd'] == 'wiseloff'):
        setgamesaved(False)
        vars.worldinfo[msg['data']]["selective"] = False
        emit('from_server', {'cmd': 'wiseloff', 'data': msg['data']}, broadcast=True)
    elif(msg['cmd'] == 'wiconstanton'):
        setgamesaved(False)
        vars.worldinfo[msg['data']]["constant"] = True
        emit('from_server', {'cmd': 'wiconstanton', 'data': msg['data']}, broadcast=True)
    elif(msg['cmd'] == 'wiconstantoff'):
        setgamesaved(False)
        vars.worldinfo[msg['data']]["constant"] = False
        emit('from_server', {'cmd': 'wiconstantoff', 'data': msg['data']}, broadcast=True)
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
        emit('from_server', {'cmd': 'buildus', 'data': {"unloaded": unloaded, "loaded": loaded}})
    elif(msg['cmd'] == 'samplerlistrequest'):
        emit('from_server', {'cmd': 'buildsamplers', 'data': vars.sampler_order})
    elif(msg['cmd'] == 'usloaded'):
        vars.userscripts = []
        for userscript in msg['data']:
            if type(userscript) is not str:
                continue
            userscript = userscript.strip()
            if len(userscript) != 0 and all(q not in userscript for q in ("..", ":")) and all(userscript[0] not in q for q in ("/", "\\")) and os.path.exists(fileops.uspath(userscript)):
                vars.userscripts.append(userscript)
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
        vars.sampler_order = sampler_order
        settingschanged()
    elif(msg['cmd'] == 'list_model'):
        sendModelSelection(menu=msg['data'])
    elif(msg['cmd'] == 'load_model'):
        logger.debug(f"Selected Model: {vars.model_selected}")
        if not os.path.exists("settings/"):
            os.mkdir("settings")
        changed = True
        if not utils.HAS_ACCELERATE:
            msg['disk_layers'] = "0"
        if os.path.exists("settings/" + vars.model_selected.replace('/', '_') + ".breakmodel"):
            with open("settings/" + vars.model_selected.replace('/', '_') + ".breakmodel", "r") as file:
                data = file.read().split('\n')[:2]
                if len(data) < 2:
                    data.append("0")
                gpu_layers, disk_layers = data
                if gpu_layers == msg['gpu_layers'] and disk_layers == msg['disk_layers']:
                    changed = False
        if changed:
            if vars.model_selected in ["NeoCustom", "GPT2Custom"]:
                filename = "settings/{}.breakmodel".format(os.path.basename(os.path.normpath(vars.custmodpth)))
            else:
                filename = "settings/{}.breakmodel".format(vars.model_selected.replace('/', '_'))
            f = open(filename, "w")
            f.write(str(msg['gpu_layers']) + '\n' + str(msg['disk_layers']))
            f.close()
        vars.colaburl = msg['url'] + "/request"
        vars.model = vars.model_selected
        if vars.model == "CLUSTER":
            if type(msg['online_model']) is not list:
                if msg['online_model'] == '':
                    vars.cluster_requested_models = []
                else:
                    vars.cluster_requested_models = [msg['online_model']]
            else:
                vars.cluster_requested_models = msg['online_model']
        load_model(use_gpu=msg['use_gpu'], gpu_layers=msg['gpu_layers'], disk_layers=msg['disk_layers'], online_model=msg['online_model'])
    elif(msg['cmd'] == 'show_model'):
        logger.info(f"Model Name: {getmodelname()}")
        emit('from_server', {'cmd': 'show_model_name', 'data': getmodelname()}, broadcast=True)
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
            if 'folder' not in msg or vars.host:
                folder = "./models"
            else:
                folder = msg['folder']
            sendModelSelection(menu=msg['data'], folder=folder)
        elif msg['data'] in ('NeoCustom', 'GPT2Custom') and 'path_modelname' in msg:
            #Here the user entered custom text in the text box. This could be either a model name or a path.
            if check_if_dir_is_model(msg['path_modelname']):
                vars.model_selected = msg['data']
                vars.custmodpth = msg['path_modelname']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                vars.model_selected = msg['path_modelname']
                try:
                    get_model_info(vars.model_selected)
                except:
                    emit('from_server', {'cmd': 'errmsg', 'data': "The model entered doesn't exist."})
        elif msg['data'] in ('NeoCustom', 'GPT2Custom'):
            if check_if_dir_is_model(msg['path']):
                vars.model_selected = msg['data']
                vars.custmodpth = msg['path']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                if vars.host:
                    sendModelSelection(menu=msg['data'], folder="./models")
                else:
                    sendModelSelection(menu=msg['data'], folder=msg['path'])
        else:
            vars.model_selected = msg['data'] 
            if 'path' in msg:
                vars.custmodpth = msg['path']
                get_model_info(msg['data'], directory=msg['path'])
            else:
                get_model_info(vars.model_selected)
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
        get_oai_models(msg['key'])
    elif(msg['cmd'] == 'Cluster_Key_Update'):
        get_cluster_models(msg)
    elif(msg['cmd'] == 'loadselect'):
        vars.loadselect = msg["data"]
    elif(msg['cmd'] == 'spselect'):
        vars.spselect = msg["data"]
    elif(msg['cmd'] == 'loadrequest'):
        loadRequest(fileops.storypath(vars.loadselect))
    elif(msg['cmd'] == 'sprequest'):
        spRequest(vars.spselect)
    elif(msg['cmd'] == 'deletestory'):
        deletesave(msg['data'])
    elif(msg['cmd'] == 'renamestory'):
        renamesave(msg['data'], msg['newname'])
    elif(msg['cmd'] == 'clearoverwrite'):    
        vars.svowname = ""
        vars.saveow   = False
    elif(msg['cmd'] == 'seqsel'):
        selectsequence(msg['data'])
    elif(msg['cmd'] == 'seqpin'):
        pinsequence(msg['data'])
    elif(msg['cmd'] == 'setnumseq'):
        vars.numseqs = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelnumseq', 'data': msg['data']})
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setwidepth'):
        vars.widepth = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelwidepth', 'data': msg['data']})
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setuseprompt'):
        vars.useprompt = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setadventure'):
        vars.adventure = msg['data']
        vars.chatmode = False
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'autosave'):
        vars.autosave = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setchatmode'):
        vars.chatmode = msg['data']
        vars.adventure = False
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setdynamicscan'):
        vars.dynamicscan = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setnopromptgen'):
        vars.nopromptgen = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setrngpersist'):
        vars.rngpersist = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setnogenmod'):
        vars.nogenmod = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setfulldeterminism'):
        vars.full_determinism = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setoutputstreaming'):
        vars.output_streaming = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setshowbudget'):
        vars.show_budget = msg['data']
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setshowprobs'):
        vars.show_probs = msg['data']
        settingschanged()
        refresh_settings()
    elif(not vars.host and msg['cmd'] == 'importwi'):
        wiimportrequest()
    elif(msg['cmd'] == 'debug'):
        vars.debug = msg['data']
        emit('from_server', {'cmd': 'set_debug', 'data': msg['data']}, broadcast=True)
        if vars.debug:
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
        max_tokens = vars.max_length - header_length - vars.sp_length - vars.genamt

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
    loaded = loaded if vars.lua_running else []
    last_userscripts = [e["filename"] for e in loaded]
    emit('from_server', {'cmd': 'usstatitems', 'data': loaded, 'flash': last_userscripts != vars.last_userscripts}, broadcast=True)
    vars.last_userscripts = last_userscripts

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
    if(vars.welcome):
        txt = kml(vars.welcome) + "<br/>"
    else:
        txt = "<span>Welcome to <span class=\"color_cyan\">KoboldAI</span>! You are running <span class=\"color_green\">"+getmodelname()+"</span>.<br/>"
    if(not vars.noai and not vars.welcome):
        txt = txt + "Please load a game or enter a prompt below to begin!</span>"
    if(vars.noai):
        txt = txt + "Please load or import a story to read. There is no AI in this mode."
    emit('from_server', {'cmd': 'updatescreen', 'gamestarted': vars.gamestarted, 'data': txt}, broadcast=True)
    emit('from_server', {'cmd': 'setgamestate', 'data': 'start'}, broadcast=True)

#==================================================================#
#  Transmit applicable settings to SocketIO to build UI sliders/toggles
#==================================================================#
def sendsettings():
    # Send settings for selected AI type
    emit('from_server', {'cmd': 'reset_menus'})
    if(vars.model != "InferKit"):
        for set in gensettings.gensettingstf:
            emit('from_server', {'cmd': 'addsetting', 'data': set})
    else:
        for set in gensettings.gensettingsik:
            emit('from_server', {'cmd': 'addsetting', 'data': set})
    
    # Send formatting options
    for frm in gensettings.formatcontrols:
        emit('from_server', {'cmd': 'addformat', 'data': frm})
        # Add format key to vars if it wasn't loaded with client.settings
        if(not frm["id"] in vars.formatoptns):
            vars.formatoptns[frm["id"]] = False;

#==================================================================#
#  Set value of gamesaved
#==================================================================#
def setgamesaved(gamesaved):
    assert type(gamesaved) is bool
    if(gamesaved != vars.gamesaved):
        emit('from_server', {'cmd': 'gamesaved', 'data': gamesaved}, broadcast=True)
    vars.gamesaved = gamesaved

#==================================================================#
#  Take input text from SocketIO and decide what to do with it
#==================================================================#

def check_for_backend_compilation():
    if(vars.checking):
        return
    vars.checking = True
    for _ in range(31):
        time.sleep(0.06276680299820175)
        if(vars.compiling):
            emit('from_server', {'cmd': 'warnmsg', 'data': 'Compiling TPU backend&mdash;this usually takes 1&ndash;2 minutes...'}, broadcast=True)
            break
    vars.checking = False

def actionsubmit(data, actionmode=0, force_submit=False, force_prompt_gen=False, disable_recentrng=False, no_generate=False, ignore_aibusy=False):
    # Ignore new submissions if the AI is currently busy
    if(not ignore_aibusy and vars.aibusy):
        return
    
    while(True):
        set_aibusy(1)

        if(vars.model in ["API","CLUSTER"]):
            global tokenizer
            if vars.model == "API":
                tokenizer_id = requests.get(
                    vars.colaburl[:-8] + "/api/v1/model",
                ).json()["result"]
            elif len(vars.cluster_requested_models) >= 1:
                # If the player has requested one or more models, we use the first one for the tokenizer
                tokenizer_id = vars.cluster_requested_models[0]
            # The cluster can return any number of possible models for each gen, but this happens after this step
            # So at this point, this is unknown
            else:
                tokenizer_id = ""
            if tokenizer_id != vars.api_tokenizer_id:
                try:
                    if(os.path.isdir(tokenizer_id)):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=args.revision, cache_dir="cache")
                        except:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=args.revision, cache_dir="cache", use_fast=False)
                    elif(os.path.isdir("models/{}".format(tokenizer_id.replace('/', '_')))):
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(tokenizer_id.replace('/', '_')), revision=args.revision, cache_dir="cache")
                        except:
                            tokenizer = AutoTokenizer.from_pretrained("models/{}".format(tokenizer_id.replace('/', '_')), revision=args.revision, cache_dir="cache", use_fast=False)
                    else:
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=args.revision, cache_dir="cache")
                        except:
                            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=args.revision, cache_dir="cache", use_fast=False)
                except:
                    logger.warning(f"Unknown tokenizer {repr(tokenizer_id)}")
                vars.api_tokenizer_id = tokenizer_id

        if(disable_recentrng):
            vars.recentrng = vars.recentrngm = None

        vars.recentback = False
        vars.recentedit = False
        vars.actionmode = actionmode

        # "Action" mode
        if(actionmode == 1):
            data = data.strip().lstrip('>')
            data = re.sub(r'\n+', ' ', data)
            if(len(data)):
                data = f"\n\n> {data}\n"
        
        # "Chat" mode
        if(vars.chatmode and vars.gamestarted):
            data = re.sub(r'\n+', ' ', data)
            if(len(data)):
                data = f"\n{vars.chatname}: {data}\n"
        
        # If we're not continuing, store a copy of the raw input
        if(data != ""):
            vars.lastact = data
        
        if(not vars.gamestarted):
            vars.submission = data
            if(not no_generate):
                execute_inmod()
            vars.submission = re.sub(r"[^\S\r\n]*([\r\n]*)$", r"\1", vars.submission)  # Remove trailing whitespace, excluding newlines
            data = vars.submission
            if(not force_submit and len(data.strip()) == 0):
                assert False
            # Start the game
            vars.gamestarted = True
            if(not no_generate and not vars.noai and vars.lua_koboldbridge.generating and (not vars.nopromptgen or force_prompt_gen)):
                # Save this first action as the prompt
                vars.prompt = data
                # Clear the startup text from game screen
                emit('from_server', {'cmd': 'updatescreen', 'gamestarted': False, 'data': 'Please wait, generating story...'}, broadcast=True)
                calcsubmit(data) # Run the first action through the generator
                if(not no_generate and not vars.abort and vars.lua_koboldbridge.restart_sequence is not None and len(vars.genseqs) == 0):
                    data = ""
                    force_submit = True
                    disable_recentrng = True
                    continue
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
                break
            else:
                # Save this first action as the prompt
                vars.prompt = data if len(data) > 0 else '"'
                for i in range(vars.numseqs):
                    vars.lua_koboldbridge.outputs[i+1] = ""
                if(not no_generate):
                    execute_outmod()
                vars.lua_koboldbridge.regeneration_required = False
                genout = []
                for i in range(vars.numseqs):
                    genout.append({"generated_text": vars.lua_koboldbridge.outputs[i+1]})
                    assert type(genout[-1]["generated_text"]) is str
                if(len(genout) == 1):
                    genresult(genout[0]["generated_text"], flash=False)
                    refresh_story()
                    if(len(vars.actions) > 0):
                        emit('from_server', {'cmd': 'texteffect', 'data': vars.actions.get_last_key() + 1}, broadcast=True)
                    if(not vars.abort and vars.lua_koboldbridge.restart_sequence is not None):
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                else:
                    if(not vars.abort and vars.lua_koboldbridge.restart_sequence is not None and vars.lua_koboldbridge.restart_sequence > 0):
                        genresult(genout[vars.lua_koboldbridge.restart_sequence-1]["generated_text"], flash=False)
                        refresh_story()
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                    genselect(genout)
                    refresh_story()
                set_aibusy(0)
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
                break
        else:
            # Apply input formatting & scripts before sending to tokenizer
            if(vars.actionmode == 0):
                data = applyinputformatting(data)
            vars.submission = data
            if(not no_generate):
                execute_inmod()
            vars.submission = re.sub(r"[^\S\r\n]*([\r\n]*)$", r"\1", vars.submission)  # Remove trailing whitespace, excluding newlines
            data = vars.submission
            # Dont append submission if it's a blank/continue action
            if(data != ""):
                # Store the result in the Action log
                if(len(vars.prompt.strip()) == 0):
                    vars.prompt = data
                else:
                    vars.actions.append(data)
                    # we now need to update the actions_metadata
                    # we'll have two conditions. 
                    # 1. This is totally new (user entered) 
                    if vars.actions.get_last_key() not in vars.actions_metadata:
                        vars.actions_metadata[vars.actions.get_last_key()] = {"Selected Text": data, "Alternative Text": []}
                    else:
                    # 2. We've selected a chunk of text that is was presented previously
                        try:
                            alternatives = [item['Text'] for item in vars.actions_metadata[len(vars.actions)-1]["Alternative Text"]]
                        except:
                            logger.debug(len(vars.actions))
                            logger.debug(vars.actions_metadata)
                            raise
                        if data in alternatives:
                            alternatives = [item for item in vars.actions_metadata[vars.actions.get_last_key() ]["Alternative Text"] if item['Text'] != data]
                            vars.actions_metadata[vars.actions.get_last_key()]["Alternative Text"] = alternatives
                        vars.actions_metadata[vars.actions.get_last_key()]["Selected Text"] = data
                update_story_chunk('last')
                send_debug()

            if(not no_generate and not vars.noai and vars.lua_koboldbridge.generating):
                # Off to the tokenizer!
                calcsubmit(data)
                if(not vars.abort and vars.lua_koboldbridge.restart_sequence is not None and len(vars.genseqs) == 0):
                    data = ""
                    force_submit = True
                    disable_recentrng = True
                    continue
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
                break
            else:
                if(not no_generate):
                    for i in range(vars.numseqs):
                        vars.lua_koboldbridge.outputs[i+1] = ""
                    execute_outmod()
                    vars.lua_koboldbridge.regeneration_required = False
                genout = []
                for i in range(vars.numseqs):
                    genout.append({"generated_text": vars.lua_koboldbridge.outputs[i+1] if not no_generate else ""})
                    assert type(genout[-1]["generated_text"]) is str
                if(len(genout) == 1):
                    genresult(genout[0]["generated_text"])
                    if(not no_generate and not vars.abort and vars.lua_koboldbridge.restart_sequence is not None):
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                else:
                    if(not no_generate and not vars.abort and vars.lua_koboldbridge.restart_sequence is not None and vars.lua_koboldbridge.restart_sequence > 0):
                        genresult(genout[vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                    genselect(genout)
                set_aibusy(0)
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
                break

def apiactionsubmit_generate(txt, minimum, maximum):
    vars.generated_tkns = 0

    if not vars.quiet:
        logger.debug(f"Prompt Min:{minimum}, Max:{maximum}")
        logger.prompt(utils.decodenewlines(tokenizer.decode(txt)).encode("unicode_escape").decode("utf-8"))

    # Clear CUDA cache if using GPU
    if(vars.hascuda and (vars.usegpu or vars.breakmodel)):
        gc.collect()
        torch.cuda.empty_cache()

    # Submit input text to generator
    _genout, already_generated = tpool.execute(_generate, txt, minimum, maximum, set())

    genout = [applyoutputformatting(utils.decodenewlines(tokenizer.decode(tokens[-already_generated:]))) for tokens in _genout]

    # Clear CUDA cache again if using GPU
    if(vars.hascuda and (vars.usegpu or vars.breakmodel)):
        del _genout
        gc.collect()
        torch.cuda.empty_cache()

    return genout

def apiactionsubmit_tpumtjgenerate(txt, minimum, maximum):
    vars.generated_tkns = 0

    if(vars.full_determinism):
        tpu_mtj_backend.set_rng_seed(vars.seed)

    if not vars.quiet:
        logger.debug(f"Prompt Min:{minimum}, Max:{maximum}")
        logger.prompt(utils.decodenewlines(tokenizer.decode(txt)).encode("unicode_escape").decode("utf-8"))

    vars._actions = vars.actions
    vars._prompt = vars.prompt
    if(vars.dynamicscan):
        vars._actions = vars._actions.copy()

    # Submit input text to generator
    soft_tokens = tpumtjgetsofttokens()
    genout = tpool.execute(
        tpu_mtj_backend.infer_static,
        np.uint32(txt),
        gen_len = maximum-minimum+1,
        temp=vars.temp,
        top_p=vars.top_p,
        top_k=vars.top_k,
        tfs=vars.tfs,
        typical=vars.typical,
        top_a=vars.top_a,
        numseqs=vars.numseqs,
        repetition_penalty=vars.rep_pen,
        rpslope=vars.rep_pen_slope,
        rprange=vars.rep_pen_range,
        soft_embeddings=vars.sp,
        soft_tokens=soft_tokens,
        sampler_order=vars.sampler_order,
    )
    genout = [applyoutputformatting(utils.decodenewlines(tokenizer.decode(txt))) for txt in genout]

    return genout

def apiactionsubmit(data, use_memory=False, use_world_info=False, use_story=False, use_authors_note=False):
    if(vars.model == "Colab"):
        raise NotImplementedError("API generation is not supported in old Colab API mode.")
    elif(vars.model == "API"):
        raise NotImplementedError("API generation is not supported in API mode.")
    elif(vars.model == "CLUSTER"):
        raise NotImplementedError("API generation is not supported in API mode.")
    elif(vars.model == "OAI"):
        raise NotImplementedError("API generation is not supported in OpenAI/GooseAI mode.")
    elif(vars.model == "ReadOnly"):
        raise NotImplementedError("API generation is not supported in read-only mode; please load a model and then try again.")

    data = applyinputformatting(data)

    if(vars.memory != "" and vars.memory[-1] != "\n"):
        mem = vars.memory + "\n"
    else:
        mem = vars.memory
    if(use_authors_note and vars.authornote != ""):
        anotetxt  = ("\n" + vars.authornotetemplate + "\n").replace("<|>", vars.authornote)
    else:
        anotetxt = ""
    MIN_STORY_TOKENS = 8
    story_tokens = []
    mem_tokens = []
    wi_tokens = []
    story_budget = lambda: vars.max_length - vars.sp_length - vars.genamt - len(tokenizer._koboldai_header) - len(story_tokens) - len(mem_tokens) - len(wi_tokens)
    budget = lambda: story_budget() + MIN_STORY_TOKENS
    if budget() < 0:
        abort(Response(json.dumps({"detail": {
            "msg": f"Your Max Tokens setting is too low for your current soft prompt and tokenizer to handle. It needs to be at least {vars.max_length - budget()}.",
            "type": "token_overflow",
        }}), mimetype="application/json", status=500))
    if use_memory:
        mem_tokens = tokenizer.encode(utils.encodenewlines(mem))[-budget():]
    if use_world_info:
        world_info, _ = checkworldinfo(data, force_use_txt=True, scan_story=use_story)
        wi_tokens = tokenizer.encode(utils.encodenewlines(world_info))[-budget():]
    if use_story:
        if vars.useprompt:
            story_tokens = tokenizer.encode(utils.encodenewlines(vars.prompt))[-budget():]
    story_tokens = tokenizer.encode(utils.encodenewlines(data))[-story_budget():] + story_tokens
    if use_story:
        for i, action in enumerate(reversed(vars.actions.values())):
            if story_budget() <= 0:
                assert story_budget() == 0
                break
            story_tokens = tokenizer.encode(utils.encodenewlines(action))[-story_budget():] + story_tokens
            if i == vars.andepth - 1:
                story_tokens = tokenizer.encode(utils.encodenewlines(anotetxt))[-story_budget():] + story_tokens
        if not vars.useprompt:
            story_tokens = tokenizer.encode(utils.encodenewlines(vars.prompt))[-budget():] + story_tokens
    tokens = tokenizer._koboldai_header + mem_tokens + wi_tokens + story_tokens
    assert story_budget() >= 0
    minimum = len(tokens) + 1
    maximum = len(tokens) + vars.genamt

    if(not vars.use_colab_tpu and vars.model not in ["Colab", "API", "CLUSTER", "OAI", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
        genout = apiactionsubmit_generate(tokens, minimum, maximum)
    elif(vars.use_colab_tpu or vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        genout = apiactionsubmit_tpumtjgenerate(tokens, minimum, maximum)

    return genout

#==================================================================#
#  
#==================================================================#
def actionretry(data):
    if(vars.noai):
        emit('from_server', {'cmd': 'errmsg', 'data': "Retry function unavailable in Read Only mode."})
        return
    if(vars.recentrng is not None):
        if(not vars.aibusy):
            randomGameRequest(vars.recentrng, memory=vars.recentrngm)
        return
    if actionback():
        actionsubmit("", actionmode=vars.actionmode, force_submit=True)
        send_debug()
    elif(not vars.useprompt):
        emit('from_server', {'cmd': 'errmsg', 'data': "Please enable \"Always Add Prompt\" to retry with your prompt."})

#==================================================================#
#  
#==================================================================#
def actionback():
    if(vars.aibusy):
        return
    # Remove last index of actions and refresh game screen
    if(len(vars.genseqs) == 0 and len(vars.actions) > 0):
        # We are going to move the selected text to alternative text in the actions_metadata variable so we can redo this action
        vars.actions_metadata[vars.actions.get_last_key() ]['Alternative Text'] = [{'Text': vars.actions_metadata[vars.actions.get_last_key() ]['Selected Text'],
                                                                    'Pinned': False,
                                                                    "Previous Selection": True,
                                                                    "Edited": False}] + vars.actions_metadata[vars.actions.get_last_key() ]['Alternative Text']
        vars.actions_metadata[vars.actions.get_last_key() ]['Selected Text'] = ""
    
        last_key = vars.actions.get_last_key()
        vars.actions.pop()
        vars.recentback = True
        remove_story_chunk(last_key + 1)
        #for the redo to not get out of whack, need to reset the max # in the actions sequence
        vars.actions.set_next_id(last_key)
        success = True
    elif(len(vars.genseqs) == 0):
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."})
        success =  False
    else:
        vars.genseqs = []
        success = True
    send_debug()
    return success
        
def actionredo():
    i = 0
    #First we need to find the next valid key
    #We might have deleted text so we don't want to show a redo for that blank chunk
    
    restore_id = vars.actions.get_last_key()+1
    if restore_id in vars.actions_metadata:
        ok_to_use = False
        while not ok_to_use:
            for item in vars.actions_metadata[restore_id]['Alternative Text']:
                if item['Previous Selection'] and item['Text'] != "":
                    ok_to_use = True
            if not ok_to_use:
                restore_id+=1
                if restore_id not in vars.actions_metadata:
                    return
            else:
                vars.actions.set_next_id(restore_id)
                
    
    if restore_id in vars.actions_metadata:
        genout = [{"generated_text": item['Text']} for item in vars.actions_metadata[restore_id]['Alternative Text'] if (item["Previous Selection"]==True)]
        if len(genout) > 0:
            genout = genout + [{"generated_text": item['Text']} for item in vars.actions_metadata[restore_id]['Alternative Text'] if (item["Pinned"]==True) and (item["Previous Selection"]==False)]
            if len(genout) == 1:
                vars.actions_metadata[restore_id]['Alternative Text'] = [item for item in vars.actions_metadata[restore_id]['Alternative Text'] if (item["Previous Selection"]!=True)]
                genresult(genout[0]['generated_text'], flash=True, ignore_formatting=True)
            else:
                # Store sequences in memory until selection is made
                vars.genseqs = genout
                
                
                # Send sequences to UI for selection
                genout = [[item['Text'], "redo"] for item in vars.actions_metadata[restore_id]['Alternative Text'] if (item["Previous Selection"]==True)]
                
                emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True)
    else:
        emit('from_server', {'cmd': 'popuperror', 'data': "There's nothing to undo"}, broadcast=True)
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
    if(vars.memory != "" and vars.memory[-1] != "\n"):
        mem = vars.memory + "\n"
    else:
        mem = vars.memory

    anotetxt = buildauthorsnote(vars.authornote, vars.authornotetemplate)

    return winfo, mem, anotetxt, found_entries

def calcsubmitbudget(actionlen, winfo, mem, anotetxt, actions, submission=None, budget_deduction=0):
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    anotetkns    = []  # Placeholder for Author's Note tokens
    lnanote      = 0   # Placeholder for Author's Note length

    lnsp = vars.sp_length

    if("tokenizer" not in globals()):
        from transformers import GPT2Tokenizer
        global tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision=args.revision, cache_dir="cache")

    lnheader = len(tokenizer._koboldai_header)

    # Calculate token budget
    prompttkns = tokenizer.encode(utils.encodenewlines(vars.comregex_ai.sub('', vars.prompt)), max_length=int(2e9), truncation=True)
    lnprompt   = len(prompttkns)

    memtokens = tokenizer.encode(utils.encodenewlines(mem), max_length=int(2e9), truncation=True)
    lnmem     = len(memtokens)
    if(lnmem > vars.max_length - lnheader - lnsp - vars.genamt - budget_deduction):
        raise OverflowError("The memory in your story is too long. Please either write a shorter memory text or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    witokens  = tokenizer.encode(utils.encodenewlines(winfo), max_length=int(2e9), truncation=True)
    lnwi      = len(witokens)
    if(lnmem + lnwi > vars.max_length - lnheader - lnsp - vars.genamt - budget_deduction):
        raise OverflowError("The current active world info keys take up too many tokens. Please either write shorter world info, decrease World Info Depth or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    if(anotetxt != ""):
        anotetkns = tokenizer.encode(utils.encodenewlines(anotetxt), max_length=int(2e9), truncation=True)
        lnanote   = len(anotetkns)
        if(lnmem + lnwi + lnanote > vars.max_length - lnheader - lnsp - vars.genamt - budget_deduction):
            raise OverflowError("The author's note in your story is too long. Please either write a shorter author's note or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    if(vars.useprompt):
        budget = vars.max_length - lnheader - lnsp - lnprompt - lnmem - lnanote - lnwi - vars.genamt - budget_deduction
    else:
        budget = vars.max_length - lnheader - lnsp - lnmem - lnanote - lnwi - vars.genamt - budget_deduction

    lnsubmission = len(tokenizer.encode(utils.encodenewlines(vars.comregex_ai.sub('', submission)), max_length=int(2e9), truncation=True)) if submission is not None else 0
    maybe_lnprompt = lnprompt if vars.useprompt and actionlen > 0 else 0

    if(lnmem + lnwi + lnanote + maybe_lnprompt + lnsubmission > vars.max_length - lnheader - lnsp - vars.genamt - budget_deduction):
        raise OverflowError("Your submission is too long. Please either write a shorter submission or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt. If you are using the Always Add Prompt setting, turning it off may help.")

    assert budget >= 0

    if(actionlen == 0):
        # First/Prompt action
        tokens = (tokenizer._koboldai_header if vars.model not in ("Colab", "API", "CLUSTER", "OAI") else []) + memtokens + witokens + anotetkns + prompttkns
        assert len(tokens) <= vars.max_length - lnsp - vars.genamt - budget_deduction
        ln = len(tokens) + lnsp
        return tokens, ln+1, ln+vars.genamt
    else:
        tokens     = []
        
        # Check if we have the action depth to hit our A.N. depth
        if(anotetxt != "" and actionlen < vars.andepth):
            forceanote = True
        
        # Get most recent action tokens up to our budget
        n = 0
        for key in reversed(actions):
            chunk = vars.comregex_ai.sub('', actions[key])
            
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
            if(n == vars.andepth-1):
                if(anotetxt != ""):
                    tokens = anotetkns + tokens # A.N. len already taken from bdgt
                    anoteadded = True
            n += 1
        
        # If we're not using the prompt every time and there's still budget left,
        # add some prompt.
        if(not vars.useprompt):
            if(budget > 0):
                prompttkns = prompttkns[-budget:]
            else:
                prompttkns = []

        # Did we get to add the A.N.? If not, do it here
        if(anotetxt != ""):
            if((not anoteadded) or forceanote):
                tokens = (tokenizer._koboldai_header if vars.model not in ("Colab", "API", "CLUSTER", "OAI") else []) + memtokens + witokens + anotetkns + prompttkns + tokens
            else:
                tokens = (tokenizer._koboldai_header if vars.model not in ("Colab", "API", "CLUSTER", "OAI") else []) + memtokens + witokens + prompttkns + tokens
        else:
            # Prepend Memory, WI, and Prompt before action tokens
            tokens = (tokenizer._koboldai_header if vars.model not in ("Colab", "API", "CLUSTER", "OAI") else []) + memtokens + witokens + prompttkns + tokens

        # Send completed bundle to generator
        assert len(tokens) <= vars.max_length - lnsp - vars.genamt - budget_deduction
        ln = len(tokens) + lnsp
        return tokens, ln+1, ln+vars.genamt

#==================================================================#
# Take submitted text and build the text to be given to generator
#==================================================================#
def calcsubmit(txt):
    anotetxt     = ""    # Placeholder for Author's Note text
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    actionlen    = len(vars.actions)

    winfo, mem, anotetxt, found_entries = calcsubmitbudgetheader(txt)
 
    # For all transformers models
    if(vars.model != "InferKit"):
        subtxt, min, max = calcsubmitbudget(actionlen, winfo, mem, anotetxt, vars.actions, submission=txt)
        if(actionlen == 0):
            if(not vars.use_colab_tpu and vars.model not in ["Colab", "API", "CLUSTER", "OAI", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(vars.model == "Colab"):
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.model == "API"):
                sendtoapi(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.model == "CLUSTER"):
                sendtocluster(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.model == "OAI"):
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.use_colab_tpu or vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
                tpumtjgenerate(subtxt, min, max, found_entries=found_entries)
        else:
            if(not vars.use_colab_tpu and vars.model not in ["Colab", "API", "CLUSTER", "OAI", "TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(vars.model == "Colab"):
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.model == "API"):
                sendtoapi(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.model == "CLUSTER"):
                sendtocluster(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.model == "OAI"):
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.use_colab_tpu or vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
                tpumtjgenerate(subtxt, min, max, found_entries=found_entries)
                    
    # For InferKit web API
    else:
        # Check if we have the action depth to hit our A.N. depth
        if(anotetxt != "" and actionlen < vars.andepth):
            forceanote = True
        
        if(vars.useprompt):
            budget = vars.ikmax - len(vars.comregex_ai.sub('', vars.prompt)) - len(anotetxt) - len(mem) - len(winfo) - 1
        else:
            budget = vars.ikmax - len(anotetxt) - len(mem) - len(winfo) - 1
            
        subtxt = ""
        prompt = vars.comregex_ai.sub('', vars.prompt)
        n = 0
        for key in reversed(vars.actions):
            chunk = vars.actions[key]
            
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
            if(not vars.useprompt):
                if(budget > 0):
                    prompt = vars.comregex_ai.sub('', vars.prompt)[-budget:]
                else:
                    prompt = ""
            
            # Inject Author's Note if we've reached the desired depth
            if(n == vars.andepth-1):
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
    if(vars.full_determinism):
        torch.manual_seed(vars.seed)

    gen_in = torch.tensor(txt, dtype=torch.long)[None]
    if(vars.sp is not None):
        soft_tokens = torch.arange(
            model.config.vocab_size,
            model.config.vocab_size + vars.sp.shape[0],
        )
        gen_in = torch.cat((soft_tokens[None], gen_in), dim=-1)
    assert gen_in.shape[-1] + vars.genamt <= vars.max_length

    if(vars.hascuda and vars.usegpu):
        gen_in = gen_in.to(vars.gpu_device)
    elif(vars.hascuda and vars.breakmodel):
        gen_in = gen_in.to(breakmodel.primary_device)
    else:
        gen_in = gen_in.to('cpu')

    model.kai_scanner_excluded_world_info = found_entries

    vars._actions = vars.actions
    vars._prompt = vars.prompt
    if(vars.dynamicscan):
        vars._actions = vars._actions.copy()

    with torch.no_grad():
        already_generated = 0
        numseqs = vars.numseqs
        while True:
            genout = generator(
                gen_in, 
                do_sample=True, 
                max_length=int(2e9),
                repetition_penalty=1.0,
                bad_words_ids=vars.badwordsids,
                use_cache=True,
                num_return_sequences=numseqs
                )
            already_generated += len(genout[0]) - len(gen_in[0])
            assert already_generated <= vars.genamt
            if(model.kai_scanner.halt or not model.kai_scanner.regeneration_required):
                break
            assert genout.ndim >= 2
            assert genout.shape[0] == vars.numseqs
            if(vars.lua_koboldbridge.generated_cols and vars.generated_tkns != vars.lua_koboldbridge.generated_cols):
                raise RuntimeError("Inconsistency detected between KoboldAI Python and Lua backends")
            if(already_generated != vars.generated_tkns):
                raise RuntimeError("WI scanning error")
            for r in range(vars.numseqs):
                for c in range(already_generated):
                    assert vars.lua_koboldbridge.generated[r+1][c+1] is not None
                    genout[r][genout.shape[-1] - already_generated + c] = vars.lua_koboldbridge.generated[r+1][c+1]
            encoded = []
            for i in range(vars.numseqs):
                txt = utils.decodenewlines(tokenizer.decode(genout[i, -already_generated:]))
                winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True, actions=vars._actions)
                found_entries[i].update(_found_entries)
                txt, _, _ = calcsubmitbudget(len(vars._actions), winfo, mem, anotetxt, vars._actions, submission=txt)
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
            if(vars.sp is not None):
                soft_tokens = torch.arange(
                    model.config.vocab_size,
                    model.config.vocab_size + vars.sp.shape[0],
                    device=genout.device,
                )
                genout = torch.cat((soft_tokens.tile(vars.numseqs, 1), genout), dim=-1)
            assert genout.shape[-1] + vars.genamt - already_generated <= vars.max_length
            diff = genout.shape[-1] - gen_in.shape[-1]
            minimum += diff
            maximum += diff
            gen_in = genout
            numseqs = 1
    
    return genout, already_generated
    

def generate(txt, minimum, maximum, found_entries=None):    
    vars.generated_tkns = 0

    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(vars.numseqs))

    if not vars.quiet:
        logger.debug(f"Prompt Min:{minimum}, Max:{maximum}")
        logger.prompt(utils.decodenewlines(tokenizer.decode(txt)).encode("unicode_escape").decode("utf-8"))

    # Store context in memory to use it for comparison with generated content
    vars.lastctx = utils.decodenewlines(tokenizer.decode(txt))

    # Clear CUDA cache if using GPU
    if(vars.hascuda and (vars.usegpu or vars.breakmodel)):
        gc.collect()
        torch.cuda.empty_cache()

    # Submit input text to generator
    try:
        genout, already_generated = tpool.execute(_generate, txt, minimum, maximum, found_entries)
    except Exception as e:
        if(issubclass(type(e), lupa.LuaError)):
            vars.lua_koboldbridge.obliterate_multiverse()
            vars.lua_running = False
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True)
            sendUSStatItems()
            logger.error('LUA ERROR: ' + str(e).replace("\033", ""))
            logger.warning("Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.")
        else:
            emit('from_server', {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'}, broadcast=True)
            logger.error(traceback.format_exc().replace("\033", ""))
        set_aibusy(0)
        return

    for i in range(vars.numseqs):
        vars.lua_koboldbridge.generated[i+1][vars.generated_tkns] = int(genout[i, -1].item())
        vars.lua_koboldbridge.outputs[i+1] = utils.decodenewlines(tokenizer.decode(genout[i, -already_generated:]))

    execute_outmod()
    if(vars.lua_koboldbridge.regeneration_required):
        vars.lua_koboldbridge.regeneration_required = False
        genout = []
        for i in range(vars.numseqs):
            genout.append({"generated_text": vars.lua_koboldbridge.outputs[i+1]})
            assert type(genout[-1]["generated_text"]) is str
    else:
        genout = [{"generated_text": utils.decodenewlines(tokenizer.decode(tokens[-already_generated:]))} for tokens in genout]
    
    if(len(genout) == 1):
        genresult(genout[0]["generated_text"])
    else:
        if(vars.lua_koboldbridge.restart_sequence is not None and vars.lua_koboldbridge.restart_sequence > 0):
            genresult(genout[vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
        else:
            genselect(genout)
    
    # Clear CUDA cache again if using GPU
    if(vars.hascuda and (vars.usegpu or vars.breakmodel)):
        del genout
        gc.collect()
        torch.cuda.empty_cache()
    
    set_aibusy(0)

#==================================================================#
#  Deal with a single return sequence from generate()
#==================================================================#
def genresult(genout, flash=True, ignore_formatting=False):
    if not vars.quiet:
        logger.generation(genout.encode("unicode_escape").decode("utf-8"))
    
    # Format output before continuing
    if not ignore_formatting:
        genout = applyoutputformatting(genout)

    vars.lua_koboldbridge.feedback = genout

    if(len(genout) == 0):
        return
    
    # Add formatted text to Actions array and refresh the game screen
    if(len(vars.prompt.strip()) == 0):
        vars.prompt = genout
    else:
        vars.actions.append(genout)
        if vars.actions.get_last_key() not in vars.actions_metadata:
            vars.actions_metadata[vars.actions.get_last_key()] = {'Selected Text': genout, 'Alternative Text': []}
        else:
            vars.actions_metadata[vars.actions.get_last_key()]['Selected Text'] = genout
    update_story_chunk('last')
    if(flash):
        emit('from_server', {'cmd': 'texteffect', 'data': vars.actions.get_last_key() + 1 if len(vars.actions) else 0}, broadcast=True)
    send_debug()

#==================================================================#
#  Send generator sequences to the UI for selection
#==================================================================#
def genselect(genout):
    i = 0
    for result in genout:
        # Apply output formatting rules to sequences
        result["generated_text"] = applyoutputformatting(result["generated_text"])
        if not vars.quiet:
            logger.info(f"Generation Result {i}")
            logger.generation(result["generated_text"].encode("unicode_escape").decode("utf-8"))
        i += 1
    
    # Add the options to the actions metadata
    # If we've already generated text for this action but haven't selected one we'll want to kill all non-pinned, non-previous selection, and non-edited options then add the new ones
    if vars.actions.get_next_id() in vars.actions_metadata:
        if (vars.actions_metadata[vars.actions.get_next_id()]['Selected Text'] == ""):
            vars.actions_metadata[vars.actions.get_next_id()]['Alternative Text'] = [{"Text": item['Text'], "Pinned": item['Pinned'], 
                                                                             "Previous Selection": item["Previous Selection"], 
                                                                             "Edited": item["Edited"]} for item in vars.actions_metadata[vars.actions.get_next_id()]['Alternative Text'] 
                                                                             if item['Pinned'] or item["Previous Selection"] or item["Edited"]] + [{"Text": text["generated_text"], 
                                                                                    "Pinned": False, "Previous Selection": False, "Edited": False} for text in genout]
        else:
            vars.actions_metadata[vars.actions.get_next_id()] = {'Selected Text': '', 'Alternative Text': [{"Text": text["generated_text"], "Pinned": False, "Previous Selection": False, "Edited": False} for text in genout]}
    else:
        vars.actions_metadata[vars.actions.get_next_id()] = {'Selected Text': '', 'Alternative Text': [{"Text": text["generated_text"], "Pinned": False, "Previous Selection": False, "Edited": False} for text in genout]}
    
    genout = [{"generated_text": item['Text']} for item in vars.actions_metadata[vars.actions.get_next_id()]['Alternative Text'] if (item["Previous Selection"]==False) and (item["Edited"]==False)]

    # Store sequences in memory until selection is made
    vars.genseqs = genout
    
    genout = [[item['Text'], "pinned" if item['Pinned'] else "normal"] for item in vars.actions_metadata[vars.actions.get_next_id()]['Alternative Text']  if (item["Previous Selection"]==False) and (item["Edited"]==False)]

    # Send sequences to UI for selection
    emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True)
    send_debug()

#==================================================================#
#  Send selected sequence to action log and refresh UI
#==================================================================#
def selectsequence(n):
    if(len(vars.genseqs) == 0):
        return
    vars.lua_koboldbridge.feedback = vars.genseqs[int(n)]["generated_text"]
    if(len(vars.lua_koboldbridge.feedback) != 0):
        vars.actions.append(vars.lua_koboldbridge.feedback)
        #We'll want to remove the option from the alternative text and put it in selected text
        vars.actions_metadata[vars.actions.get_last_key() ]['Alternative Text'] = [item for item in vars.actions_metadata[vars.actions.get_last_key()]['Alternative Text'] if item['Text'] != vars.lua_koboldbridge.feedback]
        vars.actions_metadata[vars.actions.get_last_key() ]['Selected Text'] = vars.lua_koboldbridge.feedback
        update_story_chunk('last')
        emit('from_server', {'cmd': 'texteffect', 'data': vars.actions.get_last_key() + 1 if len(vars.actions) else 0}, broadcast=True)
    emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True)
    vars.genseqs = []

    if(vars.lua_koboldbridge.restart_sequence is not None):
        actionsubmit("", actionmode=vars.actionmode, force_submit=True, disable_recentrng=True)
    send_debug()

#==================================================================#
#  Pin/Unpin the selected sequence
#==================================================================#
def pinsequence(n):
    if n.isnumeric():
        text = vars.genseqs[int(n)]['generated_text']
        if text in [item['Text'] for item in vars.actions_metadata[vars.actions.get_next_id()]['Alternative Text']]:
            alternatives = vars.actions_metadata[vars.actions.get_next_id()]['Alternative Text']
            for i in range(len(alternatives)):
                if alternatives[i]['Text'] == text:
                    alternatives[i]['Pinned'] = not alternatives[i]['Pinned']
                    break
            vars.actions_metadata[vars.actions.get_next_id()]['Alternative Text'] = alternatives
    send_debug()


#==================================================================#
#  Send transformers-style request to ngrok/colab host
#==================================================================#
def sendtocolab(txt, min, max):
    # Log request to console
    if not vars.quiet:
        print("{0}Tokens:{1}, Txt:{2}{3}".format(colors.YELLOW, min-1, txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    vars.lastctx = txt
    
    # Build request JSON data
    reqdata = {
        'text': txt,
        'min': min,
        'max': max,
        'rep_pen': vars.rep_pen,
        'rep_pen_slope': vars.rep_pen_slope,
        'rep_pen_range': vars.rep_pen_range,
        'temperature': vars.temp,
        'top_p': vars.top_p,
        'top_k': vars.top_k,
        'tfs': vars.tfs,
        'typical': vars.typical,
        'topa': vars.top_a,
        'numseqs': vars.numseqs,
        'retfultxt': False
    }
    
    # Create request
    req = requests.post(
        vars.colaburl, 
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
        
        for i in range(vars.numseqs):
            vars.lua_koboldbridge.outputs[i+1] = genout[i]

        execute_outmod()
        if(vars.lua_koboldbridge.regeneration_required):
            vars.lua_koboldbridge.regeneration_required = False
            genout = []
            for i in range(vars.numseqs):
                genout.append(vars.lua_koboldbridge.outputs[i+1])
                assert type(genout[-1]) is str

        if(len(genout) == 1):
            genresult(genout[0])
        else:
            # Convert torch output format to transformers
            seqs = []
            for seq in genout:
                seqs.append({"generated_text": seq})
            if(vars.lua_koboldbridge.restart_sequence is not None and vars.lua_koboldbridge.restart_sequence > 0):
                genresult(genout[vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
            else:
                genselect(genout)
        
        # Format output before continuing
        #genout = applyoutputformatting(getnewcontent(genout))
        
        # Add formatted text to Actions array and refresh the game screen
        #vars.actions.append(genout)
        #refresh_story()
        #emit('from_server', {'cmd': 'texteffect', 'data': vars.actions.get_last_key() + 1 if len(vars.actions) else 0})
        
        set_aibusy(0)
    else:
        errmsg = "Colab API Error: Failed to get a reply from the server. Please check the colab console."
        print("{0}{1}{2}".format(colors.RED, errmsg, colors.END))
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
        set_aibusy(0)


#==================================================================#
#  Send transformers-style request to KoboldAI API
#==================================================================#
def sendtoapi(txt, min, max):
    # Log request to console
    if not vars.quiet:
        print("{0}Tokens:{1}, Txt:{2}{3}".format(colors.YELLOW, min-1, txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    vars.lastctx = txt
    
    # Build request JSON data
    reqdata = {
        'prompt': txt,
        'max_length': max - min + 1,
        'max_context_length': vars.max_length,
        'rep_pen': vars.rep_pen,
        'rep_pen_slope': vars.rep_pen_slope,
        'rep_pen_range': vars.rep_pen_range,
        'temperature': vars.temp,
        'top_p': vars.top_p,
        'top_k': vars.top_k,
        'top_a': vars.top_a,
        'tfs': vars.tfs,
        'typical': vars.typical,
        'n': vars.numseqs,
    }
    
    # Create request
    while True:
        req = requests.post(
            vars.colaburl[:-8] + "/api/v1/generate",
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

        for i in range(vars.numseqs):
            vars.lua_koboldbridge.outputs[i+1] = genout[i]

        execute_outmod()
        if(vars.lua_koboldbridge.regeneration_required):
            vars.lua_koboldbridge.regeneration_required = False
            genout = []
            for i in range(vars.numseqs):
                genout.append(vars.lua_koboldbridge.outputs[i+1])
                assert type(genout[-1]) is str

        if(len(genout) == 1):
            genresult(genout[0])
        else:
            adjusted_genout = []
            for item in genout:
                adjusted_genout.append({"generated_text": item})
            # Convert torch output format to transformers
            seqs = []
            for seq in adjusted_genout:
                seqs.append({"generated_text": seq})
            if(vars.lua_koboldbridge.restart_sequence is not None and vars.lua_koboldbridge.restart_sequence > 0):
                genresult(adjusted_genout[vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
            else:
                genselect(adjusted_genout)

        set_aibusy(0)
        return

#==================================================================#
#  Send transformers-style request to KoboldAI Cluster
#==================================================================#
def sendtocluster(txt, min, max):
    # Log request to console
    if not vars.quiet:
        logger.debug(f"Tokens Min:{min-1}")
        logger.prompt(txt.encode("unicode_escape").decode("utf-8"))

    # Store context in memory to use it for comparison with generated content
    vars.lastctx = txt
    # Build request JSON data
    reqdata = {
        'max_length': max - min + 1,
        'max_context_length': vars.max_length,
        'rep_pen': vars.rep_pen,
        'rep_pen_slope': vars.rep_pen_slope,
        'rep_pen_range': vars.rep_pen_range,
        'temperature': vars.temp,
        'top_p': vars.top_p,
        'top_k': vars.top_k,
        'top_a': vars.top_a,
        'tfs': vars.tfs,
        'typical': vars.typical,
        'n': vars.numseqs,
    }
    cluster_metadata = {
        'prompt': txt,
        'params': reqdata,
        'models': vars.cluster_requested_models,
        'trusted_workers': False,
    }    
    client_agent = "KoboldAI:1.19.3:koboldai.org"
    cluster_headers = {
        'apikey': vars.apikey,
        "Client-Agent": client_agent
    }    
    logger.debug(f"Horde Payload: {cluster_metadata}")
    try:
        # Create request
        req = requests.post(
            vars.colaburl[:-8] + "/api/v2/generate/text/async",
            json=cluster_metadata,
            headers=cluster_headers,
        )
    except requests.exceptions.ConnectionError:
        errmsg = f"Horde unavailable. Please try again later"
        logger.error(errmsg)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
        set_aibusy(0)
        return
    if(req.status_code == 503):
        errmsg = f"KoboldAI API Error: No available KoboldAI servers found in Horde to fulfil this request using the selected models or other properties."
        logger.error(req.text)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
        set_aibusy(0)
        return
    if(not req.ok):
        errmsg = f"KoboldAI API Error: Failed to get a standard reply from the Horde. Please check the console."
        logger.error(req.text)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
        set_aibusy(0)
        return
    try:
        js = req.json()
    except requests.exceptions.JSONDecodeError:
        errmsg = f"Unexpected message received from the Horde: '{req.text}'"
        logger.error(errmsg)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
        set_aibusy(0)
        return

    request_id = js["id"]
    logger.debug("Horde Request ID: {}".format(request_id))
    
    cluster_agent_headers = {
        "Client-Agent": client_agent
    }            
    finished = False

    while not finished:
        try: 
            req = requests.get(vars.colaburl[:-8] + "/api/v2/generate/text/status/" + request_id, headers=cluster_agent_headers)
        except requests.exceptions.ConnectionError:
            errmsg = f"Horde unavailable. Please try again later"
            logger.error(errmsg)
            emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
            set_aibusy(0)
            return

        if not req.ok:
            errmsg = f"KoboldAI API Error: Failed to get a standard reply from the Horde. Please check the console."
            logger.error(req.text)
            emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
            set_aibusy(0)
            return

        try:
            req_status = req.json()
        except requests.exceptions.JSONDecodeError:
            errmsg = f"Unexpected message received from the KoboldAI Horde: '{req.text}'"
            logger.error(errmsg)
            emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
            set_aibusy(0)
            return

        if "done" not in req_status:
            errmsg = f"Unexpected response received from the KoboldAI Horde: '{js}'"
            logger.error(errmsg)
            emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
            set_aibusy(0)
            return

        finished = req_status["done"]

        if not finished:
            logger.debug(req_status)
            time.sleep(1)
    
    logger.debug("Last Horde Status Message: {}".format(js))
    if req_status["faulted"]:
        errmsg = "Horde Text generation faulted! Please try again"
        logger.error(errmsg)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
        set_aibusy(0)
        return
    
    generations = req_status['generations']
    gen_workers = [(cgen['worker_name'],cgen['worker_id']) for cgen in generations]
    logger.info(f"Generations by: {gen_workers}")






    # Just in case we want to announce it to the user
    if len(generations) == 1:        
        warnmsg = f"Text generated by {[w[0] for w in gen_workers]}"
        emit('from_server', {'cmd': 'warnmsg', 'data': warnmsg}, broadcast=True)
    genout = [cgen['text'] for cgen in generations]

    for i in range(vars.numseqs):
        vars.lua_koboldbridge.outputs[i+1] = genout[i]

    execute_outmod()
    if(vars.lua_koboldbridge.regeneration_required):
        vars.lua_koboldbridge.regeneration_required = False
        genout = []
        for i in range(vars.numseqs):
            genout.append(vars.lua_koboldbridge.outputs[i+1])
            assert type(genout[-1]) is str

    if(len(genout) == 1):
        genresult(genout[0])
    else:
        adjusted_genout = []
        for item in genout:
            adjusted_genout.append({"generated_text": item})
        # Convert torch output format to transformers
        seqs = []
        for seq in adjusted_genout:
            seqs.append({"generated_text": seq})
        if(vars.lua_koboldbridge.restart_sequence is not None and vars.lua_koboldbridge.restart_sequence > 0):
            genresult(adjusted_genout[vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
        else:
            genselect(adjusted_genout)

    set_aibusy(0)
    return

#==================================================================#
#  Send text to TPU mesh transformer backend
#==================================================================#
def tpumtjgenerate(txt, minimum, maximum, found_entries=None):
    if(vars.full_determinism):
        tpu_mtj_backend.set_rng_seed(vars.seed)

    vars.generated_tkns = 0

    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(vars.numseqs))

    if not vars.quiet:
        logger.debug(f"Prompt Min:{minimum}, Max:{maximum}")
        logger.prompt(utils.decodenewlines(tokenizer.decode(txt)).encode("unicode_escape").decode("utf-8"))

    vars._actions = vars.actions
    vars._prompt = vars.prompt
    if(vars.dynamicscan):
        vars._actions = vars._actions.copy()

    # Submit input text to generator
    try:
        soft_tokens = tpumtjgetsofttokens()

        global past

        socketio.start_background_task(copy_current_request_context(check_for_backend_compilation))

        if(vars.dynamicscan or (not vars.nogenmod and vars.has_genmod)):

            context = np.tile(np.uint32(txt), (vars.numseqs, 1))
            past = np.empty((vars.numseqs, 0), dtype=np.uint32)

            while(True):
                genout, n_generated, regeneration_required, halt = tpool.execute(
                    tpu_mtj_backend.infer_dynamic,
                    context,
                    gen_len = maximum-minimum+1,
                    numseqs=vars.numseqs,
                    soft_embeddings=vars.sp,
                    soft_tokens=soft_tokens,
                    excluded_world_info=found_entries,
                )

                past = np.pad(past, ((0, 0), (0, n_generated)))
                for r in range(vars.numseqs):
                    for c in range(vars.lua_koboldbridge.generated_cols):
                        assert vars.lua_koboldbridge.generated[r+1][c+1] is not None
                        past[r, c] = vars.lua_koboldbridge.generated[r+1][c+1]

                if(vars.abort or halt or not regeneration_required):
                    break
                print("(regeneration triggered)")

                encoded = []
                for i in range(vars.numseqs):
                    txt = utils.decodenewlines(tokenizer.decode(past[i]))
                    winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True, actions=vars._actions)
                    found_entries[i].update(_found_entries)
                    txt, _, _ = calcsubmitbudget(len(vars._actions), winfo, mem, anotetxt, vars._actions, submission=txt)
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
                temp=vars.temp,
                top_p=vars.top_p,
                top_k=vars.top_k,
                tfs=vars.tfs,
                typical=vars.typical,
                top_a=vars.top_a,
                numseqs=vars.numseqs,
                repetition_penalty=vars.rep_pen,
                rpslope=vars.rep_pen_slope,
                rprange=vars.rep_pen_range,
                soft_embeddings=vars.sp,
                soft_tokens=soft_tokens,
                sampler_order=vars.sampler_order,
            )
            past = genout
            for i in range(vars.numseqs):
                vars.lua_koboldbridge.generated[i+1] = vars.lua_state.table(*genout[i].tolist())
            vars.lua_koboldbridge.generated_cols = vars.generated_tkns = genout[0].shape[-1]

    except Exception as e:
        if(issubclass(type(e), lupa.LuaError)):
            vars.lua_koboldbridge.obliterate_multiverse()
            vars.lua_running = False
            emit('from_server', {'cmd': 'errmsg', 'data': 'Lua script error; please check console.'}, broadcast=True)
            sendUSStatItems()
            logger.error('LUA ERROR: ' + str(e).replace("\033", ""))
            logger.warning("Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.")
        else:
            emit('from_server', {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'}, broadcast=True)
            print("{0}{1}{2}".format(colors.RED, traceback.format_exc().replace("\033", ""), colors.END), file=sys.stderr)
        set_aibusy(0)
        return

    for i in range(vars.numseqs):
        vars.lua_koboldbridge.outputs[i+1] = utils.decodenewlines(tokenizer.decode(past[i]))
    genout = past

    execute_outmod()
    if(vars.lua_koboldbridge.regeneration_required):
        vars.lua_koboldbridge.regeneration_required = False
        genout = []
        for i in range(vars.numseqs):
            genout.append({"generated_text": vars.lua_koboldbridge.outputs[i+1]})
            assert type(genout[-1]["generated_text"]) is str
    else:
        genout = [{"generated_text": utils.decodenewlines(tokenizer.decode(txt))} for txt in genout]

    if(len(genout) == 1):
        genresult(genout[0]["generated_text"])
    else:
        if(vars.lua_koboldbridge.restart_sequence is not None and vars.lua_koboldbridge.restart_sequence > 0):
            genresult(genout[vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
        else:
            genselect(genout)

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
    if(vars.lastctx == ""):
        return txt
    
    # Tokenize the last context and the generated content
    ctxtokens = tokenizer.encode(utils.encodenewlines(vars.lastctx), max_length=int(2e9), truncation=True)
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
    if(vars.formatoptns["frmtadsnsp"]):
        txt = utils.addsentencespacing(txt, vars)
 
    return txt

#==================================================================#
# Applies chosen formatting options to text returned from AI
#==================================================================#
def applyoutputformatting(txt):
    # Use standard quotes and apostrophes
    txt = utils.fixquotes(txt)

    # Adventure mode clipping of all characters after '>'
    if(vars.adventure):
        txt = vars.acregex_ai.sub('', txt)
    
    # Trim incomplete sentences
    if(vars.formatoptns["frmttriminc"] and not vars.chatmode):
        txt = utils.trimincompletesentence(txt)
    # Replace blank lines
    if(vars.formatoptns["frmtrmblln"] or vars.chatmode):
        txt = utils.replaceblanklines(txt)
    # Remove special characters
    if(vars.formatoptns["frmtrmspch"]):
        txt = utils.removespecialchars(txt, vars)
	# Single Line Mode
    if(vars.formatoptns["singleline"] or vars.chatmode):
        txt = utils.singlelineprocessing(txt, vars)
    
    return txt

#==================================================================#
# Sends the current story content to the Game Screen
#==================================================================#
def refresh_story():
    text_parts = ['<chunk n="0" id="n0" tabindex="-1">', vars.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), html.escape(vars.prompt)), '</chunk>']
    for idx in vars.actions:
        item = vars.actions[idx]
        idx += 1
        item = html.escape(item)
        item = vars.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), item)  # Add special formatting to comments
        item = vars.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions
        text_parts.extend(('<chunk n="', str(idx), '" id="n', str(idx), '" tabindex="-1">', item, '</chunk>'))
    emit('from_server', {'cmd': 'updatescreen', 'gamestarted': vars.gamestarted, 'data': formatforhtml(''.join(text_parts))}, broadcast=True)


#==================================================================#
# Signals the Game Screen to update one of the chunks
#==================================================================#
def update_story_chunk(idx: Union[int, str]):
    if idx == 'last':
        if len(vars.actions) <= 1:
            # In this case, we are better off just refreshing the whole thing as the
            # prompt might not have been shown yet (with a "Generating story..."
            # message instead).
            refresh_story()
            setgamesaved(False)
            return

        idx = (vars.actions.get_last_key() if len(vars.actions) else 0) + 1

    if idx == 0:
        text = vars.prompt
    else:
        # Actions are 0 based, but in chunks 0 is the prompt.
        # So the chunk index is one more than the corresponding action index.
        if(idx - 1 not in vars.actions):
            return
        text = vars.actions[idx - 1]

    item = html.escape(text)
    item = vars.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), item)  # Add special formatting to comments
    item = vars.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions

    chunk_text = f'<chunk n="{idx}" id="n{idx}" tabindex="-1">{formatforhtml(item)}</chunk>'
    emit('from_server', {'cmd': 'updatechunk', 'data': {'index': idx, 'html': chunk_text}}, broadcast=True)

    setgamesaved(False)

    #If we've set the auto save flag, we'll now save the file
    if vars.autosave and (".json" in vars.savedir):
        save()


#==================================================================#
# Signals the Game Screen to remove one of the chunks
#==================================================================#
def remove_story_chunk(idx: int):
    emit('from_server', {'cmd': 'removechunk', 'data': idx}, broadcast=True)
    setgamesaved(False)


#==================================================================#
# Sends the current generator settings to the Game Menu
#==================================================================#
def refresh_settings():
    # Suppress toggle change events while loading state
    emit('from_server', {'cmd': 'allowtoggle', 'data': False}, broadcast=True)
    
    if(vars.model != "InferKit"):
        emit('from_server', {'cmd': 'updatetemp', 'data': vars.temp}, broadcast=True)
        emit('from_server', {'cmd': 'updatetopp', 'data': vars.top_p}, broadcast=True)
        emit('from_server', {'cmd': 'updatetopk', 'data': vars.top_k}, broadcast=True)
        emit('from_server', {'cmd': 'updatetfs', 'data': vars.tfs}, broadcast=True)
        emit('from_server', {'cmd': 'updatetypical', 'data': vars.typical}, broadcast=True)
        emit('from_server', {'cmd': 'updatetopa', 'data': vars.top_a}, broadcast=True)
        emit('from_server', {'cmd': 'updatereppen', 'data': vars.rep_pen}, broadcast=True)
        emit('from_server', {'cmd': 'updatereppenslope', 'data': vars.rep_pen_slope}, broadcast=True)
        emit('from_server', {'cmd': 'updatereppenrange', 'data': vars.rep_pen_range}, broadcast=True)
        emit('from_server', {'cmd': 'updateoutlen', 'data': vars.genamt}, broadcast=True)
        emit('from_server', {'cmd': 'updatetknmax', 'data': vars.max_length}, broadcast=True)
        emit('from_server', {'cmd': 'updatenumseq', 'data': vars.numseqs}, broadcast=True)
    else:
        emit('from_server', {'cmd': 'updatetemp', 'data': vars.temp}, broadcast=True)
        emit('from_server', {'cmd': 'updatetopp', 'data': vars.top_p}, broadcast=True)
        emit('from_server', {'cmd': 'updateikgen', 'data': vars.ikgen}, broadcast=True)
    
    emit('from_server', {'cmd': 'updateanotedepth', 'data': vars.andepth}, broadcast=True)
    emit('from_server', {'cmd': 'updatewidepth', 'data': vars.widepth}, broadcast=True)
    emit('from_server', {'cmd': 'updateuseprompt', 'data': vars.useprompt}, broadcast=True)
    emit('from_server', {'cmd': 'updateadventure', 'data': vars.adventure}, broadcast=True)
    emit('from_server', {'cmd': 'updatechatmode', 'data': vars.chatmode}, broadcast=True)
    emit('from_server', {'cmd': 'updatedynamicscan', 'data': vars.dynamicscan}, broadcast=True)
    emit('from_server', {'cmd': 'updateautosave', 'data': vars.autosave}, broadcast=True)
    emit('from_server', {'cmd': 'updatenopromptgen', 'data': vars.nopromptgen}, broadcast=True)
    emit('from_server', {'cmd': 'updaterngpersist', 'data': vars.rngpersist}, broadcast=True)
    emit('from_server', {'cmd': 'updatenogenmod', 'data': vars.nogenmod}, broadcast=True)
    emit('from_server', {'cmd': 'updatefulldeterminism', 'data': vars.full_determinism}, broadcast=True)
    
    emit('from_server', {'cmd': 'updatefrmttriminc', 'data': vars.formatoptns["frmttriminc"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtrmblln', 'data': vars.formatoptns["frmtrmblln"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtrmspch', 'data': vars.formatoptns["frmtrmspch"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtadsnsp', 'data': vars.formatoptns["frmtadsnsp"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatesingleline', 'data': vars.formatoptns["singleline"]}, broadcast=True)
    emit('from_server', {'cmd': 'updateoutputstreaming', 'data': vars.output_streaming}, broadcast=True)
    emit('from_server', {'cmd': 'updateshowbudget', 'data': vars.show_budget}, broadcast=True)
    emit('from_server', {'cmd': 'updateshowprobs', 'data': vars.show_probs}, broadcast=True)
    
    # Allow toggle events again
    emit('from_server', {'cmd': 'allowtoggle', 'data': True}, broadcast=True)

#==================================================================#
#  Sets the logical and display states for the AI Busy condition
#==================================================================#
def set_aibusy(state):
    if(vars.disable_set_aibusy):
        return
    if(state):
        vars.aibusy = True
        emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'}, broadcast=True)
    else:
        vars.aibusy = False
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True)

#==================================================================#
# 
#==================================================================#
def editrequest(n):
    if(n == 0):
        txt = vars.prompt
    else:
        txt = vars.actions[n-1]
    
    vars.editln = n
    emit('from_server', {'cmd': 'setinputtext', 'data': txt}, broadcast=True)
    emit('from_server', {'cmd': 'enablesubmit', 'data': ''}, broadcast=True)

#==================================================================#
# 
#==================================================================#
def editsubmit(data):
    vars.recentedit = True
    if(vars.editln == 0):
        vars.prompt = data
    else:
        vars.actions_metadata[vars.editln-1]['Alternative Text'] = vars.actions_metadata[vars.editln-1]['Alternative Text'] + [{"Text": vars.actions[vars.editln-1], "Pinned": False, 
                                                                         "Previous Selection": False, 
                                                                         "Edited": True}]
        vars.actions_metadata[vars.editln-1]['Selected Text'] = data
        vars.actions[vars.editln-1] = data
    
    vars.mode = "play"
    update_story_chunk(vars.editln)
    emit('from_server', {'cmd': 'texteffect', 'data': vars.editln}, broadcast=True)
    emit('from_server', {'cmd': 'editmode', 'data': 'false'})
    send_debug()

#==================================================================#
#  
#==================================================================#
def deleterequest():
    vars.recentedit = True
    # Don't delete prompt
    if(vars.editln == 0):
        # Send error message
        pass
    else:
        vars.actions_metadata[vars.editln-1]['Alternative Text'] = [{"Text": vars.actions[vars.editln-1], "Pinned": False, 
                                                      "Previous Selection": True, "Edited": False}] + vars.actions_metadata[vars.editln-1]['Alternative Text']
        vars.actions_metadata[vars.editln-1]['Selected Text'] = ''
        vars.actions[vars.editln-1] = ''
        vars.mode = "play"
        remove_story_chunk(vars.editln)
        emit('from_server', {'cmd': 'editmode', 'data': 'false'})
    send_debug()

#==================================================================#
# 
#==================================================================#
def inlineedit(chunk, data):
    vars.recentedit = True
    chunk = int(chunk)
    if(chunk == 0):
        if(len(data.strip()) == 0):
            return
        vars.prompt = data
    else:
        if(chunk-1 in vars.actions):
            vars.actions_metadata[chunk-1]['Alternative Text'] = vars.actions_metadata[chunk-1]['Alternative Text'] + [{"Text": vars.actions[chunk-1], "Pinned": False, 
                                                                             "Previous Selection": False, 
                                                                             "Edited": True}]
            vars.actions_metadata[chunk-1]['Selected Text'] = data
            vars.actions[chunk-1] = data
        else:
            logger.warning(f"Attempted to edit non-existent chunk {chunk}")

    setgamesaved(False)
    update_story_chunk(chunk)
    emit('from_server', {'cmd': 'texteffect', 'data': chunk}, broadcast=True)
    emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)
    send_debug()

#==================================================================#
#  
#==================================================================#
def inlinedelete(chunk):
    vars.recentedit = True
    chunk = int(chunk)
    # Don't delete prompt
    if(chunk == 0):
        # Send error message
        update_story_chunk(chunk)
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."})
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)
    else:
        if(chunk-1 in vars.actions):
            vars.actions_metadata[chunk-1]['Alternative Text'] = [{"Text": vars.actions[chunk-1], "Pinned": False, 
                                                                             "Previous Selection": True, 
                                                                             "Edited": False}] + vars.actions_metadata[chunk-1]['Alternative Text']
            vars.actions_metadata[chunk-1]['Selected Text'] = ''
            del vars.actions[chunk-1]
        else:
            logger.warning(f"Attempted to delete non-existent chunk {chunk}")
        setgamesaved(False)
        remove_story_chunk(chunk)
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)
    send_debug()

#==================================================================#
#   Toggles the game mode for memory editing and sends UI commands
#==================================================================#
def togglememorymode():
    if(vars.mode == "play"):
        vars.mode = "memory"
        emit('from_server', {'cmd': 'memmode', 'data': 'true'}, broadcast=True)
        emit('from_server', {'cmd': 'setinputtext', 'data': vars.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote}, broadcast=True)
        emit('from_server', {'cmd': 'setanotetemplate', 'data': vars.authornotetemplate}, broadcast=True)
    elif(vars.mode == "memory"):
        vars.mode = "play"
        emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True)

#==================================================================#
#   Toggles the game mode for WI editing and sends UI commands
#==================================================================#
def togglewimode():
    if(vars.mode == "play"):
        vars.mode = "wi"
        emit('from_server', {'cmd': 'wimode', 'data': 'true'}, broadcast=True)
    elif(vars.mode == "wi"):
        # Commit WI fields first
        requestwi()
        # Then set UI state back to Play
        vars.mode = "play"
        emit('from_server', {'cmd': 'wimode', 'data': 'false'}, broadcast=True)
    sendwi()

#==================================================================#
#   
#==================================================================#
def addwiitem(folder_uid=None):
    assert folder_uid is None or folder_uid in vars.wifolders_d
    ob = {"key": "", "keysecondary": "", "content": "", "comment": "", "folder": folder_uid, "num": len(vars.worldinfo), "init": False, "selective": False, "constant": False}
    vars.worldinfo.append(ob)
    while(True):
        uid = int.from_bytes(os.urandom(4), "little", signed=True)
        if(uid not in vars.worldinfo_u):
            break
    vars.worldinfo_u[uid] = vars.worldinfo[-1]
    vars.worldinfo[-1]["uid"] = uid
    if(folder_uid is not None):
        vars.wifolders_u[folder_uid].append(vars.worldinfo[-1])
    emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True)

#==================================================================#
#   Creates a new WI folder with an unused cryptographically secure random UID
#==================================================================#
def addwifolder():
    while(True):
        uid = int.from_bytes(os.urandom(4), "little", signed=True)
        if(uid not in vars.wifolders_d):
            break
    ob = {"name": "", "collapsed": False}
    vars.wifolders_d[uid] = ob
    vars.wifolders_l.append(uid)
    vars.wifolders_u[uid] = []
    emit('from_server', {'cmd': 'addwifolder', 'uid': uid, 'data': ob}, broadcast=True)
    addwiitem(folder_uid=uid)

#==================================================================#
#   Move the WI entry with UID src so that it immediately precedes
#   the WI entry with UID dst
#==================================================================#
def movewiitem(dst, src):
    setgamesaved(False)
    if(vars.worldinfo_u[src]["folder"] is not None):
        for i, e in enumerate(vars.wifolders_u[vars.worldinfo_u[src]["folder"]]):
            if(e is vars.worldinfo_u[src]):
                vars.wifolders_u[vars.worldinfo_u[src]["folder"]].pop(i)
                break
    if(vars.worldinfo_u[dst]["folder"] is not None):
        vars.wifolders_u[vars.worldinfo_u[dst]["folder"]].append(vars.worldinfo_u[src])
    vars.worldinfo_u[src]["folder"] = vars.worldinfo_u[dst]["folder"]
    for i, e in enumerate(vars.worldinfo):
        if(e is vars.worldinfo_u[src]):
            _src = i
        elif(e is vars.worldinfo_u[dst]):
            _dst = i
    vars.worldinfo.insert(_dst - (_dst >= _src), vars.worldinfo.pop(_src))
    sendwi()

#==================================================================#
#   Move the WI folder with UID src so that it immediately precedes
#   the WI folder with UID dst
#==================================================================#
def movewifolder(dst, src):
    setgamesaved(False)
    vars.wifolders_l.remove(src)
    if(dst is None):
        # If dst is None, that means we should move src to be the last folder
        vars.wifolders_l.append(src)
    else:
        vars.wifolders_l.insert(vars.wifolders_l.index(dst), src)
    sendwi()

#==================================================================#
#   
#==================================================================#
def sendwi():
    # Cache len of WI
    ln = len(vars.worldinfo)

    # Clear contents of WI container
    emit('from_server', {'cmd': 'wistart', 'wifolders_d': vars.wifolders_d, 'wifolders_l': vars.wifolders_l, 'data': ''}, broadcast=True)

    # Stable-sort WI entries in order of folder
    stablesortwi()

    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]

    # If there are no WI entries, send an empty WI object
    if(ln == 0):
        addwiitem()
    else:
        # Send contents of WI array
        last_folder = ...
        for wi in vars.worldinfo:
            if(wi["folder"] != last_folder):
                emit('from_server', {'cmd': 'addwifolder', 'uid': wi["folder"], 'data': vars.wifolders_d[wi["folder"]] if wi["folder"] is not None else None}, broadcast=True)
                last_folder = wi["folder"]
            ob = wi
            emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True)
    
    emit('from_server', {'cmd': 'wifinish', 'data': ''}, broadcast=True)

#==================================================================#
#  Request current contents of all WI HTML elements
#==================================================================#
def requestwi():
    list = []
    for wi in vars.worldinfo:
        list.append(wi["num"])
    emit('from_server', {'cmd': 'requestwiitem', 'data': list})

#==================================================================#
#  Stable-sort WI items so that items in the same folder are adjacent,
#  and items in different folders are sorted based on the order of the folders
#==================================================================#
def stablesortwi():
    mapping = {uid: index for index, uid in enumerate(vars.wifolders_l)}
    vars.worldinfo.sort(key=lambda x: mapping[x["folder"]] if x["folder"] is not None else float("inf"))
    last_folder = ...
    last_wi = None
    for i, wi in enumerate(vars.worldinfo):
        wi["num"] = i
        wi["init"] = True
        if(wi["folder"] != last_folder):
            if(last_wi is not None and last_folder is not ...):
                last_wi["init"] = False
            last_folder = wi["folder"]
        last_wi = wi
    if(last_wi is not None):
        last_wi["init"] = False
    for folder in vars.wifolders_u:
        vars.wifolders_u[folder].sort(key=lambda x: x["num"])

#==================================================================#
#  Extract object from server and send it to WI objects
#==================================================================#
def commitwi(ar):
    for ob in ar:
        ob["uid"] = int(ob["uid"])
        vars.worldinfo_u[ob["uid"]]["key"]          = ob["key"]
        vars.worldinfo_u[ob["uid"]]["keysecondary"] = ob["keysecondary"]
        vars.worldinfo_u[ob["uid"]]["content"]      = ob["content"]
        vars.worldinfo_u[ob["uid"]]["comment"]      = ob.get("comment", "")
        vars.worldinfo_u[ob["uid"]]["folder"]       = ob.get("folder", None)
        vars.worldinfo_u[ob["uid"]]["selective"]    = ob["selective"]
        vars.worldinfo_u[ob["uid"]]["constant"]     = ob.get("constant", False)
    stablesortwi()
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]

#==================================================================#
#  
#==================================================================#
def deletewi(uid):
    if(uid in vars.worldinfo_u):
        setgamesaved(False)
        # Store UID of deletion request
        vars.deletewi = uid
        if(vars.deletewi is not None):
            if(vars.worldinfo_u[vars.deletewi]["folder"] is not None):
                for i, e in enumerate(vars.wifolders_u[vars.worldinfo_u[vars.deletewi]["folder"]]):
                    if(e is vars.worldinfo_u[vars.deletewi]):
                        vars.wifolders_u[vars.worldinfo_u[vars.deletewi]["folder"]].pop(i)
            for i, e in enumerate(vars.worldinfo):
                if(e is vars.worldinfo_u[vars.deletewi]):
                    del vars.worldinfo[i]
                    break
            del vars.worldinfo_u[vars.deletewi]
            # Send the new WI array structure
            sendwi()
            # And reset deletewi
            vars.deletewi = None

#==================================================================#
#  
#==================================================================#
def deletewifolder(uid):
    uid = int(uid)
    del vars.wifolders_u[uid]
    del vars.wifolders_d[uid]
    del vars.wifolders_l[vars.wifolders_l.index(uid)]
    setgamesaved(False)
    # Delete uninitialized entries in the folder we're going to delete
    vars.worldinfo = [wi for wi in vars.worldinfo if wi["folder"] != uid or wi["init"]]
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
    # Move WI entries that are inside of the folder we're going to delete
    # so that they're outside of all folders
    for wi in vars.worldinfo:
        if(wi["folder"] == uid):
            wi["folder"] = None

    sendwi()

#==================================================================#
#  Look for WI keys in text to generator 
#==================================================================#
def checkworldinfo(txt, allowed_entries=None, allowed_folders=None, force_use_txt=False, scan_story=True, actions=None):
    original_txt = txt

    if(actions is None):
        actions = vars.actions

    # Dont go any further if WI is empty
    if(len(vars.worldinfo) == 0):
        return "", set()
    
    # Cache actions length
    ln = len(actions)
    
    # Don't bother calculating action history if widepth is 0
    if(vars.widepth > 0 and scan_story):
        depth = vars.widepth
        # If this is not a continue, add 1 to widepth since submitted
        # text is already in action history @ -1
        if(not force_use_txt and (txt != "" and vars.prompt != txt)):
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
            txt = vars.comregex_ai.sub('', vars.prompt) + "".join(chunks)
        elif(ln == 0):
            txt = vars.comregex_ai.sub('', vars.prompt)

    if(force_use_txt):
        txt += original_txt

    # Scan text for matches on WI keys
    wimem = ""
    found_entries = set()
    for wi in vars.worldinfo:
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
                if(vars.wirmvwhtsp):
                    ky = k.strip()
                if ky.lower() in txt.lower():
                    if wi.get("selective", False) and len(keys_secondary):
                        found = False
                        for ks in keys_secondary:
                            ksy = ks
                            if(vars.wirmvwhtsp):
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
    emit('from_server', {'cmd': 'setinputtext', 'data': data}, broadcast=True)
    # Maybe check for length at some point
    # For now just send it to storage
    if(data != vars.memory):
        setgamesaved(False)
    vars.memory = data
    vars.mode = "play"
    emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True)
    
    # Ask for contents of Author's Note field
    emit('from_server', {'cmd': 'getanote', 'data': ''})

#==================================================================#
#  Commit changes to Author's Note
#==================================================================#
def anotesubmit(data, template=""):
    assert type(data) is str and type(template) is str
    # Maybe check for length at some point
    # For now just send it to storage
    if(data != vars.authornote):
        setgamesaved(False)
    vars.authornote = data

    if(vars.authornotetemplate != template):
        vars.setauthornotetemplate = template
        settingschanged()
    vars.authornotetemplate = template

    emit('from_server', {'cmd': 'setanote', 'data': vars.authornote}, broadcast=True)
    emit('from_server', {'cmd': 'setanotetemplate', 'data': vars.authornotetemplate}, broadcast=True)

#==================================================================#
#  Assembles game data into a request to InferKit API
#==================================================================#
def ikrequest(txt):
    # Log request to console
    if not vars.quiet:
        print("{0}Len:{1}, Txt:{2}{3}".format(colors.YELLOW, len(txt), txt, colors.END))
    
    # Build request JSON data
    reqdata = {
        'forceNoEnd': True,
        'length': vars.ikgen,
        'prompt': {
            'isContinuation': False,
            'text': txt
        },
        'startFromBeginning': False,
        'streamResponse': False,
        'temperature': vars.temp,
        'topP': vars.top_p
    }
    
    # Create request
    req = requests.post(
        vars.url, 
        json    = reqdata,
        headers = {
            'Authorization': 'Bearer '+vars.apikey
            }
        )
    
    # Deal with the response
    if(req.status_code == 200):
        genout = req.json()["data"]["text"]

        vars.lua_koboldbridge.outputs[1] = genout

        execute_outmod()
        if(vars.lua_koboldbridge.regeneration_required):
            vars.lua_koboldbridge.regeneration_required = False
            genout = vars.lua_koboldbridge.outputs[1]
            assert genout is str

        if not vars.quiet:
            print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
        vars.actions.append(genout)
        if vars.actions.get_last_key() in vars.actions_metadata:
            vars.actions_metadata[vars.actions.get_last_key()] = {"Selected Text": genout, "Alternative Text": []}
        else:
        # 2. We've selected a chunk of text that is was presented previously
            alternatives = [item['Text'] for item in vars.actions_metadata[vars.actions.get_last_key()]["Alternative Text"]]
            if genout in alternatives:
                alternatives = [item for item in vars.actions_metadata[vars.actions.get_last_key()]["Alternative Text"] if item['Text'] != genout]
                vars.actions_metadata[vars.actions.get_last_key()]["Alternative Text"] = alternatives
            vars.actions_metadata[vars.actions.get_last_key()]["Selected Text"] = genout
        update_story_chunk('last')
        emit('from_server', {'cmd': 'texteffect', 'data': vars.actions.get_last_key() + 1 if len(vars.actions) else 0}, broadcast=True)
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
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
        set_aibusy(0)

#==================================================================#
#  Assembles game data into a request to OpenAI API
#==================================================================#
def oairequest(txt, min, max):
    # Log request to console
    if not vars.quiet:
        print("{0}Len:{1}, Txt:{2}{3}".format(colors.YELLOW, len(txt), txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    vars.lastctx = txt
    
    # Build request JSON data
    # GooseAI is a subntype of OAI. So to check if it's this type, we check the configname as a workaround
    # as the vars.model will always be OAI
    if 'GooseAI' in vars.configname:
        reqdata = {
            'prompt': txt,
            'max_tokens': vars.genamt,
            'temperature': vars.temp,
            'top_a': vars.top_a,
            'top_p': vars.top_p,
            'top_k': vars.top_k,
            'tfs': vars.tfs,
            'typical_p': vars.typical,
            'repetition_penalty': vars.rep_pen,
            'repetition_penalty_slope': vars.rep_pen_slope,
            'repetition_penalty_range': vars.rep_pen_range,
            'n': vars.numseqs,
            'stream': False
        }
    else:
        reqdata = {
            'prompt': txt,
            'max_tokens': vars.genamt,
            'temperature': vars.temp,
            'top_p': vars.top_p,
            'n': vars.numseqs,
            'stream': False
        }
    
    req = requests.post(
        vars.oaiurl, 
        json    = reqdata,
        headers = {
            'Authorization': 'Bearer '+vars.oaiapikey,
            'Content-Type': 'application/json'
            }
        )
    
    # Deal with the response
    if(req.status_code == 200):
        outputs = [out["text"] for out in req.json()["choices"]]

        for idx in range(len(outputs)):
            vars.lua_koboldbridge.outputs[idx+1] = outputs[idx]

        execute_outmod()
        if (vars.lua_koboldbridge.regeneration_required):
            vars.lua_koboldbridge.regeneration_required = False
            genout = []
            for i in range(len(outputs)):
                genout.append(
                    {"generated_text": vars.lua_koboldbridge.outputs[i + 1]})
                assert type(genout[-1]["generated_text"]) is str
        else:
            genout = [
                {"generated_text": utils.decodenewlines(txt)}
                for txt in outputs]

        if vars.actions.get_last_key() not in vars.actions_metadata:
            vars.actions_metadata[vars.actions.get_last_key()] = {
                "Selected Text": genout[0], "Alternative Text": []}
        else:
        # 2. We've selected a chunk of text that is was presented previously
            try:
                alternatives = [item['Text'] for item in vars.actions_metadata[len(vars.actions)-1]["Alternative Text"]]
            except:
                print(len(vars.actions))
                print(vars.actions_metadata)
                raise
            if genout in alternatives:
                alternatives = [item for item in vars.actions_metadata[vars.actions.get_last_key() ]["Alternative Text"] if item['Text'] != genout]
                vars.actions_metadata[vars.actions.get_last_key()]["Alternative Text"] = alternatives
            vars.actions_metadata[vars.actions.get_last_key()]["Selected Text"] = genout

        if (len(genout) == 1):
            genresult(genout[0]["generated_text"])
        else:
            if (vars.lua_koboldbridge.restart_sequence is not None and
                    vars.lua_koboldbridge.restart_sequence > 0):
                genresult(genout[vars.lua_koboldbridge.restart_sequence - 1][
                              "generated_text"])
            else:
                genselect(genout)

        if not vars.quiet:
            print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))

        set_aibusy(0)
    else:
        # Send error message to web client            
        er = req.json()
        if("error" in er):
            type    = er["error"]["type"]
            message = er["error"]["message"]
            
        errmsg = "OpenAI API Error: {0} - {1}".format(type, message)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg}, broadcast=True)
        set_aibusy(0)

#==================================================================#
#  Forces UI to Play mode
#==================================================================#
def exitModes():
    if(vars.mode == "edit"):
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)
    elif(vars.mode == "memory"):
        emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True)
    elif(vars.mode == "wi"):
        emit('from_server', {'cmd': 'wimode', 'data': 'false'}, broadcast=True)
    vars.mode = "play"

#==================================================================#
#  Launch in-browser save prompt
#==================================================================#
def saveas(data):
    
    name = data['name']
    savepins = data['pins']
    # Check if filename exists already
    name = utils.cleanfilename(name)
    if(not fileops.saveexists(name) or (vars.saveow and vars.svowname == name)):
        # All clear to save
        e = saveRequest(fileops.storypath(name), savepins=savepins)
        vars.saveow = False
        vars.svowname = ""
        if(e is None):
            emit('from_server', {'cmd': 'hidesaveas', 'data': ''})
        else:
            print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
            emit('from_server', {'cmd': 'popuperror', 'data': str(e)})
    else:
        # File exists, prompt for overwrite
        vars.saveow   = True
        vars.svowname = name
        emit('from_server', {'cmd': 'askforoverwrite', 'data': ''})

#==================================================================#
#  Launch in-browser story-delete prompt
#==================================================================#
def deletesave(name):
    name = utils.cleanfilename(name)
    e = fileops.deletesave(name)
    if(e is None):
        if(vars.smandelete):
            emit('from_server', {'cmd': 'hidepopupdelete', 'data': ''})
            getloadlist()
        else:
            emit('from_server', {'cmd': 'popuperror', 'data': "The server denied your request to delete this story"})
    else:
        print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
        emit('from_server', {'cmd': 'popuperror', 'data': str(e)})

#==================================================================#
#  Launch in-browser story-rename prompt
#==================================================================#
def renamesave(name, newname):
    # Check if filename exists already
    name = utils.cleanfilename(name)
    newname = utils.cleanfilename(newname)
    if(not fileops.saveexists(newname) or name == newname or (vars.saveow and vars.svowname == newname)):
        e = fileops.renamesave(name, newname)
        vars.saveow = False
        vars.svowname = ""
        if(e is None):
            if(vars.smanrename):
                emit('from_server', {'cmd': 'hidepopuprename', 'data': ''})
                getloadlist()
            else:
                emit('from_server', {'cmd': 'popuperror', 'data': "The server denied your request to rename this story"})
        else:
            print("{0}{1}{2}".format(colors.RED, str(e), colors.END))
            emit('from_server', {'cmd': 'popuperror', 'data': str(e)})
    else:
        # File exists, prompt for overwrite
        vars.saveow   = True
        vars.svowname = newname
        emit('from_server', {'cmd': 'askforoverwrite', 'data': ''})

#==================================================================#
#  Save the currently running story
#==================================================================#
def save():
    # Check if a file is currently open
    if(".json" in vars.savedir):
        saveRequest(vars.savedir)
    else:
        emit('from_server', {'cmd': 'saveas', 'data': ''})

#==================================================================#
#  Save the story via file browser
#==================================================================#
def savetofile():
    savpath = fileops.getsavepath(vars.savedir, "Save Story As", [("Json", "*.json")])
    saveRequest(savpath)

#==================================================================#
#  Save the story to specified path
#==================================================================#
def saveRequest(savpath, savepins=True):    
    if(savpath):
        # Leave Edit/Memory mode before continuing
        exitModes()
        
        # Save path for future saves
        vars.savedir = savpath
        txtpath = os.path.splitext(savpath)[0] + ".txt"
        # Build json to write
        js = {}
        js["gamestarted"] = vars.gamestarted
        js["prompt"]      = vars.prompt
        js["memory"]      = vars.memory
        js["authorsnote"] = vars.authornote
        js["anotetemplate"] = vars.authornotetemplate
        js["actions"]     = tuple(vars.actions.values())
        if savepins:
            js["actions_metadata"]     = vars.actions_metadata
        js["worldinfo"]   = []
        js["wifolders_d"] = vars.wifolders_d
        js["wifolders_l"] = vars.wifolders_l
		
        # Extract only the important bits of WI
        for wi in vars.worldinfo_i:
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
                
        txt = vars.prompt + "".join(vars.actions.values())

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
        vars.laststory = filename
        emit('from_server', {'cmd': 'setstoryname', 'data': vars.laststory}, broadcast=True)
        setgamesaved(True)
        print("{0}Story saved to {1}!{2}".format(colors.GREEN, path.basename(savpath), colors.END))

#==================================================================#
#  Show list of saved stories
#==================================================================#
def getloadlist():
    emit('from_server', {'cmd': 'buildload', 'data': fileops.getstoryfiles()})

#==================================================================#
#  Show list of soft prompts
#==================================================================#
def getsplist():
    if(vars.allowsp):
        emit('from_server', {'cmd': 'buildsp', 'data': fileops.getspfiles(vars.modeldim)})

#==================================================================#
#  Get list of userscripts
#==================================================================#
def getuslist():
    files = {i: v for i, v in enumerate(fileops.getusfiles())}
    loaded = []
    unloaded = []
    userscripts = set(vars.userscripts)
    for i in range(len(files)):
        if files[i]["filename"] not in userscripts:
            unloaded.append(files[i])
    files = {files[k]["filename"]: files[k] for k in files}
    userscripts = set(files.keys())
    for filename in vars.userscripts:
        if filename in userscripts:
            loaded.append(files[filename])
    return unloaded, loaded

#==================================================================#
#  Load a saved story via file browser
#==================================================================#
def loadfromfile():
    loadpath = fileops.getloadpath(vars.savedir, "Select Story File", [("Json", "*.json")])
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
        
        # Copy file contents to vars
        vars.gamestarted = js["gamestarted"]
        vars.prompt      = js["prompt"]
        vars.memory      = js["memory"]
        vars.worldinfo   = []
        vars.worldinfo   = []
        vars.worldinfo_u = {}
        vars.wifolders_d = {int(k): v for k, v in js.get("wifolders_d", {}).items()}
        vars.wifolders_l = js.get("wifolders_l", [])
        vars.wifolders_u = {uid: [] for uid in vars.wifolders_d}
        vars.lastact     = ""
        vars.submission  = ""
        vars.lastctx     = ""
        vars.genseqs = []

        del vars.actions
        vars.actions = structures.KoboldStoryRegister()
        actions = collections.deque(js["actions"])
        

        if "actions_metadata" in js:
            
            if type(js["actions_metadata"]) == dict:
                temp = js["actions_metadata"]
                vars.actions_metadata = {}
                #we need to redo the numbering of the actions_metadata since the actions list doesn't preserve it's number on saving
                if len(temp) > 0:
                    counter = 0
                    temp = {int(k):v for k,v in temp.items()}
                    for i in range(max(temp)+1):
                        if i in temp:
                            vars.actions_metadata[counter] = temp[i]
                            counter += 1
                del temp
            else:
                #fix if we're using the old metadata format
                vars.actions_metadata = {}
                i = 0
                
                for text in js['actions']:
                    vars.actions_metadata[i] = {'Selected Text': text, 'Alternative Text': []}
                    i+=1
        else:
            vars.actions_metadata = {}
            i = 0
            
            for text in js['actions']:
                vars.actions_metadata[i] = {'Selected Text': text, 'Alternative Text': []}
                i+=1

        footer = ""                

        if(len(vars.prompt.strip()) == 0):
            while(len(actions)):
                action = actions.popleft()
                if(len(action.strip()) != 0):
                    vars.prompt = action
                    break
            else:
                vars.gamestarted = False
        vars.prompt = vars.prompt.lstrip()
        ln = len(vars.prompt.rstrip())
        footer += vars.prompt[ln:]
        vars.prompt = vars.prompt[:ln]
        if(vars.gamestarted):
            for s in actions:
                if(len(s.strip()) == 0):
                    # If this action only contains whitespace, we merge it with the next action
                    footer += s
                    continue
                vars.actions.append(footer + s)
                footer = ""
                # If there is trailing whitespace at the end of an action, we move that whitespace to the beginning of the next action
                ln = len(vars.actions[vars.actions.get_last_key()].rstrip())
                footer += vars.actions[vars.actions.get_last_key()][ln:]
                vars.actions[vars.actions.get_last_key()] = vars.actions[vars.actions.get_last_key()][:ln]
        
        # Try not to break older save files
        if("authorsnote" in js):
            vars.authornote = js["authorsnote"]
        else:
            vars.authornote = ""
        if("anotetemplate" in js):
            vars.authornotetemplate = js["anotetemplate"]
        else:
            vars.authornotetemplate = "[Author's note: <|>]"
        
        if("worldinfo" in js):
            num = 0
            for wi in js["worldinfo"]:
                vars.worldinfo.append({
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
                    if(uid not in vars.worldinfo_u):
                        break
                vars.worldinfo_u[uid] = vars.worldinfo[-1]
                vars.worldinfo[-1]["uid"] = uid
                if(vars.worldinfo[-1]["folder"] is not None):
                    vars.wifolders_u[vars.worldinfo[-1]["folder"]].append(vars.worldinfo[-1])
                num += 1

        for uid in vars.wifolders_l + [None]:
            vars.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
            while(True):
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if(uid not in vars.worldinfo_u):
                    break
            vars.worldinfo_u[uid] = vars.worldinfo[-1]
            vars.worldinfo[-1]["uid"] = uid
            if(vars.worldinfo[-1]["folder"] is not None):
                vars.wifolders_u[vars.worldinfo[-1]["folder"]].append(vars.worldinfo[-1])
        stablesortwi()
        vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]

        # Save path for save button
        vars.savedir = loadpath
        
        # Clear loadselect var
        vars.loadselect = ""
        
        # Refresh game screen
        _filename = filename
        if(filename.endswith('.json')):
            _filename = filename[:-5]
        vars.laststory = _filename
        emit('from_server', {'cmd': 'setstoryname', 'data': vars.laststory}, broadcast=True)
        setgamesaved(True)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': vars.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote}, broadcast=True)
        emit('from_server', {'cmd': 'setanotetemplate', 'data': vars.authornotetemplate}, broadcast=True)
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True)
        emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True)
        print("{0}Story loaded from {1}!{2}".format(colors.GREEN, filename, colors.END))
        
        send_debug()

#==================================================================#
# Import an AIDungon game exported with Mimi's tool
#==================================================================#
def importRequest():
    importpath = fileops.getloadpath(vars.savedir, "Select AID CAT File", [("Json", "*.json")])
    
    if(importpath):
        # Leave Edit/Memory mode before continuing
        exitModes()
        
        # Read file contents into JSON object
        file = open(importpath, "rb")
        vars.importjs = json.load(file)
        
        # If a bundle file is being imported, select just the Adventures object
        if type(vars.importjs) is dict and "stories" in vars.importjs:
            vars.importjs = vars.importjs["stories"]
        
        # Clear Popup Contents
        emit('from_server', {'cmd': 'clearpopup', 'data': ''}, broadcast=True)
        
        # Initialize vars
        num = 0
        vars.importnum = -1
        
        # Get list of stories
        for story in vars.importjs:
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
            emit('from_server', {'cmd': 'addimportline', 'data': ob})
            num += 1
        
        # Show Popup
        emit('from_server', {'cmd': 'popupshow', 'data': True})

#==================================================================#
# Import an AIDungon game selected in popup
#==================================================================#
def importgame():
    if(vars.importnum >= 0):
        # Cache reference to selected game
        ref = vars.importjs[vars.importnum]
        
        # Copy game contents to vars
        vars.gamestarted = True
        
        # Support for different versions of export script
        if("actions" in ref):
            if(len(ref["actions"]) > 0):
                vars.prompt = ref["actions"][0]["text"]
            else:
                vars.prompt = ""
        elif("actionWindow" in ref):
            if(len(ref["actionWindow"]) > 0):
                vars.prompt = ref["actionWindow"][0]["text"]
            else:
                vars.prompt = ""
        else:
            vars.prompt = ""
        vars.memory      = ref["memory"]
        vars.authornote  = ref["authorsNote"] if type(ref["authorsNote"]) is str else ""
        vars.authornotetemplate = "[Author's note: <|>]"
        vars.actions     = structures.KoboldStoryRegister()
        vars.actions_metadata = {}
        vars.worldinfo   = []
        vars.worldinfo_i = []
        vars.worldinfo_u = {}
        vars.wifolders_d = {}
        vars.wifolders_l = []
        vars.wifolders_u = {uid: [] for uid in vars.wifolders_d}
        vars.lastact     = ""
        vars.submission  = ""
        vars.lastctx     = ""
        
        # Get all actions except for prompt
        if("actions" in ref):
            if(len(ref["actions"]) > 1):
                for act in ref["actions"][1:]:
                    vars.actions.append(act["text"])
        elif("actionWindow" in ref):
            if(len(ref["actionWindow"]) > 1):
                for act in ref["actionWindow"][1:]:
                    vars.actions.append(act["text"])
        
        # Get just the important parts of world info
        if(ref["worldInfo"] != None):
            if(len(ref["worldInfo"]) > 1):
                num = 0
                for wi in ref["worldInfo"]:
                    vars.worldinfo.append({
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
                        if(uid not in vars.worldinfo_u):
                            break
                    vars.worldinfo_u[uid] = vars.worldinfo[-1]
                    vars.worldinfo[-1]["uid"] = uid
                    if(vars.worldinfo[-1]["folder"]) is not None:
                        vars.wifolders_u[vars.worldinfo[-1]["folder"]].append(vars.worldinfo[-1])
                    num += 1

        for uid in vars.wifolders_l + [None]:
            vars.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
            while(True):
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if(uid not in vars.worldinfo_u):
                    break
            vars.worldinfo_u[uid] = vars.worldinfo[-1]
            vars.worldinfo[-1]["uid"] = uid
            if(vars.worldinfo[-1]["folder"] is not None):
                vars.wifolders_u[vars.worldinfo[-1]["folder"]].append(vars.worldinfo[-1])
        stablesortwi()
        vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
        
        # Clear import data
        vars.importjs = {}
        
        # Reset current save
        vars.savedir = getcwd()+"\\stories"
        
        # Refresh game screen
        vars.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': vars.laststory}, broadcast=True)
        setgamesaved(False)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': vars.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote}, broadcast=True)
        emit('from_server', {'cmd': 'setanotetemplate', 'data': vars.authornotetemplate}, broadcast=True)
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True)
        emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True)

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
        vars.gamestarted = True
        vars.prompt      = js["promptContent"]
        vars.memory      = js["memory"]
        vars.authornote  = js["authorsNote"]
        vars.authornotetemplate = "[Author's note: <|>]"
        vars.actions     = structures.KoboldStoryRegister()
        vars.actions_metadata = {}
        vars.worldinfo   = []
        vars.worldinfo_i = []
        vars.worldinfo_u = {}
        vars.wifolders_d = {}
        vars.wifolders_l = []
        vars.wifolders_u = {uid: [] for uid in vars.wifolders_d}
        vars.lastact     = ""
        vars.submission  = ""
        vars.lastctx     = ""
        
        if not vars.memory:
            vars.memory = ""
        if not vars.authornote:
            vars.authornote = ""
        
        num = 0
        for wi in js["worldInfos"]:
            vars.worldinfo.append({
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
                if(uid not in vars.worldinfo_u):
                    break
            vars.worldinfo_u[uid] = vars.worldinfo[-1]
            vars.worldinfo[-1]["uid"] = uid
            if(vars.worldinfo[-1]["folder"]) is not None:
                vars.wifolders_u[vars.worldinfo[-1]["folder"]].append(vars.worldinfo[-1])
            num += 1

        for uid in vars.wifolders_l + [None]:
            vars.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
            while(True):
                uid = int.from_bytes(os.urandom(4), "little", signed=True)
                if(uid not in vars.worldinfo_u):
                    break
            vars.worldinfo_u[uid] = vars.worldinfo[-1]
            vars.worldinfo[-1]["uid"] = uid
            if(vars.worldinfo[-1]["folder"] is not None):
                vars.wifolders_u[vars.worldinfo[-1]["folder"]].append(vars.worldinfo[-1])
        stablesortwi()
        vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]

        # Reset current save
        vars.savedir = getcwd()+"\\stories"
        
        # Refresh game screen
        vars.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': vars.laststory}, broadcast=True)
        setgamesaved(False)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': vars.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote}, broadcast=True)
        emit('from_server', {'cmd': 'setanotetemplate', 'data': vars.authornotetemplate}, broadcast=True)
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True)

#==================================================================#
#  Import World Info JSON file
#==================================================================#
def wiimportrequest():
    importpath = fileops.getloadpath(vars.savedir, "Select World Info File", [("Json", "*.json")])
    if(importpath):
        file = open(importpath, "rb")
        js = json.load(file)
        if(len(js) > 0):
            # If the most recent WI entry is blank, remove it.
            if(not vars.worldinfo[-1]["init"]):
                del vars.worldinfo[-1]
            # Now grab the new stuff
            num = len(vars.worldinfo)
            for wi in js:
                vars.worldinfo.append({
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
                    if(uid not in vars.worldinfo_u):
                        break
                vars.worldinfo_u[uid] = vars.worldinfo[-1]
                vars.worldinfo[-1]["uid"] = uid
                if(vars.worldinfo[-1]["folder"]) is not None:
                    vars.wifolders_u[vars.worldinfo[-1]["folder"]].append(vars.worldinfo[-1])
                num += 1
            for uid in [None]:
                vars.worldinfo.append({"key": "", "keysecondary": "", "content": "", "comment": "", "folder": uid, "num": None, "init": False, "selective": False, "constant": False, "uid": None})
                while(True):
                    uid = int.from_bytes(os.urandom(4), "little", signed=True)
                    if(uid not in vars.worldinfo_u):
                        break
                vars.worldinfo_u[uid] = vars.worldinfo[-1]
                vars.worldinfo[-1]["uid"] = uid
                if(vars.worldinfo[-1]["folder"] is not None):
                    vars.wifolders_u[vars.worldinfo[-1]["folder"]].append(vars.worldinfo[-1])
        
        if not vars.quiet:
            print("{0}".format(vars.worldinfo[0]))
                
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
    vars.gamestarted = False
    vars.prompt      = ""
    vars.memory      = ""
    vars.actions     = structures.KoboldStoryRegister()
    vars.actions_metadata = {}
    
    vars.authornote  = ""
    vars.authornotetemplate = vars.setauthornotetemplate
    vars.worldinfo   = []
    vars.worldinfo_i = []
    vars.worldinfo_u = {}
    vars.wifolders_d = {}
    vars.wifolders_l = []
    vars.lastact     = ""
    vars.submission  = ""
    vars.lastctx     = ""
    
    # Reset current save
    vars.savedir = getcwd()+"\\stories"
    
    # Refresh game screen
    vars.laststory = None
    emit('from_server', {'cmd': 'setstoryname', 'data': vars.laststory}, broadcast=True)
    setgamesaved(True)
    sendwi()
    emit('from_server', {'cmd': 'setmemory', 'data': vars.memory}, broadcast=True)
    emit('from_server', {'cmd': 'setanote', 'data': vars.authornote}, broadcast=True)
    emit('from_server', {'cmd': 'setanotetemplate', 'data': vars.authornotetemplate}, broadcast=True)
    setStartState()

def randomGameRequest(topic, memory=""): 
    if(vars.noai):
        newGameRequest()
        vars.memory = memory
        emit('from_server', {'cmd': 'setmemory', 'data': vars.memory}, broadcast=True)
        return
    vars.recentrng = topic
    vars.recentrngm = memory
    newGameRequest()
    setgamesaved(False)
    _memory = memory
    if(len(memory) > 0):
        _memory = memory.rstrip() + "\n\n"
    vars.memory      = _memory + "You generate the following " + topic + " story concept :"
    vars.lua_koboldbridge.feedback = None
    actionsubmit("", force_submit=True, force_prompt_gen=True)
    vars.memory      = memory
    emit('from_server', {'cmd': 'setmemory', 'data': vars.memory}, broadcast=True)

def final_startup():
    # Prevent tokenizer from taking extra time the first time it's used
    def __preempt_tokenizer():
        if("tokenizer" not in globals()):
            return
        utils.decodenewlines(tokenizer.decode([25678, 559]))
        tokenizer.encode(utils.encodenewlines("eunoia"))
    threading.Thread(target=__preempt_tokenizer).start()

    # Load soft prompt specified by the settings file, if applicable
    if(path.exists(get_config_filename())):
        file = open(get_config_filename(), "r")
        js   = json.load(file)
        if(vars.allowsp and "softprompt" in js and type(js["softprompt"]) is str and all(q not in js["softprompt"] for q in ("..", ":")) and (len(js["softprompt"]) == 0 or all(js["softprompt"][0] not in q for q in ("/", "\\")))):
            spRequest(js["softprompt"])
        else:
            vars.spfilename = ""
        file.close()

    # Precompile TPU backend if required
    if(vars.use_colab_tpu or vars.model in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")):
        soft_tokens = tpumtjgetsofttokens()
        if(vars.dynamicscan or (not vars.nogenmod and vars.has_genmod)):
            threading.Thread(
                target=tpu_mtj_backend.infer_dynamic,
                args=(np.tile(np.uint32((23403, 727, 20185)), (vars.numseqs, 1)),),
                kwargs={
                    "soft_embeddings": vars.sp,
                    "soft_tokens": soft_tokens,
                    "gen_len": 1,
                    "use_callback": False,
                    "numseqs": vars.numseqs,
                    "excluded_world_info": list(set() for _ in range(vars.numseqs)),
                },
            ).start()
        else:
            threading.Thread(
                target=tpu_mtj_backend.infer_static,
                args=(np.uint32((23403, 727, 20185)),),
                kwargs={
                    "soft_embeddings": vars.sp,
                    "soft_tokens": soft_tokens,
                    "gen_len": 1,
                    "numseqs": vars.numseqs,
                },
            ).start()

    # Set the initial RNG seed
    if(vars.seed is not None):
        if(vars.use_colab_tpu):
            if(vars.seed_specified):
                __import__("tpu_mtj_backend").set_rng_seed(vars.seed)
            else:
                __import__("tpu_mtj_backend").randomize_rng_seed()
        else:
            if(vars.seed_specified):
                __import__("torch").manual_seed(vars.seed)
            else:
                __import__("torch").seed()
    vars.seed = __import__("tpu_mtj_backend").get_rng_seed() if vars.use_colab_tpu else __import__("torch").initial_seed()

def send_debug():
    if vars.debug:
        debug_info = ""
        try:
            debug_info = "{}Seed: {} ({})\n".format(debug_info, repr(__import__("tpu_mtj_backend").get_rng_seed() if vars.use_colab_tpu else __import__("torch").initial_seed()), "specified by user in settings file" if vars.seed_specified else "randomly generated")
        except:
            pass
        try:
            debug_info = "{}Newline Mode: {}\n".format(debug_info, vars.newlinemode)
        except:
            pass
        try:
            debug_info = "{}Action Length: {}\n".format(debug_info, vars.actions.get_last_key())
        except:
            pass
        try:
            debug_info = "{}Actions Metadata Length: {}\n".format(debug_info, max(vars.actions_metadata) if len(vars.actions_metadata) > 0 else 0)
        except:
            pass
        try:
            debug_info = "{}Actions: {}\n".format(debug_info, [k for k in vars.actions])
        except:
            pass
        try:
            debug_info = "{}Actions Metadata: {}\n".format(debug_info, [k for k in vars.actions_metadata])
        except:
            pass
        try:
            debug_info = "{}Last Action: {}\n".format(debug_info, vars.actions[vars.actions.get_last_key()])
        except:
            pass
        try:
            debug_info = "{}Last Metadata: {}\n".format(debug_info, vars.actions_metadata[max(vars.actions_metadata)])
        except:
            pass

        emit('from_server', {'cmd': 'debug_info', 'data': debug_info}, broadcast=True)

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
# File Popup options
#==================================================================#

@socketio.on('upload_file')
def upload_file(data):
    print("upload_file {}".format(data['filename']))
    print('current_folder' in session)
    print('popup_jailed_dir' not in session)
    print(session['popup_jailed_dir'])
    print(session['current_folder'])    
    if 'current_folder' in session:
        path = os.path.abspath(os.path.join(session['current_folder'], data['filename']).replace("\\", "/")).replace("\\", "/")
        print(path)
        print(os.path.exists(path))
        if 'popup_jailed_dir' not in session:
            print("Someone is trying to upload a file to your server. Blocked.")
        elif session['popup_jailed_dir'] is None:
            if os.path.exists(path):
                print("popup error")
                emit("error_popup", "The file already exists. Please delete it or rename the file before uploading", room="UI_2");
            else:
                with open(path, "wb") as f:
                    f.write(data['data'])
                get_files_folders(session['current_folder'])
                print("saved")
        elif session['popup_jailed_dir'] in session['current_folder']:
            if os.path.exists(path):
                print("popup error")
                emit("error_popup", "The file already exists. Please delete it or rename the file before uploading", room="UI_2");
            else:
                with open(path, "wb") as f:
                    f.write(data['data'])
                get_files_folders(session['current_folder'])
                print("saved")

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

def file_popup(popup_title, starting_folder, return_event, upload=True, jailed=True, folder_only=True, renameable=False, deleteable=False, editable=False, show_breadcrumbs=True, item_check=None, show_hidden=False):
    #starting_folder = The folder we're going to get folders and/or items from
    #return_event = the socketio event that will be emitted when the load button is clicked
    #jailed = if set to true will look for the session variable jailed_folder and prevent navigation outside of that folder
    #folder_only = will only show folders, no files
    #deletable = will show the delete icons/methods.
    #editable = will show the edit icons/methods
    #show_breadcrumbs = will show the breadcrumbs at the top of the screen
    #item_check will call this function to check if the item is valid as a selection if not none. Will pass absolute directory as only argument to function
    #show_hidden = ... really, you have to ask?
    if jailed:
        session['popup_jailed_dir'] = os.path.abspath(starting_folder).replace("\\", "/")
    else:
        session['popup_jailed_dir'] = None
    session['popup_deletable'] = deleteable
    session['popup_renameable'] = renameable
    session['popup_editable'] = editable
    session['popup_show_hidden'] = show_hidden
    session['popup_item_check'] = item_check
    session['popup_folder_only'] = folder_only
    session['popup_show_breadcrumbs'] = show_breadcrumbs
    session['upload'] = upload
    
    socketio.emit("load_popup", {"popup_title": popup_title, "call_back": return_event, "renameable": renameable, "deleteable": deleteable, "editable": editable, 'upload': upload}, broadcast=True)
    
    get_files_folders(starting_folder)
    
    
def get_files_folders(starting_folder):
    import stat
    session['current_folder'] = os.path.abspath(starting_folder).replace("\\", "/")
    item_check = session['popup_item_check']
    show_breadcrumbs = session['popup_show_breadcrumbs']
    show_hidden = session['popup_show_hidden']
    folder_only = session['popup_folder_only']
    
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
        for item in os.listdir(base_path):
            item_full_path = os.path.join(base_path, item).replace("\\", "/")
            if hasattr(os.stat(item_full_path), "st_file_attributes"):
                hidden = bool(os.stat(item_full_path).st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)
            else:
                hidden = item[0] == "."
            if item_check is None:
                valid_selection = True
            else:
                valid_selection = item_check(item_full_path)
                
            if (show_hidden and hidden) or not hidden:
                if os.path.isdir(os.path.join(base_path, item)):
                    folders.append([True, item_full_path, item,  valid_selection])
                else:
                    files.append([False, item_full_path, item,  valid_selection])
        items = folders
        if not folder_only:
            items += files
            
    socketio.emit("popup_items", items, broadcast=True, include_self=True)
    if show_breadcrumbs:
        socketio.emit("popup_breadcrumbs", breadcrumbs, broadcast=True)


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
    if not vars.allowsp:
        raise ValidationError("Cannot use soft prompts with current backend.")
    if any(q in soft_prompt for q in ("/", "\\")):
        return
    z, _, _, _, _ = fileops.checksp(soft_prompt.strip(), vars.modeldim)
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
    quiet: Optional[bool] = fields.Boolean(metadata={"description": "When enabled, Generated output will not be displayed in the console."})
    sampler_order: Optional[List[int]] = fields.List(fields.Integer(), validate=[validate.Length(min=6), permutation_validator], metadata={"description": "Sampler order to be used. If N is the length of this array, then N must be greater than or equal to 6 and the array must be a permutation of the first N non-negative integers."})
    sampler_seed: Optional[int] = fields.Integer(validate=validate.Range(min=0, max=2**64 - 1), metadata={"description": "RNG seed to use for sampling. If not specified, the global RNG will be used."})
    sampler_full_determinism: Optional[bool] = fields.Boolean(metadata={"description": "If enabled, the generated text will always be the same as long as you use the same RNG seed, input and settings. If disabled, only the *sequence* of generated texts that you get when repeatedly generating text will be the same given the same RNG seed, input and settings."})

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
    if vars.aibusy or vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
    if vars.use_colab_tpu:
        import tpu_mtj_backend
    if hasattr(body, "sampler_seed"):
        # If a seed was specified, we need to save the global RNG state so we
        # can restore it later
        old_seed = vars.seed
        old_rng_state = tpu_mtj_backend.get_rng_state() if vars.use_colab_tpu else torch.get_rng_state()
        vars.seed = body.sampler_seed
        # We should try to use a previously saved RNG state with the same seed
        if body.sampler_seed in vars.rng_states:
            if vars.use_colab_tpu:
                tpu_mtj_backend.set_rng_state(vars.rng_states[body.sampler_seed])
            else:
                torch.set_rng_state(vars.rng_states[body.sampler_seed])
        else:
            if vars.use_colab_tpu:
                tpu_mtj_backend.set_rng_state(tpu_mtj_backend.new_rng_state(body.sampler_seed))
            else:
                torch.manual_seed(body.sampler_seed)
        vars.rng_states[body.sampler_seed] = tpu_mtj_backend.get_rng_state() if vars.use_colab_tpu else torch.get_rng_state()
    if hasattr(body, "sampler_order"):
        if len(body.sampler_order) < 7:
            body.sampler_order = [6] + body.sampler_order
    # This maps each property of the setting to use when sending the generate idempotently
    # To the object which typically contains it's value
    # This allows to set the property only for the API generation, and then revert the setting
    # To what it was before.
    mapping = {
        "disable_input_formatting": ("vars", "disable_input_formatting", None),
        "disable_output_formatting": ("vars", "disable_output_formatting", None),
        "rep_pen": ("vars", "rep_pen", None),
        "rep_pen_range": ("vars", "rep_pen_range", None),
        "rep_pen_slope": ("vars", "rep_pen_slope", None),
        "top_k": ("vars", "top_k", None),
        "top_a": ("vars", "top_a", None),
        "top_p": ("vars", "top_p", None),
        "tfs": ("vars", "tfs", None),
        "typical": ("vars", "typical", None),
        "temperature": ("vars", "temp", None),
        "frmtadsnsp": ("vars.formatoptns", "@frmtadsnsp", "input"),
        "frmttriminc": ("vars.formatoptns", "@frmttriminc", "output"),
        "frmtrmblln": ("vars.formatoptns", "@frmtrmblln", "output"),
        "frmtrmspch": ("vars.formatoptns", "@frmtrmspch", "output"),
        "singleline": ("vars.formatoptns", "@singleline", "output"),
        "max_length": ("vars", "genamt", None),
        "max_context_length": ("vars", "max_length", None),
        "n": ("vars", "numseqs", None),
        "quiet": ("vars", "quiet", None),
        "sampler_order": ("vars", "sampler_order", None),
        "sampler_full_determinism": ("vars", "full_determinism", None),
    }
    saved_settings = {}
    set_aibusy(1)
    disable_set_aibusy = vars.disable_set_aibusy
    vars.disable_set_aibusy = True
    _standalone = vars.standalone
    vars.standalone = True
    show_probs = vars.show_probs
    vars.show_probs = False
    output_streaming = vars.output_streaming
    vars.output_streaming = False
    for key, entry in mapping.items():
        obj = {"vars": vars, "vars.formatoptns": vars.formatoptns}[entry[0]]
        if entry[2] == "input" and vars.disable_input_formatting and not hasattr(body, key):
            setattr(body, key, False)
        if entry[2] == "output" and vars.disable_output_formatting and not hasattr(body, key):
            setattr(body, key, False)
        if getattr(body, key, None) is not None:
            if entry[1].startswith("@"):
                saved_settings[key] = obj[entry[1][1:]]
                obj[entry[1][1:]] = getattr(body, key)
            else:
                saved_settings[key] = getattr(obj, entry[1])
                setattr(obj, entry[1], getattr(body, key))
    try:
        if vars.allowsp and getattr(body, "soft_prompt", None) is not None:
            if any(q in body.soft_prompt for q in ("/", "\\")):
                raise RuntimeError
            old_spfilename = vars.spfilename
            spRequest(body.soft_prompt.strip())
        genout = apiactionsubmit(body.prompt, use_memory=body.use_memory, use_story=body.use_story, use_world_info=body.use_world_info, use_authors_note=body.use_authors_note)
        output = {"results": [{"text": txt} for txt in genout]}
    finally:
        for key in saved_settings:
            entry = mapping[key]
            obj = {"vars": vars, "vars.formatoptns": vars.formatoptns}[entry[0]]
            if getattr(body, key, None) is not None:
                if entry[1].startswith("@"):
                    if obj[entry[1][1:]] == getattr(body, key):
                        obj[entry[1][1:]] = saved_settings[key]
                else:
                    if getattr(obj, entry[1]) == getattr(body, key):
                        setattr(obj, entry[1], saved_settings[key])
        vars.disable_set_aibusy = disable_set_aibusy
        vars.standalone = _standalone
        vars.show_probs = show_probs
        vars.output_streaming = output_streaming
        if vars.allowsp and getattr(body, "soft_prompt", None) is not None:
            spRequest(old_spfilename)
        if hasattr(body, "sampler_seed"):
            vars.seed = old_seed
            if vars.use_colab_tpu:
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
    return {"result": vars.model}


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
    if vars.aibusy or vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
    set_aibusy(1)
    disable_set_aibusy = vars.disable_set_aibusy
    vars.disable_set_aibusy = True
    _standalone = vars.standalone
    vars.standalone = True
    numseqs = vars.numseqs
    vars.numseqs = 1
    try:
        actionsubmit(body.prompt, force_submit=True, no_generate=True, ignore_aibusy=True)
    finally:
        vars.disable_set_aibusy = disable_set_aibusy
        vars.standalone = _standalone
        vars.numseqs = numseqs
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
    if not vars.gamestarted:
        abort(Response(json.dumps({"detail": {
            "msg": "Could not retrieve the last action of the story because the story is empty.",
            "type": "story_empty",
        }}), mimetype="application/json", status=510))
    if len(vars.actions) == 0:
        return {"result": {"text": vars.prompt, "num": 0}}
    return {"result": {"text": vars.actions[vars.actions.get_last_key()], "num": vars.actions.get_last_key() + 1}}


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
    if not vars.gamestarted:
        abort(Response(json.dumps({"detail": {
            "msg": "Could not retrieve the last action of the story because the story is empty.",
            "type": "story_empty",
        }}), mimetype="application/json", status=510))
    if len(vars.actions) == 0:
        return {"result": {"text": 0}}
    return {"result": {"text": vars.actions.get_last_key() + 1}}


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
    if not vars.gamestarted:
        abort(Response(json.dumps({"detail": {
            "msg": "Could not retrieve the last action of the story because the story is empty.",
            "type": "story_empty",
        }}), mimetype="application/json", status=510))
    if len(vars.actions) == 0:
        return {"result": {"text": vars.prompt}}
    return {"result": {"text": vars.actions[vars.actions.get_last_key()]}}


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
    if not vars.gamestarted:
        abort(Response(json.dumps({"detail": {
            "msg": "Could not retrieve the last action of the story because the story is empty.",
            "type": "story_empty",
        }}), mimetype="application/json", status=510))
    value = body.value.rstrip()
    if len(vars.actions) == 0:
        inlineedit(0, value)
    else:
        inlineedit(vars.actions.get_last_key() + 1, value)
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
    if vars.aibusy or vars.genseqs:
        abort(Response(json.dumps({"detail": {
            "msg": "Server is busy; please try again later.",
            "type": "service_unavailable",
        }}), mimetype="application/json", status=503))
    if not vars.gamestarted or not len(vars.actions):
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
    if vars.gamestarted:
        chunks.append({"num": 0, "text": vars.prompt})
    for num, action in vars.actions.items():
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
    if vars.gamestarted:
        chunks.append(0)
    for num in vars.actions.keys():
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
        return {"result": vars.gamestarted}
    return {"result": num - 1 in vars.actions}


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
        if not vars.gamestarted:
            abort(Response(json.dumps({"detail": {
                "msg": "No chunk with the given num exists.",
                "type": "key_error",
            }}), mimetype="application/json", status=404))
        return {"result": {"text": vars.prompt, "num": num}}
    if num - 1 not in vars.actions:
        abort(Response(json.dumps({"detail": {
            "msg": "No chunk with the given num exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"result": {"text": vars.actions[num - 1], "num": num}}


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
        if not vars.gamestarted:
            abort(Response(json.dumps({"detail": {
                "msg": "No chunk with the given num exists.",
                "type": "key_error",
            }}), mimetype="application/json", status=404))
        return {"value": vars.prompt}
    if num - 1 not in vars.actions:
        abort(Response(json.dumps({"detail": {
            "msg": "No chunk with the given num exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": vars.actions[num - 1]}


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
        if not vars.gamestarted:
            abort(Response(json.dumps({"detail": {
                "msg": "No chunk with the given num exists.",
                "type": "key_error",
            }}), mimetype="application/json", status=404))
        inlineedit(0, body.value.rstrip())
        return {}
    if num - 1 not in vars.actions:
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
    if num - 1 not in vars.actions:
        abort(Response(json.dumps({"detail": {
            "msg": "No chunk with the given num exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    if vars.aibusy or vars.genseqs:
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
    if vars.aibusy or vars.genseqs:
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
    if vars.aibusy or vars.genseqs:
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
    ln = len(vars.worldinfo)
    stablesortwi()
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
    folder: Optional[list] = None
    if ln:
        last_folder = ...
        for wi in vars.worldinfo_i:
            if wi["folder"] != last_folder:
                folder = []
                if wi["folder"] is not None:
                    folders.append({"uid": wi["folder"], "name": vars.wifolders_d[wi["folder"]]["name"], "entries": folder})
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
    ln = len(vars.worldinfo)
    stablesortwi()
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
    folder: Optional[list] = None
    if ln:
        last_folder = ...
        for wi in vars.worldinfo_i:
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
    return {"result": uid in vars.worldinfo_u and vars.worldinfo_u[uid]["init"]}


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
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
    return {"folders": [{"uid": folder, **{k: v for k, v in vars.wifolders_d[folder].items() if k != "collapsed"}} for folder in vars.wifolders_l]}


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
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
    return {"folders": vars.wifolders_l}


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
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
    for wi in reversed(vars.worldinfo_i):
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
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
    for wi in reversed(vars.worldinfo_i):
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
    return {"result": uid in vars.worldinfo_u and vars.worldinfo_u[uid]["folder"] is None and vars.worldinfo_u[uid]["init"]}


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
    if uid not in vars.wifolders_d:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info folder with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    entries = []
    stablesortwi()
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
    for wi in vars.wifolders_u[uid]:
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
    if uid not in vars.wifolders_d:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info folder with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    entries = []
    stablesortwi()
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
    for wi in vars.wifolders_u[uid]:
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
    return {"result": entry_uid in vars.worldinfo_u and vars.worldinfo_u[entry_uid]["folder"] == folder_uid and vars.worldinfo_u[entry_uid]["init"]}


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
    if uid not in vars.wifolders_d:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info folder with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": vars.wifolders_d[uid]["name"]}


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
    if uid not in vars.wifolders_d:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info folder with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    vars.wifolders_d[uid]["name"] = body.value
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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    wi = vars.worldinfo_u[uid]
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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": vars.worldinfo_u[uid]["comment"]}


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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    vars.worldinfo_u[uid]["comment"] = body.value
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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": vars.worldinfo_u[uid]["content"]}


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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    vars.worldinfo_u[uid]["content"] = body.value
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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": vars.worldinfo_u[uid]["key"]}


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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    vars.worldinfo_u[uid]["key"] = body.value
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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": vars.worldinfo_u[uid]["keysecondary"]}


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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    vars.worldinfo_u[uid]["keysecondary"] = body.value
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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": vars.worldinfo_u[uid]["selective"]}


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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    vars.worldinfo_u[uid]["selective"] = body.value
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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    return {"value": vars.worldinfo_u[uid]["constant"]}


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
    if uid not in vars.worldinfo_u:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info entry with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    vars.worldinfo_u[uid]["constant"] = body.value
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
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
    setgamesaved(False)
    emit('from_server', {'cmd': 'wiexpand', 'data': vars.worldinfo[-1]["num"]}, broadcast=True)
    vars.worldinfo[-1]["init"] = True
    addwiitem(folder_uid=None)
    return {"uid": vars.worldinfo[-2]["uid"]}


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
    if uid not in vars.wifolders_d:
        abort(Response(json.dumps({"detail": {
            "msg": "No world info folder with the given uid exists.",
            "type": "key_error",
        }}), mimetype="application/json", status=404))
    stablesortwi()
    vars.worldinfo_i = [wi for wi in vars.worldinfo if wi["init"]]
    setgamesaved(False)
    emit('from_server', {'cmd': 'wiexpand', 'data': vars.wifolders_u[uid][-1]["num"]}, broadcast=True)
    vars.wifolders_u[uid][-1]["init"] = True
    addwiitem(folder_uid=uid)
    return {"uid": vars.wifolders_u[uid][-2]["uid"]}


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
    if uid not in vars.worldinfo_u:
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
    return {"uid": vars.wifolders_l[-1]}


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
    if uid not in vars.wifolders_d:
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
        _obj = {"vars": vars, "vars.formatoptns": vars.formatoptns}[obj]
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
        _obj = {"vars": vars, "vars.formatoptns": vars.formatoptns}[obj]
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
    return {"value": vars.spfilename.strip()}

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
    for sp in fileops.getspfiles(vars.modeldim):

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
    if vars.allowsp:
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
    return {"value": __import__("tpu_mtj_backend").get_rng_seed() if vars.use_colab_tpu else __import__("torch").initial_seed()}

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
    if vars.use_colab_tpu:
        import tpu_mtj_backend
        tpu_mtj_backend.set_rng_seed(body.value)
    else:
        import torch
        torch.manual_seed(body.value)
    vars.seed = body.value
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
        obj = "vars"
        var_name = "memory"
        name = "memory"
        example_yaml_value = "Memory"

@config_endpoint_schema
class AuthorsNoteSettingSchema(KoboldSchema):
    value = fields.String(required=True)
    class KoboldMeta:
        route_name = "authors_note"
        obj = "vars"
        var_name = "authornote"
        name = "author's note"
        example_yaml_value = "''"

@config_endpoint_schema
class AuthorsNoteTemplateSettingSchema(KoboldSchema):
    value = fields.String(required=True)
    class KoboldMeta:
        route_name = "authors_note_template"
        obj = "vars"
        var_name = "authornotetemplate"
        name = "author's note template"
        example_yaml_value = "\"[Author's note: <|>]\""

@config_endpoint_schema
class TopKSamplingSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=0), required=True)
    class KoboldMeta:
        route_name = "top_k"
        obj = "vars"
        var_name = "top_k"
        name = "top-k sampling"
        example_yaml_value = "0"

@config_endpoint_schema
class TopASamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0), required=True)
    class KoboldMeta:
        route_name = "top_a"
        obj = "vars"
        var_name = "top_a"
        name = "top-a sampling"
        example_yaml_value = "0.0"

@config_endpoint_schema
class TopPSamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0, max=1), required=True)
    class KoboldMeta:
        route_name = "top_p"
        obj = "vars"
        var_name = "top_p"
        name = "top-p sampling"
        example_yaml_value = "0.9"

@config_endpoint_schema
class TailFreeSamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0, max=1), required=True)
    class KoboldMeta:
        route_name = "tfs"
        obj = "vars"
        var_name = "tfs"
        name = "tail free sampling"
        example_yaml_value = "1.0"

@config_endpoint_schema
class TypicalSamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0, max=1), required=True)
    class KoboldMeta:
        route_name = "typical"
        obj = "vars"
        var_name = "typical"
        name = "typical sampling"
        example_yaml_value = "1.0"

@config_endpoint_schema
class TemperatureSamplingSettingSchema(KoboldSchema):
    value = fields.Float(validate=validate.Range(min=0, min_inclusive=False), required=True)
    class KoboldMeta:
        route_name = "temperature"
        obj = "vars"
        var_name = "temp"
        name = "temperature"
        example_yaml_value = "0.5"

@config_endpoint_schema
class GensPerActionSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=0, max=5), required=True)
    class KoboldMeta:
        route_name = "n"
        obj = "vars"
        var_name = "numseqs"
        name = "Gens Per Action"
        example_yaml_value = "1"

@config_endpoint_schema
class MaxLengthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=1, max=512), required=True)
    class KoboldMeta:
        route_name = "max_length"
        obj = "vars"
        var_name = "genamt"
        name = "max length"
        example_yaml_value = "80"

@config_endpoint_schema
class WorldInfoDepthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=1, max=5), required=True)
    class KoboldMeta:
        route_name = "world_info_depth"
        obj = "vars"
        var_name = "widepth"
        name = "world info depth"
        example_yaml_value = "3"

@config_endpoint_schema
class AuthorsNoteDepthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=1, max=5), required=True)
    class KoboldMeta:
        route_name = "authors_note_depth"
        obj = "vars"
        var_name = "andepth"
        name = "author's note depth"
        example_yaml_value = "3"

@config_endpoint_schema
class MaxContextLengthSettingSchema(KoboldSchema):
    value = fields.Integer(validate=validate.Range(min=512, max=2048), required=True)
    class KoboldMeta:
        route_name = "max_context_length"
        obj = "vars"
        var_name = "max_length"
        name = "max context length"
        example_yaml_value = "2048"

@config_endpoint_schema
class TrimIncompleteSentencesSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "frmttriminc"
        obj = "vars.formatoptns"
        var_name = "@frmttriminc"
        name = "trim incomplete sentences (output formatting)"
        example_yaml_value = "false"

@config_endpoint_schema
class RemoveBlankLinesSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "frmtrmblln"
        obj = "vars.formatoptns"
        var_name = "@frmtrmblln"
        name = "remove blank lines (output formatting)"
        example_yaml_value = "false"

@config_endpoint_schema
class RemoveSpecialCharactersSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "frmtrmspch"
        obj = "vars.formatoptns"
        var_name = "@frmtrmspch"
        name = "remove special characters (output formatting)"
        example_yaml_value = "false"

@config_endpoint_schema
class SingleLineSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "singleline"
        obj = "vars.formatoptns"
        var_name = "@singleline"
        name = "single line (output formatting)"
        example_yaml_value = "false"

@config_endpoint_schema
class AddSentenceSpacingSettingsSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "frmtadsnsp"
        obj = "vars.formatoptns"
        var_name = "@frmtadsnsp"
        name = "add sentence spacing (input formatting)"
        example_yaml_value = "false"

@config_endpoint_schema
class SamplerOrderSettingSchema(KoboldSchema):
    value = fields.List(fields.Integer(), validate=[validate.Length(min=6), permutation_validator], required=True)
    class KoboldMeta:
        route_name = "sampler_order"
        obj = "vars"
        var_name = "sampler_order"
        name = "sampler order"
        example_yaml_value = "[6, 0, 1, 2, 3, 4, 5]"

@config_endpoint_schema
class SamplerFullDeterminismSettingSchema(KoboldSchema):
    value = fields.Boolean(required=True)
    class KoboldMeta:
        route_name = "sampler_full_determinism"
        obj = "vars"
        var_name = "full_determinism"
        name = "sampler full determinism"
        example_yaml_value = "false"


for schema in config_endpoint_schemas:
    create_config_endpoint(schema=schema.__name__, method="GET")
    create_config_endpoint(schema=schema.__name__, method="PUT")


#==================================================================#
#  Final startup commands to launch Flask app
#==================================================================#
if __name__ == "__main__":

    general_startup()
    # Start flask & SocketIO
    logger.init("Flask", status="Starting")
    Session(app)
    logger.init_ok("Flask", status="OK")
    logger.init("Webserver", status="Starting")
    patch_transformers()
    #show_select_model_list()
    if vars.model == "" or vars.model is None:
        vars.model = "ReadOnly"
    load_model(initial_load=True)

    # Start Flask/SocketIO (Blocking, so this must be last method!)
    port = args.port if "port" in args and args.port is not None else 5000
    
    #socketio.run(app, host='0.0.0.0', port=port)
    if(vars.host):
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
                logger.init_ok("Webserver", status="OK")
                logger.message(f"KoboldAI has finished loading and is available at the following link: {cloudflare}")
        else:
            logger.init_ok("Webserver", status="OK")
            logger.message(f"Webserver has started, you can now connect to this machine at port: {port}")
        vars.serverstarted = True
        socketio.run(app, host='0.0.0.0', port=port)
    else:
        if args.unblock:
            if not args.no_ui:
                try:
                    import webbrowser
                    webbrowser.open_new('http://localhost:{0}'.format(port))
                except:
                    pass
            logger.init_ok("Webserver", status="OK")
            logger.message(f"Webserver started! You may now connect with a browser at http://127.0.0.1:{port}")
            vars.serverstarted = True
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
            vars.serverstarted = True
            socketio.run(app, port=port)
    logger.init("Webserver", status="Closed")


else:
    general_startup()
    # Start flask & SocketIO
    logger.init("Flask", status="Starting")
    Session(app)
    logger.init_ok("Flask", status="OK")
    patch_transformers()
    #show_select_model_list()
    if vars.model == "" or vars.model is None:
        vars.model = "ReadOnly"
    load_model(initial_load=True)
    print("{0}\nServer started in WSGI mode!{1}".format(colors.GREEN, colors.END), flush=True)
