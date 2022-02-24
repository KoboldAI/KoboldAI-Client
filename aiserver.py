#!/usr/bin/python3
#==================================================================#
# KoboldAI
# Version: 1.17.0
# By: KoboldAIDev and the KoboldAI Community
#==================================================================#

# External packages
import eventlet
eventlet.monkey_patch(all=True, thread=False)
import os
os.system("")
os.environ['EVENTLET_THREADPOOL_SIZE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from eventlet import tpool

from os import path, getcwd
import time
import re
import json
import collections
import zipfile
import packaging
import contextlib
import traceback
import threading
import markdown
import bleach
from collections.abc import Iterable
from typing import Any, Callable, TypeVar, Tuple, Union, Dict, Set, List

import requests
import html
import argparse
import sys
import gc

import lupa

# KoboldAI
import fileops
import gensettings
from utils import debounce
import utils
import structures


if lupa.LUA_VERSION[:2] != (5, 4):
    print(f"Please install lupa==1.10. You have lupa {lupa.__version__}.", file=sys.stderr)


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

# AI models
modellist = [
    ["Load a model from its directory", "NeoCustom", ""],
    ["Load an old GPT-2 model (eg CloverEdition)", "GPT2Custom", ""],
    ["Skein 6B (Hybrid)", "KoboldAI/GPT-J-6B-Skein", "16GB"],
    ["Adventure 6B", "KoboldAI/GPT-J-6B-Adventure", "16GB"],
    ["Lit 6B (NSFW)", "hakurei/lit-6B", "16GB"],
    ["C1 6B (Chatbot)", "hakurei/c1-6B", "16GB"],
    ["Janeway 2.7B (Novel)", "KoboldAI/GPT-Neo-2.7B-Janeway", "8GB"],
    ["Adventure 2.7B", "KoboldAI/GPT-Neo-2.7B-AID", "8GB"],
    ["Picard 2.7B (Novel)", "KoboldAI/GPT-Neo-2.7B-Picard", "8GB"],
    ["Horni 2.7B (NSFW)", "KoboldAI/GPT-Neo-2.7B-Horni", "8GB"],
    ["Horni-LN 2.7B (Novel)", "KoboldAI/GPT-Neo-2.7B-Horni-LN", "8GB"],
    ["Shinen 2.7B (NSFW)", "KoboldAI/GPT-Neo-2.7B-Shinen", "8GB"],
    ["GPT-J 6B", "EleutherAI/gpt-j-6B", "16GB"],
    ["GPT-Neo 2.7B", "EleutherAI/gpt-neo-2.7B", "8GB"],
    ["GPT-Neo 1.3B", "EleutherAI/gpt-neo-1.3B", "6GB"],
    ["GPT-2 XL", "gpt2-xl", "6GB"],
    ["GPT-2 Large", "gpt2-large", "4GB"],
    ["GPT-2 Med", "gpt2-medium", "2GB"],
    ["GPT-2", "gpt2", "2GB"],
    ["OpenAI API (requires API key)", "OAI", ""],
    ["InferKit API (requires API key)", "InferKit", ""],
    ["KoboldAI Server API (Old Google Colab)", "Colab", ""],
    ["Read Only (No AI)", "ReadOnly", ""]
    ]

# Variables
class vars:
    lastact     = ""     # The last action received from the user
    submission  = ""     # Same as above, but after applying input formatting
    lastctx     = ""     # The last context submitted to the generator
    model       = ""     # Model ID string chosen at startup
    model_type  = ""     # Model Type (Automatically taken from the model config)
    noai        = False  # Runs the script without starting up the transformers pipeline
    aibusy      = False  # Stops submissions while the AI is working
    max_length  = 1024    # Maximum number of tokens to submit per action
    ikmax       = 3000   # Maximum number of characters to submit to InferKit
    genamt      = 80     # Amount of text for each action to generate
    ikgen       = 200    # Number of characters for InferKit to generate
    rep_pen     = 1.1    # Default generator repetition_penalty
    rep_pen_slope = 1.0  # Default generator repetition penalty slope
    rep_pen_range = 1024 # Default generator repetition penalty range
    temp        = 0.5    # Default generator temperature
    top_p       = 0.9    # Default generator top_p
    top_k       = 0      # Default generator top_k
    tfs         = 1.0    # Default generator tfs (tail-free sampling)
    numseqs     = 1     # Number of sequences to ask the generator to create
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
    actions_metadata = [] # List of dictonaries, one dictonary for every action that contains information about the action like alternative options.
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
    spfilename  = ""     # Filename of soft prompt to load, or an empty string if not using a soft prompt
    userscripts = []     # List of userscripts to load
    last_userscripts = []  # List of previous userscript filenames from the previous time userscripts were send via usstatitems
    corescript  = "default.lua"  # Filename of corescript to load
    # badwords    = []     # Array of str/chr values that should be removed from output
    badwordsids = [[13460], [6880], [50256], [42496], [4613], [17414], [22039], [16410], [27], [29], [38430], [37922], [15913], [24618], [28725], [58], [47175], [36937], [26700], [12878], [16471], [37981], [5218], [29795], [13412], [45160], [3693], [49778], [4211], [20598], [36475], [33409], [44167], [32406], [29847], [29342], [42669], [685], [25787], [7359], [3784], [5320], [33994], [33490], [34516], [43734], [17635], [24293], [9959], [23785], [21737], [28401], [18161], [26358], [32509], [1279], [38155], [18189], [26894], [6927], [14610], [23834], [11037], [14631], [26933], [46904], [22330], [25915], [47934], [38214], [1875], [14692], [41832], [13163], [25970], [29565], [44926], [19841], [37250], [49029], [9609], [44438], [16791], [17816], [30109], [41888], [47527], [42924], [23984], [49074], [33717], [31161], [49082], [30138], [31175], [12240], [14804], [7131], [26076], [33250], [3556], [38381], [36338], [32756], [46581], [17912], [49146]] # Tokenized array of badwords used to prevent AI artifacting
    deletewi    = None   # Temporary storage for UID to delete
    wirmvwhtsp  = False  # Whether to remove leading whitespace from WI entries
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
    savedir     = getcwd()+"\stories"
    hascuda     = False  # Whether torch has detected CUDA on the system
    usegpu      = False  # Whether to launch pipeline with GPU support
    custmodpth  = ""     # Filesystem location of custom model to run
    formatoptns = {'frmttriminc': True, 'frmtrmblln': False, 'frmtrmspch': False, 'frmtadsnsp': False, 'singleline': False}     # Container for state of formatting options
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
    bmsupported = False  # Whether the breakmodel option is supported (GPT-Neo/GPT-J/XGLM only, currently)
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
    newlinemode = "n"
    quiet       = False # If set will suppress any story text from being printed to the console (will only be seen on the client web page)
    debug       = False # If set to true, will send debug information to the client for display

utils.vars = vars

#==================================================================#
# Function to get model selection at startup
#==================================================================#
def getModelSelection():
    print("    #   Model                           VRAM\n    =========================================")
    i = 1
    for m in modellist:
        print("    {0} - {1}\t\t{2}".format("{:<2}".format(i), m[0].ljust(15), m[2]))
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
            getModelSelection()

#==================================================================#
# Return all keys in tokenizer dictionary containing char
#==================================================================#
def gettokenids(char):
    keys = []
    for key in vocab_keys:
        if(key.find(char) != -1):
            keys.append(key)
    return keys

#==================================================================#
# Return Model Name
#==================================================================#
def getmodelname():
    if(args.configname):
       modelname = args.configname
       return modelname
    if(vars.model in ("NeoCustom", "GPT2Custom", "TPUMeshTransformerGPTJ")):
        modelname = os.path.basename(os.path.normpath(vars.custmodpth))
        return modelname
    else:
        modelname = vars.model
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
    print(f"{row_color}   {' '*9} N/A  {sep_color}|{row_color}     {n_layers:3}  {sep_color}|{row_color}  (CPU){colors.END}")

def device_config(model):
    global breakmodel, generator
    import breakmodel
    n_layers = model.config.num_layers if hasattr(model.config, "num_layers") else model.config.n_layer
    if(args.breakmodel_gpulayers is not None):
        try:
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
    
    print(colors.PURPLE + "\nFinal device configuration:")
    device_list(n_layers)

    # If all layers are on the same device, use the old GPU generation mode
    while(len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] == 0):
        breakmodel.gpu_blocks.pop()
    if(len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] in (-1, model.config.num_layers if hasattr(model.config, "num_layers") else model.config.n_layer)):
        vars.breakmodel = False
        vars.usegpu = True
        vars.gpu_device = len(breakmodel.gpu_blocks)-1
        model = model.half().to(vars.gpu_device)
        generator = model.generate
        return

    if(not breakmodel.gpu_blocks):
        print("Nothing assigned to a GPU, reverting to CPU only mode")
        vars.breakmodel = False
        vars.usegpu = False
        model = model.to('cpu').float()
        generator = model.generate
        return
    model.half().to('cpu')
    gc.collect()
    if(hasattr(model, "transformer")):
        model.transformer.wte.to(breakmodel.primary_device)
        model.transformer.ln_f.to(breakmodel.primary_device)
        if(hasattr(model, 'lm_head')):
            model.lm_head.to(breakmodel.primary_device)
        if(hasattr(model.transformer, 'wpe')):
            model.transformer.wpe.to(breakmodel.primary_device)
    else:
        model.model.embed_tokens.to(breakmodel.primary_device)
        model.model.layer_norm.to(breakmodel.primary_device)
        model.lm_head.to(breakmodel.primary_device)
        model.model.embed_positions.to(breakmodel.primary_device)
    gc.collect()
    GPTNeoModel.forward = breakmodel.new_forward_neo
    if("GPTJModel" in globals()):
        GPTJModel.forward = breakmodel.new_forward_neo
    if("XGLMModel" in globals()):
        XGLMModel.forward = breakmodel.new_forward_xglm
    generator = model.generate
    if(hasattr(model, "transformer")):
        breakmodel.move_hidden_layers(model.transformer)
    else:
        breakmodel.move_hidden_layers(model.model, model.model.layers)

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
    vars.modelconfig = js
    if("badwordsids" in js):
        vars.badwordsids = js["badwordsids"]
    if("nobreakmodel" in js):
        vars.nobreakmodel = js["nobreakmodel"]
    if("temp" in js):
        vars.temp       = js["temp"]
    if("top_p" in js):
        vars.top_p      = js["top_p"]
    if("top_k" in js):
        vars.top_k      = js["top_k"]
    if("tfs" in js):
        vars.tfs        = js["tfs"]
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
    js["temp"]        = vars.temp
    js["top_p"]       = vars.top_p
    js["top_k"]       = vars.top_k
    js["tfs"]         = vars.tfs
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
    js["autosave"]    = vars.autosave
    js["welcome"]     = vars.welcome
    js["newlinemode"] = vars.newlinemode

    js["antemplate"]  = vars.setauthornotetemplate

    js["userscripts"] = vars.userscripts
    js["corescript"]  = vars.corescript
    js["softprompt"]  = vars.spfilename

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
    if(path.exists("settings/" + getmodelname().replace('/', '_') + ".settings")):
        # Read file contents into JSON object
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        js   = json.load(file)
        
        # Copy file contents to vars
        if("apikey" in js):
            vars.apikey     = js["apikey"]
        if("andepth" in js):
            vars.andepth    = js["andepth"]
        if("temp" in js):
            vars.temp       = js["temp"]
        if("top_p" in js):
            vars.top_p      = js["top_p"]
        if("top_k" in js):
            vars.top_k      = js["top_k"]
        if("tfs" in js):
            vars.tfs        = js["tfs"]
        if("rep_pen" in js):
            vars.rep_pen    = js["rep_pen"]
        if("rep_pen_slope" in js):
            vars.rep_pen_slope = js["rep_pen_slope"]
        if("rep_pen_range" in js):
            vars.rep_pen_range = js["rep_pen_range"]
        if("genamt" in js):
            vars.genamt     = js["genamt"]
        if("max_length" in js):
            vars.max_length = js["max_length"]
        if("ikgen" in js):
            vars.ikgen      = js["ikgen"]
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
        if("autosave" in js):
            vars.autosave = js["autosave"]
        if("newlinemode" in js):
            vars.newlinemode = js["newlinemode"]
        if("welcome" in js):
            vars.welcome = js["welcome"]

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

        file.close()

#==================================================================#
#  Load a soft prompt from a file
#==================================================================#
def spRequest(filename):
    vars.spfilename = ""
    settingschanged()

    if(len(filename) == 0):
        vars.sp = None
        vars.sp_length = 0
        return

    global np
    if 'np' not in globals():
        import numpy as np

    z, version, shape, fortran_order, dtype = fileops.checksp(filename, vars.modeldim)
    assert isinstance(z, zipfile.ZipFile)
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

    if(vars.model in ("TPUMeshTransformerGPTJ",)):
        rows = tensor.shape[0]
        padding_amount = tpu_mtj_backend.params["seq"] - (tpu_mtj_backend.params["seq"] % -tpu_mtj_backend.params["cores_per_replica"]) - rows
        tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
        tensor = tensor.reshape(
            tpu_mtj_backend.params["cores_per_replica"],
            -1,
            tpu_mtj_backend.params["d_model"],
        )
        vars.sp = tpu_mtj_backend.shard_xmap(np.float32(tensor))
    else:
        vars.sp = torch.from_numpy(tensor)

    vars.spfilename = filename
    settingschanged()

#==================================================================#
# Startup
#==================================================================#

# Parsing Parameters
parser = argparse.ArgumentParser(description="KoboldAI Server")
parser.add_argument("--remote", action='store_true', help="Optimizes KoboldAI for Remote Play")
parser.add_argument("--ngrok", action='store_true', help="Optimizes KoboldAI for Remote Play using Ngrok")
parser.add_argument("--host", action='store_true', help="Optimizes KoboldAI for Remote Play without using a proxy service")
parser.add_argument("--model", help="Specify the Model Type to skip the Menu")
parser.add_argument("--path", help="Specify the Path for local models (For model NeoCustom or GPT2Custom)")
parser.add_argument("--cpu", action='store_true', help="By default unattended launches are on the GPU use this option to force CPU usage.")
parser.add_argument("--breakmodel", action='store_true', help=argparse.SUPPRESS)
parser.add_argument("--breakmodel_layers", type=int, help=argparse.SUPPRESS)
parser.add_argument("--breakmodel_gpulayers", type=str, help="If using a model that supports hybrid generation, this is a comma-separated list that specifies how many layers to put on each GPU device. For example to put 8 layers on device 0, 9 layers on device 1 and 11 layers on device 2, use --beakmodel_gpulayers 8,9,11")
parser.add_argument("--override_delete", action='store_true', help="Deleting stories from inside the browser is disabled if you are using --remote and enabled otherwise. Using this option will instead allow deleting stories if using --remote and prevent deleting stories otherwise.")
parser.add_argument("--override_rename", action='store_true', help="Renaming stories from inside the browser is disabled if you are using --remote and enabled otherwise. Using this option will instead allow renaming stories if using --remote and prevent renaming stories otherwise.")
parser.add_argument("--configname", help="Force a fixed configuration name to aid with config management.")
parser.add_argument("--colab", action='store_true', help="Optimize for Google Colab.")
parser.add_argument("--nobreakmodel", action='store_true', help="Disables Breakmodel support completely.")
parser.add_argument("--unblock", action='store_true', default=False, help="Unblocks the KoboldAI port to be accessible from other machines without optimizing for remote play (It is recommended to use --host instead)")
parser.add_argument("--quiet", action='store_true', default=False, help="If present will suppress any story related text from showing on the console")
parser.add_argument("--lowmem", action='store_true', help="Extra Low Memory loading for the GPU, slower but memory does not peak to twice the usage")

args: argparse.Namespace = None
if(os.environ.get("KOBOLDAI_ARGS") is not None):
    import shlex
    args = parser.parse_args(shlex.split(os.environ["KOBOLDAI_ARGS"]))
else:
    args = parser.parse_args()

vars.model = args.model;

if args.colab:
    args.remote = True;
    args.override_rename = True;
    args.override_delete = True;
    args.nobreakmodel = True;
    args.quiet = True;
    args.lowmem = True;

if args.quiet:
    vars.quiet = True

if args.nobreakmodel:
    vars.nobreakmodel = True;

if args.remote:
    vars.host = True;

if args.ngrok:
    vars.host = True;

if args.host:
    vars.host = True;

vars.smandelete = vars.host == args.override_delete
vars.smanrename = vars.host == args.override_rename

# Select a model to run
if args.model:
    print("Welcome to KoboldAI!\nYou have selected the following Model:", vars.model)
    if args.path:
        print("You have selected the following path for your Model :", args.path)
        vars.custmodpth = args.path;
        vars.colaburl = args.path + "/request"; # Lets just use the same parameter to keep it simple

else:
    print("{0}Welcome to the KoboldAI Server!\nListed RAM is the optimal VRAM and CPU ram can be up to twice the amount.\nMost models can run at less VRAM with reduced max tokens or less layers on the GPU.\nSelect an AI model to continue:{1}\n".format(colors.CYAN, colors.END))
    getModelSelection()

# If transformers model was selected & GPU available, ask to use CPU or GPU
if(not vars.model in ["InferKit", "Colab", "OAI", "ReadOnly", "TPUMeshTransformerGPTJ"]):
    vars.allowsp = True
    # Test for GPU support
    import torch
    
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
            model_config = AutoConfig.from_pretrained(vars.custmodpth.replace('/', '_'), cache_dir="cache/")
            vars.model_type = model_config.model_type
        except ValueError as e:
            vars.model_type = "not_found"
    elif(os.path.isdir("models/{}".format(vars.custmodpth.replace('/', '_')))):
        try:
            model_config = AutoConfig.from_pretrained("models/{}".format(vars.custmodpth.replace('/', '_')), cache_dir="cache/")
            vars.model_type = model_config.model_type
        except ValueError as e:
            vars.model_type = "not_found"
    else:
        try:
            model_config = AutoConfig.from_pretrained(vars.custmodpth, cache_dir="cache/")
            vars.model_type = model_config.model_type
        except ValueError as e:
            vars.model_type = "not_found"
    if(vars.model_type == "not_found" and vars.model == "NeoCustom"):
        vars.model_type = "gpt_neo"
    elif(vars.model_type == "not_found" and vars.model == "GPT2Custom"):
        vars.model_type = "gpt2"
    elif(vars.model_type == "not_found"):
        print("WARNING: No model type detected, assuming Neo (If this is a GPT2 model use the other menu option or --model GPT2Custom)")
        vars.model_type = "gpt_neo"
    loadmodelsettings()
    loadsettings()
    print("{0}Looking for GPU support...{1}".format(colors.PURPLE, colors.END), end="")
    vars.hascuda = torch.cuda.is_available()
    vars.bmsupported = vars.model_type in ("gpt_neo", "gptj", "xglm") and not vars.nobreakmodel
    if(args.breakmodel is not None and args.breakmodel):
        print("WARNING: --breakmodel is no longer supported. Breakmodel mode is now automatically enabled when --breakmodel_gpulayers is used (see --help for details).", file=sys.stderr)
    if(args.breakmodel_layers is not None):
        print("WARNING: --breakmodel_layers is deprecated. Use --breakmodel_gpulayers instead (see --help for details).", file=sys.stderr)
    if(args.model and vars.bmsupported and not args.breakmodel_gpulayers and not args.breakmodel_layers):
        print("WARNING: Model launched without the --breakmodel_gpulayers argument, defaulting to GPU only mode.", file=sys.stderr)
        vars.bmsupported = False
    if(not vars.bmsupported and (args.breakmodel_gpulayers is not None or args.breakmodel_layers is not None)):
        print("WARNING: This model does not support hybrid generation. --breakmodel_gpulayers will be ignored.", file=sys.stderr)
    if(vars.hascuda):
        print("{0}FOUND!{1}".format(colors.GREEN, colors.END))
    else:
        print("{0}NOT FOUND!{1}".format(colors.YELLOW, colors.END))
    
    if args.model:
        if(vars.hascuda):
            genselected = True
            vars.usegpu = True
            vars.breakmodel = False
        if(vars.bmsupported):
            vars.usegpu = False
            vars.breakmodel = True
        if(args.cpu):
            vars.usegpu = False
            vars.breakmodel = False
    elif(vars.hascuda):    
        if(vars.bmsupported):
            genselected = True
            vars.usegpu = False
            vars.breakmodel = True
        else:
            print("    1 - GPU\n    2 - CPU\n")
            genselected = False
    else:
        genselected = False

    if(vars.hascuda):
        while(genselected == False):
            genselect = input("Mode> ")
            if(genselect == ""):
                vars.breakmodel = False
                vars.usegpu = True
                genselected = True
            elif(genselect.isnumeric() and int(genselect) == 1):
                if(vars.bmsupported):
                    vars.breakmodel = True
                    vars.usegpu = False
                    genselected = True
                else:
                    vars.breakmodel = False
                    vars.usegpu = True
                    genselected = True
            elif(genselect.isnumeric() and int(genselect) == 2):
                vars.breakmodel = False
                vars.usegpu = False
                genselected = True
            else:
                print("{0}Please enter a valid selection.{1}".format(colors.RED, colors.END))

# Ask for API key if InferKit was selected
if(vars.model == "InferKit"):
    if(not path.exists("settings/" + getmodelname().replace('/', '_') + ".settings")):
        # If the client settings file doesn't exist, create it
        print("{0}Please enter your InferKit API key:{1}\n".format(colors.CYAN, colors.END))
        vars.apikey = input("Key> ")
        # Write API key to file
        os.makedirs('settings', exist_ok=True)
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
        try:
            js = {"apikey": vars.apikey}
            file.write(json.dumps(js, indent=3))
        finally:
            file.close()
    else:
        # Otherwise open it up
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        # Check if API key exists
        js = json.load(file)
        if("apikey" in js and js["apikey"] != ""):
            # API key exists, grab it and close the file
            vars.apikey = js["apikey"]
            file.close()
        else:
            # Get API key, add it to settings object, and write it to disk
            print("{0}Please enter your InferKit API key:{1}\n".format(colors.CYAN, colors.END))
            vars.apikey = input("Key> ")
            js["apikey"] = vars.apikey
            # Write API key to file
            file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
            try:
                file.write(json.dumps(js, indent=3))
            finally:
                file.close()

# Ask for API key if OpenAI was selected
if(vars.model == "OAI"):
    if(not path.exists("settings/" + getmodelname().replace('/', '_') + ".settings")):
        # If the client settings file doesn't exist, create it
        print("{0}Please enter your OpenAI API key:{1}\n".format(colors.CYAN, colors.END))
        vars.oaiapikey = input("Key> ")
        # Write API key to file
        os.makedirs('settings', exist_ok=True)
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
        try:
            js = {"oaiapikey": vars.oaiapikey}
            file.write(json.dumps(js, indent=3))
        finally:
            file.close()
    else:
        # Otherwise open it up
        file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
        # Check if API key exists
        js = json.load(file)
        if("oaiapikey" in js and js["oaiapikey"] != ""):
            # API key exists, grab it and close the file
            vars.oaiapikey = js["oaiapikey"]
            file.close()
        else:
            # Get API key, add it to settings object, and write it to disk
            print("{0}Please enter your OpenAI API key:{1}\n".format(colors.CYAN, colors.END))
            vars.oaiapikey = input("Key> ")
            js["oaiapikey"] = vars.oaiapikey
            # Write API key to file
            file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
            try:
                file.write(json.dumps(js, indent=3))
            finally:
                file.close()
    
    # Get list of models from OAI
    print("{0}Retrieving engine list...{1}".format(colors.PURPLE, colors.END), end="")
    req = requests.get(
        vars.oaiengines, 
        headers = {
            'Authorization': 'Bearer '+vars.oaiapikey
            }
        )
    if(req.status_code == 200):
        print("{0}OK!{1}".format(colors.GREEN, colors.END))
        print("{0}Please select an engine to use:{1}\n".format(colors.CYAN, colors.END))
        engines = req.json()["data"]
        # Print list of engines
        i = 0
        for en in engines:
            print("    {0} - {1} ({2})".format(i, en["id"], "\033[92mready\033[0m" if en["ready"] == True else "\033[91mnot ready\033[0m"))
            i += 1
        # Get engine to use
        print("")
        engselected = False
        while(engselected == False):
            engine = input("Engine #> ")
            if(engine.isnumeric() and int(engine) < len(engines)):
                vars.oaiurl = "https://api.openai.com/v1/engines/{0}/completions".format(engines[int(engine)]["id"])
                engselected = True
            else:
                print("{0}Please enter a valid selection.{1}".format(colors.RED, colors.END))
    else:
        # Something went wrong, print the message and quit since we can't initialize an engine
        print("{0}ERROR!{1}".format(colors.RED, colors.END))
        print(req.json())
        quit()

# Ask for ngrok url if Google Colab was selected
if(vars.model == "Colab"):
    if(vars.colaburl == ""):
        print("{0}NOTE: For the modern KoboldAI Colab's you open the links directly in your browser.\nThis option is only for the KoboldAI Server API, not all features are supported in this mode.\n".format(colors.YELLOW, colors.END))
        print("{0}Enter the URL of the server (For example a trycloudflare link):{1}\n".format(colors.CYAN, colors.END))
        vars.colaburl = input("URL> ") + "/request"

if(vars.model == "ReadOnly"):
    vars.noai = True

# Set logging level to reduce chatter from Flask
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Start flask & SocketIO
print("{0}Initializing Flask... {1}".format(colors.PURPLE, colors.END), end="")
from flask import Flask, render_template, Response, request, copy_current_request_context
from flask_socketio import SocketIO, emit
app = Flask(__name__)
app.config['SECRET KEY'] = 'secret!'
socketio = SocketIO(app, async_method="eventlet")
print("{0}OK!{1}".format(colors.GREEN, colors.END))

# Start transformers and create pipeline
if(not vars.model in ["InferKit", "Colab", "OAI", "ReadOnly", "TPUMeshTransformerGPTJ"]):
    if(not vars.noai):
        print("{0}Initializing transformers, please wait...{1}".format(colors.PURPLE, colors.END))
        from transformers import StoppingCriteria, GPT2TokenizerFast, GPT2LMHeadModel, GPTNeoForCausalLM, GPTNeoModel, AutoModelForCausalLM, AutoTokenizer
        for m in ("GPTJModel", "XGLMModel"):
            try:
                globals()[m] = getattr(__import__("transformers"), m)
            except:
                pass
        import transformers.generation_utils
        from transformers import __version__ as transformers_version

        # Temporary fix for XGLM positional embedding issues until
        # https://github.com/huggingface/transformers/issues/15736
        # is resolved
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

        # Patch transformers to use our soft prompt
        def patch_causallm(cls):
            old_forward = cls.forward
            def new_causallm_forward(self, *args, **kwargs):
                input_ids = kwargs.get('input_ids').to(self.device)
                assert input_ids is not None
                kwargs['input_ids'] = None
                if(vars.sp is not None):
                    shifted_input_ids = input_ids - self.config.vocab_size
                input_ids.clamp_(max=self.config.vocab_size-1)
                if(hasattr(self, "transformer")):
                    inputs_embeds = self.transformer.wte(input_ids)
                else:
                    inputs_embeds = self.model.embed_tokens(input_ids) * self.model.embed_scale
                if(vars.sp is not None):
                    vars.sp = vars.sp.to(inputs_embeds.dtype).to(inputs_embeds.device)
                    inputs_embeds = torch.where(
                        (shifted_input_ids >= 0)[..., None],
                        vars.sp[shifted_input_ids.clamp(min=0)],
                        inputs_embeds,
                    )
                kwargs['inputs_embeds'] = inputs_embeds
                return old_forward(self, *args, **kwargs)
            cls.forward = new_causallm_forward
        for cls in (GPT2LMHeadModel, GPTNeoForCausalLM):
            patch_causallm(cls)
        for c in ("GPTJForCausalLM", "XGLMForCausalLM"):
            try:
                patch_causallm(getattr(__import__("transformers"), c))
            except:
                pass


        # Patch transformers to use our custom logit warpers
        from transformers import LogitsProcessorList, LogitsWarper, LogitsProcessor, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper, RepetitionPenaltyLogitsProcessor
        from warpers import AdvancedRepetitionPenaltyLogitsProcessor, TailFreeLogitsWarper

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
        dynamic_processor_wrap(TopPLogitsWarper, "top_p", "top_p", cond=lambda x: x < 1.0)
        dynamic_processor_wrap(TailFreeLogitsWarper, "tfs", "tfs", cond=lambda x: x < 1.0)
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
        
        def new_get_logits_processor(*args, **kwargs) -> LogitsProcessorList:
            processors = new_get_logits_processor.old_get_logits_processor(*args, **kwargs)
            processors.insert(0, LuaLogitsProcessor())
            return processors
        new_get_logits_processor.old_get_logits_processor = transformers.generation_utils.GenerationMixin._get_logits_processor
        transformers.generation_utils.GenerationMixin._get_logits_processor = new_get_logits_processor

        def new_get_logits_warper(beams: int = 1,) -> LogitsProcessorList:
            warper_list = LogitsProcessorList()
            warper_list.append(TopKLogitsWarper(top_k=1, min_tokens_to_keep=1 + (beams > 1)))
            warper_list.append(TopPLogitsWarper(top_p=0.5, min_tokens_to_keep=1 + (beams > 1)))
            warper_list.append(TailFreeLogitsWarper(tfs=0.5, min_tokens_to_keep=1 + (beams > 1)))
            warper_list.append(TemperatureLogitsWarper(temperature=0.5))
            return warper_list
        
        def new_sample(self, *args, **kwargs):
            assert kwargs.pop("logits_warper", None) is not None
            kwargs["logits_warper"] = new_get_logits_warper(
                beams=1,
            )
            if(vars.newlinemode == "s"):
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
                vars.generated_tkns += 1
                if(vars.lua_koboldbridge.generated_cols and vars.generated_tkns != vars.lua_koboldbridge.generated_cols):
                    raise RuntimeError(f"Inconsistency detected between KoboldAI Python and Lua backends ({vars.generated_tkns} != {vars.lua_koboldbridge.generated_cols})")
                if(vars.abort or vars.generated_tkns >= vars.genamt):
                    self.regeneration_required = False
                    self.halt = False
                    return True

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
            stopping_criteria.insert(0, self.kai_scanner)
            return stopping_criteria
        transformers.generation_utils.GenerationMixin._get_stopping_criteria = new_get_stopping_criteria

        def get_hidden_size_from_model(model):
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
            if(always_use or (vars.hascuda and args.lowmem and (vars.usegpu or vars.breakmodel))):
                original_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float16)
                yield True
                torch.set_default_dtype(original_dtype)
            else:
                yield False

        # If custom GPT2 model was chosen
        if(vars.model == "GPT2Custom"):
            model_config = open(vars.custmodpth + "/config.json", "r")
            js   = json.load(model_config)
            with(maybe_use_float16()):
                model = GPT2LMHeadModel.from_pretrained(vars.custmodpth, cache_dir="cache/")
            tokenizer = GPT2TokenizerFast.from_pretrained(vars.custmodpth, cache_dir="cache/")
            vars.modeldim = get_hidden_size_from_model(model)
            # Is CUDA available? If so, use GPU, otherwise fall back to CPU
            if(vars.hascuda and vars.usegpu):
                model = model.half().to(vars.gpu_device)
                generator = model.generate
            else:
                model = model.to('cpu').float()
                generator = model.generate
        # Use the Generic implementation
        else:
            lowmem = maybe_low_cpu_mem_usage()
            # We must disable low_cpu_mem_usage (by setting lowmem to {}) if
            # using a GPT-2 model because GPT-2 is not compatible with this
            # feature yet
            if(vars.model_type == "gpt2"):
                lowmem = {}

            # Download model from Huggingface if it does not exist, otherwise load locally
            
            #If we specify a model and it's in the root directory, we need to move it to the models directory (legacy folder structure to new)
            if os.path.isdir(vars.model.replace('/', '_')):
                import shutil
                shutil.move(vars.model.replace('/', '_'), "models/{}".format(vars.model.replace('/', '_')))
            if(os.path.isdir(vars.custmodpth)):
               with(maybe_use_float16()):
                   try:
                       tokenizer = AutoTokenizer.from_pretrained(vars.custmodpth, cache_dir="cache")
                   except ValueError as e:
                       tokenizer = GPT2TokenizerFast.from_pretrained(vars.custmodpth, cache_dir="cache")
                   try:
                       model     = AutoModelForCausalLM.from_pretrained(vars.custmodpth, cache_dir="cache", **lowmem)
                   except ValueError as e:
                       model     = GPTNeoForCausalLM.from_pretrained(vars.custmodpth, cache_dir="cache", **lowmem)
            elif(os.path.isdir("models/{}".format(vars.model.replace('/', '_')))):
                with(maybe_use_float16()):
                   try:
                       tokenizer = AutoTokenizer.from_pretrained("models/{}".format(vars.model.replace('/', '_')), cache_dir="cache")
                   except ValueError as e:
                       tokenizer = GPT2TokenizerFast.from_pretrained("models/{}".format(vars.model.replace('/', '_')), cache_dir="cache")
                   try:
                       model     = AutoModelForCausalLM.from_pretrained("models/{}".format(vars.model.replace('/', '_')), cache_dir="cache", **lowmem)
                   except ValueError as e:
                       model     = GPTNeoForCausalLM.from_pretrained("models/{}".format(vars.model.replace('/', '_')), cache_dir="cache", **lowmem)
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(vars.model, cache_dir="cache")
                except ValueError as e:
                    tokenizer = GPT2TokenizerFast.from_pretrained(vars.model, cache_dir="cache")
                with(maybe_use_float16()):
                    try:
                        model     = AutoModelForCausalLM.from_pretrained(vars.model, cache_dir="cache", **lowmem)
                    except ValueError as e:
                        model     = GPTNeoForCausalLM.from_pretrained(vars.model, cache_dir="cache", **lowmem)
                
                if not args.colab:
                    import shutil
                    model = model.half()
                    model.save_pretrained("models/{}".format(vars.model.replace('/', '_')))
                    tokenizer.save_pretrained("models/{}".format(vars.model.replace('/', '_')))
                    shutil.rmtree("cache/")
            
            if(vars.hascuda):
                if(vars.usegpu):
                    vars.modeldim = get_hidden_size_from_model(model)
                    model = model.half().to(vars.gpu_device)
                    generator = model.generate
                elif(vars.breakmodel):  # Use both RAM and VRAM (breakmodel)
                    vars.modeldim = get_hidden_size_from_model(model)
                    device_config(model)
                else:
                    model = model.to('cpu').float()
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
		
        print("{0}OK! {1} pipeline created!{2}".format(colors.GREEN, vars.model, colors.END))
    
    else:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="cache/")
else:
    def tpumtjgetsofttokens():
        soft_tokens = None
        if(vars.sp is None):
            global np
            if 'np' not in globals():
                import numpy as np
            tensor = np.zeros((1, tpu_mtj_backend.params["d_model"]), dtype=np.float32)
            rows = tensor.shape[0]
            padding_amount = tpu_mtj_backend.params["seq"] - (tpu_mtj_backend.params["seq"] % -tpu_mtj_backend.params["cores_per_replica"]) - rows
            tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
            tensor = tensor.reshape(
                tpu_mtj_backend.params["cores_per_replica"],
                -1,
                tpu_mtj_backend.params["d_model"],
            )
            vars.sp = tpu_mtj_backend.shard_xmap(tensor)
        soft_tokens = np.arange(
            tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"],
            tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"] + vars.sp_length,
            dtype=np.uint32
        )
        return soft_tokens

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
        return {
            "top_p": float(vars.top_p),
            "temp": float(vars.temp),
            "top_k": int(vars.top_k),
            "tfs": float(vars.tfs),
            "repetition_penalty": float(vars.rep_pen),
            "rpslope": float(vars.rep_pen_slope),
            "rprange": int(vars.rep_pen_range),
        }

    # If we're running Colab or OAI, we still need a tokenizer.
    if(vars.model == "Colab"):
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-2.7B", cache_dir="cache/")
        loadsettings()
    elif(vars.model == "OAI"):
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="cache/")
        loadsettings()
    # Load the TPU backend if requested
    elif(vars.model == "TPUMeshTransformerGPTJ"):
        print("{0}Initializing Mesh Transformer JAX, please wait...{1}".format(colors.PURPLE, colors.END))
        if not vars.custmodpth or not os.path.isdir(vars.custmodpth):
            raise FileNotFoundError(f"The specified model path {repr(vars.custmodpth)} is not the path to a valid folder")
        import tpu_mtj_backend
        tpu_mtj_backend.vars = vars
        tpu_mtj_backend.warper_callback = tpumtjgenerate_warper_callback
        tpu_mtj_backend.stopping_callback = tpumtjgenerate_stopping_callback
        tpu_mtj_backend.compiling_callback = tpumtjgenerate_compiling_callback
        tpu_mtj_backend.stopped_compiling_callback = tpumtjgenerate_stopped_compiling_callback
        tpu_mtj_backend.settings_callback = tpumtjgenerate_settings_callback
        vars.allowsp = True
        loadmodelsettings()
        loadsettings()
        tpu_mtj_backend.load_model(vars.custmodpth, **vars.modelconfig)
        vars.modeldim = int(tpu_mtj_backend.params["d_model"])
        tokenizer = tpu_mtj_backend.tokenizer
    else:
        loadsettings()

# Set up Flask routes
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/download')
def download():
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

if(path.exists("settings/" + getmodelname().replace('/', '_') + ".settings")):
    file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "r")
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

def lua_log_format_name(name):
    return f"[{name}]" if type(name) is str else "CORE"

_bridged = {}
F = TypeVar("F", bound=Callable)
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
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        if(vars.serverstarted):
            set_aibusy(0)

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
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="cache/")
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
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="cache/")
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
        "setrngpersist",
        "temp",
        "topp",
        "top_p",
        "topk",
        "top_k",
        "tfs",
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
    if(setting in ("settemp", "temp")): return vars.temp
    if(setting in ("settopp", "topp", "top_p")): return vars.top_p
    if(setting in ("settopk", "topk", "top_k")): return vars.top_k
    if(setting in ("settfs", "tfs")): return vars.tfs
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
    if(setting in ("setrngpersist", "rngpersist")): return vars.rngpersist
    if(setting in ("frmttriminc", "triminc")): return vars.formatoptns["frmttriminc"]
    if(setting in ("frmtrmblln", "rmblln")): return vars.formatoptns["frmttrmblln"]
    if(setting in ("frmtrmspch", "rmspch")): return vars.formatoptns["frmttrmspch"]
    if(setting in ("frmtadsnsp", "adsnsp")): return vars.formatoptns["frmtadsnsp"]
    if(setting in ("frmtsingleline", "singleline")): return vars.formatoptns["singleline"]

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
    if(setting in ("setrngpersist", "rngpersist")): vars.rngpersist = v
    if(setting in ("setchatmode", "chatmode")): vars.chatmode = v
    if(setting in ("frmttriminc", "triminc")): vars.formatoptns["frmttriminc"] = v
    if(setting in ("frmtrmblln", "rmblln")): vars.formatoptns["frmttrmblln"] = v
    if(setting in ("frmtrmspch", "rmspch")): vars.formatoptns["frmttrmspch"] = v
    if(setting in ("frmtadsnsp", "adsnsp")): vars.formatoptns["frmtadsnsp"] = v
    if(setting in ("frmtsingleline", "singleline")): vars.formatoptns["singleline"] = v

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
    if(vars.model in ("Colab", "OAI", "InferKit")):
        return "api"
    if(vars.model not in ("TPUMeshTransformerGPTJ",) and (vars.model in ("GPT2Custom", "NeoCustom") or vars.model_type in ("gpt2", "gpt_neo", "gptj"))):
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
    if(vars.model in ("EleutherAI/gpt-j-6B",) or (vars.model == "TPUMeshTransformerGPTJ" and tpu_mtj_backend.params["d_model"] == 4096) or (vars.model_type in ("gpt_neo", "gptj") and hidden_size == 4096)):
        return "gpt-j-6B"
    return "unknown"

#==================================================================#
#  Get model backend as "transformers" or "mtj"
#==================================================================#
@bridged_kwarg()
def lua_get_modelbackend():
    if(vars.noai):
        return "readonly"
    if(vars.model in ("Colab", "OAI", "InferKit")):
        return "api"
    if(vars.model in ("TPUMeshTransformerGPTJ",)):
        return "mtj"
    return "transformers"

#==================================================================#
#  Check whether model is loaded from a custom path
#==================================================================#
@bridged_kwarg()
def lua_is_custommodel():
    return vars.model in ("GPT2Custom", "NeoCustom", "TPUMeshTransformerGPTJ")

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
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
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
        print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
        print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
        print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        set_aibusy(0)
    if(vars.lua_koboldbridge.resend_settings_required):
        vars.lua_koboldbridge.resend_settings_required = False
        lua_resend_settings()
    for k in vars.lua_edited:
        inlineedit(k, vars.actions[k])
    for k in vars.lua_deleted:
        inlinedelete(k)

#==================================================================#
#  Lua runtime startup
#==================================================================#

print("", end="", flush=True)
print(colors.PURPLE + "Initializing Lua Bridge... " + colors.END, end="", flush=True)

# Set up Lua state
vars.lua_state = lupa.LuaRuntime(unpack_returned_tuples=True)

# Load bridge.lua
bridged = {
    "corescript_path": os.path.join(os.path.dirname(os.path.realpath(__file__)), "cores"),
    "userscript_path": os.path.join(os.path.dirname(os.path.realpath(__file__)), "userscripts"),
    "config_path": os.path.join(os.path.dirname(os.path.realpath(__file__)), "userscripts"),
    "lib_paths": vars.lua_state.table(os.path.join(os.path.dirname(os.path.realpath(__file__)), "lualibs"), os.path.join(os.path.dirname(os.path.realpath(__file__)), "extern", "lualibs")),
    "vars": vars,
}
for kwarg in _bridged:
    bridged[kwarg] = _bridged[kwarg]
try:
    vars.lua_kobold, vars.lua_koboldcore, vars.lua_koboldbridge = vars.lua_state.globals().dofile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "bridge.lua"))(
        vars.lua_state.globals().python,
        bridged,
    )
except lupa.LuaError as e:
    print(colors.RED + "ERROR!" + colors.END)
    vars.lua_koboldbridge.obliterate_multiverse()
    print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
    print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
    exit(1)
print(colors.GREEN + "OK!" + colors.END)

# Load scripts
load_lua_scripts()


#============================ METHODS =============================#    

#==================================================================#
# Event triggered when browser SocketIO is loaded and connects to server
#==================================================================#
@socketio.on('connect')
def do_connect():
    print("{0}Client connected!{1}".format(colors.GREEN, colors.END))
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
        print("{0}Data received:{1}{2}".format(colors.GREEN, msg, colors.END))
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
        actionback()
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
    elif(msg['cmd'] == 'loadselect'):
        vars.loadselect = msg["data"]
    elif(msg['cmd'] == 'spselect'):
        vars.spselect = msg["data"]
    elif(msg['cmd'] == 'loadrequest'):
        loadRequest(fileops.storypath(vars.loadselect))
    elif(msg['cmd'] == 'sprequest'):
        spRequest(vars.spselect)
        emit('from_server', {'cmd': 'spstatitems', 'data': {vars.spfilename: vars.spmeta} if vars.allowsp and len(vars.spfilename) else {}}, broadcast=True)
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
    elif(not vars.host and msg['cmd'] == 'importwi'):
        wiimportrequest()
    elif(msg['cmd'] == 'debug'):
        vars.debug = msg['data']
        emit('from_server', {'cmd': 'set_debug', 'data': msg['data']}, broadcast=True)
        if vars.debug:
            send_debug()

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

def actionsubmit(data, actionmode=0, force_submit=False, force_prompt_gen=False, disable_recentrng=False):
    # Ignore new submissions if the AI is currently busy
    if(vars.aibusy):
        return
    
    while(True):
        set_aibusy(1)

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
                data = f"\n{vars.chatname} : {data}\n"
        
        # If we're not continuing, store a copy of the raw input
        if(data != ""):
            vars.lastact = data
        
        if(not vars.gamestarted):
            vars.submission = data
            execute_inmod()
            data = vars.submission
            if(not force_submit and len(data.strip()) == 0):
                assert False
            # Start the game
            vars.gamestarted = True
            if(not vars.noai and vars.lua_koboldbridge.generating and (not vars.nopromptgen or force_prompt_gen)):
                # Save this first action as the prompt
                vars.prompt = data
                # Clear the startup text from game screen
                emit('from_server', {'cmd': 'updatescreen', 'gamestarted': False, 'data': 'Please wait, generating story...'}, broadcast=True)
                calcsubmit(data) # Run the first action through the generator
                if(not vars.abort and vars.lua_koboldbridge.restart_sequence is not None and len(vars.genseqs) == 0):
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
            execute_inmod()
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
                    if len(vars.actions_metadata) < len(vars.actions):
                        vars.actions_metadata.append({"Selected Text": data, "Alternative Text": []})
                    else:
                    # 2. We've selected a chunk of text that is was presented previously
                        try:
                            alternatives = [item['Text'] for item in vars.actions_metadata[len(vars.actions)-1]["Alternative Text"]]
                        except:
                            print(len(vars.actions))
                            print(vars.actions_metadata)
                            raise
                        if data in alternatives:
                            alternatives = [item for item in vars.actions_metadata[len(vars.actions)-1]["Alternative Text"] if item['Text'] != data]
                            vars.actions_metadata[len(vars.actions)-1]["Alternative Text"] = alternatives
                        vars.actions_metadata[len(vars.actions)-1]["Selected Text"] = data
                update_story_chunk('last')
                send_debug()

            if(not vars.noai and vars.lua_koboldbridge.generating):
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
                for i in range(vars.numseqs):
                    vars.lua_koboldbridge.outputs[i+1] = ""
                execute_outmod()
                vars.lua_koboldbridge.regeneration_required = False
                genout = []
                for i in range(vars.numseqs):
                    genout.append({"generated_text": vars.lua_koboldbridge.outputs[i+1]})
                    assert type(genout[-1]["generated_text"]) is str
                if(len(genout) == 1):
                    genresult(genout[0]["generated_text"])
                    if(not vars.abort and vars.lua_koboldbridge.restart_sequence is not None):
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                else:
                    if(not vars.abort and vars.lua_koboldbridge.restart_sequence is not None and vars.lua_koboldbridge.restart_sequence > 0):
                        genresult(genout[vars.lua_koboldbridge.restart_sequence-1]["generated_text"])
                        data = ""
                        force_submit = True
                        disable_recentrng = True
                        continue
                    genselect(genout)
                set_aibusy(0)
                emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
                break

#==================================================================#
#  
#==================================================================#
def actionretry(data):
    if(vars.noai):
        emit('from_server', {'cmd': 'errmsg', 'data': "Retry function unavailable in Read Only mode."})
        return
    if(vars.aibusy):
        return
    if(vars.recentrng is not None):
        randomGameRequest(vars.recentrng, memory=vars.recentrngm)
        return
    # Remove last action if possible and resubmit
    if(vars.gamestarted if vars.useprompt else len(vars.actions) > 0):
        if(not vars.recentback and len(vars.actions) != 0 and len(vars.genseqs) == 0):  # Don't pop if we're in the "Select sequence to keep" menu or if there are no non-prompt actions
            # We are going to move the selected text to alternative text in the actions_metadata variable so we can redo this action
            vars.actions_metadata[len(vars.actions)-1]['Alternative Text'] = [{'Text': vars.actions_metadata[len(vars.actions)-1]['Selected Text'],
                                                                        'Pinned': False,
                                                                        "Previous Selection": True,
                                                                        "Edited": False}] + vars.actions_metadata[len(vars.actions)-1]['Alternative Text']
            vars.actions_metadata[len(vars.actions)-1]['Selected Text'] = ""
            
            
            
            last_key = vars.actions.get_last_key()
            vars.actions.pop()
            remove_story_chunk(last_key + 1)
        vars.recentback = False
        vars.recentedit = False
        vars.lua_koboldbridge.feedback = None
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
        vars.actions_metadata[len(vars.actions)-1]['Alternative Text'] = [{'Text': vars.actions_metadata[len(vars.actions)-1]['Selected Text'],
                                                                    'Pinned': False,
                                                                    "Previous Selection": True,
                                                                    "Edited": False}] + vars.actions_metadata[len(vars.actions)-1]['Alternative Text']
        vars.actions_metadata[len(vars.actions)-1]['Selected Text'] = ""
    
        last_key = vars.actions.get_last_key()
        vars.actions.pop()
        vars.recentback = True
        remove_story_chunk(last_key + 1)
    elif(len(vars.genseqs) == 0):
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."})
    else:
        vars.genseqs = []
    send_debug()
        
def actionredo():
    i = 0
    if len(vars.actions) < len(vars.actions_metadata):
        genout = [{"generated_text": item['Text']} for item in vars.actions_metadata[len(vars.actions)]['Alternative Text'] if (item["Previous Selection"]==True)]
        genout = genout + [{"generated_text": item['Text']} for item in vars.actions_metadata[len(vars.actions)]['Alternative Text'] if (item["Pinned"]==True) and (item["Previous Selection"]==False)]
        
        if len(genout) == 1:
            vars.actions_metadata[len(vars.actions)]['Alternative Text'] = [item for item in vars.actions_metadata[len(vars.actions)]['Alternative Text'] if (item["Previous Selection"]!=True)]
            genresult(genout[0]['generated_text'], flash=True)
        else:
            # Store sequences in memory until selection is made
            vars.genseqs = genout
            
            
            # Send sequences to UI for selection
            genout = [[item['Text'], "redo"] for item in vars.actions_metadata[len(vars.actions)]['Alternative Text'] if (item["Previous Selection"]==True)]
            emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True)
    else:
        emit('from_server', {'cmd': 'popuperror', 'data': "There's nothing to undo"}, broadcast=True)
    send_debug()

#==================================================================#
#  
#==================================================================#
def calcsubmitbudgetheader(txt, **kwargs):
    # Scan for WorldInfo matches
    winfo, found_entries = checkworldinfo(txt, **kwargs)

    # Add a newline to the end of memory
    if(vars.memory != "" and vars.memory[-1] != "\n"):
        mem = vars.memory + "\n"
    else:
        mem = vars.memory

    # Build Author's Note if set
    if(vars.authornote != ""):
        anotetxt  = ("\n" + vars.authornotetemplate + "\n").replace("<|>", vars.authornote)
    else:
        anotetxt = ""

    return winfo, mem, anotetxt, found_entries

def calcsubmitbudget(actionlen, winfo, mem, anotetxt, actions, submission=None, budget_deduction=0):
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    anotetkns    = []  # Placeholder for Author's Note tokens
    lnanote      = 0   # Placeholder for Author's Note length

    lnsp = vars.sp_length

    if("tokenizer" not in globals()):
        from transformers import GPT2TokenizerFast
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="cache/")

    # Calculate token budget
    prompttkns = tokenizer.encode(utils.encodenewlines(vars.comregex_ai.sub('', vars.prompt)), max_length=int(2e9), truncation=True)
    lnprompt   = len(prompttkns)

    memtokens = tokenizer.encode(utils.encodenewlines(mem), max_length=int(2e9), truncation=True)
    lnmem     = len(memtokens)
    if(lnmem > vars.max_length - lnsp - vars.genamt - budget_deduction):
        raise OverflowError("The memory in your story is too long. Please either write a shorter memory text or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    witokens  = tokenizer.encode(utils.encodenewlines(winfo), max_length=int(2e9), truncation=True)
    lnwi      = len(witokens)
    if(lnmem + lnwi > vars.max_length - lnsp - vars.genamt - budget_deduction):
        raise OverflowError("The current active world info keys take up too many tokens. Please either write shorter world info, decrease World Info Depth or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    if(anotetxt != ""):
        anotetkns = tokenizer.encode(utils.encodenewlines(anotetxt), max_length=int(2e9), truncation=True)
        lnanote   = len(anotetkns)
        if(lnmem + lnwi + lnanote > vars.max_length - lnsp - vars.genamt - budget_deduction):
            raise OverflowError("The author's note in your story is too long. Please either write a shorter author's note or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt.")

    if(vars.useprompt):
        budget = vars.max_length - lnsp - lnprompt - lnmem - lnanote - lnwi - vars.genamt - budget_deduction
    else:
        budget = vars.max_length - lnsp - lnmem - lnanote - lnwi - vars.genamt - budget_deduction

    lnsubmission = len(tokenizer.encode(utils.encodenewlines(vars.comregex_ai.sub('', submission)), max_length=int(2e9), truncation=True)) if submission is not None else 0
    maybe_lnprompt = lnprompt if vars.useprompt and actionlen > 0 else 0

    if(lnmem + lnwi + lnanote + maybe_lnprompt + lnsubmission > vars.max_length - lnsp - vars.genamt - budget_deduction):
        raise OverflowError("Your submission is too long. Please either write a shorter submission or increase the Max Tokens setting. If you are using a soft prompt, additionally consider using a smaller soft prompt. If you are using the Always Add Prompt setting, turning it off may help.")

    assert budget >= 0

    if(actionlen == 0):
        # First/Prompt action
        tokens = memtokens + witokens + anotetkns + prompttkns
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
                tokens = memtokens + witokens + anotetkns + prompttkns + tokens
            else:
                tokens = memtokens + witokens + prompttkns + tokens
        else:
            # Prepend Memory, WI, and Prompt before action tokens
            tokens = memtokens + witokens + prompttkns + tokens

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
            if(not vars.model in ["Colab", "OAI", "TPUMeshTransformerGPTJ"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(vars.model == "Colab"):
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.model == "OAI"):
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.model == "TPUMeshTransformerGPTJ"):
                tpumtjgenerate(subtxt, min, max, found_entries=found_entries)
        else:
            if(not vars.model in ["Colab", "OAI", "TPUMeshTransformerGPTJ"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(vars.model == "Colab"):
                sendtocolab(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.model == "OAI"):
                oairequest(utils.decodenewlines(tokenizer.decode(subtxt)), min, max)
            elif(vars.model == "TPUMeshTransformerGPTJ"):
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
                repetition_penalty=1.1,
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
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, utils.decodenewlines(tokenizer.decode(txt)), colors.END))

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
            print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
            print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
            print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
        else:
            emit('from_server', {'cmd': 'errmsg', 'data': 'Error occurred during generator call; please check console.'}, broadcast=True)
            print("{0}{1}{2}".format(colors.RED, traceback.format_exc().replace("\033", ""), colors.END), file=sys.stderr)
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
def genresult(genout, flash=True):
    if not vars.quiet:
        print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
    
    # Format output before continuing
    genout = applyoutputformatting(genout)

    vars.lua_koboldbridge.feedback = genout

    if(len(genout) == 0):
        return
    
    # Add formatted text to Actions array and refresh the game screen
    if(len(vars.prompt.strip()) == 0):
        vars.prompt = genout
    else:
        vars.actions.append(genout)
        if len(vars.actions) > len(vars.actions_metadata):
            vars.actions_metadata.append({'Selected Text': genout, 'Alternative Text': []})
        else:
            vars.actions_metadata[len(vars.actions)-1]['Selected Text'] = genout
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
            print("{0}[Result {1}]\n{2}{3}".format(colors.CYAN, i, result["generated_text"], colors.END))
        i += 1
    
    # Add the options to the actions metadata
    # If we've already generated text for this action but haven't selected one we'll want to kill all non-pinned, non-previous selection, and non-edited options then add the new ones
    if (len(vars.actions_metadata) > len(vars.actions)):
        if (vars.actions_metadata[len(vars.actions)]['Selected Text'] == ""):
            vars.actions_metadata[len(vars.actions)]['Alternative Text'] = [{"Text": item['Text'], "Pinned": item['Pinned'], 
                                                                             "Previous Selection": item["Previous Selection"], 
                                                                             "Edited": item["Edited"]} for item in vars.actions_metadata[len(vars.actions)]['Alternative Text'] 
                                                                             if item['Pinned'] or item["Previous Selection"] or item["Edited"]] + [{"Text": text["generated_text"], 
                                                                                    "Pinned": False, "Previous Selection": False, "Edited": False} for text in genout]
        else:
            vars.actions_metadata.append({'Selected Text': '', 'Alternative Text': [{"Text": text["generated_text"], "Pinned": False, "Previous Selection": False, "Edited": False} for text in genout]})
    else:
        vars.actions_metadata.append({'Selected Text': '', 'Alternative Text': [{"Text": text["generated_text"], "Pinned": False, "Previous Selection": False, "Edited": False} for text in genout]})
    
    genout = [{"generated_text": item['Text']} for item in vars.actions_metadata[len(vars.actions)]['Alternative Text'] if (item["Previous Selection"]==False) and (item["Edited"]==False)]

    # Store sequences in memory until selection is made
    vars.genseqs = genout
    
    genout = [[item['Text'], "pinned" if item['Pinned'] else "normal"] for item in vars.actions_metadata[len(vars.actions)]['Alternative Text']  if (item["Previous Selection"]==False) and (item["Edited"]==False)]

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
        vars.actions_metadata[len(vars.actions)-1]['Alternative Text'] = [item for item in vars.actions_metadata[len(vars.actions)-1]['Alternative Text'] if item['Text'] != vars.lua_koboldbridge.feedback]
        vars.actions_metadata[len(vars.actions)-1]['Selected Text'] = vars.lua_koboldbridge.feedback
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
        if text in [item['Text'] for item in vars.actions_metadata[len(vars.actions)]['Alternative Text']]:
            alternatives = vars.actions_metadata[len(vars.actions)]['Alternative Text']
            for i in range(len(alternatives)):
                if alternatives[i]['Text'] == text:
                    alternatives[i]['Pinned'] = not alternatives[i]['Pinned']
                    break
            vars.actions_metadata[len(vars.actions)]['Alternative Text'] = alternatives
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
#  Send text to TPU mesh transformer backend
#==================================================================#
def tpumtjgenerate(txt, minimum, maximum, found_entries=None):
    vars.generated_tkns = 0

    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(vars.numseqs))

    if not vars.quiet:
        print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, utils.decodenewlines(tokenizer.decode(txt)), colors.END))

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
                numseqs=vars.numseqs,
                repetition_penalty=vars.rep_pen,
                rpslope=vars.rep_pen_slope,
                rprange=vars.rep_pen_range,
                soft_embeddings=vars.sp,
                soft_tokens=soft_tokens,
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
            print("{0}{1}{2}".format(colors.RED, "***LUA ERROR***: ", colors.END), end="", file=sys.stderr)
            print("{0}{1}{2}".format(colors.RED, str(e).replace("\033", ""), colors.END), file=sys.stderr)
            print("{0}{1}{2}".format(colors.YELLOW, "Lua engine stopped; please open 'Userscripts' and press Load to reinitialize scripts.", colors.END), file=sys.stderr)
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
    emit('from_server', {'cmd': 'updatenopromptgen', 'data': vars.nopromptgen}, broadcast=True)
    emit('from_server', {'cmd': 'updaterngpersist', 'data': vars.rngpersist}, broadcast=True)
    emit('from_server', {'cmd': 'updatenogenmod', 'data': vars.nogenmod}, broadcast=True)
    
    emit('from_server', {'cmd': 'updatefrmttriminc', 'data': vars.formatoptns["frmttriminc"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtrmblln', 'data': vars.formatoptns["frmtrmblln"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtrmspch', 'data': vars.formatoptns["frmtrmspch"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtadsnsp', 'data': vars.formatoptns["frmtadsnsp"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatesingleline', 'data': vars.formatoptns["singleline"]}, broadcast=True)
    
    # Allow toggle events again
    emit('from_server', {'cmd': 'allowtoggle', 'data': True}, broadcast=True)

#==================================================================#
#  Sets the logical and display states for the AI Busy condition
#==================================================================#
def set_aibusy(state):
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
            print(f"WARNING: Attempted to edit non-existent chunk {chunk}")

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
            vars.actions[chunk-1] = ''
        else:
            print(f"WARNING: Attempted to delete non-existent chunk {chunk}")
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
                if ky in txt:
                    if wi.get("selective", False) and len(keys_secondary):
                        found = False
                        for ks in keys_secondary:
                            ksy = ks
                            if(vars.wirmvwhtsp):
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
        if len(vars.actions_metadata) < len(vars.actions):
            vars.actions_metadata.append({"Selected Text": genout, "Alternative Text": []})
        else:
        # 2. We've selected a chunk of text that is was presented previously
            alternatives = [item['Text'] for item in vars.actions_metadata[len(vars.actions)]["Alternative Text"]]
            if genout in alternatives:
                alternatives = [item for item in vars.actions_metadata[len(vars.actions)]["Alternative Text"] if item['Text'] != genout]
                vars.actions_metadata[len(vars.actions)]["Alternative Text"] = alternatives
            vars.actions_metadata[len(vars.actions)]["Selected Text"] = genout
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
    reqdata = {
        'prompt': txt,
        'max_tokens': max,
        'temperature': vars.temp,
        'top_p': vars.top_p,
        'n': 1,
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
        genout = req.json()["choices"][0]["text"]

        vars.lua_koboldbridge.outputs[1] = genout

        execute_outmod()
        if(vars.lua_koboldbridge.regeneration_required):
            vars.lua_koboldbridge.regeneration_required = False
            genout = vars.lua_koboldbridge.outputs[1]
            assert genout is str

        if not vars.quiet:
            print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
        vars.actions.append(genout)
        if len(vars.actions_metadata) < len(vars.actions):
            vars.actions_metadata.append({"Selected Text": genout, "Alternative Text": []})
        else:
        # 2. We've selected a chunk of text that is was presented previously
            alternatives = [item['Text'] for item in vars.actions_metadata[len(vars.actions)]["Alternative Text"]]
            if genout in alternatives:
                alternatives = [item for item in vars.actions_metadata[len(vars.actions)]["Alternative Text"] if item['Text'] != genout]
                vars.actions_metadata[len(vars.actions)]["Alternative Text"] = alternatives
            vars.actions_metadata[len(vars.actions)]["Selected Text"] = genout
        update_story_chunk('last')
        emit('from_server', {'cmd': 'texteffect', 'data': vars.actions.get_last_key() + 1 if len(vars.actions) else 0}, broadcast=True)
        send_debug()
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

        del vars.actions
        vars.actions = structures.KoboldStoryRegister()
        actions = collections.deque(js["actions"])

        if "actions_metadata" in js:
            vars.actions_metadata = js["actions_metadata"]
        else:
            vars.actions_metadata = [{'Selected Text': text, 'Alternative Text': []} for text in js["actions"]]
                

        if(len(vars.prompt.strip()) == 0):
            while(len(actions)):
                action = actions.popleft()
                if(len(action.strip()) != 0):
                    vars.prompt = action
                    break
            else:
                vars.gamestarted = False
        if(vars.gamestarted):
            for s in actions:
                vars.actions.append(s)
        
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
        vars.savedir = getcwd()+"\stories"
        
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
    
    urlformat = "https://prompts.aidg.club/api/"
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
        vars.worldinfo   = []
        vars.worldinfo_i = []
        vars.worldinfo_u = {}
        vars.wifolders_d = {}
        vars.wifolders_l = []
        vars.wifolders_u = {uid: [] for uid in vars.wifolders_d}
        vars.lastact     = ""
        vars.submission  = ""
        vars.lastctx     = ""
        
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
        vars.savedir = getcwd()+"\stories"
        
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
    vars.actions_metadata = []
    
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
    vars.savedir = getcwd()+"\stories"
    
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
    if(vars.allowsp and "softprompt" in js and type(js["softprompt"]) is str and all(q not in js["softprompt"] for q in ("..", ":")) and (len(js["softprompt"]) == 0 or all(js["softprompt"][0] not in q for q in ("/", "\\")))):
        spRequest(js["softprompt"])
    else:
        vars.spfilename = ""
    file.close()

# Precompile TPU backend if required
if(vars.model in ("TPUMeshTransformerGPTJ",)):
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

def send_debug():
    if vars.debug:
        debug_info = ""
        for variable in [["Newline Mode", vars.newlinemode], ["Action Length", len(vars.actions)], ["Actions Metadata Length", len(vars.actions_metadata)], ["Actions Metadata", vars.actions_metadata]]:
            debug_info = "{}{}: {}\n".format(debug_info, variable[0], variable[1])
        emit('from_server', {'cmd': 'debug_info', 'data': debug_info}, broadcast=True)
    
#==================================================================#
#  Final startup commands to launch Flask app
#==================================================================#
print("", end="", flush=True)
if __name__ == "__main__":
    print("{0}\nStarting webserver...{1}".format(colors.GREEN, colors.END), flush=True)

    # Start Flask/SocketIO (Blocking, so this must be last method!)
    
    #socketio.run(app, host='0.0.0.0', port=5000)
    if(vars.host):
        if(args.ngrok):
            from flask_ngrok import _run_ngrok
            cloudflare = _run_ngrok()
        elif(args.remote):
           from flask_cloudflared import _run_cloudflared
           cloudflare = _run_cloudflared(5000)
        if(args.ngrok or args.remote):
            with open('cloudflare.log', 'w') as cloudflarelog:
                cloudflarelog.write("KoboldAI has finished loading and is available at the following link : " + cloudflare)
                print(format(colors.GREEN) + "KoboldAI has finished loading and is available at the following link : " + cloudflare + format(colors.END))
        else:
            print("{0}Webserver has started, you can now connect to this machine at port 5000{1}".format(colors.GREEN, colors.END))
        vars.serverstarted = True
        socketio.run(app, host='0.0.0.0', port=5000)
    else:
        import webbrowser
        webbrowser.open_new('http://localhost:5000')
        print("{0}Server started!\nYou may now connect with a browser at http://127.0.0.1:5000/{1}".format(colors.GREEN, colors.END))
        vars.serverstarted = True
        if args.unblock:
            socketio.run(app, port=5000, host='0.0.0.0')
        else:
            socketio.run(app, port=5000)

else:
    print("{0}\nServer started in WSGI mode!{1}".format(colors.GREEN, colors.END), flush=True)
