#!/usr/bin/python3
#==================================================================#
# KoboldAI
# Version: 1.16.4
# By: KoboldAIDev and the KoboldAI Community
#==================================================================#

# External packages
import os
from os import path, getcwd
import re
import tkinter as tk
from tkinter import messagebox
import json
import collections
import zipfile
import packaging
import contextlib
from typing import Any, Union, Dict, Set, List

import requests
import html
import argparse
import sys
import gc

# KoboldAI
import fileops
import gensettings
from utils import debounce
import utils
import structures

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
    ["GPT-Neo 1.3B", "EleutherAI/gpt-neo-1.3B", "8GB"],
    ["GPT-Neo 2.7B", "EleutherAI/gpt-neo-2.7B", "16GB"],
    ["GPT-J 6B", "EleutherAI/gpt-j-6B", "24GB"],
    ["GPT-2", "gpt2", "1GB"],
    ["GPT-2 Med", "gpt2-medium", "2GB"],
    ["GPT-2 Large", "gpt2-large", "4GB"],
    ["GPT-2 XL", "gpt2-xl", "8GB"],
    ["InferKit API (requires API key)", "InferKit", ""],
    ["Google Colab", "Colab", ""],
    ["OpenAI API (requires API key)", "OAI", ""],
    ["Read Only (No AI)", "ReadOnly", ""]
    ]

# Variables
class vars:
    lastact     = ""     # The last action received from the user
    lastctx     = ""     # The last context submitted to the generator
    model       = ""     # Model ID string chosen at startup
    noai        = False  # Runs the script without starting up the transformers pipeline
    aibusy      = False  # Stops submissions while the AI is working
    max_length  = 1024    # Maximum number of tokens to submit per action
    ikmax       = 3000   # Maximum number of characters to submit to InferKit
    genamt      = 80     # Amount of text for each action to generate
    ikgen       = 200    # Number of characters for InferKit to generate
    rep_pen     = 1.1    # Default generator repetition_penalty
    temp        = 0.5    # Default generator temperature
    top_p       = 0.9    # Default generator top_p
    top_k       = 0      # Default generator top_k
    tfs         = 1.0    # Default generator tfs (tail-free sampling)
    numseqs     = 1     # Number of sequences to ask the generator to create
    gamestarted = False  # Whether the game has started (disables UI elements)
    prompt      = ""     # Prompt
    memory      = ""     # Text submitted to memory field
    authornote  = ""     # Text submitted to Author's Note field
    andepth     = 3      # How far back in history to append author's note
    actions     = structures.KoboldStoryRegister()  # Actions submitted by user and AI
    worldinfo   = []     # Array of World Info key/value objects
    # badwords    = []     # Array of str/chr values that should be removed from output
    badwordsids = [[13460], [6880], [50256], [42496], [4613], [17414], [22039], [16410], [27], [29], [38430], [37922], [15913], [24618], [28725], [58], [47175], [36937], [26700], [12878], [16471], [37981], [5218], [29795], [13412], [45160], [3693], [49778], [4211], [20598], [36475], [33409], [44167], [32406], [29847], [29342], [42669], [685], [25787], [7359], [3784], [5320], [33994], [33490], [34516], [43734], [17635], [24293], [9959], [23785], [21737], [28401], [18161], [26358], [32509], [1279], [38155], [18189], [26894], [6927], [14610], [23834], [11037], [14631], [26933], [46904], [22330], [25915], [47934], [38214], [1875], [14692], [41832], [13163], [25970], [29565], [44926], [19841], [37250], [49029], [9609], [44438], [16791], [17816], [30109], [41888], [47527], [42924], [23984], [49074], [33717], [31161], [49082], [30138], [31175], [12240], [14804], [7131], [26076], [33250], [3556], [38381], [36338], [32756], [46581], [17912], [49146]] # Tokenized array of badwords used to prevent AI artifacting
    deletewi    = -1     # Temporary storage for index to delete
    wirmvwhtsp  = False  # Whether to remove leading whitespace from WI entries
    widepth     = 3      # How many historical actions to scan for WI hits
    mode        = "play" # Whether the interface is in play, memory, or edit mode
    editln      = 0      # Which line was last selected in Edit Mode
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
    sp          = None   # Current soft prompt tensor (as a NumPy array)
    sp_length   = 0      # Length of current soft prompt in tokens, or 0 if not using a soft prompt
    svowname    = ""     # Filename that was flagged for overwrite confirm
    saveow      = False  # Whether or not overwrite confirm has been displayed
    genseqs     = []     # Temporary storage for generated sequences
    recentback  = False  # Whether Back button was recently used without Submitting or Retrying after
    useprompt   = False   # Whether to send the full prompt with every submit action
    breakmodel  = False  # For GPU users, whether to use both system RAM and VRAM to conserve VRAM while offering speedup compared to CPU-only
    bmsupported = False  # Whether the breakmodel option is supported (GPT-Neo/GPT-J only, currently)
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
    actionmode  = 1
    adventure   = False
    dynamicscan = False
    remote      = False

#==================================================================#
# Function to get model selection at startup
#==================================================================#
def getModelSelection():
    print("    #   Model                           V/RAM\n    =========================================")
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
        
        modpath = fileops.getdirpath(getcwd(), "Select Model Folder")
        
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
            assert sum(breakmodel.gpu_blocks) <= n_layers
            n_layers -= sum(breakmodel.gpu_blocks)
        except:
            print("WARNING: --layers is malformatted. Please use the --help option to see correct usage of --layers. Defaulting to all layers on device 0.", file=sys.stderr)
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
        model = model.half().to(len(breakmodel.gpu_blocks)-1)
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
    model.transformer.wte.to(breakmodel.primary_device)
    model.transformer.ln_f.to(breakmodel.primary_device)
    if(hasattr(model, 'lm_head')):
        model.lm_head.to(breakmodel.primary_device)
    if(hasattr(model.transformer, 'wpe')):
        model.transformer.wpe.to(breakmodel.primary_device)
    gc.collect()
    GPTNeoModel.forward = breakmodel.new_forward
    if("GPTJModel" in globals()):
        GPTJModel.forward = breakmodel.new_forward
    generator = model.generate
    breakmodel.move_hidden_layers(model.transformer)

#==================================================================#
# Startup
#==================================================================#

# Parsing Parameters
parser = argparse.ArgumentParser(description="KoboldAI Server")
parser.add_argument("--remote", action='store_true', help="Optimizes KoboldAI for Remote Play")
parser.add_argument("--ngrok", action='store_true', help="Optimizes KoboldAI for Remote Play using Ngrok")
parser.add_argument("--model", help="Specify the Model Type to skip the Menu")
parser.add_argument("--path", help="Specify the Path for local models (For model NeoCustom or GPT2Custom)")
parser.add_argument("--cpu", action='store_true', help="By default unattended launches are on the GPU use this option to force CPU usage.")
parser.add_argument("--breakmodel", action='store_true', help=argparse.SUPPRESS)
parser.add_argument("--breakmodel_layers", type=int, help=argparse.SUPPRESS)
parser.add_argument("--breakmodel_gpulayers", type=str, help="If using a model that supports hybrid generation, this is a comma-separated list that specifies how many layers to put on each GPU device. For example to put 8 layers on device 0, 9 layers on device 1 and 11 layers on device 2, use --layers 8,9,11")
parser.add_argument("--override_delete", action='store_true', help="Deleting stories from inside the browser is disabled if you are using --remote and enabled otherwise. Using this option will instead allow deleting stories if using --remote and prevent deleting stories otherwise.")
parser.add_argument("--override_rename", action='store_true', help="Renaming stories from inside the browser is disabled if you are using --remote and enabled otherwise. Using this option will instead allow renaming stories if using --remote and prevent renaming stories otherwise.")
parser.add_argument("--configname", help="Force a fixed configuration name to aid with config management.")

args = parser.parse_args()
vars.model = args.model;

if args.remote:
    vars.remote = True;

if args.ngrok:
    vars.remote = True;

vars.smandelete = vars.remote == args.override_delete
vars.smanrename = vars.remote == args.override_rename

# Select a model to run
if args.model:
    print("Welcome to KoboldAI!\nYou have selected the following Model:", vars.model)
    if args.path:
        print("You have selected the following path for your Model :", args.path)
        vars.custmodpth = args.path;
        vars.colaburl = args.path + "/request"; # Lets just use the same parameter to keep it simple

else:
    print("{0}Welcome to the KoboldAI Server!\nSelect an AI model to continue:{1}\n".format(colors.CYAN, colors.END))
    getModelSelection()

# If transformers model was selected & GPU available, ask to use CPU or GPU
if(not vars.model in ["InferKit", "Colab", "OAI", "ReadOnly", "TPUMeshTransformerGPTJ"]):
    vars.allowsp = True
    # Test for GPU support
    import torch
    print("{0}Looking for GPU support...{1}".format(colors.PURPLE, colors.END), end="")
    vars.hascuda = torch.cuda.is_available()
    vars.bmsupported = vars.model in ("EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-j-6B", "NeoCustom")
    if(args.breakmodel is not None and args.breakmodel):
        print("WARNING: --breakmodel is no longer supported. Breakmodel mode is now automatically enabled when --layers is used (see --help for details).", file=sys.stderr)
    if(args.breakmodel_layers is not None):
        print("WARNING: --breakmodel_layers is deprecated. Use --layers instead (see --help for details).", file=sys.stderr)
    if(not vars.bmsupported and (args.breakmodel_gpulayers is not None or args.breakmodel_layers is not None)):
        print("WARNING: This model does not support hybrid generation. --layers will be ignored.", file=sys.stderr)
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
        print("{0}Please enter the ngrok.io or trycloudflare.com URL displayed in Google Colab:{1}\n".format(colors.CYAN, colors.END))
        vars.colaburl = input("URL> ") + "/request"

if(vars.model == "ReadOnly"):
    vars.noai = True

# Set logging level to reduce chatter from Flask
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Start flask & SocketIO
print("{0}Initializing Flask... {1}".format(colors.PURPLE, colors.END), end="")
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
app = Flask(__name__)
app.config['SECRET KEY'] = 'secret!'
socketio = SocketIO(app)
print("{0}OK!{1}".format(colors.GREEN, colors.END))

# Start transformers and create pipeline
if(not vars.model in ["InferKit", "Colab", "OAI", "ReadOnly", "TPUMeshTransformerGPTJ"]):
    if(not vars.noai):
        print("{0}Initializing transformers, please wait...{1}".format(colors.PURPLE, colors.END))
        from transformers import StoppingCriteria, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, GPTNeoModel, AutoModelForCausalLM
        try:
            from transformers import GPTJModel
        except:
            pass
        import transformers.generation_utils
        from transformers import __version__ as transformers_version

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
                inputs_embeds = self.transformer.wte(input_ids)
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
        try:
            from transformers import GPTJForCausalLM
            patch_causallm(GPTJForCausalLM)
        except:
            pass


        # Patch transformers to use our custom logit warpers
        from transformers import LogitsProcessorList, LogitsWarper, TopKLogitsWarper, TopPLogitsWarper, TemperatureLogitsWarper
        class TailFreeLogitsWarper(LogitsWarper):

            def __init__(self, tfs: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
                tfs = float(tfs)
                if tfs < 0 or tfs > 1.0:
                    raise ValueError(f"`tfs` has to be a float > 0 and < 1, but is {tfs}")
                self.tfs = tfs
                self.filter_value = filter_value
                self.min_tokens_to_keep = min_tokens_to_keep

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                if self.filter_value >= 1.0:
                    return scores
                sorted_logits, sorted_indices = torch.sort(scores, descending=True)
                probs = sorted_logits.softmax(dim=-1)

                # Compute second derivative normalized CDF
                d2 = probs.diff().diff().abs()
                normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
                normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

                # Remove tokens with CDF value above the threshold (token with 0 are kept)
                sorted_indices_to_remove = normalized_d2_cdf > self.tfs

                # Centre the distribution around the cutoff as in the original implementation of the algorithm
                sorted_indices_to_remove = torch.cat(
                    (
                        torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                        sorted_indices_to_remove,
                        torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                    ),
                    dim=-1,
                )

                if self.min_tokens_to_keep > 1:
                    # Keep at least min_tokens_to_keep
                    sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scores = scores.masked_fill(indices_to_remove, self.filter_value)
                return scores

        def new_get_logits_warper(
            top_k: int = None,
            top_p: float = None,
            tfs: float = None,
            temp: float = None,
            beams: int = 1,
        ) -> LogitsProcessorList:
            warper_list = LogitsProcessorList()
            if(top_k is not None and top_k > 0):
                warper_list.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1 + (beams > 1)))
            if(top_p is not None and top_p < 1.0):
                warper_list.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1 + (beams > 1)))
            if(tfs is not None and tfs < 1.0):
                warper_list.append(TailFreeLogitsWarper(tfs=tfs, min_tokens_to_keep=1 + (beams > 1)))
            if(temp is not None and temp != 1.0):
                warper_list.append(TemperatureLogitsWarper(temperature=temp))
            return warper_list
        
        def new_sample(self, *args, **kwargs):
            assert kwargs.pop("logits_warper", None) is not None
            kwargs["logits_warper"] = new_get_logits_warper(
                vars.top_k,
                vars.top_p,
                vars.tfs,
                vars.temp,
                1,
            )
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
                head_length: int,
            ):
                self.any_new_entries = False
                self.tokenizer = tokenizer
                self.excluded_world_info = excluded_world_info
                self.head_length = head_length
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
                **kwargs,
            ) -> bool:
                assert input_ids.ndim == 2
                assert len(self.excluded_world_info) == input_ids.shape[0]
                self.any_new_entries = False
                if(not vars.dynamicscan):
                    return False
                tail = input_ids[..., self.head_length:]
                for i, t in enumerate(tail):
                    decoded = tokenizer.decode(t)
                    _, found = checkworldinfo(decoded, force_use_txt=True)
                    found -= self.excluded_world_info[i]
                    if(len(found) != 0):
                        self.any_new_entries = True
                        break
                return self.any_new_entries
        old_get_stopping_criteria = transformers.generation_utils.GenerationMixin._get_stopping_criteria
        def new_get_stopping_criteria(self, *args, **kwargs):
            stopping_criteria = old_get_stopping_criteria(self, *args, **kwargs)
            global tokenizer
            self.kai_scanner = DynamicWorldInfoScanCriteria(
                tokenizer=tokenizer,
                excluded_world_info=self.kai_scanner_excluded_world_info,
                head_length=self.kai_scanner_head_length,
            )
            stopping_criteria.append(self.kai_scanner)
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
            if(always_use or (vars.hascuda and (vars.usegpu or vars.breakmodel))):
                original_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float16)
                yield True
                torch.set_default_dtype(original_dtype)
            else:
                yield False

        # If custom GPT Neo model was chosen
        if(vars.model == "NeoCustom"):
            model_config = open(vars.custmodpth + "/config.json", "r")
            js   = json.load(model_config)
            with(maybe_use_float16()):
                if("model_type" in js):
                    model     = AutoModelForCausalLM.from_pretrained(vars.custmodpth, cache_dir="cache/", **maybe_low_cpu_mem_usage())
                else:
                    model     = GPTNeoForCausalLM.from_pretrained(vars.custmodpth, cache_dir="cache/", **maybe_low_cpu_mem_usage())
            vars.modeldim = get_hidden_size_from_model(model)
            tokenizer = GPT2Tokenizer.from_pretrained(vars.custmodpth, cache_dir="cache/")
            # Is CUDA available? If so, use GPU, otherwise fall back to CPU
            if(vars.hascuda):
                if(vars.usegpu):
                    model = model.half().to(0)
                    generator = model.generate
                elif(vars.breakmodel):  # Use both RAM and VRAM (breakmodel)
                    device_config(model)
                else:
                    generator = model.generate
            else:
                generator = model.generate
        # If custom GPT2 model was chosen
        elif(vars.model == "GPT2Custom"):
            model_config = open(vars.custmodpth + "/config.json", "r")
            js   = json.load(model_config)
            with(maybe_use_float16()):
                model = GPT2LMHeadModel.from_pretrained(vars.custmodpth, cache_dir="cache/", **maybe_low_cpu_mem_usage())
            tokenizer = GPT2Tokenizer.from_pretrained(vars.custmodpth, cache_dir="cache/")
            vars.modeldim = get_hidden_size_from_model(model)
            # Is CUDA available? If so, use GPU, otherwise fall back to CPU
            if(vars.hascuda and vars.usegpu):
                model = model.half().to(0)
                generator = model.generate
            else:
                generator = model.generate
        # If base HuggingFace model was chosen
        else:
            # Is CUDA available? If so, use GPU, otherwise fall back to CPU
            tokenizer = GPT2Tokenizer.from_pretrained(vars.model, cache_dir="cache/")
            if(vars.hascuda):
                if(vars.usegpu):
                    with(maybe_use_float16()):
                        model = AutoModelForCausalLM.from_pretrained(vars.model, cache_dir="cache/", **maybe_low_cpu_mem_usage())
                    vars.modeldim = get_hidden_size_from_model(model)
                    model = model.half().to(0)
                    generator = model.generate
                elif(vars.breakmodel):  # Use both RAM and VRAM (breakmodel)
                    with(maybe_use_float16()):
                        model = AutoModelForCausalLM.from_pretrained(vars.model, cache_dir="cache/", **maybe_low_cpu_mem_usage())
                    vars.modeldim = get_hidden_size_from_model(model)
                    device_config(model)
                else:
                    model = AutoModelForCausalLM.from_pretrained(vars.model, cache_dir="cache/", **maybe_low_cpu_mem_usage())
                    vars.modeldim = get_hidden_size_from_model(model)
                    generator = model.generate
            else:
                model = AutoModelForCausalLM.from_pretrained(vars.model, cache_dir="cache/", **maybe_low_cpu_mem_usage())
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
    # If we're running Colab or OAI, we still need a tokenizer.
    if(vars.model == "Colab"):
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    elif(vars.model == "OAI"):
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Load the TPU backend if requested
    elif(vars.model == "TPUMeshTransformerGPTJ"):
        print("{0}Initializing Mesh Transformer JAX, please wait...{1}".format(colors.PURPLE, colors.END))
        assert vars.model == "TPUMeshTransformerGPTJ" and vars.custmodpth and os.path.isdir(vars.custmodpth)
        import tpu_mtj_backend
        tpu_mtj_backend.load_model(vars.custmodpth)
        vars.allowsp = True
        vars.modeldim = int(tpu_mtj_backend.params["d_model"])
        tokenizer = tpu_mtj_backend.tokenizer

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
    js["actions"]     = tuple(vars.actions.values())
    js["worldinfo"]   = []
        
    # Extract only the important bits of WI
    for wi in vars.worldinfo:
        if(wi["constant"] or wi["key"] != ""):
            js["worldinfo"].append({
                "key": wi["key"],
                "keysecondary": wi["keysecondary"],
                "content": wi["content"],
                "selective": wi["selective"],
                "constant": wi["constant"]
            })
    
    save = Response(json.dumps(js, indent=3))
    filename = path.basename(vars.savedir)
    if filename[-5:] == ".json":
        filename = filename[:-5]
    save.headers.set('Content-Disposition', 'attachment', filename='%s.json' % filename)
    return(save)

#============================ METHODS =============================#    

#==================================================================#
# Event triggered when browser SocketIO is loaded and connects to server
#==================================================================#
@socketio.on('connect')
def do_connect():
    print("{0}Client connected!{1}".format(colors.GREEN, colors.END))
    emit('from_server', {'cmd': 'connected', 'smandelete': vars.smandelete, 'smanrename': vars.smanrename})
    if(vars.remote):
        emit('from_server', {'cmd': 'runs_remotely'})
    if(vars.allowsp):
        emit('from_server', {'cmd': 'allowsp', 'data': vars.allowsp})
    
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

#==================================================================#
# Event triggered when browser SocketIO sends data to the server
#==================================================================#
@socketio.on('message')
def get_message(msg):
    print("{0}Data received:{1}{2}".format(colors.GREEN, msg, colors.END))
    # Submit action
    if(msg['cmd'] == 'submit'):
        if(vars.mode == "play"):
            actionsubmit(msg['data'], actionmode=msg['actionmode'])
        elif(vars.mode == "edit"):
            editsubmit(msg['data'])
        elif(vars.mode == "memory"):
            memsubmit(msg['data'])
    # Retry Action
    elif(msg['cmd'] == 'retry'):
        actionretry(msg['data'])
    # Back/Undo Action
    elif(msg['cmd'] == 'back'):
        actionback()
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
    elif(not vars.remote and msg['cmd'] == 'savetofile'):
        savetofile()
    elif(not vars.remote and msg['cmd'] == 'loadfromfile'):
        loadfromfile()
    elif(msg['cmd'] == 'loadfromstring'):
        loadRequest(json.loads(msg['data']), filename=msg['filename'])
    elif(not vars.remote and msg['cmd'] == 'import'):
        importRequest()
    elif(msg['cmd'] == 'newgame'):
        newGameRequest()
    elif(msg['cmd'] == 'rndgame'):
        randomGameRequest(msg['data'])
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
        anotesubmit(msg['data'])
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
            vars.worldinfo[msg['data']]["init"] = True
            addwiitem()
    elif(msg['cmd'] == 'widelete'):
        deletewi(msg['data'])
    elif(msg['cmd'] == 'wiselon'):
        vars.worldinfo[msg['data']]["selective"] = True
    elif(msg['cmd'] == 'wiseloff'):
        vars.worldinfo[msg['data']]["selective"] = False
    elif(msg['cmd'] == 'wiconstanton'):
        vars.worldinfo[msg['data']]["constant"] = True
    elif(msg['cmd'] == 'wiconstantoff'):
        vars.worldinfo[msg['data']]["constant"] = False
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
        settingschanged()
        refresh_settings()
    elif(msg['cmd'] == 'setdynamicscan'):
        vars.dynamicscan = msg['data']
        settingschanged()
        refresh_settings()
    elif(not vars.remote and msg['cmd'] == 'importwi'):
        wiimportrequest()
    
#==================================================================#
#  Send start message and tell Javascript to set UI state
#==================================================================#
def setStartState():
    txt = "<span>Welcome to <span class=\"color_cyan\">KoboldAI</span>! You are running <span class=\"color_green\">"+getmodelname()+"</span>.<br/>"
    if(not vars.noai):
        txt = txt + "Please load a game or enter a prompt below to begin!</span>"
    else:
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
    js["genamt"]      = vars.genamt
    js["max_length"]  = vars.max_length
    js["ikgen"]       = vars.ikgen
    js["formatoptns"] = vars.formatoptns
    js["numseqs"]     = vars.numseqs
    js["widepth"]     = vars.widepth
    js["useprompt"]   = vars.useprompt
    js["adventure"]   = vars.adventure
    js["dynamicscan"] = vars.dynamicscan

    # Write it
    if not os.path.exists('settings'):
        os.mkdir('settings')
    file = open("settings/" + getmodelname().replace('/', '_') + ".settings", "w")
    try:
        file.write(json.dumps(js, indent=3))
    finally:
        file.close()

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
        if("dynamicscan" in js):
            vars.dynamicscan = js["dynamicscan"]
        
        file.close()

#==================================================================#
#  Allow the models to override some settings
#==================================================================#
def loadmodelsettings():
    if(path.exists(vars.custmodpth + "/config.json")):
        model_config = open(vars.custmodpth + "/config.json", "r")
        js   = json.load(model_config)
        if("badwordsids" in js):
            vars.badwordsids = js["badwordsids"]
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
        if("adventure" in js):
            vars.adventure = js["adventure"]
        if("dynamicscan" in js):
            vars.dynamicscan = js["dynamicscan"]
        if("formatoptns" in js):
            vars.formatoptns = js["formatoptns"]
        model_config.close()

#==================================================================#
#  Don't save settings unless 2 seconds have passed without modification
#==================================================================#
@debounce(2)
def settingschanged():
    print("{0}Saving settings!{1}".format(colors.GREEN, colors.END))
    savesettings()

#==================================================================#
#  Take input text from SocketIO and decide what to do with it
#==================================================================#
def actionsubmit(data, actionmode=0, force_submit=False):
    # Ignore new submissions if the AI is currently busy
    if(vars.aibusy):
        return
    set_aibusy(1)

    vars.recentback = False
    vars.recentedit = False
    vars.actionmode = actionmode

    # "Action" mode
    if(actionmode == 1):
        data = data.strip().lstrip('>')
        data = re.sub(r'\n+', ' ', data)
        if(len(data)):
            data = f"\n\n> {data}\n"
    
    # If we're not continuing, store a copy of the raw input
    if(data != ""):
        vars.lastact = data
    
    if(not vars.gamestarted):
        if(not force_submit and len(data.strip()) == 0):
            set_aibusy(0)
            return
        # Start the game
        vars.gamestarted = True
        # Save this first action as the prompt
        vars.prompt = data
        if(not vars.noai):
            # Clear the startup text from game screen
            emit('from_server', {'cmd': 'updatescreen', 'gamestarted': False, 'data': 'Please wait, generating story...'}, broadcast=True)
            calcsubmit(data) # Run the first action through the generator
            emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
        else:
            refresh_story()
            set_aibusy(0)
            emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
    else:
        # Dont append submission if it's a blank/continue action
        if(data != ""):
            # Apply input formatting & scripts before sending to tokenizer
            if(vars.actionmode == 0):
                data = applyinputformatting(data)
            # Store the result in the Action log
            if(len(vars.prompt.strip()) == 0):
                vars.prompt = data
            else:
                vars.actions.append(data)
            update_story_chunk('last')

        if(not vars.noai):
            # Off to the tokenizer!
            calcsubmit(data)
            emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
        else:
            set_aibusy(0)
            emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)

#==================================================================#
#  
#==================================================================#
def actionretry(data):
    if(vars.noai):
        emit('from_server', {'cmd': 'errmsg', 'data': "Retry function unavailable in Read Only mode."})
        return
    if(vars.aibusy):
        return
    # Remove last action if possible and resubmit
    if(vars.gamestarted if vars.useprompt else len(vars.actions) > 0):
        set_aibusy(1)
        if(not vars.recentback and len(vars.actions) != 0 and len(vars.genseqs) == 0):  # Don't pop if we're in the "Select sequence to keep" menu or if there are no non-prompt actions
            last_key = vars.actions.get_last_key()
            vars.actions.pop()
            remove_story_chunk(last_key + 1)
        vars.genseqs = []
        calcsubmit('')
        emit('from_server', {'cmd': 'scrolldown', 'data': ''}, broadcast=True)
        vars.recentback = False
        vars.recentedit = False
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
        last_key = vars.actions.get_last_key()
        vars.actions.pop()
        vars.recentback = True
        remove_story_chunk(last_key + 1)
    elif(len(vars.genseqs) == 0):
        emit('from_server', {'cmd': 'errmsg', 'data': "Cannot delete the prompt."})
    else:
        vars.genseqs = []

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
        anotetxt  = "\n[Author's note: "+vars.authornote+"]\n"
    else:
        anotetxt = ""

    return winfo, mem, anotetxt, found_entries

def calcsubmitbudget(actionlen, winfo, mem, anotetxt, actions):
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    anotetkns    = []  # Placeholder for Author's Note tokens
    lnanote      = 0   # Placeholder for Author's Note length

    # Calculate token budget
    prompttkns = tokenizer.encode(vars.comregex_ai.sub('', vars.prompt))
    lnprompt   = len(prompttkns)
    
    memtokens = tokenizer.encode(mem)
    lnmem     = len(memtokens)
    
    witokens  = tokenizer.encode(winfo)
    lnwi      = len(witokens)
    
    if(anotetxt != ""):
        anotetkns = tokenizer.encode(anotetxt)
        lnanote   = len(anotetkns)
    
    lnsp = vars.sp.shape[0] if vars.sp is not None else 0
    
    if(vars.useprompt):
        budget = vars.max_length - lnsp - lnprompt - lnmem - lnanote - lnwi - vars.genamt
    else:
        budget = vars.max_length - lnsp - lnmem - lnanote - lnwi - vars.genamt

    if(actionlen == 0):
        # First/Prompt action
        subtxt = vars.memory + winfo + anotetxt + vars.comregex_ai.sub('', vars.prompt)
        lnsub  = lnsp + lnmem + lnwi + lnprompt + lnanote
        return subtxt, lnsub+1, lnsub+vars.genamt
    else:
        tokens     = []
        
        # Check if we have the action depth to hit our A.N. depth
        if(anotetxt != "" and actionlen < vars.andepth):
            forceanote = True
        
        # Get most recent action tokens up to our budget
        n = 0
        for key in reversed(actions):
            chunk = vars.comregex_ai.sub('', actions[key])
            
            if(budget <= 0):
                break
            acttkns = tokenizer.encode(chunk)
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
        ln = len(tokens) + lnsp
        return tokenizer.decode(tokens), ln+1, ln+vars.genamt

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
        subtxt, min, max = calcsubmitbudget(actionlen, winfo, mem, anotetxt, vars.actions)
        if(actionlen == 0):
            if(not vars.model in ["Colab", "OAI", "TPUMeshTransformerGPTJ"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(vars.model == "Colab"):
                sendtocolab(subtxt, min, max)
            elif(vars.model == "OAI"):
                oairequest(subtxt, min, max)
            elif(vars.model == "TPUMeshTransformerGPTJ"):
                tpumtjgenerate(subtxt, min, max, found_entries=found_entries)
        else:
            if(not vars.model in ["Colab", "OAI", "TPUMeshTransformerGPTJ"]):
                generate(subtxt, min, max, found_entries=found_entries)
            elif(vars.model == "Colab"):
                sendtocolab(subtxt, min, max)
            elif(vars.model == "OAI"):
                oairequest(subtxt, min, max)
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
def generate(txt, minimum, maximum, found_entries=None):    
    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(vars.numseqs))

    print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    vars.lastctx = txt
    
    # Clear CUDA cache if using GPU
    if(vars.hascuda and (vars.usegpu or vars.breakmodel)):
        gc.collect()
        torch.cuda.empty_cache()
    
    # Submit input text to generator
    try:
        gen_in = tokenizer.encode(txt, return_tensors="pt", truncation=True).long()
        if(vars.sp is not None):
            soft_tokens = torch.arange(
                model.config.vocab_size,
                model.config.vocab_size + vars.sp.shape[0],
            )
            gen_in = torch.cat((soft_tokens[None], gen_in), dim=-1)

        if(vars.hascuda and vars.usegpu):
            gen_in = gen_in.to(0)
        elif(vars.hascuda and vars.breakmodel):
            gen_in = gen_in.to(breakmodel.primary_device)
        else:
            gen_in = gen_in.to('cpu')

        model.kai_scanner_head_length = gen_in.shape[-1]
        model.kai_scanner_excluded_world_info = found_entries

        actions = vars.actions
        if(vars.dynamicscan):
            actions = actions.copy()

        with torch.no_grad():
            already_generated = 0
            numseqs = vars.numseqs
            while True:
                genout = generator(
                    gen_in, 
                    do_sample=True, 
                    min_length=minimum, 
                    max_length=maximum-already_generated,
                    repetition_penalty=vars.rep_pen,
                    bad_words_ids=vars.badwordsids,
                    use_cache=True,
                    num_return_sequences=numseqs
                    )
                already_generated += len(genout[0]) - len(gen_in[0])
                if(not model.kai_scanner.any_new_entries):
                    break
                assert genout.ndim >= 2
                assert genout.shape[0] == vars.numseqs
                encoded = []
                for i in range(vars.numseqs):
                    txt = tokenizer.decode(genout[i, -already_generated:])
                    winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True)
                    found_entries[i].update(_found_entries)
                    txt, _, _ = calcsubmitbudget(len(actions), winfo, mem, anotetxt, actions)
                    encoded.append(tokenizer.encode(txt, return_tensors="pt", truncation=True)[0].long().to(genout.device))
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
                diff = genout.shape[-1] - gen_in.shape[-1]
                minimum += diff
                maximum += diff
                gen_in = genout
                model.kai_scanner_head_length = encoded.shape[-1]
                numseqs = 1

    except Exception as e:
        emit('from_server', {'cmd': 'errmsg', 'data': 'Error occured during generator call, please check console.'}, broadcast=True)
        print("{0}{1}{2}".format(colors.RED, e, colors.END))
        set_aibusy(0)
        return
    
    # Need to manually strip and decode tokens if we're not using a pipeline
    #already_generated = -(len(gen_in[0]) - len(tokens))
    genout = [{"generated_text": tokenizer.decode(tokens[-already_generated:])} for tokens in genout]
    
    if(len(genout) == 1):
        genresult(genout[0]["generated_text"])
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
def genresult(genout):
    print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
    
    # Format output before continuing
    genout = applyoutputformatting(genout)
    
    # Add formatted text to Actions array and refresh the game screen
    if(len(vars.prompt.strip()) == 0):
        vars.prompt = genout
    else:
        vars.actions.append(genout)
    update_story_chunk('last')
    emit('from_server', {'cmd': 'texteffect', 'data': vars.actions.get_last_key() if len(vars.actions) else 0}, broadcast=True)

#==================================================================#
#  Send generator sequences to the UI for selection
#==================================================================#
def genselect(genout):
    i = 0
    for result in genout:
        # Apply output formatting rules to sequences
        result["generated_text"] = applyoutputformatting(result["generated_text"])
        print("{0}[Result {1}]\n{2}{3}".format(colors.CYAN, i, result["generated_text"], colors.END))
        i += 1
    
    # Store sequences in memory until selection is made
    vars.genseqs = genout
    
    # Send sequences to UI for selection
    emit('from_server', {'cmd': 'genseqs', 'data': genout}, broadcast=True)

#==================================================================#
#  Send selected sequence to action log and refresh UI
#==================================================================#
def selectsequence(n):
    if(len(vars.genseqs) == 0):
        return
    vars.actions.append(vars.genseqs[int(n)]["generated_text"])
    update_story_chunk('last')
    emit('from_server', {'cmd': 'texteffect', 'data': vars.actions.get_last_key() if len(vars.actions) else 0}, broadcast=True)
    emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True)
    vars.genseqs = []

#==================================================================#
#  Send transformers-style request to ngrok/colab host
#==================================================================#
def sendtocolab(txt, min, max):
    # Log request to console
    print("{0}Tokens:{1}, Txt:{2}{3}".format(colors.YELLOW, min-1, txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    vars.lastctx = txt
    
    # Build request JSON data
    reqdata = {
        'text': txt,
        'min': min,
        'max': max,
        'rep_pen': vars.rep_pen,
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
        
        if(len(genout) == 1):
            genresult(genout[0])
        else:
            # Convert torch output format to transformers
            seqs = []
            for seq in genout:
                seqs.append({"generated_text": seq})
            genselect(seqs)
        
        # Format output before continuing
        #genout = applyoutputformatting(getnewcontent(genout))
        
        # Add formatted text to Actions array and refresh the game screen
        #vars.actions.append(genout)
        #refresh_story()
        #emit('from_server', {'cmd': 'texteffect', 'data': vars.actions.get_last_key() if len(vars.actions) else 0})
        
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
    if(found_entries is None):
        found_entries = set()
    found_entries = tuple(found_entries.copy() for _ in range(vars.numseqs))

    print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, minimum, maximum, txt, colors.END))

    # Submit input text to generator
    try:
        if(vars.dynamicscan):
            raise ValueError("Dynamic world info scanning is not supported by the TPU backend yet")
        
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
            vars.sp = tensor
        soft_tokens = np.arange(
            tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"],
            tpu_mtj_backend.params["n_vocab"] + tpu_mtj_backend.params["n_vocab_padding"] + vars.sp_length,
            dtype=np.uint32
        )

        genout = tpu_mtj_backend.infer(
            txt,
            gen_len = maximum-minimum+1,
            temp=vars.temp,
            top_p=vars.top_p,
            top_k=vars.top_k,
            tfs=vars.tfs,
            numseqs=vars.numseqs,
            repetition_penalty=vars.rep_pen,
            soft_embeddings=vars.sp,
            soft_tokens=soft_tokens,
        )

    except Exception as e:
        emit('from_server', {'cmd': 'errmsg', 'data': 'Error occured during generator call, please check console.'}, broadcast=True)
        print("{0}{1}{2}".format(colors.RED, e, colors.END))
        set_aibusy(0)
        return

    genout = [{"generated_text": txt} for txt in genout]

    if(len(genout) == 1):
        genresult(genout[0]["generated_text"])
    else:
        genselect(genout)

    set_aibusy(0)


#==================================================================#
# Replaces returns and newlines with HTML breaks
#==================================================================#
def formatforhtml(txt):
    return txt.replace("\\r\\n", "<br/>").replace("\\r", "<br/>").replace("\\n", "<br/>").replace("\r\n", "<br/>").replace('\n', '<br/>').replace('\r', '<br/>')

#==================================================================#
# Strips submitted text from the text returned by the AI
#==================================================================#
def getnewcontent(txt):
    # If the submitted context was blank, then everything is new
    if(vars.lastctx == ""):
        return txt
    
    # Tokenize the last context and the generated content
    ctxtokens = tokenizer.encode(vars.lastctx)
    txttokens = tokenizer.encode(txt)
    dif       = (len(txttokens) - len(ctxtokens)) * -1
    
    # Remove the context from the returned text
    newtokens = txttokens[dif:]
    
    return tokenizer.decode(newtokens)

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
    if(vars.formatoptns["frmttriminc"]):
        txt = utils.trimincompletesentence(txt)
    # Replace blank lines
    if(vars.formatoptns["frmtrmblln"]):
        txt = utils.replaceblanklines(txt)
    # Remove special characters
    if(vars.formatoptns["frmtrmspch"]):
        txt = utils.removespecialchars(txt, vars)
	# Single Line Mode
    if(vars.formatoptns["singleline"]):
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
            return

        idx = (vars.actions.get_last_key() if len(vars.actions) else 0) + 1

    if idx == 0:
        text = vars.prompt
    else:
        # Actions are 0 based, but in chunks 0 is the prompt.
        # So the chunk index is one more than the corresponding action index.
        text = vars.actions[idx - 1]

    item = html.escape(text)
    item = vars.comregex_ui.sub(lambda m: '\n'.join('<comment>' + l + '</comment>' for l in m.group().split('\n')), item)  # Add special formatting to comments
    item = vars.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions

    chunk_text = f'<chunk n="{idx}" id="n{idx}" tabindex="-1">{formatforhtml(item)}</chunk>'
    emit('from_server', {'cmd': 'updatechunk', 'data': {'index': idx, 'html': chunk_text}}, broadcast=True)


#==================================================================#
# Signals the Game Screen to remove one of the chunks
#==================================================================#
def remove_story_chunk(idx: int):
    emit('from_server', {'cmd': 'removechunk', 'data': idx}, broadcast=True)


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
    emit('from_server', {'cmd': 'updatedynamicscan', 'data': vars.dynamicscan}, broadcast=True)
    
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
        vars.actions[vars.editln-1] = data
    
    vars.mode = "play"
    update_story_chunk(vars.editln)
    emit('from_server', {'cmd': 'texteffect', 'data': vars.editln}, broadcast=True)
    emit('from_server', {'cmd': 'editmode', 'data': 'false'})

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
        del vars.actions[vars.editln-1]
        vars.mode = "play"
        remove_story_chunk(vars.editln)
        emit('from_server', {'cmd': 'editmode', 'data': 'false'})

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
        vars.actions[chunk-1] = data
    
    update_story_chunk(chunk)
    emit('from_server', {'cmd': 'texteffect', 'data': chunk}, broadcast=True)
    emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)

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
        del vars.actions[chunk-1]
        remove_story_chunk(chunk)
        emit('from_server', {'cmd': 'editmode', 'data': 'false'}, broadcast=True)

#==================================================================#
#   Toggles the game mode for memory editing and sends UI commands
#==================================================================#
def togglememorymode():
    if(vars.mode == "play"):
        vars.mode = "memory"
        emit('from_server', {'cmd': 'memmode', 'data': 'true'}, broadcast=True)
        emit('from_server', {'cmd': 'setinputtext', 'data': vars.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote}, broadcast=True)
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
def addwiitem():
    ob = {"key": "", "keysecondary": "", "content": "", "num": len(vars.worldinfo), "init": False, "selective": False, "constant": False}
    vars.worldinfo.append(ob);
    emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True)

#==================================================================#
#   
#==================================================================#
def sendwi():
    # Cache len of WI
    ln = len(vars.worldinfo)
    
    # Clear contents of WI container
    emit('from_server', {'cmd': 'clearwi', 'data': ''}, broadcast=True)
    
    # If there are no WI entries, send an empty WI object
    if(ln == 0):
        addwiitem()
    else:
        # Send contents of WI array
        for wi in vars.worldinfo:
            ob = wi
            emit('from_server', {'cmd': 'addwiitem', 'data': ob}, broadcast=True)
        # Make sure last WI item is uninitialized
        if(vars.worldinfo[-1]["init"]):
            addwiitem()

#==================================================================#
#  Request current contents of all WI HTML elements
#==================================================================#
def requestwi():
    list = []
    for wi in vars.worldinfo:
        list.append(wi["num"])
    emit('from_server', {'cmd': 'requestwiitem', 'data': list})

#==================================================================#
#  Renumber WI items consecutively
#==================================================================#
def organizewi():
    if(len(vars.worldinfo) > 0):
        count = 0
        for wi in vars.worldinfo:
            wi["num"] = count
            count += 1
        

#==================================================================#
#  Extract object from server and send it to WI objects
#==================================================================#
def commitwi(ar):
    for ob in ar:
        vars.worldinfo[ob["num"]]["key"]          = ob["key"]
        vars.worldinfo[ob["num"]]["keysecondary"] = ob["keysecondary"]
        vars.worldinfo[ob["num"]]["content"]      = ob["content"]
        vars.worldinfo[ob["num"]]["selective"]    = ob["selective"]
        vars.worldinfo[ob["num"]]["constant"]     = ob.get("constant", False)
    # Was this a deletion request? If so, remove the requested index
    if(vars.deletewi >= 0):
        del vars.worldinfo[vars.deletewi]
        organizewi()
        # Send the new WI array structure
        sendwi()
        # And reset deletewi index
        vars.deletewi = -1

#==================================================================#
#  
#==================================================================#
def deletewi(num):
    if(num < len(vars.worldinfo)):
        # Store index of deletion request
        vars.deletewi = num
        # Get contents of WI HTML inputs
        requestwi()

#==================================================================#
#  Look for WI keys in text to generator 
#==================================================================#
def checkworldinfo(txt, force_use_txt=False):
    original_txt = txt

    # Dont go any further if WI is empty
    if(len(vars.worldinfo) == 0):
        return "", set()
    
    # Cache actions length
    ln = len(vars.actions)
    
    # Don't bother calculating action history if widepth is 0
    if(vars.widepth > 0):
        depth = vars.widepth
        # If this is not a continue, add 1 to widepth since submitted
        # text is already in action history @ -1
        if(not force_use_txt and (txt != "" and vars.prompt != txt)):
            txt    = ""
            depth += 1
        
        if(ln > 0):
            chunks = collections.deque()
            i = 0
            for key in reversed(vars.actions):
                chunk = vars.actions[key]
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
        if(wi.get("constant", False)):
            wimem = wimem + wi["content"] + "\n"
            found_entries.add(id(wi))
            continue

        if(wi["key"] != ""):
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
    # Maybe check for length at some point
    # For now just send it to storage
    vars.memory = data
    vars.mode = "play"
    emit('from_server', {'cmd': 'memmode', 'data': 'false'}, broadcast=True)
    
    # Ask for contents of Author's Note field
    emit('from_server', {'cmd': 'getanote', 'data': ''})

#==================================================================#
#  Commit changes to Author's Note
#==================================================================#
def anotesubmit(data):
    # Maybe check for length at some point
    # For now just send it to storage
    vars.authornote = data

#==================================================================#
#  Assembles game data into a request to InferKit API
#==================================================================#
def ikrequest(txt):
    # Log request to console
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
        print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
        vars.actions.append(genout)
        update_story_chunk('last')
        emit('from_server', {'cmd': 'texteffect', 'data': vars.actions.get_last_key() if len(vars.actions) else 0}, broadcast=True)
        
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
        print("{0}{1}{2}".format(colors.CYAN, genout, colors.END))
        vars.actions.append(genout)
        update_story_chunk('last')
        emit('from_server', {'cmd': 'texteffect', 'data': vars.actions.get_last_key() if len(vars.actions) else 0}, broadcast=True)
        
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
def saveas(name):
    # Check if filename exists already
    name = utils.cleanfilename(name)
    if(not fileops.saveexists(name) or (vars.saveow and vars.svowname == name)):
        # All clear to save
        e = saveRequest(fileops.storypath(name))
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
def saveRequest(savpath):    
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
        js["actions"]     = tuple(vars.actions.values())
        js["worldinfo"]   = []
		
        # Extract only the important bits of WI
        for wi in vars.worldinfo:
            if(wi["constant"] or wi["key"] != ""):
                js["worldinfo"].append({
                    "key": wi["key"],
                    "keysecondary": wi["keysecondary"],
                    "content": wi["content"],
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
        vars.lastact     = ""
        vars.lastctx     = ""

        del vars.actions
        vars.actions = structures.KoboldStoryRegister()
        actions = collections.deque(js["actions"])

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
        
        if("worldinfo" in js):
            num = 0
            for wi in js["worldinfo"]:
                vars.worldinfo.append({
                    "key": wi["key"],
                    "keysecondary": wi.get("keysecondary", ""),
                    "content": wi["content"],
                    "num": num,
                    "init": True,
                    "selective": wi.get("selective", False),
                    "constant": wi.get("constant", False)
                })
                num += 1
        
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
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': vars.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote}, broadcast=True)
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True)
        emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True)
        print("{0}Story loaded from {1}!{2}".format(colors.GREEN, filename, colors.END))

#==================================================================#
#  Load a soft prompt from a file
#==================================================================#
def spRequest(filename):
    if(len(filename) == 0):
        vars.sp = None
        vars.sp_length = 0
        return

    global np
    if 'np' not in globals():
        import numpy as np

    z, version, shape, fortran_order, dtype = fileops.checksp(filename, vars.modeldim)
    assert isinstance(z, zipfile.ZipFile)
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

    vars.sp_length = tensor.shape[0]

    if(vars.model in ("TPUMeshTransformerGPTJ",)):
        rows = tensor.shape[0]
        padding_amount = tpu_mtj_backend.params["seq"] - (tpu_mtj_backend.params["seq"] % -tpu_mtj_backend.params["cores_per_replica"]) - rows
        tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
        tensor = tensor.reshape(
            tpu_mtj_backend.params["cores_per_replica"],
            -1,
            tpu_mtj_backend.params["d_model"],
        )
        vars.sp = np.float32(tensor)
    else:
        vars.sp = torch.from_numpy(tensor)

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
        vars.actions     = structures.KoboldStoryRegister()
        vars.worldinfo   = []
        vars.lastact     = ""
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
                        "num": num,
                        "init": True,
                        "selective": wi.get("selective", False),
                        "constant": wi.get("constant", False)
                    })
                    num += 1
        
        # Clear import data
        vars.importjs = {}
        
        # Reset current save
        vars.savedir = getcwd()+"\stories"
        
        # Refresh game screen
        vars.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': vars.laststory}, broadcast=True)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': vars.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote}, broadcast=True)
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
        vars.actions     = structures.KoboldStoryRegister()
        vars.worldinfo   = []
        vars.lastact     = ""
        vars.lastctx     = ""
        
        num = 0
        for wi in js["worldInfos"]:
            vars.worldinfo.append({
                "key": wi["keys"],
                "keysecondary": wi.get("keysecondary", ""),
                "content": wi["entry"],
                "num": num,
                "init": True,
                "selective": wi.get("selective", False),
                "constant": wi.get("constant", False)
            })
            num += 1
        
        # Reset current save
        vars.savedir = getcwd()+"\stories"
        
        # Refresh game screen
        vars.laststory = None
        emit('from_server', {'cmd': 'setstoryname', 'data': vars.laststory}, broadcast=True)
        sendwi()
        emit('from_server', {'cmd': 'setmemory', 'data': vars.memory}, broadcast=True)
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote}, broadcast=True)
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
                    "num": num,
                    "init": True,
                    "selective": wi.get("selective", False),
                    "constant": wi.get("constant", False)
                })
                num += 1
        
        print("{0}".format(vars.worldinfo[0]))
                
        # Refresh game screen
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
    
    vars.authornote  = ""
    vars.worldinfo   = []
    vars.lastact     = ""
    vars.lastctx     = ""
    
    # Reset current save
    vars.savedir = getcwd()+"\stories"
    
    # Refresh game screen
    vars.laststory = None
    emit('from_server', {'cmd': 'setstoryname', 'data': vars.laststory}, broadcast=True)
    sendwi()
    emit('from_server', {'cmd': 'setmemory', 'data': vars.memory}, broadcast=True)
    emit('from_server', {'cmd': 'setanote', 'data': vars.authornote}, broadcast=True)
    setStartState()

def randomGameRequest(topic): 
    newGameRequest()
    vars.memory      = "You generate the following " + topic + " story concept :"
    actionsubmit("", force_submit=True)
    vars.memory      = ""

#==================================================================#
#  Final startup commands to launch Flask app
#==================================================================#
if __name__ == "__main__":
	
    # Load settings from client.settings
    loadmodelsettings()
    loadsettings()

    # Start Flask/SocketIO (Blocking, so this must be last method!)
    
    #socketio.run(app, host='0.0.0.0', port=5000)
    if(vars.remote):
        if(args.ngrok):
            from flask_ngrok import _run_ngrok
            cloudflare = _run_ngrok()
        else:
           from flask_cloudflared import _run_cloudflared
           cloudflare = _run_cloudflared(5000)
        with open('cloudflare.log', 'w') as cloudflarelog:
            cloudflarelog.write("KoboldAI has finished loading and is available in the following link : " + cloudflare)
            print(format(colors.GREEN) + "KoboldAI has finished loading and is available in the following link : " + cloudflare + format(colors.END))
        socketio.run(app, host='0.0.0.0', port=5000)
    else:
        import webbrowser
        webbrowser.open_new('http://localhost:5000')
        print("{0}Server started!\rYou may now connect with a browser at http://127.0.0.1:5000/{1}".format(colors.GREEN, colors.END))
        socketio.run(app)
