#==================================================================#
# KoboldAI Client
# Version: 1.15.0
# By: KoboldAIDev
#==================================================================#

# External packages
from os import path, getcwd
import re
import tkinter as tk
from tkinter import messagebox
import json
import collections
from typing import Literal, Union

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
import breakmodel

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
    ["GPT Neo 1.3B", "EleutherAI/gpt-neo-1.3B", "8GB"],
    ["GPT Neo 2.7B", "EleutherAI/gpt-neo-2.7B", "16GB"],
    ["GPT-2", "gpt2", "1.2GB"],
    ["GPT-2 Med", "gpt2-medium", "2GB"],
    ["GPT-2 Large", "gpt2-large", "16GB"],
    ["GPT-2 XL", "gpt2-xl", "16GB"],
    ["InferKit API (requires API key)", "InferKit", ""],
    ["Custom Neo   (eg Neo-horni)", "NeoCustom", ""],
    ["Custom GPT-2 (eg CloverEdition)", "GPT2Custom", ""],
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
    badwords    = []     # Array of str/chr values that should be removed from output
    badwordsids = []     # Tokenized array of badwords
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
    formatoptns = {}     # Container for state of formatting options
    importnum   = -1     # Selection on import popup list
    importjs    = {}     # Temporary storage for import data
    loadselect  = ""     # Temporary storage for filename to load
    svowname    = ""     # Filename that was flagged for overwrite confirm
    saveow      = False  # Whether or not overwrite confirm has been displayed
    genseqs     = []     # Temporary storage for generated sequences
    recentback  = False  # Whether Back button was recently used without Submitting or Retrying after
    useprompt   = True   # Whether to send the full prompt with every submit action
    breakmodel  = False  # For GPU users, whether to use both system RAM and VRAM to conserve VRAM while offering speedup compared to CPU-only
    bmsupported = False  # Whether the breakmodel option is supported (GPT-Neo/GPT-J only, currently)
    acregex_ai  = re.compile(r'\n* *>(.|\n)*')  # Pattern for matching adventure actions from the AI so we can remove them
    acregex_ui  = re.compile(r'^ *(&gt;.*)$', re.MULTILINE)    # Pattern for matching actions in the HTML-escaped story so we can apply colouring, etc (make sure to encase part to format in parentheses)
    actionmode  = 1
    adventure   = False
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
# Startup
#==================================================================#

# Parsing Parameters
parser = argparse.ArgumentParser(description="KoboldAI Server")
parser.add_argument("--remote", action='store_true', help="Optimizes KoboldAI for Remote Play")
parser.add_argument("--model", help="Specify the Model Type to skip the Menu")
parser.add_argument("--path", help="Specify the Path for local models (For model NeoCustom or GPT2Custom)")
parser.add_argument("--cpu", action='store_true', help="By default unattended launches are on the GPU use this option to force CPU usage.")
parser.add_argument("--breakmodel", action='store_true', help="For models that support GPU-CPU hybrid generation, use this feature instead of GPU or CPU generation")
parser.add_argument("--breakmodel_layers", type=int, help="Specify the number of layers to commit to system RAM if --breakmodel is used")
args = parser.parse_args()
vars.model = args.model;

if args.remote:
    vars.remote = True;

# Select a model to run
if args.model:
    print("Welcome to KoboldAI!\nYou have selected the following Model:", vars.model)
    if args.path:
        print("You have selected the following path for your Model :", args.path)
        vars.custmodpth = args.path;
        vars.colaburl = args.path + "/request"; # Lets just use the same parameter to keep it simple

else:
    print("{0}Welcome to the KoboldAI Client!\nSelect an AI model to continue:{1}\n".format(colors.CYAN, colors.END))
    getModelSelection()

# If transformers model was selected & GPU available, ask to use CPU or GPU
if(not vars.model in ["InferKit", "Colab", "OAI", "ReadOnly"]):
    # Test for GPU support
    import torch
    print("{0}Looking for GPU support...{1}".format(colors.PURPLE, colors.END), end="")
    vars.hascuda = torch.cuda.is_available()
    vars.bmsupported = vars.model in ("EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B", "NeoCustom")
    if(vars.hascuda):
        print("{0}FOUND!{1}".format(colors.GREEN, colors.END))
    else:
        print("{0}NOT FOUND!{1}".format(colors.YELLOW, colors.END))
    
    if args.model:
        if(vars.hascuda):
            genselected = True
            vars.usegpu = True
            vars.breakmodel = False
        if(args.cpu):
            vars.usegpu = False
            vars.breakmodel = False
        if(vars.bmsupported and args.breakmodel):
            vars.usegpu = False
            vars.breakmodel = True
    elif(vars.hascuda):    
        if(vars.bmsupported):
            print(colors.YELLOW + "You're using a model that supports GPU-CPU hybrid generation!\nCurrently only GPT-Neo models and GPT-J-6B support this feature.")
        print("{0}Use GPU or CPU for generation?:  (Default GPU){1}".format(colors.CYAN, colors.END))
        if(vars.bmsupported):
            print(f"    1 - GPU\n    2 - CPU\n    3 - Both (slower than GPU-only but uses less VRAM)\n")
        else:
            print("    1 - GPU\n    2 - CPU\n")
        genselected = False

    if(vars.hascuda):
        while(genselected == False):
            genselect = input("Mode> ")
            if(genselect == ""):
                vars.breakmodel = False
                vars.usegpu = True
                genselected = True
            elif(genselect.isnumeric() and int(genselect) == 1):
                vars.breakmodel = False
                vars.usegpu = True
                genselected = True
            elif(genselect.isnumeric() and int(genselect) == 2):
                vars.breakmodel = False
                vars.usegpu = False
                genselected = True
            elif(vars.bmsupported and genselect.isnumeric() and int(genselect) == 3):
                vars.breakmodel = True
                vars.usegpu = False
                genselected = True
            else:
                print("{0}Please enter a valid selection.{1}".format(colors.RED, colors.END))

# Ask for API key if InferKit was selected
if(vars.model == "InferKit"):
    if(not path.exists("client.settings")):
        # If the client settings file doesn't exist, create it
        print("{0}Please enter your InferKit API key:{1}\n".format(colors.CYAN, colors.END))
        vars.apikey = input("Key> ")
        # Write API key to file
        file = open("client.settings", "w")
        try:
            js = {"apikey": vars.apikey}
            file.write(json.dumps(js, indent=3))
        finally:
            file.close()
    else:
        # Otherwise open it up
        file = open("client.settings", "r")
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
            file = open("client.settings", "w")
            try:
                file.write(json.dumps(js, indent=3))
            finally:
                file.close()

# Ask for API key if OpenAI was selected
if(vars.model == "OAI"):
    if(not path.exists("client.settings")):
        # If the client settings file doesn't exist, create it
        print("{0}Please enter your OpenAI API key:{1}\n".format(colors.CYAN, colors.END))
        vars.oaiapikey = input("Key> ")
        # Write API key to file
        file = open("client.settings", "w")
        try:
            js = {"oaiapikey": vars.oaiapikey}
            file.write(json.dumps(js, indent=3))
        finally:
            file.close()
    else:
        # Otherwise open it up
        file = open("client.settings", "r")
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
            file = open("client.settings", "w")
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
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
app = Flask(__name__)
app.config['SECRET KEY'] = 'secret!'
socketio = SocketIO(app)
print("{0}OK!{1}".format(colors.GREEN, colors.END))

# Start transformers and create pipeline
if(not vars.model in ["InferKit", "Colab", "OAI", "ReadOnly"]):
    if(not vars.noai):
        print("{0}Initializing transformers, please wait...{1}".format(colors.PURPLE, colors.END))
        from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, GPTNeoModel, AutoModel
        
        # If custom GPT Neo model was chosen
        if(vars.model == "NeoCustom"):
            model     = GPTNeoForCausalLM.from_pretrained(vars.custmodpth)
            tokenizer = GPT2Tokenizer.from_pretrained(vars.custmodpth)
            # Is CUDA available? If so, use GPU, otherwise fall back to CPU
            if(vars.hascuda):
                if(vars.usegpu):
                    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)
                elif(vars.breakmodel):  # Use both RAM and VRAM (breakmodel)
                    n_layers = model.config.num_layers
                    breakmodel.total_blocks = n_layers
                    model.half().to('cpu')
                    gc.collect()
                    model.transformer.wte.to(breakmodel.gpu_device)
                    model.transformer.ln_f.to(breakmodel.gpu_device)
                    if(hasattr(model, 'lm_head')):
                        model.lm_head.to(breakmodel.gpu_device)
                    if(not hasattr(model.config, 'rotary') or not model.config.rotary):
                        model.transformer.wpe.to(breakmodel.gpu_device)
                    gc.collect()
                    if(args.breakmodel_layers is not None):
                        breakmodel.ram_blocks = max(0, min(n_layers, args.breakmodel_layers))
                    else:
                        print(colors.CYAN + "\nHow many layers would you like to put into system RAM?")
                        print("The more of them you put into system RAM, the slower it will run,")
                        print("but it will require less VRAM")
                        print("(roughly proportional to number of layers).")
                        print(f"This model has{colors.YELLOW} {n_layers} {colors.CYAN}layers.{colors.END}\n")
                        while(True):
                            layerselect = input("# of layers> ")
                            if(layerselect.isnumeric() and 0 <= int(layerselect) <= n_layers):
                                breakmodel.ram_blocks = int(layerselect)
                                break
                            else:
                                print(f"{colors.RED}Please enter an integer between 0 and {n_layers}.{colors.END}")
                    print(f"{colors.PURPLE}Will commit{colors.YELLOW} {breakmodel.ram_blocks} {colors.PURPLE}of{colors.YELLOW} {n_layers} {colors.PURPLE}layers to system RAM.{colors.END}")
                    GPTNeoModel.forward = breakmodel.new_forward
                    generator = model.generate
                else:
                    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
            else:
                generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        # If custom GPT2 model was chosen
        elif(vars.model == "GPT2Custom"):
            model     = GPT2LMHeadModel.from_pretrained(vars.custmodpth)
            tokenizer = GPT2Tokenizer.from_pretrained(vars.custmodpth)
            # Is CUDA available? If so, use GPU, otherwise fall back to CPU
            if(vars.hascuda and vars.usegpu):
                generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)
            else:
                generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        # If base HuggingFace model was chosen
        else:
            # Is CUDA available? If so, use GPU, otherwise fall back to CPU
            tokenizer = GPT2Tokenizer.from_pretrained(vars.model)
            if(vars.hascuda):
                if(vars.usegpu):
                    generator = pipeline('text-generation', model=vars.model, device=0)
                elif(vars.breakmodel):  # Use both RAM and VRAM (breakmodel)
                    model = AutoModel.from_pretrained(vars.model)
                    n_layers = model.config.num_layers
                    breakmodel.total_blocks = n_layers
                    model.half().to('cpu')
                    gc.collect()
                    model.transformer.wte.to(breakmodel.gpu_device)
                    model.transformer.ln_f.to(breakmodel.gpu_device)
                    if(hasattr(model, 'lm_head')):
                        model.lm_head.to(breakmodel.gpu_device)
                    if(not hasattr(model.config, 'rotary') or not model.config.rotary):
                        model.transformer.wpe.to(breakmodel.gpu_device)
                    gc.collect()
                    if(args.breakmodel_layers is not None):
                        breakmodel.ram_blocks = max(0, min(n_layers, args.breakmodel_layers))
                    else:
                        print(colors.CYAN + "\nHow many layers would you like to put into system RAM?")
                        print("The more of them you put into system RAM, the slower it will run,")
                        print("but it will require less VRAM")
                        print("(roughly proportional to number of layers).")
                        print(f"This model has{colors.YELLOW} {n_layers} {colors.CYAN}layers.{colors.END}\n")
                        while(True):
                            layerselect = input("# of layers> ")
                            if(layerselect.isnumeric() and 0 <= int(layerselect) <= n_layers):
                                breakmodel.ram_blocks = int(layerselect)
                                break
                            else:
                                print(f"{colors.RED}Please enter an integer between 0 and {n_layers}.{colors.END}")
                    print(f"{colors.PURPLE}Will commit{colors.YELLOW} {breakmodel.ram_blocks} {colors.PURPLE}of{colors.YELLOW} {n_layers} {colors.PURPLE}layers to system RAM.{colors.END}")
                    GPTNeoModel.forward = breakmodel.new_forward
                    generator = model.generate
                else:
                    generator = pipeline('text-generation', model=vars.model)
            else:
                generator = pipeline('text-generation', model=vars.model)
        
        # Suppress Author's Note by flagging square brackets
        vocab         = tokenizer.get_vocab()
        vocab_keys    = vocab.keys()
        vars.badwords = gettokenids("[")
        for key in vars.badwords:
            vars.badwordsids.append([vocab[key]])
        
        print("{0}OK! {1} pipeline created!{2}".format(colors.GREEN, vars.model, colors.END))
else:
    # If we're running Colab or OAI, we still need a tokenizer.
    if(vars.model == "Colab"):
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    elif(vars.model == "OAI"):
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set up Flask routes
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

#============================ METHODS =============================#    

#==================================================================#
# Event triggered when browser SocketIO is loaded and connects to server
#==================================================================#
@socketio.on('connect')
def do_connect():
    print("{0}Client connected!{1}".format(colors.GREEN, colors.END))
    emit('from_server', {'cmd': 'connected'})
    if(vars.remote):
        emit('from_server', {'cmd': 'runs_remotely'})
    
    if(not vars.gamestarted):
        setStartState()
        sendsettings()
        refresh_settings()
        sendwi()
        vars.mode = "play"
    else:
        # Game in session, send current game data and ready state to browser
        refresh_story()
        sendsettings()
        refresh_settings()
        sendwi()
        if(vars.mode == "play"):
            if(not vars.aibusy):
                emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True)
            else:
                emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'}, broadcast=True)
        elif(vars.mode == "edit"):
            emit('from_server', {'cmd': 'editmode', 'data': 'true'}, broadcast=True)
        elif(vars.mode == "memory"):
            emit('from_server', {'cmd': 'memmode', 'data': 'true'}, broadcast=True)
        elif(vars.mode == "wi"):
            emit('from_server', {'cmd': 'wimode', 'data': 'true'}, broadcast=True)

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
    elif(msg['cmd'] == 'savetofile'):
        savetofile()
    elif(msg['cmd'] == 'loadfromfile'):
        loadfromfile()
    elif(msg['cmd'] == 'import'):
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
    elif(msg['cmd'] == 'loadselect'):
        vars.loadselect = msg["data"]
    elif(msg['cmd'] == 'loadrequest'):
        loadRequest(getcwd()+"/stories/"+vars.loadselect+".json")
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
    elif(msg['cmd'] == 'importwi'):
        wiimportrequest()
    
#==================================================================#
#  Send start message and tell Javascript to set UI state
#==================================================================#
def setStartState():
    txt = "<span>Welcome to <span class=\"color_cyan\">KoboldAI Client</span>! You are running <span class=\"color_green\">"+vars.model+"</span>.<br/>"
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
    
    # Write it
    file = open("client.settings", "w")
    try:
        file.write(json.dumps(js, indent=3))
    finally:
        file.close()

#==================================================================#
#  Read settings from client file JSON and send to vars
#==================================================================#
def loadsettings():
    if(path.exists("client.settings")):
        # Read file contents into JSON object
        file = open("client.settings", "r")
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
        
        file.close()

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
def actionsubmit(data, actionmode=0):
    # Ignore new submissions if the AI is currently busy
    if(vars.aibusy):
        return
    set_aibusy(1)

    vars.recentback = False
    vars.actionmode = actionmode

    # "Action" mode
    if(actionmode == 1):
        data = data.strip().lstrip('>')
        data = re.sub(r'\n+', ' ', data)
        data = f"\n\n> {data}\n"
    
    # If we're not continuing, store a copy of the raw input
    if(data != ""):
        vars.lastact = data
    
    if(not vars.gamestarted):
        # Start the game
        vars.gamestarted = True
        # Save this first action as the prompt
        vars.prompt = data
        if(not vars.noai):
            # Clear the startup text from game screen
            emit('from_server', {'cmd': 'updatescreen', 'gamestarted': vars.gamestarted, 'data': 'Please wait, generating story...'}, broadcast=True)
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
        vars.recentback = False
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
# Take submitted text and build the text to be given to generator
#==================================================================#
def calcsubmit(txt):
    anotetxt     = ""    # Placeholder for Author's Note text
    lnanote      = 0     # Placeholder for Author's Note length
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    actionlen    = len(vars.actions)
    
    # Scan for WorldInfo matches
    winfo = checkworldinfo(txt)
    
    # Add a newline to the end of memory
    if(vars.memory != "" and vars.memory[-1] != "\n"):
        mem = vars.memory + "\n"
    else:
        mem = vars.memory
    
    # Build Author's Note if set
    if(vars.authornote != ""):
        anotetxt  = "\n[Author's note: "+vars.authornote+"]\n"
    
    # For all transformers models
    if(vars.model != "InferKit"):
        anotetkns    = []  # Placeholder for Author's Note tokens
        
        # Calculate token budget
        prompttkns = tokenizer.encode(vars.prompt)
        lnprompt   = len(prompttkns)
        
        memtokens = tokenizer.encode(mem)
        lnmem     = len(memtokens)
        
        witokens  = tokenizer.encode(winfo)
        lnwi      = len(witokens)
        
        if(anotetxt != ""):
            anotetkns = tokenizer.encode(anotetxt)
            lnanote   = len(anotetkns)
        
        if(vars.useprompt):
            budget = vars.max_length - lnprompt - lnmem - lnanote - lnwi - vars.genamt
        else:
            budget = vars.max_length - lnmem - lnanote - lnwi - vars.genamt
        
        if(actionlen == 0):
            # First/Prompt action
            subtxt = vars.memory + winfo + anotetxt + vars.prompt
            lnsub  = lnmem + lnwi + lnprompt + lnanote
            
            if(not vars.model in ["Colab", "OAI"]):
                generate(subtxt, lnsub+1, lnsub+vars.genamt)
            elif(vars.model == "Colab"):
                sendtocolab(subtxt, lnsub+1, lnsub+vars.genamt)
            elif(vars.model == "OAI"):
                oairequest(subtxt, lnsub+1, lnsub+vars.genamt)
        else:
            tokens     = []
            
            # Check if we have the action depth to hit our A.N. depth
            if(anotetxt != "" and actionlen < vars.andepth):
                forceanote = True
            
            # Get most recent action tokens up to our budget
            n = 0
            for key in reversed(vars.actions):
                chunk = vars.actions[key]
                
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
            ln = len(tokens)
            
            if(not vars.model in ["Colab", "OAI"]):
                generate (
                    tokenizer.decode(tokens),
                    ln+1,
                    ln+vars.genamt
                    )
            elif(vars.model == "Colab"):
                sendtocolab(
                    tokenizer.decode(tokens),
                    ln+1,
                    ln+vars.genamt
                    )
            elif(vars.model == "OAI"):
                oairequest(
                    tokenizer.decode(tokens),
                    ln+1,
                    ln+vars.genamt
                    )
                    
    # For InferKit web API
    else:
        # Check if we have the action depth to hit our A.N. depth
        if(anotetxt != "" and actionlen < vars.andepth):
            forceanote = True
        
        if(vars.useprompt):
            budget = vars.ikmax - len(vars.prompt) - len(anotetxt) - len(mem) - len(winfo) - 1
        else:
            budget = vars.ikmax - len(anotetxt) - len(mem) - len(winfo) - 1
            
        subtxt = ""
        prompt = vars.prompt
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
                    prompt = vars.prompt[-budget:]
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
def generate(txt, min, max):    
    print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.YELLOW, min, max, txt, colors.END))
    
    # Store context in memory to use it for comparison with generated content
    vars.lastctx = txt
    
    # Clear CUDA cache if using GPU
    if(vars.hascuda and (vars.usegpu or vars.breakmodel)):
        gc.collect()
        torch.cuda.empty_cache()
    
    # Submit input text to generator
    try:
        top_p = vars.top_p if vars.top_p > 0.0 else None
        top_k = vars.top_k if vars.top_k > 0 else None
        tfs = vars.tfs if vars.tfs > 0.0 else None

        # generator() only accepts a torch tensor of tokens (long datatype) as
        # its first argument if we're using breakmodel, otherwise a string
        # is fine
        if(vars.hascuda and vars.breakmodel):
            gen_in = tokenizer.encode(txt, return_tensors="pt", truncation=True).long().to(breakmodel.gpu_device)
        else:
            gen_in = txt
		
        with torch.no_grad():
            genout = generator(
                gen_in, 
                do_sample=True, 
                min_length=min, 
                max_length=max,
                repetition_penalty=vars.rep_pen,
                top_p=top_p,
                top_k=top_k,
                tfs=tfs,
                temperature=vars.temp,
                bad_words_ids=vars.badwordsids,
                use_cache=True,
                return_full_text=False,
                num_return_sequences=vars.numseqs
                )
    except Exception as e:
        emit('from_server', {'cmd': 'errmsg', 'data': 'Error occured during generator call, please check console.'}, broadcast=True)
        print("{0}{1}{2}".format(colors.RED, e, colors.END))
        set_aibusy(0)
        return
    
    # Need to manually strip and decode tokens if we're not using a pipeline
    if(vars.hascuda and vars.breakmodel):
        genout = [{"generated_text": tokenizer.decode(tokens[len(gen_in[0])-len(tokens):])} for tokens in genout]
    
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
# Replaces returns and newlines with HTML breaks
#==================================================================#
def formatforhtml(txt):
    return txt.replace("\\r", "<br/>").replace("\\n", "<br/>").replace('\n', '<br/>').replace('\r', '<br/>')

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
    
    return txt

#==================================================================#
# Sends the current story content to the Game Screen
#==================================================================#
def refresh_story():
    text_parts = ['<chunk n="0" id="n0" tabindex="-1">', html.escape(vars.prompt), '</chunk>']
    for idx in vars.actions:
        item = vars.actions[idx]
        idx += 1
        item = html.escape(item)
        item = vars.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions
        text_parts.extend(('<chunk n="', str(idx), '" id="n', str(idx), '" tabindex="-1">', item, '</chunk>'))
    emit('from_server', {'cmd': 'updatescreen', 'gamestarted': vars.gamestarted, 'data': formatforhtml(''.join(text_parts))}, broadcast=True)


#==================================================================#
# Signals the Game Screen to update one of the chunks
#==================================================================#
def update_story_chunk(idx: Union[int, Literal['last']]):
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
    item = vars.acregex_ui.sub('<action>\\1</action>', item)  # Add special formatting to adventure actions

    chunk_text = f'<chunk n="{idx}" id="n{idx}" tabindex="-1">{formatforhtml(item)}</chunk>'
    emit('from_server', {'cmd': 'updatechunk', 'data': {'index': idx, 'html': chunk_text, 'last': (idx == (vars.actions.get_last_key() if len(vars.actions) else 0))}}, broadcast=True)


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
    
    emit('from_server', {'cmd': 'updatefrmttriminc', 'data': vars.formatoptns["frmttriminc"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtrmblln', 'data': vars.formatoptns["frmtrmblln"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtrmspch', 'data': vars.formatoptns["frmtrmspch"]}, broadcast=True)
    emit('from_server', {'cmd': 'updatefrmtadsnsp', 'data': vars.formatoptns["frmtadsnsp"]}, broadcast=True)
    
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
    chunk = int(chunk)
    if(chunk == 0):
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
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote})
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
    emit('from_server', {'cmd': 'requestwiitem', 'data': list}, broadcast=True)

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
def checkworldinfo(txt):
    # Dont go any further if WI is empty
    if(len(vars.worldinfo) == 0):
        return
    
    # Cache actions length
    ln = len(vars.actions)
    
    # Don't bother calculating action history if widepth is 0
    if(vars.widepth > 0):
        depth = vars.widepth
        # If this is not a continue, add 1 to widepth since submitted
        # text is already in action history @ -1
        if(txt != "" and vars.prompt != txt):
            txt    = ""
            depth += 1
        
        if(ln > 0):
            chunks = collections.deque()
            i = 0
            for key in reversed(vars.actions):
                chunk = vars.actions[key]
                chunks.appendleft(chunk)
                if(i == depth):
                    break
        
        if(ln >= depth):
            txt = "".join(chunks)
        elif(ln > 0):
            txt = vars.prompt + "".join(chunks)
        elif(ln == 0):
            txt = vars.prompt
    
    # Scan text for matches on WI keys
    wimem = ""
    for wi in vars.worldinfo:
        if(wi.get("constant", False)):
            wimem = wimem + wi["content"] + "\n"
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
                                found = True
                                break
                        if found:
                            break
                    else:
                        wimem = wimem + wi["content"] + "\n"
                        break
    
    return wimem
    
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
    emit('from_server', {'cmd': 'getanote', 'data': ''}, broadcast=True)

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
        saveRequest(getcwd()+"/stories/"+name+".json")
        emit('from_server', {'cmd': 'hidesaveas', 'data': ''})
        vars.saveow = False
        vars.svowname = ""
    else:
        # File exists, prompt for overwrite
        vars.saveow   = True
        vars.svowname = name
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
        
        # Write it
        file = open(savpath, "w")
        try:
            file.write(json.dumps(js, indent=3))
        finally:
            file.close()
        
        print("{0}Story saved to {1}!{2}".format(colors.GREEN, path.basename(savpath), colors.END))

#==================================================================#
#  Load a saved story via file browser
#==================================================================#
def getloadlist():
    emit('from_server', {'cmd': 'buildload', 'data': fileops.getstoryfiles()})

#==================================================================#
#  Load a saved story via file browser
#==================================================================#
def loadfromfile():
    loadpath = fileops.getloadpath(vars.savedir, "Select Story File", [("Json", "*.json")])
    loadRequest(loadpath)

#==================================================================#
#  Load a stored story from a file
#==================================================================#
def loadRequest(loadpath):
    if(loadpath):
        # Leave Edit/Memory mode before continuing
        exitModes()
        
        # Read file contents into JSON object
        file = open(loadpath, "r")
        js   = json.load(file)
        
        # Copy file contents to vars
        vars.gamestarted = js["gamestarted"]
        vars.prompt      = js["prompt"]
        vars.memory      = js["memory"]
        vars.worldinfo   = []
        vars.lastact     = ""
        vars.lastctx     = ""

        del vars.actions
        vars.actions = structures.KoboldStoryRegister()
        for s in js["actions"]:
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
        
        file.close()
        
        # Save path for save button
        vars.savedir = loadpath
        
        # Clear loadselect var
        vars.loadselect = ""
        
        # Refresh game screen
        sendwi()
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'}, broadcast=True)
        emit('from_server', {'cmd': 'hidegenseqs', 'data': ''}, broadcast=True)
        print("{0}Story loaded from {1}!{2}".format(colors.GREEN, path.basename(loadpath), colors.END))

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
        sendwi()
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
        sendwi()
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
    sendwi()
    setStartState()

def randomGameRequest(topic): 
    newGameRequest()
    vars.memory      = "You generate the following " + topic + " story concept :"
    actionsubmit("")
    vars.memory      = ""

#==================================================================#
#  Final startup commands to launch Flask app
#==================================================================#
if __name__ == "__main__":
	
    # Load settings from client.settings
    loadsettings()
    
    # Start Flask/SocketIO (Blocking, so this must be last method!)
    
    #socketio.run(app, host='0.0.0.0', port=5000)
    if(vars.remote):
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
