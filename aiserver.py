#==================================================================#
# KoboldAI Client
# Version: Dev-0.1
# By: KoboldAIDev
#==================================================================#

from os import path, getcwd
from tkinter import filedialog, messagebox
import tkinter as tk
import json
import torch

#==================================================================#
# Variables & Storage
#==================================================================#
# Terminal tags for colored text
class colors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKCYAN    = '\033[96m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

# Transformers models
modellist = [
    ["InferKit API (requires API key)", "InferKit", ""],
    ["GPT Neo 1.3B", "EleutherAI/gpt-neo-1.3B", "8GB"],
    ["GPT Neo 2.7B", "EleutherAI/gpt-neo-2.7B", "16GB"],
    ["GPT-2", "gpt2", "1.2GB"],
    ["GPT-2 Med", "gpt2-medium", "2GB"],
    ["GPT-2 Large", "gpt2-large", "16GB"],
    ["GPT-2 XL", "gpt2-xl", "16GB"],
    ["Custom Neo   (eg Neo-horni)", "NeoCustom", ""],
    ["Custom GPT-2 (eg CloverEdition)", "GPT2Custom", ""]
    ]

# Variables
class vars:
    lastact     = "" # The last action submitted to the generator
    model       = ""
    noai        = False # Runs the script without starting up the transformers pipeline
    aibusy      = False # Stops submissions while the AI is working
    max_length  = 500 # Maximum number of tokens to submit per action
    genamt      = 60  # Amount of text for each action to generate
    rep_pen     = 1.0 # Default generator repetition_penalty
    temp        = 0.9 # Default generator temperature
    top_p       = 1.0 # Default generator top_p
    gamestarted = False
    prompt      = ""
    memory      = ""
    authornote  = ""
    andepth     = 3     # How far back in history to append author's note
    actions     = []
    mode        = "play" # Whether the interface is in play, memory, or edit mode
    editln      = 0      # Which line was last selected in Edit Mode
    url         = "https://api.inferkit.com/v1/models/standard/generate" # InferKit API URL
    apikey      = ""     # API key to use for InferKit API calls
    savedir     = getcwd()+"\stories"
    hascuda     = False  # Whether torch has detected CUDA on the system
    usegpu      = False  # Whether to launch pipeline with GPU support
    custmodpth  = ""     # Filesystem location of custom model to run

#==================================================================#
# Function to get model selection at startup
#==================================================================#
def getModelSelection():
    print("    #   Model                   {0}\n    ==================================="
        .format("VRAM" if vars.hascuda else "    "))
    
    i = 1
    for m in modellist:
        if(vars.hascuda):
            print("    {0} - {1}\t\t{2}".format(i, m[0].ljust(15), m[2]))
        else:
            print("    {0} - {1}".format(i, m[0]))
        i += 1
    print(" ");
    modelsel = 0
    vars.model = ''
    while(vars.model == ''):
        modelsel = input("Model #> ")
        if(modelsel.isnumeric() and int(modelsel) > 0 and int(modelsel) <= len(modellist)):
            vars.model = modellist[int(modelsel)-1][1]
        else:
            print("{0}Please enter a valid selection.{1}".format(colors.FAIL, colors.ENDC))
    
    # If custom model was selected, get the filesystem location and store it
    if(vars.model == "NeoCustom" or vars.model == "GPT2Custom"):
        print("{0}Please choose the folder where pytorch_model.bin is located:{1}\n".format(colors.OKCYAN, colors.ENDC))
        
        root = tk.Tk()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(
            initialdir=getcwd(), 
            title="Select Model Folder", 
            )
        root.destroy()
        
        if(path != None and path != ""):
            # Save directory to vars
            vars.custmodpth = path
        else:
            # Print error and retry model selection
            print("{0}Model select cancelled!{1}".format(colors.FAIL, colors.ENDC))
            print("{0}Select an AI model to continue:{1}\n".format(colors.OKCYAN, colors.ENDC))
            getModelSelection()

#==================================================================#
# Startup
#==================================================================#

# Test for GPU support
print("{0}Looking for GPU support...{1}".format(colors.HEADER, colors.ENDC), end="")
vars.hascuda = torch.cuda.is_available()
if(vars.hascuda):
    print("{0}FOUND!{1}".format(colors.OKGREEN, colors.ENDC))
else:
    print("{0}NOT FOUND!{1}".format(colors.WARNING, colors.ENDC))

# Select a model to run
print("{0}Welcome to the KoboldAI Client!\nSelect an AI model to continue:{1}\n".format(colors.OKCYAN, colors.ENDC))
getModelSelection()

# If transformers model was selected & GPU available, ask to use CPU or GPU
if(vars.model != "InferKit" and vars.hascuda):
    print("{0}Use GPU or CPU for generation?:  (Default GPU){1}\n".format(colors.OKCYAN, colors.ENDC))
    print("    1 - GPU\n    2 - CPU\n")
    genselected = False
    while(genselected == False):
        genselect = input("Mode> ")
        if(genselect == ""):
            vars.usegpu = True
            genselected = True
        elif(genselect.isnumeric() and int(genselect) == 1):
            vars.usegpu = True
            genselected = True
        elif(genselect.isnumeric() and int(genselect) == 2):
            vars.usegpu = False
            genselected = True
        else:
            print("{0}Please enter a valid selection.{1}".format(colors.FAIL, colors.ENDC))

# Ask for API key if InferKit was selected
if(vars.model == "InferKit"):
    if(not path.exists("client.settings")):
        # If the client settings file doesn't exist, create it
        print("{0}Please enter your InferKit API key:{1}\n".format(colors.OKCYAN, colors.ENDC))
        vars.apikey = input("Key> ")
        # Write API key to file
        file = open("client.settings", "w")
        try:
            file.write("{\"apikey\": \""+vars.apikey+"\"}")
        finally:
            file.close()
    else:
        # Otherwise open it up and get the key
        file = open("client.settings", "r")
        vars.apikey = json.load(file)["apikey"]
        file.close()

# Set logging level to reduce chatter from Flask
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Start flask & SocketIO
print("{0}Initializing Flask... {1}".format(colors.HEADER, colors.ENDC), end="")
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
app = Flask(__name__)
app.config['SECRET KEY'] = 'secret!'
socketio = SocketIO(app)
print("{0}OK!{1}".format(colors.OKGREEN, colors.ENDC))

# Start transformers and create pipeline
if(vars.model != "InferKit"):
    if(not vars.noai):
        print("{0}Initializing transformers, please wait...{1}".format(colors.HEADER, colors.ENDC))
        from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM
        
        # If custom GPT Neo model was chosen
        if(vars.model == "NeoCustom"):
            model     = GPTNeoForCausalLM.from_pretrained(vars.custmodpth)
            tokenizer = GPT2Tokenizer.from_pretrained(vars.custmodpth)
            # Is CUDA available? If so, use GPU, otherwise fall back to CPU
            if(vars.hascuda and vars.usegpu):
                generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)
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
            if(vars.hascuda and vars.usegpu):
                generator = pipeline('text-generation', model=vars.model, device=0)
            else:
                generator = pipeline('text-generation', model=vars.model)
        
        print("{0}OK! {1} pipeline created!{2}".format(colors.OKGREEN, vars.model, colors.ENDC))
else:
    # Import requests library for HTTPS calls
    import requests
    
    # Set generator variables to match InferKit's capabilities
    vars.max_length = 3000
    vars.genamt     = 200

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
    print("{0}Client connected!{1}".format(colors.OKGREEN, colors.ENDC))
    emit('from_server', {'cmd': 'connected'})
    if(not vars.gamestarted):
        setStartState()
        refresh_settings()
    else:
        # Game in session, send current game data and ready state to browser
        refresh_story()
        refresh_settings()
        if(vars.mode == "play"):
            if(not vars.aibusy):
                emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'})
            else:
                emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'})
        elif(vars.mode == "edit"):
            emit('from_server', {'cmd': 'editmode', 'data': 'true'})
        elif(vars.mode == "memory"):
            emit('from_server', {'cmd': 'memmode', 'data': 'true'})

#==================================================================#
# Event triggered when browser SocketIO sends data to the server
#==================================================================#
@socketio.on('message')
def get_message(msg):
    print("{0}Data recieved:{1}{2}".format(colors.OKGREEN, msg, colors.ENDC))
    # Submit action
    if(msg['cmd'] == 'submit'):
        if(vars.mode == "play"):
            actionsubmit(msg['data'])
        elif(vars.mode == "edit"):
            editsubmit(msg['data'])
        elif(vars.mode == "memory"):
            memsubmit(msg['data'])
    # Retry Action
    elif(msg['cmd'] == 'retry'):
        if(vars.aibusy):
            return
        set_aibusy(1)
        # Remove last action if possible and resubmit
        if(len(vars.actions) > 0):
            vars.actions.pop()
            refresh_story()
            calcsubmit('')
    # Back/Undo Action
    elif(msg['cmd'] == 'back'):
        if(vars.aibusy):
            return
        # Remove last index of actions and refresh game screen
        if(len(vars.actions) > 0):
            vars.actions.pop()
            refresh_story()
    # EditMode Action
    elif(msg['cmd'] == 'edit'):
        if(vars.mode == "play"):
            vars.mode = "edit"
            emit('from_server', {'cmd': 'editmode', 'data': 'true'})
        elif(vars.mode == "edit"):
            vars.mode = "play"
            emit('from_server', {'cmd': 'editmode', 'data': 'false'})
    # EditLine Action
    elif(msg['cmd'] == 'editline'):
        editrequest(int(msg['data']))
    # DeleteLine Action
    elif(msg['cmd'] == 'delete'):
        deleterequest()
    elif(msg['cmd'] == 'memory'):
        togglememorymode()
    elif(msg['cmd'] == 'save'):
        saveRequest()
    elif(msg['cmd'] == 'load'):
        loadRequest()
    elif(msg['cmd'] == 'newgame'):
        newGameRequest()
    elif(msg['cmd'] == 'settemp'):
        vars.temperature = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltemp', 'data': msg['data']})
    elif(msg['cmd'] == 'settopp'):
        vars.top_p = float(msg['data'])
        emit('from_server', {'cmd': 'setlabeltopp', 'data': msg['data']})
    elif(msg['cmd'] == 'setreppen'):
        vars.rep_pen = float(msg['data'])
        emit('from_server', {'cmd': 'setlabelreppen', 'data': msg['data']})
    elif(msg['cmd'] == 'setoutput'):
        vars.genamt = int(msg['data'])
        emit('from_server', {'cmd': 'setlabeloutput', 'data': msg['data']})
    # Author's Note field update
    elif(msg['cmd'] == 'anote'):
        anotesubmit(msg['data'])
    # Author's Note depth update
    elif(msg['cmd'] == 'anotedepth'):
        vars.andepth = int(msg['data'])
        emit('from_server', {'cmd': 'setlabelanotedepth', 'data': msg['data']})

#==================================================================#
#   
#==================================================================#
def setStartState():
    emit('from_server', {'cmd': 'updatescreen', 'data': '<span>Welcome to <span class="color_cyan">KoboldAI Client</span>! You are running <span class="color_green">'+vars.model+'</span>.<br/>Please load a game or enter a prompt below to begin!</span>'})
    emit('from_server', {'cmd': 'setgamestate', 'data': 'start'})

#==================================================================#
# 
#==================================================================#
def actionsubmit(data):
    if(vars.aibusy):
        return
    set_aibusy(1)
    if(not vars.gamestarted):
        vars.gamestarted = True # Start the game
        vars.prompt = data # Save this first action as the prompt
        emit('from_server', {'cmd': 'updatescreen', 'data': 'Please wait, generating story...'}) # Clear the startup text from game screen
        calcsubmit(data) # Run the first action through the generator
    else:
        # Dont append submission if it's a blank/continue action
        if(data != ""):
            vars.actions.append(data)
        calcsubmit(data)

#==================================================================#
# Take submitted text and build the text to be given to generator
#==================================================================#
def calcsubmit(txt):
    vars.lastact = txt   # Store most recent action in memory (is this still needed?)
    anotetxt     = ""    # Placeholder for Author's Note text
    lnanote      = 0     # Placeholder for Author's Note length
    forceanote   = False # In case we don't have enough actions to hit A.N. depth
    anoteadded   = False # In case our budget runs out before we hit A.N. depth
    actionlen    = len(vars.actions)
    
    # Build Author's Note if set
    if(vars.authornote != ""):
        anotetxt  = "\n[Author's note: "+vars.authornote+"]\n"
    
    # For all transformers models
    if(vars.model != "InferKit"):
        anotetkns    = []  # Placeholder for Author's Note tokens
        
        # Calculate token budget
        prompttkns = tokenizer.encode(vars.prompt)
        lnprompt   = len(prompttkns)
        
        memtokens = tokenizer.encode(vars.memory)
        lnmem     = len(memtokens)
        
        if(anotetxt != ""):
            anotetkns = tokenizer.encode(anotetxt)
            lnanote   = len(anotetkns)
        
        budget = vars.max_length - lnprompt - lnmem - lnanote - vars.genamt
        
        if(actionlen == 0):
            # First/Prompt action
            subtxt = vars.memory + anotetxt + vars.prompt
            lnsub  = lnmem + lnprompt + lnanote
            
            generate(subtxt, lnsub+1, lnsub+vars.genamt)
        else:
            tokens     = []
            
            # Check if we have the action depth to hit our A.N. depth
            if(anotetxt != "" and actionlen < vars.andepth):
                forceanote = True
            
            # Get most recent action tokens up to our budget
            for n in range(actionlen):
                
                if(budget <= 0):
                    break
                acttkns = tokenizer.encode(vars.actions[(-1-n)])
                tknlen = len(acttkns)
                if(tknlen < budget):
                    tokens = acttkns + tokens
                    budget -= tknlen
                else:
                    count = budget * -1
                    tokens = acttkns[count:] + tokens
                    break
                
                # Inject Author's Note if we've reached the desired depth
                if(n == vars.andepth-1):
                    if(anotetxt != ""):
                        tokens = anotetkns + tokens # A.N. len already taken from bdgt
                        anoteadded = True
                    
            # Did we get to add the A.N.? If not, do it here
            if(anotetxt != ""):
                if((not anoteadded) or forceanote):
                    tokens = memtokens + anotetkns + prompttkns + tokens
                else:
                    tokens = memtokens + prompttkns + tokens
            else:
                # Prepend Memory and Prompt before action tokens
                tokens = memtokens + prompttkns + tokens
            
            # Send completed bundle to generator
            ln = len(tokens)
            generate (
                tokenizer.decode(tokens),
                ln+1,
                ln+vars.genamt
                )
    # For InferKit web API
    else:
        
        # Check if we have the action depth to hit our A.N. depth
        if(anotetxt != "" and actionlen < vars.andepth):
            forceanote = True
        
        budget = vars.max_length - len(vars.prompt) - len(anotetxt) - len(vars.memory) - 1
        subtxt = ""
        for n in range(actionlen):
            
            if(budget <= 0):
                    break
            actlen = len(vars.actions[(-1-n)])
            if(actlen < budget):
                subtxt = vars.actions[(-1-n)] + subtxt
                budget -= actlen
            else:
                count = budget * -1
                subtxt = vars.actions[(-1-n)][count:] + subtxt
                break
            
            # Inject Author's Note if we've reached the desired depth
            if(n == vars.andepth-1):
                if(anotetxt != ""):
                    subtxt = anotetxt + subtxt # A.N. len already taken from bdgt
                    anoteadded = True
        
        # Format memory for inclusion (adding newline separator)
        memsub = ""
        if(vars.memory != ""):
            memsub = vars.memory + "\n"
        
        # Did we get to add the A.N.? If not, do it here
        if(anotetxt != ""):
            if((not anoteadded) or forceanote):
                subtxt = memsub + anotetxt + vars.prompt + subtxt
            else:
                subtxt = memsub + vars.prompt + subtxt
        else:
            subtxt = memsub + vars.prompt + subtxt
        
        # Send it!
        ikrequest(subtxt)

#==================================================================#
# Send text to generator and deal with output
#==================================================================#
def generate(txt, min, max):    
    print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.WARNING, min, max, txt, colors.ENDC))
    
    # Clear CUDA cache if using GPU
    if(vars.hascuda and vars.usegpu):
        torch.cuda.empty_cache()
    
    # Submit input text to generator
    genout = generator(
        txt, 
        do_sample=True, 
        min_length=min, 
        max_length=max,
        repetition_penalty=vars.rep_pen,
        temperature=vars.temp
        )[0]["generated_text"]
    print("{0}{1}{2}".format(colors.OKCYAN, genout, colors.ENDC))
    vars.actions.append(getnewcontent(genout))
    refresh_story()
    emit('from_server', {'cmd': 'texteffect', 'data': len(vars.actions)})
    
    # Clear CUDA cache again if using GPU
    if(vars.hascuda and vars.usegpu):
        torch.cuda.empty_cache()
    
    set_aibusy(0)

#==================================================================#
# Replaces returns and newlines with HTML breaks
#==================================================================#
def formatforhtml(txt):
    return txt.replace("\\r", "<br/>").replace("\\n", "<br/>").replace('\n', '<br/>').replace('\r', '<br/>')

#==================================================================#
#  Strips submitted text from the text returned by the AI
#==================================================================#
def getnewcontent(txt):
    ln = len(vars.actions)
    if(ln == 0):
        delim = vars.prompt
    else:
        delim = vars.actions[-1]
    
    return (txt.split(delim)[-1])

#==================================================================#
# Sends the current story content to the Game Screen
#==================================================================#
def refresh_story():
    txt = '<chunk n="0" id="n0">'+vars.prompt+'</chunk>'
    i = 1
    for item in vars.actions:
        txt = txt + '<chunk n="'+str(i)+'" id="n'+str(i)+'">'+item+'</chunk>'
        i += 1
    emit('from_server', {'cmd': 'updatescreen', 'data': formatforhtml(txt)})

#==================================================================#
# Sends the current generator settings to the Game Menu
#==================================================================#
def refresh_settings():
    emit('from_server', {'cmd': 'updatetemp', 'data': vars.temp})
    emit('from_server', {'cmd': 'updatetopp', 'data': vars.top_p})
    emit('from_server', {'cmd': 'updatereppen', 'data': vars.rep_pen})
    emit('from_server', {'cmd': 'updateoutlen', 'data': vars.genamt})
    emit('from_server', {'cmd': 'updatanotedepth', 'data': vars.andepth})

#==================================================================#
#  Sets the logical and display states for the AI Busy condition
#==================================================================#
def set_aibusy(state):
    if(state):
        vars.aibusy = True
        emit('from_server', {'cmd': 'setgamestate', 'data': 'wait'})
    else:
        vars.aibusy = False
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'})

#==================================================================#
# 
#==================================================================#
def editrequest(n):
    if(n == 0):
        txt = vars.prompt
    else:
        txt = vars.actions[n-1]
    
    vars.editln = n
    emit('from_server', {'cmd': 'setinputtext', 'data': txt})
    emit('from_server', {'cmd': 'enablesubmit', 'data': ''})

#==================================================================#
# 
#==================================================================#
def editsubmit(data):
    if(vars.editln == 0):
        vars.prompt = data
    else:
        vars.actions[vars.editln-1] = data
    
    vars.mode = "play"
    refresh_story()
    emit('from_server', {'cmd': 'texteffect', 'data': vars.editln})
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
        refresh_story()
        emit('from_server', {'cmd': 'editmode', 'data': 'false'})

#==================================================================#
#   Toggles the game mode for memory editing and sends UI commands
#==================================================================#
def togglememorymode():
    if(vars.mode == "play"):
        vars.mode = "memory"
        emit('from_server', {'cmd': 'memmode', 'data': 'true'})
        emit('from_server', {'cmd': 'setinputtext', 'data': vars.memory})
        emit('from_server', {'cmd': 'setanote', 'data': vars.authornote})
    elif(vars.mode == "memory"):
        vars.mode = "play"
        emit('from_server', {'cmd': 'memmode', 'data': 'false'})

#==================================================================#
#  Commit changes to Memory storage
#==================================================================#
def memsubmit(data):
    # Maybe check for length at some point
    # For now just send it to storage
    vars.memory = data
    vars.mode = "play"
    emit('from_server', {'cmd': 'memmode', 'data': 'false'})
    
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
    print("{0}Len:{1}, Txt:{2}{3}".format(colors.WARNING, len(txt), txt, colors.ENDC))
    
    # Build request JSON data
    reqdata = {
        'forceNoEnd': True,
        'length': vars.genamt,
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
        print("{0}{1}{2}".format(colors.OKCYAN, genout, colors.ENDC))
        vars.actions.append(genout)
        refresh_story()
        emit('from_server', {'cmd': 'texteffect', 'data': len(vars.actions)})
        
        set_aibusy(0)
    else:
        # Send error message to web client
        er = req.json()
        if("error" in er):
            code = er["error"]["extensions"]["code"]
        elif("errors" in er):
            code = er["errors"][0]["extensions"]["code"]
            
        errmsg = "InferKit API Error: {0} - {1}".format(req.status_code, code)
        emit('from_server', {'cmd': 'errmsg', 'data': errmsg})
        set_aibusy(0)

#==================================================================#
#  Forces UI to Play mode
#==================================================================#
def exitModes():
    if(vars.mode == "edit"):
        emit('from_server', {'cmd': 'editmode', 'data': 'false'})
    elif(vars.mode == "memory"):
        emit('from_server', {'cmd': 'memmode', 'data': 'false'})
    vars.mode = "play"

#==================================================================#
#  Save the story to a file
#==================================================================#
def saveRequest():    
    root = tk.Tk()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfile(
        initialdir=vars.savedir, 
        title="Save Story As", 
        filetypes = [("Json", "*.json")]
        )
    root.destroy()
    
    if(path != None and path != ''):
        # Leave Edit/Memory mode before continuing
        exitModes()
        # Save path for future saves
        vars.savedir = path
        # Build json to write
        js = {}
        #js["maxlegth"]    = vars.max_length # This causes problems when switching to/from InfraKit
        #js["genamt"]      = vars.genamt
        #js["rep_pen"]     = vars.rep_pen
        #js["temp"]        = vars.temp
        js["gamestarted"] = vars.gamestarted
        js["prompt"]      = vars.prompt
        js["memory"]      = vars.memory
        js["authorsnote"] = vars.authornote
        js["actions"]     = vars.actions
        #js["savedir"]     = path.name  # For privacy, don't include savedir in save file      
        # Write it
        file = open(path.name, "w")
        try:
            file.write(json.dumps(js))
        finally:
            file.close()

#==================================================================#
#  Load a stored story from a file
#==================================================================#
def loadRequest():    
    root = tk.Tk()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        initialdir=vars.savedir, 
        title="Select Story File", 
        filetypes = [("Json", "*.json")]
        )
    root.destroy()
    
    if(path != None and path != ''):
        # Leave Edit/Memory mode before continuing
        exitModes()
        # Read file contents into JSON object
        file = open(path, "r")
        js = json.load(file)
        # Copy file contents to vars
        #vars.max_length  = js["maxlegth"] # This causes problems when switching to/from InfraKit
        #vars.genamt      = js["genamt"]
        #vars.rep_pen     = js["rep_pen"]
        #vars.temp        = js["temp"]
        vars.gamestarted = js["gamestarted"]
        vars.prompt      = js["prompt"]
        vars.memory      = js["memory"]
        vars.actions     = js["actions"]
        #vars.savedir     = js["savedir"] # For privacy, don't include savedir in save file
        
        # Try not to break older save files
        if("authorsnote" in js):
            vars.authornote = js["authorsnote"]
        
        file.close()
        # Refresh game screen
        refresh_story()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'})

#==================================================================#
#  Starts a new story
#==================================================================#
def newGameRequest(): 
    # Ask for confirmation
    root = tk.Tk()
    root.attributes("-topmost", True)
    confirm = messagebox.askquestion("Confirm New Game", "Really start new Story?")
    root.destroy()
    
    if(confirm == "yes"):
        # Leave Edit/Memory mode before continuing
        exitModes()
        # Clear vars values
        vars.gamestarted = False
        vars.prompt      = ""
        vars.memory      = ""
        vars.actions     = []
        vars.savedir     = getcwd()+"\stories\\newstory.json"
        # Refresh game screen
        setStartState()


#==================================================================#
# Start Flask/SocketIO (Blocking, so this must be last method!)
#==================================================================#
if __name__ == "__main__":
    print("{0}Server started!\rYou may now connect with a browser at http://127.0.0.1:5000/{1}".format(colors.OKGREEN, colors.ENDC))
    socketio.run(app)
