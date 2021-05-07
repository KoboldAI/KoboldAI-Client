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
    scripts = { # Text form of javascript story scripts
      'shared': "",
      'inputModifier': "",
      'contextModifier': "",
      'outputModifier': "",
    }
    script_state = {} # Saved data from clientside scripts.
    last_context = "" # The last context received from the client. Used when retrying to avoid having the make + request a context again.

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
        refresh_story({'all': True})
        send_script_state()
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
            actionsubmit(msg['data'], msg['stop'])
        elif(vars.mode == "edit"):
            editsubmit(msg['data'])
        elif(vars.mode == "memory"):
            memsubmit(msg['data'])
    # Retry Action
    elif(msg['cmd'] == 'retry'):
        if(vars.aibusy):
            return
       # Remove last action if possible and resubmit
        if(len(vars.actions) > 0):
            vars.actions.pop()
            refresh_story({'actions': True})
            # Use last received context (if present)
            if(vars.last_context != ""):
              generate(vars.last_context)
            else:
              # Otherwise, make the context over again
              prepare_generation()
    # Back/Undo Action
    elif(msg['cmd'] == 'back'):
        if(vars.aibusy):
            return
        # Remove last index of actions and refresh game screen
        if(len(vars.actions) > 0):
            vars.actions.pop()
            refresh_story({'actions': True})
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
    elif(msg['cmd'] == 'generate'):
        generate(msg['data'])
    elif(msg['cmd'] == 'newoutput'):
        add_new_output(msg['data'])
    elif(msg['cmd'] == 'recordscriptstate'):
        # Update script state record
        vars.script_state = msg['data']

#==================================================================#
# Adds on a new action, treating it as a fresh output
#==================================================================#
def add_new_output(text):
    vars.actions.append(text)
    refresh_story({'actions': True})
    emit('from_server', {'cmd': 'texteffect', 'data': len(vars.actions)})

#==================================================================#
# Triggers generation of a new AI output using the provided input
#==================================================================#
def generate(context=""):
  if(vars.aibusy):
    return
  set_aibusy(1)

  # Format the context appropriately for the given model
  # For all transformers models
  if(vars.model != "InferKit"):
    format_return = format_transformers(context)
    vars.last_context = tokenizer.decode(format_return) # Store the context in case the player wants to retry this session.
  # For InferKit web API
  else:
    format_return = format_inferkit(context)
    vars.last_context = format_return # Store the context in case the player wants to retry this session.

  # Submit the formatted context to the appropriate model
  if(vars.model != "InferKit"): # transformer
    generated_output = generate_transformers(format_return)
  else: # InferKit
    generated_output = generate_inferkit(format_return)

  set_aibusy(0)
  # Don't add the output to actions yet
  # Instead, send it to the client to be processed through its outputModifier script
  emit('from_server', {'cmd': 'modoutput', 'data': generated_output})

#==================================================================#
# Formats given input for use in transformers models
#==================================================================#
def format_transformers(context):
  # (Is vars.genamt supposed to be tokens or text? Original `calcsubmit` function suggests its supposed to be tokens.)
  # (for now, treating it as tokens)
  tokens = tokenizer.encode(context)

  if ((len(tokens) + vars.genamt) > vars.max_length):
    # Cut off excess tokens so it doesn't go over the limit (removing tokens from the start)
    # while also allowing space for generated tokens
    tokens = tokens[len(tokens) - vars.max_length + vars.genamt:]

  return tokens

#==================================================================#
# Formats given input for use in the InferKit model
#==================================================================#
def format_inferkit(context):
  new_context = context

  if(len(new_context) > vars.max_length):
    # Cut off excess text so it doesn't go over the limit (cutting away text off the start)
    new_context = new_context[len(new_context) - vars.max_length:]

  return new_context

#==================================================================#
# Generates an output using the transformers model
#==================================================================#
def generate_transformers(tokens):
  token_length = len(tokens)

  txt = tokenizer.decode(tokens)
  min = token_length+1
  max = token_length+vars.genamt

  print("{0}Min:{1}, Max:{2}, Txt:{3}{4}".format(colors.WARNING, min, max, txt, colors.ENDC))

  # Clear CUDA cache if using GPU
  if(vars.hascuda and vars.usegpu):
    torch.cuda.empty_cache()

  # Submit input text to generator
  generator_output = generator(
    txt,
    do_sample=True,
    min_length=min,
    max_length=max,
    repetition_penalty=vars.rep_pen,
    temperature=vars.temp
    )[0]["generated_text"]
  print("{0}{1}{2}".format(colors.OKCYAN, generator_output, colors.ENDC))

  # Clear CUDA cache again if using GPU
  if(vars.hascuda and vars.usegpu):
    torch.cuda.empty_cache()

  set_aibusy(0)

  return getnewcontent(generator_output)

#==================================================================#
# Generates an output using the InferKit API
#==================================================================#
def generate_inferkit(txt):
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
        'topP': 0.9
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

        set_aibusy(0)

        return genout
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
# Sends request to edit base context to client (starting off chain leading to generating an output)
#==================================================================#
def prepare_generation():
  # Create a base context and send it off to the client to edit with its contextModifier script
  # If the client responds, generation will begin
  emit('from_server', {'cmd': 'modcontext', 'data': build_base_context()})

#==================================================================#
# Generates a context that will ultimately be fed to the generator after going through client's contextModifier
#==================================================================#
def build_base_context():
  if(vars.model != "InferKit"): #Transformers model
    return build_base_context_transformers()
  else: #InferKit API
    return build_base_context_inferkit()

#==================================================================#
# Creates a context for transformers-based models
#==================================================================#
def build_base_context_transformers():
  # Current rules:
  # - Memory cannot exceed half the max token amount, and the start will be cut off to fit
  # - Any remaining space will consist of as much history as possible (most recent actions, going back to and including prompt)
  # - Space is made for vars.genamt if necessary by shortening history
  # - Author's note will only be included if set
  # - If there's not enough depth for authors notes, it will be inserted after memory, before history
  # - Author note depth starts from last history entry, counting back. Depth 1 means author's note will be posted after all actions.
  # Order:
  # - Memory
  # - History
  # - Authors note (at its appropriate depth)
  # - History (remaining)

  # (Note: I couldn't work out if transformers max_length works in string length, or token amount.
  # Based on previous code, I assume it's token amount, which I'll be operating off for here)
  memory_tokens = []
  history_tokens = []

  remaining_tokens = vars.max_length - vars.genamt

  author_note_tokens = []
  author_note_text = "" # Placeholder for Author's Note text
  author_note_pos = -1 # Records number of entries from the end of history that author's note tokens should be inserted into

  # Build Author's Note if set
  if(vars.authornote != ""):
    author_note_text  = "\n[Author's note: "+vars.authornote+"]\n"
    author_note_tokens = tokenizer.encode(author_note_text)
    remaining_tokens -= len(author_note_tokens)

  # Memory
  memory_tokens = tokenizer.encode(vars.memory)[-int(vars.max_length / 2):] # Turn memory into tokens and cut any excess off the start

  # History
  history = vars.actions[:] # Copy actions list
  history.insert(0, vars.prompt) # Add the prompt to the start, for ease of use

  depth = 0 # Record depth for inserting authors note
  for current_history in reversed(history):
    if(remaining_tokens <= 0):
      break

    # Author's note
    depth += 1
    if(author_note_text != "" and depth == vars.andepth):
      # Record how far from the end that the author note tokens should be inserted
      author_note_pos = len(history_tokens)

    current_tokens = tokenizer.encode(current_history)
    history_tokens = current_tokens[-remaining_tokens:] + history_tokens # Insert up to remaining tokens (cutting off the start if necessary) to start (as we're working backwards)

    remaining_tokens -= len(current_tokens) # (will go below zero, but that should be fine for our purposes)

  # Add author's note before finishing
  if (author_note_text != ""):
    if (author_note_pos != -1):
      # Insert author's note tokens into the right place in history tokens
      history_tokens[len(history_tokens)-author_note_pos:len(history_tokens)-author_note_pos] = author_note_tokens

      return tokenizer.decode(memory_tokens + history_tokens)
    else:
      # Didn't get to appropriate depth!
      # Add the author's note between memory and history
      return tokenizer.decode(memory_tokens + author_note_tokens + history_tokens)
  else:
    # Return the combined memory + history, decoded back into text
    return tokenizer.decode(memory_tokens + history_tokens)

#==================================================================#
# Creates a context for the InferKit API
#==================================================================#
def build_base_context_inferkit():
  # Current rules:
  # - Memory cannot exceed half the max length, and the start will be cut off to fit
  # - Any remaining space will consist of as much history as possible (most recent actions, going back to and including prompt)
  # - History will have the start cut off to fit
  # - Author's note will only be included if set
  # - If there's not enough depth for authors notes, it will be inserted after memory, before history
  # - Author note depth starts from last history entry, counting back. Depth 1 means author's note will be posted after all actions.
  # Order:
  # - Memory
  # - History
  # - Authors note (at its appropriate depth)
  # - History (remaining)

  memory_section = ""
  history_section = ""
  author_note_text = "" # Placeholder for Author's Note text
  author_note_pos = -1 # Records number of entries from the end of history that author's note tokens should be inserted into

  remaining_length = vars.max_length

  # Build Author's Note if set
  if(vars.authornote != ""):
    author_note_text = "\n[Author's note: "+vars.authornote+"]\n"
    remaining_length -= len(author_note_text)

  # Memory
  memory_section = vars.memory[-int(vars.max_length / 2):]
  remaining_length -= len(memory_section)

  # History
  history = vars.actions[:] # Copy actions list
  history.insert(0, vars.prompt) # Add the prompt to the start, for ease of use

  depth = 0 # Record depth for inserting authors note
  for current_history in reversed(history):
    if(remaining_length <= 0):
      break

    # Author's notes
    depth += 1
    if(author_note_text != "" and depth == vars.andepth):
      # Record how far from the end that the author note tokens should be inserted
      author_note_pos = len(history_section)

    history_section = current_history[-remaining_length:] + history_section # Insert up to remaining length (cutting off the start if necessary) to start (as we're working backwards)
    remaining_length -= len(current_history) # (will go below zero, but that should be fine for our purposes)

  # Add author's note before finishing
  if (author_note_text != ""):
    if (author_note_pos != -1):
      # Insert author's note into the right place in the history
      # (Temporarily make it into a list so its easier to insert the text in a specific place)
      temp_list = list(history_section)
      temp_list.insert(len(history_section)-author_note_pos, author_note_text)

      history_section = "".join(temp_list)

      return memory_section + history_section
    else:
      # Didn't get to appropriate depth!
      # Add the author's note between memory and history
      return memory_section + author_note_text + history_section
  else:
    # Return the combined memory + history
    return memory_section + history_section
#==================================================================#
#
#==================================================================#
def setStartState():
    emit('from_server', {'cmd': 'updatescreen', 'data': '<span>Welcome to <span class="color_cyan">KoboldAI Client</span>! You are running <span class="color_green">'+vars.model+'</span>.<br/>Please load a game or enter a prompt below to begin!</span>'})

    # Send client basic script info, so they can execute scripts from the start
    vars.scripts = get_default_scripts()
    vars.script_state = {}
    send_script_state()

    update_client_data({'all': True}) # Update all their stored data (data from previous stories would otherwise persist until the start prompt is given)

    emit('from_server', {'cmd': 'setgamestate', 'data': 'start'})

#==================================================================#
#
#==================================================================#
def actionsubmit(text, should_stop):
    # Special case if game hasn't started yet:
    # Ignore should_stop and bypass requesting context. Generate right away using only given text.
    if(not vars.gamestarted):
        vars.gamestarted = True # Start the game
        vars.prompt = text # Save this first action as the prompt
        update_client_data({'prompt': True}) # Tell the client to record the prompt
        emit('from_server', {'cmd': 'updatescreen', 'data': 'Please wait, generating story...'}) # Clear the startup text from game screen
        prepare_generation() # Begin generation
    else:
        # Dont append submission if it's a blank/continue action
        if(text != ""):
            vars.actions.append(text)
            refresh_story({'actions': True}) # Update the client's view + action record

        if(not should_stop):
          # Continue on to requesting context from the client.
          # If a response is given, it'll eventually move on to generating
          prepare_generation()

#==================================================================#
# Replaces returns and newlines with HTML breaks
#==================================================================#
def formatforhtml(txt):
    return txt.replace("\\r", "<br/>").replace("\\n", "<br/>").replace('\n', '<br/>').replace('\r', '<br/>')

#==================================================================#
#  Strips submitted text from the text returned by the AI
#==================================================================#
def getnewcontent(txt):
    return (txt.split(vars.last_context)[-1])

#==================================================================#
#  Updates client's data records
#==================================================================#
def update_client_data(parameters=None):
  package = {'cmd': 'updatedata'}

  do_all = False
  if(parameters is None or parameters.get('all')):
    do_all = True

  if(do_all or parameters.get('actions')):
    package['actions'] = vars.actions

  if(do_all or parameters.get('prompt')):
    package['prompt'] = vars.prompt

  if(do_all or parameters.get('memory')):
    package['memory'] = vars.memory

  # TODO: Allow for each script to be handled individually
  if(do_all or parameters.get('scripts')):
    package['scripts'] = vars.scripts

  # (Technically could utilise the getter and setter events clientside for updating its record)
  if(do_all or parameters.get('authornote')):
    package['authornote'] = vars.authornote

  # script_state is handled elsewhere
  emit('from_server', package)

#==================================================================#
#  Sends the loaded script state to the client
#==================================================================#
def send_script_state():
  emit('from_server', {'cmd': 'setscriptstate', 'data': vars.script_state})

#==================================================================#
# Sends the current story content to the Game Screen
#==================================================================#
def refresh_story(data_parameters=None):
    update_client_data(data_parameters)

    emit('from_server', {'cmd': 'updatescreen'}) # Client handles generating the html

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
        update_client_data({'prompt': True})
    else:
        vars.actions[vars.editln-1] = data

    vars.mode = "play"
    refresh_story({'actions': True})
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
    update_client_data({'memory': True})

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
    update_client_data({'authornote': True})

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
        js["scripts"] = vars.scripts
        js["script_state"] = vars.script_state
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

        if("scripts" in js):
            vars.scripts = js["scripts"]
        else:
            vars.scripts = get_default_scripts()

        if("script_state" in js):
            vars.script_state = js["script_state"]
        else:
            vars.script_state = {}


        file.close()
        # Refresh game screen
        refresh_story({'all': True})
        send_script_state()
        emit('from_server', {'cmd': 'setgamestate', 'data': 'ready'})

#==================================================================#
#  Provides the default starting scripts, loaded from files
#==================================================================#

def get_default_scripts():
  path = getcwd()+"\\templates\\"
  scripts = {}

  file = open(path+"shared.js", "r")
  scripts['shared'] = file.read()
  file.close()

  file = open(path+"inputModifier.js", "r")
  scripts['inputModifier'] = file.read()
  file.close()

  file = open(path+"contextModifier.js", "r")
  scripts['contextModifier'] = file.read()
  file.close()

  file = open(path+"outputModifier.js", "r")
  scripts['outputModifier'] = file.read()
  file.close()

  return scripts

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
        vars.scripts = get_default_scripts()
        vars.script_state = {}
        vars.last_context = ""
        # Refresh game screen
        setStartState()


#==================================================================#
# Start Flask/SocketIO (Blocking, so this must be last method!)
#==================================================================#
if __name__ == "__main__":
    print("{0}Server started!\rYou may now connect with a browser at http://127.0.0.1:5000/{1}".format(colors.OKGREEN, colors.ENDC))
    socketio.run(app)
