from flask_socketio import emit, join_room, leave_room, rooms
import os
import re

socketio = None

def clean_var_for_emit(value):
    if isinstance(value, KoboldStoryRegister):
        return value.to_json()
    elif isinstance(value, set):
        return list(value)
    else:
        return value

def process_variable_changes(classname, name, value, old_value):
    #Special Case for KoboldStoryRegister
    if isinstance(value, KoboldStoryRegister):
        print("resetting")
        socketio.emit("reset_story", {}, broadcast=True, room="UI_2")
        for i in range(len(value.actions)):
            socketio.emit("var_changed", {"classname": "actions", "name": "Selected Text", "old_value": None, "value": {"id": i, "text": value[i]}}, broadcast=True, room="UI_2")
            socketio.emit("var_changed", {"classname": "actions", "name": "Options", "old_value": None, "value": {"id": i, "options": value.actions[i]['Options']}}, broadcast=True, room="UI_2")
    else:
        #print("{}: {} changed from {} to {}".format(classname, name, old_value, value))
        #if name == "Selected Text":
        #    print({"classname": classname, "name": name, "old_value": clean_var_for_emit(old_value), "value": clean_var_for_emit(value)})
        socketio.emit("var_changed", {"classname": classname, "name": name, "old_value": clean_var_for_emit(old_value), "value": clean_var_for_emit(value)}, broadcast=True, room="UI_2")

        
class settings(object):
    def send_to_ui(self):
        if socketio is not None:
            for (name, value) in vars(self).items():
                if name not in self.local_only_variables and name[0] != "_":
                    process_variable_changes(self.__class__.__name__.replace("_settings", ""), name, value, None)
                    
    

class model_settings(settings):
    local_only_variables = ['badwordsids', 'apikey', '_class_init']
    settings_name = "model"
    __class_initialized = False
    def __init__(self):
        self.model       = ""     # Model ID string chosen at startup
        self.model_type  = ""     # Model Type (Automatically taken from the model config)
        self.modelconfig = {}     # Raw contents of the model's config.json, or empty dictionary if none found
        self.custmodpth  = ""     # Filesystem location of custom model to run
        self.max_length  = 2048    # Maximum number of tokens to submit per action
        self.ikmax       = 3000    # Maximum number of characters to submit to InferKit
        self.genamt      = 80      # Amount of text for each action to generate
        self.ikgen       = 200     # Number of characters for InferKit to generate
        self.rep_pen     = 1.1     # Default generator repetition_penalty
        self.rep_pen_slope = 0.7   # Default generator repetition penalty slope
        self.rep_pen_range = 1024  # Default generator repetition penalty range
        self.temp        = 0.5     # Default generator temperature
        self.top_p       = 0.9     # Default generator top_p
        self.top_k       = 0       # Default generator top_k
        self.top_a       = 0.0     # Default generator top-a
        self.tfs         = 1.0     # Default generator tfs (tail-free sampling)
        self.typical     = 1.0     # Default generator typical sampling threshold
        self.numseqs     = 1       # Number of sequences to ask the generator to create
        self.badwordsids = []
        self.fp32_model  = False  # Whether or not the most recently loaded HF model was in fp32 format
        self.url         = "https://api.inferkit.com/v1/models/standard/generate" # InferKit API URL
        self.oaiurl      = "" # OpenAI API URL
        self.oaiengines  = "https://api.openai.com/v1/engines"
        self.colaburl    = ""     # Ngrok url for Google Colab mode
        self.apikey      = ""     # API key to use for InferKit API calls
        self.oaiapikey   = ""     # API key to use for OpenAI API calls
        self.modeldim    = -1     # Embedding dimension of your model (e.g. it's 4096 for GPT-J-6B and 2560 for GPT-Neo-2.7B)
        self.sampler_order = [0, 1, 2, 3, 4, 5]
        self.newlinemode = "n"
        self.lazy_load   = True # Whether or not to use torch_lazy_loader.py for transformers models in order to reduce CPU memory usage
        self.revision    = None
        self.presets     = []   # Holder for presets
        self.selected_preset = ""
        
        #Must be at end of __init__
        self.__class_initialized = True
        
    def __setattr__(self, name, value):
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        if self.__class_initialized and name != '__class_initialized':
            #Put variable change actions here
            if name not in self.local_only_variables and name[0] != "_":
                process_variable_changes(self.__class__.__name__.replace("_settings", ""), name, value, old_value)
        
                        
class story_settings(settings):
    local_only_variables = []
    settings_name = "story"
    __class_initialized = False
    def __init__(self):
        self.lastact     = ""     # The last action received from the user
        self.submission  = ""     # Same as above, but after applying input formatting
        self.lastctx     = ""     # The last context submitted to the generator
        self.gamestarted = False  # Whether the game has started (disables UI elements)
        self.gamesaved   = True   # Whether or not current game is saved
        self.prompt      = ""     # Prompt
        self.memory      = ""     # Text submitted to memory field
        self.authornote  = ""     # Text submitted to Author's Note field
        self.authornotetemplate = "[Author's note: <|>]"  # Author's note template
        self.setauthornotetemplate = self.authornotetemplate  # Saved author's note template in settings
        self.andepth     = 3      # How far back in history to append author's note
        self.actions     = KoboldStoryRegister()  # Actions submitted by user and AI
        self.actions_metadata = {} # List of dictonaries, one dictonary for every action that contains information about the action like alternative options.
                              # Contains at least the same number of items as actions. Back action will remove an item from actions, but not actions_metadata
                              # Dictonary keys are:
                              # Selected Text: (text the user had selected. None when this is a newly generated action)
                              # Alternative Generated Text: {Text, Pinned, Previous Selection, Edited}
                              # 
        self.worldinfo   = []     # List of World Info key/value objects
        self.worldinfo_i = []     # List of World Info key/value objects sans uninitialized entries
        self.worldinfo_u = {}     # Dictionary of World Info UID - key/value pairs
        self.wifolders_d = {}     # Dictionary of World Info folder UID-info pairs
        self.wifolders_l = []     # List of World Info folder UIDs
        self.wifolders_u = {}     # Dictionary of pairs of folder UID - list of WI UID
        self.lua_edited  = set()  # Set of chunk numbers that were edited from a Lua generation modifier
        self.lua_deleted = set()  # Set of chunk numbers that were deleted from a Lua generation modifier
        self.generated_tkns = 0   # If using a backend that supports Lua generation modifiers, how many tokens have already been generated, otherwise 0
        self.deletewi    = None   # Temporary storage for UID to delete
        self.mode        = "play" # Whether the interface is in play, memory, or edit mode
        self.editln      = 0      # Which line was last selected in Edit Mode
        self.genseqs     = []     # Temporary storage for generated sequences
        self.recentback  = False  # Whether Back button was recently used without Submitting or Retrying after
        self.recentrng   = None   # If a new random game was recently generated without Submitting after, this is the topic used (as a string), otherwise this is None
        self.recentrngm  = None   # If a new random game was recently generated without Submitting after, this is the memory used (as a string), otherwise this is None
        self.useprompt   = False   # Whether to send the full prompt with every submit action
        self.chatmode    = False
        self.chatname    = "You"
        self.adventure   = False
        self.actionmode  = 1
        self.dynamicscan = False
        self.recentedit  = False
        
        #Must be at end of __init__
        self.__class_initialized = True
        
    def __setattr__(self, name, value):
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        if self.__class_initialized and name != '__class_initialized':
            #Put variable change actions here
            if name not in self.local_only_variables and name[0] != "_":
                process_variable_changes(self.__class__.__name__.replace("_settings", ""), name, value, old_value)
                    
class user_settings(settings):
    local_only_variables = []
    settings_name = "user"
    __class_initialized = False
    def __init__(self):
        self.wirmvwhtsp  = False             # Whether to remove leading whitespace from WI entries
        self.widepth     = 3                 # How many historical actions to scan for WI hits
        self.formatoptns = {'frmttriminc': True, 'frmtrmblln': False, 'frmtrmspch': False, 'frmtadsnsp': False, 'singleline': False}     # Container for state of formatting options
        self.importnum   = -1                # Selection on import popup list
        self.importjs    = {}                # Temporary storage for import data
        self.loadselect  = ""                # Temporary storage for story filename to load
        self.spselect    = ""                # Temporary storage for soft prompt filename to load
        self.svowname    = ""                # Filename that was flagged for overwrite confirm
        self.saveow      = False             # Whether or not overwrite confirm has been displayed
        self.autosave    = False             # Whether or not to automatically save after each action
        self.laststory   = None              # Filename (without extension) of most recent story JSON file we loaded
        self.sid         = ""                # session id for the socketio client (request.sid)
        self.username    = "Default User"    # Displayed Username
        self.nopromptgen = False
        self.rngpersist  = False
        self.nogenmod    = False
        self.debug       = False    # If set to true, will send debug information to the client for display
        
        #Must be at end of __init__
        self.__class_initialized = True
        
    def __setattr__(self, name, value):
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        if self.__class_initialized and name != '__class_initialized':
            #Put variable change actions here
            if name not in self.local_only_variables and name[0] != "_":
                process_variable_changes(self.__class__.__name__.replace("_settings", ""), name, value, old_value)
        

class system_settings(settings):
    local_only_variables = ['lua_state', 'lua_logname', 'lua_koboldbridge', 'lua_kobold', 'lua_koboldcore', 'regex_sl', 'acregex_ai', 'acregex_ui', 'comregex_ai', 'comregex_ui']
    settings_name = "system"
    __class_initialized = False
    def __init__(self):
        self.noai        = False  # Runs the script without starting up the transformers pipeline
        self.aibusy      = False  # Stops submissions while the AI is working
        self.serverstarted = False  # Whether or not the Flask server has started
        self.lua_state   = None   # Lua state of the Lua scripting system
        self.lua_koboldbridge = None  # `koboldbridge` from bridge.lua
        self.lua_kobold  = None   # `kobold` from` bridge.lua
        self.lua_koboldcore = None  # `koboldcore` from bridge.lua
        self.lua_logname = ...    # Name of previous userscript that logged to terminal
        self.lua_running = False  # Whether or not Lua is running (i.e. wasn't stopped due to an error)
        self.abort       = False  # Whether or not generation was aborted by clicking on the submit button during generation
        self.compiling   = False  # If using a TPU Colab, this will be set to True when the TPU backend starts compiling and then set to False again
        self.checking    = False  # Whether or not we are actively checking to see if TPU backend is compiling or not
        self.sp_changed  = False  # This gets set to True whenever a userscript changes the soft prompt so that check_for_sp_change() can alert the browser that the soft prompt has changed
        self.spfilename  = ""     # Filename of soft prompt to load, or an empty string if not using a soft prompt
        self.userscripts = []     # List of userscripts to load
        self.last_userscripts = []  # List of previous userscript filenames from the previous time userscripts were send via usstatitems
        self.corescript  = "default.lua"  # Filename of corescript to load
        
        self.gpu_device  = 0      # Which PyTorch device to use when using pure GPU generation
        self.savedir     = os.getcwd()+"\\stories"
        self.hascuda     = False  # Whether torch has detected CUDA on the system
        self.usegpu      = False  # Whether to launch pipeline with GPU support
        self.spselect    = ""     # Temporary storage for soft prompt filename to load
        self.spmeta      = None   # Metadata of current soft prompt, or None if not using a soft prompt
        self.sp          = None   # Current soft prompt tensor (as a NumPy array)
        self.sp_length   = 0      # Length of current soft prompt in tokens, or 0 if not using a soft prompt
        self.has_genmod  = False  # Whether or not at least one loaded Lua userscript has a generation modifier
        self.breakmodel  = False  # For GPU users, whether to use both system RAM and VRAM to conserve VRAM while offering speedup compared to CPU-only
        self.bmsupported = False  # Whether the breakmodel option is supported (GPT-Neo/GPT-J/XGLM/OPT only, currently)
        self.nobreakmodel = False  # Something specifically requested Breakmodel to be disabled (For example a models config)
        self.smandelete  = False  # Whether stories can be deleted from inside the browser
        self.smanrename  = False  # Whether stories can be renamed from inside the browser
        self.allowsp     = False  # Whether we are allowed to use soft prompts (by default enabled if we're using GPT-2, GPT-Neo or GPT-J)
        self.regex_sl    = re.compile(r'\n*(?<=.) *\n(.|\n)*')  # Pattern for limiting the output to a single line
        self.acregex_ai  = re.compile(r'\n* *>(.|\n)*')  # Pattern for matching adventure actions from the AI so we can remove them
        self.acregex_ui  = re.compile(r'^ *(&gt;.*)$', re.MULTILINE)    # Pattern for matching actions in the HTML-escaped story so we can apply colouring, etc (make sure to encase part to format in parentheses)
        self.comregex_ai = re.compile(r'(?:\n<\|(?:.|\n)*?\|>(?=\n|$))|(?:<\|(?:.|\n)*?\|>\n?)')  # Pattern for matching comments to remove them before sending them to the AI
        self.comregex_ui = re.compile(r'(&lt;\|(?:.|\n)*?\|&gt;)')  # Pattern for matching comments in the editor
        self.host        = False
        self.flaskwebgui = False
        self.welcome     = False # Custom Welcome Text (False is default)
        self.quiet       = False # If set will suppress any story text from being printed to the console (will only be seen on the client web page)
        self.use_colab_tpu  = os.environ.get("COLAB_TPU_ADDR", "") != "" or os.environ.get("TPU_NAME", "") != ""  # Whether or not we're in a Colab TPU instance or Kaggle TPU instance and are going to use the TPU rather than the CPU
        self.aria2_port  = 6799 #Specify the port on which aria2's RPC interface will be open if aria2 is installed (defaults to 6799)
        
        #Must be at end of __init__
        self.__class_initialized = True
        
    def __setattr__(self, name, value):
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        if self.__class_initialized and name != '__class_initialized':
            #Put variable change actions here
            if name not in self.local_only_variables and name[0] != "_":
                process_variable_changes(self.__class__.__name__.replace("_settings", ""), name, value, old_value)
        
        
class KoboldStoryRegister(object):
    def __init__(self, sequence=[]):
        self.actions = {}
        self.action_count = -1
        for item in sequence:
            self.append(item)
        
    def __str__(self):
        return "".join([x['Selected Text'] for ignore, x in sorted(self.actions.items())])
        
    def __repr__(self):
        return self.__str__()
        
    def __iter__(self):
        self.itter = -1
        return self
        
    def __next__(self):
        self.itter += 1
        if self.itter < len(self.actions):
            return self.itter
        else:
            raise StopIteration
        
    def __getitem__(self, i):
        return self.actions[i]["Selected Text"]
        
    def __setitem__(self, i, text):
        if i in self.actions:
            old_text = self.actions[i]["Selected Text"]
            self.actions[i]["Selected Text"] = text
            if "Options" in self.actions[i]:
                for j in range(len(self.actions[i]["Options"])):
                    if self.actions[i]["Options"][j]["text"] == text:
                        del self.actions[i]["Options"][j]
            if old_text != "":
                self.actions[i]["Options"].append({"text": old_text, "Pinned": False, "Previous Selection": False, "Edited": True})
        else:
            old_text = None
            self.actions[i] = {"Selected Text": text, "Options": []}
        process_variable_changes("actions", "Selected Text", {"id": i, "text": text}, {"id": i, "text": old_text})
    
    def __len__(self):
        return self.action_count if self.action_count >=0 else 0
    
    def __reversed__(self):
        return reversed(range(self.action_count+1))
    
    def values(self):
        return [self.actions[k]["Selected Text"] for k in self.actions]
    
    def to_json(self):
        return {"action_count": self.action_count, "actions": self.actions}
        
    def load_json(self, json_data):
        if type(json_data) == str:
            import json
            json_data = json.loads(json_data)
        #JSON forces keys to be strings, so let's fix that
        temp = {}
        for item in json_data['actions']:
            temp[int(item)] = json_data['actions'][item]
            process_variable_changes("actions", "Selected Text", {"id": int(item), "text": json_data['actions'][item]["Selected Text"]}, None)
            if "Options" in json_data['actions'][item]:
                process_variable_changes("actions", "Options", {"id": int(item), "options": json_data['actions'][item]["Options"]}, None)
            
        self.action_count = json_data['action_count']
        self.actions = temp
            
    def get_action(self, action_id):
        if action_id not in actions:
            return None
        if "Selected Text" not in self.actions[action_id]:
            return None
        return self.actions[action_id]["Selected Text"]
        
    def get_action_list(self):
        return [x['Selected Text'] for ignore, x in sorted(self.actions.items()) if x['Selected Text'] is not None]
    
    def append(self, text):
        self.clear_unused_options()
        self.action_count+=1
        if self.action_count in self.actions:
            self.actions[self.action_count]["Selected Text"] = text
            print("looking for old option that matches")
            for item in self.actions[self.action_count]["Options"]:
                if item['text'] == text:
                    print("found it")
                    old_options = self.actions[self.action_count]["Options"]
                    del item
                    print("old: ")
                    print(old_options)
                    print()
                    print("New: ")
                    print(self.actions[self.action_count]["Options"])
                    process_variable_changes("actions", "Options", {"id": self.action_count, "options": self.actions[self.action_count]["Options"]}, {"id": self.action_count, "options": old_options})
                    
        else:
            self.actions[self.action_count] = {"Selected Text": text, "Options": []}
        process_variable_changes("actions", "Selected Text", {"id": self.action_count, "text": text}, None)
    
    def append_options(self, option_list):
        if self.action_count+1 in self.actions:
            print("1")
            old_options = self.actions[self.action_count+1]["Options"]
            self.actions[self.action_count+1]['Options'].extend([{"text": x, "Pinned": False, "Previous Selection": False, "Edited": False} for x in option_list])
            for item in option_list:
                process_variable_changes("actions", "Options", {"id": self.action_count+1, "options": self.actions[self.action_count+1]["Options"]}, {"id": self.action_count+1, "options": old_options})
        else:
            print("2")
            old_options = None
            self.actions[self.action_count+1] = {"Selected Text": "", "Options": [{"text": x, "Pinned": False, "Previous Selection": False, "Edited": False} for x in option_list]}
        process_variable_changes("actions", "Options", {"id": self.action_count+1, "options": self.actions[self.action_count+1]["Options"]}, {"id": self.action_count+1, "options": old_options})
            
    def clear_unused_options(self, pointer=None):
        new_options = []
        old_options = None
        if pointer is None:
            pointer = self.action_count+1
        if pointer in self.actions:
            old_options = self.actions[pointer]["Options"]
            self.actions[pointer]["Options"] = [x for x in self.actions[pointer]["Options"] if x["Pinned"] or x["Previous Selection"] or x["Edited"]]
            new_options = self.actions[pointer]["Options"]
        process_variable_changes("actions", "Options", {"id": pointer, "options": new_options}, {"id": pointer, "options": old_options})
    
    def set_pin(self, action_step, option_number):
        if action_step in self.actions:
            if option_number < len(self.actions[action_step]['Options']):
                old_options = self.actions[action_step]["Options"]
                self.actions[action_step]['Options'][option_number]['Pinned'] = True
                process_variable_changes("actions", "Options", {"id": action_step, "options": self.actions[action_step]["Options"]}, {"id": action_step, "options": old_options})
    
    def unset_pin(self, action_step, option_number):
        if action_step in self.actions:
            old_options = self.actions[action_step]["Options"]
            if option_number < len(self.actions[action_step]['Options']):
                self.actions[action_step]['Options'][option_number]['Pinned'] = False
                process_variable_changes("actions", "Options", {"id": action_step, "options": self.actions[action_step]["Options"]}, {"id": action_step, "options": old_options})
    
    def use_option(self, action_step, option_number):
        if action_step in self.actions:
            old_options = self.actions[action_step]["Options"]
            old_text = self.actions[action_step]["Selected Text"]
            if option_number < len(self.actions[action_step]['Options']):
                self.actions[action_step]["Selected Text"] = self.actions[action_step]['Options'][option_number]['text']
                del self.actions[action_step]['Options'][option_number]
                process_variable_changes("actions", "Options", {"id": action_step, "options": self.actions[action_step]["Options"]}, {"id": action_step, "options": old_options})
                process_variable_changes("actions", "Selected Text", {"id": action_step, "text": self.actions[action_step]["Selected Text"]}, {"id": action_step, "Selected Text": old_text})
    
    def delete_action(self, action_id):
        if action_id in self.actions:
            old_options = self.actions[action_id]["Options"]
            old_text = self.actions[action_id]["Selected Text"]
            self.actions[action_id]["Options"].append({"text": self.actions[action_id]["Selected Text"], "Pinned": False, "Previous Selection": True, "Edited": False})
            self.actions[action_id]["Selected Text"] = ""
            self.action_count -= 1
            process_variable_changes("actions", "Selected Text", {"id": action_id, "text": None}, {"id": action_id, "text": old_text})
            process_variable_changes("actions", "Options", {"id": action_id, "options": self.actions[action_id]["Options"]}, {"id": action_id, "options": old_options})
            
    def pop(self):
        if self.action_count >= 0:
            text = self.actions[self.action_count]
            self.delete_action(self.action_count)
            process_variable_changes("actions", "Selected Text", {"id": self.action_count, "text": None}, {"id": self.action_count, "text": text})
            return text
        else:
            return None
            
    def get_first_key(self):
        if self.action_count >= 0:
            text = ""
            i = 0
            while text == "" and i <= self.action_count:
                if "selected Text" in self.actions[i]:
                    text = self.actions[i]["Selected Text"]
                i+=1
            return text

    def get_last_key(self):
        if self.action_count >= 0:
            return self.action_count
        else:
            return 0
    
    def get_last_item(self):
        if self.action_count >= 0:
            return self.actions[self.action_count]
     
    def increment_id(self):
        self.action_count += 1
        
    def get_next_id(self):
        return self.action_count+1
        
    def set_next_id(self, x: int):
        self.action_count = x
        
    def get_options(self, action_id):
        if action_id in self.actions:
            return self.actions[action_id]["Options"]
        else:
            return []
    
    def get_current_options(self):
        if self.action_count+1 in self.actions:
            return self.actions[self.action_count+1]["Options"]
        else:
            return []
            
    def get_current_options_no_edits(self):
        if self.action_count+1 in self.actions:
            return [x for x in self.actions[self.action_count+1]["Options"] if x["Edited"] == False]
        else:
            return []
    
    def get_pins(self, action_id):
        if action_id in self.actions:
            return [x for x in self.actions[action_id]["Options"] if x["Pinned"]]
        else:
            return []
            
    def get_prev_selections(self, action_id):
        if action_id in self.actions:
            return [x for x in self.actions[action_id]["Options"] if x["Previous Selection"]]
        else:
            return []
    
    def get_edits(self, action_id):
        if action_id in self.actions:
            return [x for x in self.actions[action_id]["Options"] if x["Edited"]]
        else:
            return []

    def get_redo_options(self):
        pointer = max(self.actions)
        while pointer > self.action_count:
            if pointer in self.actions:
                for item in self.actions[pointer]["Options"]:
                    if item["Previous Selection"] or item["Pinned"]:
                        return self.actions[pointer]["Options"]
            pointer-=1
        return []


        
badwordsids_default = [[13460], [6880], [50256], [42496], [4613], [17414], [22039], [16410], [27], [29], [38430], [37922], [15913], [24618], [28725], [58], [47175], [36937], [26700], [12878], [16471], [37981], [5218], [29795], [13412], [45160], [3693], [49778], [4211], [20598], [36475], [33409], [44167], [32406], [29847], [29342], [42669], [685], [25787], [7359], [3784], [5320], [33994], [33490], [34516], [43734], [17635], [24293], [9959], [23785], [21737], [28401], [18161], [26358], [32509], [1279], [38155], [18189], [26894], [6927], [14610], [23834], [11037], [14631], [26933], [46904], [22330], [25915], [47934], [38214], [1875], [14692], [41832], [13163], [25970], [29565], [44926], [19841], [37250], [49029], [9609], [44438], [16791], [17816], [30109], [41888], [47527], [42924], [23984], [49074], [33717], [31161], [49082], [30138], [31175], [12240], [14804], [7131], [26076], [33250], [3556], [38381], [36338], [32756], [46581], [17912], [49146]] # Tokenized array of badwords used to prevent AI artifacting
badwordsids_neox = [[0], [1], [44162], [9502], [12520], [31841], [36320], [49824], [34417], [6038], [34494], [24815], [26635], [24345], [3455], [28905], [44270], [17278], [32666], [46880], [7086], [43189], [37322], [17778], [20879], [49821], [3138], [14490], [4681], [21391], [26786], [43134], [9336], [683], [48074], [41256], [19181], [29650], [28532], [36487], [45114], [46275], [16445], [15104], [11337], [1168], [5647], [29], [27482], [44965], [43782], [31011], [42944], [47389], [6334], [17548], [38329], [32044], [35487], [2239], [34761], [7444], [1084], [12399], [18990], [17636], [39083], [1184], [35830], [28365], [16731], [43467], [47744], [1138], [16079], [40116], [45564], [18297], [42368], [5456], [18022], [42696], [34476], [23505], [23741], [39334], [37944], [45382], [38709], [33440], [26077], [43600], [34418], [36033], [6660], [48167], [48471], [15775], [19884], [41533], [1008], [31053], [36692], [46576], [20095], [20629], [31759], [46410], [41000], [13488], [30952], [39258], [16160], [27655], [22367], [42767], [43736], [49694], [13811], [12004], [46768], [6257], [37471], [5264], [44153], [33805], [20977], [21083], [25416], [14277], [31096], [42041], [18331], [33376], [22372], [46294], [28379], [38475], [1656], [5204], [27075], [50001], [16616], [11396], [7748], [48744], [35402], [28120], [41512], [4207], [43144], [14767], [15640], [16595], [41305], [44479], [38958], [18474], [22734], [30522], [46267], [60], [13976], [31830], [48701], [39822], [9014], [21966], [31422], [28052], [34607], [2479], [3851], [32214], [44082], [45507], [3001], [34368], [34758], [13380], [38363], [4299], [46802], [30996], [12630], [49236], [7082], [8795], [5218], [44740], [9686], [9983], [45301], [27114], [40125], [1570], [26997], [544], [5290], [49193], [23781], [14193], [40000], [2947], [43781], [9102], [48064], [42274], [18772], [49384], [9884], [45635], [43521], [31258], [32056], [47686], [21760], [13143], [10148], [26119], [44308], [31379], [36399], [23983], [46694], [36134], [8562], [12977], [35117], [28591], [49021], [47093], [28653], [29013], [46468], [8605], [7254], [25896], [5032], [8168], [36893], [38270], [20499], [27501], [34419], [29547], [28571], [36586], [20871], [30537], [26842], [21375], [31148], [27618], [33094], [3291], [31789], [28391], [870], [9793], [41361], [47916], [27468], [43856], [8850], [35237], [15707], [47552], [2730], [41449], [45488], [3073], [49806], [21938], [24430], [22747], [20924], [46145], [20481], [20197], [8239], [28231], [17987], [42804], [47269], [29972], [49884], [21382], [46295], [36676], [34616], [3921], [26991], [27720], [46265], [654], [9855], [40354], [5291], [34904], [44342], [2470], [14598], [880], [19282], [2498], [24237], [21431], [16369], [8994], [44524], [45662], [13663], [37077], [1447], [37786], [30863], [42854], [1019], [20322], [4398], [12159], [44072], [48664], [31547], [18736], [9259], [31], [16354], [21810], [4357], [37982], [5064], [2033], [32871], [47446], [62], [22158], [37387], [8743], [47007], [17981], [11049], [4622], [37916], [36786], [35138], [29925], [14157], [18095], [27829], [1181], [22226], [5709], [4725], [30189], [37014], [1254], [11380], [42989], [696], [24576], [39487], [30119], [1092], [8088], [2194], [9899], [14412], [21828], [3725], [13544], [5180], [44679], [34398], [3891], [28739], [14219], [37594], [49550], [11326], [6904], [17266], [5749], [10174], [23405], [9955], [38271], [41018], [13011], [48392], [36784], [24254], [21687], [23734], [5413], [41447], [45472], [10122], [17555], [15830], [47384], [12084], [31350], [47940], [11661], [27988], [45443], [905], [49651], [16614], [34993], [6781], [30803], [35869], [8001], [41604], [28118], [46462], [46762], [16262], [17281], [5774], [10943], [5013], [18257], [6750], [4713], [3951], [11899], [38791], [16943], [37596], [9318], [18413], [40473], [13208], [16375]]
badwordsids_opt = [[44717], [46613], [48513], [49923], [50185], [48755], [8488], [43303], [49659], [48601], [49817], [45405], [48742], [49925], [47720], [11227], [48937], [48784], [50017], [42248], [49310], [48082], [49895], [50025], [49092], [49007], [8061], [44226], [0], [742], [28578], [15698], [49784], [46679], [39365], [49281], [49609], [48081], [48906], [46161], [48554], [49670], [48677], [49721], [49632], [48610], [48462], [47457], [10975], [46077], [28696], [48709], [43839], [49798], [49154], [48203], [49625], [48395], [50155], [47161], [49095], [48833], [49420], [49666], [48443], [22176], [49242], [48651], [49138], [49750], [40389], [48021], [21838], [49070], [45333], [40862], [1], [49915], [33525], [49858], [50254], [44403], [48992], [48872], [46117], [49853], [47567], [50206], [41552], [50068], [48999], [49703], [49940], [49329], [47620], [49868], [49962], [2], [44082], [50236], [31274], [50260], [47052], [42645], [49177], [17523], [48691], [49900], [49069], [49358], [48794], [47529], [46479], [48457], [646], [49910], [48077], [48935], [46386], [48902], [49151], [48759], [49803], [45587], [48392], [47789], [48654], [49836], [49230], [48188], [50264], [46844], [44690], [48505], [50161], [27779], [49995], [41833], [50154], [49097], [48520], [50018], [8174], [50084], [49366], [49526], [50193], [7479], [49982], [3]]