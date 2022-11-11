from dataclasses import dataclass
import os, re, time, threading, json, pickle, base64, copy, tqdm, datetime, sys
from typing import Union
from io import BytesIO
from flask import has_request_context, session
from flask_socketio import SocketIO, join_room, leave_room
from collections import OrderedDict
import multiprocessing
from logger import logger
import eventlet

serverstarted = False
queue = None
multi_story = False


def clean_var_for_emit(value):
    if isinstance(value, KoboldStoryRegister) or isinstance(value, KoboldWorldInfo):
        return value.to_json()
    elif isinstance(value, set):
        return list(value)
    elif isinstance(value, datetime.datetime):
        return str(value)
    else:
        return value

def process_variable_changes(socketio, classname, name, value, old_value, debug_message=None):
    global multi_story
    if serverstarted and name != "serverstarted":
        transmit_time = str(datetime.datetime.now())
        if debug_message is not None:
            print("{} {}: {} changed from {} to {}".format(debug_message, classname, name, old_value, value))
        if value != old_value:
            #Get which room we'll send the messages to
            if multi_story:
                if classname != 'story':
                    room = 'UI_2'
                else:
                    if has_request_context():
                        room = 'default' if 'story' not in session else session['story']
                    else:
                        logger.error("We tried to access the story register outside of an http context. Will not work in multi-story mode")
                        return
            else:
                room = "UI_2"
            #logger.debug("sending data to room (multi_story={},classname={}): {}".format(multi_story, classname, room))
            #Special Case for KoboldStoryRegister
            if isinstance(value, KoboldStoryRegister):
                #To speed up loading time we will only transmit the last 100 actions to the UI, then rely on scrolling triggers to load more as needed
                if not has_request_context():
                    if queue is not None:
                        #logger.debug("Had to use queue")
                        queue.put(["var_changed", {"classname": "actions", "name": "Action Count", "old_value": None, "value":value.action_count, "transmit_time": transmit_time}, {"broadcast":True, "room":room}])
                        
                        data_to_send = []
                        for i in list(value.actions)[-100:]:
                            data_to_send.append({"id": i, "action": value.actions[i]})
                        queue.put(["var_changed", {"classname": "story", "name": "actions", "old_value": None, "value":data_to_send, "transmit_time": transmit_time}, {"broadcast":True, "room":room}])
                
                else:
                    if socketio is not None:
                        socketio.emit("var_changed", {"classname": "actions", "name": "Action Count", "old_value": None, "value":value.action_count, "transmit_time": transmit_time}, broadcast=True, room=room)
                    
                    data_to_send = []
                    for i in list(value.actions)[-100:]:
                        data_to_send.append({"id": i, "action": value.actions[i]})
                    if socketio is not None:
                        socketio.emit("var_changed", {"classname": "story", "name": "actions", "old_value": None, "value": data_to_send, "transmit_time": transmit_time}, broadcast=True, room=room)
            elif isinstance(value, KoboldWorldInfo):
                value.send_to_ui()
            else:
                #If we got a variable change from a thread other than what the app is run it, eventlet seems to block and no further messages are sent. Instead, we'll rely the message to the app and have the main thread send it
                if not has_request_context():
                    data = ["var_changed", {"classname": classname, "name": name, "old_value": clean_var_for_emit(old_value), "value": clean_var_for_emit(value), "transmit_time": transmit_time}, {"include_self":True, "broadcast":True, "room":room}]
                    if queue is not None:
                        #logger.debug("Had to use queue")
                        queue.put(data)
                        
                else:
                    if socketio is not None:
                        socketio.emit("var_changed", {"classname": classname, "name": name, "old_value": clean_var_for_emit(old_value), "value": clean_var_for_emit(value), "transmit_time": transmit_time}, include_self=True, broadcast=True, room=room)

class koboldai_vars(object):
    def __init__(self, socketio):
        self._model_settings = model_settings(socketio, self)
        self._user_settings = user_settings(socketio)
        self._system_settings = system_settings(socketio, self)
        self._story_settings = {'default': story_settings(socketio, self)}
        self.socketio = socketio
        self.tokenizer = None
    
    def get_story_name(self):
        global multi_story
        if multi_story:
            if has_request_context():
                story = 'default' if 'story' not in session else session['story']
            else:
                logger.error("We tried to access the story register outside of an http context. Will not work in multi-story mode")
                assert multi_story and has_request_context(), "Tried to access story data outside context in multi_story mode"
        else:
            story = "default"
        return story
    
    def to_json(self, classname):
        if classname == 'story_settings':
            return self._story_settings[self.get_story_name()].to_json()
#            data = {}
#            for story in self._story_settings:
#                data[story] = json.loads(self._story_settings[story].to_json())
#            return json.dumps(data)
        return self.__dict__["_{}".format(classname)].to_json()
        
    def load_story(self, story_name, json_data):
        #Story name here is intended for multiple users on multiple stories. Now always uses default
        #If we can figure out a way to get flask sessions into/through the lua bridge we could re-enable
        global multi_story
        original_story_name = story_name
        if not multi_story:
            story_name = 'default'
        #Leave the old room and join the new one
        logger.debug("Leaving room {}".format(session['story']))
        leave_room(session['story'])
        logger.debug("Joining room {}".format(story_name))
        join_room(story_name)
        session['story'] = story_name
        logger.debug("Sending story reset")
        self._story_settings[story_name].socketio.emit("reset_story", {}, broadcast=True, room=story_name)
        if story_name in self._story_settings:
            self._story_settings[story_name].no_save = True
            self._story_settings[story_name].from_json(json_data)
            logger.debug("Calcing AI text after load story")
            ignore = self.calc_ai_text()
            self._story_settings[story_name].no_save = False
        else:
            #self._story_settings[story_name].no_save = True
            self.create_story(story_name, json_data=json_data)
            logger.debug("Calcing AI text after create story")
            ignore = self.calc_ai_text()
            #self._story_settings[story_name].no_save = False
        self._system_settings.story_loads[original_story_name] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        with open("settings/system_settings.v2_settings", "w") as settings_file:
            settings_file.write(self._system_settings.to_json())
        
    def save_story(self):
        logger.debug("Saving story from koboldai_vars.save_story()")
        self._story_settings[self.get_story_name()].save_story()
    
    def download_story(self):
        return self._story_settings[self.get_story_name()].to_json()
    
    def save_revision(self):
        self._story_settings[self.get_story_name()].save_revision()
    
    def create_story(self, story_name, json_data=None):
        #Story name here is intended for multiple users on multiple stories. Now always uses default
        #If we can figure out a way to get flask sessions into/through the lua bridge we could re-enable
        #story_name = 'default'
        if not multi_story:
            story_name = 'default'
        self._story_settings[story_name] = story_settings(self.socketio, self)
        #self._story_settings[story_name].reset()
        if json_data is not None:
            self.load_story(story_name, json_data)
        else:
            #Leave the old room and join the new one
            logger.debug("Leaving room {}".format(session['story']))
            leave_room(session['story'])
            logger.debug("Joining room {}".format(story_name))
            join_room(story_name)
            session['story'] = story_name
            logger.debug("Sending story reset")
            self._story_settings[story_name].socketio.emit("reset_story", {}, broadcast=True, room=story_name)
            self._story_settings[story_name].send_to_ui()
        session['story'] = story_name
        
    def story_list(self):
        return [x for x in self._story_settings]
    
    def send_to_ui(self):
        self._model_settings.send_to_ui()
        self._user_settings.send_to_ui()
        self._system_settings.send_to_ui()
        self._story_settings[self.get_story_name()].send_to_ui()
    
    def reset_model(self):
        self._model_settings.reset_for_model_load()
    
    def get_token_representation(self, text: Union[str, list, None]) -> list:
        if not self.tokenizer or not text:
            return []
        
        if isinstance(text, str):
            encoded = self.tokenizer.encode(text)
        else:
            encoded = text

        # TODO: This might be ineffecient, should we cache some of this?
        return [[token, self.tokenizer.decode(token)] for token in encoded]
    
    def calc_ai_text(self, submitted_text="", return_text=False, send_context=True):
        #start_time = time.time()
        if self.tokenizer is None:
            if return_text:
                return ""
            return [], 0, 0+self.genamt, []
            
        if self.alt_gen:
            method = 2
        else:
            method = 1
        #Context and Game Context are lists of chunks of text that will go to the AI. Game Context will be appended after context when we're done
        context = []
        game_context = []
        token_budget = self.max_length - self.genamt
        used_world_info = []
        used_tokens = 0 if self.sp_length is None else self.sp_length + len(self.tokenizer._koboldai_header)
        
        if self.sp_length > 0:
            context.append({"type": "soft_prompt", "text": f"<{self.sp_length} tokens of Soft Prompt.>", "tokens": [[-1, ""]] * self.sp_length})
        
        self.worldinfo_v2.reset_used_in_game()
        
        ######################################### Add memory ########################################################
        memory_text = self.memory
        if memory_text != "":
            if memory_text[-1] not in [" ", '\n']:
                memory_text += " "
            memory_tokens = self.tokenizer.encode(memory_text)
        else:
            memory_tokens = []
        if len(memory_tokens) > self.max_memory_length:
            memory_tokens = memory_tokens[:self.max_memory_length]
            memory_length = self.max_memory_length
        memory_data = [[x, self.tokenizer.decode(x)] for x in memory_tokens]
        
        #We have so much memory that we've run out of budget without going over the max_memory_length. Just stop
        if len(memory_tokens) > token_budget:
            return [], 0, 0+self.genamt, []
        
        #Actually Add Memory
        if len(memory_tokens) != 0:
            context.append({"type": "memory", 
                            "text": "".join([x[1] for x in memory_data]), 
                            "tokens": memory_data,
                            "attention_multiplier": self.memory_attn_bias})
            used_tokens += len(memory_tokens)
        
        
        ######################################### Constant World Info ########################################################
        #Add constant world info entries to memory
        for wi in self.worldinfo_v2:
            if wi['constant']:
                wi_length = len(self.tokenizer.encode(wi['content']))
                if used_tokens + wi_length <= token_budget:
                    used_tokens+=wi_length
                    used_world_info.append(wi['uid'])
                    self.worldinfo_v2.set_world_info_used(wi['uid'])
                    wi_text = wi['content']+" " if wi['content'] != "" and wi['content'][-1] not in [" ", "\n"] else wi['content']
                    wi_tokens = self.tokenizer.encode(wi_text)
                    context.append({
                        "type": "world_info",
                        "text": wi_text,
                        "uid": wi['uid'],
                        "tokens": [[x, self.tokenizer.decode(x)] for x in wi_tokens],
                    })
                    used_tokens += len(wi_tokens)
        
        
        
        ######################################### Get Action Text by Sentence ########################################################
        action_text_split = self.actions.to_sentences(submitted_text=submitted_text)
        
        
        ######################################### Prompt ########################################################
        #Add prompt lenght/text if we're set to always use prompt
        if self.useprompt:
            prompt_length = 0
            prompt_data = []
            for item in reversed(action_text_split):
                if -1 in item[1]:
                    tokenized_data = [[x, self.tokenizer.decode(x)] for x in self.tokenizer.encode(item[0])]
                    item[2] = len(tokenized_data)
                    if prompt_length + item[2] <= self.max_prompt_length:
                        prompt_length += item[2]
                        item[3] = True
                        prompt_data = tokenized_data + prompt_data
            prompt_text = self.tokenizer.decode([x[0] for x in prompt_data])
            #wi_search = re.sub("[^A-Za-z\ 0-9\'\"]", "", prompt_text)
            wi_search = prompt_text
            if prompt_length + used_tokens <= token_budget:
                used_tokens += prompt_length
                #We'll add the prompt text AFTER we go through the game text as the world info needs to come first if we're in method 1 rather than method 2
                self.prompt_in_ai = True
                #Find World Info entries in prompt
                for wi in self.worldinfo_v2:
                    if wi['uid'] not in used_world_info:
                        #Check to see if we have the keys/secondary keys in the text so far
                        match = False
                        for key in wi['key']:
                            if key in wi_search:
                                match = True
                                break
                        if wi['selective'] and match:
                            match = False
                            for key in wi['keysecondary']:
                                if key in wi_search:
                                    match=True
                                    break
                        if match:
                            wi_length = len(self.tokenizer.encode(wi['content']))
                            if used_tokens+wi_length <= token_budget:
                                used_tokens+=wi_length
                                used_world_info.append(wi['uid'])
                                wi_text = wi['content']+" " if wi['content'] != "" and wi['content'][-1] not in [" ", "\n"] else wi['content']
                                wi_tokens = self.tokenizer.encode(wi_text)
                                context.append({
                                    "type": "world_info",
                                    "text": wi_text,
                                    "uid": wi['uid'],
                                    "tokens": [[x, self.tokenizer.decode(x)] for x in wi_tokens],
                                })
                                used_tokens += len(wi_tokens)
                                self.worldinfo_v2.set_world_info_used(wi['uid'])
                   
            else:
                self.prompt_in_ai = False
        
        
        ######################################### Setup Author's Note Data ########################################################
        authors_note_text = self.authornotetemplate.replace("<|>", self.authornote)
        if len(authors_note_text) > 0 and authors_note_text[-1] not in [" ", "\n"]:
            authors_note_text += " "
        authors_note_data = [[x, self.tokenizer.decode(x)] for x in self.tokenizer.encode(authors_note_text)]
        if used_tokens + len(authors_note_data) <= token_budget:
            used_tokens += len(authors_note_data)
        
        
        ######################################### Actions ########################################################
        #Start going through the actions backwards, adding it to the text if it fits and look for world info entries
        used_all_tokens = False
        actions_seen = [] #Used to track how many actions we've seen so we can insert author's note in the appropriate place as well as WI depth stop
        inserted_author_note = False

        for i in range(len(action_text_split)-1, -1, -1):
            if action_text_split[i][3]:
                #We've hit an item we've already included. Stop
                break
            for action in action_text_split[i][1]:
                if action not in actions_seen:
                    actions_seen.append(action)

            #Add our author's note if we've hit andepth
            if not inserted_author_note and len(actions_seen) >= self.andepth and self.authornote != "":
                game_context.insert(0, {"type": "authors_note", "text": authors_note_text, "tokens": authors_note_data, "attention_multiplier": self.an_attn_bias})
                inserted_author_note = True

            action_data = [[x, self.tokenizer.decode(x)] for x in self.tokenizer.encode(action_text_split[i][0])]
            length = len(action_data)

            if length+used_tokens <= token_budget:
                used_tokens += length
                action_text_split[i][3] = True

                action_type = "action"
                if action_text_split[i][1] == [self.actions.action_count+1]:
                    action_type = "submit"
                elif -1 in action_text_split[i][1]:
                    action_type = "prompt"
                
                game_context.insert(0, {
                    "type": action_type,
                    "text": action_text_split[i][0],
                    "tokens": action_data,
                    "action_ids": action_text_split[i][1]
                })
                #wi_search = re.sub("[^A-Za-z\ 0-9\'\"]", "", action_text_split[i][0])
                wi_search = action_text_split[i][0]


                #Now we need to check for used world info entries
                for wi in self.worldinfo_v2:
                    if wi['uid'] not in used_world_info:
                        #Check to see if we have the keys/secondary keys in the text so far
                        match = False
                        for key in wi['key']:
                            if key in wi_search:
                                match = True
                                break
                        if wi['selective'] and match:
                            match = False
                            for key in wi['keysecondary']:
                                if key in wi_search:
                                    match=True
                                    break
                        if method == 1:
                            if len(actions_seen) > self.widepth:
                                match = False
                        if match:
                            wi_length = len(self.tokenizer.encode(wi['content']))
                            if used_tokens+wi_length <= token_budget:
                                used_tokens+=wi_length
                                used_world_info.append(wi['uid'])
                                wi_text = wi['content']+" " if wi['content'] != "" and wi['content'][-1] not in [" ", "\n"] else wi['content']
                                wi_tokens = self.tokenizer.encode(wi_text)
                                if method == 1:
                                    context.append({
                                        "type": "world_info",
                                        "text": wi_text,
                                        "uid": wi['uid'],
                                        "tokens": [[x, self.tokenizer.decode(x)] for x in wi_tokens],
                                    })
                                else:
                                    #for method 2 we add the game text before the current action
                                    game_context.insert(0, {
                                        "type": "world_info",
                                        "text": wi_text,
                                        "uid": wi['uid'],
                                        "tokens": [[x, self.tokenizer.decode(x)] for x in wi_tokens],
                                    })
                                used_tokens += len(wi_tokens)
                                self.worldinfo_v2.set_world_info_used(wi['uid'])
            else:
                used_all_tokens = True
                break
             
             
        ######################################### Verify Author's Note Data in AI Text ########################################################
        #if we don't have enough actions to get to author's note depth then we just add it right before the game text
        if not inserted_author_note and self.authornote != "":
            game_context.insert(0, {"type": "authors_note", "text": authors_note_text, "tokens": authors_note_data, "attention_multiplier": self.an_attn_bias})
        
        
        ######################################### Add our prompt data ########################################################
        if self.useprompt and len(prompt_data) != 0:
            context.append({"type": "prompt", "text": prompt_text, "tokens": prompt_data})
        
        context += game_context
        
        if len(context) == 0:
            tokens = []
        else:
            tokens = []
            for item in context:
                tokens.extend([x[0] for x in item['tokens']])
        
        if send_context:
            self.context = context

        #logger.debug("Calc_AI_text: {}s".format(time.time()-start_time))
        logger.debug("Token Budget: {}. Used Tokens: {}".format(token_budget, used_tokens))
        if return_text:
            return "".join([x['text'] for x in context])
        return tokens, used_tokens, used_tokens+self.genamt, used_world_info

    def is_model_torch(self) -> bool:
        if self.use_colab_tpu:
            return False

        if self.model in ["Colab", "API", "CLUSTER", "ReadOnly", "OAI"]:
            return False

        return True
    
    def assign_world_info_to_actions(self, *args, **kwargs):
        self._story_settings[self.get_story_name()].assign_world_info_to_actions(*args, **kwargs)
    
    def __setattr__(self, name, value):
        if name[0] == "_" or name == "tokenizer":
            super().__setattr__(name, value)
        if name[0] != "_":
            #Send it to the corrent _setting class
            if name in self._model_settings.__dict__:
                setattr(self._model_settings, name, value)
            elif name in self._user_settings.__dict__:
                setattr(self._user_settings, name, value)
            elif name in self._system_settings.__dict__:
                setattr(self._system_settings, name, value)
            else:
                setattr(self._story_settings[self.get_story_name()], name, value)


    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        elif name in self._model_settings.__dict__:
            return getattr(self._model_settings, name)
        elif name in self._user_settings.__dict__:
            return getattr(self._user_settings, name)
        elif name in self._system_settings.__dict__:
            return getattr(self._system_settings, name)
        else:
            return getattr(self._story_settings[self.get_story_name()], name)


class settings(object):
    def to_json(self):
        json_data = {'file_version': 2}
        for (name, value) in vars(self).items():
            if name not in self.no_save_variables and name[0] != "_":
                json_data[name] = value
        def to_base64(data):
            if isinstance(data, KoboldStoryRegister):
                return data.to_json()
            elif isinstance(data, KoboldWorldInfo):
                return data.to_json()
            elif isinstance(data, datetime.datetime):
                return str(data)
            output = BytesIO()
            pickle.dump(data, output)
            output.seek(0)
            return "base64:{}".format(base64.encodebytes(output.read()).decode())
        return json.dumps(json_data, default=to_base64, indent="\t")
    
    def from_json(self, data):
        if isinstance(data, str):
            json_data = json.loads(data)
        else:
            json_data = data
        #since loading will trigger the autosave, we need to disable it
        if 'no_save' in self.__dict__:
            setattr(self, 'no_save', True)
        for key, value in json_data.items():
            start_time = time.time()
            if key in self.__dict__ and key not in self.no_save_variables:
                if key == 'sampler_order':
                    if(len(value) < 7):
                        value = [6] + value
                if key == 'autosave':
                    autosave = value
                if isinstance(value, str):
                    if value[:7] == 'base64:':
                        value = pickle.loads(base64.b64decode(value[7:]))
                #Need to fix the data type of value to match the module
                if type(getattr(self, key)) == int:
                    setattr(self, key, int(value))
                elif type(getattr(self, key)) == float:
                    setattr(self, key, float(value))
                elif type(getattr(self, key)) == bool:
                    setattr(self, key, bool(value))
                elif type(getattr(self, key)) == str:
                    setattr(self, key, str(value))
                elif isinstance(getattr(self, key), KoboldStoryRegister):
                    getattr(self, key).load_json(value)
                elif isinstance(getattr(self, key), KoboldWorldInfo):
                    getattr(self, key).load_json(value)
                else:
                    setattr(self, key, value)
            logger.debug("Loading {} took {}s".format(key, time.time()- start_time))
        
        #check from prompt_wi_highlighted_text since that wasn't always in the V2 save format
        if 'prompt' in json_data and 'prompt_wi_highlighted_text' not in json_data:
            self.prompt_wi_highlighted_text[0]['text'] = self.prompt
            self.assign_world_info_to_actions(action_id=-1)
            process_variable_changes(self.socketio, "story", 'prompt_wi_highlighted_text', self.prompt_wi_highlighted_text, None)
        
        if 'no_save' in self.__dict__:
            setattr(self, 'no_save', False)
            
        
    def send_to_ui(self):
        for (name, value) in vars(self).items():
            if name not in self.local_only_variables and name[0] != "_":
                try:
                    process_variable_changes(self.socketio, self.__class__.__name__.replace("_settings", ""), name, value, None)
                except:
                    print("{} is of type {} and I can't transmit".format(name, type(value)))
                    raise

class model_settings(settings):
    local_only_variables = ['badwordsids', 'apikey', 'tqdm', 'socketio', 'default_preset', 'koboldai_vars']
    no_save_variables = ['tqdm', 'tqdm_progress', 'tqdm_rem_time', 'socketio', 'modelconfig', 'custmodpth', 'generated_tkns', 
                         'loaded_layers', 'total_layers', 'total_download_chunks', 'downloaded_chunks', 'presets', 'default_preset', 
                         'koboldai_vars', 'welcome', 'welcome_default']
    settings_name = "model"
    def __init__(self, socketio, koboldai_vars):
        self.socketio = socketio
        self.reset_for_model_load()
        self.model       = ""     # Model ID string chosen at startup
        self.model_type  = ""     # Model Type (Automatically taken from the model config)
        self.modelconfig = {}     # Raw contents of the model's config.json, or empty dictionary if none found
        self.custmodpth  = ""     # Filesystem location of custom model to run
        self.generated_tkns = 0    # If using a backend that supports Lua generation modifiers, how many tokens have already been generated, otherwise 0
        self.loaded_layers = 0     # Used in UI 2 to show model loading progress
        self.total_layers = 0      # Same as above
        self.total_download_chunks = 0 # tracks how much of the model has downloaded for the UI 2
        self.downloaded_chunks = 0 #as above
        self.tqdm        = tqdm.tqdm(total=self.genamt, file=self.ignore_tqdm())    # tqdm agent for generating tokens. This will allow us to calculate the remaining time
        self.tqdm_progress = 0     # TQDP progress
        self.tqdm_rem_time = 0     # tqdm calculated reemaining time
        self.url         = "https://api.inferkit.com/v1/models/standard/generate" # InferKit API URL
        self.oaiurl      = "" # OpenAI API URL
        self.oaiengines  = "https://api.openai.com/v1/engines"
        self.colaburl    = ""     # Ngrok url for Google Colab mode
        self.apikey      = ""     # API key to use for InferKit API calls
        self.oaiapikey   = ""     # API key to use for OpenAI API calls
        self.configname = None
        self.online_model = ''
        self.welcome_default = """  <img id='welcome-logo' src='static/Welcome_Logo.png' draggable='False'>
                                    <div class='welcome_text'>
                                        <div id="welcome-text-content">Please load a model from the left.<br/>
                                            If you encounter any issues, please click the Download debug dump link in the Home tab on the left flyout and attach the downloaded file to your error report on <a href='https://github.com/ebolam/KoboldAI/issues'>Github</a>, <a href='https://www.reddit.com/r/KoboldAI/'>Reddit</a>, or <a href='https://discord.gg/XuQWadgU9k'>Discord</a>. 
                                            A redacted version (without story text) is available.
                                        </div>
                                    </div>""" # Custom Welcome Text
        self.welcome     = self.welcome_default
        self.koboldai_vars = koboldai_vars
        
    def reset_for_model_load(self):
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
        self.penalty_alpha = 0.0   # Default generator penalty_alpha (contrastive search)
        self.numseqs     = 1       # Number of sequences to ask the generator to create
        self.badwordsids = []
        self.fp32_model  = False  # Whether or not the most recently loaded HF model was in fp32 format
        self.modeldim    = -1     # Embedding dimension of your model (e.g. it's 4096 for GPT-J-6B and 2560 for GPT-Neo-2.7B)
        self.sampler_order = [6, 0, 1, 2, 3, 4, 5]
        self.newlinemode = "n"
        self.lazy_load   = True # Whether or not to use torch_lazy_loader.py for transformers models in order to reduce CPU memory usage
        self.revision    = None
        self.presets     = []   # Holder for presets
        self.selected_preset = ""
        self.uid_presets = []
        self.default_preset = {}
        self.horde_wait_time = 0
        self.horde_queue_position = 0
        self.horde_queue_size = 0

        
    #dummy class to eat the tqdm output
    class ignore_tqdm(object):
        def write(self, bar):
            pass
        
    def __setattr__(self, name, value):
        new_variable = name not in self.__dict__
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        #Put variable change actions here
        
        if not new_variable and (name == 'max_length' or name == 'genamt'):
            ignore = self.koboldai_vars.calc_ai_text()
        
        #set preset values
        if name == 'selected_preset' and value != "":
            if int(value) in self.uid_presets:
                for preset_key, preset_value in self.uid_presets[int(value)].items():
                    if preset_key in self.__dict__:
                        if type(getattr(self, preset_key)) == int:
                            preset_value = int(preset_value)
                        elif type(getattr(self, preset_key)) == float:
                            preset_value = float(preset_value)
                        elif type(getattr(self, preset_key)) == bool:
                            preset_value = bool(preset_value)
                        elif type(getattr(self, preset_key)) == str:
                            preset_value = str(preset_value)
                        if preset_key == "sampler_order":
                            if 6 not in preset_value:
                                preset_value.insert(0, 6)
                        setattr(self, preset_key, preset_value)
        #Setup TQDP for token generation
        elif name == "generated_tkns" and 'tqdm' in self.__dict__:
            if value == 0:
                self.tqdm.reset(total=self.genamt)
                self.tqdm_progress = 0
            else:
                self.tqdm.update(value-self.tqdm.n)
                self.tqdm_progress = int(float(self.generated_tkns)/float(self.genamt)*100)
                if self.tqdm.format_dict['rate'] is not None:
                    self.tqdm_rem_time = str(datetime.timedelta(seconds=int(float(self.genamt-self.generated_tkns)/self.tqdm.format_dict['rate'])))
        #Setup TQDP for model loading
        elif name == "loaded_layers" and 'tqdm' in self.__dict__:
            if value == 0:
                self.tqdm.reset(total=self.total_layers)
                self.tqdm_progress = 0
            else:
                self.tqdm.update(1)
                self.tqdm_progress = int(float(self.loaded_layers)/float(self.total_layers)*100)
                if self.tqdm.format_dict['rate'] is not None:
                    self.tqdm_rem_time = str(datetime.timedelta(seconds=int(float(self.total_layers-self.loaded_layers)/self.tqdm.format_dict['rate'])))  
        #Setup TQDP for model downloading
        elif name == "total_download_chunks" and 'tqdm' in self.__dict__:
            self.tqdm.reset(total=value)
            self.tqdm_progress = 0
        elif name == "downloaded_chunks" and 'tqdm' in self.__dict__:
            if value == 0:
                self.tqdm.reset(total=self.total_download_chunks)
                self.tqdm_progress = 0
            else:
                self.tqdm.update(value-old_value)
                if self.total_download_chunks is not None:
                    if self.total_download_chunks==0:
                        self.tqdm_progress = 0
                    elif float(self.downloaded_chunks) > float(self.total_download_chunks):
                        self.tqdm_progress = 100
                    else: 
                        self.tqdm_progress = round(float(self.downloaded_chunks)/float(self.total_download_chunks)*100, 1)
                else:
                    self.tqdm_progress = 0
                if self.tqdm.format_dict['rate'] is not None:
                    self.tqdm_rem_time = str(datetime.timedelta(seconds=int(float(self.total_download_chunks-self.downloaded_chunks)/self.tqdm.format_dict['rate'])))  
        
        
        
        if name not in self.local_only_variables and name[0] != "_" and not new_variable:
            process_variable_changes(self.socketio, self.__class__.__name__.replace("_settings", ""), name, value, old_value)
            
class story_settings(settings):
    local_only_variables = ['socketio', 'tokenizer', 'koboldai_vars', 'no_save', 'revisions', 'prompt']
    no_save_variables = ['socketio', 'tokenizer', 'koboldai_vars', 'context', 'no_save', 'prompt_in_ai', 'authornote_length', 'prompt_length', 'memory_length']
    settings_name = "story"
    def __init__(self, socketio, koboldai_vars, tokenizer=None):
        self.socketio = socketio
        self.tokenizer = tokenizer
        self.koboldai_vars = koboldai_vars
        self.privacy_mode = False
        self.privacy_password = ""
        self.story_name  = "New Game"   # Title of the story
        self.lastact     = ""     # The last action received from the user
        self.submission  = ""     # Same as above, but after applying input formatting
        self.lastctx     = ""     # The last context submitted to the generator
        self.gamestarted = False  # Whether the game has started (disables UI elements)
        self.gamesaved   = True   # Whether or not current game is saved
        self.autosave    = False             # Whether or not to automatically save after each action
        self.prompt      = ""     # Prompt
        self.prompt_wi_highlighted_text = [{"text": self.prompt, "WI matches": None, "WI Text": ""}]
        self.memory      = ""     # Text submitted to memory field
        self.auto_memory = ""
        self.authornote  = ""     # Text submitted to Author's Note field
        self.authornotetemplate = "[Author's note: <|>]"  # Author's note template
        self.setauthornotetemplate = self.authornotetemplate  # Saved author's note template in settings
        self.andepth     = 3      # How far back in history to append author's note
        self.actions     = KoboldStoryRegister(socketio, self, koboldai_vars, tokenizer=tokenizer)  # Actions submitted by user and AI
        self.actions_metadata = {} # List of dictonaries, one dictonary for every action that contains information about the action like alternative options.
                              # Contains at least the same number of items as actions. Back action will remove an item from actions, but not actions_metadata
                              # Dictonary keys are:
                              # Selected Text: (text the user had selected. None when this is a newly generated action)
                              # Alternative Generated Text: {Text, Pinned, Previous Selection, Edited}
                              # 
        self.worldinfo_v2 = KoboldWorldInfo(socketio, self, koboldai_vars, tokenizer=self.tokenizer)
        self.worldinfo   = []     # List of World Info key/value objects
        self.worldinfo_i = []     # List of World Info key/value objects sans uninitialized entries
        self.worldinfo_u = {}     # Dictionary of World Info UID - key/value pairs
        self.wifolders_d = {}     # Dictionary of World Info folder UID-info pairs
        self.wifolders_l = []     # List of World Info folder UIDs
        self.wifolders_u = {}     # Dictionary of pairs of folder UID - list of WI UID
        self.lua_edited  = set()  # Set of chunk numbers that were edited from a Lua generation modifier
        self.lua_deleted = set()  # Set of chunk numbers that were deleted from a Lua generation modifier
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
        self.actionmode  = 0
        self.storymode   = 0
        self.dynamicscan = False
        self.recentedit  = False
        self.notes       = ""    #Notes for the story. Does nothing but save
        self.biases      = {} # should look like {"phrase": [score, completion_threshold]}
        self.story_id    = int.from_bytes(os.urandom(16), 'little', signed=True) # this is a unique ID for the story. We'll use this to ensure when we save that we're saving the same story
        self.memory_length = 0
        self.prompt_length = 0
        self.authornote_length = 0
        self.max_memory_length = 512
        self.max_prompt_length = 512
        self.max_authornote_length = 512
        self.prompt_in_ai = False
        self.context = []
        self.last_story_load = None
        self.revisions = []
        self.picture = "" #base64 of the image shown for the story
        self.picture_prompt = "" #Prompt used to create picture
        self.substitutions = [
            {"target": "--", "substitution": "–", "enabled": False},
            {"target": "---", "substitution": "—", "enabled": False},
            {"target": "...", "substitution": "…", "enabled": False},
            # {"target": "(c)", "substitution": "©", "enabled": False},
            # {"target": "(r)", "substitution": "®", "enabled": False},
            # {"target": "(tm)", "substitution": "™", "enabled": False},
        ]
        
        # bias experiment
        self.memory_attn_bias = 1
        self.an_attn_bias = 1
        
        
        ################### must be at bottom #########################
        self.no_save = False  #Temporary disable save (doesn't save with the file)

        
        
    def save_story(self):
        if not self.no_save:
            if self.prompt != "" or self.memory != "" or self.authornote != "" or len(self.actions) > 0 or len(self.worldinfo_v2) > 0:
                logger.debug("Saving story from koboldai_vars.story_settings.save_story()")
                logger.info("Saving")
                save_name = self.story_name if self.story_name != "" else "untitled"
                adder = ""
                while True:
                    if os.path.exists("stories/{}{}_v2.json".format(save_name, adder)):
                        with open("stories/{}{}_v2.json".format(save_name, adder), "r") as f:
                            temp = json.load(f)
                        if 'story_id' in temp:
                            if self.story_id != temp['story_id']:
                                adder = 0 if adder == "" else adder+1
                            else:
                                break
                        else:
                            adder = 0 if adder == "" else adder+1
                    else:
                        break
                with open("stories/{}{}_v2.json".format(save_name, adder), "w") as settings_file:
                    settings_file.write(self.to_json())
                self.gamesaved = True
    
    def save_revision(self):
        game = json.loads(self.to_json())
        del game['revisions']
        self.revisions.append(game)
        self.gamesaved = False
    
    def reset(self):
        self.no_save = True
        self.socketio.emit("reset_story", {}, broadcast=True, room="UI_2")
        self.__init__(self.socketio, self.koboldai_vars, tokenizer=self.tokenizer)
        self.no_save = False
      
    def sync_worldinfo_v1_to_v2(self):
        new_world_info = KoboldWorldInfo(None, self, self.koboldai_vars, tokenizer=self.tokenizer)
        for wi in self.worldinfo:
            if wi['init'] == True:
                new_world_info.add_item([x.strip() for x in wi["key"].split(",")][0], 
                                        wi["key"], 
                                        wi.get("keysecondary", ""), 
                                        "root" if wi["folder"] is None else self.wifolders_d[wi['folder']]['name'], 
                                        wi.get("constant", False), 
                                        wi["content"], 
                                        wi.get("comment", ""), 
                                        v1_uid=wi['uid'], sync=False)
        
        new_world_info.socketio = self.socketio
        self.worldinfo_v2 = new_world_info
        
    def assign_world_info_to_actions(self, action_id=None, wuid=None, no_transmit=False):
        logger.debug("Calcing WI Assignment for action_id: {} wuid: {}".format(action_id, wuid))
        if action_id != -1 and (action_id is None or action_id not in self.actions.actions):
            actions_to_check = self.actions.actions
        else:
            actions_to_check = {}
        if wuid is None or wuid not in self.worldinfo_v2.world_info:
            wi_to_check = self.worldinfo_v2.world_info
        else:
            wi_to_check = {wuid: self.worldinfo_v2.world_info[wuid]}
            
        for action_id, action in actions_to_check.items():
            for uid, wi in wi_to_check.items():
                for key in sorted(wi['key'], key=len, reverse=True):
                    if key in action['Selected Text']:
                        self.actions.add_wi_to_action(action_id, key, wi['content'], uid, no_transmit=no_transmit)
                        break
                        
        #Do prompt if no action_id was sent
        if action_id is None or action_id == -1:
            for uid, wi in wi_to_check.items():
                for key in sorted(wi['key'], key=len, reverse=True):
                    if key in self.prompt:
                        if wi['keysecondary'] != []:
                            for key2 in wi['keysecondary']:
                                if key2 in self.prompt:
                                    self.actions.add_wi_to_action(-1, key, wi['content'], uid, no_transmit=no_transmit)
                                    break
                        else:
                            self.actions.add_wi_to_action(-1, key, wi['content'], uid, no_transmit=no_transmit)
                            break
    
    def __setattr__(self, name, value):
        new_variable = name not in self.__dict__
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        #Put variable change actions here
        if name not in self.local_only_variables and name[0] != "_" and not new_variable and old_value != value:
            process_variable_changes(self.socketio, self.__class__.__name__.replace("_settings", ""), name, value, old_value)
        #We want to automatically set gamesaved to false if something happens to the actions list (pins, redos, generations, text, etc)
        #To do that we need to give the actions list a copy of this data so it can set the gamesaved variable as needed
        
        #autosave action
        if name == "gamesaved" and value == False and self.autosave:
            logger.debug("Saving story from gamesaved change and on autosave")
            self.save_story()
        if not new_variable and old_value != value:
            #Change game save state
            if name in ['story_name', 'prompt', 'memory', 'authornote', 'authornotetemplate', 'andepth', 'chatname', 'actionmode', 'dynamicscan', 'notes', 'biases']:
                self.gamesaved = False
        
            if name == 'useprompt':
                ignore = self.koboldai_vars.calc_ai_text()
            elif name == 'actions':
                self.actions.story_settings = self
                logger.debug("Calcing AI text after setting actions")
                ignore = self.koboldai_vars.calc_ai_text()
            elif name == 'story_name':
                #reset the story id if we change the name
                self.story_id = int.from_bytes(os.urandom(16), 'little', signed=True)
            
            #Recalc AI Text
            elif name == 'authornote':
                ignore = self.koboldai_vars.calc_ai_text()
            elif name == 'authornotetemplate':
                ignore = self.koboldai_vars.calc_ai_text()
            elif name == 'memory':
                ignore = self.koboldai_vars.calc_ai_text()
            elif name == 'prompt':
                self.prompt_wi_highlighted_text = [{"text": self.prompt, "WI matches": None, "WI Text": ""}]
                self.assign_world_info_to_actions(action_id=-1, wuid=None)
                process_variable_changes(self.socketio, self.__class__.__name__.replace("_settings", ""), 'prompt_wi_highlighted_text', self.prompt_wi_highlighted_text, None)
                ignore = self.koboldai_vars.calc_ai_text()
            
            #Because we have seperate variables for action types, this syncs them
            elif name == 'storymode':
                if value == 0:
                    self.adventure = False
                    self.chatmode = False
                elif value == 1:
                    self.adventure = True
                    self.chatmode = False
                elif value == 2:
                    self.adventure = False
                    self.chatmode = True
            elif name == 'adventure' and value == True:
                self.chatmode = False
                self.storymode = 1
            elif name == 'adventure' and value == False and self.chatmode == False:
                self.storymode = 0
            elif name == 'chatmode' and value == True:
                self.adventure = False
                self.storymode = 2
            elif name == 'chatmode' and value == False and self.adventure == False:
                self.storymode = 0
                
class user_settings(settings):
    local_only_variables = ['socketio', 'importjs']
    no_save_variables = ['socketio', 'importnum', 'importjs', 'loadselect', 'spselect', 'svowname', 'saveow', 'laststory', 'sid']
    settings_name = "user"
    def __init__(self, socketio):
        self.socketio = socketio
        self.wirmvwhtsp  = False             # Whether to remove leading whitespace from WI entries
        self.widepth     = 3                 # How many historical actions to scan for WI hits
        self.formatoptns = {'frmttriminc': True, 'frmtrmblln': False, 'frmtrmspch': False, 'frmtadsnsp': False, 'singleline': False}     # Container for state of formatting options
        self.frmttriminc = True
        self.frmtrmblln  = False
        self.frmtrmspch  = False
        self.frmtadsnsp  = False
        self.singleline  = False
        self.remove_double_space = True
        self.importnum   = -1                # Selection on import popup list
        self.importjs    = {}                # Temporary storage for import data
        self.loadselect  = ""                # Temporary storage for story filename to load
        self.spselect    = ""                # Temporary storage for soft prompt filename to load
        self.svowname    = ""                # Filename that was flagged for overwrite confirm
        self.saveow      = False             # Whether or not overwrite confirm has been displayed
        self.laststory   = None              # Filename (without extension) of most recent story JSON file we loaded
        self.sid         = ""                # session id for the socketio client (request.sid)
        self.username    = "Default User"    # Displayed Username
        self.nopromptgen = False
        self.rngpersist  = False
        self.nogenmod    = False
        self.debug       = False    # If set to true, will send debug information to the client for display
        self.output_streaming = True
        self.show_probs = False # Whether or not to show token probabilities
        self.beep_on_complete = False
        self.img_gen_priority = 1
        self.show_budget = False
        self.img_gen_api_url = "http://127.0.0.1:7860/"
        self.cluster_requested_models = [] # The models which we allow to generate during cluster mode
        
        
    def __setattr__(self, name, value):
        new_variable = name not in self.__dict__
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        #Put variable change actions here
        if name not in self.local_only_variables and name[0] != "_" and not new_variable:
            process_variable_changes(self.socketio, self.__class__.__name__.replace("_settings", ""), name, value, old_value)
        
class system_settings(settings):
    local_only_variables = ['socketio', 'lua_state', 'lua_logname', 'lua_koboldbridge', 'lua_kobold', 
                            'lua_koboldcore', 'regex_sl', 'acregex_ai', 'acregex_ui', 'comregex_ai', 
                            'comregex_ui', 'sp', '_horde_pid', 'inference_config', 'image_pipeline', 
                            'summarizer', 'summary_tokenizer']
    no_save_variables = ['socketio', 'lua_state', 'lua_logname', 'lua_koboldbridge', 'lua_kobold', 
                         'lua_koboldcore', 'sp', 'sp_length', '_horde_pid', 'horde_share', 'aibusy', 
                         'serverstarted', 'inference_config', 'image_pipeline', 'summarizer', 
                         'summary_tokenizer', 'use_colab_tpu', 'noai', 'disable_set_aibusy', 'cloudflare_link']
    settings_name = "system"
    def __init__(self, socketio, koboldai_var):
        self.socketio = socketio
        self.noai        = False  # Runs the script without starting up the transformers pipeline
        self.aibusy      = False  # Stops submissions while the AI is working
        self.status_message = ""
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
        self.splist      = []
        self.spselect    = ""     # Temporary storage for soft prompt filename to load
        self.spmeta      = None   # Metadata of current soft prompt, or None if not using a soft prompt
        self.spname      = "Not in Use"     # Name of the soft prompt    
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
        self.quiet       = False # If set will suppress any story text from being printed to the console (will only be seen on the client web page)
        self.use_colab_tpu  = os.environ.get("COLAB_TPU_ADDR", "") != "" or os.environ.get("TPU_NAME", "") != ""  # Whether or not we're in a Colab TPU instance or Kaggle TPU instance and are going to use the TPU rather than the CPU
        self.aria2_port  = 6799 #Specify the port on which aria2's RPC interface will be open if aria2 is installed (defaults to 6799)
        self.standalone = False
        self.api_tokenizer_id = None
        self.disable_set_aibusy = False
        self.disable_input_formatting = False
        self.disable_output_formatting = False
        self.full_determinism = False  # Whether or not full determinism is enabled
        self.seed_specified = False  # Whether or not the current RNG seed was specified by the user (in their settings file)
        self.seed        = None   # The current RNG seed (as an int), or None if unknown
        self.alt_gen = False # Use the calc_ai_text method for generating text to go to the AI
        self.theme_list = [".".join(f.split(".")[:-1]) for f in os.listdir("./themes") if os.path.isfile(os.path.join("./themes", f))]
        self.cloudflare_link = ""
        self.story_loads = {} #dict of when each story was last loaded
        self.standalone = False
        self.disable_set_aibusy = False
        self.disable_input_formatting = False
        self.disable_output_formatting = False
        self.api_tokenizer_id = None
        self.port = 5000
        try:
            import google.colab
            self.on_colab = True
        except:
            self.on_colab = False
            pass
        print("Colab Check: {}".format(self.on_colab))
        self.horde_share = False
        self._horde_pid = None
        self.sh_apikey   = ""     # API key to use for txt2img from the Stable Horde.
        self.generating_image = False #The current status of image generation
        self.image_pipeline = None
        self.summarizer = None
        self.summary_tokenizer = None
        self.keep_img_gen_in_memory = False
        self.cookies = {} #cookies for colab since colab's URL changes, cookies are lost
        self.experimental_features = False
        
        @dataclass
        class _inference_config:
            do_streaming: bool = False
            do_dynamic_wi: bool = False
            # Genamt stopping is mostly tied to Dynamic WI
            stop_at_genamt: bool = False
            do_core: bool = True
        self.inference_config = _inference_config()
        
        self._koboldai_var = koboldai_var
        
        
    def __setattr__(self, name, value):
        new_variable = name not in self.__dict__
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        
        #Put variable change actions here
        if name == 'serverstarted':
            global serverstarted
            serverstarted = value
        if name not in self.local_only_variables and name[0] != "_" and not new_variable:
            process_variable_changes(self.socketio, self.__class__.__name__.replace("_settings", ""), name, value, old_value)
            
            #if name == "aibusy" and value == False and self.abort == True:
            #    koboldai_vars.abort = False
            
            #for original UI
            if name == 'sp_changed':
                self.socketio.emit('from_server', {'cmd': 'spstatitems', 'data': {self.spfilename: self.spmeta} if self.allowsp and len(self.spfilename) else {}}, namespace=None, broadcast=True, room="UI_1")
                super().__setattr__("sp_changed", False)
            
            if name == 'keep_img_gen_in_memory' and value == False:
                self.image_pipeline = None
            
            if name == 'alt_gen' and old_value != value:
                logger.debug("Calcing AI Text from setting alt_gen")
                self._koboldai_var.calc_ai_text()
            
            if name == 'horde_share':
                if self.on_colab == False:
                    if os.path.exists("./KoboldAI-Horde"):
                        if value == True:
                            if self._horde_pid is None:
                                import subprocess
                                import random
                                random.seed()
                                if os.path.exists('./KoboldAI-Horde/venv/scripts/python.exe'):
                                    self._horde_pid = subprocess.Popen(['./KoboldAI-Horde/venv/scripts/python.exe', './KoboldAI-Horde/bridge.py', 
                                                                            '--kai_url', 'http://127.0.0.1:{}'.format(self.port), '--cluster_url', "http://koboldai.net"])
                                else:
                                    self._horde_pid = subprocess.Popen(['./KoboldAI-Horde/venv/bin/python', './KoboldAI-Horde/bridge.py', 
                                                                            '--kai_url', 'http://127.0.0.1:{}'.format(self.port), '--cluster_url', "http://koboldai.net"])
                        else:
                            if self._horde_pid is not None:
                                print("kill bridge")
                                self._horde_pid.terminate()
                                self._horde_pid = None
                
class KoboldStoryRegister(object):
    def __init__(self, socketio, story_settings, koboldai_vars, tokenizer=None, sequence=[]):
        self.socketio = socketio
        self.koboldai_vars = koboldai_vars
        #### DO NOT DIRECTLY EDIT THE ACTIONS DICT. IT WILL NOT TRANSMIT TO CLIENT. USE FUCTIONS BELOW TO DO SO ###
        #### doing actions[x] = game text is OK
        self.actions = {} #keys = "Selected Text", "Wi_highlighted_text", "Options", "Selected Text Length", "Probabilities". 
                          #Options being a list of dict with keys of "text", "Pinned", "Previous Selection", "Edited", "Probabilities"
        self.action_count = -1
        self.story_settings = story_settings
        for item in sequence:
            self.append(item)
    
    def reset(self, sequence=[]):
        self.__init__(self.socketio, self.story_settings, self.koboldai_vars, sequence=sequence)
        
    def add_wi_to_action(self, action_id, key, content, uid, no_transmit=False):
        old = self.story_settings.prompt_wi_highlighted_text.copy() if action_id == -1 else self.actions[action_id].copy()
        #First check to see if we have the wi_highlighted_text variable
        if action_id != -1:
            if 'wi_highlighted_text' not in self.actions[action_id]:
                self.actions[action_id]['wi_highlighted_text'] = [{"text": self.actions[action_id]['Selected Text'], "WI matches": None, "WI Text": ""}]
            action = self.actions[action_id]['wi_highlighted_text']
        else:
            action = self.story_settings.prompt_wi_highlighted_text
        
        i=0
        while i < len(action):
            if action[i]['WI matches'] is None and key in action[i]['text'] and action[i]['text'] != "":
                chunk = action[i]
                for j in range(len(chunk['text'])-1, -1, -1):
                    if key in chunk['text'][j:]:
                        #OK, we found the start of our key. We'll split our text into before key, key, after key

                        pre_text = chunk['text'][:j] if j != 0 else None                            
                        post_text = chunk['text'][j+len(key):] if j != len(chunk['text']) else None
                        text = chunk['text'][j:j+len(key)]

                        action[i]['text'] = text
                        action[i]['WI matches'] = uid
                        action[i]['WI Text'] = content
                        adder = 1
                        if pre_text is not None:
                            action.insert(i, {"text": pre_text, "WI matches": None, "WI Text": ""})
                            adder += 1
                        if post_text is not None:   
                            action.insert(i+adder, {"text": post_text, "WI matches": None, "WI Text": ""})
                        break;
            else:
                i+=1
        if action_id != -1:
            if old != self.actions[action_id]:
                if not no_transmit:
                    process_variable_changes(self.socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, old)
        else:
            if old != self.story_settings.prompt_wi_highlighted_text:
                if not no_transmit:
                    process_variable_changes(self.socketio, "story", 'prompt_wi_highlighted_text', self.story_settings.prompt_wi_highlighted_text, old)
                    
        
    def __str__(self):
        if len(self.actions) > 0:
            return "".join([x['Selected Text'] for ignore, x in sorted(self.actions.items())])
        else:
            return ""
        
    def __repr__(self):
        return self.__str__()
        
    def __iter__(self):
        self.itter = -1
        return self
        
    def __next__(self):
        self.itter += 1
        if self.itter <= self.action_count:
            return self.itter
        else:
            raise StopIteration
        
    def __getitem__(self, i):
        if isinstance(i, slice):
            temp = [self.actions[x]["Selected Text"] for x in list(self.actions)[i]]
            return temp
        else:
            if i < 0:
                return self.actions[self.action_count+i+1]["Selected Text"]
            return self.actions[i]["Selected Text"]
        
    def __setitem__(self, i, text):
        if self.koboldai_vars.remove_double_space:
            while "  " in text:
                text = text.replace("  ", " ")
        if i in self.actions:
            old = self.actions[i]
            old_text = self.actions[i]["Selected Text"]
            if self.actions[i]["Selected Text"] != text:
                self.actions[i]["Selected Text"] = text
                self.actions[i]["Probabilities"] = []
            if "Options" in self.actions[i]:
                for j in range(len(self.actions[i]["Options"])):
                    if self.actions[i]["Options"][j]["text"] == text:
                        del self.actions[i]["Options"][j]
            if old_text != "":
                self.actions[i]["Options"].append({"text": old_text, "Pinned": False, "Previous Selection": False, "Edited": True})
        else:
            old_text = None
            old_length = None
            old = None
            self.actions[i] = {"Selected Text": text, "Probabilities": [], "Options": []}
            
        self.story_settings.assign_world_info_to_actions(action_id=i, no_transmit=True)
        process_variable_changes(self.socketio, "story", 'actions', {"id": i, 'action':  self.actions[i]}, old)
        logger.debug("Calcing AI Text from Action __setitem__")
        ignore = self.koboldai_vars.calc_ai_text()
        self.set_game_saved()
    
    def __len__(self):
        return self.action_count+1 if self.action_count >=0 else 0
    
    def __reversed__(self):
        return reversed(range(self.action_count+1))
    
    def values(self):
        return [self.actions[k]["Selected Text"] for k in self.actions]
    
    def options(self, ui_version=2):
        if ui_version == 1:
            return [{"Selected Text": self.actions[k]["Selected Text"], "Alternative Text": self.actions[k]["Options"]} for k in self.actions]
        else:
            return [self.actions[k]["Options"] for k in self.actions]
    
    def to_json(self):
        return {"action_count": self.action_count, "actions": self.actions}
        
    def load_json(self, json_data):
        if type(json_data) == str:
            import json
            json_data = json.loads(json_data)
        self.action_count = int(json_data['action_count'])
        #JSON forces keys to be strings, so let's fix that
        temp = {}
        data_to_send = []
        for item in json_data['actions']:
            temp[int(item)] = json_data['actions'][item]
            if int(item) >= self.action_count-100: #sending last 100 items to UI
                data_to_send.append({"id": item, 'action':  temp[int(item)]})
        
        process_variable_changes(self.socketio, "story", 'actions', data_to_send, None)
        
        self.actions = temp
        #Check if our json has our new world info highlighting data
        if len(self.actions) > 0:
            if 'wi_highlighted_text' not in self.actions[0]:
                self.story_settings.assign_world_info_to_actions(no_transmit=True)
        
        logger.debug("Calcing AI Text from Action load from json")
        ignore = self.koboldai_vars.calc_ai_text()
        self.set_game_saved()
        
    def append(self, text, action_id_offset=0, recalc=True):
        if self.koboldai_vars.remove_double_space:
            while "  " in text:
                text = text.replace("  ", " ")
        self.clear_unused_options()
        self.action_count+=1
        action_id = self.action_count + action_id_offset
        if action_id in self.actions:
            if self.actions[action_id]["Selected Text"] != text:
                self.actions[action_id]["Selected Text"] = text
                self.actions[action_id]["Probabilities"] = []
            selected_text_length = 0
            self.actions[action_id]["Selected Text Length"] = selected_text_length
            for item in self.actions[action_id]["Options"]:
                if item['text'] == text:
                    old_options = self.actions[action_id]["Options"]
                    del item
                    
        else:
            selected_text_length = 0
            
            self.actions[action_id] = {"Selected Text": text, "Selected Text Length": selected_text_length, 
                                               "Options": [], "Probabilities": []}
            
        if self.story_settings is not None:
            self.story_settings.assign_world_info_to_actions(action_id=action_id, no_transmit=True)
            process_variable_changes(self.socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, None)
        self.set_game_saved()
        if recalc:
            logger.debug("Calcing AI Text from Action Append")
            ignore = self.koboldai_vars.calc_ai_text()
    
    def append_options(self, option_list):
        if self.action_count+1 in self.actions:
            #First let's check if we did streaming, as we can just replace those items with these
            old_options = copy.deepcopy(self.actions[self.action_count+1]["Options"])
            i=-1
            for option in option_list:
                i+=1
                found = False
                for item in self.actions[self.action_count+1]["Options"]:
                    if 'stream_id' in item and item['stream_id'] == i:
                        item['text'] = option
                        del item['stream_id']
                        found = True
                        break
                    elif item['text'] == option:
                        found = True
                        if 'stream_id' in item:
                            del item['stream_id']
                        found = True
                        break
                        
                if not found:
                    self.actions[self.action_count+1]['Options'].append({"text": option, "Pinned": False, "Previous Selection": False, "Edited": False, "Probabilities": []})
        else:
            old_options = None
            self.actions[self.action_count+1] = {"Selected Text": "", "Selected Text Length": 0, "Options": [{"text": x, "Pinned": False, "Previous Selection": False, "Edited": False, "Probabilities": []} for x in option_list]}
        process_variable_changes(self.socketio, "story", 'actions', {"id": self.action_count+1, 'action':  self.actions[self.action_count+1]}, None)
        self.set_game_saved()
            
    def set_options(self, option_list, action_id):
        if action_id not in self.actions:
            old_options = None
            self.actions[action_id] = {"Selected Text": "", "Options": option_list}
        else:
            old_options = self.actions[action_id]["Options"]
            self.actions[action_id]["Options"] = []
            for item in option_list:
                for old_item in old_options:
                    if item['text'] == old_item['text']:
                        #We already have this option, so we need to save the probabilities
                        item['Probabilities'] = old_item['Probabilities']
                    self.actions[action_id]["Options"].append(item)
        process_variable_changes(self.socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, None)
    
    def clear_unused_options(self, pointer=None):
        new_options = []
        old_options = None
        if pointer is None:
            pointer = self.action_count+1
        if pointer in self.actions:
            old_options = copy.deepcopy(self.actions[pointer]["Options"])
            self.actions[pointer]["Options"] = [x for x in self.actions[pointer]["Options"] if x["Pinned"] or x["Previous Selection"] or x["Edited"]]
            new_options = self.actions[pointer]["Options"]
            process_variable_changes(self.socketio, "story", 'actions', {"id": pointer, 'action':  self.actions[pointer]}, None)
        self.set_game_saved()
    
    def set_pin(self, action_step, option_number):
        if action_step in self.actions:
            if option_number < len(self.actions[action_step]['Options']):
                old_options = copy.deepcopy(self.actions[action_step]["Options"])
                self.actions[action_step]['Options'][option_number]['Pinned'] = True
                process_variable_changes(self.socketio, "story", 'actions', {"id": action_step, 'action':  self.actions[action_step]}, None)
                self.set_game_saved()
    
    def unset_pin(self, action_step, option_number):
        if action_step in self.actions:
            old_options = copy.deepcopy(self.actions[action_step]["Options"])
            if option_number < len(self.actions[action_step]['Options']):
                self.actions[action_step]['Options'][option_number]['Pinned'] = False
                process_variable_changes(self.socketio, "story", 'actions', {"id": action_step, 'action':  self.actions[action_step]}, None)
                self.set_game_saved()
    
    def toggle_pin(self, action_step, option_number):
        if action_step in self.actions:
            old_options = copy.deepcopy(self.actions[action_step]["Options"])
            if option_number < len(self.actions[action_step]['Options']):
                self.actions[action_step]['Options'][option_number]['Pinned'] = not self.actions[action_step]['Options'][option_number]['Pinned']
                process_variable_changes(self.socketio, "story", 'actions', {"id": action_step, 'action':  self.actions[action_step]}, None)
                self.set_game_saved()
    
    def use_option(self, option_number, action_step=None):
        if action_step is None:
            action_step = self.action_count+1
        if action_step in self.actions:
            old_options = copy.deepcopy(self.actions[action_step]["Options"])
            old_text = self.actions[action_step]["Selected Text"]
            old_length = self.actions[action_step]["Selected Text Length"]
            if option_number < len(self.actions[action_step]['Options']):
                self.actions[action_step]["Selected Text"] = self.actions[action_step]['Options'][option_number]['text']
                if 'Probabilities' in self.actions[action_step]['Options'][option_number]:
                    self.actions[action_step]["Probabilities"] = self.actions[action_step]['Options'][option_number]['Probabilities']
                if self.koboldai_vars.tokenizer is not None:
                    self.actions[action_step]['Selected Text Length'] = len(self.koboldai_vars.tokenizer.encode(self.actions[action_step]['Options'][option_number]['text']))
                else:
                    self.actions[action_step]['Selected Text Length'] = 0
                del self.actions[action_step]['Options'][option_number]
                #If this is the current spot in the story, advance
                if action_step-1 == self.action_count:
                    self.action_count+=1
                    self.socketio.emit("var_changed", {"classname": "actions", "name": "Action Count", "old_value": None, "value":self.action_count, "transmit_time": str(datetime.datetime.now())}, broadcast=True, room="UI_2")
                self.story_settings.assign_world_info_to_actions(action_id=action_step, no_transmit=True)
                self.clear_unused_options(pointer=action_step)
                #process_variable_changes(self.socketio, "story", 'actions', {"id": action_step, 'action':  self.actions[action_step]}, None)
                self.set_game_saved()
                logger.debug("Calcing AI Text from Action Use Option")
                ignore = self.koboldai_vars.calc_ai_text()
    
    def delete_option(self, option_number, action_step=None):
        if action_step is None:
            action_step = self.action_count+1
        if action_step in self.actions:
            if option_number < len(self.actions[action_step]['Options']):
                del self.actions[action_step]['Options'][option_number]
                process_variable_changes(self.socketio, "story", 'actions', {"id": action_step, 'action':  self.actions[action_step]}, None)
                self.set_game_saved()
    
    def delete_action(self, action_id):
        if action_id in self.actions:
            old_options = copy.deepcopy(self.actions[action_id]["Options"])
            old_text = self.actions[action_id]["Selected Text"]
            old_length = self.actions[action_id]["Selected Text Length"]
            self.actions[action_id]["Options"].append({"text": self.actions[action_id]["Selected Text"], "Pinned": False, "Previous Selection": True, "Edited": False})
            self.actions[action_id]["Selected Text"] = ""
            if "wi_highlighted_text" in self.actions[action_id]:
                del self.actions[action_id]["wi_highlighted_text"]
            self.actions[action_id]['Selected Text Length'] = 0
            self.action_count -= 1
            process_variable_changes(self.socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, None)
            self.set_game_saved()
            logger.debug("Calcing AI Text from Action Delete")
            ignore = self.koboldai_vars.calc_ai_text()
            
    def pop(self):
        if self.action_count >= 0:
            text = self.actions[self.action_count]['Selected Text']
            self.delete_action(self.action_count)
            logger.debug("Calcing AI Text from Action Pop")
            return text
        else:
            return None
            
    def get_last_key(self):
        return self.action_count
            
    def get_current_options(self):
        if self.action_count+1 in self.actions:
            return self.actions[self.action_count+1]["Options"]
        else:
            return []
            
    def get_current_options_no_edits(self, ui=2):
        if ui==2:
            if self.action_count+1 in self.actions:
                return [x for x in self.actions[self.action_count+1]["Options"] if x["Edited"] == False and x['Previous Selection'] == False]
            else:
                return []
        else:
            if self.action_count+1 in self.actions:
                return [[x['text'], "pinned" if x['Pinned'] else 'normal'] for x in self.actions[self.action_count+1]["Options"] if x["Edited"] == False and x['Previous Selection'] == False]
            else:
                return []
    
    def get_redo_options(self):
        if self.action_count+1 in self.actions:
            return [x for x in self.actions[self.action_count+1]['Options'] if x['Pinned'] or x['Previous Selection']]
        else:
            return []
        
    def set_game_saved(self):
        if self.story_settings is not None:
            self.story_settings.gamesaved = False
    
    def stream_tokens(self, text_list):
        if len(text_list) > 1:
            if self.action_count+1 in self.actions:
                for i in range(len(text_list)):
                    found = False
                    for j in range(len(self.actions[self.action_count+1]['Options'])):
                        if 'stream_id' in self.actions[self.action_count+1]['Options'][j]:
                            if self.actions[self.action_count+1]['Options'][j]['stream_id'] == i:
                                found = True
                                self.actions[self.action_count+1]['Options'][j]['text'] = "{}{}".format(self.actions[self.action_count+1]['Options'][j]['text'], text_list[i])
                    if not found:
                        self.actions[self.action_count+1]['Options'].append({"text": text_list[i], "Pinned": False, "Previous Selection": False, "Edited": False, "Probabilities": [], "stream_id": i})
            else:
                self.actions[self.action_count+1] = {"Selected Text": "", "Selected Text Length": 0, "Options": []}
                for i in range(len(text_list)):
                    self.actions[self.action_count+1]['Options'].append({"text": text_list[i], "Pinned": False, "Previous Selection": False, "Edited": False, "Probabilities": [], "stream_id": i})
        
            #We need to see if this is the last token being streamed. If so due to the rely it will come in AFTER the actual trimmed final text overwriting it in the UI
            if self.koboldai_vars.tokenizer is not None:
                if len(self.koboldai_vars.tokenizer.encode(self.actions[self.action_count+1]["Options"][0]['text'])) != self.koboldai_vars.genamt:
                    #process_variable_changes(self.socketio, "actions", "Options", {"id": self.action_count+1, "options": self.actions[self.action_count+1]["Options"]}, {"id": self.action_count+1, "options": None})
                    process_variable_changes(self.socketio, "story", 'actions', {"id": self.action_count+1, 'action':  self.actions[self.action_count+1]}, None)
        else:
            #We're streaming single options so our output is our selected
            #First we need to see if this is actually the prompt. If so we'll just not do streaming:
            if self.story_settings.prompt != "":
                if self.action_count+1 in self.actions:
                    if self.koboldai_vars.tokenizer is not None:
                        selected_text_length = len(self.koboldai_vars.tokenizer.encode(self.actions[self.action_count+1]['Selected Text']))
                    else:
                        selected_text_length = 0
                    self.actions[self.action_count+1]['Selected Text'] = "{}{}".format(self.actions[self.action_count+1]['Selected Text'], text_list[0])
                    self.actions[self.action_count+1]['Selected Text Length'] = selected_text_length
                else:
                    if self.koboldai_vars.tokenizer is not None:
                        selected_text_length = len(self.koboldai_vars.tokenizer.encode(text_list[0]))
                    else:
                        selected_text_length = 0
                    self.actions[self.action_count+1] = {"Selected Text": text_list[0], "Selected Text Length": selected_text_length, "Options": []}
                
                
                
                if self.koboldai_vars.tokenizer is not None:
                    if len(self.koboldai_vars.tokenizer.encode(self.actions[self.action_count+1]['Selected Text'])) != self.koboldai_vars.genamt:
                        #ui1
                        if queue is not None:
                            queue.put(["from_server", {"cmd": "streamtoken", "data": [{'decoded': text_list[0]}]}, {"broadcast":True, "room":"UI_1"}])
                        #process_variable_changes(self.socketio, "actions", "Options", {"id": self.action_count+1, "options": self.actions[self.action_count+1]["Options"]}, {"id": self.action_count+1, "options": None})
                        process_variable_changes(self.socketio, "story", 'actions', {"id": self.action_count+1, 'action':  self.actions[self.action_count+1]}, None)
    
    def set_probabilities(self, probabilities, action_id=None):
        if action_id is None:
            action_id = self.action_count+1
        if action_id in self.actions:
            self.actions[action_id]['Probabilities'].append(probabilities)
            process_variable_changes(self.socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, None)
            
    def set_option_probabilities(self, probabilities, option_number, action_id=None):
        if action_id is None:
            action_id = self.action_count+1
        if action_id in self.actions:
            old_options = self.actions[action_id]["Options"]
            if option_number < len(self.actions[action_id]["Options"]):
                if "Probabilities" not in self.actions[action_id]["Options"][option_number]:
                    self.actions[action_id]["Options"][option_number]["Probabilities"] = []
                self.actions[action_id]["Options"][option_number]['Probabilities'].append(probabilities)
                process_variable_changes(self.socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, None)
    
    def to_sentences(self, submitted_text=""):
        #start_time = time.time()
        #we're going to split our actions by sentence for better context. We'll add in which actions the sentence covers. Prompt will be added at a -1 ID
        actions = {i: self.actions[i]['Selected Text'] for i in self.actions}
        if self.story_settings is None:
            actions[-1] = ""
        else:
            actions[-1] = self.story_settings.prompt
        if submitted_text != "":
            actions[self.action_count+1] = submitted_text
        action_text = self.__str__()
        action_text = "{}{}{}".format("" if self.story_settings is None else self.story_settings.prompt, action_text, submitted_text)
        ###########action_text_split = [sentence, actions used in sentence, token length, included in AI context]################
        action_text_split = [[x, [], 0, False] for x in re.findall(".*?[.!?]\s+", action_text, re.S)]
        #The above line can trim out the last sentence if it's incomplete. Let's check for that and add it back in
        if len("".join([x[0] for x in action_text_split])) < len(action_text):
            action_text_split.append([action_text[len("".join([x[0] for x in action_text_split])):], [], 0, False])
        #The last action shouldn't have the extra space from the sentence splitting, so let's remove it
        if len(action_text_split) == 0:
            return []
        
        Action_Position = [-1, len(actions[-1])] #First element is the action item, second is how much text is left
        Sentence_Position = [0, len(action_text_split[0][0])]
        while True:
            advance_action = False
            advance_sentence = False
            if Action_Position[1] <= Sentence_Position[1]:
                #We have enough text in the sentence to completely cover the action. Advance it to the next action
                advance_action = True
            if Sentence_Position[1] <= Action_Position[1]:
                advance_sentence = True
            if Action_Position[0] not in action_text_split[Sentence_Position[0]][1]:
                #Since this action is in the sentence, add it to the list if it's not already there
                action_text_split[Sentence_Position[0]][1].append(Action_Position[0])
            #Fix the text length leftovers first since they interact with each other
            if not advance_action:
                Action_Position[1] -= Sentence_Position[1]
            if not advance_sentence:
                Sentence_Position[1] -= Action_Position[1]
                
            if advance_action:
                Action_Position[0] += 1
                if Action_Position[0] > max(actions):
                    break
                Action_Position[1] = len(actions[Action_Position[0]])
            if advance_sentence:
                Sentence_Position[0] += 1
                if Sentence_Position[0] >= len(action_text_split):
                    break
                Sentence_Position[1] = len(action_text_split[Sentence_Position[0]][0])
        #OK, action_text_split now contains a list of [sentence including trailing space if needed, [action IDs that sentence includes]]
        #logger.debug("to_sentences: {}s".format(time.time()-start_time))
        return action_text_split
    
    def __setattr__(self, name, value):
        new_variable = name not in self.__dict__
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        if name == 'action_count' and not new_variable:
            process_variable_changes(self.socketio, "actions", "Action Count", value, old_value)

class KoboldWorldInfo(object):
    
    def __init__(self, socketio, story_settings, koboldai_vars, tokenizer=None):
        self.socketio = socketio
        self.koboldai_vars = koboldai_vars
        self.world_info = {}
        self.world_info_folder = OrderedDict()
        self.world_info_folder['root'] = []
        self.story_settings = story_settings
        
    def reset(self):
        self.__init__(self.socketio, self.story_settings, self.koboldai_vars)
    
    def __iter__(self):
        self.itter = -1
        return self
        
    def __next__(self):
        self.itter += 1
        if self.itter < len(self.world_info):
            try:
                return self.world_info[list(self.world_info)[self.itter]].copy()
            except:
                print(self.itter)
                print(list(self.world_info))
                raise
        else:
            raise StopIteration
        
    def __getitem__(self, i):
        return self.self.world_info[i].copy()
    
    def __len__(self):
        return len(self.world_info)
    
    def add_folder(self, folder):
        if folder in self.world_info_folder:
            i=0
            while "{} {}".format(folder, i) in self.world_info_folder:
                i+=1
            folder = "{} {}".format(folder, i)
        self.world_info_folder[folder] = []
        self.story_settings.gamesaved = False
        self.sync_world_info_to_old_format()
        if self.socketio is not None:
            self.socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
        
    def delete_folder(self, folder):
        keys = [key for key in self.world_info]
        for key in keys:
            if self.world_info[key]['folder'] == folder:
                self.delete(key)
        if folder in self.world_info_folder:
            del self.world_info_folder[folder]
        if self.socketio is not None:
            self.socketio.emit("delete_world_info_folder", folder, broadcast=True, room="UI_2")
        logger.debug("Calcing AI Text from WI Folder Delete")
        ignore = self.koboldai_vars.calc_ai_text()
        
    def add_item_to_folder(self, uid, folder, before=None):
        if uid in self.world_info:
            #fiirst we need to remove the item from whatever folder it's in
            for temp in self.world_info_folder:
                if uid in self.world_info_folder[temp]:
                    self.world_info_folder[temp].remove(uid)
            #Now we add it to the folder list
            if folder not in self.world_info_folder:
                self.world_info_folder[folder] = []
            if before is None:
                self.world_info_folder[folder].append(uid)
            else:
                for i in range(len(self.world_info_folder[folder])):
                    if self.world_info_folder[folder][i] == before:
                        self.world_info_folder[folder].insert(i, uid)
                        break
                
            #Finally, adjust the folder tag in the element
            self.world_info[uid]['folder'] = folder
            self.story_settings.gamesaved = False
            self.sync_world_info_to_old_format()
        if self.socketio is not None:
            self.socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
                
    def add_item(self, title, key, keysecondary, folder, constant, manual_text, 
                 comment, use_wpp=False, wpp={'name': "", 'type': "", 'format': "W++", 'attributes': {}}, 
                 v1_uid=None, recalc=True, sync=True, send_to_ui=True):
        if len(self.world_info) == 0:
            uid = 0
        else:
            uid = max(self.world_info)+1
        if use_wpp:
            if wpp['format'] == "W++":
                content = '[{}("{}")\n{{\n'.format(wpp['type'], wpp['name'])
                for attribute in wpp['attributes']:
                    content = "{}{}({})\n".format(content, attribute, " + ".join(['"{}"'.format(x) for x in wpp['attributes'][attribute]]))
                content = "{}}}]".format(content)
            else:
                content = '[ {}: "{}";'.format(wpp['type'], wpp['name'])
                for attribute in wpp['attributes']:
                    content = "{} {}: {};".format(content, attribute, ", ".join(['"{}"'.format(x) for x in wpp['attributes'][attribute]]))
                content = "{} ]".format(content[:-1])
        else:
            content = manual_text
        if self.koboldai_vars.tokenizer is not None:
            token_length = len(self.koboldai_vars.tokenizer.encode(content))
        else:
            token_length = 0
        if folder is None:
            folder = "root"
        
        if isinstance(key, str):
            key = [x.strip() for x in key.split(",")]
            if key == [""]:
                key = []
        if isinstance(keysecondary, str):
            keysecondary = [x.strip() for x in keysecondary.split(",")]
            if keysecondary == [""]:
                keysecondary = []
        
        try:
            self.world_info[uid] = {"uid": uid,
                                    "title": title,
                                    "key": key,
                                    "keysecondary": keysecondary,
                                    "folder": folder,
                                    "constant": constant,
                                    'manual_text': manual_text,
                                    "content": content,
                                    "comment": comment,
                                    "token_length": token_length,
                                    "selective": len(keysecondary) > 0,
                                    "used_in_game": constant,
                                    'wpp': wpp,
                                    'use_wpp': use_wpp,
                                    'v1_uid': v1_uid
                                    }
        except:
            print("Error:")
            print(key)
            print(title)
            raise
        if folder not in self.world_info_folder:
            self.world_info_folder[folder] = []
        self.world_info_folder[folder].append(uid)
        self.story_settings.gamesaved = False
        if sync:
            self.sync_world_info_to_old_format()
        
        self.story_settings.assign_world_info_to_actions(wuid=uid)
        
        if self.socketio is not None and send_to_ui:
            self.socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
            self.socketio.emit("world_info_entry", self.world_info[uid], broadcast=True, room="UI_2")
        if recalc:
            logger.debug("Calcing AI Text from WI Add")
            ignore = self.koboldai_vars.calc_ai_text()
        return uid
        
    def edit_item(self, uid, title, key, keysecondary, folder, constant, manual_text, comment, use_wpp=False, before=None, wpp={'name': "", 'type': "", 'format': "W++", 'attributes': {}}):
        logger.debug("Editing World Info {}: {}".format(uid, title))
        old_folder = self.world_info[uid]['folder']
        #move the world info entry if the folder changed or if there is a new order requested
        if old_folder != folder or before is not None:
            self.add_item_to_folder(uid, folder, before=before)
        if use_wpp:
            if wpp['format'] == "W++":
                content = '[{}("{}")\n{{\n'.format(wpp['type'], wpp['name'])
                for attribute in wpp['attributes']:
                    content = "{}{}({})\n".format(content, attribute, " + ".join(['"{}"'.format(x) for x in wpp['attributes'][attribute]]))
                content = "{}}}]".format(content)
            else:
                content = '[ {}: "{}";'.format(wpp['type'], wpp['name'])
                for attribute in wpp['attributes']:
                    content = "{} {}: {};".format(content, attribute, ", ".join(['"{}"'.format(x) for x in wpp['attributes'][attribute]]))
                content = "{} ]".format(content[:-1])
        else:
            content = manual_text
        if self.koboldai_vars.tokenizer is not None:
            token_length = len(self.koboldai_vars.tokenizer.encode(content))
        else:
            token_length = 0
        if folder is None:
            folder = "root"
            
        self.world_info[uid] = {"uid": uid,
                                "title": title,
                                "key": key,
                                "keysecondary": keysecondary,
                                "folder": folder,
                                "constant": constant,
                                'manual_text': manual_text,
                                "content": content,
                                "comment": comment,
                                "token_length": token_length,
                                "selective": len(keysecondary) > 0,
                                "used_in_game": constant,
                                'wpp': wpp,
                                'use_wpp': use_wpp
                                }
                                
        self.story_settings.gamesaved = False
        self.sync_world_info_to_old_format()
        self.story_settings.assign_world_info_to_actions(wuid=uid)
        logger.debug("Calcing AI Text from WI Edit")
        ignore = self.koboldai_vars.calc_ai_text()
        
        if self.socketio is not None:
            self.socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
            self.socketio.emit("world_info_entry", self.world_info[uid], broadcast=True, room="UI_2")
        
    def delete(self, uid):
        del self.world_info[uid]
        for folder in self.world_info_folder:
            if uid in self.world_info_folder[folder]:
                self.world_info_folder[folder].remove(uid)
        
        self.story_settings.gamesaved = False
        self.sync_world_info_to_old_format()
        if self.socketio is not None:
            self.socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
            self.socketio.emit("delete_world_info_entry", uid, broadcast=True, room="UI_2")
        logger.debug("Calcing AI Text from WI Delete")
        ignore = self.koboldai_vars.calc_ai_text()
    
    def rename_folder(self, old_folder, folder):
        self.story_settings.gamesaved = False
        if folder in self.world_info_folder:
            i=0
            while "{} {}".format(folder, i) in self.world_info_folder:
                i+=1
            folder = "{} {}".format(folder, i)
        
        self.world_info_folder[folder] = self.world_info_folder[old_folder]
        #The folder dict is ordered and we want the new key to be in the same location as the last key. To do that we need to find all of the keys after old_folder
        after_old = False
        folder_order = [x for x in self.world_info_folder]
        for check_folder in folder_order:
            if check_folder == old_folder:
                after_old = True
            elif after_old and check_folder != folder:
                self.world_info_folder.move_to_end(check_folder)
        del self.world_info_folder[old_folder]
        
        # Need to change the folder properties on each affected world info item
        for uid in self.world_info:
            if self.world_info[uid]['folder'] == old_folder:
                self.world_info[uid]['folder'] = folder
        
        self.story_settings.gamesaved = False
        self.sync_world_info_to_old_format()
        if self.socketio is not None:
            self.socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
    
    def reorder(self, uid, before):
        self.add_item_to_folder(uid, self.world_info[before]['folder'], before=before)
        self.sync_world_info_to_old_format()
    
    def send_to_ui(self):
        if self.socketio is not None:
            self.socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
            logger.debug("Sending all world info from send_to_ui")
            self.socketio.emit("world_info_entry", [self.world_info[uid] for uid in self.world_info], broadcast=True, room="UI_2")
    
    def to_json(self, folder=None):
        if folder is None:
            return {
                    "folders": {x: self.world_info_folder[x] for x in self.world_info_folder},
                    "entries": self.world_info
                   }
        else:
            return {
                    "folders": {x: self.world_info_folder[x] for x in self.world_info_folder if x == folder},
                    "entries": {x: self.world_info[x] for x in self.world_info if self.world_info[x]['folder'] == folder}
                   }
    
    def load_json(self, data, folder=None):
        if folder is None:
            self.world_info = {int(x): data['entries'][x] for x in data['entries']}
            self.world_info_folder = data['folders']
        
        #Add the item
        for uid, item in data['entries'].items():
            start_time = time.time()
            self.add_item(item['title'] if 'title' in item else item['key'][0], 
                          item['key'] if 'key' in item else [], 
                          item['keysecondary'] if 'keysecondary' in item else [], 
                          folder, 
                          item['constant'] if 'constant' in item else False, 
                          item['manual_text'] if 'manual_text' in item else item['content'], 
                          item['comment'] if 'comment' in item else '',
                          use_wpp=item['use_wpp'] if 'use_wpp' in item else False, 
                          wpp=item['wpp'] if 'wpp' in item else {'name': "", 'type': "", 'format': "W++", 'attributes': {}},
                          recalc=False, sync=False)
            logger.debug("Load World Info {} took {}s".format(uid, time.time()-start_time))
        try:
            start_time = time.time()
            self.sync_world_info_to_old_format()
            logger.debug("Syncing WI2 to WI1 took {}s".format(time.time()-start_time))
        except:
            print(self.world_info)
            print(data)
            raise
        self.send_to_ui()
    
    def sync_world_info_to_old_format(self):
        #Since the old UI uses world info entries for folders, we need to make some up
        folder_entries = {}
        i=-1
        for folder in self.world_info_folder:
            folder_entries[folder] = i
            i-=1
    
    
        #self.worldinfo_i = []     # List of World Info key/value objects sans uninitialized entries
        self.story_settings.worldinfo_i = [{
                                            "comment": self.world_info[x]['comment'],
                                            "constant": self.world_info[x]['constant'],
                                            "content": self.world_info[x]['content'],
                                            "folder": folder_entries[self.world_info[x]['folder']],
                                            "init": True,
                                            "key": ",".join(self.world_info[x]['key']),
                                            "keysecondary": ",".join(self.world_info[x]['keysecondary']),
                                            "num": x,
                                            "selective": len(self.world_info[x]['keysecondary'])>0,
                                            "uid": self.world_info[x]['uid'] if 'v1_uid' not in self.world_info[x] or self.world_info[x]['v1_uid'] is None else self.world_info[x]['v1_uid']
                                        } for x in self.world_info]
                                        
        #self.worldinfo   = []     # List of World Info key/value objects
        self.story_settings.worldinfo = [x for x in self.story_settings.worldinfo_i]
        #We have to have an uninitialized blank entry for every folder or the old method craps out
        for folder in folder_entries:
            self.story_settings.worldinfo.append({
                                            "comment": "",
                                            "constant": False,
                                            "content": "",
                                            "folder": folder_entries[folder],
                                            "init": False,
                                            "key": "",
                                            "keysecondary": "",
                                            "num": (0 if len(self.world_info) == 0 else max(self.world_info))+(folder_entries[folder]*-1),
                                            "selective": False,
                                            "uid": folder_entries[folder]
                                        })
        
        #self.wifolders_d = {}     # Dictionary of World Info folder UID-info pairs
        self.story_settings.wifolders_d = {folder_entries[x]: {'collapsed': False, 'name': x} for x in folder_entries}
        
        #self.worldinfo_u = {}     # Dictionary of World Info UID - key/value pairs
        self.story_settings.worldinfo_u = {x['uid']: x for x in self.story_settings.worldinfo}
        
        #self.wifolders_l = []     # List of World Info folder UIDs
        self.story_settings.wifolders_l = [folder_entries[x] for x in folder_entries]
        
        #self.wifolders_u = {}     # Dictionary of pairs of folder UID - list of WI UID
        self.story_settings.wifolders_u = {folder_entries[x]: [y for y in self.story_settings.worldinfo if y['folder'] == x] for x in folder_entries}
        
    def reset_used_in_game(self):
        for key in self.world_info:
            if self.world_info[key]["used_in_game"] != self.world_info[key]["constant"]:
                self.world_info[key]["used_in_game"] = self.world_info[key]["constant"]
                if self.socketio is not None:
                    self.socketio.emit("world_info_entry_used_in_game", {"uid": key, "used_in_game": False}, broadcast=True, room="UI_2")
        
    def set_world_info_used(self, uid):
        if uid in self.world_info:
            self.world_info[uid]["used_in_game"] = True
        else:
            logger.warning("Something tried to set world info UID {} to in game, but it doesn't exist".format(uid))
        if self.socketio is not None:
            self.socketio.emit("world_info_entry_used_in_game", {"uid": uid, "used_in_game": True}, broadcast=True, room="UI_2")
    
    def get_used_wi(self):
        return [x['content'] for x in self.world_info if x['used_in_game']]
   

badwordsids_default = [[13460], [6880], [50256], [42496], [4613], [17414], [22039], [16410], [27], [29], [38430], [37922], [15913], [24618], [28725], [58], [47175], [36937], [26700], [12878], [16471], [37981], [5218], [29795], [13412], [45160], [3693], [49778], [4211], [20598], [36475], [33409], [44167], [32406], [29847], [29342], [42669], [685], [25787], [7359], [3784], [5320], [33994], [33490], [34516], [43734], [17635], [24293], [9959], [23785], [21737], [28401], [18161], [26358], [32509], [1279], [38155], [18189], [26894], [6927], [14610], [23834], [11037], [14631], [26933], [46904], [22330], [25915], [47934], [38214], [1875], [14692], [41832], [13163], [25970], [29565], [44926], [19841], [37250], [49029], [9609], [44438], [16791], [17816], [30109], [41888], [47527], [42924], [23984], [49074], [33717], [31161], [49082], [30138], [31175], [12240], [14804], [7131], [26076], [33250], [3556], [38381], [36338], [32756], [46581], [17912], [49146]] # Tokenized array of badwords used to prevent AI artifacting
badwordsids_neox = [[0], [1], [44162], [9502], [12520], [31841], [36320], [49824], [34417], [6038], [34494], [24815], [26635], [24345], [3455], [28905], [44270], [17278], [32666], [46880], [7086], [43189], [37322], [17778], [20879], [49821], [3138], [14490], [4681], [21391], [26786], [43134], [9336], [683], [48074], [41256], [19181], [29650], [28532], [36487], [45114], [46275], [16445], [15104], [11337], [1168], [5647], [29], [27482], [44965], [43782], [31011], [42944], [47389], [6334], [17548], [38329], [32044], [35487], [2239], [34761], [7444], [1084], [12399], [18990], [17636], [39083], [1184], [35830], [28365], [16731], [43467], [47744], [1138], [16079], [40116], [45564], [18297], [42368], [5456], [18022], [42696], [34476], [23505], [23741], [39334], [37944], [45382], [38709], [33440], [26077], [43600], [34418], [36033], [6660], [48167], [48471], [15775], [19884], [41533], [1008], [31053], [36692], [46576], [20095], [20629], [31759], [46410], [41000], [13488], [30952], [39258], [16160], [27655], [22367], [42767], [43736], [49694], [13811], [12004], [46768], [6257], [37471], [5264], [44153], [33805], [20977], [21083], [25416], [14277], [31096], [42041], [18331], [33376], [22372], [46294], [28379], [38475], [1656], [5204], [27075], [50001], [16616], [11396], [7748], [48744], [35402], [28120], [41512], [4207], [43144], [14767], [15640], [16595], [41305], [44479], [38958], [18474], [22734], [30522], [46267], [60], [13976], [31830], [48701], [39822], [9014], [21966], [31422], [28052], [34607], [2479], [3851], [32214], [44082], [45507], [3001], [34368], [34758], [13380], [38363], [4299], [46802], [30996], [12630], [49236], [7082], [8795], [5218], [44740], [9686], [9983], [45301], [27114], [40125], [1570], [26997], [544], [5290], [49193], [23781], [14193], [40000], [2947], [43781], [9102], [48064], [42274], [18772], [49384], [9884], [45635], [43521], [31258], [32056], [47686], [21760], [13143], [10148], [26119], [44308], [31379], [36399], [23983], [46694], [36134], [8562], [12977], [35117], [28591], [49021], [47093], [28653], [29013], [46468], [8605], [7254], [25896], [5032], [8168], [36893], [38270], [20499], [27501], [34419], [29547], [28571], [36586], [20871], [30537], [26842], [21375], [31148], [27618], [33094], [3291], [31789], [28391], [870], [9793], [41361], [47916], [27468], [43856], [8850], [35237], [15707], [47552], [2730], [41449], [45488], [3073], [49806], [21938], [24430], [22747], [20924], [46145], [20481], [20197], [8239], [28231], [17987], [42804], [47269], [29972], [49884], [21382], [46295], [36676], [34616], [3921], [26991], [27720], [46265], [654], [9855], [40354], [5291], [34904], [44342], [2470], [14598], [880], [19282], [2498], [24237], [21431], [16369], [8994], [44524], [45662], [13663], [37077], [1447], [37786], [30863], [42854], [1019], [20322], [4398], [12159], [44072], [48664], [31547], [18736], [9259], [31], [16354], [21810], [4357], [37982], [5064], [2033], [32871], [47446], [62], [22158], [37387], [8743], [47007], [17981], [11049], [4622], [37916], [36786], [35138], [29925], [14157], [18095], [27829], [1181], [22226], [5709], [4725], [30189], [37014], [1254], [11380], [42989], [696], [24576], [39487], [30119], [1092], [8088], [2194], [9899], [14412], [21828], [3725], [13544], [5180], [44679], [34398], [3891], [28739], [14219], [37594], [49550], [11326], [6904], [17266], [5749], [10174], [23405], [9955], [38271], [41018], [13011], [48392], [36784], [24254], [21687], [23734], [5413], [41447], [45472], [10122], [17555], [15830], [47384], [12084], [31350], [47940], [11661], [27988], [45443], [905], [49651], [16614], [34993], [6781], [30803], [35869], [8001], [41604], [28118], [46462], [46762], [16262], [17281], [5774], [10943], [5013], [18257], [6750], [4713], [3951], [11899], [38791], [16943], [37596], [9318], [18413], [40473], [13208], [16375]]
badwordsids_opt = [[44717], [46613], [48513], [49923], [50185], [48755], [8488], [43303], [49659], [48601], [49817], [45405], [48742], [49925], [47720], [11227], [48937], [48784], [50017], [42248], [49310], [48082], [49895], [50025], [49092], [49007], [8061], [44226], [0], [742], [28578], [15698], [49784], [46679], [39365], [49281], [49609], [48081], [48906], [46161], [48554], [49670], [48677], [49721], [49632], [48610], [48462], [47457], [10975], [46077], [28696], [48709], [43839], [49798], [49154], [48203], [49625], [48395], [50155], [47161], [49095], [48833], [49420], [49666], [48443], [22176], [49242], [48651], [49138], [49750], [40389], [48021], [21838], [49070], [45333], [40862], [1], [49915], [33525], [49858], [50254], [44403], [48992], [48872], [46117], [49853], [47567], [50206], [41552], [50068], [48999], [49703], [49940], [49329], [47620], [49868], [49962], [2], [44082], [50236], [31274], [50260], [47052], [42645], [49177], [17523], [48691], [49900], [49069], [49358], [48794], [47529], [46479], [48457], [646], [49910], [48077], [48935], [46386], [48902], [49151], [48759], [49803], [45587], [48392], [47789], [48654], [49836], [49230], [48188], [50264], [46844], [44690], [48505], [50161], [27779], [49995], [41833], [50154], [49097], [48520], [50018], [8174], [50084], [49366], [49526], [50193], [7479], [49982], [3]]
genres = ['Absurdist', 'Action & Adventure', 'Adaptations & Pastiche', 'African American & Black/General', 'African American & Black/Christian', 'African American & Black/Erotica', 'African American & Black/Historical', 'African American & Black/Mystery & Detective', 'African American & Black/Urban & Street Lit', 'African American & Black/Women', 'Alternative History', 'Amish & Mennonite', 'Animals', 'Anthologies (multiple authors)', 'Asian American', 'Biographical', 'Buddhist', 'Christian/General', 'Christian/Biblical', 'Christian/Classic & Allegory', 'Christian/Collections & Anthologies', 'Christian/Contemporary', 'Christian/Fantasy', 'Christian/Futuristic', 'Christian/Historical', 'Christian/Romance/General', 'Christian/Romance/Historical', 'Christian/Romance/Suspense', 'Christian/Suspense', 'Christian/Western', 'City Life', 'Classics', 'Coming of Age', 'Crime', 'Cultural Heritage', 'Disabilities & Special Needs', 'Disaster', 'Dystopian', 'Epistolary', 'Erotica/General', 'Erotica/BDSM', 'Erotica/Collections & Anthologies', 'Erotica/Historical', 'Erotica/LGBTQ+/General', 'Erotica/LGBTQ+/Bisexual', 'Erotica/LGBTQ+/Gay', 'Erotica/LGBTQ+/Lesbian', 'Erotica/LGBTQ+/Transgender', 'Erotica/Science Fiction, Fantasy & Horror', 'Fairy Tales, Folk Tales, Legends & Mythology', 'Family Life/General', 'Family Life/Marriage & Divorce', 'Family Life/Siblings', 'Fantasy/General', 'Fantasy/Action & Adventure', 'Fantasy/Arthurian', 'Fantasy/Collections & Anthologies', 'Fantasy/Contemporary', 'Fantasy/Dark Fantasy', 'Fantasy/Dragons & Mythical Creatures', 'Fantasy/Epic', 'Fantasy/Gaslamp', 'Fantasy/Historical', 'Fantasy/Humorous', 'Fantasy/Military', 'Fantasy/Paranormal', 'Fantasy/Romance', 'Fantasy/Urban', 'Feminist', 'Friendship', 'Ghost', 'Gothic', 'Hispanic & Latino', 'Historical/General', 'Historical/Ancient', 'Historical/Civil War Era', 'Historical/Colonial America & Revolution', 'Historical/Medieval', 'Historical/Renaissance', 'Historical/World War I', 'Historical/World War II', 'Holidays', 'Horror', 'Humorous/General', 'Humorous/Black Humor', 'Indigenous', 'Jewish', 'Legal', 'LGBTQ+/General', 'LGBTQ+/Bisexual', 'LGBTQ+/Gay', 'LGBTQ+/Lesbian', 'LGBTQ+/Transgender', 'Literary', 'LitRPG (Literary Role-Playing Game) *', 'Magical Realism', 'Mashups', 'Media Tie-In', 'Medical', 'Multiple Timelines', 'Muslim', 'Mystery & Detective/General', 'Mystery & Detective/Amateur Sleuth', 'Mystery & Detective/Collections & Anthologies', 'Mystery & Detective/Cozy/General', 'Mystery & Detective/Cozy/Animals', 'Mystery & Detective/Cozy/Crafts', 'Mystery & Detective/Cozy/Culinary', 'Mystery & Detective/Cozy/Holidays & Vacation *', 'Mystery & Detective/Cozy/Paranormal *', 'Mystery & Detective/Hard-Boiled', 'Mystery & Detective/Historical', 'Mystery & Detective/International Crime & Mystery', 'Mystery & Detective/Jewish *', 'Mystery & Detective/Police Procedural', 'Mystery & Detective/Private Investigators', 'Mystery & Detective/Traditional', 'Mystery & Detective/Women Sleuths', 'Nature & the Environment', 'Noir', 'Occult & Supernatural', 'Own Voices', 'Political', 'Psychological', 'Religious', 'Romance/General', 'Romance/Action & Adventure', 'Romance/African American & Black', 'Romance/Billionaires', 'Romance/Clean & Wholesome', 'Romance/Collections & Anthologies', 'Romance/Contemporary', 'Romance/Erotic', 'Romance/Fantasy', 'Romance/Firefighters', 'Romance/Historical/General', 'Romance/Historical/American', 'Romance/Historical/Ancient World', 'Romance/Historical/Gilded Age', 'Romance/Historical/Medieval', 'Romance/Historical/Regency', 'Romance/Historical/Renaissance', 'Romance/Historical/Scottish', 'Romance/Historical/Tudor', 'Romance/Historical/20th Century', 'Romance/Historical/Victorian', 'Romance/Historical/Viking', 'Romance/Holiday', 'Romance/Later in Life', 'Romance/LGBTQ+/General', 'Romance/LGBTQ+/Bisexual', 'Romance/LGBTQ+/Gay', 'Romance/LGBTQ+/Lesbian', 'Romance/LGBTQ+/Transgender', 'Romance/Medical', 'Romance/Military', 'Romance/Multicultural & Interracial', 'Romance/New Adult', 'Romance/Paranormal/General', 'Romance/Paranormal/Shifters', 'Romance/Paranormal/Vampires', 'Romance/Paranormal/Witches', 'Romance/Police & Law Enforcement', 'Romance/Polyamory', 'Romance/Rock Stars', 'Romance/Romantic Comedy', 'Romance/Royalty', 'Romance/Science Fiction', 'Romance/Sports', 'Romance/Suspense', 'Romance/Time Travel', 'Romance/Western', 'Romance/Workplace', 'Sagas', 'Satire', 'Science Fiction/General', 'Science Fiction/Action & Adventure', 'Science Fiction/Alien Contact', 'Science Fiction/Apocalyptic & Post-Apocalyptic', 'Science Fiction/Collections & Anthologies', 'Science Fiction/Crime & Mystery', 'Science Fiction/Cyberpunk', 'Science Fiction/Genetic Engineering', 'Science Fiction/Hard Science Fiction', 'Science Fiction/Humorous', 'Science Fiction/Military', 'Science Fiction/Space Exploration', 'Science Fiction/Space Opera', 'Science Fiction/Steampunk', 'Science Fiction/Time Travel', 'Sea Stories', 'Short Stories (single author)', 'Small Town & Rural', 'Southern', 'Sports', 'Superheroes', 'Thrillers/General', 'Thrillers/Crime', 'Thrillers/Domestic', 'Thrillers/Espionage', 'Thrillers/Historical', 'Thrillers/Legal', 'Thrillers/Medical', 'Thrillers/Military', 'Thrillers/Political', 'Thrillers/Psychological', 'Thrillers/Supernatural', 'Thrillers/Suspense', 'Thrillers/Technological', 'Thrillers/Terrorism', 'Urban & Street Lit', 'Visionary & Metaphysical', 'War & Military', 'Westerns', 'Women', 'World Literature/Africa/General', 'World Literature/Africa/East Africa', 'World Literature/Africa/Nigeria', 'World Literature/Africa/Southern Africa', 'World Literature/Africa/West Africa', 'World Literature/American/General', 'World Literature/American/Colonial & Revolutionary Periods', 'World Literature/American/19th Century', 'World Literature/American/20th Century', 'World Literature/American/21st Century', 'World Literature/Argentina', 'World Literature/Asia (General)', 'World Literature/Australia', 'World Literature/Austria', 'World Literature/Brazil', 'World Literature/Canada/General', 'World Literature/Canada/Colonial & 19th Century', 'World Literature/Canada/20th Century', 'World Literature/Canada/21st Century', 'World Literature/Caribbean & West Indies', 'World Literature/Central America', 'World Literature/Chile', 'World Literature/China/General', 'World Literature/China/19th Century', 'World Literature/China/20th Century', 'World Literature/China/21st Century', 'World Literature/Colombia', 'World Literature/Czech Republic', 'World Literature/Denmark', 'World Literature/England/General', 'World Literature/England/Early & Medieval Periods', 'World Literature/England/16th & 17th Century', 'World Literature/England/18th Century', 'World Literature/England/19th Century', 'World Literature/England/20th Century', 'World Literature/England/21st Century', 'World Literature/Europe (General)', 'World Literature/Finland', 'World Literature/France/General', 'World Literature/France/18th Century', 'World Literature/France/19th Century', 'World Literature/France/20th Century', 'World Literature/France/21st Century', 'World Literature/Germany/General', 'World Literature/Germany/20th Century', 'World Literature/Germany/21st Century', 'World Literature/Greece', 'World Literature/Hungary', 'World Literature/India/General', 'World Literature/India/19th Century', 'World Literature/India/20th Century', 'World Literature/India/21st Century', 'World Literature/Ireland/General', 'World Literature/Ireland/19th Century', 'World Literature/Ireland/20th Century', 'World Literature/Ireland/21st Century', 'World Literature/Italy', 'World Literature/Japan', 'World Literature/Korea', 'World Literature/Mexico', 'World Literature/Middle East/General', 'World Literature/Middle East/Arabian Peninsula', 'World Literature/Middle East/Egypt & North Africa', 'World Literature/Middle East/Israel', 'World Literature/Netherlands', 'World Literature/New Zealand', 'World Literature/Norway', 'World Literature/Oceania', 'World Literature/Pakistan', 'World Literature/Peru', 'World Literature/Poland', 'World Literature/Portugal', 'World Literature/Russia/General', 'World Literature/Russia/19th Century', 'World Literature/Russia/20th Century', 'World Literature/Russia/21st Century', 'World Literature/Scotland/General', 'World Literature/Scotland/19th Century', 'World Literature/Scotland/20th Century', 'World Literature/Scotland/21st Century', 'World Literature/South America (General)', 'World Literature/Southeast Asia', 'World Literature/Spain/General', 'World Literature/Spain/19th Century', 'World Literature/Spain/20th Century', 'World Literature/Spain/21st Century', 'World Literature/Sweden', 'World Literature/Turkey', 'World Literature/Uruguay', 'World Literature/Wales']
