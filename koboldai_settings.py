from __future__ import annotations
from dataclasses import dataclass
import difflib
import importlib
import os, re, time, threading, json, pickle, base64, copy, tqdm, datetime, sys
import shutil
from typing import List, Union
from io import BytesIO
from flask import has_request_context, session, request
from flask_socketio import join_room, leave_room
from collections import OrderedDict
import multiprocessing
from logger import logger
import torch
import numpy as np
import random
import inspect

serverstarted = False
queue = None
multi_story = False
global enable_whitelist
enable_whitelist = False

if importlib.util.find_spec("tortoise") is not None:
    from tortoise import api
    from tortoise.utils.audio import load_voices
    
password_vars = ["horde_api_key", "privacy_password", "img_gen_api_password"]

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
    global multi_story, koboldai_vars_main
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
                    if not koboldai_vars_main.host or name not in password_vars:
                        data = ["var_changed", {"classname": classname, "name": name, "old_value": clean_var_for_emit(old_value), "value": clean_var_for_emit(value), "transmit_time": transmit_time}, {"include_self":True, "broadcast":True, "room":room}]
                    else:
                        data = ["var_changed", {"classname": classname, "name": name, "old_value": "*" * len(old_value) if old_value is not None else "", "value": "*" * len(value) if value is not None else "", "transmit_time": transmit_time}, {"include_self":True, "broadcast":True, "room":room}]
                    if queue is not None:
                        #logger.debug("Had to use queue")
                        queue.put(data)
                        
                else:
                    if socketio is not None:
                        if not koboldai_vars_main.host or name not in password_vars:
                            socketio.emit("var_changed", {"classname": classname, "name": name, "old_value": clean_var_for_emit(old_value), "value": clean_var_for_emit(value), "transmit_time": transmit_time}, include_self=True, broadcast=True, room=room)
                        else:
                            socketio.emit("var_changed", {"classname": classname, "name": name, "old_value":  "*" * len(old_value) if old_value is not None else "", "value": "*" * len(value) if value is not None else "", "transmit_time": transmit_time}, include_self=True, broadcast=True, room=room)

class koboldai_vars(object):
    def __init__(self, socketio):
        self._model_settings = model_settings(socketio, self)
        self._user_settings = user_settings(socketio)
        self._system_settings = system_settings(socketio, self)
        self._story_settings = {'default': story_settings(socketio, self)}
        self._undefined_settings = undefined_settings()
        self._socketio = socketio
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

        # Leave the old room and join the new one if in socket context
        if hasattr(request, "sid"):
            logger.debug("Leaving room {}".format(session['story']))
            leave_room(session['story'])
            logger.debug("Joining room {}".format(story_name))
            join_room(story_name)

        session['story'] = story_name
        logger.debug("Sending story reset")
        self._story_settings[story_name]._socketio.emit("reset_story", {}, broadcast=True, room=story_name)
        if story_name in self._story_settings:
            self._story_settings[story_name].no_save = True
            self._story_settings[story_name].worldinfo_v2.reset()
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
        self._story_settings[story_name] = story_settings(self._socketio, self)
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
            self._story_settings[story_name]._socketio.emit("reset_story", {}, broadcast=True, room=story_name)
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
    
    def is_chat_v2(self):
        return self.chat_style > 0 and self.chatmode
    
    def get_token_representation(self, text: Union[str, list, None]) -> list:
        if not self.tokenizer or not text:
            return []
        
        if isinstance(text, str):
            encoded = self.tokenizer.encode(text)
        else:
            encoded = text

        # TODO: This might be ineffecient, should we cache some of this?
        return [[token, self.tokenizer.decode(token)] for token in encoded]
    
    def calc_ai_text(self, submitted_text=None, return_text=False, send_context=True, allowed_wi_entries=None, allowed_wi_folders=None):
        """Compute the context that would be submitted to the AI.

        submitted_text: Optional override to the player-submitted text.
        return_text: If True, return the context as a string.  Otherwise, return a tuple consisting of:
            (tokens, used_tokens, used_tokens+self.genamt, set(used_world_info))
        send_context: If True, the context is prepared for submission to the AI (by marking used world info and setting self.context
        allowed_wi_entries: If not None, only world info entries with uids in the given set are allowed to be used.
        allowed_wi_folders: If not None, only world info folders with uids in the given set are allowed to be used.
        """
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
        
        if send_context:
            self.worldinfo_v2.reset_used_in_game()
        
        def allow_wi(wi):
            """Return True if adding this world info entry is allowed."""
            wi_uid = wi['uid']
            if wi_uid in used_world_info:
                return False
            if wi["type"] == "commentator" and not (allowed_wi_entries and wi_uid in allowed_wi_entries):
                return False
            if allowed_wi_entries is not None and wi_uid not in allowed_wi_entries:
                return False
            if allowed_wi_folders is not None and wi['folder'] not in allowed_wi_folders:
                return False
            return True

        # Add Genres #
        if self.genres:
            # Erebus, Nerys, Janeway, Picard (probably)
            genre_template = "[Genre: %s]"
            model_name = self.model.lower()
            if "skein" in model_name or "adventure" in model_name:
                genre_template = "[Themes: %s]"
            elif "shinen" in model_name:
                genre_template = "[Theme: %s]"

            genre_text = genre_template % (", ".join(self.genres))
            genre_tokens = self.tokenizer.encode(genre_text)
            genre_data = [[x, self.tokenizer.decode(x)] for x in genre_tokens]

            context.append({
                "type": "genre",
                "text": genre_text,
                "tokens": genre_data,
            })
            used_tokens += len(genre_tokens)


        ######################################### Add memory ########################################################
        max_memory_length = int(token_budget * self.max_memory_fraction)
        memory_text = self.memory
        if memory_text != "":
            if memory_text[-1] not in [" ", '\n']:
                memory_text += " "
            memory_tokens = self.tokenizer.encode(memory_text)
        else:
            memory_tokens = []
        if len(memory_tokens) > max_memory_length:
            memory_tokens = memory_tokens[:max_memory_length]
            memory_length = max_memory_length
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
            if wi['constant'] and allow_wi(wi):
                wi_length = len(self.tokenizer.encode(wi['content']))
                if used_tokens + wi_length <= token_budget:
                    used_tokens+=wi_length
                    used_world_info.append(wi['uid'])
                    if send_context:
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

        # Always add newlines on chat v2
        if action_text_split and self.is_chat_v2():
            action_text_split[-1][0] = action_text_split[-1][0].strip() + "\n"
        
        ######################################### Prompt ########################################################
        #Add prompt length/text if we're set to always use prompt
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
                    if allow_wi(wi):
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
                                if send_context:
                                    self.worldinfo_v2.set_world_info_used(wi['uid'])
                   
            else:
                self.prompt_in_ai = False
        
        
        ######################################### Setup Author's Note Data ########################################################
        authors_note_text = self.authornotetemplate.replace("<|>", self.authornote)
        if len(authors_note_text) > 0 and authors_note_text[0] not in [" ", "\n"]:
            authors_note_text = " " + authors_note_text
        authors_note_data = [[x, self.tokenizer.decode(x)] for x in self.tokenizer.encode(authors_note_text)]
        if used_tokens + len(authors_note_data) <= token_budget:
            used_tokens += len(authors_note_data)
        
        
        ######################################### Actions ########################################################
        #Start going through the actions backwards, adding it to the text if it fits and look for world info entries
        used_all_tokens = False
        actions_seen = [] #Used to track how many actions we've seen so we can insert author's note in the appropriate place as well as WI depth stop
        inserted_author_note = False

        sentences_seen = 0
        for i in range(len(action_text_split)-1, -1, -1):
            if action_text_split[i][3]:
                #We've hit an item we've already included. Stop
                break
            sentences_seen += 1

            #Add our author's note if we've hit andepth
            if not inserted_author_note and len(actions_seen) >= self.andepth and sentences_seen > self.andepth and self.authornote != "":
                game_context.insert(0, {"type": "authors_note", "text": authors_note_text, "tokens": authors_note_data, "attention_multiplier": self.an_attn_bias})
                inserted_author_note = True

            # Add to actions_seen after potentially inserting the author note, since we want to insert the author note
            # after the sentence that pushes our action count above the author note depth.
            for action in action_text_split[i][1]:
                if action not in actions_seen:
                    actions_seen.append(action)

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
                wi_search = action_text_split[i][0]

                #Now we need to check for used world info entries
                for wi in self.worldinfo_v2:
                    if allow_wi(wi):
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
                            if len(actions_seen) > self.widepth and sentences_seen > self.widepth:
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
                                if send_context:
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
                if item['type'] != 'soft_prompt':
                    tokens.extend([x[0] for x in item['tokens']])
        
        if send_context:
            self.context = context

        #logger.debug("Calc_AI_text: {}s".format(time.time()-start_time))
        logger.debug("Token Budget: {}. Used Tokens: {}".format(token_budget, used_tokens))
        if return_text:
            return "".join([x['text'] for x in context])
        return tokens, used_tokens, used_tokens+self.genamt, set(used_world_info)
    
    def is_model_torch(self) -> bool:
        if self.use_colab_tpu:
            return False

        if self.model in ["Colab", "API", "CLUSTER", "ReadOnly", "OAI"]:
            return False

        return True
    
    def assign_world_info_to_actions(self, *args, **kwargs):
        self._story_settings[self.get_story_name()].assign_world_info_to_actions(*args, **kwargs)
    
    def reset_for_model_load(self):
        self._model_settings.reset_for_model_load()
    
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
            elif name in self._story_settings[self.get_story_name()].__dict__:
                setattr(self._story_settings[self.get_story_name()], name, value)
            else:
                setattr(self._undefined_settings, name, value)

    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        elif name in self._model_settings.__dict__ or hasattr(self._model_settings, name):
            return getattr(self._model_settings, name)
        elif name in self._user_settings.__dict__ or hasattr(self._user_settings, name):
            return getattr(self._user_settings, name)
        elif name in self._system_settings.__dict__ or hasattr(self._system_settings, name):
            return getattr(self._system_settings, name)
        elif name in self._story_settings[self.get_story_name()].__dict__ or hasattr(self._story_settings[self.get_story_name()], name):
            return getattr(self._story_settings[self.get_story_name()], name)
        else:
            return getattr(self._undefined_settings, name)


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
                elif key == 'autosave':
                    autosave = value
                elif key in ['worldinfo_u', 'wifolders_d']:
                    # Fix UID keys to be ints
                    value = {int(k): v for k, v in value.items()}

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
            process_variable_changes(self._socketio, "story", 'prompt_wi_highlighted_text', self.prompt_wi_highlighted_text, None)
        
        if 'no_save' in self.__dict__:
            setattr(self, 'no_save', False)
            
        
    def send_to_ui(self):
        for (name, value) in vars(self).items():
            if name not in self.local_only_variables and name[0] != "_":
                try:
                    process_variable_changes(self._socketio, self.__class__.__name__.replace("_settings", ""), name, value, None)
                except:
                    print("{} is of type {} and I can't transmit".format(name, type(value)))
                    raise

class model_settings(settings):
    local_only_variables = ['apikey', 'default_preset']
    no_save_variables = ['modelconfig', 'custmodpth', 'generated_tkns', 
                         'loaded_layers', 'total_layers', 'total_download_chunks', 'downloaded_chunks', 'presets', 'default_preset', 
                         'welcome', 'welcome_default', 'simple_randomness', 'simple_creativity', 'simple_repitition',
                         'badwordsids', 'uid_presets', 'model', 'model_type', 'lazy_load', 'fp32_model', 'modeldim', 'horde_wait_time', 'horde_queue_position', 'horde_queue_size', 'newlinemode', 'tqdm_progress', 'tqdm_rem_time', '_tqdm']
    settings_name = "model"
    default_settings = {"rep_pen" : 1.1, "rep_pen_slope": 1.0, "rep_pen_range": 2048, "temp": 0.5, "top_p": 0.9, "top_k": 0, "top_a": 0.0, "tfs": 1.0, "typical": 1.0,
                        "sampler_order": [6,0,1,2,3,4,5]}
    def __init__(self, socketio, koboldai_vars):
        self.enable_whitelist = False
        self._socketio = socketio
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
        self._tqdm        = tqdm.tqdm(total=self.genamt, file=self.ignore_tqdm())    # tqdm agent for generating tokens. This will allow us to calculate the remaining time
        self.tqdm_progress = 0     # TQDP progress
        self.tqdm_rem_time = 0     # tqdm calculated reemaining time
        self.configname = None
        self.online_model = ''
        self.welcome_default = """<style>#welcome_container { display: block; } #welcome_text { display: flex; height: 100%; } .welcome_text { align-self: flex-end; }</style>
        <div id='welcome-logo-container'><img id='welcome-logo' src='static/Welcome_Logo.png' draggable='False'></div>
        <div class='welcome_text'>
            <div id="welcome-text-content">Please load a model from the left.<br/>
                If you encounter any issues, please click the Download debug dump link in the Home tab on the left flyout and attach the downloaded file to your error report on <a href='https://github.com/ebolam/KoboldAI/issues'>Github</a>, <a href='https://www.reddit.com/r/KoboldAI/'>Reddit</a>, or <a href='https://koboldai.org/discord'>Discord</a>.
                A redacted version (without story text) is available.
            </div>
        </div>""" # Custom Welcome Text
        self.welcome     = self.welcome_default
        self._koboldai_vars = koboldai_vars
        self.alt_multi_gen = False
        self.bit_8_available = None
        self.use_default_badwordsids = True
        self.supported_gen_modes = []
        
    def reset_for_model_load(self):
        self.simple_randomness = 0 #Set first as this affects other outputs
        self.simple_creativity = 0 #Set first as this affects other outputs
        self.simple_repitition = 0 #Set first as this affects other outputs
        self.max_length  = 2048    # Maximum number of tokens to submit per action
        self.ikmax       = 3000    # Maximum number of characters to submit to InferKit
        self.genamt      = 200      # Amount of text for each action to generate
        self.ikgen       = 200     # Number of characters for InferKit to generate
        self.rep_pen     = 1.1     # Default generator repetition_penalty
        self.rep_pen_slope = 1.0   # Default generator repetition penalty slope
        self.rep_pen_range = 2048  # Default generator repetition penalty range
        self.temp        = 0.5     # Default generator temperature
        self.top_p       = 0.9     # Default generator top_p
        self.top_k       = 0       # Default generator top_k
        self.top_a       = 0.0     # Default generator top-a
        self.tfs         = 1.0     # Default generator tfs (tail-free sampling)
        self.typical     = 1.0     # Default generator typical sampling threshold
        self.numseqs     = 1       # Number of sequences to ask the generator to create
        self.generated_tkns = 0    # If using a backend that supports Lua generation modifiers, how many tokens have already been generated, otherwise 0
        self.badwordsids = []
        self.fp32_model  = False  # Whether or not the most recently loaded HF model was in fp32 format
        self.modeldim    = -1     # Embedding dimension of your model (e.g. it's 4096 for GPT-J-6B and 2560 for GPT-Neo-2.7B)
        self.sampler_order = [6, 0, 1, 2, 3, 4, 5]
        self.newlinemode = "n"
        self.presets     = []   # Holder for presets
        self.selected_preset = ""
        self.uid_presets = []
        self.default_preset = {}
        self.horde_wait_time = 0
        self.horde_queue_position = 0
        self.horde_queue_size = 0
        self.use_alt_rep_pen = False
        
        


        
    #dummy class to eat the tqdm output
    class ignore_tqdm(object):
        def write(self, bar):
            pass
        
    def __setattr__(self, name, value):
        new_variable = name not in self.__dict__
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        #Put variable change actions here
                
        if name in ['simple_randomness', 'simple_creativity', 'simple_repitition'] and not new_variable:
            #We need to disable all of the samplers
            self.top_k = 0
            self.top_a = 0.0
            self.tfs = 1.0
            self.typical = 1.0
            self.rep_pen_range = 1024
            self.rep_pen_slope = 0.7
            
            #Now we setup our other settings
            if self.simple_randomness < 0:
                self.temp = default_rand_range[1]+(default_rand_range[1]-default_rand_range[0])/100*self.simple_randomness
            else:
                self.temp = default_rand_range[1]+(default_rand_range[2]-default_rand_range[1])/100*self.simple_randomness
        
            self.top_p = (default_creativity_range[1]-default_creativity_range[0])/200*self.simple_creativity+sum(default_creativity_range)/2
            self.rep_pen = (default_rep_range[1]-default_rep_range[0])/200*self.simple_repitition+sum(default_rep_range)/2
        
        if not new_variable and (name == 'max_length' or name == 'genamt'):
            ignore = self._koboldai_vars.calc_ai_text()
            
        #set preset values
        if name == 'selected_preset' and value != "":
            logger.info("Changing preset to {}".format(value))
            if value in self.uid_presets:
                for default_key, default_value in self.default_settings.items():
                    setattr(self, default_key, default_value)
                for preset_key, preset_value in self.uid_presets[value].items():
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
        elif name == "generated_tkns" and '_tqdm' in self.__dict__:
            if value == 0:
                self._tqdm.reset(total=self.genamt * (self.numseqs if self.alt_multi_gen else 1) )
                self.tqdm_progress = 0
            else:
                self._tqdm.update(value-self._tqdm.n)
                self.tqdm_progress = int(float(self.generated_tkns)/float(self.genamt * (self.numseqs if self.alt_multi_gen else 1))*100)
                if self._tqdm.format_dict['rate'] is not None:
                    self.tqdm_rem_time = str(datetime.timedelta(seconds=int(float((self.genamt * (self.numseqs if self.alt_multi_gen else 1))-self.generated_tkns)/self._tqdm.format_dict['rate'])))
        #Setup TQDP for model loading
        elif name == "loaded_layers" and '_tqdm' in self.__dict__:
            if value == 0:
                self._tqdm.reset(total=self.total_layers)
                self.tqdm_progress = 0
            else:
                self._tqdm.update(1)
                self.tqdm_progress = int(float(self.loaded_layers)/float(self.total_layers)*100)
                if self._tqdm.format_dict['rate'] is not None:
                    self.tqdm_rem_time = str(datetime.timedelta(seconds=int(float(self.total_layers-self.loaded_layers)/self._tqdm.format_dict['rate'])))  
        #Setup TQDP for model downloading
        elif name == "total_download_chunks" and '_tqdm' in self.__dict__:
            self._tqdm.reset(total=value)
            self.tqdm_progress = 0
        elif name == "downloaded_chunks" and '_tqdm' in self.__dict__:
            if value == 0:
                self._tqdm.reset(total=self.total_download_chunks)
                self.tqdm_progress = 0
            else:
                self._tqdm.update(value-old_value)
                if self.total_download_chunks is not None:
                    if self.total_download_chunks==0:
                        self.tqdm_progress = 0
                    elif float(self.downloaded_chunks) > float(self.total_download_chunks):
                        self.tqdm_progress = 100
                    else: 
                        self.tqdm_progress = round(float(self.downloaded_chunks)/float(self.total_download_chunks)*100, 1)
                else:
                    self.tqdm_progress = 0
                if self._tqdm.format_dict['rate'] is not None:
                    self.tqdm_rem_time = str(datetime.timedelta(seconds=int(float(self.total_download_chunks-self.downloaded_chunks)/self._tqdm.format_dict['rate'])))  
        
        
        
        if name not in self.local_only_variables and name[0] != "_" and not new_variable:
            process_variable_changes(self._socketio, self.__class__.__name__.replace("_settings", ""), name, value, old_value)
            
class story_settings(settings):
    local_only_variables = ['tokenizer', 'no_save', 'revisions', 'prompt', 'save_paths']
    no_save_variables = ['tokenizer', 'context', 'no_save', 'prompt_in_ai', 'authornote_length', 'prompt_length', 'memory_length', 'save_paths']
    settings_name = "story"
    def __init__(self, socketio, koboldai_vars, tokenizer=None):
        self._socketio = socketio
        self.tokenizer = tokenizer
        self._koboldai_vars = koboldai_vars
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
        self.botname    = "Bot"
        self.stop_sequence = []     #use for configuring stop sequences
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
        self.max_memory_fraction = 0.5  # Tokens from memory are allowed to use this much of the token budget
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
        self.gen_audio = False
        self.prompt_picture_filename = ""
        self.prompt_picture_prompt = ""

        # It's important to only use "=" syntax on this to ensure syncing; no
        # .append() or the like
        self.genres = []
        
        # bias experiment
        self.memory_attn_bias = 1
        self.an_attn_bias = 1
        self.chat_style = 0

        # In percent!!!
        self.commentary_chance = 0
        self.commentary_enabled = False
        
        self.save_paths = SavePaths(os.path.join("stories", self.story_name or "Untitled"))

        ################### must be at bottom #########################
        self.no_save = False  #Temporary disable save (doesn't save with the file)
    
    def save_story(self) -> None:
        if self.no_save:
            return
        
        if not any([self.prompt, self.memory, self.authornote, len(self.actions), len(self.worldinfo_v2)]):
            return

        logger.info("Saving")

        save_name = self.story_name or "Untitled"

        # Disambiguate stories by adding (n) if needed
        disambiguator = 0
        self.save_paths.base = os.path.join("stories", save_name)
        while os.path.exists(self.save_paths.base):
            try:
                # If the stories share a story id, overwrite the existing one.
                with open(self.save_paths.story, "r", encoding="utf-8") as file:
                    j = json.load(file)
                    if self.story_id == j["story_id"]:
                        break
            except FileNotFoundError:
                logger.error(f"Malformed save file: Missing story.json in {self.save_paths.base}. Populating it with new data.")
                break

            disambiguator += 1
            self.save_paths.base = os.path.join("stories", save_name + (f" ({disambiguator})" if disambiguator else ""))
        
        # Setup the directory structure.
        for path in self.save_paths.required_paths:
            try:
                os.mkdir(path)
            except FileExistsError:
                pass

        # Convert v2 if applicable
        v2_path = os.path.join("stories", f"{self.story_name}_v2.json")
        if os.path.exists(v2_path):
            logger.info("Migrating v2 save")
            with open(v2_path, "r", encoding="utf-8") as file:
                v2j = json.load(file)
            
            if v2j["story_id"] == self.story_id:
                shutil.move(v2_path, os.path.join(self.save_paths.base, ".v2_old.json"))
            else:
                logger.warning(f"Story mismatch in v2 migration. Existing file had story id {v2j['story_id']} but we have {self.story_id}")

        self.gamesaved = True
        with open(self.save_paths.story, "w", encoding="utf-8") as file:
            file.write(self.to_json())
    
    def update_story_path_structure(self, path: str) -> None:
        # Upon loading a file, makes directories that are required for certain
        # functionality.
        sp = SavePaths(path)

        for path in sp.required_paths:
            try:
                os.mkdir(path)
            except FileExistsError:
                pass

    def save_revision(self):
        game = json.loads(self.to_json())
        del game['revisions']
        self.revisions.append(game)
        self.gamesaved = False
    
    def reset(self):
        self.no_save = True
        self._socketio.emit("reset_story", {}, broadcast=True, room="UI_2")
        self.__init__(self._socketio, self._koboldai_vars, tokenizer=self.tokenizer)
        self.no_save = False
      
    def sync_worldinfo_v1_to_v2(self):
        new_world_info = KoboldWorldInfo(None, self, self._koboldai_vars, tokenizer=self.tokenizer)
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
        
        new_world_info._socketio = self._socketio
        self.worldinfo_v2 = new_world_info
        
    def assign_world_info_to_actions(self, action_id=None, wuid=None, no_transmit=False):
        logger.debug("Calcing WI Assignment for action_id: {} wuid: {}".format(action_id, wuid))
        if action_id != -1 and (action_id is None or action_id not in self.actions.actions):
            actions_to_check = self.actions.actions
        elif action_id == -1:
            actions_to_check = {}
        else:
            actions_to_check = {action_id: self.actions.actions[action_id]}
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
            process_variable_changes(self._socketio, self.__class__.__name__.replace("_settings", ""), name, value, old_value)
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
            
            if name == "gen_audio" and value:
                self.actions.gen_all_audio()
            elif name == 'useprompt':
                ignore = self._koboldai_vars.calc_ai_text()
            elif name == 'actions':
                self.actions.story_settings = self
                logger.debug("Calcing AI text after setting actions")
                ignore = self._koboldai_vars.calc_ai_text()
            elif name == 'story_name':
                #reset the story id if we change the name
                self.story_id = int.from_bytes(os.urandom(16), 'little', signed=True)

                # Story name influences save base
                self.save_paths.base = os.path.join("stories", self.story_name or "Untitled")
            
            #Recalc AI Text
            elif name == 'authornote':
                ignore = self._koboldai_vars.calc_ai_text()
            elif name == 'authornotetemplate':
                ignore = self._koboldai_vars.calc_ai_text()
            elif name == 'memory':
                ignore = self._koboldai_vars.calc_ai_text()
            elif name == "genres":
                self._koboldai_vars.calc_ai_text()
            elif name == 'prompt':
                self.prompt_wi_highlighted_text = [{"text": self.prompt, "WI matches": None, "WI Text": ""}]
                self.assign_world_info_to_actions(action_id=-1, wuid=None)
                process_variable_changes(self._socketio, self.__class__.__name__.replace("_settings", ""), 'prompt_wi_highlighted_text', self.prompt_wi_highlighted_text, None)
                ignore = self._koboldai_vars.calc_ai_text()
                self.actions.gen_audio(action_id=-1)
            
            #Because we have seperate variables for action types, this syncs them
            elif name == 'storymode':
                if value == 0:
                    self.adventure = False
                    self.chatmode = False
                    self.actionmode = 0
                elif value == 1:
                    self.adventure = True
                    self.chatmode = False
                elif value == 2:
                    self.adventure = False
                    self.chatmode = True
                    self.actionmode = 0
            elif name == 'adventure' and value == True:
                self.chatmode = False
                self.storymode = 1
            elif name == 'adventure' and value == False and self.chatmode == False:
                self.storymode = 0
            elif name == 'chatmode' and value == True:
                self.adventure = False
                self.storymode = 2
                self.actionmode = 0
            elif name == 'chatmode' and value == False and self.adventure == False:
                self.storymode = 0
                self.actionmode = 0
                
class user_settings(settings):
    local_only_variables = ['importjs']
    no_save_variables = ['importnum', 'importjs', 'loadselect', 'spselect', 'svowname', 'saveow', 'laststory', 'sid', "revision", "model_selected"]
    settings_name = "user"
    def __init__(self, socketio):
        self._socketio = socketio
        self.wirmvwhtsp  = False             # Whether to remove leading whitespace from WI entries
        self.widepth     = 3                 # How many historical actions to scan for WI hits
        self.formatoptns = {'frmttriminc': True, 'frmtrmblln': False, 'frmtrmspch': False, 'frmtadsnsp': True, 'singleline': False}     # Container for state of formatting options
        self.frmttriminc = True
        self.frmtrmblln  = False
        self.frmtrmspch  = False
        self.frmtadsnsp  = True
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
        self.smooth_streaming = True
        self.show_probs = False # Whether or not to show token probabilities
        self.beep_on_complete = False
        self.img_gen_priority = 1
        self.show_budget = False
        self.ui_level    = 2
        self.img_gen_api_url = "http://127.0.0.1:7860"
        self.img_gen_art_guide = "masterpiece, digital painting, <|>, dramatic lighting, highly detailed, trending"
        self.img_gen_negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"
        self.img_gen_api_username = ""
        self.img_gen_api_password = ""
        self.img_gen_steps = 30
        self.img_gen_cfg_scale = 7.0
        self.cluster_requested_models = [] # The models which we allow to generate during cluster mode
        self.wigen_use_own_wi = False
        self.wigen_amount = 80
        self.screenshot_show_story_title = True
        self.screenshot_show_author_name = True
        self.screenshot_author_name = "Anonymous"
        self.screenshot_use_boring_colors = False
        self.oaiurl      = "" # OpenAI API URL
        self.revision    = None
        self.oaiengines  = "https://api.openai.com/v1/engines"
        self.url         = "https://api.inferkit.com/v1/models/standard/generate" # InferKit API URL
        self.colaburl    = ""     # Ngrok url for Google Colab mode
        self.apikey      = ""     # API key to use for InferKit API calls
        self.oaiapikey   = ""     # API key to use for OpenAI API calls
        self.horde_api_key = "0000000000"
        self.horde_worker_name = "My Awesome Instance"
        self.horde_url = "https://horde.koboldai.net"
        self.model_selected = ""
        
    def __setattr__(self, name, value):
        new_variable = name not in self.__dict__
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        #Put variable change actions here
        if name not in self.local_only_variables and name[0] != "_" and not new_variable:
            process_variable_changes(self._socketio, self.__class__.__name__.replace("_settings", ""), name, value, old_value)

class undefined_settings(settings):
    def __init__(self):
        pass
        
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        logger.error("{} just set {} to {} in koboldai_vars. That variable isn't defined!".format(inspect.stack()[1].function, name, value))
        
class system_settings(settings):
    local_only_variables = ['lua_state', 'lua_logname', 'lua_koboldbridge', 'lua_kobold', 
                            'lua_koboldcore', 'regex_sl', 'acregex_ai', 'acregex_ui', 'comregex_ai', 'comregex_ui',
                            'sp', '_horde_pid', 'inference_config', 'image_pipeline', 
                            'summarizer', 'summary_tokenizer', 'tts_model', 'rng_states', 'comregex_ai', 'comregex_ui', 'colab_arg']
    no_save_variables = ['lua_state', 'lua_logname', 'lua_koboldbridge', 'lua_kobold', 
                         'lua_koboldcore', 'sp', 'sp_length', '_horde_pid', 'horde_share', 'aibusy', 
                         'serverstarted', 'inference_config', 'image_pipeline', 'summarizer', 'on_colab'
                         'summary_tokenizer', 'use_colab_tpu', 'noai', 'disable_set_aibusy', 'cloudflare_link', 'tts_model',
                         'generating_image', 'bit_8_available', 'host', 'hascuda', 'usegpu', 'rng_states', 'comregex_ai', 'comregex_ui', 'git_repository', 'git_branch', 'colab_arg']
    settings_name = "system"
    def __init__(self, socketio, koboldai_var):
        self._socketio = socketio
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
        self.hascuda     = torch.cuda.is_available()  # Whether torch has detected CUDA on the system
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
        self.comregex_ai = re.compile(r'(?:\n\[<\|(?:.|\n)*?\|>\](?=\n|$))|(?:\[<\|(?:.|\n)*?\|>\]\n?)')  # Pattern for matching comments to remove them before sending them to the AI
        self.comregex_ui = re.compile(r'(\[&lt;\|(?:.|\n)*?\|&gt;\])')  # Pattern for matching comments in the editor
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
        self.rng_states = {} # creates an empty dictionary to store the random number generator (RNG) states for a given seed, which is used to restore the RNG state later on
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
        self.colab_arg = False
        try:
            import google.colab
            self.on_colab = True
        except:
            self.on_colab = self.colab_arg
        print(f"Colab Check: {self.on_colab}, TPU: {self.use_colab_tpu}")
        self.horde_share = False
        self._horde_pid = None
        self.generating_image = False #The current status of image generation
        self.image_pipeline = None
        self.summarizer = None
        self.summary_tokenizer = None
        self.keep_img_gen_in_memory = False
        self.cookies = {} #cookies for colab since colab's URL changes, cookies are lost
        self.experimental_features = False
        self.seen_messages = []
        self.git_repository = ""
        self.git_branch = ""
        
        
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
            process_variable_changes(self._socketio, self.__class__.__name__.replace("_settings", ""), name, value, old_value)
            
            #if name == "aibusy" and value == False and self.abort == True:
            #    koboldai_vars.abort = False
            
            #for original UI
            if name == 'sp_changed':
                self._socketio.emit('from_server', {'cmd': 'spstatitems', 'data': {self.spfilename: self.spmeta} if self.allowsp and len(self.spfilename) else {}}, namespace=None, broadcast=True, room="UI_1")
                super().__setattr__("sp_changed", False)
            
            if name == 'keep_img_gen_in_memory' and value == False:
                self.image_pipeline = None
            
            if name == 'alt_gen' and old_value != value:
                logger.debug("Calcing AI Text from setting alt_gen")
                self._koboldai_var.calc_ai_text()
            
            if name == 'horde_share':
                if self.on_colab is True:
                    return
                if not os.path.exists("./AI-Horde-Worker"):
                    return
                if value is True:
                    if self._horde_pid is None:
                        self._horde_pid = "Pending" # Hack to make sure we don't launch twice while it loads
                        logger.info("Starting Horde bridge")
                        logger.debug("Clearing command line args in sys.argv before AI Horde Scribe load")
                        sys_arg_bkp = sys.argv.copy()
                        sys.argv = sys.argv[:1]
                        bd_module = importlib.import_module("AI-Horde-Worker.worker.bridge_data.scribe")
                        bridge_data = bd_module.KoboldAIBridgeData()
                        sys.argv = sys_arg_bkp
                        bridge_data.reload_data()
                        bridge_data.kai_url = f'http://127.0.0.1:{self.port}'
                        bridge_data.horde_url = self._koboldai_var.horde_url
                        bridge_data.api_key = self._koboldai_var.horde_api_key
                        bridge_data.scribe_name = self._koboldai_var.horde_worker_name
                        bridge_data.max_length = self._koboldai_var.genamt
                        bridge_data.max_context_length = self._koboldai_var.max_length
                        bridge_data.disable_terminal_ui = self._koboldai_var.host
                        if bridge_data.worker_name == "My Awesome Instance":
                            bridge_data.worker_name = f"KoboldAI UI Instance #{random.randint(-100000000, 100000000)}"
                        worker_module = importlib.import_module("AI-Horde-Worker.worker.workers.scribe")
                        self._horde_pid = worker_module.ScribeWorker(bridge_data)
                        new_thread = threading.Thread(target=self._horde_pid.start)
                        new_thread.daemon = True
                        new_thread.start()

                else:
                    if self._horde_pid is not None:
                        logger.info("Killing Horde bridge")
                        self._horde_pid.stop()
                        self._horde_pid = None

                
class KoboldStoryRegister(object):
    def __init__(self, socketio, story_settings, koboldai_vars, tokenizer=None, sequence=[]):
        self._socketio = socketio
        self._koboldai_vars = koboldai_vars
        #### DO NOT DIRECTLY EDIT THE ACTIONS DICT. IT WILL NOT TRANSMIT TO CLIENT. USE FUCTIONS BELOW TO DO SO ###
        #### doing actions[x] = game text is OK
        self.actions = {} #keys = "Selected Text", "Wi_highlighted_text", "Options", "Selected Text Length", "Probabilities". 
                          #Options being a list of dict with keys of "text", "Pinned", "Previous Selection", "Edited", "Probabilities"
        self.action_count = -1
        # The id of the last submission action, or 0 if the last append was not a submission
        self.submission_id = 0
        # A regular expression used to break the story into sentences so that the author's
        # note can be inserted with minimal disruption. Avoid ending a sentance with
        # whitespace because most tokenizers deal better with whitespace at the beginning of text.
        # Search for sentence end delimeters (i.e. .!?) and also capture closing parenthesis and quotes.
        self.sentence_re = re.compile(r".*?[.!?]+(?=[.!?\]\)}>'\"›»\s])[.!?\]\)}>'\"›»]*", re.S)
        self.story_settings = story_settings
        self.tts_model = None
        self.tortoise = None
        self.make_audio_thread = None
        self.make_audio_queue = multiprocessing.Queue()
        self.make_audio_thread_slow = None
        self.make_audio_queue_slow = multiprocessing.Queue()
        self.probability_buffer = None
        for item in sequence:
            self.append(item)
    
    def reset(self, sequence=[]):
        self.__init__(self._socketio, self.story_settings, self._koboldai_vars, sequence=sequence)
        
    def add_wi_to_action(self, action_id, key, content, uid, no_transmit=False):
        old = self.story_settings.prompt_wi_highlighted_text.copy() if action_id == -1 else self.actions[action_id].copy()
        force_changed = False
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
            elif action[i]['WI matches'] == uid:
                action[i]['WI Text'] = content
                force_changed = True
                i+=1
            else:
                i+=1
        if action_id != -1:
            if old != self.actions[action_id] or force_changed:
                if not no_transmit:
                    process_variable_changes(self._socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, old)
        else:
            if old != self.story_settings.prompt_wi_highlighted_text or force_changed:
                if not no_transmit:
                    process_variable_changes(self._socketio, "story", 'prompt_wi_highlighted_text', self.story_settings.prompt_wi_highlighted_text, old)
        
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
        if self._koboldai_vars.remove_double_space:
            while "  " in text:
                text = text.replace("  ", " ")
            if i > 0 and text != "" and i-1 in self.actions and self.actions[i-1]['Selected Text'] != "":
                if self.actions[i-1]['Selected Text'][-1] == " " and text[0] == " ":
                    text = text[1:]
        if i in self.actions:
            old = self.actions[i]
            old_text = self.actions[i]["Selected Text"]
            if self.actions[i]["Selected Text"] != text:
                self.actions[i]["Selected Text"] = text
                if 'wi_highlighted_text' in self.actions[i]:
                    del self.actions[i]['wi_highlighted_text']
                if self._koboldai_vars.tokenizer is not None:
                    tokens = self._koboldai_vars.tokenizer.encode(text)
                    if 'Probabilities' in self.actions[i]:
                        for token_num in range(len(self.actions[i]["Probabilities"])):
                            for token_option in range(len(self.actions[i]["Probabilities"][token_num])):
                                if token_num < len(tokens):
                                    self.actions[i]["Probabilities"][token_num][token_option]["Used"] = tokens[token_num] == self.actions[i]["Probabilities"][token_num][token_option]["tokenId"]
                                else:
                                    self.actions[i]["Probabilities"][token_num][token_option]["Used"] = False
            if "Options" in self.actions[i]:
                for j in reversed(range(len(self.actions[i]["Options"]))):
                    if self.actions[i]["Options"][j]["text"] == text:
                        del self.actions[i]["Options"][j]
            if old_text != "":
                self.actions[i]["Options"].append({"text": old_text, "Pinned": False, "Previous Selection": False, "Edited": True})
        else:
            old_text = None
            old_length = None
            old = None
            self.actions[i] = {"Selected Text": text, "Probabilities": [], "Options": [], "Time": int(time.time())}
            
        self.story_settings.assign_world_info_to_actions(action_id=i, no_transmit=True)
        process_variable_changes(self._socketio, "story", 'actions', {"id": i, 'action':  self.actions[i]}, old)
        logger.debug("Calcing AI Text from Action __setitem__")
        ignore = self._koboldai_vars.calc_ai_text()
        self.set_game_saved()
        self.gen_audio(i)
    
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
            if "Time" not in json_data["actions"][item]:
                json_data["actions"][item]["Time"] = int(time.time())

            if "Original Text" not in json_data["actions"][item]:
                json_data["actions"][item]["Original Text"] = json_data["actions"][item]["Selected Text"]

            temp[int(item)] = json_data['actions'][item]
            if int(item) >= self.action_count-100: #sending last 100 items to UI
                data_to_send.append({"id": item, 'action':  temp[int(item)]})
        
        process_variable_changes(self._socketio, "story", 'actions', data_to_send, None)
        
        self.actions = temp
        #Check if our json has our new world info highlighting data
        if len(self.actions) > 0:
            if 'wi_highlighted_text' not in self.actions[0]:
                self.story_settings.assign_world_info_to_actions(no_transmit=True)
        
        logger.debug("Calcing AI Text from Action load from json")
        ignore = self._koboldai_vars.calc_ai_text()
        self.set_game_saved()
        self.gen_all_audio()

    def append(self, text, action_id_offset=0, recalc=True, submission=False):
        """Create a new action and append it to the table of actions.

        text: The text of the action.
        action_id_offset: An optional offset added to action_count when assiging an action_id.
        recalc: If True, recalculate the context and generate audio.
        submission: If True, mark this action as a user submission.
        """
        if self._koboldai_vars.remove_double_space:
            while "  " in text:
                text = text.replace("  ", " ")
            if action_id_offset > 0:
                if self.actions[action_id_offset-1]['Selected Text'][-1] == " " and text[0] == " ":
                    text = text[1:]
        self.clear_unused_options(clear_probabilities=False)
        self.action_count+=1
        action_id = self.action_count + action_id_offset
        if action_id in self.actions:
            if not self.actions[action_id].get("Original Text"):
                self.actions[action_id]["Original Text"] = text

            if self.actions[action_id]["Selected Text"] != text:
                self.actions[action_id]["Selected Text"] = text
                self.actions[action_id]["Time"] = self.actions[action_id].get("Time", int(time.time()))
                if 'buffer' in self.actions[action_id]:
                    if self._koboldai_vars.tokenizer is not None:
                        tokens = self._koboldai_vars.tokenizer.encode(text)
                        for token_num in range(len(self.actions[action_id]["Probabilities"])):
                            for token_option in range(len(self.actions[action_id]["Probabilities"][token_num])):
                                if token_num < len(tokens):
                                    self.actions[action_id]["Probabilities"][token_num][token_option]["Used"] = tokens[token_num] == self.actions[action_id]["Probabilities"][token_num][token_option]["tokenId"]
                                else:
                                    self.actions[action_id]["Probabilities"][token_num][token_option]["Used"] = False
            selected_text_length = 0
            self.actions[action_id]["Selected Text Length"] = selected_text_length
            for item in self.actions[action_id]["Options"]:
                if item['text'] == text:
                    old_options = self.actions[action_id]["Options"]
                    del item
                    
        else:
            selected_text_length = 0
            
            self.actions[action_id] = {
                "Selected Text": text,
                "Selected Text Length": selected_text_length,
                "Options": [],
                "Probabilities": [],
                "Time": int(time.time()),
                "Original Text": text,
                "Origin": "user" if submission else "ai"
            }

        if submission:
            self.submission_id = action_id
        else:
            self.submission_id = 0

        if self.story_settings is not None:
            self.story_settings.assign_world_info_to_actions(action_id=action_id, no_transmit=True)
            process_variable_changes(self._socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, None)
        self.set_game_saved()
        if recalc:
            logger.debug("Calcing AI Text from Action Append")
            ignore = self._koboldai_vars.calc_ai_text()
            self.gen_audio(action_id)
    
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
                        if 'Probabilities' in item:
                            if self._koboldai_vars.tokenizer is not None:
                                tokens = self._koboldai_vars.tokenizer.encode(option)
                                for token_num in range(len(item["Probabilities"])):
                                    for token_option in range(len(item["Probabilities"][token_num])):
                                        if token_num < len(tokens):
                                            item["Probabilities"][token_num][token_option]["Used"] = tokens[token_num] == item["Probabilities"][token_num][token_option]["tokenId"]
                                        else:
                                            item["Probabilities"][token_num][token_option]["Used"] = False
                        break
                    elif item['text'] == option:
                        found = True
                        if 'stream_id' in item:
                            del item['stream_id']
                        found = True
                        if 'Probabilities' in item:
                            if self._koboldai_vars.tokenizer is not None:
                                tokens = self._koboldai_vars.tokenizer.encode(option)
                                for token_num in range(len(item["Probabilities"])):
                                    for token_option in range(len(item["Probabilities"][token_num])):
                                        if token_num < len(tokens):
                                            item["Probabilities"][token_num][token_option]["Used"] = tokens[token_num] == item["Probabilities"][token_num][token_option]["tokenId"]
                                        else:
                                            item["Probabilities"][token_num][token_option]["Used"] = False
                        break
                        
                if not found:
                    self.actions[self.action_count+1]['Options'].append({"text": option, "Pinned": False, "Previous Selection": False, "Edited": False, "Probabilities": []})
        else:
            old_options = None
            self.actions[self.action_count+1] = {
                "Selected Text": "",
                "Selected Text Length": 0,
                "Options": [{"text": x, "Pinned": False, "Previous Selection": False, "Edited": False, "Probabilities": []} for x in option_list],
                "Time": int(time.time()),
            }
        process_variable_changes(self._socketio, "story", 'actions', {"id": self.action_count+1, 'action':  self.actions[self.action_count+1]}, None)
        self.set_game_saved()
            
    def set_options(self, option_list, action_id):
        if action_id not in self.actions:
            old_options = None
            self.actions[action_id] = {
                "Selected Text": "",
                "Options": option_list,
                "Time": int(time.time()),
            }
        else:
            old_options = self.actions[action_id]["Options"]
            self.actions[action_id]["Options"] = []
            for item in option_list:
                for old_item in old_options:
                    if item['text'] == old_item['text']:
                        #We already have this option, so we need to save the probabilities
                        if 'Probabilities' in old_item:
                            item['Probabilities'] = old_item['Probabilities']
                    self.actions[action_id]["Options"].append(item)
        process_variable_changes(self._socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, None)
    
    def clear_unused_options(self, pointer=None, clear_probabilities=True):
        new_options = []
        old_options = None
        if pointer is None:
            pointer = self.action_count+1
        if pointer in self.actions:
            old_options = copy.deepcopy(self.actions[pointer]["Options"])
            self.actions[pointer]["Options"] = [x for x in self.actions[pointer]["Options"] if x["Pinned"] or x["Previous Selection"] or x["Edited"]]
            new_options = self.actions[pointer]["Options"]
            if clear_probabilities:
                self.actions[pointer]['Probabilities'] = []
            process_variable_changes(self._socketio, "story", 'actions', {"id": pointer, 'action':  self.actions[pointer]}, None)
        self.set_game_saved()
    
    def clear_all_options(self):
        for i in self.actions:
            self.actions[i]["Options"] = []
    
    def set_pin(self, action_step, option_number):
        if action_step in self.actions:
            if option_number < len(self.actions[action_step]['Options']):
                old_options = copy.deepcopy(self.actions[action_step]["Options"])
                self.actions[action_step]['Options'][option_number]['Pinned'] = True
                process_variable_changes(self._socketio, "story", 'actions', {"id": action_step, 'action':  self.actions[action_step]}, None)
                self.set_game_saved()
    
    def unset_pin(self, action_step, option_number):
        if action_step in self.actions:
            old_options = copy.deepcopy(self.actions[action_step]["Options"])
            if option_number < len(self.actions[action_step]['Options']):
                self.actions[action_step]['Options'][option_number]['Pinned'] = False
                process_variable_changes(self._socketio, "story", 'actions', {"id": action_step, 'action':  self.actions[action_step]}, None)
                self.set_game_saved()
    
    def toggle_pin(self, action_step, option_number):
        if action_step in self.actions:
            old_options = copy.deepcopy(self.actions[action_step]["Options"])
            if option_number < len(self.actions[action_step]['Options']):
                self.actions[action_step]['Options'][option_number]['Pinned'] = not self.actions[action_step]['Options'][option_number]['Pinned']
                process_variable_changes(self._socketio, "story", 'actions', {"id": action_step, 'action':  self.actions[action_step]}, None)
                self.set_game_saved()
    
    def go_forward(self):
        action_step = self.action_count+1
        if action_step not in self.actions:
            return

        self.show_options(len(self.get_current_options()) > 1)

        if len(self.get_current_options()) == 1:
            logger.debug("Going forward with this text: {}".format(self.get_current_options()[0]["text"]))
            self.use_option([x['text'] for x in self.actions[action_step]["Options"]].index(self.get_current_options()[0]["text"]))

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
                if self._koboldai_vars.tokenizer is not None:
                    self.actions[action_step]['Selected Text Length'] = len(self._koboldai_vars.tokenizer.encode(self.actions[action_step]['Options'][option_number]['text']))
                else:
                    self.actions[action_step]['Selected Text Length'] = 0
                del self.actions[action_step]['Options'][option_number]
                #If this is the current spot in the story, advance
                if action_step-1 == self.action_count:
                    self.action_count+=1
                    self._socketio.emit("var_changed", {"classname": "actions", "name": "Action Count", "old_value": None, "value":self.action_count, "transmit_time": str(datetime.datetime.now())}, broadcast=True, room="UI_2")
                self.story_settings.assign_world_info_to_actions(action_id=action_step, no_transmit=True)
                self.clear_unused_options(pointer=action_step)
                #process_variable_changes(self._socketio, "story", 'actions', {"id": action_step, 'action':  self.actions[action_step]}, None)
                self.set_game_saved()
                logger.debug("Calcing AI Text from Action Use Option")
                ignore = self._koboldai_vars.calc_ai_text()
                self.gen_audio(action_step)
    
    def delete_option(self, option_number, action_step=None):
        if action_step is None:
            action_step = self.action_count+1
        if action_step in self.actions:
            if option_number < len(self.actions[action_step]['Options']):
                del self.actions[action_step]['Options'][option_number]
                process_variable_changes(self._socketio, "story", 'actions', {"id": action_step, 'action':  self.actions[action_step]}, None)
                self.set_game_saved()
    
    def show_options(
        self,
        should_show: bool,
        force: bool = False,

    ) -> None:
        if self._koboldai_vars.aibusy and not force:
            return
        self._socketio.emit("show_options", should_show, broadcast=True, room="UI_2")
    
    def delete_action(self, action_id, keep=True):
        if action_id in self.actions:
            old_options = copy.deepcopy(self.actions[action_id]["Options"])
            old_text = self.actions[action_id]["Selected Text"]
            old_length = self.actions[action_id]["Selected Text Length"]
            if keep:
                if {"text": self.actions[action_id]["Selected Text"], "Pinned": False, "Previous Selection": True, "Edited": False} not in self.actions[action_id]["Options"]:
                    self.actions[action_id]["Options"].append({"text": self.actions[action_id]["Selected Text"], "Pinned": False, "Previous Selection": True, "Edited": False})
            self.actions[action_id]["Selected Text"] = ""
            if "wi_highlighted_text" in self.actions[action_id]:
                del self.actions[action_id]["wi_highlighted_text"]
            self.actions[action_id]['Selected Text Length'] = 0
            self.action_count -= 1
            process_variable_changes(self._socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, None)
            self.set_game_saved()
            logger.debug("Calcing AI Text from Action Delete")
            ignore = self._koboldai_vars.calc_ai_text()
            
    def pop(self, keep=True):
        if self.action_count >= 0:
            text = self.actions[self.action_count]['Selected Text']
            self.delete_action(self.action_count, keep=keep)
            logger.debug("Calcing AI Text from Action Pop")
            return text
        else:
            return None
            
    def get_last_key(self):
        return self.action_count
            
    def get_current_options(self):
        if self.action_count+1 in self.actions:
            return [x for x in self.actions[self.action_count+1]["Options"] if x['Edited'] != True]
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
        if len(text_list) > 1 or (self._koboldai_vars.alt_multi_gen and self._koboldai_vars.numseqs > 1):
            if self._koboldai_vars.alt_multi_gen: 
                #since alt_multi_gen is really just several single gens the text list is always 1 deep, so we need some 
                #other way to figure out wich spot in our options list we're on. We'll figure it out by seeing how many
                #tokens we generated vs how many each option should take
                stream_offset = int((self._koboldai_vars.generated_tkns-1) / self._koboldai_vars.genamt)
            else:
                stream_offset = 0
            option_offset = 0
            if self.action_count+1 in self.actions:
                for x in range(len(self.actions[self.action_count+1]['Options'])):
                    option = self.actions[self.action_count+1]['Options'][x]
                    if option['Pinned'] or option["Previous Selection"] or option["Edited"]:
                        option_offset = x+1
            if self.action_count+1 in self.actions:
                for i in range(len(text_list)):
                    if i+stream_offset+option_offset < len(self.actions[self.action_count+1]['Options']):
                        self.actions[self.action_count+1]['Options'][i+stream_offset+option_offset]['text'] = "{}{}".format(self.actions[self.action_count+1]['Options'][i+stream_offset+option_offset]['text'], text_list[i])
                    else:
                        logger.info(self.actions[self.action_count+1]['Options'])
                        self.actions[self.action_count+1]['Options'].append({"text": text_list[i], "Pinned": False, "Previous Selection": False, "Edited": False, "Probabilities": [], "stream_id": i+stream_offset})
            else:
                self.actions[self.action_count+1] = {"Selected Text": "", "Selected Text Length": 0, "Options": [], "Time": int(time.time())}
                for i in range(len(text_list)):
                    self.actions[self.action_count+1]['Options'].append({"text": text_list[i], "Pinned": False, "Previous Selection": False, "Edited": False, "Probabilities": [], "stream_id": i+stream_offset})
        
            #We need to see if this is the last token being streamed. If so due to the rely it will come in AFTER the actual trimmed final text overwriting it in the UI
            if self._koboldai_vars.tokenizer is not None:
                if len(self._koboldai_vars.tokenizer.encode(self.actions[self.action_count+1]["Options"][0]['text'])) != self._koboldai_vars.genamt:
                    #process_variable_changes(self._socketio, "actions", "Options", {"id": self.action_count+1, "options": self.actions[self.action_count+1]["Options"]}, {"id": self.action_count+1, "options": None})
                    process_variable_changes(self._socketio, "story", 'actions', {"id": self.action_count+1, 'action':  self.actions[self.action_count+1]}, None)
        else:
            #We're streaming single options so our output is our selected
            queue.put(["stream_tokens", text_list, {"broadcast": True, "room": "UI_2"}])

            # UI1
            queue.put([
                "from_server", {
                    "cmd": "streamtoken",
                    "data": [{
                        "decoded": text_list[0],
                        "probabilities": self.probability_buffer
                    }],
                },
                {"broadcast":True, "room": "UI_1"}
            ])
    
    def set_probabilities(self, probabilities, action_id=None):
        self.probability_buffer = probabilities

        if action_id is None:
            action_id = self.action_count+1
        if action_id in self.actions:
            if 'Probabilities' not in self.actions[action_id]:
                self.actions[action_id]['Probabilities'] = []
            self.actions[action_id]['Probabilities'].append(probabilities)
        else:
            self.actions[action_id] = {
                "Selected Text": "",
                "Selected Text Length": 0,
                "Options": [],
                "Probabilities": [probabilities],
                "Time": int(time.time()),
            }
            
        process_variable_changes(self._socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, None)
            
    def set_option_probabilities(self, probabilities, option_number, action_id=None):
        if action_id is None:
            action_id = self.action_count+1
        if action_id in self.actions:
            old_options = self.actions[action_id]["Options"]
            if option_number < len(self.actions[action_id]["Options"]):
                if "Probabilities" not in self.actions[action_id]["Options"][option_number]:
                    self.actions[action_id]["Options"][option_number]["Probabilities"] = []
                self.actions[action_id]["Options"][option_number]['Probabilities'].append(probabilities)
                process_variable_changes(self._socketio, "story", 'actions', {"id": action_id, 'action':  self.actions[action_id]}, None)
            else:
                self.actions[action_id]['Options'].append({"temp_prob": True, "text": "", "Pinned": False, "Previous Selection": False, "Edited": False, "Probabilities": [probabilities], "stream_id": option_number})
        else:
            self.actions[action_id] = {
                "Selected Text": "",
                "Selected Text Length": 0,
                "Options": [{"temp_prob": True, "text": "", "Pinned": False, "Previous Selection": False, "Edited": False, "Probabilities": [probabilities], "stream_id": option_number}],
                "Time": int(time.time()),
            }
    
    def to_sentences(self, submitted_text=None, max_action_id=None):
        """Return a list of the actions split into sentences.
        submitted_text: Optional additional text to append to the actions, the text just submitted by the player.
        returns: List of [sentence text, actions used, token length, included in AI context]
        """
        #start_time = time.time()
        #we're going to split our actions by sentence for better context. We'll add in which actions the sentence covers. Prompt will be added at a -1 ID
        if max_action_id is None:
            actions = {i: self.actions[i]['Selected Text'] for i in self.actions}
        else:
            actions = {i: self.actions[i]['Selected Text'] for i in self.actions if i <= max_action_id}
        if self.story_settings is None:
            actions[-1] = ""
        else:
            actions[-1] = self.story_settings.prompt
        # During generation, the action with id action_count+1 holds streamed tokens, and action_count holds the last
        # submitted text.  Outside of generation, action_count+1 is missing/empty.  Strip streaming tokens and
        # replace submitted if specified.
        if self.action_count+1 in self.actions:
            actions[self.action_count+1] = ""
        if submitted_text:
            if self.submission_id:
                # Text has been submitted
                actions[self.submission_id] = submitted_text
            elif self.action_count == -1:
                # The only submission is the prompt
                actions[-1] = submitted_text
            else:
                # Add submitted_text to the end
                actions[self.action_count + 1] = submitted_text
        action_text = "".join(txt for _, txt in sorted(actions.items()))
        ###########action_text_split = [sentence, actions used in sentence, token length, included in AI context]################
        action_text_split = [[x, [], 0, False] for x in self.sentence_re.findall(action_text)]
        #The above line can trim out the last sentence if it's incomplete. Let's check for that and add it back in
        text_len = sum(len(x[0]) for x in action_text_split)
        if text_len < len(action_text):
            action_text_split.append([action_text[text_len:], [], 0, False])
        #If we don't have any actions at this point, just return.
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
                #Only add actions with non-empty text
                if len(actions[Action_Position[0]]) > 0:
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
        return action_text_split
    
    def gen_audio(self, action_id=None, overwrite=True):
        if self.story_settings.gen_audio and self._koboldai_vars.experimental_features:
            if action_id is None:
                action_id = self.action_count

            if self.tts_model is None:
                language = 'en'
                model_id = 'v3_en'
                self.tts_model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                 model='silero_tts',
                                                 language=language,
                                                 speaker=model_id)
                #self.tts_model.to(torch.device(0))  # gpu or cpu
                self.tts_model.to(torch.device("cpu"))  # gpu or cpu
            
            filename = os.path.join(self._koboldai_vars.save_paths.generated_audio, f"{action_id}.ogg")
            filename_slow = os.path.join(self._koboldai_vars.save_paths.generated_audio, f"{action_id}_slow.ogg")
                
            if overwrite or not os.path.exists(filename):
                if action_id == -1:
                    self.make_audio_queue.put((self._koboldai_vars.prompt, filename))
                else:
                    self.make_audio_queue.put((self.actions[action_id]['Selected Text'], filename))
                if self.make_audio_thread_slow is None or not self.make_audio_thread_slow.is_alive():
                    self.make_audio_thread_slow = threading.Thread(target=self.create_wave_slow, args=(self.make_audio_queue_slow, ))
                    self.make_audio_thread_slow.start()
            
            if overwrite or not os.path.exists(filename_slow):
                if action_id == -1:
                    self.make_audio_queue_slow.put((self._koboldai_vars.prompt, filename_slow))
                else:
                    self.make_audio_queue_slow.put((self.actions[action_id]['Selected Text'], filename_slow))
                if self.make_audio_thread_slow is None or not self.make_audio_thread_slow.is_alive():
                    self.make_audio_thread_slow = threading.Thread(target=self.create_wave_slow, args=(self.make_audio_queue_slow, ))
                    self.make_audio_thread_slow.start()
                
    def create_wave(self, make_audio_queue):
        import pydub
        sample_rate = 24000
        speaker = 'en_5'
        while not make_audio_queue.empty():
            (text, filename) = make_audio_queue.get()
            logger.info("Creating audio for {}".format(os.path.basename(filename)))
            if text.strip() == "":
                shutil.copy("data/empty_audio.ogg", filename)
            else:
                if len(text) > 2000:
                    text = self.sentence_re.findall(text)
                else:
                    text = [text]
                output = None
                for process_text in text:
                    audio = self.tts_model.apply_tts(text=process_text,
                                            speaker=speaker,
                                            sample_rate=sample_rate)
                                            #audio_path=filename)
                    channels = 2 if (audio.ndim == 2 and audio.shape[1] == 2) else 1
                    if output is None:
                        output = pydub.AudioSegment(np.int16(audio * 2 ** 15).tobytes(), frame_rate=sample_rate, sample_width=2, channels=channels)
                    else:
                        output = output + pydub.AudioSegment(np.int16(audio * 2 ** 15).tobytes(), frame_rate=sample_rate, sample_width=2, channels=channels)
                output.export(filename, format="ogg", bitrate="16k")
    
    def create_wave_slow(self, make_audio_queue_slow):
        import pydub
        sample_rate = 24000
        speaker = 'train_daws'
        if self.tortoise is None and importlib.util.find_spec("tortoise") is not None:
           self.tortoise=api.TextToSpeech()
        
        if importlib.util.find_spec("tortoise") is not None:
            voice_samples, conditioning_latents = load_voices([speaker])
            while not make_audio_queue_slow.empty():
                start_time = time.time()
                (text, filename) = make_audio_queue_slow.get()
                text_length = len(text)
                logger.info("Creating audio for {}".format(os.path.basename(filename)))
                if text.strip() == "":
                    shutil.copy("data/empty_audio.ogg", filename)
                else:
                    if len(text) > 20000:
                        text = self.sentence_re.findall(text)
                    else:
                        text = [text]
                output = None
                for process_text in text:
                    audio = self.tortoise.tts_with_preset(process_text, preset='ultra_fast', voice_samples=voice_samples, conditioning_latents=conditioning_latents).numpy()
                    channels = 2 if (audio.ndim == 2 and audio.shape[1] == 2) else 1
                    if output is None:
                        output = pydub.AudioSegment(np.int16(audio * 2 ** 15).tobytes(), frame_rate=sample_rate, sample_width=2, channels=channels)
                    else:
                        output = output + pydub.AudioSegment(np.int16(audio * 2 ** 15).tobytes(), frame_rate=sample_rate, sample_width=2, channels=channels)
                output.export(filename, format="ogg", bitrate="16k")
                logger.info("Slow audio took {} for {} characters".format(time.time()-start_time, text_length))
    
    def gen_all_audio(self, overwrite=False):
        if self.story_settings.gen_audio and self._koboldai_vars.experimental_features:
            for i in reversed([-1]+list(self.actions.keys())):
                self.gen_audio(i, overwrite=False)
        #else:
        #    print("{} and {}".format(self.story_settings.gen_audio, self._koboldai_vars.experimental_features))
    
    def set_picture(self, action_id, filename, prompt):
        if action_id == -1:
            self.story_settings.prompt_picture_filename = filename
            self.story_settings.prompt_picture_prompt = prompt
        elif action_id in self.actions:
            self.actions[action_id]['picture_filename'] = filename
            self.actions[action_id]['picture_prompt'] = prompt
    
    def get_picture(self, action_id):
        if action_id == -1:
            if self.story_settings.prompt_picture_filename == "":
                return None, None
            filename = os.path.join(self._koboldai_vars.save_paths.generated_images, self.story_settings.prompt_picture_filename)
            prompt = self.story_settings.prompt_picture_prompt
        elif action_id in self.actions and 'picture_filename' in self.actions[action_id]:
            filename = os.path.join(self._koboldai_vars.save_paths.generated_images, self.actions[action_id]['picture_filename'])
            prompt = self.actions[action_id]['picture_prompt']
        else:
            #Let's find the last picture if there is one
            found = False
            for i in reversed(range(-1, action_id)):
                if i in self.actions and 'picture_filename' in self.actions[i]:
                    filename = os.path.join(self._koboldai_vars.save_paths.generated_images, self.actions[i]['picture_filename'])
                    prompt = self.actions[i]['picture_prompt']
                    found = True
                    break
                elif i == -1:
                    if self.story_settings.prompt_picture_filename == "":
                        return None, None
                    filename = os.path.join(self._koboldai_vars.save_paths.generated_images, self.story_settings.prompt_picture_filename)
                    prompt = self.story_settings.prompt_picture_prompt
                    found = True
            if not found:
                return None, None
        
        if os.path.exists(filename):
            return filename, prompt
        return None, None
    
    def get_action_composition(self, action_id: int) -> List[dict]:
        """
        Returns a list of chunks that comprise an action in dictionaries
        formatted as follows:
            type: string identifying chunk type ("ai", "user", "edit", or "prompt")
            content: the actual content of the chunk
        """
        # Prompt doesn't need standard edit data
        if action_id == -1:
            if self._koboldai_vars.prompt:
                return [{"type": "prompt", "content": self._koboldai_vars.prompt}]
            return []

        current_text = self.actions[action_id]["Selected Text"]
        action_original_type = self.actions[action_id].get("Origin", "ai")
        original = self.actions[action_id]["Original Text"]

        matching_blocks = difflib.SequenceMatcher(
            None,
            self.actions[action_id]["Original Text"],
            current_text
        ).get_matching_blocks()

        chunks = []
        base = 0
        for chunk_match in matching_blocks:
            inserted = current_text[base:chunk_match.b]
            content = current_text[chunk_match.b:chunk_match.b + chunk_match.size]

            base = chunk_match.b + chunk_match.size

            if inserted:
                chunks.append({"type": "edit", "content": inserted})
            if content:
                chunks.append({"type": action_original_type, "content": content})

        return chunks

    def __setattr__(self, name, value):
        new_variable = name not in self.__dict__
        old_value = getattr(self, name, None)
        super().__setattr__(name, value)
        if name == 'action_count' and not new_variable:
            process_variable_changes(self._socketio, "actions", "Action Count", value, old_value)

class KoboldWorldInfo(object):
    
    def __init__(self, socketio, story_settings, koboldai_vars, tokenizer=None):
        self._socketio = socketio
        self._koboldai_vars = koboldai_vars
        self.world_info = {}
        self.world_info_folder = OrderedDict()
        self.world_info_folder['root'] = []
        self.story_settings = story_settings
        
    def reset(self):
        self.__init__(self._socketio, self.story_settings, self._koboldai_vars)
    
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
        if self._socketio is not None:
            self._socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
        
    def delete_folder(self, folder):
        keys = [key for key in self.world_info]
        for key in keys:
            if self.world_info[key]['folder'] == folder:
                self.delete(key)
        if folder in self.world_info_folder:
            del self.world_info_folder[folder]
        self.sync_world_info_to_old_format()
        if self._socketio is not None:
            self._socketio.emit("delete_world_info_folder", folder, broadcast=True, room="UI_2")
            self._socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
        logger.debug("Calcing AI Text from WI Folder Delete")
        ignore = self._koboldai_vars.calc_ai_text()
        
    def add_item_to_folder(self, uid, folder, before=None):
        if uid in self.world_info:
            #first we need to remove the item from whatever folder it's in
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
        if self._socketio is not None:
            self._socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
                
    def add_item(self, title, key, keysecondary, folder, constant, manual_text,
                 comment, wi_type="wi", use_wpp=False,
                 wpp={'name': "", 'type': "", 'format': "W++", 'attributes': {}},
                 v1_uid=None, recalc=True, sync=True, send_to_ui=True, object_type=None):
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
        if self._koboldai_vars.tokenizer is not None:
            token_length = len(self._koboldai_vars.tokenizer.encode(content))
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
                                    "type": wi_type,
                                    "token_length": token_length,
                                    "selective": len(keysecondary) > 0,
                                    "used_in_game": constant,
                                    'wpp': wpp,
                                    'use_wpp': use_wpp,
                                    'v1_uid': v1_uid,
                                    "object_type": object_type,
                                    }
        except:
            print("Error:")
            print(key)
            print(title)
            raise
        if folder not in self.world_info_folder:
            self.world_info_folder[folder] = []
        if uid not in self.world_info_folder[folder]:
            self.world_info_folder[folder].append(uid)
        self.story_settings.gamesaved = False
        if sync:
            self.sync_world_info_to_old_format()
        
        self.story_settings.assign_world_info_to_actions(wuid=uid)
        
        if self._socketio is not None and send_to_ui:
            self._socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
            self._socketio.emit("world_info_entry", self.world_info[uid], broadcast=True, room="UI_2")
        if recalc:
            logger.debug("Calcing AI Text from WI Add")
            ignore = self._koboldai_vars.calc_ai_text()
        return uid
        
    def edit_item(
            self,
            uid,
            title,
            key,
            keysecondary,
            folder,
            constant,
            manual_text,
            comment,
            wi_type,
            use_wpp=False,
            before=None,
            wpp={'name': "", 'type': "", 'format': "W++", 'attributes': {}},
            object_type=None,
        ):
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
        if self._koboldai_vars.tokenizer is not None:
            token_length = len(self._koboldai_vars.tokenizer.encode(content))
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
                                "type": wi_type,
                                "token_length": token_length,
                                "selective": len(keysecondary) > 0,
                                "used_in_game": constant,
                                'wpp': wpp,
                                'use_wpp': use_wpp,
                                "object_type": object_type,
                                }
                                
        self.story_settings.gamesaved = False
        self.sync_world_info_to_old_format()
        self.story_settings.assign_world_info_to_actions(wuid=uid)
        logger.debug("Calcing AI Text from WI Edit")
        ignore = self._koboldai_vars.calc_ai_text()
        
        if self._socketio is not None:
            self._socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
            self._socketio.emit("world_info_entry", self.world_info[uid], broadcast=True, room="UI_2")
        
    def delete(self, uid):
        del self.world_info[uid]
        
        try:
            os.remove(os.path.join(self._koboldai_vars.save_paths.wi_images, str(uid)))
        except FileNotFoundError:
            pass

        for folder in self.world_info_folder:
            if uid in self.world_info_folder[folder]:
                self.world_info_folder[folder].remove(uid)
        
        
        self.story_settings.gamesaved = False
        self.sync_world_info_to_old_format()
        if self._socketio is not None:
            self._socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
            self._socketio.emit("delete_world_info_entry", uid, broadcast=True, room="UI_2")
        logger.debug("Calcing AI Text from WI Delete")
        ignore = self._koboldai_vars.calc_ai_text()
    
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
        if self._socketio is not None:
            self._socketio.emit("delete_world_info_folder", old_folder, broadcast=True, room="UI_2")
            self.send_to_ui()
    
    def reorder(self, uid, before):
        self.add_item_to_folder(uid, self.world_info[before]['folder'], before=before)
        self.sync_world_info_to_old_format()
    
    def send_to_ui(self):
        if self._socketio is not None:
            self._socketio.emit("world_info_folder", {x: self.world_info_folder[x] for x in self.world_info_folder}, broadcast=True, room="UI_2")
            logger.debug("Sending all world info from send_to_ui")
            self._socketio.emit("world_info_entry", [self.world_info[uid] for uid in self.world_info], broadcast=True, room="UI_2")
    
    def to_json(self, folder=None):
        if folder is None:
            return {
                    "folders": {x: self.world_info_folder[x] for x in self.world_info_folder},
                    "entries": self.world_info,
                   }
        else:
            return {
                    "folders": {x: self.world_info_folder[x] for x in self.world_info_folder if x == folder},
                    "entries": {x: self.world_info[x] for x in self.world_info if self.world_info[x]['folder'] == folder},
                   }
    
    def upgrade_entry(self, wi_entry: dict) -> dict:
        # If we do not have a type, or it is incorrect, set to WI.
        if wi_entry.get("type") not in ["constant", "chatcharacter", "wi", "commentator"]:
            wi_entry["type"] = "wi"
        
        if wi_entry["type"] in ["commentator", "constant"]:
            wi_entry["constant"] = True

        return wi_entry
    
    def load_json(self, data, folder=None):
        # Legacy WI images (stored in json)
        if "images" in data:
            for uid, image_b64 in data["images"].items():
                image_b64 = image_b64.split(",")[-1]
                image_path = os.path.join(
                    self._koboldai_vars.save_paths.wi_images,
                    str(uid)
                )
                with open(image_path, "wb") as file:
                    file.write(base64.b64decode(image_b64))
        
        data["entries"] = {int(k): self.upgrade_entry(v) for k,v in data["entries"].items()}
        
        #Add the item
        start_time = time.time()
        for uid, item in data['entries'].items():
            self.add_item(item['title'] if 'title' in item else item['key'][0], 
                          item['key'] if 'key' in item else [], 
                          item['keysecondary'] if 'keysecondary' in item else [], 
                          folder if folder is not None else item['folder'] if 'folder' in item else 'root', 
                          item['constant'] if 'constant' in item else False, 
                          item['manual_text'] if 'manual_text' in item else item['content'], 
                          item['comment'] if 'comment' in item else '',
                          wi_type=item["type"],
                          use_wpp=item['use_wpp'] if 'use_wpp' in item else False, 
                          wpp=item['wpp'] if 'wpp' in item else {'name': "", 'type': "", 'format': "W++", 'attributes': {}},
                          object_type=item.get("object_type"),
                          v1_uid=item.get("v1_uid"),
                          recalc=False, sync=False)

        logger.debug("Load World Info took {}s".format(time.time()-start_time))
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
        if "root" not in self.world_info_folder:
            old_world_info_folder = self.world_info_folder
            self.world_info_folder = OrderedDict()
            self.world_info_folder["root"] = []
            self.world_info_folder.update(old_world_info_folder)
        folder_entries = {}
        i=-1
        for folder in self.world_info_folder:
            folder_entries[folder] = i
            i-=1
        
        #self.wifolders_l = []     # List of World Info folder UIDs
        self.story_settings.wifolders_l = [folder_entries[x] for x in folder_entries if x != "root"]
        
        #self.worldinfo_i = []     # List of World Info key/value objects sans uninitialized entries
        self.story_settings.worldinfo_i = [{
                                            "key": ",".join(self.world_info[x]['key']),
                                            "keysecondary": ",".join(self.world_info[x]['keysecondary']),
                                            "content": self.world_info[x]['content'],
                                            "comment": self.world_info[x]['comment'],
                                            "folder": folder_entries[self.world_info[x]['folder']] if self.world_info[x]['folder'] != "root" else None,
                                            "num": x,
                                            "init": True,
                                            "selective": len(self.world_info[x]['keysecondary'])>0,
                                            "constant": self.world_info[x]['constant'],
                                            "uid": self.world_info[x]['uid'] if 'v1_uid' not in self.world_info[x] or self.world_info[x]['v1_uid'] is None else self.world_info[x]['v1_uid']
                                        } for x in self.world_info]
        
        #self.worldinfo   = []     # List of World Info key/value objects
        self.story_settings.worldinfo = [x for x in self.story_settings.worldinfo_i]
        #We have to have an uninitialized blank entry for every folder or the old method craps out
        for folder in folder_entries:
            self.story_settings.worldinfo.append({
                                            "key": "",
                                            "keysecondary": "",
                                            "content": "",
                                            "comment": "",
                                            "folder": folder_entries[folder] if folder != "root" else None,
                                            "num": (0 if len(self.world_info) == 0 else max(self.world_info))+(folder_entries[folder]*-1),
                                            "init": False,
                                            "selective": False,
                                            "constant": False,
                                            "uid": folder_entries[folder]
                                        })
        
        mapping = {uid: index for index, uid in enumerate(self.story_settings.wifolders_l)}
        self.story_settings.worldinfo.sort(key=lambda x: mapping[x["folder"]] if x["folder"] is not None else float("inf"))
        
        #self.wifolders_d = {}     # Dictionary of World Info folder UID-info pairs
        self.story_settings.wifolders_d = {folder_entries[x]: {'name': x, 'collapsed': False} for x in folder_entries if x != "root"}
        
        #self.worldinfo_u = {}     # Dictionary of World Info UID - key/value pairs
        self.story_settings.worldinfo_u = {y["uid"]: y for x in folder_entries for y in self.story_settings.worldinfo if y["folder"] == (folder_entries[x] if x != "root" else None)}
        
        #self.wifolders_u = {}     # Dictionary of pairs of folder UID - list of WI UID
        self.story_settings.wifolders_u = {folder_entries[x]: [y for y in self.story_settings.worldinfo if y['folder'] == folder_entries[x]] for x in folder_entries if x != "root"}
        
    def reset_used_in_game(self):
        for key in self.world_info:
            if self.world_info[key]["used_in_game"] != self.world_info[key]["constant"]:
                self.world_info[key]["used_in_game"] = self.world_info[key]["constant"]
                if self._socketio is not None:
                    self._socketio.emit("world_info_entry_used_in_game", {"uid": key, "used_in_game": False}, broadcast=True, room="UI_2")
        
    def set_world_info_used(self, uid):
        if uid in self.world_info:
            self.world_info[uid]["used_in_game"] = True
        else:
            logger.warning("Something tried to set world info UID {} to in game, but it doesn't exist".format(uid))
        if self._socketio is not None:
            self._socketio.emit("world_info_entry_used_in_game", {"uid": uid, "used_in_game": True}, broadcast=True, room="UI_2")
    
    def get_used_wi(self):
        return [x['content'] for x in self.world_info if x['used_in_game']]
    
    def to_wi_fewshot_format(self, excluding_uid: int) -> List[str]:
        """
        Returns a list of strings representing applicable (has title, text, and
        object type) World Info entries. Intended for feeding into the fewshot
        generator.
        """
        the_collection = []
        for entry in self.world_info.values():
            if entry["uid"] == excluding_uid:
                continue

            if not (
                entry["title"]
                and entry["manual_text"]
                and entry["object_type"]
            ):
                continue
                
            processed_desc = entry["manual_text"].replace("\n", " ")
            while "  " in processed_desc:
                processed_desc = processed_desc.replace("  ", " ")
            processed_desc = processed_desc.strip()

            the_collection.append(
                f"Title: {entry['title']}\n" \
                f"Type: {entry['object_type']}\n"
                f"Description: {processed_desc}"
            )
        
        return the_collection
    
    def get_commentators(self) -> List[dict]:
        ret = []
        for entry in self.world_info.values():
            if entry["type"] != "commentator":
                continue
            ret.append(entry)
        return ret


@dataclass
class SavePaths:
    base: str

    @property
    def required_paths(self) -> List[str]:
        return [
            self.base,
            self.generated_audio,
            self.generated_images,
            self.wi_images
        ]

    @property
    def story(self) -> str:
        return os.path.join(self.base, "story.json")

    @property
    def generated_audio(self) -> str:
        return os.path.join(self.base, "generated_audio")
    
    @property
    def generated_images(self) -> str:
        return os.path.join(self.base, "generated_images")

    @property
    def wi_images(self) -> str:
        return os.path.join(self.base, "wi_images")
   
default_rand_range = [0.44, 1, 2]
default_creativity_range = [0.5, 1]
default_rep_range = [1.0, 1.3]
default_preset = {
        "preset": "Default",
        "description": "Known working settings.",
        "Match": "Recommended",
        "Preset Category": "Official",
        "temp": 0.5,
        "genamt": 80,
        "top_k": 0,
        "top_p": 0.9,
        "top_a": 0.0,
        "typical": 1.0,
        "tfs": 1.0,
        "rep_pen": 1.1,
        "rep_pen_range": 1024,
        "rep_pen_slope": 0.7,
        "sampler_order": [
            6,
            0,
            1,
            2,
            3,
            4,
            5
        ]
    }
badwordsids_default = [[6880], [50256], [42496], [4613], [17414], [22039], [16410], [27], [29], [38430], [37922], [15913], [24618], [28725], [58], [47175], [36937], [26700], [12878], [16471], [37981], [5218], [29795], [13412], [45160], [3693], [49778], [4211], [20598], [36475], [33409], [44167], [32406], [29847], [29342], [42669], [685], [25787], [7359], [3784], [5320], [33994], [33490], [34516], [43734], [17635], [24293], [9959], [23785], [21737], [28401], [18161], [26358], [32509], [1279], [38155], [18189], [26894], [6927], [14610], [23834], [11037], [14631], [26933], [46904], [22330], [25915], [47934], [38214], [1875], [14692], [41832], [13163], [25970], [29565], [44926], [19841], [37250], [49029], [9609], [44438], [16791], [17816], [30109], [41888], [47527], [42924], [23984], [49074], [33717], [31161], [49082], [30138], [31175], [12240], [14804], [7131], [26076], [33250], [3556], [38381], [36338], [32756], [46581], [17912], [49146]] # Tokenized array of badwords used to prevent AI artifacting
badwordsids_neox = [[0], [1], [44162], [9502], [12520], [31841], [36320], [49824], [34417], [6038], [34494], [24815], [26635], [24345], [3455], [28905], [44270], [17278], [32666], [46880], [7086], [43189], [37322], [17778], [20879], [49821], [3138], [14490], [4681], [21391], [26786], [43134], [9336], [683], [48074], [41256], [19181], [29650], [28532], [36487], [45114], [46275], [16445], [15104], [11337], [1168], [5647], [29], [27482], [44965], [43782], [31011], [42944], [47389], [6334], [17548], [38329], [32044], [35487], [2239], [34761], [7444], [1084], [12399], [18990], [17636], [39083], [1184], [35830], [28365], [16731], [43467], [47744], [1138], [16079], [40116], [45564], [18297], [42368], [5456], [18022], [42696], [34476], [23505], [23741], [39334], [37944], [45382], [38709], [33440], [26077], [43600], [34418], [36033], [6660], [48167], [48471], [15775], [19884], [41533], [1008], [31053], [36692], [46576], [20095], [20629], [31759], [46410], [41000], [13488], [30952], [39258], [16160], [27655], [22367], [42767], [43736], [49694], [13811], [12004], [46768], [6257], [37471], [5264], [44153], [33805], [20977], [21083], [25416], [14277], [31096], [42041], [18331], [33376], [22372], [46294], [28379], [38475], [1656], [5204], [27075], [50001], [16616], [11396], [7748], [48744], [35402], [28120], [41512], [4207], [43144], [14767], [15640], [16595], [41305], [44479], [38958], [18474], [22734], [30522], [46267], [60], [13976], [31830], [48701], [39822], [9014], [21966], [31422], [28052], [34607], [2479], [3851], [32214], [44082], [45507], [3001], [34368], [34758], [13380], [38363], [4299], [46802], [30996], [12630], [49236], [7082], [8795], [5218], [44740], [9686], [9983], [45301], [27114], [40125], [1570], [26997], [544], [5290], [49193], [23781], [14193], [40000], [2947], [43781], [9102], [48064], [42274], [18772], [49384], [9884], [45635], [43521], [31258], [32056], [47686], [21760], [13143], [10148], [26119], [44308], [31379], [36399], [23983], [46694], [36134], [8562], [12977], [35117], [28591], [49021], [47093], [28653], [29013], [46468], [8605], [7254], [25896], [5032], [8168], [36893], [38270], [20499], [27501], [34419], [29547], [28571], [36586], [20871], [30537], [26842], [21375], [31148], [27618], [33094], [3291], [31789], [28391], [870], [9793], [41361], [47916], [27468], [43856], [8850], [35237], [15707], [47552], [2730], [41449], [45488], [3073], [49806], [21938], [24430], [22747], [20924], [46145], [20481], [20197], [8239], [28231], [17987], [42804], [47269], [29972], [49884], [21382], [46295], [36676], [34616], [3921], [26991], [27720], [46265], [654], [9855], [40354], [5291], [34904], [44342], [2470], [14598], [880], [19282], [2498], [24237], [21431], [16369], [8994], [44524], [45662], [13663], [37077], [1447], [37786], [30863], [42854], [1019], [20322], [4398], [12159], [44072], [48664], [31547], [18736], [9259], [31], [16354], [21810], [4357], [37982], [5064], [2033], [32871], [47446], [62], [22158], [37387], [8743], [47007], [17981], [11049], [4622], [37916], [36786], [35138], [29925], [14157], [18095], [27829], [1181], [22226], [5709], [4725], [30189], [37014], [1254], [11380], [42989], [696], [24576], [39487], [30119], [1092], [8088], [2194], [9899], [14412], [21828], [3725], [13544], [5180], [44679], [34398], [3891], [28739], [14219], [37594], [49550], [11326], [6904], [17266], [5749], [10174], [23405], [9955], [38271], [41018], [13011], [48392], [36784], [24254], [21687], [23734], [5413], [41447], [45472], [10122], [17555], [15830], [47384], [12084], [31350], [47940], [11661], [27988], [45443], [905], [49651], [16614], [34993], [6781], [30803], [35869], [8001], [41604], [28118], [46462], [46762], [16262], [17281], [5774], [10943], [5013], [18257], [6750], [4713], [3951], [11899], [38791], [16943], [37596], [9318], [18413], [40473], [13208], [16375]]
