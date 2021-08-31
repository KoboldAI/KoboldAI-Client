import tkinter as tk
from tkinter import filedialog
from os import getcwd, listdir, path
import os
import json
import string

#==================================================================#
#  Generic Method for prompting for file path
#==================================================================#
def getsavepath(dir, title, types):
    root = tk.Tk()
    root.attributes("-topmost", True)
    path = tk.filedialog.asksaveasfile(
        initialdir=dir, 
        title=title, 
        filetypes = types,
        defaultextension="*.*"
        )
    root.destroy()
    if(path != "" and path != None):
        return path.name
    else:
        return None

#==================================================================#
#  Generic Method for prompting for file path
#==================================================================#
def getloadpath(dir, title, types):
    root = tk.Tk()
    root.attributes("-topmost", True)
    path = tk.filedialog.askopenfilename(
        initialdir=dir, 
        title=title, 
        filetypes = types
        )
    root.destroy()
    if(path != "" and path != None):
        return path
    else:
        return None

#==================================================================#
#  Generic Method for prompting for directory path
#==================================================================#
def getdirpath(dir, title):
    root = tk.Tk()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(
        initialdir=dir, 
        title=title
        )
    root.destroy()
    if(path != "" and path != None):
        return path
    else:
        return None

#==================================================================#
#  Returns the path (as a string) to the given story by its name
#==================================================================#
def storypath(name):
    return path.join(path.dirname(path.realpath(__file__)), "stories", name + ".json")

#==================================================================#
#  Returns an array of dicts containing story files in /stories
#==================================================================#
def getstoryfiles():
    list = []
    for file in listdir(path.dirname(path.realpath(__file__))+"/stories"):
        if file.endswith(".json"):
            ob = {}
            ob["name"] = file.replace(".json", "")
            f = open(path.dirname(path.realpath(__file__))+"/stories/"+file, "r")
            try:
                js = json.load(f)
            except:
                print(f"Browser loading error: {file} is malformed or not a JSON file.")
                f.close()
                continue
            f.close()
            try:
                ob["actions"] = len(js["actions"])
            except TypeError:
                print(f"Browser loading error: {file} has incorrect format.")
                continue
            list.append(ob)
    return list

#==================================================================#
#  Returns True if json file exists with requested save name
#==================================================================#
def saveexists(name):
    return path.exists(storypath(name))

#==================================================================#
#  Delete save file by name; returns None if successful, or the exception if not
#==================================================================#
def deletesave(name):
    try:
        os.remove(storypath(name))
    except Exception as e:
        return e

#==================================================================#
#  Rename save file; returns None if successful, or the exception if not
#==================================================================#
def renamesave(name, new_name):
    try:
        os.replace(storypath(name), storypath(new_name))
    except Exception as e:
        return e
