from os import getcwd, listdir, path
from typing import Tuple, Union, Optional
import os
import json
import zipfile
from logger import logger

#==================================================================#
#  Generic Method for prompting for file path
#==================================================================#
def getsavepath(dir, title, types):
    import tkinter as tk
    from tkinter import filedialog
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
    import tkinter as tk
    from tkinter import filedialog
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
    import tkinter as tk
    from tkinter import filedialog
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
    return path.join("stories", name + ".json")

#==================================================================#
#  Returns the path (as a string) to the given soft prompt by its filename
#==================================================================#
def sppath(filename):
    return path.join("softprompts", filename)

#==================================================================#
#  Returns the path (as a string) to the given username by its filename
#==================================================================#
def uspath(filename):
    return path.join("userscripts", filename)

#==================================================================#
#  Returns an array of dicts containing story files in /stories
#==================================================================#
def getstoryfiles():
    list = []
    for file in listdir("stories"):
        if file.endswith(".json") and not file.endswith(".v2.json"):
            ob = {}
            ob["name"] = file.replace(".json", "")
            f = open("stories/"+file, "r")
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
#  Checks if the given soft prompt file is valid
#==================================================================#
def checksp(filename: str, model_dimension: int) -> Tuple[Union[zipfile.ZipFile, int], Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[bool], Optional['np.dtype']]:
    global np
    if 'np' not in globals():
        import numpy as np
    try:
        z = zipfile.ZipFile("softprompts/"+filename)
        with z.open('tensor.npy') as f:
            # Read only the header of the npy file, for efficiency reasons
            version: Tuple[int, int] = np.lib.format.read_magic(f)
            shape: Tuple[int, int]
            fortran_order: bool
            dtype: np.dtype
            shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
            assert len(shape) == 2
    except:
        try:
            z.close()
        except UnboundLocalError:
            pass
        return 1, None, None, None, None
    if dtype not in ('V2', np.float16, np.float32):
        z.close()
        return 2, version, shape, fortran_order, dtype
    if shape[1] != model_dimension:
        z.close()
        return 3, version, shape, fortran_order, dtype
    if shape[0] >= 2048:
        z.close()
        return 4, version, shape, fortran_order, dtype
    return z, version, shape, fortran_order, dtype

#==================================================================#
#  Returns an array of dicts containing softprompt files in /softprompts
#==================================================================#
def getspfiles(model_dimension: int):
    lst = []
    os.makedirs("softprompts", exist_ok=True)
    for file in listdir("softprompts"):
        if not file.endswith(".zip"):
            continue
        z, version, shape, fortran_order, dtype = checksp(file, model_dimension)
        if z == 1:
            logger.warning(f"Softprompt {file} is malformed or not a soft prompt ZIP file.")
            continue
        if z == 2:
            logger.warning(f"Softprompt {file} tensor.npy has unsupported dtype '{dtype.name}'.")
            continue
        if z == 3:
            logger.debug(f"Softprompt {file} tensor.npy has model dimension {shape[1]} which does not match your model's model dimension of {model_dimension}. This usually means this soft prompt is not compatible with your model.")
            continue
        if z == 4:
            logger.warning(f"Softprompt {file} tensor.npy has {shape[0]} tokens but it is supposed to have less than 2048 tokens.")
            continue
        assert isinstance(z, zipfile.ZipFile)
        try:
            with z.open('meta.json') as f:
                ob = json.load(f)
        except:
            ob = {}
        z.close()
        ob["filename"] = file
        ob["n_tokens"] = shape[-2]
        lst.append(ob)
    return lst

#==================================================================#
#  Returns an array of dicts containing userscript files in /userscripts
#==================================================================#
def getusfiles(long_desc=False):
    lst = []
    os.makedirs("userscripts", exist_ok=True)
    for file in listdir("userscripts"):
        if file.endswith(".lua"):
            ob = {}
            ob["filename"] = file
            description = []
            multiline = False
            with open(uspath(file)) as f:
                ob["modulename"] = f.readline().strip().replace("\033", "")
                if ob["modulename"][:2] != "--":
                    ob["modulename"] = file
                else:
                    ob["modulename"] = ob["modulename"][2:]
                    if ob["modulename"][:2] == "[[":
                        ob["modulename"] = ob["modulename"][2:]
                        multiline = True
                    ob["modulename"] = ob["modulename"].lstrip("-").strip()
                    for line in f:
                        line = line.strip().replace("\033", "")
                        if multiline:
                            index = line.find("]]")
                            if index > -1:
                                description.append(line[:index])
                                if index != len(line) - 2:
                                    break
                                multiline = False
                            else:
                                description.append(line)
                        else:
                            if line[:2] != "--":
                                break
                            line = line[2:]
                            if line[:2] == "[[":
                                multiline = True
                                line = line[2:]
                            description.append(line.strip())
            ob["description"] = "\n".join(description)
            if not long_desc:
                if len(ob["description"]) > 250:
                    ob["description"] = ob["description"][:247] + "..."
            lst.append(ob)
    return lst

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
