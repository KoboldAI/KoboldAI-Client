import tkinter as tk
from tkinter import filedialog

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