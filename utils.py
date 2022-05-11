from threading import Timer, local
import re
import shutil
import json
import subprocess
import tempfile
import requests
import os

vars = None

#==================================================================#
# Decorator to prevent a function's actions from being run until
# at least x seconds have passed without the function being called
#==================================================================#
def debounce(wait): 
    def decorator(fun):
        def debounced(*args, **kwargs):
            def call_it():
                fun(*args, **kwargs)
 
            try:
                debounced.t.cancel()
            except AttributeError:
                pass
 
            debounced.t = Timer(wait, call_it)
            debounced.t.start()
 
        return debounced
 
    return decorator

#==================================================================#
# Replace fancy quotes and apostrope's with standard ones
#==================================================================#
def fixquotes(txt):
    txt = txt.replace("“", '"')
    txt = txt.replace("”", '"')
    txt = txt.replace("’", "'")
    txt = txt.replace("`", "'")
    return txt

#==================================================================#
# 
#==================================================================#
def trimincompletesentence(txt):
    # Cache length of text
    ln = len(txt)
    # Find last instance of punctuation (Borrowed from Clover-Edition by cloveranon)
    lastpunc = max(txt.rfind("."), txt.rfind("!"), txt.rfind("?"))
    # Is this the end of a quote?
    if(lastpunc < ln-1):
        if(txt[lastpunc+1] == '"'):
            lastpunc = lastpunc + 1
    if(lastpunc >= 0):
        txt = txt[:lastpunc+1]
    return txt

#==================================================================#
# 
#==================================================================#
def replaceblanklines(txt):
    txt = txt.replace("\n\n", "\n")
    return txt

#==================================================================#
# 
#==================================================================#
def removespecialchars(txt, vars=None):
    if vars is None or vars.actionmode == 0:
        txt = re.sub(r"[#/@%<>{}+=~|\^]", "", txt)
    else:
        txt = re.sub(r"[#/@%{}+=~|\^]", "", txt)
    return txt

#==================================================================#
# If the next action follows a sentence closure, add a space
#==================================================================#
def addsentencespacing(txt, vars):
    # Get last character of last action
    if(len(vars.actions) > 0):
        if(len(vars.actions[vars.actions.get_last_key()]) > 0):
            action = vars.actions[vars.actions.get_last_key()]
            lastchar = action[-1] if len(action) else ""
        else:
            # Last action is blank, this should never happen, but
            # since it did let's bail out.
            return txt
    else:
        action = vars.prompt
        lastchar = action[-1] if len(action) else ""
    if(lastchar == "." or lastchar == "!" or lastchar == "?" or lastchar == "," or lastchar == ";" or lastchar == ":"):
        txt = " " + txt
    return txt
	
def singlelineprocessing(txt, vars):
    txt = vars.regex_sl.sub('', txt)
    if(len(vars.actions) > 0):
        if(len(vars.actions[vars.actions.get_last_key()]) > 0):
            action = vars.actions[vars.actions.get_last_key()]
            lastchar = action[-1] if len(action) else ""
        else:
            # Last action is blank, this should never happen, but
            # since it did let's bail out.
            return txt
    else:
        action = vars.prompt
        lastchar = action[-1] if len(action) else ""
    if(lastchar != "\n"):
        txt = txt + "\n"
    return txt

#==================================================================#
#  Cleans string for use in file name
#==================================================================#
def cleanfilename(filename):
    filteredcharacters = ('/','\\')
    filename = "".join(c for c in filename if c not in filteredcharacters).rstrip()
    return filename
    
#==================================================================#
#  Newline substitution for fairseq models
#==================================================================#
def encodenewlines(txt):
    if(vars.newlinemode == "s"):
        return txt.replace('\n', "</s>")
    return txt

def decodenewlines(txt):
    if(vars.newlinemode == "s"):
        return txt.replace("</s>", '\n')
    return txt

#==================================================================#
#  Downloads sharded huggingface checkpoints using aria2c if possible
#==================================================================#
def aria2_hook(pretrained_model_name_or_path: str, force_download=False, cache_dir=None, proxies=None, resume_download=False, local_files_only=False, use_auth_token=None, user_agent=None, revision=None, mirror=None, **kwargs):
    import transformers
    import transformers.modeling_utils
    from huggingface_hub import HfFolder
    if shutil.which("aria2c") is None:  # Don't do anything if aria2 is not installed
        return
    if local_files_only:  # If local_files_only is true, we obviously don't need to download anything
        return
    if os.path.isdir(pretrained_model_name_or_path) or os.path.isfile(pretrained_model_name_or_path) or os.path.isfile(pretrained_model_name_or_path + ".index") or transformers.modeling_utils.is_remote_url(pretrained_model_name_or_path):
        return
    if proxies:
        print("WARNING:  KoboldAI does not support using aria2 to download models from huggingface.co through a proxy.  Disabling aria2 download mode.")
        return
    if use_auth_token:
        if isinstance(use_auth_token, str):
            token = use_auth_token
        else:
            token = HfFolder.get_token()
            if token is None:
                raise EnvironmentError("You specified use_auth_token=True, but a huggingface token was not found.")
    _cache_dir = str(cache_dir) if cache_dir is not None else transformers.TRANSFORMERS_CACHE
    sharded = False
    headers = {"user-agent": transformers.file_utils.http_user_agent(user_agent)}
    if use_auth_token:
        headers["authorization"] = f"Bearer {use_auth_token}"
    def is_cached(url):
        try:
            transformers.file_utils.get_from_cache(url, cache_dir=cache_dir, local_files_only=True)
        except FileNotFoundError:
            return False
        return True
    while True:  # Try to get the huggingface.co URL of the model's pytorch_model.bin or pytorch_model.bin.index.json file
        try:
            filename = transformers.modeling_utils.WEIGHTS_INDEX_NAME if sharded else transformers.modeling_utils.WEIGHTS_NAME
        except AttributeError:
            return
        url = transformers.file_utils.hf_bucket_url(pretrained_model_name_or_path, filename, revision=revision, mirror=mirror)
        if is_cached(url) or requests.head(url, allow_redirects=True, proxies=proxies, headers=headers):
            break
        if sharded:
            return
        else:
            sharded = True
    if not sharded:  # If the model has a pytorch_model.bin file, that's the only file to download
        filenames = [transformers.modeling_utils.WEIGHTS_NAME]
    else:  # Otherwise download the pytorch_model.bin.index.json and then let aria2 download all the pytorch_model-#####-of-#####.bin files mentioned inside it
        map_filename = transformers.file_utils.cached_path(url, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, use_auth_token=use_auth_token, user_agent=user_agent)
        with open(map_filename) as f:
            map_data = json.load(f)
        filenames = set(map_data["weight_map"].values())
    urls = [transformers.file_utils.hf_bucket_url(pretrained_model_name_or_path, n, revision=revision, mirror=mirror) for n in filenames]
    if not force_download:
        if all(is_cached(u) for u in urls):
            return
        elif local_files_only:
            raise FileNotFoundError("Cannot find the requested files in the cached path and outgoing traffic has been disabled. To enable model look-ups and downloads online, set 'local_files_only' to False.")
    etags = [h.get("X-Linked-Etag") or h.get("ETag") for u in urls for h in [requests.head(u, headers=headers, allow_redirects=False, proxies=proxies, timeout=10).headers]]
    filenames = [transformers.file_utils.url_to_filename(u, t) for u, t in zip(urls, etags)]
    if force_download:
        for n in filenames:
            path = os.path.join(_cache_dir, n + ".json")
            if os.path.exists(path):
                os.remove(path)
    aria2_config = "\n".join(f"{u}\n  out={n}" for u, n in zip(urls, filenames)).encode()
    with tempfile.NamedTemporaryFile("w+b", delete=False) as f:
        f.write(aria2_config)
        f.flush()
        p = subprocess.Popen(["aria2c", "-x", "10", "-s", "10", "-j", "10", "--disable-ipv6", "--file-allocation=none", "-d", _cache_dir, "-i", f.name, "-U", transformers.file_utils.http_user_agent(user_agent)] + (["-c"] if not force_download else []) + ([f"--header='Authorization: Bearer {token}'"] if use_auth_token else []), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            print(line.decode(), end="", flush=True)
        path = f.name
    try:
        os.remove(path)
    except OSError:
        pass
    for u, t, n in zip(urls, etags, filenames):
        with open(os.path.join(_cache_dir, n + ".json"), "w") as f:
            json.dump({"url": u, "etag": t}, f)
