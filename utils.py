from threading import Timer
import re
import shutil
import json
import subprocess
import tempfile
import requests
import requests.adapters
import time
from tqdm.auto import tqdm
import os
import itertools
import hashlib
import huggingface_hub
from typing import Optional

vars = None
num_shards: Optional[int] = None
current_shard = 0
from_pretrained_model_name = ""
from_pretrained_index_filename: Optional[str] = None
from_pretrained_kwargs = {}
bar = None

default_sampler_order = [0, 1, 2, 3, 4, 5]

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
    # Don't add sentence spacing if submission is empty or starts with whitespace
    if(len(txt) == 0 or len(txt) != len(txt.lstrip())):
        return txt
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
    if(vars.newlinemode == "ns"):
        return txt.replace("</s>", '')
    return txt

#==================================================================#
#  Returns number of layers given an HF model config
#==================================================================#
def num_layers(config):
    return config.num_layers if hasattr(config, "num_layers") else config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers

#==================================================================#
#  Downloads huggingface checkpoints using aria2c if possible
#==================================================================#
def aria2_hook(pretrained_model_name_or_path: str, force_download=False, cache_dir=None, proxies=None, resume_download=False, local_files_only=False, use_auth_token=None, user_agent=None, revision=None, **kwargs):
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
            huggingface_hub.cached_download(url, cache_dir=cache_dir, local_files_only=True)
        except ValueError:
            return False
        return True
    while True:  # Try to get the huggingface.co URL of the model's pytorch_model.bin or pytorch_model.bin.index.json file
        try:
            filename = transformers.modeling_utils.WEIGHTS_INDEX_NAME if sharded else transformers.modeling_utils.WEIGHTS_NAME
        except AttributeError:
            return
        url = huggingface_hub.hf_hub_url(pretrained_model_name_or_path, filename, revision=revision)
        if is_cached(url) or requests.head(url, allow_redirects=True, proxies=proxies, headers=headers):
            break
        if sharded:
            return
        else:
            sharded = True
    if not sharded:  # If the model has a pytorch_model.bin file, that's the only file to download
        filenames = [transformers.modeling_utils.WEIGHTS_NAME]
    else:  # Otherwise download the pytorch_model.bin.index.json and then let aria2 download all the pytorch_model-#####-of-#####.bin files mentioned inside it
        map_filename = huggingface_hub.cached_download(url, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, use_auth_token=use_auth_token, user_agent=user_agent)
        with open(map_filename) as f:
            map_data = json.load(f)
        filenames = set(map_data["weight_map"].values())
    urls = [huggingface_hub.hf_hub_url(pretrained_model_name_or_path, n, revision=revision) for n in filenames]
    if not force_download:
        urls = [u for u in urls if not is_cached(u)]
        if not urls:
            return
    etags = [h.get("X-Linked-Etag") or h.get("ETag") for u in urls for h in [requests.head(u, headers=headers, allow_redirects=False, proxies=proxies, timeout=10).headers]]
    headers = [requests.head(u, headers=headers, allow_redirects=True, proxies=proxies, timeout=10).headers for u in urls]
    filenames = [hashlib.sha256(u.encode("utf-8")).hexdigest() + "." + hashlib.sha256(t.encode("utf-8")).hexdigest() for u, t in zip(urls, etags)]
    for n in filenames:
        path = os.path.join(_cache_dir, "kai-tempfile." + n + ".aria2")
        if os.path.exists(path):
            os.remove(path)
        path = os.path.join(_cache_dir, "kai-tempfile." + n)
        if os.path.exists(path):
            os.remove(path)
        if force_download:
            path = os.path.join(_cache_dir, n + ".json")
            if os.path.exists(path):
                os.remove(path)
            path = os.path.join(_cache_dir, n)
            if os.path.exists(path):
                os.remove(path)
    total_length = sum(int(h["Content-Length"]) for h in headers)
    lengths = {}
    aria2_config = "\n".join(f"{u}\n  out=kai-tempfile.{n}" for u, n in zip(urls, filenames)).encode()
    s = requests.Session()
    s.mount("http://", requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(total=120, backoff_factor=1)))
    bar = None
    done = False
    secret = os.urandom(17).hex()
    try:
        with tempfile.NamedTemporaryFile("w+b", delete=False) as f:
            f.write(aria2_config)
            f.flush()
            p = subprocess.Popen(["aria2c", "-x", "10", "-s", "10", "-j", "10", "--enable-rpc=true", f"--rpc-secret={secret}", "--rpc-listen-port", str(vars.aria2_port), "--disable-ipv6", "--file-allocation=trunc", "--allow-overwrite", "--auto-file-renaming=false", "-d", _cache_dir, "-i", f.name, "-U", transformers.file_utils.http_user_agent(user_agent)] + (["-c"] if not force_download else []) + ([f"--header='Authorization: Bearer {token}'"] if use_auth_token else []), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while p.poll() is None:
                r = s.post(f"http://localhost:{vars.aria2_port}/jsonrpc", json={"jsonrpc": "2.0", "id": "kai", "method": "aria2.tellActive", "params": [f"token:{secret}"]}).json()["result"]
                if not r:
                    s.close()
                    if bar is not None:
                        bar.n = bar.total
                        bar.close()
                    p.terminate()
                    done = True
                    break
                if bar is None:
                    bar = tqdm(total=total_length, desc=f"[aria2] Downloading model", unit="B", unit_scale=True, unit_divisor=1000)
                visited = set()
                for x in r:
                    filename = x["files"][0]["path"]
                    lengths[filename] = (int(x["completedLength"]), int(x["totalLength"]))
                    visited.add(filename)
                for k, v in lengths.items():
                    if k not in visited:
                        lengths[k] = (v[1], v[1])
                bar.n = sum(v[0] for v in lengths.values())
                bar.update()
                time.sleep(0.1)
            path = f.name
    except Exception as e:
        p.terminate()
        raise e
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    code = p.wait()
    if not done and code:
        raise OSError(f"aria2 exited with exit code {code}")
    for u, t, n in zip(urls, etags, filenames):
        os.rename(os.path.join(_cache_dir, "kai-tempfile." + n), os.path.join(_cache_dir, n))
        with open(os.path.join(_cache_dir, n + ".json"), "w") as f:
            json.dump({"url": u, "etag": t}, f)

#==================================================================#
#  Given the path to a pytorch_model.bin.index.json, returns how many
#  shards there are in the model
#==================================================================#
def get_num_shards(filename):
    with open(filename) as f:
        map_data = json.load(f)
    return len(set(map_data["weight_map"].values()))

#==================================================================#
#  Given the name/path of a sharded model and the path to a
#  pytorch_model.bin.index.json, returns a list of weight names in the
#  sharded model.  Requires lazy loader to be enabled to work properl
#==================================================================#
def get_sharded_checkpoint_num_tensors(pretrained_model_name_or_path, filename, cache_dir=None, force_download=False, proxies=None, resume_download=False, local_files_only=False, use_auth_token=None, user_agent=None, revision=None, **kwargs):
    import transformers.modeling_utils
    import torch
    shard_paths, _ = transformers.modeling_utils.get_checkpoint_shard_files(pretrained_model_name_or_path, filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, use_auth_token=use_auth_token, user_agent=user_agent, revision=revision)
    return list(itertools.chain(*(torch.load(p, map_location="cpu").keys() for p in shard_paths)))
