from threading import Timer
import re
import shutil
import json
import subprocess
import tempfile
from urllib.error import HTTPError
import requests
import requests.adapters
import time
from transformers import PreTrainedModel
import packaging.version
from tqdm.auto import tqdm
import os
import itertools
import hashlib
import huggingface_hub
import packaging.version
from pathlib import Path
from typing import List, Optional

koboldai_vars = None
args = None
num_shards: Optional[int] = None
current_shard = 0
from_pretrained_model_name = ""
from_pretrained_index_filename: Optional[str] = None
from_pretrained_kwargs = {}
bar = None

layers_module_names: Optional[List[str]] = None
module_names: Optional[List[str]] = None
named_buffers: Optional[List[tuple]] = None

default_sampler_order = [6, 0, 1, 2, 3, 4, 5]

emit = None

# Hack for socket stuff that needs app context
flask_app = None

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
    return txt.replace("\n\n", "\n")

#==================================================================#
# 
#==================================================================#
def removespecialchars(txt, koboldai_vars=None):
    if koboldai_vars is None or koboldai_vars.actionmode == 0:
        txt = re.sub(r"[/@%<>{}+=~|\^]", "", txt)
    else:
        txt = re.sub(r"[/@%{}+=~|\^]", "", txt)
    return txt

#==================================================================#
# If the next action follows a sentence closure, add a space
#==================================================================#
def addsentencespacing(txt, koboldai_vars):
    # Don't add sentence spacing if submission is empty or starts with whitespace
    if(len(txt) == 0 or len(txt) != len(txt.lstrip())):
        return txt
    # Get last character of last action
    if(len(koboldai_vars.actions) > 0):
        if(len(koboldai_vars.actions[koboldai_vars.actions.get_last_key()]) > 0):
            action = koboldai_vars.actions[koboldai_vars.actions.get_last_key()]
            lastchar = action[-1] if len(action) else ""
        else:
            # Last action is blank, this should never happen, but
            # since it did let's bail out.
            return txt
    else:
        action = koboldai_vars.prompt
        lastchar = action[-1] if len(action) else ""
    if(lastchar != " "):
        txt = " " + txt
    return txt
	
def singlelineprocessing(txt, koboldai_vars):
    txt = koboldai_vars.regex_sl.sub('', txt)
    if(len(koboldai_vars.actions) > 0):
        if(len(koboldai_vars.actions[-1]) > 0):
            action = koboldai_vars.actions[-1]
            lastchar = action[-1] if len(action) else ""
        else:
            # Last action is blank, this should never happen, but
            # since it did let's bail out.
            return txt
    else:
        action = koboldai_vars.prompt
        lastchar = action[-1] if len(action) else ""
    if(lastchar != "\n"):
        txt = txt + "\n"
    return txt

def chatmodeprocessing(txt, koboldai_vars):
    chatregex = re.compile(r'\s+%s:[.|\n|\W|\w]*'%koboldai_vars.chatname)
    txt = chatregex.sub('', txt)
    if(len(koboldai_vars.actions) > 0):
        if(len(koboldai_vars.actions[-1]) > 0):
            action = koboldai_vars.actions[-1]
        else:
            # Last action is blank, this should never happen, but
            # since it did let's bail out.
            return txt
    else:
        action = koboldai_vars.prompt
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
    if(koboldai_vars.newlinemode == "s"):
        return txt.replace('\n', "</s>")
    return txt

def decodenewlines(txt):
    if(koboldai_vars.newlinemode == "s"):
        return txt.replace("</s>", '\n')
    if(koboldai_vars.newlinemode == "ns"):
        return txt.replace("</s>", '')
    return txt

#==================================================================#
#  Returns number of layers given an HF model config
#==================================================================#
def num_layers(config):
    return config["n_layer"] if isinstance(config, dict) else config.num_layers if hasattr(config, "num_layers") else config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else config.n_layers if hasattr(config, "n_layers") else None

#==================================================================#
#  Downloads huggingface checkpoints using aria2c if possible
#==================================================================#
from flask_socketio import emit
            
def _download_with_aria2(aria2_config: str, total_length: int, directory: str = ".", user_agent=None, force_download=False, use_auth_token=None):
    import transformers
    aria2_port = 6799 if koboldai_vars is None else koboldai_vars.aria2_port
    lengths = {}
    path = None
    s = requests.Session()
    s.mount("http://", requests.adapters.HTTPAdapter(max_retries=requests.adapters.Retry(total=120, backoff_factor=1)))
    bar = None
    done = False
    secret = os.urandom(17).hex()
    try:
        with tempfile.NamedTemporaryFile("w+b", delete=False) as f:
            f.write(aria2_config)
            f.flush()
            p = subprocess.Popen(["aria2c", "-x", "10", "-s", "10", "-j", "10", "--enable-rpc=true", f"--rpc-secret={secret}", "--rpc-listen-port", str(aria2_port), "--disable-ipv6", "--file-allocation=trunc", "--allow-overwrite", "--auto-file-renaming=false", "-d", directory, "-i", f.name, "-U", transformers.file_utils.http_user_agent(user_agent)] + (["-c"] if not force_download else []) + ([f"--header='Authorization: Bearer {use_auth_token}'"] if use_auth_token else []), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while p.poll() is None:
                r = s.post(f"http://localhost:{aria2_port}/jsonrpc", json={"jsonrpc": "2.0", "id": "kai", "method": "aria2.tellActive", "params": [f"token:{secret}"]}).json()["result"]
                if not r:
                    s.close()
                    if bar is not None:
                        bar.n = bar.total
                        bar.close()
                        koboldai_vars.downloaded_chunks = bar.total
                    p.terminate()
                    done = True
                    break
                visited = set()
                for x in r:
                    filename = x["files"][0]["path"]
                    lengths[filename] = (int(x["completedLength"]), int(x["totalLength"]))
                    visited.add(filename)
                for k, v in lengths.items():
                    if k not in visited:
                        lengths[k] = (v[1], v[1])
                if bar is None:
                    bar = tqdm(total=total_length, desc=f"[aria2] Downloading model", unit="B", unit_scale=True, unit_divisor=1000, file=UIProgressBarFile())
                    koboldai_vars.status_message = "Download Model"
                    koboldai_vars.total_download_chunks = sum(v[1] for v in lengths.values())
                koboldai_vars.downloaded_chunks = sum(v[0] for v in lengths.values())
                bar.n = koboldai_vars.downloaded_chunks
                bar.update()
                time.sleep(0.1)
            koboldai_vars.status_message = ""
            path = f.name
    except Exception as e:
        p.terminate()
        raise e
    finally:
        try:
            if path is not None:
                if os.path.exists(path):
                    os.remove(path)
        except OSError:
            pass
    code = p.wait()
    if not done and code:
        raise OSError(f"aria2 exited with exit code {code}")

def _transformers22_aria2_hook(pretrained_model_name_or_path: str, force_download=False, cache_dir=None, proxies=None, resume_download=False, local_files_only=False, use_auth_token=None, user_agent=None, revision=None, **kwargs):
    import transformers
    import transformers.modeling_utils
    from huggingface_hub import HfFolder
    if use_auth_token:
        if isinstance(use_auth_token, str):
            token = use_auth_token
        else:
            token = HfFolder.get_token()
            if token is None:
                raise EnvironmentError("You specified use_auth_token=True, but a huggingface token was not found.")
    _cache_dir = str(cache_dir) if cache_dir is not None else transformers.TRANSFORMERS_CACHE
    _revision = koboldai_vars.revision if koboldai_vars.revision is not None else huggingface_hub.constants.DEFAULT_REVISION
    headers = {"user-agent": transformers.file_utils.http_user_agent(user_agent)}
    if use_auth_token:
        headers["authorization"] = f"Bearer {use_auth_token}"
    storage_folder = os.path.join(_cache_dir, huggingface_hub.file_download.repo_folder_name(repo_id=pretrained_model_name_or_path, repo_type="model"))
    os.makedirs(storage_folder, exist_ok=True)

    def is_cached(filename):
        try:
            huggingface_hub.hf_hub_download(pretrained_model_name_or_path, filename, cache_dir=cache_dir, local_files_only=True, revision=revision)
        except ValueError:
            return False
        return True

    filename = None

    # NOTE: For now sharded Safetensors models are not supported. Haven't seen
    # one of these out in the wild yet, probably due to how Safetensors has a
    # lot of benifits of sharding built in
    for possible_filename in [
        transformers.modeling_utils.SAFE_WEIGHTS_INDEX_NAME,
        transformers.modeling_utils.SAFE_WEIGHTS_NAME,
        transformers.modeling_utils.WEIGHTS_INDEX_NAME,
        transformers.modeling_utils.WEIGHTS_NAME
    ]:
        # Try to get the huggingface.co URL of the model's weights file(s)
        url = huggingface_hub.hf_hub_url(pretrained_model_name_or_path, possible_filename, revision=_revision)

        if is_cached(possible_filename) or requests.head(url, allow_redirects=True, proxies=proxies, headers=headers):
            filename = possible_filename
            break

    if not filename:
        return

    if filename not in [
        transformers.modeling_utils.SAFE_WEIGHTS_INDEX_NAME,
        transformers.modeling_utils.WEIGHTS_INDEX_NAME
    ]:
        # If the model isn't sharded, theres only one file to download
        filenames = [filename]
    else:
        # Otherwise download the pytorch_model.bin.index.json and then let aria2 download all the pytorch_model-#####-of-#####.bin files mentioned inside it
        map_filename = huggingface_hub.hf_hub_download(pretrained_model_name_or_path, filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, use_auth_token=use_auth_token, user_agent=user_agent, revision=revision)
        with open(map_filename) as f:
            map_data = json.load(f)
        filenames = set(map_data["weight_map"].values())

    urls = [huggingface_hub.hf_hub_url(pretrained_model_name_or_path, n, revision=_revision) for n in filenames]
    if not force_download:
        urls = [u for u, n in zip(urls, filenames) if not is_cached(n)]
        if not urls:
            return
    
    blob_paths = []

    # This section is a modified version of hf_hub_download from huggingface_hub
    # See https://github.com/huggingface/huggingface_hub/blob/main/LICENSE for license
    for u, n in zip(urls, filenames):
        relative_filename = os.path.join(*n.split("/"))
        if not local_files_only:
            try:
                r = huggingface_hub.file_download._request_wrapper(
                    method="HEAD",
                    url=u,
                    headers=headers,
                    allow_redirects=False,
                    follow_relative_redirects=True,
                    proxies=proxies,
                    timeout=10,
                )
                try:
                    r.raise_for_status()
                except HTTPError as e:
                    error_code = r.headers.get("X-Error-Code")
                    if error_code != "EntryNotFound":
                        raise RuntimeError(f"HEAD {u} failed with error code {r.status_code}")
                    commit_hash = r.headers.get(huggingface_hub.file_download.HUGGINGFACE_HEADER_X_REPO_COMMIT)
                    if commit_hash is not None:
                        no_exist_file_path = (
                            Path(storage_folder)
                            / ".no_exist"
                            / commit_hash
                            / relative_filename
                        )
                        no_exist_file_path.parent.mkdir(parents=True, exist_ok=True)
                        no_exist_file_path.touch()
                        huggingface_hub.file_download._cache_commit_hash_for_specific_revision(
                            storage_folder, _revision, commit_hash
                        )
                    raise
                commit_hash = r.headers[huggingface_hub.file_download.HUGGINGFACE_HEADER_X_REPO_COMMIT]
                if commit_hash is None:
                    raise OSError(
                        "Distant resource does not seem to be on huggingface.co (missing"
                        " commit header)."
                    )
                etag = r.headers.get(huggingface_hub.file_download.HUGGINGFACE_HEADER_X_LINKED_ETAG) or r.headers.get(
                    "ETag"
                )
                # We favor a custom header indicating the etag of the linked resource, and
                # we fallback to the regular etag header.
                # If we don't have any of those, raise an error.
                if etag is None:
                    raise OSError(
                        "Distant resource does not have an ETag, we won't be able to"
                        " reliably ensure reproducibility."
                    )
                etag = huggingface_hub.file_download._normalize_etag(etag)
                # In case of a redirect, save an extra redirect on the request.get call,
                # and ensure we download the exact atomic version even if it changed
                # between the HEAD and the GET (unlikely, but hey).
                # Useful for lfs blobs that are stored on a CDN.
                if 300 <= r.status_code <= 399:
                    url_to_download = r.headers["Location"]
                    if (
                        "lfs.huggingface.co" in url_to_download
                        or "lfs-staging.huggingface.co" in url_to_download
                    ):
                        # Remove authorization header when downloading a LFS blob
                        headers.pop("authorization", None)
            except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
                # Actually raise for those subclasses of ConnectionError
                raise
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                huggingface_hub.file_download.OfflineModeIsEnabled,
            ):
                # Otherwise, our Internet connection is down.
                # etag is None
                pass
        if etag is None:
            # In those cases, we cannot force download.
            if force_download:
                raise ValueError(
                    "We have no connection or you passed local_files_only, so"
                    " force_download is not an accepted option."
                )
            if huggingface_hub.file_download.REGEX_COMMIT_HASH.match(_revision):
                commit_hash = _revision
            else:
                ref_path = os.path.join(storage_folder, "refs", _revision)
                with open(ref_path) as f:
                    commit_hash = f.read()
            pointer_path = os.path.join(
                storage_folder, "snapshots", commit_hash, relative_filename
            )
            if os.path.exists(pointer_path):
                return pointer_path
            # If we couldn't find an appropriate file on disk,
            # raise an error.
            # If files cannot be found and local_files_only=True,
            # the models might've been found if local_files_only=False
            # Notify the user about that
            if local_files_only:
                raise huggingface_hub.file_download.LocalEntryNotFoundError(
                    "Cannot find the requested files in the disk cache and"
                    " outgoing traffic has been disabled. To enable hf.co look-ups"
                    " and downloads online, set 'local_files_only' to False."
                )
            else:
                raise huggingface_hub.file_download.LocalEntryNotFoundError(
                    "Connection error, and we cannot find the requested files in"
                    " the disk cache. Please try again or make sure your Internet"
                    " connection is on."
                )
        # From now on, etag and commit_hash are not None.
        blob_path = os.path.join(storage_folder, "blobs", etag)
        pointer_path = os.path.join(
            storage_folder, "snapshots", commit_hash, relative_filename
        )
        os.makedirs(os.path.dirname(blob_path), exist_ok=True)
        os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
        # if passed revision is not identical to commit_hash
        # then revision has to be a branch name or tag name.
        # In that case store a ref.
        huggingface_hub.file_download._cache_commit_hash_for_specific_revision(storage_folder, _revision, commit_hash)
        if os.path.exists(pointer_path) and not force_download:
            return pointer_path
        if os.path.exists(blob_path) and not force_download:
            # we have the blob already, but not the pointer
            huggingface_hub.file_download.logger.info("creating pointer to %s from %s", blob_path, pointer_path)
            huggingface_hub.file_download._create_relative_symlink(blob_path, pointer_path)
            return pointer_path
        # Some Windows versions do not allow for paths longer than 255 characters.
        # In this case, we must specify it is an extended path by using the "\\?\" prefix.
        if os.name == "nt" and len(os.path.abspath(blob_path)) > 255:
            blob_path = "\\\\?\\" + os.path.abspath(blob_path)
        blob_paths.append(blob_path)

    filenames = blob_paths
    headers = [requests.head(u, headers=headers, allow_redirects=True, proxies=proxies, timeout=10).headers for u in urls]

    for n in filenames:
        prefix, suffix = n.rsplit(os.sep, 1)
        path = os.path.join(prefix, "kai-tempfile." + suffix + ".aria2")
        if os.path.exists(path):
            os.remove(path)
        path = os.path.join(prefix, "kai-tempfile." + suffix)
        if os.path.exists(path):
            os.remove(path)
    total_length = sum(int(h["Content-Length"]) for h in headers)
    aria2_config = "\n".join(f"{u}\n  out={os.path.join(prefix, 'kai-tempfile.' + suffix)}" for u, n in zip(urls, filenames) for prefix, suffix in [n.rsplit(os.sep, 1)]).encode()
    _download_with_aria2(aria2_config, total_length, use_auth_token=token if use_auth_token else None, user_agent=user_agent, force_download=force_download)
    for u, n in zip(urls, filenames):
        prefix, suffix = n.rsplit(os.sep, 1)
        os.rename(os.path.join(prefix, "kai-tempfile." + suffix), os.path.join(prefix, suffix))

def aria2_hook(pretrained_model_name_or_path: str, force_download=False, cache_dir=None, proxies=None, resume_download=False, local_files_only=False, use_auth_token=None, user_agent=None, revision=None, **kwargs):
    import transformers
    import transformers.modeling_utils
    from huggingface_hub import HfFolder
    _revision = koboldai_vars.revision if koboldai_vars.revision is not None else huggingface_hub.constants.DEFAULT_REVISION
    if shutil.which("aria2c") is None:  # Don't do anything if aria2 is not installed
        return
    if local_files_only:  # If local_files_only is true, we obviously don't need to download anything
        return
    if os.path.isdir(pretrained_model_name_or_path) or os.path.isfile(pretrained_model_name_or_path) or os.path.isfile(pretrained_model_name_or_path + ".index") or transformers.modeling_utils.is_remote_url(pretrained_model_name_or_path):
        return
    if proxies:
        print("WARNING:  KoboldAI does not support using aria2 to download models from huggingface.co through a proxy.  Disabling aria2 download mode.")
        return
    if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.22.0.dev0"):
        return _transformers22_aria2_hook(pretrained_model_name_or_path, force_download=force_download, cache_dir=cache_dir, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, use_auth_token=use_auth_token, revision=revision, **kwargs)
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
        url = huggingface_hub.hf_hub_url(pretrained_model_name_or_path, filename, revision=_revision)
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
    urls = [huggingface_hub.hf_hub_url(pretrained_model_name_or_path, n, revision=_revision) for n in filenames]
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
    aria2_config = "\n".join(f"{u}\n  out=kai-tempfile.{n}" for u, n in zip(urls, filenames)).encode()
    _download_with_aria2(aria2_config, total_length, directory=_cache_dir, use_auth_token=token if use_auth_token else None, user_agent=user_agent, force_download=force_download)
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
#  sharded model.  Requires lazy loader to be enabled to work properly
#==================================================================#
def get_sharded_checkpoint_num_tensors(pretrained_model_name_or_path, filename, is_safetensors=False, cache_dir=None, force_download=False, proxies=None, resume_download=False, local_files_only=False, use_auth_token=None, user_agent=None, revision=None, **kwargs):
    import transformers.modeling_utils
    _revision = koboldai_vars.revision if koboldai_vars.revision is not None else huggingface_hub.constants.DEFAULT_REVISION
    shard_paths, _ = transformers.modeling_utils.get_checkpoint_shard_files(pretrained_model_name_or_path, filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, use_auth_token=use_auth_token, user_agent=user_agent, revision=_revision)

    if is_safetensors:
        from safetensors import safe_open
        return list(itertools.chain(*(safe_open(p, framework="pt", device="cpu").keys() for p in shard_paths)))

    # Torch
    import torch
    return list(itertools.chain(*(torch.load(p, map_location="cpu").keys() for p in shard_paths)))

#==================================================================#
#  Given a PreTrainedModel, returns the list of module names that correspond
#  to the model's hidden layers.
#==================================================================#
def get_layers_module_names(model: PreTrainedModel) -> List[str]:
    names: List[str] = []
    def recurse(module, head=""):
        for c in module.named_children():
            name = head + c[0]
            if c[0].isnumeric() and any(c[1].__class__.__name__.endswith(suffix) for suffix in ("Block", "Layer")):
                names.append(name)
            else:
                recurse(c[1], head=name + ".")
    recurse(model)
    return names

#==================================================================#
#  Given a PreTrainedModel, returns the module name that corresponds
#  to the model's input embeddings.
#==================================================================#
def get_input_embeddings_module_name(model: PreTrainedModel) -> str:
    embeddings = model.get_input_embeddings()
    def recurse(module, head=""):
        for c in module.named_children():
            name = head + c[0]
            if c[1] is embeddings:
                return name
            else:
                return recurse(c[1], head=name + ".")
    return recurse(model)

#==================================================================#
#  Given a PreTrainedModel and a list of module names, returns a list
#  of module names such that the union of the set of modules given as input
#  and the set of modules returned as output contains all modules in the model.
#==================================================================#
def get_missing_module_names(model: PreTrainedModel, names: List[str]) -> List[str]:
    missing_names: List[str] = []
    def recurse(module, head=""):
        for c in module.named_children():
            name = head + c[0]
            if any(name.startswith(n) for n in names):
                continue
            if next(c[1].named_children(), None) is None:
                missing_names.append(name)
            else:
                recurse(c[1], head=name + ".")
    recurse(model)
    return missing_names

class UIProgressBarFile(object):
    """Write TQDM progress to the UI."""
    def __init__(self, emit_func=emit) -> None:
        self.emit_func = emit_func

    def write(self, bar):
        bar = bar.replace("\r", "").replace("\n", "").replace(chr(0), "")
        if bar != "" and [ord(num) for num in bar] != [27, 91, 65]: #No idea why we're getting the 27, 1, 65 character set, just killing to so we can move on
            #logger.info(bar)
            print('\r' + bar, end='')
            time.sleep(0.01)
            try:
                with flask_app.app_context():
                    self.emit_func('from_server', {'cmd': 'model_load_status', 'data': bar.replace(" ", "&nbsp;")}, broadcast=True, room="UI_1", namespace="/")
            except Exception as e:
                pass
        
    def flush(self):
        pass

#==================================================================#
# Strips submitted text from the text returned by the AI
#==================================================================#
def getnewcontent(txt, tokenizer):
    # If the submitted context was blank, then everything is new
    if(koboldai_vars.lastctx == ""):
        return txt
    
    # Tokenize the last context and the generated content
    ctxtokens = tokenizer.encode(encodenewlines(koboldai_vars.lastctx), max_length=int(2e9), truncation=True)
    txttokens = tokenizer.encode(encodenewlines(txt), max_length=int(2e9), truncation=True)
    dif       = (len(txttokens) - len(ctxtokens)) * -1
    
    # Remove the context from the returned text
    newtokens = txttokens[dif:]
    
    return decodenewlines(tokenizer.decode(newtokens))

#==================================================================#
# Applies chosen formatting options to text returned from AI
#==================================================================#
def applyoutputformatting(txt, no_sentence_trimming=False, no_single_line=False):
    #remove null ascii character (used to kill chat mode text in multi-generation)
    txt = txt.replace(chr(0), "")
    if len(txt) == 0:
        return txt
    
    # Handle <|endoftext|> for models that want this
    # In the future it would be nice if we could extend this to all EOS models.
    # However, since EOS detection may have unforseen consequences for now we hardcode <|endoftext|> until more can be tested
    # - Henk
    eotregex = re.compile(r'<\|endoftext\|>[.|\n|\W|\w]*')
    txt = eotregex.sub('', txt)

    # Cleanup stray </s>
    txt = txt.replace("</s>", "")

    # Use standard quotes and apostrophes
    # txt = fixquotes(txt)

    # Adventure mode clipping of all characters after '>'
    if(koboldai_vars.adventure):
        txt = koboldai_vars.acregex_ai.sub('', txt)
    
    # Trim incomplete sentences
    if(koboldai_vars.frmttriminc and not koboldai_vars.chatmode and not no_sentence_trimming):
        txt = trimincompletesentence(txt)

    # Replace blank lines
    if(koboldai_vars.frmtrmblln or koboldai_vars.chatmode):
        txt = replaceblanklines(txt)

    # trim off starting new lines in replies if we're in chat mode
    if koboldai_vars.chatmode and txt and txt[0] == "\n":
        txt = txt[1:]

    # Remove special characters
    if(koboldai_vars.frmtrmspch):
        txt = removespecialchars(txt, koboldai_vars)

	# Single Line Mode
    if(koboldai_vars.singleline and not no_single_line):
        txt = singlelineprocessing(txt, koboldai_vars)

 	# Chat Mode Trimming
    if(koboldai_vars.chatmode):
        txt = chatmodeprocessing(txt, koboldai_vars)   

    for sub in koboldai_vars.substitutions:
        if not sub["enabled"]:
            continue
        i = 0
        while sub["trueTarget"] in txt or sub["target"] in txt:
            i += 1
            if i > 1000:
                print("[substitutions] Infinite recursion :^(")
                break
            txt = txt.replace(sub["trueTarget"], sub["substitution"])
            txt = txt.replace(sub["target"], sub["substitution"])
    
    return txt
