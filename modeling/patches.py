from __future__ import annotations

import copy
import requests
from typing import Iterable, List
from tqdm.auto import tqdm

import transformers
from transformers import (
    PreTrainedModel,
    modeling_utils,
)

import utils


def patch_transformers_download():
    def http_get(
        url: str,
        temp_file,
        proxies=None,
        resume_size=0,
        headers=None,
        file_name=None,
    ):
        """
        Download remote file. Do not gobble up errors.
        """
        headers = copy.deepcopy(headers)
        if resume_size > 0:
            headers["Range"] = f"bytes={resume_size}-"
        r = requests.get(url, stream=True, proxies=proxies, headers=headers)
        transformers.utils.hub._raise_for_status(r)
        content_length = r.headers.get("Content-Length")
        total = (
            resume_size + int(content_length) if content_length is not None else None
        )

        # `tqdm` behavior is determined by `utils.logging.is_progress_bar_enabled()`
        # and can be set using `utils.logging.enable/disable_progress_bar()`
        if url[-11:] != "config.json":
            progress = tqdm.tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=total,
                initial=resume_size,
                desc=f"Downloading {file_name}"
                if file_name is not None
                else "Downloading",
                file=utils.UIProgressBarFile(),
            )
            utils.koboldai_vars.status_message = "Download Model"
            utils.koboldai_vars.total_download_chunks = total

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                if url[-11:] != "config.json":
                    progress.update(len(chunk))
                    utils.koboldai_vars.downloaded_chunks += len(chunk)
                temp_file.write(chunk)

        if url[-11:] != "config.json":
            progress.close()

        utils.koboldai_vars.status_message = ""

    transformers.utils.hub.http_get = http_get


def patch_transformers_loader() -> None:
    """
    Patch the Transformers loader to use aria2 and our shard tracking.
    Universal for TPU/MTJ and Torch.
    """
    old_from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def new_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        utils.koboldai_vars.fp32_model = False
        utils.num_shards = None
        utils.current_shard = 0
        utils.from_pretrained_model_name = pretrained_model_name_or_path
        utils.from_pretrained_index_filename = None
        utils.from_pretrained_kwargs = kwargs
        utils.bar = None
        if not utils.args.no_aria2:
            utils.aria2_hook(pretrained_model_name_or_path, **kwargs)
        return old_from_pretrained(
            cls, pretrained_model_name_or_path, *model_args, **kwargs
        )

    if not hasattr(PreTrainedModel, "_kai_patched"):
        PreTrainedModel.from_pretrained = new_from_pretrained
        PreTrainedModel._kai_patched = True

    if hasattr(modeling_utils, "get_checkpoint_shard_files"):
        old_get_checkpoint_shard_files = modeling_utils.get_checkpoint_shard_files

        def new_get_checkpoint_shard_files(
            pretrained_model_name_or_path, index_filename, *args, **kwargs
        ):
            utils.num_shards = utils.get_num_shards(index_filename)
            utils.from_pretrained_index_filename = index_filename
            return old_get_checkpoint_shard_files(
                pretrained_model_name_or_path, index_filename, *args, **kwargs
            )

        modeling_utils.get_checkpoint_shard_files = new_get_checkpoint_shard_files


def patch_transformers_generation() -> None:
    # Not sure why this global is needed...
    global transformers

    # Allow bad words filter to ban <|endoftext|> token
    import transformers.generation.logits_process

    def new_init(self, bad_words_ids: List[List[int]], eos_token_id: int):
        return new_init.old_init(self, bad_words_ids, -1)

    new_init.old_init = (
        transformers.generation.logits_process.NoBadWordsLogitsProcessor.__init__
    )
    transformers.generation.logits_process.NoBadWordsLogitsProcessor.__init__ = new_init


def patch_transformers() -> None:
    patch_transformers_download()
    patch_transformers_loader()

    # Doesn't do anything for TPU
    patch_transformers_generation()
