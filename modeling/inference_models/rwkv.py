from __future__ import annotations
import os


import time
from typing import Dict, List, Optional, Union
import numpy as np
import requests
from tokenizers import Tokenizer
from tqdm import tqdm
from huggingface_hub import hf_hub_url

import torch
from torch.nn import functional as F

# Must be defined before import
os.environ["RWKV_JIT_ON"] = "1"
# TODO: Include compiled kernel
os.environ["RWKV_CUDA_ON"] = "1"
from rwkv.model import RWKV

import utils
from logger import logger

from modeling import warpers
from modeling.warpers import Warper
from modeling.stoppers import Stoppers
from modeling.post_token_hooks import PostTokenHooks
from modeling.tokenizer import GenericTokenizer
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
    ModelCapabilities,
)

TOKENIZER_URL = (
    "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json"
)
TOKENIZER_PATH = "models/rwkv/20b_tokenizer.json"

REPO_OWNER = "BlinkDL"
MODEL_FILES = {
    "rwkv-4-pile-14b": "RWKV-4-Pile-14B-20230213-8019.pth",
    # NOTE: Still in progress(?)
    "rwkv-4-pile-14b:ctx4096": "RWKV-4-Pile-14B-20230228-ctx4096-test663.pth",
    "rwkv-4-pile-7b": "RWKV-4-Pile-7B-20221115-8047.pth",
    "rwkv-4-pile-7b:ctx4096": "RWKV-4-Pile-7B-20230109-ctx4096.pth",
    "rwkv-4-pile-3b": "RWKV-4-Pile-3B-20221008-8023.pth",
    "rwkv-4-pile-3b:ctx4096": "RWKV-4-Pile-3B-20221110-ctx4096.pth",
    "rwkv-4-pile-1b5": "RWKV-4-Pile-1B5-20220903-8040.pth",
    "rwkv-4-pile-1b5:ctx4096": "RWKV-4-Pile-1B5-20220929-ctx4096.pth",
    "rwkv-4-pile-430m": "RWKV-4-Pile-430M-20220808-8066.pth",
    "rwkv-4-pile-169m": "RWKV-4-Pile-169M-20220807-8023.pth",
}


class RWKVInferenceModel(InferenceModel):
    def __init__(
        self,
        model_name: str,
    ) -> None:
        super().__init__()
        self.model_name = model_name

        self.post_token_hooks = [
            PostTokenHooks.stream_tokens,
        ]

        self.stopper_hooks = [
            Stoppers.core_stopper,
            Stoppers.dynamic_wi_scanner,
            Stoppers.singleline_stopper,
            Stoppers.chat_mode_stopper,
            Stoppers.stop_sequence_stopper,
        ]

        self.capabilties = ModelCapabilities(
            embedding_manipulation=False,
            post_token_hooks=True,
            stopper_hooks=True,
            post_token_probs=True,
        )
        self._old_stopping_criteria = None

    def _ensure_directory_structure(self) -> None:
        for path in ["models/rwkv", "models/rwkv/models"]:
            try:
                os.mkdir(path)
            except FileExistsError:
                pass

    def _get_tokenizer(self) -> GenericTokenizer:
        if not os.path.exists(TOKENIZER_PATH):
            logger.info("RWKV tokenizer not found, downloading...")

            r = requests.get(TOKENIZER_URL)
            with open(TOKENIZER_PATH, "wb") as file:
                file.write(r.content)

        return GenericTokenizer(Tokenizer.from_file(TOKENIZER_PATH))

    def _download_model(self, model_path: str, model_class: str) -> None:
        logger.info(f"{self.model_name} not found, downloading...")

        url = hf_hub_url(
            repo_id=f"{REPO_OWNER}/{model_class}",
            filename=MODEL_FILES[self.model_name],
        )

        # TODO: Use aria2
        # https://stackoverflow.com/a/57030446
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            bar = tqdm(
                desc="Downloading RWKV Model",
                unit="B",
                unit_scale=True,
                total=int(r.headers["Content-Length"]),
            )
            with open(model_path, "wb") as file:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    file.write(chunk)
                    bar.update(len(chunk))

    def _load(self, save_model: bool, initial_load: bool) -> None:
        self._ensure_directory_structure()
        self.tokenizer = self._get_tokenizer()

        # Parse model name
        model_class, _, special = self.model_name.partition(":")
        special = special or None

        model_dir = os.path.join("models", "rwkv", "models", model_class)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        # Download model if we need to
        model_path = os.path.join(model_dir, MODEL_FILES[self.model_name])
        if not os.path.exists(model_path):
            self._download_model(model_path, model_class)

        # Now we load!

        # TODO: Breakmodel to strat
        self.model = RWKV(model=model_path, strategy="cuda:0 fp16")

    def _apply_warpers(
        self, scores: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        warpers.update_settings()
        for sid in utils.koboldai_vars.sampler_order:
            warper = Warper.from_id(sid)

            if not warper.value_is_valid():
                continue

            if warper == warpers.RepetitionPenalty:
                # Rep pen needs more data than other samplers
                scores = warper.torch(scores, input_ids=input_ids)
            else:
                scores = warper.torch(scores)
        return scores

    def _sample_token(self, logits: torch.Tensor, input_ids: torch.Tensor) -> int:
        probs = F.softmax(logits.float(), dim=-1)

        if probs.device == torch.device("cpu"):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]

            probs = self._apply_warpers(probs[None, :], input_ids)

            # TODO: is this right?
            probs[probs == -torch.inf] = 0.0

            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))

            probs = self._apply_warpers(probs[None, :], input_ids)

            # TODO: is this right?
            probs[probs == -torch.inf] = 0.0

            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:
        if seed is not None:
            torch.manual_seed(seed)

        aux_device = utils.get_auxilary_device()
        context = torch.tensor(prompt_tokens)[None, :].to(aux_device)
        out = []

        start_time = time.time()
        with torch.no_grad():
            logits, state = self.model.forward(prompt_tokens, None)
            last_token = prompt_tokens[-1]

            for _ in range(max_new):

                logits, state = self.model.forward([last_token], state)
                last_token = self._sample_token(logits, context)
                out.append(last_token)
                add = torch.tensor([[last_token]]).to(aux_device)
                context = torch.cat((context, add), dim=-1)
                self._post_token_gen(context)

        logger.debug(
            "torch_raw_generate: run generator {}s".format(time.time() - start_time)
        )

        return GenerationResult(
            self,
            out_batches=torch.tensor([out]),
            prompt=prompt_tokens,
            is_whole_generation=False,
            output_includes_prompt=True,
        )
