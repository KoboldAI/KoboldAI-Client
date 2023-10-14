from __future__ import annotations

import os
import torch
import numpy as np
from eventlet import tpool
from typing import List, Optional, Tuple, Union

import utils
import koboldai_settings
from logger import logger, Colors

from modeling import warpers
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    ModelCapabilities,
)
from modeling.inference_models.hf import HFInferenceModel
from modeling.tokenizer import GenericTokenizer

model_backend_name = "Huggingface MTJ"
model_backend_type = "Huggingface" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)


class model_backend(HFInferenceModel):
    def __init__(
        self,
        #model_name: str,
    ) -> None:
        super().__init__()
        import importlib
        dependency_exists = importlib.util.find_spec("jax")
        if dependency_exists:
            self.hf_torch = False
            self.model_config = None
            self.capabilties = ModelCapabilities(
                embedding_manipulation=False,
                post_token_hooks=False,
                stopper_hooks=False,
                post_token_probs=False,
                uses_tpu=True,
            )
        else:
            logger.debug("Jax is not installed, hiding TPU backend")
            self.disable = True
        
    def is_valid(self, model_name, model_path, menu_path):
        # This file shouldn't be imported unless using the TPU
        return utils.koboldai_vars.use_colab_tpu and super().is_valid(model_name, model_path, menu_path)

    def setup_mtj(self) -> None:
        import tpu_mtj_backend
        def mtj_warper_callback(scores) -> "np.array":
            scores_shape = scores.shape
            scores_list = scores.tolist()
            utils.koboldai_vars.lua_koboldbridge.logits = (
                utils.koboldai_vars.lua_state.table()
            )
            for r, row in enumerate(scores_list):
                utils.koboldai_vars.lua_koboldbridge.logits[
                    r + 1
                ] = utils.koboldai_vars.lua_state.table(*row)
            utils.koboldai_vars.lua_koboldbridge.vocab_size = scores_shape[-1]

            utils.koboldai_vars.lua_koboldbridge.execute_genmod()

            scores = np.array(
                tuple(
                    tuple(row.values())
                    for row in utils.koboldai_vars.lua_koboldbridge.logits.values()
                ),
                dtype=scores.dtype,
            )
            assert scores.shape == scores_shape

            return scores

        def mtj_stopping_callback(
            generated, n_generated
        ) -> Tuple[List[set], bool, bool]:
            utils.koboldai_vars.generated_tkns += 1

            regeneration_required = (
                utils.koboldai_vars.lua_koboldbridge.regeneration_required
            )
            halt = (
                utils.koboldai_vars.abort
                or not utils.koboldai_vars.lua_koboldbridge.generating
                or utils.koboldai_vars.generated_tkns >= utils.koboldai_vars.genamt
            )
            utils.koboldai_vars.lua_koboldbridge.regeneration_required = False

            # Not sure what the deal is with this variable. It's been undefined
            # as far back as I can trace it.
            global past

            for i in range(utils.koboldai_vars.numseqs):
                utils.koboldai_vars.lua_koboldbridge.generated[i + 1][
                    utils.koboldai_vars.generated_tkns
                ] = int(
                    generated[i, tpu_mtj_backend.params["seq"] + n_generated - 1].item()
                )

            if not utils.koboldai_vars.dynamicscan or halt:
                return regeneration_required, halt

            for i, t in enumerate(generated):
                decoded = utils.decodenewlines(
                    self.tokenizer.decode(past[i])
                ) + utils.decodenewlines(
                    self.tokenizer.decode(
                        t[
                            tpu_mtj_backend.params["seq"] : tpu_mtj_backend.params[
                                "seq"
                            ]
                            + n_generated
                        ]
                    )
                )
                # _, found = checkworldinfo(decoded, force_use_txt=True, actions=koboldai_vars.actions)
                _, _, _, used_world_info = utils.koboldai_vars.calc_ai_text(
                    submitted_text=decoded
                )
                # found -= excluded_world_info[i]
                if used_world_info:
                    regeneration_required = True
                    break
            return regeneration_required, halt

        def mtj_compiling_callback() -> None:
            print(Colors.GREEN + "TPU backend compilation triggered" + Colors.END)
            utils.koboldai_vars.compiling = True

        def mtj_stopped_compiling_callback() -> None:
            if utils.koboldai_vars.compiling:
                print(Colors.GREEN + "TPU backend compilation stopped" + Colors.END)
            utils.koboldai_vars.compiling = False

        def mtj_settings_callback() -> dict:
            sampler_order = utils.koboldai_vars.sampler_order[:]
            if (
                len(sampler_order) < 7
            ):  # Add repetition penalty at beginning if it's not present
                sampler_order = [6] + sampler_order
            return {
                "sampler_order": utils.koboldai_vars.sampler_order,
                "top_p": float(utils.koboldai_vars.top_p),
                "temp": float(utils.koboldai_vars.temp),
                "top_k": int(utils.koboldai_vars.top_k),
                "tfs": float(utils.koboldai_vars.tfs),
                "typical": float(utils.koboldai_vars.typical),
                "top_a": float(utils.koboldai_vars.top_a),
                "repetition_penalty": float(utils.koboldai_vars.rep_pen),
                "rpslope": float(utils.koboldai_vars.rep_pen_slope),
                "rprange": int(utils.koboldai_vars.rep_pen_range),
            }

        tpu_mtj_backend.socketio = utils.socketio

        if self.model_name == "TPUMeshTransformerGPTNeoX":
            utils.koboldai_vars.badwordsids = utils.koboldai_vars.badwordsids_neox

        print(
            "{0}Initializing Mesh Transformer JAX, please wait...{1}".format(
                Colors.PURPLE, Colors.END
            )
        )
        if self.model_name in (
            "TPUMeshTransformerGPTJ",
            "TPUMeshTransformerGPTNeoX",
        ) and (
            not utils.koboldai_vars.custmodpth
            or not os.path.isdir(utils.koboldai_vars.custmodpth)
        ):
            raise FileNotFoundError(
                f"The specified model path {repr(utils.koboldai_vars.custmodpth)} is not the path to a valid folder"
            )
        if self.model_name == "TPUMeshTransformerGPTNeoX":
            tpu_mtj_backend.pad_token_id = 2

        tpu_mtj_backend.koboldai_vars = utils.koboldai_vars
        tpu_mtj_backend.warper_callback = mtj_warper_callback
        tpu_mtj_backend.stopping_callback = mtj_stopping_callback
        tpu_mtj_backend.compiling_callback = mtj_compiling_callback
        tpu_mtj_backend.stopped_compiling_callback = mtj_stopped_compiling_callback
        tpu_mtj_backend.settings_callback = mtj_settings_callback

    def _load(self, save_model: bool, initial_load: bool) -> None:
        import tpu_mtj_backend
        self.setup_mtj()
        self.init_model_config()
        utils.koboldai_vars.allowsp = True

        logger.info(self.model_name)
        tpu_mtj_backend.load_model(
            self.model_name,
            hf_checkpoint=self.model_name
            not in ("TPUMeshTransformerGPTJ", "TPUMeshTransformerGPTNeoX")
            and utils.koboldai_vars.use_colab_tpu,
            socketio_queue=koboldai_settings.queue,
            initial_load=initial_load,
            logger=logger,
            **self.model_config.to_dict(),
        )

        utils.koboldai_vars.modeldim = int(
            tpu_mtj_backend.params.get("d_embed", tpu_mtj_backend.params["d_model"])
        )
        self.tokenizer = GenericTokenizer(tpu_mtj_backend.tokenizer)

        if (
            utils.koboldai_vars.badwordsids is koboldai_settings.badwordsids_default
            and self.model_type not in ("gpt2", "gpt_neo", "gptj")
        ):
            utils.koboldai_vars.badwordsids = [
                [v]
                for k, v in self.tokenizer.get_vocab().items()
                if any(c in str(k) for c in "[]")
            ]

    def get_soft_tokens(self) -> np.array:
        import tpu_mtj_backend
        soft_tokens = None

        if utils.koboldai_vars.sp is None:
            tensor = np.zeros(
                (
                    1,
                    tpu_mtj_backend.params.get(
                        "d_embed", tpu_mtj_backend.params["d_model"]
                    ),
                ),
                dtype=np.float32,
            )
            rows = tensor.shape[0]
            padding_amount = (
                tpu_mtj_backend.params["seq"]
                - (
                    tpu_mtj_backend.params["seq"]
                    % -tpu_mtj_backend.params["cores_per_replica"]
                )
                - rows
            )
            tensor = np.pad(tensor, ((0, padding_amount), (0, 0)))
            tensor = tensor.reshape(
                tpu_mtj_backend.params["cores_per_replica"],
                -1,
                tpu_mtj_backend.params.get(
                    "d_embed", tpu_mtj_backend.params["d_model"]
                ),
            )
            utils.koboldai_vars.sp = tpu_mtj_backend.shard_xmap(tensor)

        soft_tokens = np.arange(
            tpu_mtj_backend.params["n_vocab"]
            + tpu_mtj_backend.params["n_vocab_padding"],
            tpu_mtj_backend.params["n_vocab"]
            + tpu_mtj_backend.params["n_vocab_padding"]
            + utils.koboldai_vars.sp_length,
            dtype=np.uint32,
        )
        return soft_tokens

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
        import tpu_mtj_backend
        warpers.update_settings()

        soft_tokens = self.get_soft_tokens()

        dynamic_inference = kwargs.get("tpu_dynamic_inference", False)

        if seed is not None:
            tpu_mtj_backend.set_rng_seed(seed)

        if not dynamic_inference:
            genout = tpool.execute(
                tpu_mtj_backend.infer_static,
                np.uint32(prompt_tokens),
                gen_len=max_new,
                temp=gen_settings.temp,
                top_p=gen_settings.top_p,
                top_k=gen_settings.top_k,
                tfs=gen_settings.tfs,
                typical=gen_settings.typical,
                top_a=gen_settings.top_a,
                numseqs=batch_count,
                repetition_penalty=gen_settings.rep_pen,
                rpslope=gen_settings.rep_pen_slope,
                rprange=gen_settings.rep_pen_range,
                soft_embeddings=utils.koboldai_vars.sp,
                soft_tokens=soft_tokens,
                sampler_order=gen_settings.sampler_order,
            )
            genout = np.array(genout)
        else:
            global past
            context = np.tile(
                np.uint32(prompt_tokens), (utils.koboldai_vars.numseqs, 1)
            )
            past = np.empty((utils.koboldai_vars.numseqs, 0), dtype=np.uint32)
            self.gen_state["wi_scanner_excluded_keys"] = set()

            while True:
                genout, n_generated, regeneration_required, halt = tpool.execute(
                    tpu_mtj_backend.infer_dynamic,
                    context,
                    gen_len=max_new,
                    numseqs=utils.koboldai_vars.numseqs,
                    soft_embeddings=utils.koboldai_vars.sp,
                    soft_tokens=soft_tokens,
                )

                past = np.pad(past, ((0, 0), (0, n_generated)))
                for r in range(utils.koboldai_vars.numseqs):
                    for c in range(utils.koboldai_vars.lua_koboldbridge.generated_cols):
                        assert (
                            utils.koboldai_vars.lua_koboldbridge.generated[r + 1][c + 1]
                            is not None
                        )
                        past[r, c] = utils.koboldai_vars.lua_koboldbridge.generated[
                            r + 1
                        ][c + 1]

                if utils.koboldai_vars.abort or halt or not regeneration_required:
                    break

                encoded = []
                for i in range(utils.koboldai_vars.numseqs):
                    txt = utils.decodenewlines(self.tokenizer.decode(past[i]))
                    # _, _, _, _found_entries = utils.koboldai_vars.calc_ai_text(
                    #     self.tokenizer.decode(prompt_tokens)
                    # )
                    # # utils.koboldai_vars.calc_ai_text()
                    # print(_found_entries)
                    # self.gen_state["wi_scanner_excluded_keys"].update(_found_entries)
                    encoded.append(np.array(txt, dtype=np.uint32))

                max_length = len(max(encoded, key=len))
                encoded = np.stack(
                    tuple(
                        np.pad(
                            e,
                            (max_length - len(e), 0),
                            constant_values=tpu_mtj_backend.pad_token_id,
                        )
                        for e in encoded
                    )
                )
                context = np.concatenate(
                    (
                        encoded,
                        past,
                    ),
                    axis=-1,
                )
            # genout = tpool.execute(
            #     tpu_mtj_backend.infer_dynamic,
            #     context=np.uint32(prompt_tokens),
            #     numseqs=batch_count,
            #     gen_len=max_new,
            #     soft_embeddings=utils.koboldai_vars.sp,
            #     soft_tokens=soft_tokens,
            #     # TODO: Fix Dynamic WI on TPU
            #     excluded_world_info=set(),
            #     use_callback=True
            # )
            # print(genout)
            # print(type(genout))
            genout = np.array(genout)

        return GenerationResult(
            self,
            out_batches=genout,
            prompt=prompt_tokens,
            is_whole_generation=True,
            single_line=single_line,
        )
