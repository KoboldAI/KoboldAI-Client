from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional, Union

from enum import Enum
from logger import logger

import torch
import numpy as np
import transformers
from transformers import (
    GPT2Tokenizer,
    AutoTokenizer,
    LlamaTokenizer,
)
from modeling.stoppers import Stoppers
from modeling.tokenizer import GenericTokenizer
from modeling import logits_processors

import utils


# We only want to use logit manipulations and such on our core text model
class use_core_manipulations:
    """Use in a `with` block to patch functions for core story model sampling."""

    # These must be set by wherever they get setup
    get_logits_processor: callable = None
    sample: callable = None
    get_stopping_criteria: callable = None

    # We set these automatically
    old_get_logits_processor: callable = None
    old_sample: callable = None
    old_get_stopping_criteria: callable = None

    def __enter__(self):
        if use_core_manipulations.get_logits_processor:
            use_core_manipulations.old_get_logits_processor = (
                transformers.GenerationMixin._get_logits_processor
            )
            transformers.GenerationMixin._get_logits_processor = (
                use_core_manipulations.get_logits_processor
            )

        if use_core_manipulations.sample:
            use_core_manipulations.old_sample = transformers.GenerationMixin.sample
            transformers.GenerationMixin.sample = use_core_manipulations.sample

        if use_core_manipulations.get_stopping_criteria:
            use_core_manipulations.old_get_stopping_criteria = (
                transformers.GenerationMixin._get_stopping_criteria
            )
            transformers.GenerationMixin._get_stopping_criteria = (
                use_core_manipulations.get_stopping_criteria
            )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if use_core_manipulations.old_get_logits_processor:
            transformers.GenerationMixin._get_logits_processor = (
                use_core_manipulations.old_get_logits_processor
            )
        else:
            assert (
                not use_core_manipulations.get_logits_processor
            ), "Patch leak: THE MONKEYS HAVE ESCAPED"

        if use_core_manipulations.old_sample:
            transformers.GenerationMixin.sample = use_core_manipulations.old_sample
        else:
            assert (
                not use_core_manipulations.sample
            ), "Patch leak: THE MONKEYS HAVE ESCAPED"

        if use_core_manipulations.old_get_stopping_criteria:
            transformers.GenerationMixin._get_stopping_criteria = (
                use_core_manipulations.old_get_stopping_criteria
            )
        else:
            assert (
                not use_core_manipulations.get_stopping_criteria
            ), "Patch leak: THE MONKEYS HAVE ESCAPED"


class GenerationResult:
    """A container for easily accessing different forms of model outputs. Returned by most generate functions."""

    def __init__(
        self,
        model: InferenceModel,
        out_batches: list,
        prompt: list,
        # Controls if generate() does it's looping thing. This should only be
        # done for HF models that use that StoppingCondition
        is_whole_generation: bool,
        # Controls if we should trim output by prompt length
        output_includes_prompt: bool = False,
        # Lazy filter to cut off extra lines where we can't manipulate
        # probabilities
        single_line: bool = False,
    ):
        # Shave prompt off of encoded response when needed (HF). Decoded does
        # not return prompt.
        if output_includes_prompt:
            self.encoded = out_batches[:, len(prompt) :]
        else:
            self.encoded = out_batches

        self.prompt = prompt
        self.is_whole_generation = is_whole_generation

        self.decoded = [
            utils.decodenewlines(model.tokenizer.decode(enc)) for enc in self.encoded
        ]

        if single_line:
            self.decoded = [x.split("\n", 1)[0] for x in self.decoded]
            self.encoded = np.array(model.tokenizer(self.decoded).input_ids)


class GenerationSettings:
    """Structure for holding temporarily overwritten settings."""

    def __init__(self, **overrides) -> None:
        for setting in [
            "temp",
            "top_p",
            "top_k",
            "tfs",
            "typical",
            "top_a",
            "rep_pen",
            "rep_pen_slope",
            "rep_pen_range",
            "sampler_order",
        ]:
            setattr(
                self,
                setting,
                overrides.get(setting, getattr(utils.koboldai_vars, setting)),
            )


@dataclass
class ModelCapabilities:
    embedding_manipulation: bool = False
    post_token_hooks: bool = False

    # Used to gauge if manual stopping is possible
    stopper_hooks: bool = False

    # TODO: Support non-live probabilities from APIs
    post_token_probs: bool = False

    # Some models cannot be hosted over the API, namely the API itself.
    api_host: bool = True

    # Some models need to warm up the TPU before use
    uses_tpu: bool = False

class GenerationMode(Enum):
    STANDARD = "standard"
    FOREVER = "forever"
    UNTIL_EOS = "until_eos"
    UNTIL_NEWLINE = "until_newline"
    UNTIL_SENTENCE_END = "until_sentence_end"

class InferenceModel:
    """Root class for all models."""

    def __init__(self) -> None:
        self.abort = False
        self.gen_state = {}
        self.post_token_hooks = []
        self.stopper_hooks = []
        self.logits_processors = [
            logits_processors.LuaIntegration(),
            logits_processors.PhraseBiasLogitsProcessor(),
            logits_processors.ProbabilityVisualization(),
        ]
        self.tokenizer = None
        self.capabilties = ModelCapabilities()
        self.model_name = "Not Defined"
    
    def is_valid(self, model_name, model_path, menu_path):
        return True
        
    def requested_parameters(self, model_name, model_path, menu_path):
        return {}
        
    def set_input_parameters(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])
        return

    def get_auxilary_device(self) -> Union[str, int, torch.device]:
        """Get device auxilary tensors like inputs should be stored on."""

        # NOTE: TPU isn't a torch device, so TPU stuff gets sent to CPU.
        if utils.koboldai_vars.hascuda and utils.koboldai_vars.usegpu:
            return utils.koboldai_vars.gpu_device
        elif utils.koboldai_vars.hascuda:
            return "cuda"
        return "cpu"

    def load(self, save_model: bool = False, initial_load: bool = False) -> None:
        """User-facing load function. Do not override this; try `_load()` instead."""

        self._pre_load()
        self._load(save_model=save_model, initial_load=initial_load)
        self._post_load()
        self._save_settings()

    def unload(self):
        return

    def _pre_load(self) -> None:
        """Pre load hook. Called before `_load()`."""

    def _post_load(self) -> None:
        """Post load hook. Called after `_load()`."""
    
    def _save_settings(self) -> None:
        """Save settings hook. Called after `_post_load()`."""

    def _load(self, save_model: bool, initial_load: bool) -> None:
        """Main load method. All logic related to loading the model onto the
        selected device(s) and preparing it for inference should be implemented here."""
        raise NotImplementedError

    def _get_tokenizer(self, location: str) -> GenericTokenizer:
        """Returns the appropiate tokenizer for the location. Should be ran once and result stored in `tokenizer`.

        Args:
            location (str): Either a local model directory path or a HuggingFace model ID.

        Returns:
            AutoTokenizer: Tokenizer deemed fit for the location string. May be a fallback tokenizer.
        """

        std_kwargs = {"revision": utils.koboldai_vars.revision, "cache_dir": "cache"}

        suppliers = [
            # Fast tokenizer disabled by default as per HF docs:
            # > Note: Make sure to pass use_fast=False when loading
            #   OPT’s tokenizer with AutoTokenizer to get the correct
            #   tokenizer.
            lambda: AutoTokenizer.from_pretrained(
                location, use_fast=False, **std_kwargs
            ),
            lambda: AutoTokenizer.from_pretrained(location, **std_kwargs),
            # Attempt more basic GPT2 Tokenizer
            lambda: GPT2Tokenizer.from_pretrained(location, **std_kwargs),
            # Fallback to generic LLaMA Tokenizer
            lambda: LlamaTokenizer.from_pretrained("KoboldAI/llama2-tokenizer", use_fast=False, **std_kwargs),
        ]

        for i, try_get_tokenizer in enumerate(suppliers):
            try:
                return GenericTokenizer(try_get_tokenizer())
            except Exception as e:
                logger.warning(f"Tokenizer falling back due to {e} (This can be normal behavior for some architectures that lack a slow tokenizer such as NeoX)")
                # If we error on each attempt, raise the last one
                if i == len(suppliers) - 1:
                    raise

    def core_generate(
        self,
        text: list,
        found_entries: set,
        gen_mode: GenerationMode = GenerationMode.STANDARD,
    ):
        """Generate story text. Heavily tied to story-specific parameters; if
        you are making a new generation-based feature, consider `generate_raw()`.

        Args:
            text (list): Encoded input tokens
            found_entries (set): Entries found for Dynamic WI
            gen_mode (GenerationMode): The GenerationMode to pass to raw_generate. Defaults to GenerationMode.STANDARD

        Raises:
            RuntimeError: if inconsistancies are detected with the internal state and Lua state -- sanity check
            RuntimeError: if inconsistancies are detected with the internal state and core stopper -- sanity check
        """

        start_time = time.time()
        gen_in = torch.tensor(text, dtype=torch.long)[None]
        logger.debug(
            "core_generate: torch.tensor time {}s".format(time.time() - start_time)
        )

        start_time = time.time()
        if utils.koboldai_vars.is_model_torch():
            # Torch stuff
            if utils.koboldai_vars.sp is not None:
                assert self.capabilties.embedding_manipulation
                soft_tokens = torch.arange(
                    self.model.config.vocab_size,
                    self.model.config.vocab_size + utils.koboldai_vars.sp.shape[0],
                )
                gen_in = torch.cat((soft_tokens[None], gen_in), dim=-1)

        logger.debug(
            "core_generate: Model Setup (SP, etc) time {}s".format(
                time.time() - start_time
            )
        )

        if (
            gen_in.shape[-1] + utils.koboldai_vars.genamt
            > utils.koboldai_vars.max_length
        ):
            logger.error("gen_in.shape[-1]: {}".format(gen_in.shape[-1]))
            logger.error(
                "utils.koboldai_vars.genamt: {}".format(utils.koboldai_vars.genamt)
            )
            logger.error(
                "utils.koboldai_vars.max_length: {}".format(
                    utils.koboldai_vars.max_length
                )
            )
        assert (
            gen_in.shape[-1] + utils.koboldai_vars.genamt
            <= utils.koboldai_vars.max_length
        )

        start_time = time.time()
        gen_in = gen_in.to(self.get_auxilary_device())

        logger.debug(
            "core_generate: gen_in to device time {}s".format(time.time() - start_time)
        )
        start_time = time.time()

        found_entries = found_entries or set()

        self.gen_state["wi_scanner_excluded_keys"] = found_entries

        utils.koboldai_vars._prompt = utils.koboldai_vars.prompt

        with torch.no_grad():
            already_generated = 0
            numseqs = utils.koboldai_vars.numseqs
            total_gens = None

            for i in range(
                utils.koboldai_vars.numseqs if utils.koboldai_vars.alt_multi_gen else 1
            ):
                while True:
                    # The reason this is a loop is due to how Dynamic WI works. We
                    # cannot simply add the WI to the context mid-generation, so we
                    # stop early, and then insert WI, then continue generating. That
                    # stopping and continuing is this loop.

                    start_time = time.time()
                    result = self.raw_generate(
                        gen_in[0],
                        max_new=utils.koboldai_vars.genamt,
                        do_streaming=utils.koboldai_vars.output_streaming,
                        do_dynamic_wi=utils.koboldai_vars.dynamicscan,
                        batch_count=numseqs
                        if not utils.koboldai_vars.alt_multi_gen
                        else 1,
                        # Real max length is handled by CoreStopper.
                        bypass_hf_maxlength=utils.koboldai_vars.dynamicscan,
                        is_core=True,
                        tpu_dynamic_inference=utils.koboldai_vars.dynamicscan
                        or (
                            not utils.koboldai_vars.nogenmod
                            and utils.koboldai_vars.has_genmod
                        ),
                        seed=utils.koboldai_vars.seed
                        if utils.koboldai_vars.full_determinism
                        else None,
                        gen_mode=gen_mode
                    )
                    logger.debug(
                        "core_generate: run raw_generate pass {} {}s".format(
                            already_generated, time.time() - start_time
                        )
                    )

                    genout = result.encoded

                    already_generated += len(genout[0])

                    try:
                        assert (
                            already_generated
                            <= utils.koboldai_vars.genamt * utils.koboldai_vars.numseqs
                            if utils.koboldai_vars.alt_multi_gen
                            else 1
                        )
                    except AssertionError:
                        print("AlreadyGenerated", already_generated)
                        print("genamt", utils.koboldai_vars.genamt)
                        raise

                    if result.is_whole_generation:
                        break

                    # Generation stopped; why?
                    # If we have been told to halt, we have reached our target token
                    # amount (controlled by halt), or Dynamic WI has not told us to
                    # stop temporarily to insert WI, we can assume that we are done
                    # generating. We shall break.
                    if self.gen_state.get("halt") or not self.gen_state.get(
                        "regeneration_required"
                    ):
                        break

                    # Now we are doing stuff for Dynamic WI.
                    assert genout.ndim >= 2
                    assert genout.shape[0] == utils.koboldai_vars.numseqs

                    if (
                        utils.koboldai_vars.lua_koboldbridge.generated_cols
                        and utils.koboldai_vars.generated_tkns
                        != utils.koboldai_vars.lua_koboldbridge.generated_cols
                    ):
                        raise RuntimeError(
                            f"Inconsistency detected between KoboldAI Python and Lua backends ({utils.koboldai_vars.generated_tkns} != {utils.koboldai_vars.lua_koboldbridge.generated_cols})"
                        )

                    if already_generated != utils.koboldai_vars.generated_tkns:
                        print("already_generated: {}".format(already_generated))
                        print(
                            "generated_tkns: {}".format(
                                utils.koboldai_vars.generated_tkns
                            )
                        )
                        raise RuntimeError("WI scanning error")

                    for r in range(utils.koboldai_vars.numseqs):
                        for c in range(already_generated):
                            assert (
                                utils.koboldai_vars.lua_koboldbridge.generated[r + 1][
                                    c + 1
                                ]
                                is not None
                            )
                            genout[r][
                                genout.shape[-1] - already_generated + c
                            ] = utils.koboldai_vars.lua_koboldbridge.generated[r + 1][
                                c + 1
                            ]

                    encoded = []

                    for i in range(utils.koboldai_vars.numseqs):
                        txt = utils.decodenewlines(
                            self.tokenizer.decode(genout[i, -already_generated:])
                        )
                        # winfo, mem, anotetxt, _found_entries = calcsubmitbudgetheader(txt, force_use_txt=True, actions=utils.koboldai_vars.actions)
                        # txt, _, _ = calcsubmitbudget(len(utils.koboldai_vars.actions), winfo, mem, anotetxt, utils.koboldai_vars.actions, submission=txt)
                        txt, _, _, _found_entries = utils.koboldai_vars.calc_ai_text(
                            submitted_text=txt, send_context=False
                        )
                        found_entries[i].update(_found_entries)
                        encoded.append(
                            torch.tensor(txt, dtype=torch.long, device=genout.device)
                        )

                    max_length = len(max(encoded, key=len))
                    encoded = torch.stack(
                        tuple(
                            torch.nn.functional.pad(
                                e,
                                (max_length - len(e), 0),
                                value=self.model.config.pad_token_id
                                or self.model.config.eos_token_id,
                            )
                            for e in encoded
                        )
                    )
                    genout = torch.cat(
                        (
                            encoded,
                            genout[..., -already_generated:],
                        ),
                        dim=-1,
                    )

                    if utils.koboldai_vars.sp is not None:
                        soft_tokens = torch.arange(
                            self.model.config.vocab_size,
                            self.model.config.vocab_size
                            + utils.koboldai_vars.sp.shape[0],
                            device=genout.device,
                        )
                        genout = torch.cat(
                            (soft_tokens.tile(utils.koboldai_vars.numseqs, 1), genout),
                            dim=-1,
                        )

                    assert (
                        genout.shape[-1]
                        + utils.koboldai_vars.genamt
                        - already_generated
                        <= utils.koboldai_vars.max_length
                    )
                    gen_in = genout
                    numseqs = 1
                if total_gens is None:
                    total_gens = genout
                else:
                    total_gens = torch.cat((total_gens, genout))

        return total_gens, already_generated

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
        **kwargs,
    ) -> GenerationResult:
        """Lowest level model-agnostic generation function. To be overridden by model implementation.

        Args:
            prompt_tokens (Union[List[int], torch.Tensor]): Prompt as encoded token IDs
            max_new (int): Maximum amount of new tokens to generate
            gen_settings (GenerationSettings): State to pass in single-generation setting overrides
            single_line (bool, optional): Generate one line only. Defaults to False.
            batch_count (int, optional): How big of a batch to generate. Defaults to 1.
            seed (int, optional): If not None, this seed will be used to make reproducible generations. Defaults to None.

        Returns:
            GenerationResult: The model's output
        """
        raise NotImplementedError

    def raw_generate(
        self,
        # prompt is either a string (text) or a list (token ids)
        prompt: Union[str, list, np.ndarray],
        max_new: int,
        do_streaming: bool = False,
        do_dynamic_wi: bool = False,
        batch_count: int = 1,
        bypass_hf_maxlength: bool = False,
        generation_settings: Optional[dict] = None,
        is_core: bool = False,
        single_line: bool = False,
        found_entries: set = (),
        tpu_dynamic_inference: bool = False,
        seed: Optional[int] = None,
        gen_mode: GenerationMode = GenerationMode.STANDARD,
        **kwargs,
    ) -> GenerationResult:
        """A wrapper around `_raw_generate()` that handles gen_state and other stuff. Use this to generate text outside of the story.

        Args:
            prompt (Union[str, list, np.ndarray]): The prompt as a string or encoded token IDs
            max_new (int): Maximum amount of new tokens to generate
            do_streaming (bool, optional): Whether to stream tokens to the user or not. Defaults to False.
            do_dynamic_wi (bool, optional): Whether to use Dynamic WI context injections. Defaults to False.
            batch_count (int, optional): How big of a batch to generate. Defaults to 1.
            bypass_hf_maxlength (bool, optional): Whether to ignore model-provided max length limits. Defaults to False.
            generation_settings (GenerationSettings): State to pass in single-generation setting overrides. Defaults to None
            is_core (bool, optional): Whether this generation is a core story generation. Defaults to False.
            single_line (bool, optional): Generate one line only.. Defaults to False.
            found_entries (set, optional): Entries found for Dynamic WI. Defaults to ().
            gen_mode (GenerationMode): Special generation mode. Defaults to GenerationMode.STANDARD.

        Raises:
            ValueError: If prompt type is weird
            NotImplementedError: If model is ReadOnly

        Returns:
            GenerationResult: The model's output
        """
        # TODO: Support singleline outside of torch

        self.gen_state["do_streaming"] = do_streaming
        self.gen_state["do_dynamic_wi"] = do_dynamic_wi

        # Dynamic WI depends on this!!! This is a main gen call.
        self.gen_state["stop_at_genamt"] = do_dynamic_wi

        # Makes stopping criteria hook happy
        self.gen_state["wi_scanner_excluded_keys"] = self.gen_state.get(
            "wi_scanner_excluded_keys", set()
        )

        self.gen_state["allow_eos"] = False

        temp_stoppers = []

        if gen_mode not in self.get_supported_gen_modes():
            gen_mode = GenerationMode.STANDARD
            logger.warning(f"User requested unsupported GenerationMode '{gen_mode}'!")

        if gen_mode == GenerationMode.FOREVER:
            self.gen_state["stop_at_genamt"] = False
            max_new = 1e7
        elif gen_mode == GenerationMode.UNTIL_EOS:
            self.gen_state["allow_eos"] = True
            self.gen_state["stop_at_genamt"] = False
            max_new = 1e7
        elif gen_mode == GenerationMode.UNTIL_NEWLINE:
            # TODO: Look into replacing `single_line` with `generation_mode`
            temp_stoppers.append(Stoppers.newline_stopper)
        elif gen_mode == GenerationMode.UNTIL_SENTENCE_END:
            temp_stoppers.append(Stoppers.sentence_end_stopper)

        self.stopper_hooks += temp_stoppers

        utils.koboldai_vars.inference_config.do_core = is_core
        gen_settings = GenerationSettings(*(generation_settings or {}))

        if isinstance(prompt, torch.Tensor):
            prompt_tokens = prompt.cpu().numpy()
        elif isinstance(prompt, list):
            prompt_tokens = np.array(prompt)
        elif isinstance(prompt, str):
            prompt_tokens = np.array(self.tokenizer.encode(prompt))
        else:
            raise ValueError(f"Prompt is {type(prompt)}. Not a fan!")

        assert isinstance(prompt_tokens, np.ndarray)
        assert len(prompt_tokens.shape) == 1

        time_start = time.time()

        with use_core_manipulations():
            result = self._raw_generate(
                prompt_tokens=prompt_tokens,
                max_new=max_new,
                batch_count=batch_count,
                gen_settings=gen_settings,
                single_line=single_line,
                tpu_dynamic_inference=tpu_dynamic_inference,
                seed=seed,
            )

        time_end = round(time.time() - time_start, 2)

        try:
            tokens_per_second = round(len(result.encoded[0]) / time_end, 2)
        except ZeroDivisionError:
            # Introducing KoboldAI's fastest model: ReadOnly!
            tokens_per_second = 0

        if not utils.koboldai_vars.quiet:
            logger.info(
                f"Generated {len(result.encoded[0])} tokens in {time_end} seconds, for an average rate of {tokens_per_second} tokens per second."
            )

        for stopper in temp_stoppers:
            self.stopper_hooks.remove(stopper)

        return result

    def generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new_tokens: int,
        do_streaming: bool = False,
        do_dynamic_wi: bool = False,
        single_line: bool = False,
        batch_count: int = 1,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _post_token_gen(self, input_ids: torch.LongTensor) -> None:
        for hook in self.post_token_hooks:
            hook(self, input_ids)

    def abort_generation(self, abort=True):
        self.abort=abort
    
    def get_supported_gen_modes(self) -> List[GenerationMode]:
        """Returns a list of compatible `GenerationMode`s for the current model.

        Returns:
            List[GenerationMode]: A list of compatible `GenerationMode`s.
        """
        ret = [GenerationMode.STANDARD]

        if self.capabilties.stopper_hooks:
            ret += [
                GenerationMode.FOREVER,
                GenerationMode.UNTIL_NEWLINE,
                GenerationMode.UNTIL_SENTENCE_END,
            ]
        return ret