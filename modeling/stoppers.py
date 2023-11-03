from __future__ import annotations

import torch

import utils
from modeling import inference_model

class Stoppers:
    @staticmethod
    def core_stopper(
        model: inference_model.InferenceModel,
        input_ids: torch.LongTensor,
    ) -> bool:
        if not utils.koboldai_vars.inference_config.do_core:
            return False

        utils.koboldai_vars.generated_tkns += 1

        if (
            not utils.koboldai_vars.standalone
            and utils.koboldai_vars.lua_koboldbridge.generated_cols
            and utils.koboldai_vars.generated_tkns
            != utils.koboldai_vars.lua_koboldbridge.generated_cols
        ):
            raise RuntimeError(
                f"Inconsistency detected between KoboldAI Python and Lua backends ({utils.koboldai_vars.generated_tkns} != {utils.koboldai_vars.lua_koboldbridge.generated_cols})"
            )

        if model.abort or (
            utils.koboldai_vars.inference_config.stop_at_genamt
            and utils.koboldai_vars.generated_tkns >= utils.koboldai_vars.genamt
        ):
            model.abort = False
            model.gen_state["regeneration_required"] = False
            model.gen_state["halt"] = False
            return True

        if utils.koboldai_vars.standalone:
            return False

        assert input_ids.ndim == 2

        model.gen_state[
            "regeneration_required"
        ] = utils.koboldai_vars.lua_koboldbridge.regeneration_required
        model.gen_state["halt"] = not utils.koboldai_vars.lua_koboldbridge.generating
        utils.koboldai_vars.lua_koboldbridge.regeneration_required = False

        for i in (
            range(utils.koboldai_vars.numseqs)
            if not utils.koboldai_vars.alt_multi_gen
            else range(1)
        ):
            utils.koboldai_vars.lua_koboldbridge.generated[i + 1][
                utils.koboldai_vars.generated_tkns
            ] = int(input_ids[i, -1].item())

        return model.gen_state["regeneration_required"] or model.gen_state["halt"]

    @staticmethod
    def dynamic_wi_scanner(
        model: inference_model.InferenceModel,
        input_ids: torch.LongTensor,
    ) -> bool:
        if not utils.koboldai_vars.inference_config.do_dynamic_wi:
            return False

        if not utils.koboldai_vars.dynamicscan:
            return False

        if len(model.gen_state["wi_scanner_excluded_keys"]) != input_ids.shape[0]:
            print(model.tokenizer.decode(model.gen_state["wi_scanner_excluded_keys"]))
            print(model.tokenizer.decode(input_ids.shape[0]))

        assert len(model.gen_state["wi_scanner_excluded_keys"]) == input_ids.shape[0]

        tail = input_ids[..., -utils.koboldai_vars.generated_tkns :]
        for i, t in enumerate(tail):
            decoded = utils.decodenewlines(model.tokenizer.decode(t))
            _, _, _, found = utils.koboldai_vars.calc_ai_text(
                submitted_text=decoded, send_context=False
            )
            found = list(
                set(found) - set(model.gen_state["wi_scanner_excluded_keys"][i])
            )
            if found:
                print("FOUNDWI", found)
                return True
        return False

    @staticmethod
    def chat_mode_stopper(
        model: inference_model.InferenceModel,
        input_ids: torch.LongTensor,
    ) -> bool:
        if not utils.koboldai_vars.chatmode:
            return False

        data = [model.tokenizer.decode(x) for x in input_ids]
        # null_character = model.tokenizer.encode(chr(0))[0]
        if "completed" not in model.gen_state:
            model.gen_state["completed"] = [False] * len(input_ids)

        for i in range(len(input_ids)):
            if (
                data[i][-1 * (len(utils.koboldai_vars.chatname) + 1) :]
                == utils.koboldai_vars.chatname + ":"
            ):
                model.gen_state["completed"][i] = True
        if all(model.gen_state["completed"]):
            utils.koboldai_vars.generated_tkns = utils.koboldai_vars.genamt
            del model.gen_state["completed"]
            return True
        return False

    @staticmethod
    def stop_sequence_stopper(
        model: inference_model.InferenceModel,
        input_ids: torch.LongTensor,
    ) -> bool:
                
        data = [model.tokenizer.decode(x) for x in input_ids]
        # null_character = model.tokenizer.encode(chr(0))[0]
        if "completed" not in model.gen_state:
            model.gen_state["completed"] = [False] * len(input_ids)
        if utils.koboldai_vars.adventure:
            extra_options = [">", "\n>"]
            for option in extra_options:
                if option not in utils.koboldai_vars.stop_sequence:
                    utils.koboldai_vars.stop_sequence.append(option)

        #one issue is that the stop sequence may not actual align with the end of token 
        #if its a subsection of a longer token
        for stopper in utils.koboldai_vars.stop_sequence:
            for i in range(len(input_ids)):
                if (
                    data[i][-1 * (len(stopper)) :]
                    == stopper
                ):
                    model.gen_state["completed"][i] = True

        if all(model.gen_state["completed"]):
            utils.koboldai_vars.generated_tkns = utils.koboldai_vars.genamt
            del model.gen_state["completed"]
            if utils.koboldai_vars.adventure: # Remove added adventure mode stop sequences
                for option in extra_options:
                    if option in utils.koboldai_vars.stop_sequence:
                        utils.koboldai_vars.stop_sequence.remove(option)
            return True
        return False

    @staticmethod
    def singleline_stopper(
        model: inference_model.InferenceModel,
        input_ids: torch.LongTensor,
    ) -> bool:
        """Stop on occurances of newlines **if singleline is enabled**."""

        # It might be better just to do this further up the line
        if not utils.koboldai_vars.singleline:
            return False
        return Stoppers.newline_stopper(model, input_ids)

    @staticmethod
    def newline_stopper(
        model: inference_model.InferenceModel,
        input_ids: torch.LongTensor,
    ) -> bool:
        """Stop on occurances of newlines."""
        # Keep track of presence of newlines in each sequence; we cannot stop a
        # batch member individually, so we must wait for all of them to contain
        # a newline.
        if "newline_in_sequence" not in model.gen_state:
            model.gen_state["newline_in_sequence"] = [False] * len(input_ids)

        for sequence_idx, batch_sequence in enumerate(input_ids):
            if model.tokenizer.decode(batch_sequence[-1]) == "\n":
                model.gen_state["newline_in_sequence"][sequence_idx] = True

        if all(model.gen_state["newline_in_sequence"]):
            del model.gen_state["newline_in_sequence"]
            return True
        return False

    @staticmethod
    def sentence_end_stopper(
        model: inference_model.InferenceModel,
        input_ids: torch.LongTensor,
    ) -> bool:
        """Stops at the end of sentences."""

        # TODO: Make this more robust
        SENTENCE_ENDS = [".", "?", "!"]

        # We need to keep track of stopping for each batch, since we can't stop
        # one individually.
        if "sentence_end_in_sequence" not in model.gen_state:
            model.gen_state["sentence_end_sequence"] = [False] * len(input_ids)

        for sequence_idx, batch_sequence in enumerate(input_ids):
            decoded = model.tokenizer.decode(batch_sequence[-1])
            for end in SENTENCE_ENDS:
                if end in decoded:
                    model.gen_state["sentence_end_sequence"][sequence_idx] = True
                    break

        if all(model.gen_state["sentence_end_sequence"]):
            del model.gen_state["sentence_end_sequence"]
            return True
        return False