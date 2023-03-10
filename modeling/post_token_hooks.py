import torch

import utils
from modeling.inference_model import InferenceModel


class PostTokenHooks:
    @staticmethod
    def stream_tokens(
        model: InferenceModel,
        input_ids: torch.LongTensor,
    ) -> None:
        if not model.gen_state.get("do_streaming"):
            return

        if not utils.koboldai_vars.output_streaming:
            return

        data = [
            utils.applyoutputformatting(
                utils.decodenewlines(model.tokenizer.decode(x[-1])),
                no_sentence_trimming=True,
                no_single_line=True,
            )
            for x in input_ids
        ]
        utils.koboldai_vars.actions.stream_tokens(data)
