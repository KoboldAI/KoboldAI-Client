import torch
import torch.nn.functional as F
from torch.nn import Embedding
import transformers
from mkultra.tuning import GPTPromptTuningMixin


class _WTEMixin:
    @property
    def wte(self):
        return self.get_input_embeddings()
    
    @wte.setter
    def wte(self, v):
        self.set_input_embeddings(v)


class UniversalPromptTuningMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        if not hasattr(model, "transformer"):
            model.transformer = _WTEMixin()
        elif not hasattr(model.transformer, "wte"):
            assert isinstance(model.transformer, type)
            model.transformer.__class__ = type("_UniversalPromptTuning" + model.transformer.__class__.__name__, (_WTEMixin, model.transformer.__class__), {})

        model.__class__ = type("_UniversalPromptTuning" + model.__class__.__name__, (UniversalPromptTuningMixin, model.__class__), {})

        for param in model.parameters():
            param.requires_grad = False
        model.initialize_soft_prompt()

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        assert input_ids is not None
        assert input_ids.ndim == 2

        input_ids = F.pad(input_ids, (self.learned_embedding.size(0), 0, 0, 0), value=self.transformer.wte.weight.size(0) // 2)

        if labels is not None:
            labels = self._extend_labels(labels)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        old_embedding_call = Embedding.__call__
        model = self

        def new_embedding_call(self, input_ids, *args, **kwargs):
            inputs_embeds = old_embedding_call(self, input_ids, *args, **kwargs)
            if model.transformer.wte is self:
                assert inputs_embeds.ndim == 3
                inputs_embeds[:, :model.learned_embedding.size(0), :] = model.learned_embedding[None]
            return inputs_embeds

        Embedding.__call__ = new_embedding_call

        try:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                return_dict=return_dict,
            )
        finally:
            Embedding.__call__ = old_embedding_call

for k in dir(GPTPromptTuningMixin):
    if not hasattr(UniversalPromptTuningMixin, k):
        setattr(UniversalPromptTuningMixin, k, getattr(GPTPromptTuningMixin, k))


class AutoPromptTuningLM(UniversalPromptTuningMixin, transformers.AutoModelForCausalLM):
    pass
