'''
This is a MODIFIED version of arrmansa's low VRAM patch.
https://github.com/arrmansa/Basic-UI-for-GPT-J-6B-with-low-vram/blob/main/GPT-J-6B-Low-Vram-UI.ipynb
The ORIGINAL version of the patch is released under the Apache License 2.0
Copyright 2021 arrmansa
Copyright 2021 finetuneanon
Copyright 2018, 2022 The Hugging Face team


                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''


import torch
from torch import nn
import torch.cuda.comm
import copy
import gc
import os
import sys
import itertools
import bisect
import random
import utils
from typing import Dict, List, Optional, Union

from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions

from transformers.utils import logging
logger = logging.get_logger(__name__)


breakmodel = True
gpu_blocks = []
disk_blocks = 0
primary_device = 0 if torch.cuda.device_count() > 0 else "cpu"


if utils.HAS_ACCELERATE:
    from accelerate.hooks import attach_align_device_hook_on_blocks
    from accelerate.utils import OffloadedWeightsLoader, check_device_map, extract_submodules_state_dict, offload_state_dict
    from accelerate import dispatch_model

def dispatch_model_ex(
    model: nn.Module,
    device_map: Dict[str, Union[str, int, torch.device]],
    main_device: Optional[torch.device] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    offload_dir: Union[str, os.PathLike] = None,
    offload_buffers: bool = False,
    **kwargs,
):
    """
    This is a modified version of
    https://github.com/huggingface/accelerate/blob/eeaba598f455fbd2c48661d7e816d3ff25ab050b/src/accelerate/big_modeling.py#L130
    that still works when the main device is the CPU.

    Dispatches a model according to a given device map. Layers of the model might be spread across GPUs, offloaded on
    the CPU or even the disk.

    Args:
        model (`torch.nn.Module`):
            The model to dispatch.
        device_map (`Dict[str, Union[str, int, torch.device]]`):
            A dictionary mapping module names in the models `state_dict` to the device they should go to. Note that
            `"disk"` is accepted even if it's not a proper value for `torch.device`.
        main_device (`str`, `int` or `torch.device`, *optional*):
            The main execution device. Will default to the first device in the `device_map` different from `"cpu"` or
            `"disk"`.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the part of the model that will be kept on CPU.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
    """
    if main_device != "cpu":
        return dispatch_model(model, device_map, main_device, state_dict, offload_dir=offload_dir, offload_buffers=offload_buffers, **kwargs)

    # Error early if the device map is incomplete.
    check_device_map(model, device_map)

    offload_devices = ["cpu", "disk"] if main_device != "cpu" else ["disk"]

    if main_device is None:
        main_device = [d for d in device_map.values() if d not in offload_devices][0]

    cpu_modules = [name for name, device in device_map.items() if device == "cpu"] if main_device != "cpu" else []
    if state_dict is None and len(cpu_modules) > 0:
        state_dict = extract_submodules_state_dict(model.state_dict(), cpu_modules)

    disk_modules = [name for name, device in device_map.items() if device == "disk"]
    if offload_dir is None and len(disk_modules) > 0:
        raise ValueError(
            "We need an `offload_dir` to dispatch this model according to this `device_map`, the following submodules "
            f"need to be offloaded: {', '.join(disk_modules)}."
        )
    if len(disk_modules) > 0 and (
        not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json"))
    ):
        disk_state_dict = extract_submodules_state_dict(model.state_dict(), disk_modules)
        offload_state_dict(offload_dir, disk_state_dict)

    execution_device = {
        name: main_device if device in offload_devices else device for name, device in device_map.items()
    }
    offload = {name: device in offload_devices for name, device in device_map.items()}
    save_folder = offload_dir if len(disk_modules) > 0 else None
    if state_dict is not None or save_folder is not None:
        weights_map = OffloadedWeightsLoader(state_dict=state_dict, save_folder=save_folder)
    else:
        weights_map = None

    attach_align_device_hook_on_blocks(
        model,
        execution_device=execution_device,
        offload=offload,
        offload_buffers=offload_buffers,
        weights_map=weights_map,
        **kwargs,
    )
    model.hf_device_map = device_map
    return model


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def move_hidden_layers(transformer, h=None):
    if h is None:
        h = transformer.h

    assert len(gpu_blocks) <= torch.cuda.device_count()
    assert sum(gpu_blocks) <= len(h)
    ram_blocks = len(h) - sum(gpu_blocks)

    transformer.extrastorage = {}
    torch.cuda.empty_cache()
    
    able_to_pin_layers = True
    for i in range(ram_blocks):
        h[i].to("cpu")
        transformer.extrastorage[i] = copy.deepcopy(h[i])
        smalltensor = torch.tensor(0).to(primary_device)
        for param1 in h[i].parameters():
            param1.data = smalltensor
        h[i].to(primary_device)
        for param in transformer.extrastorage[i].parameters():
            param.requires_grad = False
            param.data = param.data.detach()
            if able_to_pin_layers:
                try:
                    param.data = param.data.pin_memory()
                except:
                    able_to_pin_layers = False
                    print(f"WARNING:  You only have enough shared GPU memory for {i} out of {ram_blocks} CPU layers.  Expect suboptimal speed.", file=sys.stderr)
            gc.collect()
            torch.cuda.empty_cache()

    if ram_blocks:
        for param1,param2 in zip(h[0].parameters(),transformer.extrastorage[0].parameters()):
            param1.data = param2.data.to(primary_device, non_blocking=False).detach()

        for param1,param2 in zip(h[ram_blocks-1].parameters(),transformer.extrastorage[ram_blocks-1].parameters()):
            param1.data = param2.data.to(primary_device, non_blocking=False).detach()

    i = ram_blocks
    for j in range(len(gpu_blocks)):
        for _ in range(gpu_blocks[j]):
            h[i].to(j)
            i += 1


def new_forward_neo(
    self,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    embs=None,
):
    assert len(gpu_blocks) <= torch.cuda.device_count()
    assert sum(gpu_blocks) <= len(self.h)
    ram_blocks = len(self.h) - sum(gpu_blocks)
    cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))


    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0][0].size(-2)

    device = primary_device if breakmodel else input_ids.device if input_ids is not None else inputs_embeds.device
    if position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # Attention mask.
    if attention_mask is not None:
        assert batch_size > 0, "batch_size has to be defined and > 0"
        attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x num_heads x N x N
    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = self.get_head_mask(head_mask, getattr(self.config, "num_layers", None) or self.config.n_layer)

    if inputs_embeds is None:
        if breakmodel:
            input_ids = input_ids.to(primary_device)
        inputs_embeds = self.wte(input_ids)

    if embs is not None and not (use_cache is not None and use_cache and past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None):
        offset = 0
        for pos, emb in embs:
            pos += offset
            if len(emb.shape) == 2:
                emb = emb.repeat(input_shape[0], 1, 1)
            inputs_embeds[:, pos:pos+emb.shape[1]] = emb
            offset += emb.shape[1]

    if getattr(self, "wpe", None) is None:
        hidden_states = inputs_embeds
    else:
        if breakmodel:
            position_ids = position_ids.to(primary_device)
        position_embeds = self.wpe(position_ids)
        if breakmodel:
            position_embeds = position_embeds.to(primary_device)
        hidden_states = inputs_embeds + position_embeds

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    if breakmodel and ram_blocks:
        copystream = torch.cuda.Stream(device=primary_device, priority=-1)

    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

        if breakmodel:
            if i in range(ram_blocks):
                index1 = (i+1)%ram_blocks
                for param1,param2 in zip(self.h[index1].parameters(),self.h[(i-1)%ram_blocks].parameters()):
                    param1.data = param2.data
                for param1,param2 in zip(self.h[index1].parameters(),self.extrastorage[index1].parameters()):
                    with torch.cuda.stream(copystream):
                        torch.cuda.comm.broadcast(param2.data,out = [param1.data])

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.cpu(),)

        if getattr(self.config, "gradient_checkpointing", False) and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                None,
                attention_mask,
                head_mask[i],
            )
        else:
            if breakmodel:
                device = primary_device if i < ram_blocks else bisect.bisect_right(cumulative_gpu_blocks, i - ram_blocks)
            outputs = block(
                hidden_states.to(device) if breakmodel and hidden_states is not None else hidden_states,
                layer_past=tuple(v.to(device) for v in layer_past if v is not None) if breakmodel and layer_past is not None and i >= ram_blocks and len(layer_past) and layer_past[0].device.index != device else layer_past,
                attention_mask=attention_mask.to(device) if breakmodel and attention_mask is not None else attention_mask,
                head_mask=head_mask[i].to(device) if breakmodel and head_mask[i] is not None else head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)


        if breakmodel:
            if i in range(ram_blocks):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    if breakmodel:
        if ram_blocks:
            del copystream
        torch.cuda.empty_cache()
        hidden_states = hidden_states.to(primary_device)
    hidden_states = self.ln_f(hidden_states)
    if breakmodel:
        hidden_states = hidden_states.to(primary_device)

    hidden_states = hidden_states.view(*output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def new_forward_xglm(
    self,
    input_ids=None,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    head_mask=None,
    cross_attn_head_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    assert len(gpu_blocks) <= torch.cuda.device_count()
    assert sum(gpu_blocks) <= len(self.layers)
    ram_blocks = len(self.layers) - sum(gpu_blocks)
    cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))


    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if inputs_embeds is None:
        if breakmodel:
            input_ids = input_ids.to(primary_device)
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )

    # expand encoder attention mask
    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

    # embed positions
    if breakmodel:
        inputs_embeds = inputs_embeds.to(primary_device)
    positions = self.embed_positions(input_ids, inputs_embeds, past_key_values_length)
    if breakmodel:
        positions = positions.to(primary_device)

    hidden_states = inputs_embeds + positions

    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
    next_decoder_cache = () if use_cache else None

    if breakmodel and ram_blocks:
        copystream = torch.cuda.Stream(device=primary_device, priority=-1)

    # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
        if attn_mask is not None:
            assert attn_mask.size()[0] == (
                len(self.layers)
            ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
    for idx, decoder_layer in enumerate(self.layers):
        i = idx
        if breakmodel:
            if i in range(ram_blocks):
                index1 = (i+1)%ram_blocks
                for param1,param2 in zip(self.layers[index1].parameters(),self.layers[(i-1)%ram_blocks].parameters()):
                    param1.data = param2.data
                for param1,param2 in zip(self.layers[index1].parameters(),self.extrastorage[index1].parameters()):
                    with torch.cuda.stream(copystream):
                        torch.cuda.comm.broadcast(param2.data,out = [param1.data])

        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):
            continue

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache = True` is incompatible with gradient checkpointing`. Setting `use_cache = False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, use_cache)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                head_mask[idx] if head_mask is not None else None,
                cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                None,
            )
        else:
            if breakmodel:
                device = primary_device if i < ram_blocks else bisect.bisect_right(cumulative_gpu_blocks, i - ram_blocks)
            layer_outputs = decoder_layer(
                hidden_states.to(device) if breakmodel and hidden_states is not None else hidden_states,
                attention_mask=attention_mask.to(device) if breakmodel and attention_mask is not None else attention_mask,
                encoder_hidden_states=encoder_hidden_states.to(device) if breakmodel and encoder_hidden_states is not None else encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask.to(device) if breakmodel and encoder_attention_mask is not None else encoder_attention_mask,
                layer_head_mask=((head_mask[idx].to(device) if breakmodel and head_mask[idx] is not None else head_mask[idx]) if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    (cross_attn_head_mask[idx].to(device) if breakmodel and cross_attn_head_mask[idx] is not None else cross_attn_head_mask[idx]) if cross_attn_head_mask is not None else None
                ),
                past_key_value=tuple(v.to(device) for v in past_key_value if v is not None) if breakmodel and past_key_value is not None and i >= ram_blocks and len(past_key_value) and past_key_value[0].device.index != device else past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)
        
        if breakmodel:
            if i in range(ram_blocks):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    if breakmodel:
        if ram_blocks:
            del copystream
        torch.cuda.empty_cache()
        hidden_states = hidden_states.to(primary_device)
    hidden_states = self.layer_norm(hidden_states)
    if breakmodel:
        hidden_states = hidden_states.to(primary_device)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        cross_attentions=all_cross_attentions,
    )


def new_forward_opt(
    self,
    input_ids=None,
    attention_mask=None,
    head_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    assert len(gpu_blocks) <= torch.cuda.device_count()
    assert sum(gpu_blocks) <= len(self.layers)
    ram_blocks = len(self.layers) - sum(gpu_blocks)
    cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))


    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if inputs_embeds is None:
        if breakmodel:
            input_ids = input_ids.to(primary_device) 
        inputs_embeds = self.embed_tokens(input_ids)

    # embed positions
    if breakmodel:
        inputs_embeds = inputs_embeds.to(primary_device) 
    if attention_mask is None:
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)

    positions = self.embed_positions(attention_mask)[:, past_key_values_length:, :]
    if breakmodel:
        positions = positions.to(primary_device) 

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )

    if self.project_in is not None:
        inputs_embeds = self.project_in(inputs_embeds)

    hidden_states = inputs_embeds + positions

    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    if breakmodel and ram_blocks:
        copystream = torch.cuda.Stream(device=primary_device, priority=-1)

    # check if head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
        if attn_mask is not None:
            if attn_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

    for idx, decoder_layer in enumerate(self.layers):
        i = idx
        if breakmodel:
            if i in range(ram_blocks):
                index1 = (i+1)%ram_blocks
                for param1,param2 in zip(self.layers[index1].parameters(),self.layers[(i-1)%ram_blocks].parameters()):
                    param1.data = param2.data
                for param1,param2 in zip(self.layers[index1].parameters(),self.extrastorage[index1].parameters()):
                    with torch.cuda.stream(copystream):
                        torch.cuda.comm.broadcast(param2.data,out = [param1.data])

        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        dropout_probability = random.uniform(0, 1)
        if self.training and (dropout_probability < self.layerdrop):
            continue

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                head_mask[idx] if head_mask is not None else None,
                None,
            )
        else:
            if breakmodel:
                device = primary_device if i < ram_blocks else bisect.bisect_right(cumulative_gpu_blocks, i - ram_blocks)
            layer_outputs = decoder_layer(
                hidden_states.to(device) if breakmodel and hidden_states is not None else hidden_states,
                attention_mask=attention_mask.to(device) if breakmodel and attention_mask is not None else attention_mask,
                layer_head_mask=((head_mask[idx].to(device) if breakmodel and head_mask[idx] is not None else head_mask[idx]) if head_mask is not None else None),
                past_key_value=tuple(v.to(device) for v in past_key_value if v is not None) if breakmodel and past_key_value is not None and i >= ram_blocks and len(past_key_value) and past_key_value[0].device.index != device else past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
        
        if breakmodel:
            if i in range(ram_blocks):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    if breakmodel:
        if ram_blocks:
            del copystream
        torch.cuda.empty_cache()
        hidden_states = hidden_states.to(primary_device)
    if self.project_out is not None:
        hidden_states = self.project_out(hidden_states)
    if breakmodel:
        hidden_states = hidden_states.to(primary_device)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
