# TODO:
# - Intertwine stoppers and streaming and such
# - Add raw_generate functions to this
# - Support TPU
# - Support APIs
# - Support RWKV

import bisect
import gc
import shutil
import contextlib
import functools
import itertools
import json
import os
import traceback
import zipfile
import utils
import breakmodel

import torch
from torch.nn import Embedding

from tqdm.auto import tqdm
from logger import logger
import torch_lazy_loader
from typing import Dict, List, Optional, Union
from transformers import StoppingCriteria, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, GPTNeoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, modeling_utils, AutoModelForTokenClassification, AutoConfig

# Previously under condition HAS_ACCELERATE, but I'm quite sure accelerate
# is now a dependency.
import accelerate.utils

import koboldai_settings

class InferenceModel:
    def __init__(self) -> None:
        self.gen_config = {}
        self.token_gen_hooks = []

    def generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new_tokens: int,
        do_streaming: bool = False,
        do_dynamic_wi: bool = False,
        single_line: bool = False,
        batch_count: int = 1,
    ) -> torch.Tensor:
        raise NotImplementedError("generate() was not overridden")

    def _post_token_gen(self, input_ids: torch.LongTensor) -> None:
        for hook in self.token_gen_hooks:
            hook(input_ids)


class HFTorchInferenceModel:
    def __init__(
        self,
        model_name: str,
        lazy_load: bool,
        low_mem: bool,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.lazy_load = lazy_load
        self.low_mem = low_mem

        self.model = None
        self.tokenizer = None
        self.model_config = None

    def generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new_tokens: int,
        do_streaming: bool = False,
        do_dynamic_wi: bool = False,
        single_line: bool = False,
        batch_count: int = 1,
    ) -> torch.Tensor:
        raise NotImplementedError("AHHHH")

        self.gen_config = {
            "do_streaming": do_streaming,
            "do_dynamic_wi": do_dynamic_wi,
            "stop_at_genamt": do_dynamic_wi,
        }
    
    def _get_model(self, location: str, tf_kwargs: Dict):
        try:
            return AutoModelForCausalLM.from_pretrained(
                location,
                revision=utils.koboldai_vars.revision,
                cache_dir="cache",
                **tf_kwargs
            )
        except Exception as e:
            if "out of memory" in traceback.format_exc().lower():
                raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
            return GPTNeoForCausalLM.from_pretrained(
                location,
                revision=utils.koboldai_vars.revision,
                cache_dir="cache",
                **tf_kwargs
            )
    
    def _get_tokenizer(self, location: str):
        std_kwargs = {"revision": utils.koboldai_vars.revision, "cache_dir": "cache"}

        suppliers = [
            # Fast tokenizer disabled by default as per HF docs:
            # > Note: Make sure to pass use_fast=False when loading
            #   OPT’s tokenizer with AutoTokenizer to get the correct 
            #   tokenizer.
            lambda: AutoTokenizer.from_pretrained(location, use_fast=False, **std_kwargs),
            lambda: AutoTokenizer.from_pretrained(location, **std_kwargs),

            # Fallback to GPT2Tokenizer
            lambda: GPT2Tokenizer.from_pretrained(location, **std_kwargs),
            lambda: GPT2Tokenizer.from_pretrained("gpt2", **std_kwargs),
        ]

        for i, try_get_tokenizer in enumerate(suppliers):
            try:
                return try_get_tokenizer()
            except Exception as e:
                # If we error on each attempt, raise the last one
                if i == len(suppliers) - 1:
                    raise e
    
    def get_local_model_path(
        self,
        legacy: bool = False,
        ignore_existance: bool = False
    ) -> Optional[str]:
        """
        Returns a string of the model's path locally, or None if it is not downloaded.
        If ignore_existance is true, it will always return a path.
        """

        basename = utils.koboldai_vars.model.replace("/", "_")
        if legacy:
            ret = basename
        else:
            ret = os.path.join("models", basename)
        
        if os.path.isdir(ret) or ignore_existance:
            return ret
        return None

    
    def get_hidden_size(self) -> int:
        return self.model.get_input_embeddings().embedding_dim


    def _move_to_devices(self) -> None:
        if not utils.koboldai_vars.breakmodel:
            if utils.koboldai_vars.usegpu:
                self.model = self.model.half().to(utils.koboldai_vars.gpu_device)
            else:
                self.model = self.model.to('cpu').float()
            return

        for key, value in self.model.state_dict().items():
            target_dtype = torch.float32 if breakmodel.primary_device == "cpu" else torch.float16
            if value.dtype is not target_dtype:
                accelerate.utils.set_module_tensor_to_device(self.model, key, target_dtype)

        disk_blocks = breakmodel.disk_blocks
        gpu_blocks = breakmodel.gpu_blocks
        ram_blocks = len(utils.layers_module_names) - sum(gpu_blocks)
        cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))
        device_map = {}

        for name in utils.layers_module_names:
            layer = int(name.rsplit(".", 1)[1])
            device = ("disk" if layer < disk_blocks else "cpu") if layer < ram_blocks else bisect.bisect_right(cumulative_gpu_blocks, layer - ram_blocks)
            device_map[name] = device

        for name in utils.get_missing_module_names(self.model, list(device_map.keys())):
            device_map[name] = breakmodel.primary_device

        breakmodel.dispatch_model_ex(
            self.model,
            device_map,
            main_device=breakmodel.primary_device,
            offload_buffers=True,
            offload_dir="accelerate-disk-cache"
        )

        gc.collect()
        return

        # == Old non-accelerate stuff
        # model.half()
        # gc.collect()

        # if(hasattr(model, "transformer")):
        #     model.transformer.wte.to(breakmodel.primary_device)
        #     model.transformer.ln_f.to(breakmodel.primary_device)
        #     if(hasattr(model, 'lm_head')):
        #         model.lm_head.to(breakmodel.primary_device)
        #     if(hasattr(model.transformer, 'wpe')):
        #         model.transformer.wpe.to(breakmodel.primary_device)
        # elif(not hasattr(model.model, "decoder")):
        #     model.model.embed_tokens.to(breakmodel.primary_device)
        #     model.model.layer_norm.to(breakmodel.primary_device)
        #     model.lm_head.to(breakmodel.primary_device)
        #     model.model.embed_positions.to(breakmodel.primary_device)
        # else:
        #     model.model.decoder.embed_tokens.to(breakmodel.primary_device)
        #     if(model.model.decoder.project_in is not None):
        #         model.model.decoder.project_in.to(breakmodel.primary_device)
        #     if(model.model.decoder.project_out is not None):
        #         model.model.decoder.project_out.to(breakmodel.primary_device)
        #     model.model.decoder.embed_positions.to(breakmodel.primary_device)
        # gc.collect()
        # GPTNeoModel.forward = breakmodel.new_forward_neo
        # if("GPTJModel" in globals()):
        #     GPTJModel.forward = breakmodel.new_forward_neo # type: ignore
        # if("XGLMModel" in globals()):
        #     XGLMModel.forward = breakmodel.new_forward_xglm # type: ignore
        # if("OPTDecoder" in globals()):
        #     OPTDecoder.forward = breakmodel.new_forward_opt # type: ignore
        # generator = model.generate
        # if(hasattr(model, "transformer")):
        #     breakmodel.move_hidden_layers(model.transformer)
        # elif(not hasattr(model.model, "decoder")):
        #     breakmodel.move_hidden_layers(model.model, model.model.layers)
        # else:
        #     breakmodel.move_hidden_layers(model.model.decoder, model.model.decoder.layers)
    
    # Function to patch transformers to use our soft prompt
    def patch_embedding(self) -> None:
        if getattr(Embedding, "_koboldai_patch_causallm_model", None):
            Embedding._koboldai_patch_causallm_model = self.model
            return

        old_embedding_call = Embedding.__call__

        kai_model = self
        def new_embedding_call(self, input_ids, *args, **kwargs):
            # Don't touch embeddings for models other than the core inference model (that's us!)
            if Embedding._koboldai_patch_causallm_model.get_input_embeddings() is not self:
                return old_embedding_call(self, input_ids, *args, **kwargs)

            assert input_ids is not None

            if utils.koboldai_vars.sp is not None:
                shifted_input_ids = input_ids - kai_model.model.config.vocab_size

            input_ids.clamp_(max=kai_model.model.config.vocab_size - 1)
            inputs_embeds = old_embedding_call(self, input_ids, *args, **kwargs)

            if utils.koboldai_vars.sp is not None:
                utils.koboldai_vars.sp = utils.koboldai_vars.sp.to(inputs_embeds.dtype).to(inputs_embeds.device)
                inputs_embeds = torch.where(
                    (shifted_input_ids >= 0)[..., None],
                    utils.koboldai_vars.sp[shifted_input_ids.clamp(min=0)],
                    inputs_embeds,
                )

            return inputs_embeds

        Embedding.__call__ = new_embedding_call
        Embedding._koboldai_patch_causallm_model = self.model


    def _get_lazy_load_callback(self, n_layers: int, convert_to_float16: bool = True):
        if not self.lazy_load:
            return

        if utils.args.breakmodel_disklayers is not None:
            breakmodel.disk_blocks = utils.args.breakmodel_disklayers

        disk_blocks = breakmodel.disk_blocks
        gpu_blocks = breakmodel.gpu_blocks
        ram_blocks = ram_blocks = n_layers - sum(gpu_blocks)
        cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))

        def lazy_load_callback(
            model_dict: Dict[str, Union[torch_lazy_loader.LazyTensor, torch.Tensor]],
            f,
            **_,
        ):
            if lazy_load_callback.nested:
                return
            lazy_load_callback.nested = True

            device_map: Dict[str, Union[str, int]] = {}

            @functools.lru_cache(maxsize=None)
            def get_original_key(key):
                return max(
                    (
                        original_key
                        for original_key in utils.module_names
                        if original_key.endswith(key)
                    ),
                    key=len,
                )

            for key, value in model_dict.items():
                original_key = get_original_key(key)
                if isinstance(value, torch_lazy_loader.LazyTensor) and not any(
                    original_key.startswith(n) for n in utils.layers_module_names
                ):
                    device_map[key] = (
                        utils.koboldai_vars.gpu_device
                        if utils.koboldai_vars.hascuda and utils.koboldai_vars.usegpu
                        else "cpu"
                        if not utils.koboldai_vars.hascuda or not utils.koboldai_vars.breakmodel
                        else breakmodel.primary_device
                    )
                else:
                    layer = int(
                        max(
                            (
                                n
                                for n in utils.layers_module_names
                                if original_key.startswith(n)
                            ),
                            key=len,
                        ).rsplit(".", 1)[1]
                    )
                    device = (
                        utils.koboldai_vars.gpu_device
                        if utils.koboldai_vars.hascuda and utils.koboldai_vars.usegpu
                        else "disk"
                        if layer < disk_blocks and layer < ram_blocks
                        else "cpu"
                        if not utils.koboldai_vars.hascuda or not utils.koboldai_vars.breakmodel
                        else "shared"
                        if layer < ram_blocks
                        else bisect.bisect_right(
                            cumulative_gpu_blocks, layer - ram_blocks
                        )
                    )
                    device_map[key] = device

            if utils.num_shards is None or utils.current_shard == 0:
                utils.offload_index = {}
                if utils.HAS_ACCELERATE:
                    if os.path.isdir("accelerate-disk-cache"):
                        # Delete all of the files in the disk cache folder without deleting the folder itself to allow people to create symbolic links for this folder
                        # (the folder doesn't contain any subfolders so os.remove will do just fine)
                        for filename in os.listdir("accelerate-disk-cache"):
                            try:
                                os.remove(
                                    os.path.join("accelerate-disk-cache", filename)
                                )
                            except OSError:
                                pass
                    os.makedirs("accelerate-disk-cache", exist_ok=True)
                if utils.num_shards is not None:
                    num_tensors = len(
                        utils.get_sharded_checkpoint_num_tensors(
                            utils.from_pretrained_model_name,
                            utils.from_pretrained_index_filename,
                            **utils.from_pretrained_kwargs,
                        )
                    )
                else:
                    num_tensors = len(device_map)
                print(flush=True)
                utils.koboldai_vars.status_message = "Loading model"
                utils.koboldai_vars.total_layers = num_tensors
                utils.koboldai_vars.loaded_layers = 0
                utils.bar = tqdm(
                    total=num_tensors,
                    desc="Loading model tensors",
                    file=utils.UIProgressBarFile(),
                )

            with zipfile.ZipFile(f, "r") as z:
                try:
                    last_storage_key = None
                    zipfolder = os.path.basename(os.path.normpath(f)).split(".")[0]
                    f = None
                    current_offset = 0
                    able_to_pin_layers = True
                    if utils.num_shards is not None:
                        utils.current_shard += 1
                    for key in sorted(
                        device_map.keys(),
                        key=lambda k: (model_dict[k].key, model_dict[k].seek_offset),
                    ):
                        storage_key = model_dict[key].key
                        if (
                            storage_key != last_storage_key
                            or model_dict[key].seek_offset < current_offset
                        ):
                            last_storage_key = storage_key
                            if isinstance(f, zipfile.ZipExtFile):
                                f.close()
                            try:
                                f = z.open(f"archive/data/{storage_key}")
                            except:
                                f = z.open(f"{zipfolder}/data/{storage_key}")
                            current_offset = 0
                        if current_offset != model_dict[key].seek_offset:
                            f.read(model_dict[key].seek_offset - current_offset)
                            current_offset = model_dict[key].seek_offset
                        device = device_map[key]
                        size = functools.reduce(
                            lambda x, y: x * y, model_dict[key].shape, 1
                        )
                        dtype = model_dict[key].dtype
                        nbytes = (
                            size
                            if dtype is torch.bool
                            else size
                            * (
                                (
                                    torch.finfo
                                    if dtype.is_floating_point
                                    else torch.iinfo
                                )(dtype).bits
                                >> 3
                            )
                        )
                        # print(f"Transferring <{key}>  to  {f'({device.upper()})' if isinstance(device, str) else '[device ' + str(device) + ']'} ... ", end="", flush=True)
                        model_dict[key] = model_dict[key].materialize(
                            f, map_location="cpu"
                        )
                        if model_dict[key].dtype is torch.float32:
                            utils.koboldai_vars.fp32_model = True
                        if (
                            convert_to_float16
                            and breakmodel.primary_device != "cpu"
                            and utils.koboldai_vars.hascuda
                            and (utils.koboldai_vars.breakmodel or utils.koboldai_vars.usegpu)
                            and model_dict[key].dtype is torch.float32
                        ):
                            model_dict[key] = model_dict[key].to(torch.float16)
                        if breakmodel.primary_device == "cpu" or (
                            not utils.koboldai_vars.usegpu
                            and not utils.koboldai_vars.breakmodel
                            and model_dict[key].dtype is torch.float16
                        ):
                            model_dict[key] = model_dict[key].to(torch.float32)
                        if device == "shared":
                            model_dict[key] = model_dict[key].to("cpu").detach_()
                            if able_to_pin_layers and utils.HAS_ACCELERATE:
                                try:
                                    model_dict[key] = model_dict[key].pin_memory()
                                except:
                                    able_to_pin_layers = False
                        elif device == "disk":
                            accelerate.utils.offload_weight(
                                model_dict[key],
                                get_original_key(key),
                                "accelerate-disk-cache",
                                index=utils.offload_index,
                            )
                            model_dict[key] = model_dict[key].to("meta")
                        else:
                            model_dict[key] = model_dict[key].to(device)
                        # print("OK", flush=True)
                        current_offset += nbytes
                        utils.bar.update(1)
                        utils.koboldai_vars.loaded_layers += 1
                finally:
                    if (
                        utils.num_shards is None
                        or utils.current_shard >= utils.num_shards
                    ):
                        if utils.offload_index:
                            for name, tensor in utils.named_buffers:
                                dtype = tensor.dtype
                                if (
                                    convert_to_float16
                                    and breakmodel.primary_device != "cpu"
                                    and utils.koboldai_vars.hascuda
                                    and (
                                        utils.koboldai_vars.breakmodel or utils.koboldai_vars.usegpu
                                    )
                                ):
                                    dtype = torch.float16
                                if breakmodel.primary_device == "cpu" or (
                                    not utils.koboldai_vars.usegpu
                                    and not utils.koboldai_vars.breakmodel
                                ):
                                    dtype = torch.float32
                                if (
                                    name in model_dict
                                    and model_dict[name].dtype is not dtype
                                ):
                                    model_dict[name] = model_dict[name].to(dtype)
                                if tensor.dtype is not dtype:
                                    tensor = tensor.to(dtype)
                                if name not in utils.offload_index:
                                    accelerate.utils.offload_weight(
                                        tensor,
                                        name,
                                        "accelerate-disk-cache",
                                        index=utils.offload_index,
                                    )
                            accelerate.utils.save_offload_index(
                                utils.offload_index, "accelerate-disk-cache"
                            )
                        utils.bar.close()
                        utils.bar = None
                        utils.koboldai_vars.status_message = ""
                    lazy_load_callback.nested = False
                    if isinstance(f, zipfile.ZipExtFile):
                        f.close()

        lazy_load_callback.nested = False
        return lazy_load_callback

    @contextlib.contextmanager
    def _maybe_use_float16(self, always_use: bool = False):
        if always_use or (utils.koboldai_vars.hascuda and self.low_mem and (utils.koboldai_vars.usegpu or utils.koboldai_vars.breakmodel)):
            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float16)
            yield True
            torch.set_default_dtype(original_dtype)
        else:
            yield False

    def breakmodel_device_list(self, n_layers, primary=None, selected=None):
        # TODO: Find a better place for this or rework this

        # HACK: Tttttttterrrible structure_hack
        class colors:
            PURPLE    = '\033[95m'
            BLUE      = '\033[94m'
            CYAN      = '\033[96m'
            GREEN     = '\033[92m'
            YELLOW    = '\033[93m'
            RED       = '\033[91m'
            END       = '\033[0m'
            UNDERLINE = '\033[4m'

        device_count = torch.cuda.device_count()
        if(device_count < 2):
            primary = None
        gpu_blocks = breakmodel.gpu_blocks + (device_count - len(breakmodel.gpu_blocks))*[0]
        print(f"{colors.YELLOW}       DEVICE ID  |  LAYERS  |  DEVICE NAME{colors.END}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            if(len(name) > 47):
                name = "..." + name[-44:]
            row_color = colors.END
            sep_color = colors.YELLOW
            print(f"{row_color}{colors.YELLOW + '->' + row_color if i == selected else '  '} {'(primary)' if i == primary else ' '*9} {i:3}  {sep_color}|{row_color}     {gpu_blocks[i]:3}  {sep_color}|{row_color}  {name}{colors.END}")
        row_color = colors.END
        sep_color = colors.YELLOW
        if(utils.HAS_ACCELERATE):
            print(f"{row_color}{colors.YELLOW + '->' + row_color if -1 == selected else '  '} {' '*9} N/A  {sep_color}|{row_color}     {breakmodel.disk_blocks:3}  {sep_color}|{row_color}  (Disk cache){colors.END}")
        print(f"{row_color}   {' '*9} N/A  {sep_color}|{row_color}     {n_layers:3}  {sep_color}|{row_color}  (CPU){colors.END}")

    def breakmodel_device_config(self, config):
        # TODO: Find a better place for this or rework this

        # HACK: Tttttttterrrible structure_hack
        class colors:
            PURPLE    = '\033[95m'
            BLUE      = '\033[94m'
            CYAN      = '\033[96m'
            GREEN     = '\033[92m'
            YELLOW    = '\033[93m'
            RED       = '\033[91m'
            END       = '\033[0m'
            UNDERLINE = '\033[4m'

        global breakmodel, generator
        import breakmodel
        n_layers = utils.num_layers(config)

        if utils.args.cpu:
            breakmodel.gpu_blocks = [0]*n_layers
            return

        elif(utils.args.breakmodel_gpulayers is not None or (utils.HAS_ACCELERATE and utils.args.breakmodel_disklayers is not None)):
            try:
                if(not utils.args.breakmodel_gpulayers):
                    breakmodel.gpu_blocks = []
                else:
                    breakmodel.gpu_blocks = list(map(int, utils.args.breakmodel_gpulayers.split(',')))
                assert len(breakmodel.gpu_blocks) <= torch.cuda.device_count()
                s = n_layers
                for i in range(len(breakmodel.gpu_blocks)):
                    if(breakmodel.gpu_blocks[i] <= -1):
                        breakmodel.gpu_blocks[i] = s
                        break
                    else:
                        s -= breakmodel.gpu_blocks[i]
                assert sum(breakmodel.gpu_blocks) <= n_layers
                n_layers -= sum(breakmodel.gpu_blocks)
                if(utils.args.breakmodel_disklayers is not None):
                    assert utils.args.breakmodel_disklayers <= n_layers
                    breakmodel.disk_blocks = utils.args.breakmodel_disklayers
                    n_layers -= utils.args.breakmodel_disklayers
            except:
                logger.warning("--breakmodel_gpulayers is malformatted. Please use the --help option to see correct usage of --breakmodel_gpulayers. Defaulting to all layers on device 0.")
                breakmodel.gpu_blocks = [n_layers]
                n_layers = 0
        elif(utils.args.breakmodel_layers is not None):
            breakmodel.gpu_blocks = [n_layers - max(0, min(n_layers, utils.args.breakmodel_layers))]
            n_layers -= sum(breakmodel.gpu_blocks)
        elif(utils.args.model is not None):
            logger.info("Breakmodel not specified, assuming GPU 0")
            breakmodel.gpu_blocks = [n_layers]
            n_layers = 0
        else:
            device_count = torch.cuda.device_count()
            if(device_count > 1):
                print(colors.CYAN + "\nPlease select one of your GPUs to be your primary GPU.")
                print("VRAM usage in your primary GPU will be higher than for your other ones.")
                print("It is recommended you make your fastest GPU your primary GPU.")
                self.breakmodel_device_list(n_layers)
                while(True):
                    primaryselect = input("device ID> ")
                    if(primaryselect.isnumeric() and 0 <= int(primaryselect) < device_count):
                        breakmodel.primary_device = int(primaryselect)
                        break
                    else:
                        print(f"{colors.RED}Please enter an integer between 0 and {device_count-1}.{colors.END}")
            else:
                breakmodel.primary_device = 0

            print(colors.PURPLE + "\nIf you don't have enough VRAM to run the model on a single GPU")
            print("you can split the model between your CPU and your GPU(s), or between")
            print("multiple GPUs if you have more than one.")
            print("By putting more 'layers' on a GPU or CPU, more computations will be")
            print("done on that device and more VRAM or RAM will be required on that device")
            print("(roughly proportional to number of layers).")
            print("It should be noted that GPUs are orders of magnitude faster than the CPU.")
            print(f"This model has{colors.YELLOW} {n_layers} {colors.PURPLE}layers.{colors.END}\n")

            for i in range(device_count):
                self.breakmodel_device_list(n_layers, primary=breakmodel.primary_device, selected=i)
                print(f"{colors.CYAN}\nHow many of the remaining{colors.YELLOW} {n_layers} {colors.CYAN}layers would you like to put into device {i}?\nYou can also enter -1 to allocate all remaining layers to this device.{colors.END}\n")
                while(True):
                    layerselect = input("# of layers> ")
                    if((layerselect.isnumeric() or layerselect.strip() == '-1') and -1 <= int(layerselect) <= n_layers):
                        layerselect = int(layerselect)
                        layerselect = n_layers if layerselect == -1 else layerselect
                        breakmodel.gpu_blocks.append(layerselect)
                        n_layers -= layerselect
                        break
                    else:
                        print(f"{colors.RED}Please enter an integer between -1 and {n_layers}.{colors.END}")
                if(n_layers == 0):
                    break

            if(utils.HAS_ACCELERATE and n_layers > 0):
                self.breakmodel_device_list(n_layers, primary=breakmodel.primary_device, selected=-1)
                print(f"{colors.CYAN}\nHow many of the remaining{colors.YELLOW} {n_layers} {colors.CYAN}layers would you like to put into the disk cache?\nYou can also enter -1 to allocate all remaining layers to this device.{colors.END}\n")
                while(True):
                    layerselect = input("# of layers> ")
                    if((layerselect.isnumeric() or layerselect.strip() == '-1') and -1 <= int(layerselect) <= n_layers):
                        layerselect = int(layerselect)
                        layerselect = n_layers if layerselect == -1 else layerselect
                        breakmodel.disk_blocks = layerselect
                        n_layers -= layerselect
                        break
                    else:
                        print(f"{colors.RED}Please enter an integer between -1 and {n_layers}.{colors.END}")

        logger.init_ok("Final device configuration:", status="Info")
        self.breakmodel_device_list(n_layers, primary=breakmodel.primary_device)

        # If all layers are on the same device, use the old GPU generation mode
        while(len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] == 0):
            breakmodel.gpu_blocks.pop()
        if(len(breakmodel.gpu_blocks) and breakmodel.gpu_blocks[-1] in (-1, utils.num_layers(config))):
            utils.koboldai_vars.breakmodel = False
            utils.koboldai_vars.usegpu = True
            utils.koboldai_vars.gpu_device = len(breakmodel.gpu_blocks)-1
            return

        if(not breakmodel.gpu_blocks):
            logger.warning("Nothing assigned to a GPU, reverting to CPU only mode")
            import breakmodel
            breakmodel.primary_device = "cpu"
            utils.koboldai_vars.breakmodel = False
            utils.koboldai_vars.usegpu = False
            return


class GenericHFTorchInferenceModel(HFTorchInferenceModel):
    def _load(self, save_model: bool) -> None:
        utils.koboldai_vars.allowsp = True

        # Make model path the same as the model name to make this consistent
        # with the other loading method if it isn't a known model type. This
        # code is not just a workaround for below, it is also used to make the
        # behavior consistent with other loading methods - Henk717
        # if utils.koboldai_vars.model not in ["NeoCustom", "GPT2Custom"]:
        #     utils.koboldai_vars.custmodpth = utils.koboldai_vars.model

        if utils.koboldai_vars.model == "NeoCustom":
            utils.koboldai_vars.model = os.path.basename(os.path.normpath(utils.koboldai_vars.custmodpth))

        # If we specify a model and it's in the root directory, we need to move
        # it to the models directory (legacy folder structure to new)
        if self.get_local_model_path(legacy=True):
            shutil.move(
                self.get_local_model_path(legacy=True, ignore_existance=True),
                self.get_local_model_path(ignore_existance=True)
            )
        
        # Get the model_type from the config or assume a model type if it isn't present
        try:
            model_config = AutoConfig.from_pretrained(self.get_local_model_path() or utils.koboldai_vars.model, revision=utils.koboldai_vars.revision, cache_dir="cache")
            utils.koboldai_vars.model_type = model_config.model_type
        except ValueError as e:
            utils.koboldai_vars.model_type = {
                "NeoCustom": "gpt_neo",
                "GPT2Custom": "gpt2",
            }.get(utils.koboldai_vars.model)

            if not utils.koboldai_vars.model_type:
                logger.warning("No model type detected, assuming Neo (If this is a GPT2 model use the other menu option or --model GPT2Custom)")
                utils.koboldai_vars.model_type = "gpt_neo"


        tf_kwargs = {
            "low_cpu_mem_usage": True,
        }

        if utils.koboldai_vars.model_type == "gpt2":
            # We must disable low_cpu_mem_usage and if using a GPT-2 model
            # because GPT-2 is not compatible with this feature yet.
            tf_kwargs.pop("low_cpu_mem_usage", None)

            # Also, lazy loader doesn't support GPT-2 models
            utils.koboldai_vars.lazy_load = False
        
        # If we're using torch_lazy_loader, we need to get breakmodel config
        # early so that it knows where to load the individual model tensors
        if utils.koboldai_vars.lazy_load and utils.koboldai_vars.hascuda and utils.koboldai_vars.breakmodel and not utils.koboldai_vars.nobreakmodel:
            self.breakmodel_device_config(model_config)

        if utils.koboldai_vars.lazy_load:
            # If we're using lazy loader, we need to figure out what the model's hidden layers are called
            with torch_lazy_loader.use_lazy_torch_load(dematerialized_modules=True, use_accelerate_init_empty_weights=True):
                try:
                    metamodel = AutoModelForCausalLM.from_config(model_config)
                except Exception as e:
                    metamodel = GPTNeoForCausalLM.from_config(model_config)
                utils.layers_module_names = utils.get_layers_module_names(metamodel)
                utils.module_names = list(metamodel.state_dict().keys())
                utils.named_buffers = list(metamodel.named_buffers(recurse=True))

        # Download model from Huggingface if it does not exist, otherwise load locally
        with self._maybe_use_float16(), torch_lazy_loader.use_lazy_torch_load(
            enable=utils.koboldai_vars.lazy_load,
            callback=self._get_lazy_load_callback(utils.num_layers(model_config)) if utils.koboldai_vars.lazy_load else None,
            dematerialized_modules=True
        ):
            if utils.koboldai_vars.lazy_load:
                # torch_lazy_loader.py and low_cpu_mem_usage can't be used at the same time
                tf_kwargs.pop("low_cpu_mem_usage", None)

            self.tokenizer = self._get_tokenizer(self.get_local_model_path())

            if self.get_local_model_path():
                # Model is stored locally, load it.
                self.model = self._get_model(self.get_local_model_path(), tf_kwargs)
            else:
                # Model not stored locally, we need to download it.

                # _rebuild_tensor patch for casting dtype and supporting LazyTensors
                old_rebuild_tensor = torch._utils._rebuild_tensor
                def new_rebuild_tensor(
                    storage: Union[torch_lazy_loader.LazyTensor, torch.Storage],
                    storage_offset,
                    shape,
                    stride
                ):
                    if not isinstance(storage, torch_lazy_loader.LazyTensor):
                        dtype = storage.dtype
                    else:
                        dtype = storage.storage_type.dtype
                        if not isinstance(dtype, torch.dtype):
                            dtype = storage.storage_type(0).dtype
                    if dtype is torch.float32 and len(shape) >= 2:
                        utils.koboldai_vars.fp32_model = True
                    return old_rebuild_tensor(storage, storage_offset, shape, stride)

                torch._utils._rebuild_tensor = new_rebuild_tensor
                self.model = self._get_model(utils.koboldai_vars.model, tf_kwargs)
                torch._utils._rebuild_tensor = old_rebuild_tensor

                if save_model:
                    self.tokenizer.save_pretrained(self.get_local_model_path(ignore_existance=True))

                    if utils.koboldai_vars.fp32_model and not breakmodel.disk_blocks:
                        # Use save_pretrained to convert fp32 models to fp16,
                        # unless we are using disk cache because save_pretrained
                        # is not supported in that case
                        model = model.half()
                        model.save_pretrained(self.get_local_model_path(ignore_existance=True), max_shard_size="500MiB")

                    else:
                        # For fp16 models, we can just copy the model files directly
                        import transformers.configuration_utils
                        import transformers.modeling_utils
                        import transformers.file_utils
                        import huggingface_hub

                        legacy = packaging.version.parse(transformers_version) < packaging.version.parse("4.22.0.dev0")
                        # Save the config.json
                        shutil.move(
                            os.path.realpath(huggingface_hub.hf_hub_download(
                                utils.koboldai_vars.model,
                                transformers.configuration_utils.CONFIG_NAME,
                                revision=utils.koboldai_vars.revision,
                                cache_dir="cache",
                                local_files_only=True,
                                legacy_cache_layout=legacy
                            )),
                            os.path.join(
                                self.get_local_model_path(ignore_existance=True),
                                transformers.configuration_utils.CONFIG_NAME
                            )
                        )

                        if utils.num_shards is None:
                            # Save the pytorch_model.bin or model.safetensors of an unsharded model
                            for possible_weight_name in [transformers.modeling_utils.WEIGHTS_NAME, "model.safetensors"]:
                                try:
                                    shutil.move(
                                        os.path.realpath(huggingface_hub.hf_hub_download(
                                            utils.koboldai_vars.model,
                                            possible_weight_name,
                                            revision=utils.koboldai_vars.revision,
                                            cache_dir="cache",
                                            local_files_only=True,
                                            legacy_cache_layout=legacy
                                        )),
                                        os.path.join(
                                            self.get_local_model_path(ignore_existance=True),
                                            possible_weight_name,
                                        )
                                    )
                                except Exception as e:
                                    if possible_weight_name == "model.safetensors":
                                        raise e
                        else:
                            # Handle saving sharded models

                            with open(utils.from_pretrained_index_filename) as f:
                                map_data = json.load(f)
                            filenames = set(map_data["weight_map"].values())
                            # Save the pytorch_model.bin.index.json of a sharded model
                            shutil.move(
                                os.path.realpath(utils.from_pretrained_index_filename),
                                os.path.join(
                                    self.get_local_model_path(ignore_existance=True),
                                    transformers.modeling_utils.WEIGHTS_INDEX_NAME
                                )
                            )
                            # Then save the pytorch_model-#####-of-#####.bin files
                            for filename in filenames:
                                shutil.move(
                                    os.path.realpath(huggingface_hub.hf_hub_download(
                                        utils.koboldai_vars.model,
                                        filename,
                                        revision=utils.koboldai_vars.revision,
                                        cache_dir="cache",
                                        local_files_only=True,
                                        legacy_cache_layout=legacy
                                    )),
                                    os.path.join(
                                        self.get_local_model_path(ignore_existance=True),
                                        filename
                                    )
                                )
                    shutil.rmtree("cache/")

        if utils.koboldai_vars.badwordsids is koboldai_settings.badwordsids_default and utils.koboldai_vars.model_type not in ("gpt2", "gpt_neo", "gptj"):
            utils.koboldai_vars.badwordsids = [[v] for k, v in self.tokenizer.get_vocab().items() if any(c in str(k) for c in "<>[]") if utils.koboldai_vars.newlinemode != "s" or str(k) != "</s>"]

        self.patch_embedding()

        if utils.koboldai_vars.hascuda:
            if utils.koboldai_vars.usegpu:
                # Use just VRAM
                model = model.half().to(utils.koboldai_vars.gpu_device)
            elif utils.koboldai_vars.breakmodel:
                # Use both RAM and VRAM (breakmodel)
                if not utils.koboldai_vars.lazy_load:
                    self.breakmodel_device_config(model.config)
                self._move_to_devices()
            elif breakmodel.disk_blocks > 0:
                # Use disk
                self._move_to_devices()
            elif breakmodel.disk_blocks > 0:
                self._move_to_devices()
            else:
                # Use CPU
                self.model = self.model.to('cpu').float()
        elif breakmodel.disk_blocks > 0:
            self._move_to_devices()
        else:
            self.model = self.model.to('cpu').float()
        utils.koboldai_vars.modeldim = self.get_hidden_size()


class CustomGPT2HFTorchInferenceModel(HFTorchInferenceModel):
    def _load(self, save_model: bool) -> None:
        utils.koboldai_vars.lazy_load = False

        model_path = None

        for possible_config_path in [
            utils.koboldai_vars.custmodpth,
            os.path.join("models", utils.koboldai_vars.custmodpth)
        ]:
            try:
                with open(os.path.join(possible_config_path, "config.json"), "r") as file:
                    # Unused?
                    self.model_config = json.load(file)
                model_path = possible_config_path
                break
            except FileNotFoundError:
                pass
        
        if not model_path:
            raise RuntimeError("Empty model_path!")

        with self._maybe_use_float16():
            try:
                self.model = GPT2LMHeadModel.from_pretrained(utils.koboldai_vars.custmodpth, revision=utils.koboldai_vars.revision, cache_dir="cache")
                self.tokenizer = GPT2Tokenizer.from_pretrained(utils.koboldai_vars.custmodpth, revision=utils.koboldai_vars.revision, cache_dir="cache")
            except Exception as e:
                if "out of memory" in traceback.format_exc().lower():
                    raise RuntimeError("One of your GPUs ran out of memory when KoboldAI tried to load your model.")
                raise e

        if save_model:
            self.model.save_pretrained(self.get_local_model_path(ignore_existance=True), max_shard_size="500MiB")
            self.tokenizer.save_pretrained(self.get_local_model_path(ignore_existance=True))

        utils.koboldai_vars.modeldim = self.get_hidden_size()

        # Is CUDA available? If so, use GPU, otherwise fall back to CPU
        if utils.koboldai_vars.hascuda and utils.koboldai_vars.usegpu:
            self.model = self.model.half().to(utils.koboldai_vars.gpu_device)
        else:
            self.model = self.model.to("cpu").float()

        self.patch_causal_lm()