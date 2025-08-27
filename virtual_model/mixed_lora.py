from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.module import Module, _EXTRA_STATE_KEY_SUFFIX

from peft.tuners.lora import LoraLayer, LoraModel
from peft.utils import _get_submodules
from transformers import GenerationConfig
from transformers.generation import LogitsProcessorList
from transformers.modeling_utils import PreTrainedModel
from warnings import warn

from contextlib import contextmanager

# Typing
from typing import Any, Callable, Dict, Iterator, List, Optional, OrderedDict, Tuple, Union

# Kernel Utils
from .kernel_utils import add_lora, BatchLenInfo

from punica.utils.kvcache import BatchedKvCache

class MixedLoraContext:
    def __init__(self):
        self._blen = None
        self._lora_ids = None
        self._lora_len = None
        self._segment = None
        self._train_idx = []
    
    def update(
        self,
        blen: Optional[BatchLenInfo] = None,
        lora_ids: Optional[List[int]] = None,
        lora_len: Optional[torch.Tensor] = None,
        segment: Optional[torch.Tensor] = None,
        train_idx: List[int] = []
    ):
        self._blen = blen
        self._lora_ids = lora_ids
        self._lora_len = lora_len
        self._segment = segment
        self._train_idx = train_idx
    
    @property
    def blen(self) -> Optional[BatchLenInfo]:
        return self._blen
    
    @property
    def lora_ids(self) -> Optional[List[int]]:
        return self._lora_ids

    @property
    def lora_len(self) -> Optional[torch.Tensor]:
        return self._lora_len
    
    @property
    def segment(self) -> Optional[torch.Tensor]:
        return self._segment
    
    @property
    def train_idx(self) -> List[int]:
        return self._train_idx

CURRENT_TRAINER_LORA_ID = 0

@contextmanager
def set_saving_lora_id(num):
    global CURRENT_TRAINER_LORA_ID
    original_lora_id = CURRENT_TRAINER_LORA_ID
    CURRENT_TRAINER_LORA_ID = num
    try:
        yield
    finally:
        CURRENT_TRAINER_LORA_ID = original_lora_id

@dataclass
class NamedLoraLayer:
    adapter_name: str
    wa: torch.Tensor
    wb: torch.Tensor
    dropout: Union[nn.Dropout, nn.Identity]
    scaling: Optional[float] = None

class MixedLoraLinear(nn.Module):
    def __init__(
        self,
        root_context: MixedLoraContext,
        base_layer: Any,
        adapters: List[NamedLoraLayer],
        rank: int,
        dtype: torch.dtype,
        is_scaled: bool = False
    ):
        super().__init__()
        self.root_context = root_context
        self.base_layer = base_layer
        self.device = base_layer.weight.device
        self.dtype = dtype
        self.is_scaled = is_scaled
        if is_scaled:
            self.scaling = None
        self.update_adapters(adapters, rank)
    
    def add_adapter(self, adapter: NamedLoraLayer):
        '''
        New adapter should have the same rank as the existing adapters.
        '''
        self.adapters.append(adapter)
        self.wa_ptrs = torch.cat((
            self.wa_ptrs, torch.tensor(
                [adapter.wa.data_ptr()],
                dtype=torch.int64,
                device=self.device
            )
        ))
        self.wb_ptrs = torch.cat((
            self.wb_ptrs, torch.tensor(
                [adapter.wb.data_ptr()],
                dtype=torch.int64,
                device=self.device
            )
        ))
        if not self.is_scaled:
            self.scaling = torch.cat((
                self.scaling, torch.tensor(
                    [adapter.scaling],
                    dtype=self.dtype,
                    device=self.device
                )
            ))

    def del_adapter(self, adapter_idx: int):
        assert adapter_idx < len(self.adapters)
        del self.adapters[adapter_idx]
        self.wa_ptrs = torch.tensor(
            [adapter.wa.data_ptr() for adapter in self.adapters],
            dtype=torch.int64, device=self.device
        )
        self.wb_ptrs = torch.tensor(
            [adapter.wb.data_ptr() for adapter in self.adapters],
            dtype=torch.int64, device=self.device
        )
        if not self.is_scaled:
            self.scaling = torch.tensor(
                [adapter.scaling for adapter in self.adapters],
                dtype=self.dtype, device=self.device
            )

    def update_adapters(self, adapters: List[NamedLoraLayer], rank: int):
        self.adapters = adapters
        self.rank = rank
        self.wa_ptrs = torch.tensor(
            [adapter.wa.data_ptr() for adapter in adapters],
            dtype=torch.int64, device=self.device
        )
        self.wb_ptrs = torch.tensor(
            [adapter.wb.data_ptr() for adapter in adapters],
            dtype=torch.int64, device=self.device
        )
        if not self.is_scaled:
            self.scaling = torch.tensor(
                [adapter.scaling for adapter in adapters],
                dtype=self.dtype, device=self.device
            )
    
    def set_trainable(self, train_idx: List[int]):
        for idx in train_idx:
            self.register_parameter(f'lora-{idx}-A', nn.Parameter(self.adapters[idx].wa))
            self.register_parameter(f'lora-{idx}-B', nn.Parameter(self.adapters[idx].wb))
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        blen = self.root_context.blen
        lora_ids = self.root_context.lora_ids
        lora_len = self.root_context.lora_len
        segment = self.root_context.segment.to(self.device)
        train_idx = self.root_context.train_idx
        wa_ptr, wb_ptr = self.wa_ptrs[lora_ids], self.wb_ptrs[lora_ids]

        x = x.to(self.dtype)
        if blen.tlen > 0:
            dropped_x = []
            for idx, pi in zip(train_idx, range(len(blen.train_partition) - 1)):
                start, end = blen.train_partition[pi], blen.train_partition[pi + 1]
                dropped_x.append(self.adapters[idx].dropout(
                    x[start:end, :]
                ))
            x = torch.cat(dropped_x + [x[blen.tlen:, :]])
        # lora_res = x @ self.get_parameter(f'lora-{0}-A') @ self.get_parameter(f'lora-{0}-B')
        # result = result + lora_res
        # return result

        if self.is_scaled:
            lora_res = add_lora.apply(
                result.shape, x, wa_ptr, wb_ptr,
                segment, self.rank, blen.tlen, blen.train_partition,
                *[w for idx in train_idx for w in (
                    self.get_parameter(f'lora-{idx}-A'),
                    self.get_parameter(f'lora-{idx}-B')
                )]
            )
        else:
            lora_res = add_lora.apply(
                result.shape, x, wa_ptr, wb_ptr,
                segment, self.rank, blen.tlen, blen.train_partition,
                *[w for idx in train_idx for w in (
                    self.get_parameter(f'lora-{idx}-A'),
                    self.get_parameter(f'lora-{idx}-B')
                )]
            ) * torch.repeat_interleave(
                self.scaling[lora_ids], lora_len
            ).unsqueeze(-1)
        result = result + lora_res
        return result.to(torch_result_dtype)
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        '''
        Rename lora modules from `lora-{idx}-[A|B]` to `lora_[A|B].weight`.
        '''
        for name, param in self._parameters.items():
            lora_info = name.split('-')
            lora_id, a_b = int(lora_info[1]), lora_info[2]
            if lora_id == CURRENT_TRAINER_LORA_ID and param is not None:
                destination[prefix + 'lora_' + a_b + '.weight'] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "get_extra_state", Module.get_extra_state) is not Module.get_extra_state:
            destination[extra_state_key] = self.get_extra_state()


class MixedLoraModel(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        lora_models: Dict[str, LoraModel],
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[torch.device, str]] = None,
        max_rank: int = 0,
        apply_scaling: Optional[str] = None,
        map_device: Optional[Dict[str, torch.device]] = None,
    ):
        '''
        *Note: `apply_scaling` should be 'A', 'B' or None.*
        '''
        super().__init__()
        self.model = base_model
        self.dtype = dtype
        self.rank = max_rank
        if apply_scaling is None or apply_scaling == 'A' or apply_scaling == 'B':
            self.apply_scaling = apply_scaling
        else:
            warn(RuntimeWarning(
                f'Param apply_scaling should be \'A\' or \'B\' or None, but got {apply_scaling}.\n'
                f'apply_scaling set to None.'
            ))
            self.apply_scaling = None
        self.map_device = map_device if map_device is not None else {}
        self.device = device if device is not None else base_model.device
        self.root_context = MixedLoraContext()
        self.adapter_lists = {}
        self.target_module_saves = {}
        self.used_loras = set()
        self.training_loras = set()
        self.lora_name_to_id = {}

        target_modules = {}
        for lora_name, lora_model in lora_models.items():
            self.adapter_lists[lora_name] = []
            self.rank = max(self.rank, lora_model.peft_config[lora_model.active_adapter].r)
            for target_module_name in lora_model.targeted_module_names:
                if target_module_name not in target_modules:
                    target_modules[target_module_name] = [lora_name]
                else:
                    target_modules[target_module_name].append(lora_name)
        self._update_lora_name_to_id()
        for target_module_name, lora_names in target_modules.items():
            lora_layers = [
                _get_submodules(lora_models[lora_name].model, target_module_name)[1]
                for lora_name in lora_names
            ]
            named_lora_layers = [NamedLoraLayer(
                lora_name,
                self._prepare_weight(
                    lora_layer.lora_A[lora_models[lora_name].active_adapter].weight,
                    self.map_device.get(lora_name, self.device),
                    scaling=lora_layer.scaling[lora_models[lora_name].active_adapter]
                    if self.apply_scaling == 'A' else None
                ),
                self._prepare_weight(
                    lora_layer.lora_B[lora_models[lora_name].active_adapter].weight,
                    self.map_device.get(lora_name, self.device),
                    rank_dim=1,
                    scaling=lora_layer.scaling[lora_models[lora_name].active_adapter]
                    if self.apply_scaling == 'B' else None
                ),
                lora_layer.lora_dropout[lora_models[lora_name].active_adapter],
                None if self.apply_scaling is None else
                lora_layer.scaling[lora_models[lora_name].active_adapter]
            ) for lora_name, lora_layer in zip(lora_names, lora_layers)]
            for named_lora_layer in named_lora_layers:
                self.adapter_lists[named_lora_layer.adapter_name].append(named_lora_layer)
            self.target_module_saves[target_module_name] = named_lora_layers
            self._create_and_replace(
                target_module_name, named_lora_layers
            )
    
    def _set_training_loras(self, lora_names: List[str]):
        self.training_loras = set(lora_names)
    
    def _update_lora_name_to_id(self):
        for i, lora_name in enumerate(self.adapter_lists.keys()):
            self.lora_name_to_id[lora_name] = i
    
    def _get_lora_id(self, lora_name: str) -> int:
        return self.lora_name_to_id.get(lora_name, 0)
    
    def _update_used_loras(self, lora_names: List[str]):
        self.used_loras = set(lora_names)
    
    def add_lora_models(
            self,
            lora_models: Dict[str, LoraModel],
            apply_scaling: Optional[str] = None
        ):
        if apply_scaling is None:
            apply_scaling = self.apply_scaling
        elif self.apply_scaling is None:
            warn(RuntimeWarning(
                f'Param apply_scaling should be None as MixedLoraModel.apply_scaling is None, '
                f'but got {apply_scaling}.\napply_scaling set to None.'
            ))
            apply_scaling = None
        target_modules = {}
        for lora_name, lora_model in lora_models.items():
            r = lora_model.peft_config[lora_model.active_adapter].r
            if self.rank < r:
                warn(RuntimeWarning(
                    f'Rank of added lora models should be lower than self.rank, but got {r}.\n'
                    f'Lora model named {lora_name} skipped.'
                ))
                continue
            self.adapter_lists[lora_name] = []
            for target_module_name in lora_model.targeted_module_names:
                if target_module_name not in target_modules:
                    target_modules[target_module_name] = [lora_name]
                else:
                    target_modules[target_module_name].append(lora_name)
        self._update_lora_name_to_id()
        for target_module_name, lora_names in target_modules.items():
            lora_layers = [
                _get_submodules(lora_models[lora_name].model, target_module_name)[1]
                for lora_name in lora_names
            ]
            named_lora_layers = [NamedLoraLayer(
                lora_name,
                self._prepare_weight(
                    lora_layer.lora_A[lora_models[lora_name].active_adapter].weight,
                    self.map_device.get(lora_name, self.device),
                    scaling=lora_layer.scaling[lora_models[lora_name].active_adapter]
                    if apply_scaling == 'A' else None
                ),
                self._prepare_weight(
                    lora_layer.lora_B[lora_models[lora_name].active_adapter].weight,
                    self.map_device.get(lora_name, self.device),
                    rank_dim=1,
                    scaling=lora_layer.scaling[lora_models[lora_name].active_adapter]
                    if apply_scaling == 'B' else None
                ),
                lora_layer.lora_dropout[lora_models[lora_name].active_adapter],
                None if apply_scaling is None else
                lora_layer.scaling[lora_models[lora_name].active_adapter]
            ) for lora_name, lora_layer in zip(lora_names, lora_layers)]
            self.target_module_saves[target_module_name] = sorted(
                filter(
                    lambda layer: layer.adapter_name not in lora_names,
                    self.target_module_saves[target_module_name] + named_lora_layers,
                ),
                key=lambda layer: self._get_lora_id(layer.adapter_name)
            )
            parent, target, target_name = _get_submodules(self.model, target_module_name)
            target.update_adapters(self.target_module_saves[target_module_name], self.rank)
    
    def remove_lora_models(self, lora_names: List[str]):
        self.adapter_lists = dict(sorted(
            filter(
                lambda item: item[0] not in lora_names,
                self.adapter_lists.items()
            ),
            key=lambda item: (0 if item[0] in self.training_loras else
                              1 if item[0] in self.used_loras else 2)
        ))
        for lora_name in lora_names:
            del self.lora_name_to_id[lora_name]
        self._update_lora_name_to_id()
        for target_module_name in self.target_module_saves.keys():
            self.target_module_saves[target_module_name] = sorted(
                filter(
                    lambda layer: layer.adapter_name not in lora_names,
                    self.target_module_saves[target_module_name],
                ),
                key=lambda layer: self._get_lora_id(layer.adapter_name)
            )
            parent, target, target_name = _get_submodules(self.model, target_module_name)
            target.update_adapters(self.target_module_saves[target_module_name], self.rank)
    
    def re_sort_loras(self):
        self.adapter_lists = dict(sorted(
            self.adapter_lists.items(),
            key=lambda item: (0 if item[0] in self.training_loras else
                              1 if item[0] in self.used_loras else 2)
        ))
        self._update_lora_name_to_id()
        for target_module_name in self.target_module_saves.keys():
            self.target_module_saves[target_module_name] = sorted(
                self.target_module_saves[target_module_name],
                key=lambda layer: self._get_lora_id(layer.adapter_name)
            )
            parent, target, target_name = _get_submodules(self.model, target_module_name)
            target.update_adapters(self.target_module_saves[target_module_name], self.rank)
    
    def set_trainable(self, train_idx: List[int]):
        for target_module_name in self.target_module_saves.keys():
            parent, target, target_name = _get_submodules(self.model, target_module_name)
            target.set_trainable(train_idx)
    
    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[List[torch.Tensor]] = None,
        **kwargs
    ):
        '''
        Prepare inputs for MixedLoraModel generation.

        Args:
            input_ids (torch.LongTensor): Input ids.
            attention_mask (Optional[List[torch.Tensor]]): Attention masks for training inputs. Defaults to None.
            **kwargs:
                - blen (BatchLenInfo): Batched input length info.
                - lora_ids (List[int]): Lora ids for input with the same lora.
                - lora_len (torch.Tensor): Each input length with the same lora.
                - segment (torch.Tensor): Segment for input length with the same lora.
                - train_idx (List[int]): Index of the lora being trained.
                - used_lora_names (Optional[List[str]]): Name list of lora that are going to be used. Defaults to None.
        '''
        blen = kwargs.get('blen')
        lora_ids = kwargs.get('lora_ids')
        lora_len = kwargs.get('lora_len')
        segment = kwargs.get('segment')
        train_idx = kwargs.get('train_idx')
        used_lora_names = kwargs.get('lora_names', [])
        for lora_name in used_lora_names:
            self.move_to_device(lora_name, self.device)
        self.root_context.update(
            blen, lora_ids, lora_len, segment, train_idx
        )
        return self.model.prepare_inputs_for_generation(
            input_ids, attention_mask, **kwargs
        )
    
    def _get_generation_mode(self, generation_config: GenerationConfig):
        return self.model._get_generation_mode(generation_config, None)
    
    def _get_logits_warper(self, *args, **kwargs) -> LogitsProcessorList:
        return self.model._get_logits_warper(*args, **kwargs)
    
    def _get_logits_processor(self, *args, **kwargs) -> LogitsProcessorList:
        return self.model._get_logits_processor(*args, **kwargs)

    def move_to_device(self, lora_name: str, device: torch.device = torch.device('cpu')):
        if lora_name not in self.map_device or self.map_device[lora_name] != device:
            self.map_device[lora_name] = device
            for named_lora_layer in self.adapter_lists[lora_name]:
                named_lora_layer.wa = named_lora_layer.wa.to(device)
                named_lora_layer.wb = named_lora_layer.wb.to(device)

    def _create_and_replace(
        self,
        target_module_name: Any,
        adapters: List[NamedLoraLayer]
    ):
        parent, target, target_name = _get_submodules(self.model, target_module_name)
        setattr(parent, target_name, MixedLoraLinear(
            self.root_context,
            target,
            adapters,
            self.rank,
            self.dtype,
            self.apply_scaling is not None
        ))

    def _prepare_weight(
            self,
            weight: torch.Tensor,
            device: torch.device,
            rank_dim: int = 0,
            scaling: Optional[float] = None,
        ):
        weight = weight.detach()
        if weight.size(rank_dim) == self.rank:
            if scaling is None:
                return weight.T.to(dtype=self.dtype, device=device).contiguous()
            else:
                return (weight.T * scaling).to(dtype=self.dtype, device=device).contiguous()
        new_size = list(weight.size())
        new_size[rank_dim] = self.rank
        new_weight = torch.zeros(new_size[::-1], dtype=self.dtype, device=device)
        if scaling is None:
            new_weight[:weight.size(1), :weight.size(0)].copy_(weight.T, non_blocking=True)
        else:
            new_weight[:weight.size(1), :weight.size(0)].copy_(weight.T * scaling, non_blocking=True)
        return new_weight
    
    def _prepare_special_tokens(self, *args, **kwargs):
        return self.model._prepare_special_tokens(*args, **kwargs)
    
    def get_nb_trainable_parameters(self):
        '''
        Copied from peft > peft_model.py > `PeftModel`.

        Returns the number of trainable parameters and number of all parameters in the model.
        '''
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self):
        '''
        Copied from peft > peft_model.py > `PeftModel`.

        Prints the number of trainable parameters in the model.
        '''
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )
    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        '''
        Mixed Lora Model only returns modules with `lora`, as a trick for saving lora linears only.
        '''
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default
            warn("Positional args are being deprecated, use kwargs instead. Refer to "
                 "https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
                 " for details.")

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        destination = OrderedDict(filter(lambda kv: 'lora' in kv[0], destination.items()))
        return destination


class MixedLoraLinearForTrainer(nn.Module):
    def __init__(self, base_linear: MixedLoraLinear, lora_id: int):
        self.base_linear = base_linear
        self.trainer_lora_id = lora_id

        self.root_context = self.base_linear.root_context
        self.base_layer = self.base_linear.base_layer
        self.device = self.base_linear.device
        self.dtype = self.base_linear.dtype
        self.is_scaled = self.base_linear.is_scaled
    
    def named_parameters(
        self,
        prefix: str = '',
        recurse: bool = True,
        remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        '''
        Mixed Lora Linear for Trainer filters lora parameters by lora id.
        '''
        gen = filter(
            lambda np: 'lora' not in np[0] or int(np[0].split('-')[1]) == self.trainer_lora_id,
            self._named_members(
                lambda module: module._parameters.items(),
                prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
            )
        )
        yield from gen
    
    def _save_to_state_dict(self, *args, **kwargs):
        return self.base_linear._save_to_state_dict(*args, **kwargs)


class MixedLoraModelForTrainer(nn.Module):
    def __init__(self, base_model: MixedLoraModel, lora_id: int):
        super().__init__()
        self.base_model = base_model
        self.trainer_lora_id = lora_id

        self.device = self.base_model.device
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        blen: Optional[BatchLenInfo] = None,
        attention_mask: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[List[torch.LongTensor]] = None,
        prefill_kv: Optional[BatchedKvCache] = None,
        decode_kv: Optional[BatchedKvCache] = None,
        labels: List[torch.LongTensor] = [],
        *args: Any,
        **kwargs: Any
    ):
        blen = BatchLenInfo([input_ids.shape], [], 0, self.device)
        self.base_model.root_context.update(
            blen, [0],
            torch.tensor([input_ids.shape[0] * input_ids.shape[1]], dtype=torch.int32, device=self.device),
            torch.tensor([0, input_ids.shape[0] * input_ids.shape[1]], dtype=torch.int32, device=self.device),
            [0]
        )
        input_ids = input_ids.view(-1)
        labels = [labels]
        return self.base_model.forward(
            input_ids,
            blen=blen,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prefill_kv=prefill_kv,
            decode_kv=decode_kv,
            labels=labels,
            *args,
            **kwargs
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def named_parameters(
        self,
        prefix: str = '',
        recurse: bool = True,
        remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        '''
        Mixed Lora Model for Trainer filters lora parameters by lora id.
        '''
        gen = filter(
            lambda np: 'lora' not in np[0] or int(np[0].split('-')[1]) == self.trainer_lora_id,
            self._named_members(
                lambda module: module._parameters.items(),
                prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
            )
        )
        yield from gen
    
    def get_nb_trainable_parameters(self):
        '''
        Copied from peft > peft_model.py > `PeftModel`.

        Returns the number of trainable parameters and number of all parameters in the model.
        '''
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self):
        '''
        Copied from peft > peft_model.py > `PeftModel`.

        Prints the number of trainable parameters in the model.
        '''
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )
    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        '''
        Mixed Lora Model only returns modules with `lora`, as a trick for saving lora linears only.
        '''
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default
            warn("Positional args are being deprecated, use kwargs instead. Refer to "
                 "https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
                 " for details.")

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        destination = OrderedDict(filter(lambda kv: 'lora' in kv[0], destination.items()))
        return destination

