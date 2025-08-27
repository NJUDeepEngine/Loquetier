from collections import OrderedDict
import torch
from torch.nn import Linear
from torch.nn.modules import Module, ModuleList
from torch.nn.modules.module import _addindent

from peft.auto import AutoModelForCausalLM
from peft.tuners.lora import Linear as LoraLinear

# For Type Check
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast

# For Multi-Processing
import threading
from queue import Queue

# For In-Context Device Change
from torch.nn import Parameter

# For Generation Rewrites
import copy
import inspect
import warnings
import torch.distributed as dist

# Enum Utils
from .enums import *

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import (
    GenerationConfig,
    GenerationMode,
    BeamSearchScorer,
    ConstrainedBeamSearchScorer,
    DisjunctiveConstraint,
    PhrasalConstraint,
)
import transformers.generation as transformer_generation
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
)
from transformers.utils import (
    logging,
)

logger = logging.get_logger(__name__)

# For Llama Forward Rewrite
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import (
    # _prepare_4d_causal_attention_mask,
    LlamaModel,
)

import time


class SimpleContextManager:
    def __init__(self, value):
        self.value = value

    def set(self, value):
        self.value = value
        return self.value
    
    def get(self):
        return self.value


def create_virtual_module(target_module, root_context, path = ''):
    pass

def create_virtual_linear(target_linear, root_context, path = '', **kwargs):
    base_class = type(target_linear)

    class VirtualLinear(base_class):
        def __init__(self, module, path, **kwargs):
            self.path = path
            if 'submodules' in kwargs:
                new_module_dict = kwargs['submodules']
            else:
                new_module_dict = OrderedDict()
                for submodule in module._modules:
                    if isinstance(module.get_submodule(submodule), Linear):
                        new_module_dict[submodule] = create_virtual_linear(
                            module.get_submodule(submodule), root_context, self.path + '.' + submodule
                        )
                    else:
                        new_module_dict[submodule] = create_virtual_module(
                            module.get_submodule(submodule), root_context, self.path + '.' + submodule
                        )
            
            self._module = module
            self._modules = new_module_dict
            for name, submodule in new_module_dict.items():
                setattr(self, name, submodule)
            
            self.real_to = self.to
            def to(self, *args, **kwargs):
                print('Virtual Model won\'t be moved!')
                return self
            self.to = to
            self.root_context = root_context
            self.is_virtual = True
        
        def __getattr__(self, name):
            if name in self.__dict__:
                return self.__dict__[name]
            else:
                return getattr(self._module, name)
        
        def __setattr__(self, name, value):
            if (name == "in_features" or name == "out_features" or 
                name == "bias" or name == "device" or name == "dtype" or
                name == "to" or name == "real_to"
                ):
                super(Linear, self).__setattr__(name, value)
            elif (name == "_module" or name == "_modules" or name == "path" or
                  name == "root_context" or name == "is_virtual"):
                self.__dict__[name] = value
            elif name in self._modules:
                super(Module, self).__setattr__(name, value)
                self._modules[name] = value
                # self.__dict__[name] = value
            else:
                setattr(self._module, name, value)
        
        def _get_name(self):
            return '[Virtual Linear] ' + self._module._get_name()

        def __repr__(self):
            # Copy from Module.__repr__
            # in torch/nn/modules/module.py
            extra_lines = []
            extra_repr = self._module.extra_repr()
            if extra_repr:
                extra_lines = extra_repr.split('\n')
            child_lines = []
            for key, module in self._modules.items():
                mod_str = repr(module)
                mod_str = _addindent(mod_str, 2)
                child_lines.append('(' + key + '): ' + mod_str)
            lines = extra_lines + child_lines

            main_str = self._get_name() + '('
            if lines:
                if len(extra_lines) == 1 and not child_lines:
                    main_str += extra_lines[0]
                else:
                    main_str += '\n  ' + '\n  '.join(lines) + '\n'
            main_str += ')'
            return main_str
        
        # Change move methods
        def _apply(self, fn, recurse=True):
            if recurse:
                for module in self.children():
                    module._apply(fn)
            
            if self.root_context.get():
                return self

            def compute_should_use_set_data(tensor, tensor_applied):
                if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                    return not torch.__future__.get_overwrite_module_params_on_conversion()
                else:
                    return False

            for key, param in self._parameters.items():
                if param is None:
                    continue
                with torch.no_grad():
                    param_applied = fn(param)
                should_use_set_data = compute_should_use_set_data(param, param_applied)
                if should_use_set_data:
                    param.data = param_applied
                    out_param = param
                else:
                    assert isinstance(param, Parameter)
                    assert param.is_leaf
                    out_param = Parameter(param_applied, param.requires_grad)
                    self._parameters[key] = out_param

                if param.grad is not None:
                    with torch.no_grad():
                        grad_applied = fn(param.grad)
                    should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
                    if should_use_set_data:
                        assert out_param.grad is not None
                        out_param.grad.data = grad_applied
                    else:
                        assert param.grad.is_leaf
                        out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)

            for key, buf in self._buffers.items():
                if buf is not None:
                    self._buffers[key] = fn(buf)

            return self
    
    return VirtualLinear(target_linear, path, **kwargs)

def create_virtual_module(target_module, root_context, path = '', **kwargs):
    base_class = type(target_module)

    class VirtualModule(base_class, Module):
        def __init__(self, module, path, **kwargs):
            self.path = path
            if 'submodules' in kwargs:
                new_module_dict = kwargs['submodules']
            else:
                new_module_dict = OrderedDict()
                for submodule in module._modules:
                    if isinstance(module.get_submodule(submodule), Linear):
                        new_module_dict[submodule] = create_virtual_linear(
                            module.get_submodule(submodule), root_context, self.path + '.' + submodule
                        )
                    else:
                        new_module_dict[submodule] = create_virtual_module(
                            module.get_submodule(submodule), root_context, self.path + '.' + submodule
                        )
            
            self._module = module
            self._modules = new_module_dict
            for name, submodule in new_module_dict.items():
                setattr(self, name, submodule)
            
            self.real_to = self.to
            def to(self, *args, **kwargs):
                print('Virtual Model won\'t be moved!')
                return self
            self.to = to
            self.root_context = root_context
            self.is_virtual = True
        
        # Change move methods
        def _apply(self, fn, recurse=True):
            if recurse:
                for module in self.children():
                    module._apply(fn)
            
            if self.root_context.get():
                return self

            def compute_should_use_set_data(tensor, tensor_applied):
                if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                    return not torch.__future__.get_overwrite_module_params_on_conversion()
                else:
                    return False

            for key, param in self._parameters.items():
                if param is None:
                    continue
                with torch.no_grad():
                    param_applied = fn(param)
                should_use_set_data = compute_should_use_set_data(param, param_applied)
                if should_use_set_data:
                    param.data = param_applied
                    out_param = param
                else:
                    assert isinstance(param, Parameter)
                    assert param.is_leaf
                    out_param = Parameter(param_applied, param.requires_grad)
                    self._parameters[key] = out_param

                if param.grad is not None:
                    with torch.no_grad():
                        grad_applied = fn(param.grad)
                    should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
                    if should_use_set_data:
                        assert out_param.grad is not None
                        out_param.grad.data = grad_applied
                    else:
                        assert param.grad.is_leaf
                        out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)

            for key, buf in self._buffers.items():
                if buf is not None:
                    self._buffers[key] = fn(buf)

            return self
        
        if isinstance(target_module, ModuleList):
            def __getattr__(self, name):
                if name in self.__dict__:
                    return self.__dict__[name]
                elif name in self._modules:
                    return self._modules[name]
                else:
                    return getattr(self._module, name)
            
            def __setattr__(self, name, value):
                if (name == "_module" or name == "_modules" or name == "path" or
                    name == "root_context" or name == "is_virtual"):
                    self.__dict__[name] = value
                elif name in self._modules:
                    self._modules[name] = value
                    # super(Module, self).__setattr__(name, value)
                    # self.__dict__[name] = value
                else:
                    setattr(self._module, name, value)
            
            def _get_name(self):
                return '[Virtual ModuleList] ' + self._module._get_name()
            
            def __repr__(self):
                # Copy from ModuleList.__repr__
                # in torch/nn/modules/module.py
                list_of_reprs = [repr(item) for item in self]
                if len(list_of_reprs) == 0:
                    return self._module._get_name() + '()'

                start_end_indices = [[0, 0]]
                repeated_blocks = [list_of_reprs[0]]
                for i, r in enumerate(list_of_reprs[1:], 1):
                    if r == repeated_blocks[-1]:
                        start_end_indices[-1][1] += 1
                        continue

                    start_end_indices.append([i, i])
                    repeated_blocks.append(r)

                lines = []
                main_str = self._get_name() + '('
                for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
                    local_repr = f"({start_id}): {b}"  # default repr

                    if start_id != end_id:
                        n = end_id - start_id + 1
                        local_repr = f"({start_id}-{end_id}): {n} x {b}"

                    local_repr = _addindent(local_repr, 2)
                    lines.append(local_repr)

                main_str += '\n  ' + '\n  '.join(lines) + '\n'
                main_str += ')'
                return main_str
        
        else:
            def __getattr__(self, name):
                if name in self.__dict__:
                    return self.__dict__[name]
                else:
                    return getattr(self._module, name)
            
            def __setattr__(self, name, value):
                if (name == "_module" or name == "_modules" or name == "path" or
                    name == "to" or name == "real_to" or
                    name == "root_context" or name == "is_virtual"):
                    self.__dict__[name] = value
                elif name in self._modules:
                    super(Module, self).__setattr__(name, value)
                    self._modules[name] = value
                    # self.__dict__[name] = value
                else:
                    setattr(self._module, name, value)
            
            def _get_name(self):
                return '[Virtual Module] ' + self._module._get_name()
            
            def __repr__(self):
                # Copy from Module.__repr__
                # in torch/nn/modules/module.py
                extra_lines = []
                extra_repr = self._module.extra_repr()
                if extra_repr:
                    extra_lines = extra_repr.split('\n')
                child_lines = []
                for key, module in self._modules.items():
                    mod_str = repr(module)
                    mod_str = _addindent(mod_str, 2)
                    child_lines.append('(' + key + '): ' + mod_str)
                lines = extra_lines + child_lines

                main_str = self._get_name() + '('
                if lines:
                    if len(extra_lines) == 1 and not child_lines:
                        main_str += extra_lines[0]
                    else:
                        main_str += '\n  ' + '\n  '.join(lines) + '\n'
                main_str += ')'
                return main_str

    return VirtualModule(target_module, path, **kwargs)

def create_virtual_model(pretrained_model, **kwargs):
    base_class = type(pretrained_model)

    class VirtualModel(base_class, Module):
        def __init__(self, pretrained_model, **kwargs):
            if 'root_context' in kwargs:
                self.root_context = kwargs['root_context']
            else:
                self.root_context = SimpleContextManager(False)
            self.path = ''

            base_model = pretrained_model
            if 'submodules' in kwargs:
                new_module_dict = kwargs['submodules']
            else:
                new_module_dict = OrderedDict()
                for submodule in base_model._modules:
                    if isinstance(base_model.get_submodule(submodule), Linear):
                        new_module_dict[submodule] = create_virtual_linear(
                            base_model.get_submodule(submodule), self.root_context, submodule
                        )
                    else:
                        new_module_dict[submodule] = create_virtual_module(
                            base_model.get_submodule(submodule), self.root_context, submodule
                        )

            self._model = base_model
            self._modules = new_module_dict
            for name, submodule in new_module_dict.items():
                setattr(self, name, submodule)
            
            self.real_to = self.to
            def to(self, *args, **kwargs):
                print('Virtual Model won\'t be moved!')
                return self
            self.to = to
            self.is_virtual = True
        
        def __getattr__(self, name):
            if name in self.__dict__:
                return self.__dict__[name]
            elif name == "_model":
                return super().__getattr__(name)
            else:
                return getattr(self._model, name)
        
        def __setattr__(self, name, value):
            if (name == "_model" or name == "_modules" or name == "to" or name == "real_to" or
                name == "path" or name == "root_context" or name == "is_virtual"):
                self.__dict__[name] = value
            elif name in self._modules:
                super(Module, self).__setattr__(name, value)
                self._modules[name] = value
                # self.__dict__[name] = value
            else:
                setattr(self._model, name, value)
        
        def _get_name(self):
            return '[Virtual Model] ' + self._model._get_name()
        
        def __repr__(self):
            # Copy from Module.__repr__
            # in torch/nn/modules/module.py
            extra_lines = []
            extra_repr = self._model.extra_repr()
            if extra_repr:
                extra_lines = extra_repr.split('\n')
            child_lines = []
            for key, module in self._modules.items():
                mod_str = repr(module)
                mod_str = _addindent(mod_str, 2)
                child_lines.append('(' + key + '): ' + mod_str)
            lines = extra_lines + child_lines

            main_str = self._get_name() + '('
            if lines:
                if len(extra_lines) == 1 and not child_lines:
                    main_str += extra_lines[0]
                else:
                    main_str += '\n  ' + '\n  '.join(lines) + '\n'
            main_str += ')'
            return main_str
        
        # Change move methods
        def _apply(self, fn, recurse=True):
            if recurse:
                for module in self.children():
                    module._apply(fn)
            
            if self.root_context.get():
                return self

            def compute_should_use_set_data(tensor, tensor_applied):
                if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                    return not torch.__future__.get_overwrite_module_params_on_conversion()
                else:
                    return False

            for key, param in self._parameters.items():
                if param is None:
                    continue
                with torch.no_grad():
                    param_applied = fn(param)
                should_use_set_data = compute_should_use_set_data(param, param_applied)
                if should_use_set_data:
                    param.data = param_applied
                    out_param = param
                else:
                    assert isinstance(param, Parameter)
                    assert param.is_leaf
                    out_param = Parameter(param_applied, param.requires_grad)
                    self._parameters[key] = out_param

                if param.grad is not None:
                    with torch.no_grad():
                        grad_applied = fn(param.grad)
                    should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
                    if should_use_set_data:
                        assert out_param.grad is not None
                        out_param.grad.data = grad_applied
                    else:
                        assert param.grad.is_leaf
                        out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)

            for key, buf in self._buffers.items():
                if buf is not None:
                    self._buffers[key] = fn(buf)

            return self
        
        def light_to(self, *args, **kwargs):
            device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

            if dtype is not None:
                if not (dtype.is_floating_point or dtype.is_complex):
                    raise TypeError('nn.Module.to only accepts floating point or complex '
                                    f'dtypes, but got desired dtype={dtype}')
                if dtype.is_complex:
                    warnings.warn(
                        "Complex modules are a new feature under active development whose design may change, "
                        "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                        "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
                        "if a complex module does not work as expected.")

            def convert(t):
                if convert_to_format is not None and t.dim() in (4, 5):
                    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                                non_blocking, memory_format=convert_to_format)
                return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

            class VirtualTo:
                def __init__(self, value):
                    self.value = value

                def __enter__(self):
                    self.value.set(True)
                
                def __exit__(self, exc_type, exc_value, traceback):
                    self.value.set(False)

            with VirtualTo(self.root_context):
                return self._apply(convert)
        
    return VirtualModel(pretrained_model, **kwargs)

def is_virtual(module):
    return 'is_virtual' in module.__dict__ and module.__dict__['is_virtual']

import importlib
import os
from typing import Optional

from transformers import AutoTokenizer
from peft.config import PeftConfig
from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from peft.peft_model import PeftModelForCausalLM
from peft.utils.constants import TOKENIZER_CONFIG_NAME
from peft.utils.other import check_file_exists_on_hf_hub

class VirtualLinearT(Linear):
    def __init__(self, module, path = '', **kwargs):
        self.path = path
        if 'submodules' in kwargs:
            new_module_dict = kwargs['submodules']
        else:
            new_module_dict = OrderedDict()
            for submodule in module._modules:
                target_module = module.get_submodule(submodule)
                target_path = target_module.path if hasattr(
                    target_module, 'path'
                ) else self.path + '.' + submodule
                if isinstance(target_module, Linear):
                    new_module_dict[submodule] = VirtualLinearT(
                        target_module, target_path
                    )
                elif isinstance(target_module, ModuleList):
                    new_module_dict[submodule] = VirtualModuleListT(
                        target_module, target_path
                    )
                else:
                    new_module_dict[submodule] = VirtualModuleT(
                        target_module, target_path
                    )
        
        self._module = module
        self._modules = new_module_dict
        for name, submodule in new_module_dict.items():
            setattr(self, name, submodule)
        self.is_virtual = True
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self._module, name)
    
    def __setattr__(self, name, value):
        if (name == "in_features" or name == "out_features" or 
            name == "bias" or name == "device" or name == "dtype" or
            name == "to" or name == "real_to"
            ):
            super(Linear, self).__setattr__(name, value)
        elif (name == "_module" or name == "_modules" or name == "path" or
              name == "root_context" or name == "is_virtual"):
            self.__dict__[name] = value
        elif name in self._modules:
            super(Module, self).__setattr__(name, value)
            self._modules[name] = value
            # self.__dict__[name] = value
        else:
            setattr(self._module, name, value)
    
    def _get_name(self):
        return '[VirtualLinearT] ' + self._module._get_name()

class VirtualModuleT(Module):
    def __init__(self, module, path = '', **kwargs):
        self.path = path
        if 'submodules' in kwargs:
            new_module_dict = kwargs['submodules']
        else:
            new_module_dict = OrderedDict()
            for submodule in module._modules:
                target_module = module.get_submodule(submodule)
                target_path = target_module.path if hasattr(
                    target_module, 'path'
                ) else self.path + '.' + submodule
                if isinstance(target_module, Linear):
                    new_module_dict[submodule] = VirtualLinearT(
                        target_module, target_path
                    )
                elif isinstance(target_module, ModuleList):
                    new_module_dict[submodule] = VirtualModuleListT(
                        target_module, target_path
                    )
                else:
                    new_module_dict[submodule] = VirtualModuleT(
                        target_module, target_path
                    )
        
        self._module = module
        self._modules = new_module_dict
        for name, submodule in new_module_dict.items():
            setattr(self, name, submodule)
        self.is_virtual = True
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self._module, name)
    
    def __setattr__(self, name, value):
        if (name == "_module" or name == "_modules" or name == "path" or
            name == "to" or name == "real_to" or
            name == "root_context" or name == "is_virtual"):
            self.__dict__[name] = value
        elif name in self._modules:
            super(Module, self).__setattr__(name, value)
            self._modules[name] = value
            # self.__dict__[name] = value
        else:
            setattr(self._module, name, value)
        
    def _get_name(self):
        return '[VirtualModuleT] ' + self._module._get_name()

class VirtualModuleListT(ModuleList):
    def __init__(self, module, path = '', **kwargs):
        self.path = path
        if 'submodules' in kwargs:
            new_module_dict = kwargs['submodules']
        else:
            new_module_dict = OrderedDict()
            for submodule in module._modules:
                target_module = module.get_submodule(submodule)
                target_path = target_module.path if hasattr(
                    target_module, 'path'
                ) else self.path + '.' + submodule
                if isinstance(target_module, Linear):
                    new_module_dict[submodule] = VirtualLinearT(
                        target_module, target_path
                    )
                elif isinstance(target_module, ModuleList):
                    new_module_dict[submodule] = VirtualModuleListT(
                        target_module, target_path
                    )
                else:
                    new_module_dict[submodule] = VirtualModuleT(
                        target_module, target_path
                    )
        
        self._module = module
        self._modules = new_module_dict
        for name, submodule in new_module_dict.items():
            setattr(self, name, submodule)
        self.is_virtual = True
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self._modules:
            return self._modules[name]
        else:
            return getattr(self._module, name)
    
    def __setattr__(self, name, value):
        if (name == "_module" or name == "_modules" or name == "path" or
            name == "root_context" or name == "is_virtual"):
            self.__dict__[name] = value
        elif name in self._modules:
            self._modules[name] = value
            # super(Module, self).__setattr__(name, value)
            # self.__dict__[name] = value
        else:
            setattr(self._module, name, value)
        
    def _get_name(self):
        return '[VirtualModuleListT] ' + self._module._get_name()

class VirtualModelT(Module):
    def __init__(self, pretrained_model, **kwargs):
        self.path = ''
        base_model = pretrained_model
        if 'submodules' in kwargs:
            new_module_dict = kwargs['submodules']
        else:
            new_module_dict = OrderedDict()
            for submodule in base_model._modules:
                target_module = base_model.get_submodule(submodule)
                target_path = target_module.path if hasattr(
                    target_module, 'path'
                ) else submodule
                if isinstance(target_module, Linear):
                    new_module_dict[submodule] = VirtualLinearT(
                        target_module, target_path
                    )
                elif isinstance(target_module, ModuleList):
                    new_module_dict[submodule] = VirtualModuleListT(
                        target_module, target_path
                    )
                else:
                    new_module_dict[submodule] = VirtualModuleT(
                        target_module, target_path
                    )

        self._model = base_model
        self._modules = new_module_dict
        for name, submodule in new_module_dict.items():
            setattr(self, name, submodule)
        self.is_virtual = True
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self._model, name)
    
    def __setattr__(self, name, value):
        if (name == "_model" or name == "_modules" or name == "to" or name == "real_to" or
            name == "path" or name == "root_context" or name == "is_virtual"):
            self.__dict__[name] = value
        elif name in self._modules:
            super(Module, self).__setattr__(name, value)
            self._modules[name] = value
            # self.__dict__[name] = value
        else:
            setattr(self._model, name, value)
    
    def _get_name(self):
        return '[VirtualModelT] ' + self._model._get_name()


def virtual_module_instantiate(module, real_module, root_context):
    new_module_dict = OrderedDict()
    for submodule in module._modules:
        target = module._modules[submodule]
        new_module_dict[submodule] = virtual_module_instantiate(
            target,
            target._module if is_virtual(target) else target,
            root_context
        )
    
    if is_virtual(module):
        if isinstance(real_module, Linear):
            return create_virtual_linear(
                real_module,
                root_context,
                module.path,
                submodules=new_module_dict
            )
        else:
            return create_virtual_module(
                real_module,
                root_context,
                module.path,
                submodules=new_module_dict
            )
    else:
        for name, submodule in new_module_dict.items():
            setattr(module, name, submodule)
        return module

def virtual_model_instantiate(model, real_model):
    root_context = SimpleContextManager(False)
    new_module_dict = OrderedDict()
    for submodule in model._modules:
        target = model._modules[submodule]
        new_module_dict[submodule] = virtual_module_instantiate(
            target,
            target._module if is_virtual(target) else target,
            root_context
        )
    
    return create_virtual_model(
        real_model,
        root_context=root_context,
        submodules=new_module_dict
    )


def virtual_module_uninstantiate(module, real_module):
    new_module_dict = OrderedDict()
    for submodule in module._modules:
        target = module._modules[submodule]
        new_module_dict[submodule] = virtual_module_uninstantiate(
            target,
            target._module if is_virtual(target) else target
        )
    
    if is_virtual(module):
        if isinstance(real_module, Linear):
            return VirtualLinearT(
                real_module,
                module.path,
                submodules=new_module_dict
            )
        elif isinstance(real_module, ModuleList):
            return VirtualModuleListT(
                real_module,
                module.path,
                submodules=new_module_dict
            )
        else:
            return VirtualModuleT(
                real_module,
                module.path,
                submodules=new_module_dict
            )
    else:
        for name, submodule in new_module_dict.items():
            setattr(module, name, submodule)
        return module

def virtual_model_uninstantiate(model, real_model):
    new_module_dict = OrderedDict()
    for submodule in model._modules:
        target = model._modules[submodule]
        new_module_dict[submodule] = virtual_module_uninstantiate(
            target,
            target._module if is_virtual(target) else target
        )
    
    return VirtualModelT(
        real_model,
        submodules=new_module_dict
    )

class VoidModule(Module):
    def __init__(self, path, submodule):
        self.path = path
        self._modules = submodule
        self._parameters = {}
        self._buffers = {}

def virtual_module_void(lora_module: Module):
    new_submodules = {
        lora_submodule: virtual_module_void(
            lora_module._modules[lora_submodule]
        ) for lora_submodule in lora_module._modules
    }
    if is_virtual(lora_module):
        return VoidModule(lora_module.path, new_submodules)
    else:
        lora_module._modules = new_submodules
        return lora_module

def virtual_module_unvoid(lora_module: Module, virtual_model: Module):
    new_submodules = {
        lora_submodule: virtual_module_unvoid(
            lora_module._modules[lora_submodule], virtual_model
        ) for lora_submodule in lora_module._modules
    }
    if isinstance(lora_module, VoidModule):
        target_module = virtual_model.get_submodule(lora_module.path)
        for name, submodule in new_submodules.items():
            target_module.add_module(name, submodule)
        return target_module
    else:
        for name, submodule in new_submodules.items():
            lora_module.add_module(name, submodule)
        return lora_module
                

class LoquetierFramework:
    BaseModelList = {}

    def __init__(self):
        pass

    @classmethod
    def load_base_model(cls, base_model_path=None, base_model_list=None, **kwargs):
        if base_model_list is None:
            base_model_list = cls.BaseModelList
        if base_model_path:
            if base_model_path not in base_model_list:
                base_model_list[base_model_path] = AutoModelForCausalLM.from_pretrained(
                    base_model_path, **kwargs
                )
        else:
            base_model_path = kwargs['pretrained_model_name_or_path']
            if base_model_path not in base_model_list:
                base_model_list[base_model_path] = AutoModelForCausalLM.from_pretrained(
                    **kwargs
                )
        return base_model_list[base_model_path]

    @classmethod
    def load_model(
        cls,
        pretrained_model_name_or_path,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        base_model_list: Optional[dict] = None,
        use_static_model: Optional[bool] = False,
        **kwargs,
    ):
        peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        base_model_path = peft_config.base_model_name_or_path

        task_type = getattr(peft_config, "task_type", None)

        target_class = AutoModelForCausalLM

        if task_type is not None:
            expected_target_class = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type]
            if PeftModelForCausalLM.__name__ != expected_target_class.__name__:
                raise ValueError(
                    f"Expected target PEFT class: {expected_target_class.__name__}, but you have asked for: {PeftModelForCausalLM.__name__ }"
                    " make sure that you are loading the correct model for your task type."
                )
        elif task_type is None and getattr(peft_config, "auto_mapping", None) is not None:
            auto_mapping = getattr(peft_config, "auto_mapping", None)
            base_model_class = auto_mapping["base_model_class"]
            parent_library_name = auto_mapping["parent_library"]

            parent_library = importlib.import_module(parent_library_name)
            target_class = getattr(parent_library, base_model_class)
        else:
            raise ValueError(
                "Cannot infer the auto class from the config, please make sure that you are loading the correct model for your task type."
            )

        pretrained_model = cls.load_base_model(base_model_path, base_model_list, **kwargs)
        base_model = VirtualModelT(
            pretrained_model
        ) if use_static_model else create_virtual_model(
            pretrained_model
        )

        tokenizer_exists = False
        if os.path.exists(os.path.join(pretrained_model_name_or_path, TOKENIZER_CONFIG_NAME)):
            tokenizer_exists = True
        else:
            token = kwargs.get("token", None)
            if token is None:
                token = kwargs.get("use_auth_token", None)

            tokenizer_exists = check_file_exists_on_hf_hub(
                repo_id=pretrained_model_name_or_path,
                filename=TOKENIZER_CONFIG_NAME,
                revision=kwargs.get("revision", None),
                repo_type=kwargs.get("repo_type", None),
                token=token,
            )

        if tokenizer_exists:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                use_fast=kwargs.get("use_fast", True),
                trust_remote_code=kwargs.get("trust_remote_code", False)
            )
            base_model.resize_token_embeddings(len(tokenizer))

        return PeftModelForCausalLM.from_pretrained(
            base_model,
            pretrained_model_name_or_path,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            config=config,
            **kwargs,
        )
