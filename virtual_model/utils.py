from dataclasses import dataclass
from multiprocessing import Queue
from time import time
import torch
from typing import Any, Optional

from .enums import *


def peft_decouple(model):
    model.base_model_prepare_inputs_for_generation = None

def peft_recover(model):
    model.base_model_prepare_inputs_for_generation = model.base_model.prepare_inputs_for_generation


def final_output_callback(output_queue, output):
    output_queue.put(output)

from torch import tensor

class StreamRecord:
    def __init__(self):
        self.streams = {}
        self.mark_ids = None

    def stream_output_callback(self, output, output_queue):
        if output[0] == InferOutputType.TokensOutput:
            for mid, o in zip(self.mark_ids, output[1]):
                if mid in self.streams:
                    self.streams[mid].append(o.item())
                else:
                    self.streams[mid] = [o.item()]
            output_queue.put((True, (self.mark_ids, output[1].tolist())))
        elif output[0] == InferOutputType.FinishedSignal:
            ret_dict = {}
            for mid_end in output[1].tolist():
                ret_dict[mid_end] = None
                del self.streams[mid_end]
            output_queue.put((False, ret_dict))
        else:
            self.mark_ids = output[1].tolist()

class GPUMemoryUsageRecorder:
    def __init__(self, device):
        self.device = device
        self.token_memory_map = {}
        self.current_tokens = 1
        self.memory_basis = 0
        self.record_end = True
    
    def record_prepare(self):
        self.token_memory_map = {}
        self.current_tokens = 1
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats(device=self.device)
        torch.cuda.reset_accumulated_memory_stats(device=self.device)
        self.record_end = False

    def start_record(self):
        pass

    def set_memory_basis(self):
        self.memory_basis = torch.cuda.memory_allocated(device=self.device)
    
    def get_memory_usage(self, output, current_tokens = None, output_queue = None):
        if self.record_end:
            if output[0] == InferOutputType.MarkIdUpdateAdded:
                self.record_prepare()
                self.start_record()
            else:
                return
        elif output[0] == InferOutputType.MarkIdUpdateAdded:
            self.start_record()
        elif output[0] == InferOutputType.FinishedSignal:
            print(f'Finished Signal Triggered, Output Queue {bool(output_queue)}')
            self.record_end = True
            if output_queue:
                output_queue.put(self.token_memory_map)
        elif output[0] == InferOutputType.TokensOutput:
            if current_tokens is None:
                current_tokens = self.current_tokens
                self.current_tokens += 1
            else:
                self.current_tokens = current_tokens + 1
            self.token_memory_map[current_tokens] = torch.cuda.max_memory_allocated(self.device)
            ''' For the 1st time running in an infer process,
                use max_memory_allocated is accurate.
                For running multiple times in the same infer process,
                use memory_allocated instead.
            '''

class SchedulerStreamRecord(StreamRecord):
    def __init__(self, manager_id = -1, config_id = '', check_length = 10, ref_id = None):
        super().__init__()
        self.token_length = 0
        self.manager_id = manager_id
        self.config_id = config_id
        self.check_length = check_length
        self.ref_id = ref_id if ref_id is not None else config_id
    
    def scheduler_output_callback(self, output, output_queue: Queue, inter_output_queue: Queue):
        self.stream_output_callback(output, output_queue)
        self.token_length += 1
        if output[0] == InferOutputType.FinishedSignal:
            inter_output_queue.put((
                InteractiveSyncType.FinishSync,
                time(),
                self.manager_id,
                self.config_id,
                self.ref_id,
                output[1].tolist()
            ))
        elif output[0] == InferOutputType.MarkIdUpdateAdded:
            inter_output_queue.put((
                InteractiveSyncType.StartSync,
                time(),
                self.manager_id,
                self.config_id,
                self.ref_id,
                output[1].tolist()
            ))
        elif output[0] == InferOutputType.TokensOutput and self.token_length >= self.check_length:
            self.token_length = 0
            inter_output_queue.put((
                InteractiveSyncType.OutputSync,
                time(),
                self.manager_id,
                self.config_id,
                self.ref_id,
                [(mark_id, len(self.streams[mark_id])) for mark_id in self.mark_ids]
            ))


from torch import cat, full, repeat_interleave


def pre_op_method_stay_alive(model, **kwargs):
    model.stay_alive = True

def pre_op_method_output_mode(model, **kwargs):
    model.output_mode = True

def pre_op_method_max_calc_cost(model, **kwargs):
    max_calc_cost = kwargs.get(
        'max_calc_cost', kwargs.get(
            'default_max_calc_cost', 1
    ))
    model.max_calc_cost = max_calc_cost

from functools import partial

def pre_op_method_default_callback(model, **kwargs):
    output_queue = kwargs.get('output_queue')
    model.callback = partial(final_output_callback, output_queue)

def pre_op_method_stream_callback(model, **kwargs):
    output_queue = kwargs.get('output_queue')
    recorder = kwargs.get('recorder', StreamRecord())
    model.callback = partial(recorder.stream_output_callback, output_queue=output_queue)

def pre_op_method_scheduler_callback(model, **kwargs):
    output_queue = kwargs.get('output_queue')
    inter_output_queue = kwargs.get('inter_output_queue')
    manager_id = kwargs.get('manager_id', -1)
    config_id = kwargs.get('config_id', '')
    check_length = kwargs.get('check_length', 10)
    ref_id = kwargs.get('ref_id', config_id)
    recorder = kwargs.get('recorder', SchedulerStreamRecord(
        manager_id=manager_id, config_id=config_id, check_length=check_length, ref_id=ref_id
    ))
    model.callback = partial(
        recorder.scheduler_output_callback,
        output_queue=output_queue,
        inter_output_queue=inter_output_queue)

def pre_op_method_gpu_usage_callback(model, **kwargs):
    output_queue = kwargs.get('output_queue')
    recorder = kwargs.get('recorder', GPUMemoryUsageRecorder(model.device))
    model.callback = partial(recorder.get_memory_usage, output_queue=output_queue)

def bloom_default_pre_op_method(model, **kwargs):
    pre_op_method_max_calc_cost(model, **kwargs)
    pre_op_method_default_callback(model, **kwargs)

def bloom_decoder_pre_op_method(model, **kwargs):
    pre_op_method_stay_alive(model, **kwargs)
    pre_op_method_output_mode(model, **kwargs)
    pre_op_method_max_calc_cost(model, **kwargs)
    pre_op_method_stream_callback(model, **kwargs)

def bloom_prefiller_pre_op_method(model, **kwargs):
    pre_op_method_max_calc_cost(model, **kwargs)
    pre_op_method_default_callback(model, **kwargs)

def bloom_decoder_scheduler_pre_op_method(model, **kwargs):
    pre_op_method_stay_alive(model, **kwargs)
    pre_op_method_output_mode(model, **kwargs)
    pre_op_method_max_calc_cost(model, **kwargs)
    pre_op_method_scheduler_callback(model, **kwargs)

def bloom_measurement_pre_op_method(model, **kwargs):
    pre_op_method_stay_alive(model, **kwargs)
    pre_op_method_output_mode(model, **kwargs)
    pre_op_method_gpu_usage_callback(model, **kwargs)

def llama_decoder_scheduler_pre_op_method(model, **kwargs):
    pre_op_method_stay_alive(model, **kwargs)
    pre_op_method_output_mode(model, **kwargs)
    pre_op_method_max_calc_cost(model, **kwargs)
    pre_op_method_scheduler_callback(model, **kwargs)

def llama_measurement_pre_op_method(model, **kwargs):
    pre_op_method_stay_alive(model, **kwargs)
    pre_op_method_output_mode(model, **kwargs)
    pre_op_method_gpu_usage_callback(model, **kwargs)


PRE_OP_METHODS = {}

PRE_OP_METHODS['llama_decoder_scheduler_pre_op_method'] = llama_decoder_scheduler_pre_op_method
PRE_OP_METHODS['llama_measurement_pre_op_method'] = llama_measurement_pre_op_method

