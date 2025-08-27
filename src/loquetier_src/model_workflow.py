from dataclasses import dataclass
from os import system
from time import time, sleep
import torch
from transformers import Trainer
from transformers.tokenization_utils import PreTrainedTokenizer

# Typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .enums import InferStatus, TrainerProcessType
from .model_generation import InputRequest, ModelGeneration
from .trainer_utils import WrappedTrainingLoopGenerator


FINETUNE_ERROR_MESSAGE = (
    'Function `{0}` doesn\'t support different training stage. '
    'To train multiple loras with different step setups, you have to '
    'implement loop logic on your own.\n'
    'To put it simply, you only need to run each step that has at least '
    'one training generator once to process generator logic, and always '
    'run backward stage at first.\n'
    'Notice: you can run training forward and evaluate forward together.'
)


@dataclass
class RequestWithTime:
    request: InputRequest
    start_time: float

@dataclass
class GenerateInfo:
    prefill_tokens: int
    decode_tokens: int
    time_cost: float
    time_idle: float


def grouped_output(
    full_dict: Dict[int, str],
    ended_total: int,
    tokenizer: PreTrainedTokenizer,
    output_dict: Dict[int, List[int]],
    ended_output: Set
):
    system('clear')
    print(f'FINISHED {ended_total:03} | {len(full_dict):03} TOTAL')
    output_dict = dict(sorted([
        (k, v) for k, v in output_dict.items()
    ] + [
        (eo, []) for eo in ended_output
    ], key=lambda x: x[0]))
    for k, v in output_dict.items():
        full_dict[k] += tokenizer.decode(
            v, skip_special_tokens=True
        )
        print('-' * 20)
        print(f'{k:02} | {full_dict[k]}')
    return

def run_inference(
    model: ModelGeneration,
    tokenizer: PreTrainedTokenizer,
    input_list: List[RequestWithTime],
    max_batch_size: int,
    max_serve_wait: Optional[float] = None,
    collect_output: bool = False
) -> Tuple[List[GenerateInfo], float, List[float]]:
    generation_info_lst = []
    slo_info = []
    prefill_tokens = 0
    time_idle = 0
    if collect_output:
        full_dict = {
            next_input.request.mark_id: tokenizer.decode(
                next_input.request.next_input_ids,
                skip_special_tokens=True
            ) for next_input in input_list
        }
        ended_total = 0
        ended_sets = {}
    
    start_time = time()
    for next_input in input_list:
        request, next_time = next_input.request, next_input.start_time

        current_time = time() - start_time
        if current_time < next_time and model.requests_len == 0:
            sleep(next_time - current_time)
            time_idle += next_time - current_time
        
        while (model.requests_len >= max_batch_size or
               (current_time < next_time and model.requests_len > 0)):
            model_start = time()
            model.generate()
            model_end = time()
            generation_info_lst.append(GenerateInfo(
                prefill_tokens, len(model.output_with_mark_ids),
                model_end - model_start, time_idle
            ))
            prefill_tokens = 0
            time_idle = 0
            for mark_id in model.finished_mark_ids:
                input_list[mark_id - 1].request.input_ids = input_list[
                    mark_id - 1].request.input_ids.to('cpu')
            if collect_output:
                for mark_id in list(ended_sets.keys()):
                    if ended_sets[mark_id] == 0:
                        del ended_sets[mark_id]
                    else:
                        ended_sets[mark_id] -= 1
                ended_sets.update({
                    mark_id: 5 for mark_id in model.finished_mark_ids
                })
                ended_total += len(model.finished_mark_ids)
                grouped_output(
                    full_dict,
                    ended_total,
                    tokenizer,
                    model.output_with_mark_ids,
                    ended_sets
                )
            current_time = time() - start_time
        
        current_delta = time() - start_time - next_time
        if max_serve_wait is not None and current_delta > max_serve_wait:
            slo_info.append(-1)
            continue
        if current_delta < 0:
            sleep(-current_delta)
            time_idle += -current_delta
            slo_info.append(0)
        else:
            slo_info.append(current_delta)
        prefill_tokens += len(request.next_input_ids)
        model.add_requests([request])
    
    while model.requests_len > 0:
        model_start = time()
        model.generate()
        generation_info_lst.append(GenerateInfo(
            prefill_tokens, len(model.output_with_mark_ids),
            model_end - model_start, time_idle
        ))
        prefill_tokens = 0
        time_idle = 0
        for mark_id in model.finished_mark_ids:
            input_list[mark_id - 1].request.input_ids = input_list[
                mark_id - 1].request.input_ids.to('cpu')
        if collect_output:
            for mark_id in list(ended_sets.keys()):
                if ended_sets[mark_id] == 0:
                    del ended_sets[mark_id]
                else:
                    ended_sets[mark_id] -= 1
            ended_sets.update({
                mark_id: 5 for mark_id in model.finished_mark_ids
            })
            ended_total += len(model.finished_mark_ids)
            grouped_output(
                full_dict,
                ended_total,
                tokenizer,
                model.output_with_mark_ids,
                ended_sets
            )
    
    end_time = time()
    print()
    return generation_info_lst, end_time - start_time, slo_info


@dataclass
class TrainerCollect:
    lora_name: str
    generator: WrappedTrainingLoopGenerator
    training_accum: int = 1
    evaluate_accum: int = 1
    status: Optional[TrainerProcessType] = None
    inputs: Optional[Union[
        Dict[str, Union[torch.Tensor, Any]],
        torch.Tensor
    ]] = None

@dataclass
class FinetuneInfo:
    process_type: TrainerProcessType
    process_tokens: int
    time_cost: float
    time_idle: float


def is_any_training_stopped(trainer_collects: List[TrainerCollect]) -> bool:
    for trainer_collect in trainer_collects:
        if trainer_collect.generator.is_training_stopped:
            return True
    return False

def run_finetune(
    model: ModelGeneration,
    trainers: Dict[str, Trainer],
) -> Tuple[List[FinetuneInfo], float]:
    finetune_info_lst = []
    training_collects = [
        TrainerCollect(
            lora_name,
            WrappedTrainingLoopGenerator(trainer),
            (trainer.args.gradient_accumulation_steps
             if trainer.args.gradient_accumulation_steps is not None else 1),
            (trainer.args.eval_accumulation_steps
             if trainer.args.eval_accumulation_steps is not None else 1),
        )
        for lora_name, trainer in trainers.items()
    ]

    start_time = time()
    while not is_any_training_stopped(training_collects):
        current_stage = None
        for training_collect in training_collects:
            (training_collect.status,
             training_collect.inputs) = training_collect.generator.next_value
            if current_stage is None:
                current_stage = training_collect.status
            elif current_stage != training_collect.status:
                raise RuntimeError(FINETUNE_ERROR_MESSAGE.format('run_finetune'))
        if current_stage == TrainerProcessType.TrainForward:
            process_tokens = 0
            for training_collect in training_collects:
                inputs = training_collect.inputs
                bsz, seqlen = inputs['input_ids'].shape
                model.add_requests([InputRequest(
                    InferStatus.Train,
                    inputs['input_ids'].view(-1),
                    inputs['input_ids'].view(-1).tolist(),
                    0,
                    training_collect.lora_name,
                    attention_mask=inputs['attention_mask'],
                    batch_info=(bsz, seqlen),
                    labels=inputs['labels'],
                    accumulation_steps=training_collect.training_accum
                )])
                process_tokens += bsz * seqlen
            
            model_start = time()
            loss_lst, logits_lst = model.generate()
            model_end = time()

            trainer_start = time()
            for training_collect, loss, logits in zip(
                training_collects, loss_lst, logits_lst
            ):
                training_collect.generator.get_next_value((
                    loss, logits
                ))
            trainer_end = time()

            finetune_info_lst.append(FinetuneInfo(
                current_stage,
                process_tokens,
                model_end - model_start,
                trainer_end - trainer_start
            ))
        
        elif current_stage == TrainerProcessType.EvaluateForward:
            process_tokens = 0
            requests = []
            seq_lens = []
            for training_collect in training_collects:
                inputs = training_collect.inputs
                bsz = inputs['input_ids'].shape[0]
                for i in range(bsz):
                    input_ids = inputs['input_ids'][i, ...][
                        inputs['attention_mask'][i, ...] != 0
                    ].contiguous()
                    requests.append(InputRequest(
                        InferStatus.Prefill,
                        input_ids,
                        input_ids.tolist(),
                        0,
                        training_collect.lora_name,
                        batch_info=(1, input_ids.shape[0]),
                        labels=inputs['labels'][i, ...][
                            inputs['attention_mask'][i, ...] != 0
                        ],
                        accumulation_steps=training_collect.evaluate_accum
                    ))
                    seq_lens.append(input_ids.shape[0])
                    process_tokens += input_ids.shape[0]
            model.add_requests(requests)
            del requests

            model_start = time()
            loss_lst, logits_lst = model.generate()
            model_end = time()

            trainer_sum = 0
            for training_collect in training_collects:
                inputs = training_collect.inputs
                bsz = inputs['input_ids'].shape[0]
                current_loss_lst, loss_lst = loss_lst[:bsz], loss_lst[bsz:]
                current_seq_lens, seq_lens = seq_lens[:bsz], seq_lens[bsz:]
                current_logits_lst, logits_lst = logits_lst[:bsz], logits_lst[bsz:]
                loss = sum(loss * seq for loss, seq in zip(
                    current_loss_lst, current_seq_lens
                )) / sum(seq for seq in current_seq_lens)
                logits = torch.cat(current_logits_lst, dim=1)

                trainer_start = time()
                training_collect.generator.get_next_value((loss, logits))
                trainer_end = time()
                trainer_sum += trainer_end - trainer_start
            
            finetune_info_lst.append(FinetuneInfo(
                current_stage,
                process_tokens,
                model_end - model_start,
                trainer_sum
            ))
        
        else: # current_stage == TrainerProcessType.Backward
            loss = sum(training_collect.inputs for training_collect in training_collects)

            model_start = time()
            loss.backward()
            model_end = time()

            trainer_start = time()
            for training_collect in training_collects:
                training_collect.generator.get_next_value()
            trainer_end = time()

            finetune_info_lst.append(FinetuneInfo(
                current_stage,
                0,
                model_end - model_start,
                trainer_end - trainer_start
            ))
    
    end_time = time()
    print()
    return finetune_info_lst, end_time - start_time


def run_finetune_inference(
    model: ModelGeneration,
    tokenizer: PreTrainedTokenizer,
    trainers: Dict[str, Trainer],
    input_list: List[RequestWithTime],
    max_batch_size: int,
    max_serve_wait: Optional[float] = None,
    infer_steps_per_finetune_step: int = 5,
    collect_output: bool = False
) -> Tuple[List[GenerateInfo], List[FinetuneInfo], float, List[float]]:
    generation_info_lst = []
    slo_info = []
    prefill_tokens = 0
    if collect_output:
        full_dict = {
            next_input.request.mark_id: tokenizer.decode(
                next_input.request.next_input_ids,
                skip_special_tokens=True
            ) for next_input in input_list
        }
        ended_total = 0
        ended_sets = {}
    finetune_info_lst = []
    training_collects = [
        TrainerCollect(
            lora_name,
            WrappedTrainingLoopGenerator(trainer),
            (trainer.args.gradient_accumulation_steps
             if trainer.args.gradient_accumulation_steps is not None else 1),
            (trainer.args.eval_accumulation_steps
             if trainer.args.eval_accumulation_steps is not None else 1),
        )
        for lora_name, trainer in trainers.items()
    ]

    input_ind, input_limit = 0, len(input_list)
    current_infer_steps = 0
    request, next_time = (input_list[input_ind].request,
                          input_list[input_ind].start_time)
    current_stage = None
    for training_collect in training_collects:
        (training_collect.status,
         training_collect.inputs) = training_collect.generator.next_value
        if current_stage is None:
            current_stage = training_collect.status
        elif current_stage != training_collect.status:
            raise RuntimeError(FINETUNE_ERROR_MESSAGE.format('run_finetune_inference'))
    
    start_time = time()
    while True:
        if input_ind < input_limit:
            current_time = time() - start_time

            while (max_serve_wait is not None and
                   current_time > next_time + max_serve_wait):
                slo_info.append(-1)
                input_ind += 1
                if input_ind >= input_limit:
                    break
                request, next_time = (input_list[input_ind].request,
                                    input_list[input_ind].start_time)
            
            while (current_time >= next_time and
                model.requests_len < max_batch_size):
                slo_info.append(current_time - next_time)
                prefill_tokens += len(request.next_input_ids)
                model.add_requests([request])
                input_ind += 1
                if input_ind >= input_limit:
                    break
                request, next_time = (input_list[input_ind].request,
                                    input_list[input_ind].start_time)
        
        if current_stage == TrainerProcessType.EvaluateForward:
            current_infer_steps = 0
            process_tokens = 0
            requests = []
            seq_lens = []
            for training_collect in training_collects:
                inputs = training_collect.inputs
                bsz = inputs['input_ids'].shape[0]
                for i in range(bsz):
                    input_ids = inputs['input_ids'][i, ...][
                        inputs['attention_mask'][i, ...] != 0
                    ].contiguous()
                    requests.append(InputRequest(
                        InferStatus.Prefill,
                        input_ids,
                        input_ids.tolist(),
                        0,
                        training_collect.lora_name,
                        batch_info=(1, input_ids.shape[0]),
                        labels=inputs['labels'][i, ...][
                            inputs['attention_mask'][i, ...] != 0
                        ],
                        accumulation_steps=training_collect.evaluate_accum
                    ))
                    seq_lens.append(input_ids.shape[0])
                    process_tokens += input_ids.shape[0]
            model.add_requests(requests)
            del requests

            model_start = time()
            loss_lst, logits_lst = model.generate()
            model_end = time()

            trainer_sum = 0
            for training_collect in training_collects:
                inputs = training_collect.inputs
                bsz = inputs['input_ids'].shape[0]
                current_loss_lst, loss_lst = loss_lst[:bsz], loss_lst[bsz:]
                current_seq_lens, seq_lens = seq_lens[:bsz], seq_lens[bsz:]
                current_logits_lst, logits_lst = logits_lst[:bsz], logits_lst[bsz:]
                loss = sum(loss * seq for loss, seq in zip(
                    current_loss_lst, current_seq_lens
                )) / sum(seq for seq in current_seq_lens)
                logits = torch.cat(current_logits_lst, dim=1)

                trainer_start = time()
                training_collect.generator.get_next_value((loss, logits))
                trainer_end = time()
                trainer_sum += trainer_end - trainer_start
            
            finetune_info_lst.append(FinetuneInfo(
                current_stage,
                process_tokens,
                model_end - model_start,
                trainer_sum
            ))

            generation_info_lst.append(GenerateInfo(
                prefill_tokens, len(model.output_with_mark_ids),
                model_end - model_start, 0
            ))
            prefill_tokens = 0
            for mark_id in model.finished_mark_ids:
                input_list[mark_id - 1].request.input_ids = input_list[
                    mark_id - 1].request.input_ids.to('cpu')
            if collect_output:
                for mark_id in list(ended_sets.keys()):
                    if ended_sets[mark_id] == 0:
                        del ended_sets[mark_id]
                    else:
                        ended_sets[mark_id] -= 1
                ended_sets.update({
                    mark_id: 5 for mark_id in model.finished_mark_ids
                })
                ended_total += len(model.finished_mark_ids)
                grouped_output(
                    full_dict,
                    ended_total,
                    tokenizer,
                    model.output_with_mark_ids,
                    ended_sets
                )

            current_stage = None
            if is_any_training_stopped(training_collects):
                break
            for training_collect in training_collects:
                (training_collect.status,
                training_collect.inputs) = training_collect.generator.next_value
                if current_stage is None:
                    current_stage = training_collect.status
                elif current_stage != training_collect.status:
                    raise RuntimeError(FINETUNE_ERROR_MESSAGE.format('run_finetune_inference'))
        
        elif current_infer_steps < infer_steps_per_finetune_step and model.requests_len > 0:
            current_infer_steps += 1
            model_start = time()
            model.generate()
            model_end = time()
            generation_info_lst.append(GenerateInfo(
                prefill_tokens, len(model.output_with_mark_ids),
                model_end - model_start, 0
            ))
            prefill_tokens = 0
            for mark_id in model.finished_mark_ids:
                input_list[mark_id - 1].request.input_ids = input_list[
                    mark_id - 1].request.input_ids.to('cpu')
            if collect_output:
                for mark_id in list(ended_sets.keys()):
                    if ended_sets[mark_id] == 0:
                        del ended_sets[mark_id]
                    else:
                        ended_sets[mark_id] -= 1
                ended_sets.update({
                    mark_id: 5 for mark_id in model.finished_mark_ids
                })
                ended_total += len(model.finished_mark_ids)
                grouped_output(
                    full_dict,
                    ended_total,
                    tokenizer,
                    model.output_with_mark_ids,
                    ended_sets
                )
        
        elif current_stage == TrainerProcessType.TrainForward:
            current_infer_steps = 0
            process_tokens = 0
            for training_collect in training_collects:
                inputs = training_collect.inputs
                bsz, seqlen = inputs['input_ids'].shape
                model.add_requests([InputRequest(
                    InferStatus.Train,
                    inputs['input_ids'].view(-1),
                    inputs['input_ids'].view(-1).tolist(),
                    0,
                    training_collect.lora_name,
                    attention_mask=inputs['attention_mask'],
                    batch_info=(bsz, seqlen),
                    labels=inputs['labels'],
                    accumulation_steps=training_collect.training_accum
                )])
                process_tokens += bsz * seqlen
            
            model_start = time()
            loss_lst, logits_lst = model.generate()
            model_end = time()

            trainer_start = time()
            for training_collect, loss, logits in zip(
                training_collects, loss_lst, logits_lst
            ):
                training_collect.generator.get_next_value((
                    loss, logits
                ))
            trainer_end = time()

            finetune_info_lst.append(FinetuneInfo(
                current_stage,
                process_tokens,
                model_end - model_start,
                trainer_end - trainer_start
            ))
            
            generation_info_lst.append(GenerateInfo(
                prefill_tokens, len(model.output_with_mark_ids),
                model_end - model_start, 0
            ))
            prefill_tokens = 0
            for mark_id in model.finished_mark_ids:
                input_list[mark_id - 1].request.input_ids = input_list[
                    mark_id - 1].request.input_ids.to('cpu')
            if collect_output:
                for mark_id in list(ended_sets.keys()):
                    if ended_sets[mark_id] == 0:
                        del ended_sets[mark_id]
                    else:
                        ended_sets[mark_id] -= 1
                ended_sets.update({
                    mark_id: 5 for mark_id in model.finished_mark_ids
                })
                ended_total += len(model.finished_mark_ids)
                grouped_output(
                    full_dict,
                    ended_total,
                    tokenizer,
                    model.output_with_mark_ids,
                    ended_sets
                )

            current_stage = None
            if is_any_training_stopped(training_collects):
                break
            for training_collect in training_collects:
                (training_collect.status,
                training_collect.inputs) = training_collect.generator.next_value
                if current_stage is None:
                    current_stage = training_collect.status
                elif current_stage != training_collect.status:
                    raise RuntimeError(FINETUNE_ERROR_MESSAGE.format('run_finetune_inference'))

        else: # current_stage == TrainerProcessType.Backward
            current_infer_steps = 0
            loss = sum(training_collect.inputs for training_collect in training_collects)

            model_start = time()
            loss.backward()
            model_end = time()

            trainer_start = time()
            for training_collect in training_collects:
                training_collect.generator.get_next_value()
            trainer_end = time()

            finetune_info_lst.append(FinetuneInfo(
                current_stage,
                0,
                model_end - model_start,
                trainer_end - trainer_start
            ))

            generation_info_lst.append(GenerateInfo(
                0, 0, 0, model_end - model_start
            ))

            current_stage = None
            if is_any_training_stopped(training_collects):
                break
            for training_collect in training_collects:
                (training_collect.status,
                training_collect.inputs) = training_collect.generator.next_value
                if current_stage is None:
                    current_stage = training_collect.status
                elif current_stage != training_collect.status:
                    raise RuntimeError(FINETUNE_ERROR_MESSAGE.format('run_finetune_inference'))
    
    time_idle = 0
    for next_input in input_list[input_ind:]:
        request, next_time = next_input.request, next_input.start_time

        current_time = time() - start_time
        if current_time < next_time and model.requests_len == 0:
            sleep(next_time - current_time)
            time_idle += next_time - current_time
        
        while (model.requests_len >= max_batch_size or
               (current_time < next_time and model.requests_len > 0)):
            model_start = time()
            model.generate()
            model_end = time()
            generation_info_lst.append(GenerateInfo(
                prefill_tokens, len(model.output_with_mark_ids),
                model_end - model_start, time_idle
            ))
            prefill_tokens = 0
            time_idle = 0
            for mark_id in model.finished_mark_ids:
                input_list[mark_id - 1].request.input_ids = input_list[
                    mark_id - 1].request.input_ids.to('cpu')
            if collect_output:
                for mark_id in list(ended_sets.keys()):
                    if ended_sets[mark_id] == 0:
                        del ended_sets[mark_id]
                    else:
                        ended_sets[mark_id] -= 1
                ended_sets.update({
                    mark_id: 5 for mark_id in model.finished_mark_ids
                })
                ended_total += len(model.finished_mark_ids)
                grouped_output(
                    full_dict,
                    ended_total,
                    tokenizer,
                    model.output_with_mark_ids,
                    ended_sets
                )
            current_time = time() - start_time
        
        current_delta = time() - start_time - next_time
        if max_serve_wait is not None and current_delta > max_serve_wait:
            slo_info.append(-1)
            continue
        if current_delta < 0:
            sleep(-current_delta)
            time_idle += -current_delta
            slo_info.append(0)
        else:
            slo_info.append(current_delta)
        prefill_tokens += len(request.next_input_ids)
        model.add_requests([request])
    
    while model.requests_len > 0:
        model_start = time()
        model.generate()
        generation_info_lst.append(GenerateInfo(
            prefill_tokens, len(model.output_with_mark_ids),
            model_end - model_start, time_idle
        ))
        prefill_tokens = 0
        time_idle = 0
        for mark_id in model.finished_mark_ids:
            input_list[mark_id - 1].request.input_ids = input_list[
                mark_id - 1].request.input_ids.to('cpu')
        if collect_output:
            for mark_id in list(ended_sets.keys()):
                if ended_sets[mark_id] == 0:
                    del ended_sets[mark_id]
                else:
                    ended_sets[mark_id] -= 1
            ended_sets.update({
                mark_id: 5 for mark_id in model.finished_mark_ids
            })
            ended_total += len(model.finished_mark_ids)
            grouped_output(
                full_dict,
                ended_total,
                tokenizer,
                model.output_with_mark_ids,
                ended_sets
            )
    
    end_time = time()
    print()
    return generation_info_lst, finetune_info_lst, end_time - start_time, slo_info


def run_finetune_inference_scalable(
    model: ModelGeneration,
    tokenizer: PreTrainedTokenizer,
    trainers: Dict[str, Trainer],
    input_list: List[RequestWithTime],
    max_batch_size: int,
    max_serve_wait: Optional[float] = None,
    infer_steps_per_finetune_step: int = 5,
    collect_output: bool = False
) -> Tuple[List[GenerateInfo], List[FinetuneInfo], float, List[float]]:
    generation_info_lst = []
    slo_info = []
    prefill_tokens = 0
    if collect_output:
        full_dict = {
            next_input.request.mark_id: tokenizer.decode(
                next_input.request.next_input_ids,
                skip_special_tokens=True
            ) for next_input in input_list
        }
        ended_total = 0
        ended_sets = {}
    finetune_info_lst = []
    training_collects = [
        TrainerCollect(
            lora_name,
            WrappedTrainingLoopGenerator(trainer),
            (trainer.args.gradient_accumulation_steps
             if trainer.args.gradient_accumulation_steps is not None else 1),
            (trainer.args.eval_accumulation_steps
             if trainer.args.eval_accumulation_steps is not None else 1),
        )
        for lora_name, trainer in trainers.items()
    ]

    input_ind, input_limit = 0, len(input_list)
    current_infer_steps = 0
    request, next_time = (input_list[input_ind].request,
                          input_list[input_ind].start_time)
    current_stage = None
    for training_collect in training_collects:
        (training_collect.status,
         training_collect.inputs) = training_collect.generator.next_value
        if current_stage is None:
            current_stage = training_collect.status
        elif current_stage != training_collect.status:
            raise RuntimeError(FINETUNE_ERROR_MESSAGE.format('run_finetune_inference'))
    
    start_time = time()
    while True:
        # print(infer_steps_per_finetune_step)
        if input_ind < input_limit:
            current_time = time() - start_time

            while (max_serve_wait is not None and
                   current_time > next_time + max_serve_wait):
                slo_info.append(-1)
                input_ind += 1
                if input_ind >= input_limit:
                    break
                request, next_time = (input_list[input_ind].request,
                                    input_list[input_ind].start_time)
            
            while (current_time >= next_time and
                model.requests_len < max_batch_size):
                slo_info.append(current_time - next_time)
                prefill_tokens += len(request.next_input_ids)
                model.add_requests([request])
                input_ind += 1
                if input_ind >= input_limit:
                    break
                request, next_time = (input_list[input_ind].request,
                                    input_list[input_ind].start_time)
        
        infer_steps_per_finetune_step = int((model.requests_len * 5 - 1) / max_batch_size) + 1
        
        if current_stage == TrainerProcessType.EvaluateForward:
            current_infer_steps = 0
            process_tokens = 0
            requests = []
            seq_lens = []
            for training_collect in training_collects:
                inputs = training_collect.inputs
                bsz = inputs['input_ids'].shape[0]
                for i in range(bsz):
                    input_ids = inputs['input_ids'][i, ...][
                        inputs['attention_mask'][i, ...] != 0
                    ].contiguous()
                    requests.append(InputRequest(
                        InferStatus.Prefill,
                        input_ids,
                        input_ids.tolist(),
                        0,
                        training_collect.lora_name,
                        batch_info=(1, input_ids.shape[0]),
                        labels=inputs['labels'][i, ...][
                            inputs['attention_mask'][i, ...] != 0
                        ],
                        accumulation_steps=training_collect.evaluate_accum
                    ))
                    seq_lens.append(input_ids.shape[0])
                    process_tokens += input_ids.shape[0]
            model.add_requests(requests)
            del requests

            model_start = time()
            loss_lst, logits_lst = model.generate()
            model_end = time()

            trainer_sum = 0
            for training_collect in training_collects:
                inputs = training_collect.inputs
                bsz = inputs['input_ids'].shape[0]
                current_loss_lst, loss_lst = loss_lst[:bsz], loss_lst[bsz:]
                current_seq_lens, seq_lens = seq_lens[:bsz], seq_lens[bsz:]
                current_logits_lst, logits_lst = logits_lst[:bsz], logits_lst[bsz:]
                loss = sum(loss * seq for loss, seq in zip(
                    current_loss_lst, current_seq_lens
                )) / sum(seq for seq in current_seq_lens)
                logits = torch.cat(current_logits_lst, dim=1)

                trainer_start = time()
                training_collect.generator.get_next_value((loss, logits))
                trainer_end = time()
                trainer_sum += trainer_end - trainer_start
            
            finetune_info_lst.append(FinetuneInfo(
                current_stage,
                process_tokens,
                model_end - model_start,
                trainer_sum
            ))

            generation_info_lst.append(GenerateInfo(
                prefill_tokens, len(model.output_with_mark_ids),
                model_end - model_start, 0
            ))
            prefill_tokens = 0
            for mark_id in model.finished_mark_ids:
                input_list[mark_id - 1].request.input_ids = input_list[
                    mark_id - 1].request.input_ids.to('cpu')
            if collect_output:
                for mark_id in list(ended_sets.keys()):
                    if ended_sets[mark_id] == 0:
                        del ended_sets[mark_id]
                    else:
                        ended_sets[mark_id] -= 1
                ended_sets.update({
                    mark_id: 5 for mark_id in model.finished_mark_ids
                })
                ended_total += len(model.finished_mark_ids)
                grouped_output(
                    full_dict,
                    ended_total,
                    tokenizer,
                    model.output_with_mark_ids,
                    ended_sets
                )

            current_stage = None
            if is_any_training_stopped(training_collects):
                break
            for training_collect in training_collects:
                (training_collect.status,
                training_collect.inputs) = training_collect.generator.next_value
                if current_stage is None:
                    current_stage = training_collect.status
                elif current_stage != training_collect.status:
                    raise RuntimeError(FINETUNE_ERROR_MESSAGE.format('run_finetune_inference'))
        
        elif current_infer_steps < infer_steps_per_finetune_step and model.requests_len > 0:
            current_infer_steps += 1
            model_start = time()
            model.generate()
            model_end = time()
            # finetune_info_lst.append(FinetuneInfo(
            #     3, 0, model_end - model_start, 0
            # ))
            generation_info_lst.append(GenerateInfo(
                prefill_tokens, len(model.output_with_mark_ids),
                model_end - model_start, 0
            ))
            prefill_tokens = 0
            for mark_id in model.finished_mark_ids:
                input_list[mark_id - 1].request.input_ids = input_list[
                    mark_id - 1].request.input_ids.to('cpu')
            if collect_output:
                for mark_id in list(ended_sets.keys()):
                    if ended_sets[mark_id] == 0:
                        del ended_sets[mark_id]
                    else:
                        ended_sets[mark_id] -= 1
                ended_sets.update({
                    mark_id: 5 for mark_id in model.finished_mark_ids
                })
                ended_total += len(model.finished_mark_ids)
                grouped_output(
                    full_dict,
                    ended_total,
                    tokenizer,
                    model.output_with_mark_ids,
                    ended_sets
                )
        
        elif current_stage == TrainerProcessType.TrainForward:
            current_infer_steps = 0
            process_tokens = 0
            for training_collect in training_collects:
                inputs = training_collect.inputs
                bsz, seqlen = inputs['input_ids'].shape
                model.add_requests([InputRequest(
                    InferStatus.Train,
                    inputs['input_ids'].view(-1),
                    inputs['input_ids'].view(-1).tolist(),
                    0,
                    training_collect.lora_name,
                    attention_mask=inputs['attention_mask'],
                    batch_info=(bsz, seqlen),
                    labels=inputs['labels'],
                    accumulation_steps=training_collect.training_accum
                )])
                process_tokens += bsz * seqlen
            
            model_start = time()
            loss_lst, logits_lst = model.generate()
            model_end = time()

            trainer_start = time()
            for training_collect, loss, logits in zip(
                training_collects, loss_lst, logits_lst
            ):
                training_collect.generator.get_next_value((
                    loss, logits
                ))
            trainer_end = time()

            finetune_info_lst.append(FinetuneInfo(
                current_stage,
                process_tokens,
                model_end - model_start,
                trainer_end - trainer_start
            ))
            
            generation_info_lst.append(GenerateInfo(
                prefill_tokens, len(model.output_with_mark_ids),
                model_end - model_start, 0
            ))
            prefill_tokens = 0
            for mark_id in model.finished_mark_ids:
                input_list[mark_id - 1].request.input_ids = input_list[
                    mark_id - 1].request.input_ids.to('cpu')
            if collect_output:
                for mark_id in list(ended_sets.keys()):
                    if ended_sets[mark_id] == 0:
                        del ended_sets[mark_id]
                    else:
                        ended_sets[mark_id] -= 1
                ended_sets.update({
                    mark_id: 5 for mark_id in model.finished_mark_ids
                })
                ended_total += len(model.finished_mark_ids)
                grouped_output(
                    full_dict,
                    ended_total,
                    tokenizer,
                    model.output_with_mark_ids,
                    ended_sets
                )

            current_stage = None
            if is_any_training_stopped(training_collects):
                break
            for training_collect in training_collects:
                (training_collect.status,
                training_collect.inputs) = training_collect.generator.next_value
                if current_stage is None:
                    current_stage = training_collect.status
                elif current_stage != training_collect.status:
                    raise RuntimeError(FINETUNE_ERROR_MESSAGE.format('run_finetune_inference'))

        else: # current_stage == TrainerProcessType.Backward
            current_infer_steps = 0
            loss = sum(training_collect.inputs for training_collect in training_collects)

            model_start = time()
            loss.backward()
            model_end = time()

            trainer_start = time()
            for training_collect in training_collects:
                training_collect.generator.get_next_value()
            trainer_end = time()

            finetune_info_lst.append(FinetuneInfo(
                current_stage,
                0,
                model_end - model_start,
                trainer_end - trainer_start
            ))

            generation_info_lst.append(GenerateInfo(
                0, 0, 0, model_end - model_start
            ))

            current_stage = None
            if is_any_training_stopped(training_collects):
                break
            for training_collect in training_collects:
                (training_collect.status,
                training_collect.inputs) = training_collect.generator.next_value
                if current_stage is None:
                    current_stage = training_collect.status
                elif current_stage != training_collect.status:
                    raise RuntimeError(FINETUNE_ERROR_MESSAGE.format('run_finetune_inference'))
    
    time_idle = 0
    for next_input in input_list[input_ind:]:
        request, next_time = next_input.request, next_input.start_time

        current_time = time() - start_time
        if current_time < next_time and model.requests_len == 0:
            sleep(next_time - current_time)
            time_idle += next_time - current_time
        
        while (model.requests_len >= max_batch_size or
               (current_time < next_time and model.requests_len > 0)):
            model_start = time()
            model.generate()
            model_end = time()
            generation_info_lst.append(GenerateInfo(
                prefill_tokens, len(model.output_with_mark_ids),
                model_end - model_start, time_idle
            ))
            prefill_tokens = 0
            time_idle = 0
            for mark_id in model.finished_mark_ids:
                input_list[mark_id - 1].request.input_ids = input_list[
                    mark_id - 1].request.input_ids.to('cpu')
            if collect_output:
                for mark_id in list(ended_sets.keys()):
                    if ended_sets[mark_id] == 0:
                        del ended_sets[mark_id]
                    else:
                        ended_sets[mark_id] -= 1
                ended_sets.update({
                    mark_id: 5 for mark_id in model.finished_mark_ids
                })
                ended_total += len(model.finished_mark_ids)
                grouped_output(
                    full_dict,
                    ended_total,
                    tokenizer,
                    model.output_with_mark_ids,
                    ended_sets
                )
            current_time = time() - start_time
        
        current_delta = time() - start_time - next_time
        if max_serve_wait is not None and current_delta > max_serve_wait:
            slo_info.append(-1)
            continue
        if current_delta < 0:
            sleep(-current_delta)
            time_idle += -current_delta
            slo_info.append(0)
        else:
            slo_info.append(current_delta)
        prefill_tokens += len(request.next_input_ids)
        model.add_requests([request])
    
    while model.requests_len > 0:
        model_start = time()
        model.generate()
        generation_info_lst.append(GenerateInfo(
            prefill_tokens, len(model.output_with_mark_ids),
            model_end - model_start, time_idle
        ))
        prefill_tokens = 0
        time_idle = 0
        for mark_id in model.finished_mark_ids:
            input_list[mark_id - 1].request.input_ids = input_list[
                mark_id - 1].request.input_ids.to('cpu')
        if collect_output:
            for mark_id in list(ended_sets.keys()):
                if ended_sets[mark_id] == 0:
                    del ended_sets[mark_id]
                else:
                    ended_sets[mark_id] -= 1
            ended_sets.update({
                mark_id: 5 for mark_id in model.finished_mark_ids
            })
            ended_total += len(model.finished_mark_ids)
            grouped_output(
                full_dict,
                ended_total,
                tokenizer,
                model.output_with_mark_ids,
                ended_sets
            )
    
    end_time = time()
    print()
    return generation_info_lst, finetune_info_lst, end_time - start_time, slo_info
