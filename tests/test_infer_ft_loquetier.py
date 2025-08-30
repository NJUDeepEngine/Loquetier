import argparse
from functools import partial
import json
import os

from datasets import load_dataset
import numpy as np
from peft import LoraConfig, get_peft_model
import torch
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    Trainer,
    TrainingArguments
)

import os
import sys
sys.path.append('../src/')
from loquetier_src import LoquetierFramework, create_virtual_model
from loquetier_src.enums import TrainerProcessType
from loquetier_src.models.mixed_llama import MixedLlamaForCausalLM
from loquetier_src.models.mixed_lora import *
from loquetier_src.model_generation import *
from loquetier_src.model_workflow import (
    RequestWithTime,
    run_finetune_inference as run_offline,
    run_finetune_inference_scalable as run_scalable
)
from interval_generator import *

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-i', '--infer_config', type=str,
                        help='inference config file')
    parser.add_argument('-f', '--ft_config', type=str,
                        help='fintune config file')
    parser.add_argument('-d', '--device', type=int, help='cuda device index')
    parser.add_argument('-b', '--max_batch_size', type=int,
                        default=16, help='max batch size')
    parser.add_argument('-s', '--steps', type=int,
                        default=5, help='infer steps per finetune step')
    parser.add_argument('-t', '--dtype', type=str, choices=['float16', 'bfloat16'],
                        default='bfloat16', help='torch dtype')
    parser.add_argument('-o', '--output_file', type=str,
                        default='result.txt', help='output result filename')

    return parser.parse_args()

def tokenize_func(examples, tokenizer):
    return tokenizer(
        examples['prompt'],
        truncation=True,
        max_length=1024,
        padding='max_length',
        return_tensors='pt'
    )

if __name__ == "__main__":
    os.environ["TRANSFORMERS_OFFLINE"] = "TRUE"
    os.environ["HF_DATASETS_OFFLINE"] = "TRUE"

    args = parse_args()

    infer_config_file = args.infer_config
    ft_config_file = args.ft_config
    max_batch_size = args.max_batch_size
    output_file = args.output_file
    steps = args.steps
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    with open(ft_config_file, 'r') as f:
        ft_config = json.load(f)
    with open(infer_config_file, 'r') as f:
        infer_config = json.load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    torch.cuda.device_count.cache_clear()
    device_str = "cuda:0"

    if ft_config['base_model_path'] != infer_config['base_model_path']:
        raise ValueError(
            'Inference config should have the same base model path as finetune config, but got '
            f'{infer_config["base_model_path"]} and {ft_config["base_model_path"]}'
        )
    base_model_path = ft_config['base_model_path']
    lora_model_path = infer_config['lora_model_path']
    infer_lora_names = infer_config['lora_names']
    save_model_path = ft_config['save_model_path']
    lora_config = LoraConfig(**ft_config['lora_config'])
    training_args = ft_config['training_args']
    testcases = ft_config['testcases']
    ft_lora_names = [testcase['lora_name'] for testcase in testcases]

    base_model = MixedLlamaForCausalLM.from_pretrained(
        base_model_path, low_cpu_mem_usage=True, torch_dtype=dtype
    ).to(device_str)
    LoquetierFramework.BaseModelList[base_model_path] = base_model
    tokenizer_llama = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=False
    )
    tokenizer_llama.pad_token = tokenizer_llama.eos_token
    tokenizer_llama.padding_side = 'left'

    if len(set(ft_lora_names).intersection(infer_lora_names)) > 0:
        raise ValueError('Finetune and inference should have no same lora names.')
    model_with_loras = []
    for lora_name in ft_lora_names:
        vml = create_virtual_model(base_model)
        model_with_loras.append(get_peft_model(
            vml, lora_config
        ))
    for lora_name in infer_lora_names:
        model_with_loras.append(LoquetierFramework.load_model(
        lora_model_path + lora_name, use_fast=False
    ))
    
    vml = create_virtual_model(base_model)
    mixed_model = MixedLoraModel(
        vml,
        { lora_name: model_with_lora.base_model
        for lora_name, model_with_lora in zip(ft_lora_names + infer_lora_names, model_with_loras) },
        apply_scaling='B'
    )
    mixed_model.set_trainable(list(range(len(ft_lora_names))))
    mixed_training_models = {
        lora_name: MixedLoraModelForTrainer(mixed_model, i)
        for i, lora_name in enumerate(ft_lora_names)
    }

    model_config = transformers.LlamaConfig.from_pretrained(
        base_model_path
    )
    generation_config = GenerationConfig.from_dict(
        ft_config['generation_config']
    ) if 'generation_config' in ft_config else GenerationConfig.from_pretrained(
        base_model_path
    )
    model_gen = ModelGeneration(mixed_model, model_config, generation_config, torch.device(device_str))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_llama,
        mlm=False
    )

    trainers = {}
    for testcase in ft_config['testcases']:
        lora_name = testcase['lora_name']
        dataset_type = testcase['dataset_type']
        dataset_files = testcase['dataset_files']
        output_dir = testcase['output_dir']
        
        dataset = load_dataset(dataset_type, data_files=dataset_files)
        tokenized_dataset = dataset.map(
            partial(tokenize_func, tokenizer=tokenizer_llama),
            batched=True,
            num_proc=4,
            remove_columns=['prompt']
        )
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            **training_args
        )
        trainers[lora_name] = Trainer(
            model=mixed_training_models[lora_name],
            args=training_arguments,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=(tokenized_dataset['eval'] if 'eval' in tokenized_dataset
                          else tokenized_dataset['train']),
            data_collator=data_collator
        )
    
    all_requests = {}
    mark_id = 1
    for testcase in infer_config['testcases']:
        jsonl_file = testcase['jsonl_path']
        lora_name = testcase['lora_name']
        sample_num = testcase['sample_num']
        sample_offset = testcase['sample_offset'] if 'sample_offset' in testcase else 0
        request_start = testcase['request_start']
        request_sample_method = INTERVAL_GENERATORS.get(testcase['request_sample_method'], None)
        if request_sample_method is not None:
            intervals = np.cumsum(list(
                request_sample_method(
                    sample_num,
                    testcase['request_sample_interval'],
                    testcase['request_sample_deviation']
            )))
        else:
            intervals = [request_start] * sample_num
        
        length = 0
        requests = []
        with open(jsonl_file, 'r') as file:
            line = file.readline()
            for _ in range(sample_offset):
                line = file.readline()
            while line and sample_num > 0:
                input_ids = None
                d = json.loads(line)['conversations']
                for conv in d:
                    if conv['from'] == 'human':
                        input_ids = tokenizer_llama.encode(
                            conv['value']
                        )
                        break
                if input_ids is None:
                    line = file.readline()
                    continue
                requests.append(
                    InputRequest(
                        InferStatus.Prefill,
                        torch.tensor(input_ids, dtype=torch.long, device=device_str),
                        input_ids,
                        mark_id,
                        lora_name
                    )
                )
                length = max(length, len(input_ids))
                sample_num -= 1
                mark_id += 1
                line = file.readline()
        
        all_requests[lora_name] = (
            requests, request_start, intervals, length
        )
    
    input_list = sorted(
        (
            RequestWithTime(request, start_time + cumulation)
            for _, (requests, start_time, cumulations, _) in all_requests.items()
            for request, cumulation in zip(requests, cumulations)
        ),
        key=lambda rt: rt.start_time
    )
    if len(infer_config['testcases']) > 1:
        for i, r in enumerate(input_list):
            r.request.mark_id = i + 1
    
    model_gen.max_length = max(
        min(
            model_gen.max_length,
            max(all_requests.values(), key=lambda rs: rs[3])[3] + 200
        ),
        1024
    )
    
    gen_res, ft_res, time_cost, slo_info = run_offline(
        model_gen,
        tokenizer_llama,
        trainers,
        input_list,
        max_batch_size,
        max_serve_wait=6,
        infer_steps_per_finetune_step=steps
    )
    
    slo_attainment = (len(input_list) - slo_info.count(-1)) / len(input_list)
    gen_model_time_cost = sum(r.time_cost for r in gen_res)
    prefill_tokens = sum(r.prefill_tokens for r in gen_res)
    decode_tokens = sum(r.decode_tokens for r in gen_res)

    ft_model_time_cost = sum(r.time_cost for r in ft_res)
    trainer_time_cost = sum(r.time_idle for r in ft_res)
    finetune_tokens = sum(r.process_tokens for r in ft_res if r.process_type == TrainerProcessType.TrainForward)
    evaluate_tokens = sum(r.process_tokens for r in ft_res if r.process_type == TrainerProcessType.EvaluateForward)
    finetune_time_cost = sum(r.time_cost for r in ft_res if r.process_type == TrainerProcessType.TrainForward)
    backward_time_cost = sum(r.time_cost for r in ft_res if r.process_type == TrainerProcessType.Backward)
    
    print('='*20 + 'Generation Info' + '='*20)
    print(f'Model Generation {gen_model_time_cost:.3f} Secs / {time_cost:.3f} Secs In Total')
    print(f'Prefill Tokens {prefill_tokens} | Decode Tokens {decode_tokens}')
    print(f'Mean Throughput {(prefill_tokens/gen_model_time_cost):.2f} | {(decode_tokens/gen_model_time_cost):.2f} Tokens Per Sec')
    print(f'SLO Attainment {(slo_attainment * 100):.2f}% | {len(input_list) - slo_info.count(-1)} of {len(input_list)}')
    print()
    print('='*20 + 'Finetune Info' + '='*20)
    print(f'Model Generation {ft_model_time_cost:.3f} Secs / {time_cost:.3f} Secs In Total')
    print(f'Finetune Tokens {finetune_tokens} | Evaluate Tokens {evaluate_tokens}')
    print(f'Finetune Throughput {(finetune_tokens/finetune_time_cost):.2f} Tokens Per Sec')
    print(f'Backward Throughput {(finetune_tokens/backward_time_cost):.2f} Tokens Per Sec')
    print(f'Mean Throughput {(finetune_tokens/(finetune_time_cost+backward_time_cost)):.2f} Tokens Per Sec')
    print()
    print('* NOTICE! * The mean throughput above is a simple division of tokens and time, which may not be the real throughput.')

    if not os.path.exists('test/result'):
        os.mkdir('test/result')
    filename, filetype = output_file.split(".")
    with open(f'test/result/{filename}_I.{filetype}', 'w') as f:
        f.writelines(
            ' '.join(f'{s:.3f}' if s >= 0 else str(s) for s in slo_info)
        )
        f.writelines('\n\n')
        f.writelines(
            f'{generate_info.prefill_tokens} {generate_info.decode_tokens} '
            f'{generate_info.time_cost:.3f} {generate_info.time_idle:.3f}\n'
            for generate_info in gen_res
        )
    with open(f'test/result/{filename}_F.{filetype}', 'w') as f:
        f.writelines(
            f'{finetune_info.process_type.value} {finetune_info.process_tokens} '
            f'{finetune_info.time_cost:.3f} {finetune_info.time_idle:.3f}\n'
            for finetune_info in ft_res
        )
        # use writelines below in scalable test
        # f.writelines(
        #     f'{finetune_info.process_type if isinstance(finetune_info.process_type, int) else finetune_info.process_type.value} '
        #     f'{finetune_info.process_tokens} {finetune_info.time_cost:.3f} {finetune_info.time_idle:.3f}\n'
        #     for finetune_info in ft_res
        # )
