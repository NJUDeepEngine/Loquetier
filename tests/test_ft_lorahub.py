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
from loquetier_src.model_workflow import run_finetune as run_offline

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-c', '--config', type=str, help='config file')
    parser.add_argument('-d', '--device', type=int, help='cuda device index')
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

    config_file = args.config
    output_file = args.output_file
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    with open(config_file, 'r') as f:
        config = json.load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    torch.cuda.device_count.cache_clear()
    device_str = "cuda:0"

    base_model_path = config['base_model_path']
    save_model_path = config['save_model_path']
    lora_config = LoraConfig(**config['lora_config'])
    training_args = config['training_args']
    testcases = config['testcases']
    lora_names = [testcase['lora_name'] for testcase in testcases]

    base_model = MixedLlamaForCausalLM.from_pretrained(
        base_model_path, low_cpu_mem_usage=True, torch_dtype=dtype
    ).to(device_str)
    LoquetierFramework.BaseModelList[base_model_path] = base_model
    tokenizer_llama = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=False
    )
    tokenizer_llama.pad_token = tokenizer_llama.eos_token
    tokenizer_llama.padding_side = 'left'

    model_with_loras = []
    for lora_name in lora_names:
        vml = create_virtual_model(base_model)
        model_with_loras.append(get_peft_model(
            vml, lora_config
        ))
    
    vml = create_virtual_model(base_model)
    mixed_model = MixedLoraModel(
        vml,
        { lora_name: model_with_lora.base_model
        for lora_name, model_with_lora in zip(lora_names, model_with_loras) },
        apply_scaling='B'
    )
    mixed_model.set_trainable(list(range(len(lora_names))))
    mixed_training_models = {
        lora_name: MixedLoraModelForTrainer(mixed_model, i)
        for i, lora_name in enumerate(lora_names)
    }

    model_config = transformers.LlamaConfig.from_pretrained(
        base_model_path
    )
    generation_config = GenerationConfig.from_dict(
        config['generation_config']
    ) if 'generation_config' in config else GenerationConfig.from_pretrained(
        base_model_path
    )
    model_gen = ModelGeneration(mixed_model, model_config, generation_config, torch.device(device_str))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_llama,
        mlm=False
    )

    trainers = {}
    for testcase in config['testcases']:
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
    
    model_gen.max_length = max(model_gen.max_length, 1024)
    
    ft_res, time_cost = run_offline(model_gen, trainers)
    
    model_time_cost = sum(r.time_cost for r in ft_res)
    trainer_time_cost = sum(r.time_idle for r in ft_res)
    finetune_tokens = sum(r.process_tokens for r in ft_res if r.process_type == TrainerProcessType.TrainForward)
    evaluate_tokens = sum(r.process_tokens for r in ft_res if r.process_type == TrainerProcessType.EvaluateForward)
    finetune_time_cost = sum(r.time_cost for r in ft_res if r.process_type == TrainerProcessType.TrainForward)
    backward_time_cost = sum(r.time_cost for r in ft_res if r.process_type == TrainerProcessType.Backward)
    print(f'Model Generation {model_time_cost:.3f} Secs / {time_cost:.3f} Secs In Total')
    print(f'Finetune Tokens {finetune_tokens} | Evaluate Tokens {evaluate_tokens}')
    print(f'Finetune Throughput {(finetune_tokens/finetune_time_cost):.2f} Tokens Per Sec')
    print(f'Backward Throughput {(finetune_tokens/backward_time_cost):.2f} Tokens Per Sec')
    print(f'Mean Throughput {(finetune_tokens/(finetune_time_cost+backward_time_cost)):.2f} Tokens Per Sec')
    print()
    print('* NOTICE! * The mean throughput above is a simple division of tokens and time, which may not be the real throughput.')

    if not os.path.exists('test/result'):
        os.mkdir('test/result')
    with open(f'test/result/{output_file}', 'w') as f:
        f.writelines(
            f'{finetune_info.process_type.value} {finetune_info.process_tokens} '
            f'{finetune_info.time_cost:.3f} {finetune_info.time_idle:.3f}\n'
            for finetune_info in ft_res
        )
