import argparse
import json
import os

import numpy as np
from scipy.stats import truncnorm
import torch
import transformers
from transformers import AutoTokenizer

import os
import sys
sys.path.append('../src/')
from loquetier_src import LoquetierFramework, create_virtual_model
from loquetier_src.models.mixed_lora import *
from loquetier_src.models.mixed_llama import MixedLlamaForCausalLM
from loquetier_src.model_generation import *
from loquetier_src.model_workflow import (
    RequestWithTime,
    run_inference as run_offline
)


def NormalInterval(length, total_time, mean_ratio=0.5, std_ratio=0.1, **kwargs):
    offset = kwargs.get('offest', 0)
    intervals = sorted(
        truncnorm.rvs(
            -mean_ratio / std_ratio,
            (1 - mean_ratio) / std_ratio,
            loc=total_time * mean_ratio,
            scale=total_time * std_ratio,
            size=length
        )
    )
    last_interval = -offset
    for interval in intervals:
        yield interval - last_interval
        last_interval = interval
    return

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-c', '--config', type=str, help='config file')
    parser.add_argument('-d', '--device', type=int, help='cuda device index')
    parser.add_argument('-b', '--max_batch_size', type=int,
                        default=16, help='max batch size')
    parser.add_argument('-t', '--dtype', type=str, choices=['float16', 'bfloat16'],
                        default='bfloat16', help='torch dtype')
    parser.add_argument('-o', '--output_file', type=str,
                        default='result.txt', help='output result filename')

    return parser.parse_args()

if __name__ == "__main__":
    os.environ["TRANSFORMERS_OFFLINE"] = "TRUE"
    os.environ["HF_DATASETS_OFFLINE"] = "TRUE"

    args = parse_args()

    config_file = args.config
    device_str = f"cuda:{args.device}"
    max_batch_size = args.max_batch_size
    output_file = args.output_file
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    with open(config_file, 'r') as f:
        config = json.load(f)

    base_model_path = config['base_model_path']
    lora_model_path = config['lora_model_path']
    lora_names = config['lora_names']

    base_model = MixedLlamaForCausalLM.from_pretrained(
        base_model_path, low_cpu_mem_usage=True, torch_dtype=dtype
    ).to(device_str)
    LoquetierFramework.BaseModelList[base_model_path] = base_model
    tokenizer_llama = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=False
    )
    tokenizer_llama.pad_token = tokenizer_llama.eos_token
    tokenizer_llama.padding_side = 'left'

    model_with_loras = [LoquetierFramework.load_model(
        lora_model_path + lora_name, use_fast=False
    ) for lora_name in lora_names]
    
    vml = create_virtual_model(base_model)
    mixed_model = MixedLoraModel(
        vml,
        { lora_name: model_with_lora.base_model
        for lora_name, model_with_lora in zip(lora_names, model_with_loras) },
        apply_scaling = 'A'
    )

    model_config = transformers.LlamaConfig.from_pretrained(
        base_model_path
    )
    generation_config = GenerationConfig.from_dict(
        config['generation_config']
    ) if 'generation_config' in config else GenerationConfig.from_pretrained(
        base_model_path
    )
    model_gen = ModelGeneration(mixed_model, model_config, generation_config, torch.device(device_str))

    all_requests = {}
    mark_id = 1
    for testcase in config['testcases']:
        jsonl_file = testcase['jsonl_path']
        lora_name = testcase['lora_name']
        sample_num = testcase['sample_num']
        sample_offset = testcase['sample_offset'] if 'sample_offset' in testcase else 0
        request_start = testcase['request_start']
        request_sample_method = NormalInterval
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
    
    model_gen.max_length = min(model_gen.max_length, max(all_requests.values(), key=lambda rs: rs[3])[3] + 400)
    
    input_list = sorted(
        (
            RequestWithTime(request, start_time + cumulation)
            for _, (requests, start_time, cumulations, _) in all_requests.items()
            for request, cumulation in zip(requests, cumulations)
        ),
        key=lambda rt: rt.start_time
    )
    if len(config['testcases']) > 1:
        for i, r in enumerate(input_list):
            r.request.mark_id = i + 1
    
    gen_res, time_cost, slo_info = run_offline(
        model_gen,
        tokenizer_llama,
        input_list,
        max_batch_size,
        max_serve_wait=6
    )
    
    slo_attainment = (len(input_list) - slo_info.count(-1)) / len(input_list)
    model_time_cost = sum(r.time_cost for r in gen_res)
    prefill_tokens = sum(r.prefill_tokens for r in gen_res)
    decode_tokens = sum(r.decode_tokens for r in gen_res)
    print(f'Model Generation {model_time_cost:.3f} Secs / {time_cost:.3f} Secs In Total')
    print(f'Prefill Tokens {prefill_tokens} | Decode Tokens {decode_tokens}')
    print(f'Mean Throughput {(prefill_tokens/model_time_cost):.2f} | {(decode_tokens/model_time_cost):.2f} Tokens Per Sec')
    print(f'SLO Attainment {(slo_attainment * 100):.2f}% | {len(input_list) - slo_info.count(-1)} of {len(input_list)}')
    print()
    print('* NOTICE! * The mean throughput above is a simple division of tokens and time, which is not the real throughput.')

    if not os.path.exists('test/result'):
        os.mkdir('test/result')
    with open(f'test/result/{output_file}', 'w') as f:
        f.writelines(
            ' '.join(f'{s:.3f}' if s >= 0 else str(s) for s in slo_info)
        )
        f.writelines('\n\n')
        f.writelines(
            f'{generate_info.prefill_tokens} {generate_info.decode_tokens} '
            f'{generate_info.time_cost:.3f} {generate_info.time_idle:.3f}\n'
            for generate_info in gen_res
        )
