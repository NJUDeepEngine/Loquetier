import torch
import transformers
from transformers.models.llama import LlamaForCausalLM
from transformers.generation import GenerationConfig, LogitsProcessorList
from loquetier.utils import KvPool, KvCache, BatchedKvCache
from time import time

import os
import sys
sys.path.append('../src/')
from loquetier_src.models.mixed_llama import MixedLlamaForCausalLM
from loquetier_src.kernel_utils import *

if __name__ == "__main__":
    final_output = ''
    model_path = ''
    assert model_path, (
        'Put your model path here. This is a simple test for llama 3 models, '
        'if you\'re using other models, you may have to modify this test file.'
    )
    device = 'cuda:0'
    max_new_length = 600
    dtype = torch.bfloat16
    model = MixedLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=dtype).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    generation_config = GenerationConfig.from_pretrained(model_path)
    prompt = "[INST] Give me some travel tips for Tokyo\n[/INST]"
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)

    model_config = transformers.LlamaConfig.from_pretrained(model_path)
    logits_processor = LogitsProcessorList()
    logits_processor.append(
        transformers.TemperatureLogitsWarper(0.9)
    )
    logits_processor.append(
        transformers.RepetitionPenaltyLogitsProcessor(1.1)
    )
    logits_processor.append(transformers.TopPLogitsWarper(0.8))

    if model_config.num_key_value_heads is None:
        num_kv_heads = model_config.num_attention_heads
    else:
        num_kv_heads = model_config.num_key_value_heads
    kvpool = KvPool(
        num_layers=model_config.num_hidden_layers,
        num_heads=num_kv_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        page_len=16,
        dtype=dtype,
        device=device,
    )
    kvcache = KvCache(kvpool, len(input_ids))
    x = torch.arange(10, device='cuda:0')

    in_model_time = 0
    real_start = time()
    start = real_start
    output_ids = model(
        input_ids=input_ids,
        blen=BatchLenInfo([(0, 0)], [len(input_ids)], 0, [], device),
        prefill_kv=BatchedKvCache([kvcache]),
        decode_kv=None
    )
    end = time()
    in_model_time += end - start
    output_len = 1

    next_token_logits = output_ids[1][-1, :]
    next_token_scores = logits_processor(input_ids.view(1, -1), next_token_logits.view(1, -1)).view(-1)
    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    output = tokenizer.decode(
        next_token_id,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=True
    )
    final_output += output
    print(output, end="", flush=False)

    while next_token_id.item() != generation_config.eos_token_id and output_len < max_new_length:
        kvcache.acquire_one()
        start = time()
        output_ids = model(
            input_ids=torch.tensor([next_token_id], dtype=torch.long, device=device),
            blen=BatchLenInfo([(0, 0)], [], 1, [], device),
            prefill_kv=None,
            decode_kv=BatchedKvCache([kvcache])
        )
        end = time()
        in_model_time += end - start
        output_len += 1

        next_token_logits = output_ids[1][-1, :]
        next_token_scores = logits_processor(input_ids.view(1, -1), next_token_logits.view(1, -1)).view(-1)
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        output = tokenizer.decode(
            next_token_id,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True
        )
        final_output += output
        print(output, end="", flush=False)

    real_end = time()
    in_tokens = input_ids.shape[0]
    out_tokens = output_len
    print(final_output)
    print(f'Prefill {in_tokens} | Decode {out_tokens}')
    print(f'Total {(real_end-real_start):.3f} Secs | Model {in_model_time:.3f} Secs')
    print(f'Average Throughput {in_tokens/in_model_time:.3f} {out_tokens/in_model_time:.3f}')

    kvcache.release()
