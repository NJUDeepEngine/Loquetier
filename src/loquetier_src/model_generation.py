from dataclasses import dataclass

import torch
from torch import nn
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PretrainedConfig
)
from transformers.generation.utils import GenerationMode
from transformers.modeling_utils import PreTrainedModel

# Typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Enums
from .enums import InferStatus

# Kernel Utils
from .kernel_utils import BatchLenInfo

# KV Management from Punica
from loquetier.utils import KvPool, KvCache, BatchedKvCache

def contrastive_search(next_token_scores: torch.FloatTensor) -> torch.Tensor:
    next_token_ids = torch.topk(next_token_scores, 2)[1][:, 0]
    return next_token_ids

def greedy_search(next_token_scores: torch.FloatTensor) -> torch.Tensor:
    next_token_ids = torch.argmax(next_token_scores, dim=-1)
    return next_token_ids

def sample(next_token_scores: torch.FloatTensor) -> torch.Tensor:
    probs = nn.functional.softmax(next_token_scores, dim=-1)
    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
    return next_token_ids

# def beam_search():
#     pass

# def beam_sample():
#     pass

# def group_beam_search():
#     pass

# def constrained_beam_search():
#     pass

def get_generation_strategy(mode: GenerationMode) -> Callable:
    if mode == GenerationMode.CONTRASTIVE_SEARCH:
        return contrastive_search
    elif mode == GenerationMode.GREEDY_SEARCH:
        return greedy_search
    elif mode == GenerationMode.SAMPLE:
        return sample
    else:
        raise NotImplementedError(f"GenerationMode {mode} Not Supported.")


@dataclass
class InputRequest:
    status: InferStatus
    input_ids: torch.LongTensor
    next_input_ids: List[int]
    mark_id: int
    lora_name: str
    lora_id: int = 0
    logits_processor: Optional[LogitsProcessorList] = None
    attention_mask: Optional[torch.Tensor] = None
    kvcache: Optional[KvCache] = None
    batch_info: Optional[Tuple[int, ...]] = None
    labels: Optional[torch.LongTensor] = None
    max_length: Optional[int] = None
    accumulation_steps: int = 1


class ModelGeneration:
    def __init__(
        self,
        model: PreTrainedModel,
        model_config: PretrainedConfig,
        generation_config: GenerationConfig,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.model_config = model_config
        self.generation_config = generation_config
        self.device = device
        self.dtype = dtype
        self.mode = generation_config.get_generation_mode()
        if isinstance(generation_config.eos_token_id, list):
            self.eos_token_ids = generation_config.eos_token_id
        else:
            self.eos_token_ids = [generation_config.eos_token_id]
        if not hasattr(generation_config, 'max_length'):
            raise ValueError('generation_config must have attribute \'max_length\'.')
        self.max_length = generation_config.max_length
        self.generation_strategy = get_generation_strategy(self.mode)
        self.model._prepare_special_tokens(generation_config, True)
        self.requests: List[InputRequest] = []
        self.train_requests: List[InputRequest] = []
        self.eval_requests: List[InputRequest] = []
        self.prefill_requests: List[InputRequest] = []
        self.output_with_mark_ids = {}
        self.finished_mark_ids = set()
        self.new_decode_request = False
        self.decode_request_changed = False
        self.cached_decode_kv = []

        if self.model_config.num_key_value_heads is None:
            num_kv_heads = self.model_config.num_attention_heads
        else:
            num_kv_heads = self.model_config.num_key_value_heads
        self.kvpool = KvPool(
            num_layers=self.model_config.num_hidden_layers,
            num_heads=num_kv_heads,
            head_dim=self.model_config.hidden_size // self.model_config.num_attention_heads,
            page_len=16,
            dtype=self.dtype,
            device=self.device
        )
    
    @property
    def requests_len(self) -> int:
        return (len(self.requests) + len(self.eval_requests) +
                len(self.prefill_requests) + len(self.train_requests))
    
    def add_requests(self, requests: List[InputRequest]):
        for r in requests:
            if r.max_length is None:
                r.max_length = self.max_length
            r.lora_id = self.model._get_lora_id(r.lora_name)
            r.input_ids = r.input_ids.to(self.device)
            if r.logits_processor is None:
                r.logits_processor = LogitsProcessorList(
                    self.model._get_logits_processor(
                        self.generation_config,
                        r.input_ids.size(-1),
                        r.input_ids,
                        None,
                        LogitsProcessorList()
                    ) + self.model._get_logits_warper(
                        self.generation_config, self.device
                    )
                )
            if r.labels is not None:
                r.labels = r.labels.to(self.device)
            if r.status == InferStatus.Prefill:
                r.kvcache = KvCache(self.kvpool, len(r.next_input_ids))
                if r.labels is None:
                    self.prefill_requests.append(r)
                else:
                    r.max_length = len(r.next_input_ids) + 1
                    self.eval_requests.append(r)
            elif r.status == InferStatus.Train:
                self.train_requests.append(r)
            else: # r.status == InferStatus.Decode, preserve for migration
                # KvCache should be moved as now this function won't do it
                self.new_decode_request = True
                self.requests.append(r)
    
    @property
    def new_train_request(self) -> bool:
        return len(self.train_requests) > 0
    
    @property
    def new_eval_request(self) -> bool:
        return len(self.eval_requests) > 0
    
    @property
    def new_prefill_request(self) -> bool:
        return len(self.prefill_requests) > 0
    
    def pre_process(self):
        self.finished_mark_ids = set()
        self.output_with_mark_ids = {}

        train_idx, prefill_kv = [], None
        bl_trains, bl_evals, bl_prefills = [], [], []
        labels_lst, accum_lst = [], []
        bl_decodes = len(self.requests)

        if self.new_decode_request:
            self.requests = sorted(self.requests, key=lambda r: (r.status, r.lora_id))
            self.new_decode_request = False
            self.cached_decode_kv = [r.kvcache for r in self.requests]
        elif self.decode_request_changed:
            self.cached_decode_kv = [r.kvcache for r in self.requests]
        
        if self.new_prefill_request or self.new_eval_request:
            self.prefill_requests = self.eval_requests + sorted(self.prefill_requests, key=lambda r: r.lora_id)
            prefill_kv = BatchedKvCache([r.kvcache for r in self.prefill_requests])
            bl_prefills = [len(r.next_input_ids) for r in self.prefill_requests]
            self.requests = self.prefill_requests + self.requests
            self.prefill_requests = []

        if self.new_train_request:
            for r in self.train_requests:
                train_idx.append(r.lora_id)
                bl_trains.append(r.batch_info)
                labels_lst.append(r.labels)
                accum_lst.append(r.accumulation_steps)
            self.requests = self.train_requests + self.requests
            self.train_requests = []
        
        if self.new_eval_request:
            for r in self.eval_requests:
                bl_evals.append(r.batch_info)
                labels_lst.append(r.labels)
                accum_lst.append(r.accumulation_steps)
            self.eval_requests = []
        
        blen = BatchLenInfo(
            bl_trains, bl_prefills, bl_decodes, bl_evals, self.device
        )
        
        for kvcache in self.cached_decode_kv:
            kvcache.acquire_one()
        
        return (
            blen, train_idx, labels_lst, accum_lst, prefill_kv,
            BatchedKvCache(self.cached_decode_kv) if len(self.cached_decode_kv) > 0 else None,
            torch.enable_grad if blen.tlen > 0 or len(bl_evals) > 0 else torch.no_grad
        )

    def post_process(self, next_token_ids: torch.LongTensor):
        '''
        Post process after model forward.

        *Note: All requests are either prefill (-> decode) or decode.*
        '''
        next_requests = []
        for i, r in enumerate(self.requests):
            r.next_input_ids = [next_token_ids[i].item()]
            self.output_with_mark_ids[r.mark_id] = r.next_input_ids
            if r.next_input_ids[0] not in self.eos_token_ids and r.input_ids.size(-1) + 1 < r.max_length:
                next_requests.append(r)
                r.input_ids = torch.cat((r.input_ids, next_token_ids[i].view(1)))
                if r.status == InferStatus.Prefill:
                    r.status = InferStatus.Decode
                    self.decode_request_changed = True
            else:
                r.kvcache.release()
                self.finished_mark_ids.add(r.mark_id)
                self.decode_request_changed = True
        self.requests = sorted(next_requests, key=lambda r: (r.status, r.lora_id))
        self.decode_request_changed &= len(self.requests) > 0

    def generate(self) -> Tuple[List[torch.Tensor], ...]:
        if self.requests_len == 0:
            return
        
        blen, train_idx, labels_lst, accum_lst, prefill_kv, decode_kv, grad_func = self.pre_process()
        
        input_ids = torch.tensor(
            sum([r.next_input_ids for r in self.requests], []),
            dtype=torch.long,
            device=self.device
        )
        if blen.tlen > 0:
            attention_mask = [r.attention_mask for r in self.requests if r.attention_mask is not None]
        else:
            attention_mask = None

        lora_ids, lora_lens = [self.requests[0].lora_id], [0]
        for r in self.requests:
            if lora_ids[-1] == r.lora_id:
                lora_lens[-1] += len(r.next_input_ids)
            else:
                lora_ids.append(r.lora_id)
                lora_lens.append(len(r.next_input_ids))

        model_kwargs = self.model.prepare_inputs_for_generation(
            input_ids, attention_mask, blen=blen, lora_ids=lora_ids,
            lora_len=torch.tensor(lora_lens, dtype=torch.int32, device=self.device),
            segment=torch.cumsum(
                torch.tensor([0] + lora_lens, dtype=torch.int32, device=self.device),
                dim=0, dtype=torch.int32
            ),
            train_idx=train_idx
        )
        model_kwargs.update({
            'blen': blen,
            'labels': labels_lst,
            'accums': accum_lst,
            'prefill_kv': prefill_kv,
            'decode_kv': decode_kv
        })

        with grad_func():
            loss, logits = self.model(**model_kwargs)
        
        training_logits = []
        if blen.tlen > 0:
            for train_batch, pi in zip(blen.train_batches, range(len(blen.train_partition) - 1)):
                bsz, seq_length = train_batch
                start, end = blen.train_partition[pi], blen.train_partition[pi + 1]
                training_logits.append(logits[start:end].view(bsz, seq_length, -1))
            self.requests = self.requests[len(blen.train_batches):]
        if len(blen.eval_batches) > 0:
            for eval_batch, pi, r in zip(
                blen.eval_batches, range(len(blen.eval_partition) - 1), self.requests
            ):
                r.kvcache.release()
                bsz, seq_length = eval_batch
                start, end = blen.eval_partition[pi], blen.eval_partition[pi + 1]
                training_logits.append(logits[start:end].view(bsz, seq_length, -1))
            self.requests = self.requests[len(blen.eval_batches):]

        eval_skips = len(blen.eval_batches)
        if len(blen.prefills) - eval_skips > 0:
            if blen.decode > 0:
                next_token_logits = torch.cat(
                    [logits[blen.indptr[eval_skips+1:] - 1], logits[blen.doff:]]
                )
            else:
                next_token_logits = logits[blen.indptr[eval_skips+1:] - 1]
        elif blen.decode > 0:
            next_token_logits = logits[blen.doff:]
        else:
            return loss, training_logits
        
        next_token_scores = torch.empty_like(
            next_token_logits,
            dtype=next_token_logits.dtype,
            device=next_token_logits.device
        )
        for i, r in enumerate(self.requests):
            next_token_scores[i, ...] = r.logits_processor(r.input_ids.view(1, -1), next_token_logits[i, ...].view(1, -1)).view(-1)
        next_token_ids = self.generation_strategy(
            next_token_scores
        )
        self.post_process(next_token_ids)

        return loss, training_logits
