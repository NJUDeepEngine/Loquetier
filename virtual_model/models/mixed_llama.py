import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaMLP,
    LlamaAttention,
    PreTrainedModel,
)

from ..kernel_utils import (
    BatchLenInfo,
    rms_norm
)

from punica.ops import (
    append_kv,
    batch_decode,
    batch_prefill,
    init_kv
)
from punica.utils.kvcache import BatchedKvCache

logger = logging.get_logger(__name__)


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


class MixedLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.bfloat16):
        """
        MixedLlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=self.dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)
        return rms_norm.apply(hidden_states, self.weight, self.variance_epsilon).to(input_dtype)


ALL_LAYERNORM_LAYERS.append(MixedLlamaRMSNorm)


class MixedLlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MixedLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.dtype = self.q_proj.weight.dtype
        self.layer_idx = layer_idx
        self.sqrt_head_dim = math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        blen: BatchLenInfo,
        attention_mask: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[List[torch.LongTensor]] = None,
        prefill_kv: Optional[BatchedKvCache] = None,
        decode_kv: Optional[BatchedKvCache] = None,
        **kwargs
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        stack_attn_output = []

        if blen.tlen > 0:
            for train_batch, pos_ids, mask, pi in zip(
                blen.train_batches, position_ids, attention_mask,
                range(len(blen.train_partition) - 1)
            ):
                bsz, q_len = train_batch
                start, end = blen.train_partition[pi], blen.train_partition[pi + 1]
                q = query_states[start:end, ...].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = key_states[start:end, ...].view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                v = value_states[start:end, ...].view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                cos, sin = self.rotary_emb(v, position_ids=pos_ids)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)

                k = repeat_kv(k, self.num_key_value_groups)
                v = repeat_kv(v, self.num_key_value_groups)
                attn_weights = torch.matmul(q, k.transpose(2, 3)) / self.sqrt_head_dim

                if attn_weights.size() != (bsz, self.num_heads, q_len, q_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, self.num_heads, q_len, q_len)}, but is"
                        f" {attn_weights.size()}"
                    )
                if mask is not None:
                    if mask.size() != (bsz, 1, q_len, q_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, q_len)}, but is {mask.size()}"
                        )
                    attn_weights = attn_weights + mask

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
                attn_output = torch.matmul(attn_weights, v)

                if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                    raise ValueError(
                        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                        f" {attn_output.size()}"
                    )

                attn_output = attn_output.transpose(1, 2).contiguous().view(end - start, self.hidden_size)
                stack_attn_output.append(attn_output)
        
        query_states = query_states.to(dtype=self.dtype)
        key_states = key_states.to(dtype=self.dtype)
        value_states = value_states.to(dtype=self.dtype)

        if len(blen.prefills) > 0:
            assert prefill_kv is not None
            assert blen.indptr is not None
            q = query_states[blen.tlen:blen.doff, ...].view(blen.doff-blen.tlen, self.num_heads, self.head_dim)
            k = key_states[blen.tlen:blen.doff, ...].view(blen.doff-blen.tlen, self.num_key_value_heads, self.head_dim)
            v = value_states[blen.tlen:blen.doff, ...].view(blen.doff-blen.tlen, self.num_key_value_heads, self.head_dim)
            init_kv(prefill_kv, k, v, blen.indptr, self.layer_idx)
            attn_output = batch_prefill(q, blen.indptr, prefill_kv, self.rope_theta, self.layer_idx)
            attn_output = attn_output.view(blen.doff-blen.tlen, self.hidden_size)
            stack_attn_output.append(attn_output.to(self.dtype))

        if blen.decode > 0:
            q = query_states[blen.doff :].view(blen.decode, self.num_heads, self.head_dim)
            k = key_states[blen.doff :].view(blen.decode, self.num_key_value_heads, self.head_dim)
            v = value_states[blen.doff :].view(blen.decode, self.num_key_value_heads, self.head_dim)

            assert decode_kv is not None
            append_kv(decode_kv, k, v, self.layer_idx)

            attn_output = batch_decode(q, decode_kv, self.rope_theta, self.layer_idx)
            attn_output = attn_output.view(blen.decode, self.hidden_size)
            stack_attn_output.append(attn_output.to(self.dtype))

        if len(stack_attn_output) == 1:
            attn_output = stack_attn_output[0]
        else:
            attn_output = torch.cat(stack_attn_output, dim=0)

        attn_output = self.o_proj(attn_output.to(input_dtype))

        return attn_output


class MixedLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            MixedLlamaAttention(config=config, layer_idx=layer_idx)
        )
        self.mlp = MixedLlamaMLP(config)
        self.input_layernorm = MixedLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixedLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        blen: BatchLenInfo,
        attention_mask: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[List[torch.LongTensor]] = None,
        prefill_kv: Optional[BatchedKvCache] = None,
        decode_kv: Optional[BatchedKvCache] = None,
        **kwargs
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            blen=blen,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prefill_kv=prefill_kv,
            decode_kv=decode_kv,
            **kwargs
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MixedLlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MixedLlamaDecoderLayer"]
    _supports_flash_attn_2 = False


class MixedLlamaModel(MixedLlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            MixedLlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = MixedLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        blen: BatchLenInfo,
        attention_mask: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[List[torch.LongTensor]] = None,
        prefill_kv: Optional[BatchedKvCache] = None,
        decode_kv: Optional[BatchedKvCache] = None
    ) -> torch.FloatTensor:
        inputs_embeds = self.embed_tokens(input_ids)

        if blen.tlen > 0:
            if position_ids is None:
                position_ids = [None] * (len(blen.train_partition) - 1)
            prepared_attention_mask = []
            prepared_position_ids = []
            for train_batch, pos_ids, mask in zip(
                blen.train_batches, position_ids, attention_mask
            ):
                batch_size, seq_length = train_batch

                if pos_ids is None:
                    device = input_ids.device
                    pos_ids = torch.arange(
                        seq_length, dtype=torch.long, device=device
                    )
                    pos_ids = pos_ids.unsqueeze(0)
                mask = _prepare_4d_causal_attention_mask(
                    mask, (batch_size, seq_length), inputs_embeds, 0
                )
                prepared_attention_mask.append(mask)
                prepared_position_ids.append(pos_ids)
            attention_mask = prepared_attention_mask
            position_ids = prepared_position_ids

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                blen,
                attention_mask=attention_mask,
                position_ids=position_ids,
                prefill_kv=prefill_kv,
                decode_kv=decode_kv
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class MixedLlamaForCausalLM(MixedLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MixedLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        blen: BatchLenInfo = None,
        attention_mask: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[List[torch.LongTensor]] = None,
        prefill_kv: Optional[BatchedKvCache] = None,
        decode_kv: Optional[BatchedKvCache] = None,
        labels: List[torch.LongTensor] = [],
        accums: List[int] = [],
        **kwargs
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        hidden_states = self.model(
            input_ids=input_ids,
            blen=blen,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prefill_kv=prefill_kv,
            decode_kv=decode_kv
        )

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = []
        if len(labels) > 0:
            label_batches = blen.train_batches + blen.eval_batches
            train_partition_len = (len(blen.train_partition) - 1 if len(blen.train_partition) > 0 else 0)
            partition_len = (train_partition_len +
                             (len(blen.eval_partition) - 1 if len(blen.eval_partition) > 0 else 0))
            if len(label_batches) != len(labels):
                raise ValueError(
                    f'Expect length of labels to be {len(blen.train_batches)}, but got {len(labels)}.'
                )
            if len(label_batches) != len(accums):
                raise ValueError(
                    f'Expect length of accumulation steps to be {len(blen.train_batches)}, '
                    f'but got {len(accums)}.'
                )
            for label_batch, label, accum_step, pi in zip(
                label_batches, labels, accums, range(partition_len)
            ):
                bsz, seq_length = label_batch
                if pi >= train_partition_len:
                    start, end = (blen.eval_partition[pi - train_partition_len],
                                  blen.eval_partition[pi - train_partition_len + 1])
                else:
                    start, end = blen.train_partition[pi], blen.train_partition[pi + 1]
                
                shift_logits = logits[start:end, :].view(bsz, seq_length, -1)[:, :-1, ...].contiguous()
                shift_labels = label[..., 1:].contiguous()
                
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                
                shift_labels = shift_labels.to(shift_logits.device)
                loss.append(loss_fct(shift_logits, shift_labels) / accum_step)

        return loss, logits
        return logits, loss

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = []
            for mask in attention_mask:
                pos_ids = mask.long().cumsum(-1) - 1
                pos_ids.masked_fill_(mask == 0, 1)
                position_ids.append(pos_ids)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }
