import torch

import loquetier.ops._kernels as _kernels
from loquetier.utils.kvcache import BatchedKvCache

_cache_buf = {}


def _get_cache_buf(name: str, bytes: int, device: torch.device):
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = torch.empty(bytes, dtype=torch.uint8, device=device)
        _cache_buf[key] = buf
    return buf


def batch_prefill(
    q: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv: BatchedKvCache,
    base: float,
    layer_idx: int,
) -> torch.Tensor:
    """
    Perform self-attention with rotary position encoding
    for each input in the batch.
    `q` and `kv` should have same length.
    Both `q` and the `k` in `kv` should NOT be position encoded.

    Args:
      q: Shape: `[sum(seqlen[i]), num_heads, head_dim]`. \
        Query projection (`X @ W_q`).
      kv: Batched key-value cache.
      layer_idx: Layer index of the KV cache.

    Returns:
      Shape: `[sum(seqlen[i]), num_heads, head_dim]`. \
      Output of the self-attention.
    """
    tmp = _get_cache_buf("flashinfer_tmp", 64 << 20, q.device)
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _kernels.batch_prefill(
        o,
        q,
        qo_indptr,
        kv.ptrs,
        kv.indptr,
        kv.last_page_offset,
        tmp,
        kv.pool.num_layers,
        layer_idx,
        kv.pool.num_heads,
        base,
        kv.pool.page_len,
    )
    return o


def batch_decode(
    q: torch.Tensor,
    kv: BatchedKvCache,
    base: float,
    layer_idx: int,
) -> torch.Tensor:
    """
    Perform self-attention with rotary position encoding
    for each input in the batch.

    All inputs in the batch should be in the decode stage,
    i.e., each input should only have one token.

    Both `q` and the `k` in `kv` should NOT be position encoded.

    Args:
      q: Shape: `[batch_size, num_heads, head_dim]`. \
        Query projection (`X @ W_q`).
      kv: Batched key-value cache.
      layer_idx: Layer index of the KV cache.

    Returns:
      Shape: `[batch_size, num_heads, head_dim]`. \
      Output of the self-attention.
    """
    tmp = _get_cache_buf("flashinfer_tmp", 64 << 20, q.device)
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _kernels.batch_decode(
        o,
        q,
        kv.ptrs,
        kv.indptr,
        kv.last_page_offset,
        tmp,
        kv.pool.num_layers,
        layer_idx,
        kv.pool.num_heads,
        base,
        kv.pool.page_len,
    )
    return o


def init_kv(
    kv: BatchedKvCache,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlen_indptr: torch.Tensor,
    layer_idx: int,
):
    """
    Copy `k` and `v` to the `kv` cache.
    Pages of each sequence in the batch are already allocated in `kv`.

    Args:
      kv: Batched key-value cache.
      k: Shape: `[sum(seqlen[i]), num_heads, head_dim]`. \
        Key projection. (`X @ W_k`)
      v: Shape: `[sum(seqlen[i]), num_heads, head_dim]`. \
        Value projection. (`X @ W_v`)
      seqlen_indptr: Shape: `[B + 1]`. Indptr of the sequence lengths. \
        `seqlen_indptr[i + 1] == sum(seqlen[:i])`.
      layer_idx: Layer index of the KV cache.
    """
    _kernels.init_kv(
        kv.ptrs,
        kv.indptr,
        kv.last_page_offset,
        k,
        v,
        seqlen_indptr,
        kv.pool.num_layers,
        layer_idx,
        kv.pool.num_heads,
        kv.pool.page_len,
    )


def append_kv(
    kv: BatchedKvCache,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_idx: int,
):
    """
    Append the new token's `k` and `v` to the `kv` cache.
    Page for the new token of each sequence in the batch
    is already allocated in `kv`.

    Args:
      kv: Batched key-value cache.
      k: Shape: `[batch_size, num_heads, head_dim]`. \
        Key projection. (`X @ W_k`)
      v: Shape: `[batch_size, num_heads, head_dim]`. \
        Value projection. (`X @ W_v`)
      layer_idx: Layer index of the KV cache.
    """
    _kernels.append_kv(
        kv.ptrs,
        kv.indptr,
        kv.last_page_offset,
        k,
        v,
        kv.pool.num_layers,
        layer_idx,
        kv.pool.num_heads,
        kv.pool.page_len,
    )
