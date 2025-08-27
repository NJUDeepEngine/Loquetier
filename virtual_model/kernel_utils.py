from collections.abc import Sequence
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.autograd.function import Function, FunctionCtx

from punica.ops import _kernels as kernel


class BatchLenInfo:
    def __init__(
        self,
        trains: List[Tuple[int, ...]],
        prefills: List[int],
        decode: int,
        evals: List[Tuple[int, ...]],
        indptr_device: torch.device,
        indptr_dtype: torch.dtype = torch.int32,
    ):
        self._train_batches = trains
        tlen = [train[0] * train[1] for train in trains]
        tmp = [0] + tlen

        if len(tlen) > 0:
            cumsum = np.cumsum(tmp)
            self._train_partition = cumsum
            self._tlen = cumsum[-1]
        else:
            self._train_partition = []
            self._tlen = 0
        
        self._eval_batches = evals
        if len(evals) > 0:
            elen = [eval[0] * eval[1] for eval in evals]
            cumsum = np.cumsum([self._tlen] + elen)
            self._eval_partition = cumsum
        else:
            self._eval_partition = []
        
        tmp = [0] + prefills
        self._prefills = tmp[1:]
        self._decode = decode

        if len(prefills) > 0:
            cumsum = np.cumsum(tmp)
            self._indptr = torch.tensor(
                cumsum, dtype=indptr_dtype, device=indptr_device
            )
            self._doff = cumsum[-1] + self._tlen
        else:
            self._indptr = None
            self._doff = self._tlen
    
    @property
    def train_batches(self) -> List[Tuple[int, ...]]:
        """Batches and length of train requests."""
        return self._train_batches
    
    @property
    def train_partition(self) -> Sequence[int]:
        """Partition of train batches."""
        return self._train_partition
    
    @property
    def eval_batches(self) -> List[Tuple[int, ...]]:
        """Batches and length of evaluate requests."""
        return self._eval_batches
    
    @property
    def eval_partition(self) -> Sequence[int]:
        """Partition of evaluate batches."""
        return self._eval_partition

    @property
    def prefills(self) -> List[int]:
        """Length of each prefill request."""
        return self._prefills

    @property
    def decode(self) -> int:
        """Number of decode requests."""
        return self._decode
    
    @property
    def tlen(self) -> int:
        """Total length of trains."""
        return self._tlen

    @property
    def doff(self) -> int:
        """Index of the first decode request. Equivalently, total length of prefills."""
        return self._doff

    @property
    def indptr(self) -> Optional[torch.Tensor]:
        """`indptr[i] := sum(prefills[:i])`. None if no prefill."""
        return self._indptr


class add_lora(Function):
    @staticmethod
    def forward(
        y_shape: torch.Size,
        x: torch.Tensor,
        wa_ptr: torch.Tensor,
        wb_ptr: torch.Tensor,
        s: torch.Tensor,
        lora_rank: int,
        train_len: int,
        train_partition: Sequence[int],
        *args
    ) -> torch.Tensor:
        y = torch.zeros(y_shape, dtype=x.dtype, device=x.device)
        tmp_size = kernel.sgmv_cutlass_tmp_size(wa_ptr.size(0))
        tmp = torch.empty((tmp_size,), dtype=torch.uint8, device=x.device)
        v = torch.zeros((x.size(0), lora_rank), dtype=x.dtype, device=x.device)
        kernel.sgmv_cutlass(v, x, wa_ptr, s, tmp)
        kernel.sgmv_cutlass(y, v, wb_ptr, s, tmp)
        return y
    
    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: Tuple, output: torch.Tensor):
        _, x, _, _, _, _, tl, tp, *weight_list = inputs
        ctx.tp = tp
        ctx.save_for_backward(x[:tl, ...], *weight_list)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):
        x, *weight_list = ctx.saved_tensors
        tp = ctx.tp
        if len(weight_list) == 0:
            return None, None, None, None, None, None, None, None
        
        assert len(weight_list) % 2 == 0
        grad_x = torch.zeros((grad_output.size(0), x.size(1)), dtype=grad_output.dtype, device=grad_output.device)
        grad_ws = []

        for wa, wb, pi in zip(weight_list[::2], weight_list[1::2], range(len(tp) - 1)):
            start, end = tp[pi], tp[pi + 1]
            # Compute and reuse grad_o @ wb.T to reduce computation cost.
            mid = grad_output[start:end, ...] @ wb.T
            grad_x[start:end, ...] = mid @ wa.T
            grad_ws.append(x[start:end].T @ mid)
            grad_ws.append(wa.T @ x[start:end].T @ grad_output[start:end, ...])

        return None, grad_x, None, None, None, None, None, None, *grad_ws

class rms_norm(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float = 1e-6,
    ):
        m = torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device)
        o = torch.empty_like(x)
        kernel.rms_norm(o, m, x, w, eps)
        ctx.save_for_backward(x, w, m)
        return o
    
    # @staticmethod
    # def setup_context(ctx: FunctionCtx, inputs: Tuple, output: torch.Tensor):
    #     x, w, eps = inputs
    #     ctx.save_for_backward(x, w)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):
        x, w, rms_1 = ctx.saved_tensors
        d_weight = (x * rms_1 * grad_output).sum(dim=0)

        d_x_hat = grad_output * w
        dot_p = (d_x_hat * x).sum(dim=1, keepdim=True)
        dx = (d_x_hat - (x * dot_p * rms_1 ** 2) / x.size(1)) * rms_1
        return dx, d_weight, None