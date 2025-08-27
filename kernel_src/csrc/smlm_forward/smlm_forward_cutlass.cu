#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cutlass/layout/matrix.h"
#include "smlm_forward_cutlass.cuh"

template bool smlm_forward<nv_half, cutlass::layout::RowMajor>(
    nv_half *y, nv_half *x, nv_half **w, int32_t *s, void *tmp_d,
    int num_problems, int d_in, int d_out, int w_ld, cudaStream_t stream
);

template bool smlm_forward<nv_half, cutlass::layout::ColumnMajor>(
    nv_half *y, nv_half *x, nv_half **w, int32_t *s, void *tmp_d,
    int num_problems, int d_in, int d_out, int w_ld, cudaStream_t stream
);

template bool smlm_forward<nv_bfloat16, cutlass::layout::RowMajor>(
    nv_bfloat16 *y, nv_bfloat16 *x, nv_bfloat16 **w, int32_t *s, void *tmp_d,
    int num_problems, int d_in, int w_ld, int d_out, cudaStream_t stream
);

template bool smlm_forward<nv_bfloat16, cutlass::layout::ColumnMajor>(
    nv_bfloat16 *y, nv_bfloat16 *x, nv_bfloat16 **w, int32_t *s, void *tmp_d,
    int num_problems, int d_in, int w_ld, int d_out, cudaStream_t stream
);
