#pragma once
#include <cuda_runtime.h>

#include <cstdint>

template <typename DType, typename LayoutType>
bool smlm_forward(
    DType *y, DType *x, DType **w, int32_t *s, void *tmp_d,
    int num_problems, int d_in, int d_out, int w_ld, cudaStream_t stream
);

size_t smlm_forward_tmp_size(int num_problems);
