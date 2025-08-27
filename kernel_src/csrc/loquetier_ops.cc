#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

#include "cutlass/layout/matrix.h"
#include "flashinfer_adapter/flashinfer_config.h"
#include "rms_norm/rms_norm.h"
#include "smlm_forward/smlm_forward.h"

namespace {

//====== utils ======

inline void check_shape(const torch::Tensor& a, const torch::Tensor& b,
                        const char* a_name, const char* b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) \
  TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) \
  TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

//====== dispatch pytorch dtype ======

#define _DISPATCH_SWITCH(cond, ...) \
  [&]() -> bool {                   \
    switch (cond) {                 \
      __VA_ARGS__                   \
      default:                      \
        return false;               \
    }                               \
  }()

#define _DISPATCH_DTYPE_CASE(enum_type, c_type_, ...) \
  case enum_type: {                                   \
    using c_type = c_type_;                           \
    return __VA_ARGS__();                             \
  }

#define _DISPATCH_DTYPE_CASES(...)                                 \
  _DISPATCH_DTYPE_CASE(at::ScalarType::Half, nv_half, __VA_ARGS__) \
  _DISPATCH_DTYPE_CASE(at::ScalarType::BFloat16, nv_bfloat16, __VA_ARGS__)

#define DISPATCH_TORCH_DTYPE(scalar_type, ...) \
  _DISPATCH_SWITCH(scalar_type, _DISPATCH_DTYPE_CASES(__VA_ARGS__))

//====== dispatch layout ======

#define DISPATCH_LAYOUT(is_row_major, ...)      \
  if (is_row_major) {                            \
    using layout = cutlass::layout::RowMajor;    \
    __VA_ARGS__                                  \
  }                                              \
  else {                                         \
    using layout = cutlass::layout::ColumnMajor; \
    __VA_ARGS__                                  \
  }

//====== flashinfer ======

void batch_prefill(torch::Tensor o, torch::Tensor q, torch::Tensor qo_indptr,
                   torch::Tensor kv_ptrs, torch::Tensor kv_indptr,
                   torch::Tensor last_page_offset, torch::Tensor tmpbuf,
                   int num_layers, int layer_idx, int num_kv_heads,
                   float base, int page_size) {
  CHECK_INPUT(o);
  CHECK_INPUT(q);
  CHECK_INPUT(qo_indptr);
  CHECK_INPUT(kv_ptrs);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(last_page_offset);

  CHECK_DIM(3, o);                 // [qo_indptr[-1], N, D]
  CHECK_DIM(3, q);                 // [qo_indptr[-1], N, D]
  CHECK_DIM(1, qo_indptr);         // [B+1]
  CHECK_DIM(1, kv_ptrs);           // [kv_indptr[-1]] ptr to a  [L, 2, N, P, D]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, last_page_offset);  // [B]

  int batch_size = static_cast<int>(last_page_offset.size(0));
  int num_qo_heads = static_cast<int>(o.size(1));
  int head_dim = static_cast<int>(o.size(2));
  int group_size = num_qo_heads / num_kv_heads;
  CHECK_SHAPE(o, q);
  CHECK_EQ(num_qo_heads, group_size * num_kv_heads);
  CHECK_EQ(qo_indptr.size(0), batch_size + 1);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_GE(tmpbuf.nbytes(), sizeof(int32_t) * (4 * batch_size + 1));
  CHECK_GE(tmpbuf.nbytes(), 64 << 20);

  auto device = kv_indptr.device();
  const c10::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index());

  bool ok = DISPATCH_TORCH_DTYPE(q.scalar_type(), [&] {
    return FlashInferBatchPrefillKernel(
        static_cast<c_type*>(o.data_ptr()), static_cast<c_type*>(q.data_ptr()),
        qo_indptr.data_ptr<int32_t>(),
        reinterpret_cast<c_type**>(kv_ptrs.data_ptr<int64_t>()),
        kv_indptr.data_ptr<int32_t>(), last_page_offset.data_ptr<int32_t>(),
        tmpbuf.data_ptr(), head_dim, num_layers, layer_idx, group_size,
        num_kv_heads, base, page_size, batch_size, stream);
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", q.scalar_type(),
              " page_size=", page_size, " group_size=", group_size,
              " head_dim=", head_dim);
}

void batch_decode(torch::Tensor o, torch::Tensor q, torch::Tensor kv_ptrs,
                  torch::Tensor kv_indptr, torch::Tensor last_page_offset,
                  torch::Tensor tmpbuf, int num_layers, int layer_idx,
                  int num_kv_heads, float base, int page_size) {
  CHECK_INPUT(o);
  CHECK_INPUT(q);
  CHECK_INPUT(kv_ptrs);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(last_page_offset);

  CHECK_DIM(3, o);                 // [B, N, D]
  CHECK_DIM(3, q);                 // [B, N, D]
  CHECK_DIM(1, kv_ptrs);           // [kv_indptr[-1]] ptr to a  [L, 2, N, P, D]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, last_page_offset);  // [B]

  int batch_size = static_cast<int>(o.size(0));
  int num_qo_heads = static_cast<int>(o.size(1));
  int head_dim = static_cast<int>(o.size(2));
  int group_size = num_qo_heads / num_kv_heads;
  CHECK_SHAPE(o, q);
  CHECK_EQ(num_qo_heads, group_size * num_kv_heads);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_GE(tmpbuf.nbytes(), sizeof(int32_t) * (4 * batch_size + 1));
  CHECK_GE(tmpbuf.nbytes(), 64 << 20);

  auto device = tmpbuf.device();
  const c10::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index());

  bool ok = DISPATCH_TORCH_DTYPE(q.scalar_type(), [&] {
    return FlashInferBatchDecodeKernel(
        static_cast<c_type*>(o.data_ptr()), static_cast<c_type*>(q.data_ptr()),
        reinterpret_cast<c_type**>(kv_ptrs.data_ptr<int64_t>()),
        kv_indptr.data_ptr<int32_t>(), last_page_offset.data_ptr<int32_t>(),
        tmpbuf.data_ptr(), head_dim, num_layers, layer_idx, group_size,
        num_kv_heads, base, page_size, batch_size, stream);
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", q.scalar_type(),
              " page_size=", page_size, " group_size=", group_size,
              " head_dim=", head_dim);
}

void init_kv(torch::Tensor kv_ptrs, torch::Tensor kv_indptr,
             torch::Tensor last_page_offset, torch::Tensor k, torch::Tensor v,
             torch::Tensor seqlen_indptr, int num_layers, int layer_idx,
             int num_kv_heads, int page_size) {
  CHECK_INPUT(kv_ptrs);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(seqlen_indptr);

  CHECK_DIM(1, kv_ptrs);           // [kv_indptr[-1]] ptr to a  [L, 2, N, P, D]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, v);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(1, seqlen_indptr);     // [B+1]

  int head_dim = static_cast<int>(k.size(2));
  int batch_size = static_cast<int>(last_page_offset.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(seqlen_indptr.size(0), batch_size + 1);
  CHECK_SHAPE(k, v);
  auto device = kv_indptr.device();
  const c10::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index());

#define CASE(dim, _)                                                           \
  case dim:                                                                    \
    return FlashInferInitKvKernel<dim, c_type>(                                \
        reinterpret_cast<c_type**>(kv_ptrs.data_ptr<int64_t>()),               \
        kv_indptr.data_ptr<int32_t>(), last_page_offset.data_ptr<int32_t>(),   \
        static_cast<c_type*>(k.data_ptr()),                                    \
        static_cast<c_type*>(v.data_ptr()), seqlen_indptr.data_ptr<int32_t>(), \
        num_layers, layer_idx, num_kv_heads, page_size, batch_size, stream);

  bool ok = DISPATCH_TORCH_DTYPE(k.scalar_type(), [&] {
    switch (head_dim) {
      FOR_FlashInferBatchDecode_D(CASE);
      default:
        return false;
    }
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", k.scalar_type(),
              " head_dim=", head_dim);
#undef CASE
}

void append_kv(torch::Tensor kv_ptrs, torch::Tensor kv_indptr,
               torch::Tensor last_page_offset, torch::Tensor k, torch::Tensor v,
               int num_layers, int layer_idx, int num_kv_heads, int page_size) {
  CHECK_INPUT(kv_ptrs);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  CHECK_DIM(1, kv_ptrs);           // [kv_indptr[-1]] ptr to a  [L, 2, N, P, D]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [B, N, D]
  CHECK_DIM(3, v);                 // [B, N, D]

  int head_dim = static_cast<int>(k.size(2));
  int batch_size = static_cast<int>(k.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_SHAPE(k, v);
  auto device = kv_indptr.device();
  const c10::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index());

#define CASE(dim, _)                                                         \
  case dim:                                                                  \
    return FlashInferAppendKvKernel<dim, c_type>(                            \
        reinterpret_cast<c_type**>(kv_ptrs.data_ptr<int64_t>()),             \
        kv_indptr.data_ptr<int32_t>(), last_page_offset.data_ptr<int32_t>(), \
        static_cast<c_type*>(k.data_ptr()),                                  \
        static_cast<c_type*>(v.data_ptr()), num_layers, layer_idx,           \
        num_kv_heads, page_size, batch_size, stream);

  bool ok = DISPATCH_TORCH_DTYPE(k.scalar_type(), [&] {
    switch (head_dim) {
      FOR_FlashInferBatchDecode_D(CASE);
      default:
        return false;
    }
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", k.scalar_type(),
              " head_dim=", head_dim);
#undef CASE
}

//====== smlm forward ======

void dispatch_smlm_forward_cutlass(
  torch::Tensor y, torch::Tensor x,
  torch::Tensor w_ptr, torch::Tensor s,
  torch::Tensor tmp, bool is_row_major
) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w_ptr);
  CHECK_INPUT(s);
  CHECK_INPUT(tmp);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(1, w_ptr);
  CHECK_DIM(1, s);
  CHECK_DIM(1, tmp);

  int num_problems = s.size(0) - 1;
  int d_in = x.size(1);
  int d_out = y.size(1);
  int w_ld;
  if (is_row_major) {
    w_ld = d_out;
  }
  else {
    w_ld = d_in;
  }
  CHECK_EQ(tmp.size(0), static_cast<int64_t>(smlm_forward_tmp_size(num_problems)));
  auto device = tmp.device();
  const c10::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index());
  bool ok;
  DISPATCH_LAYOUT(is_row_major,
    ok = DISPATCH_TORCH_DTYPE(x.scalar_type(), [&] {
      return smlm_forward<c_type, layout>(
        (c_type*)y.data_ptr(), (c_type*)x.data_ptr(), (c_type**)w_ptr.data_ptr(),
        s.data_ptr<int32_t>(), tmp.data_ptr<uint8_t>(), num_problems,
        d_in, d_out, w_ld, stream
      );
    });
  );
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", x.scalar_type());
}

//====== smlm backward ======
// TODO

//====== rms_norm ======

void dispatch_rms_norm(torch::Tensor output, torch::Tensor rms_output, torch::Tensor input,
                       torch::Tensor weight, float epsilon) {
  CHECK_INPUT(output);
  CHECK_INPUT(rms_output);
  CHECK_INPUT(input);
  CHECK_INPUT(weight);

  CHECK_DIM(2, input);
  CHECK_DIM(2, rms_output);
  CHECK_DIM(1, weight);
  CHECK_SHAPE(output, input);
  CHECK_EQ(input.size(0), rms_output.size(0))
  CHECK_EQ(1, rms_output.size(1))
  CHECK_EQ(input.size(input.dim() - 1), weight.size(0));
  CHECK_EQ(input.scalar_type(), weight.scalar_type());
  CHECK_EQ(input.scalar_type(), output.scalar_type());

  int rows = input.size(0);
  int columns = input.size(1);
  auto device = input.device();
  const c10::cuda::OptionalCUDAGuard device_guard(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index());

  bool ok = DISPATCH_TORCH_DTYPE(input.scalar_type(), [&] {
    return rms_norm<c_type>(static_cast<c_type*>(output.data_ptr()),
                            static_cast<float*>(rms_output.data_ptr()),
                            static_cast<c_type*>(input.data_ptr()),
                            static_cast<c_type*>(weight.data_ptr()), rows,
                            columns, epsilon, stream);
  });

  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", input.scalar_type(),
              " columns=", columns);
}

}  // namespace

//====== pybind ======

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_prefill", &batch_prefill, "");
  m.def("batch_decode", &batch_decode, "");
  m.def("init_kv", &init_kv, "");
  m.def("append_kv", &append_kv, "");

  m.def("smlm_forward_cutlass", &dispatch_smlm_forward_cutlass, "");
  m.def("smlm_forward_cutlass_tmp_size", &smlm_forward_tmp_size, "");
  m.def("rms_norm", &dispatch_rms_norm, "");
}
