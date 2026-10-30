// Functional SM90 transport for the original planner fields. Scalar chains
// follow low.cu/high.cu at the pinned author revision; this is not a tuned MMA
// backend and does not certify the native SM90 FP32 target.
#include "../score.cuh"

namespace DOSA_NAMESPACE {
namespace sm90_screen_reference {
struct Args {
  const uint8_t *fine, *keys;
  const float *dd, *meta, *raw, *out, *scale, *tau;
  const int *hint, *rho, *starts, *ends;
  int* error;
  float* upper;
  uint8_t* keep;
  int rows, nk;
};

__device__ __forceinline__ float decode(uint8_t value) {
  return float(*reinterpret_cast<cutlass::float_e4m3_t*>(&value));
}
__device__ __forceinline__ float2 add(float2 a, float2 b) {
  return make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
}
__device__ __forceinline__ float2 fma(float2 a, float2 b, float2 c) {
  return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
}

template <int Paid, bool Direct>
__global__ void screen(Args a) {
  constexpr int Width = Paid + 2, MetaStride = Paid <= 16 ? 64 : 96;
  const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= a.rows * a.nk) return;
  const int row = i / a.nk, key = i % a.nk;
  const int start = a.starts[row], end = a.ends[row];
  const bool interval_ok = start >= 0 && start <= end && end <= a.nk;
  const bool visible = !interval_ok || (key >= start && key < end);
  if (!visible) {
    a.upper[i] = -INFINITY;
    a.keep[i] = 0;
    return;
  }
  const float scale = a.scale[key];
  const uint32_t sb = __float_as_uint(scale);
  const int code = a.rho[key];
  const bool admitted = interval_ok && a.hint[row] == 2 && code >= 0 &&
                        code < 127 && sb >= 0x09800000u && sb <= 0x5F800000u &&
                        (sb & 0x007FFFFFu) == 0u && isfinite(a.tau[row]);
  if (!admitted) {
    atomicOr(a.error + row, 1);
    a.upper[i] = INFINITY;
    a.keep[i] = 1;
    return;
  }
  float acc[Width] = {};
#pragma unroll 1
  for (int d = 0; d < 128; ++d) {
    const float k = decode(a.keys[int64_t(key) * 128 + d]);
#pragma unroll
    for (int h = 0; h < Width; ++h)
      acc[h] = __fmaf_rn(decode(a.fine[(int64_t(row) * Width + h) * 128 + d]),
                         k, acc[h]);
  }
  const float* meta = a.meta + row * MetaStride;
  const float lo0 = a.dd[row * Width], lo1 = a.dd[row * Width + 1];
  float2 s0 = make_float2(0, 0), s1 = make_float2(0, 0);
  float core;
  if constexpr (Direct && Paid == 14) {
    // Original h16_abs_core: four RN chains, including the signed DD terms.
#pragma unroll
    for (int h = 0; h < 16; h += 4) {
      const float2 a0 = h == 0 ? make_float2(acc[0], acc[1])
                               : make_float2(fabsf(acc[h]), fabsf(acc[h + 1]));
      const float2 w0 = h == 0 ? make_float2(-lo0, -lo1)
                               : make_float2(__fmul_rn(.5f, meta[h]),
                                             __fmul_rn(.5f, meta[h + 1]));
      const float2 w1 =
          make_float2(__fmul_rn(.5f, meta[h + 2]), __fmul_rn(.5f, meta[h + 3]));
      s0 = fma(a0, w0, s0);
      s1 = fma(make_float2(fabsf(acc[h + 2]), fabsf(acc[h + 3])), w1, s1);
    }
    const float2 pair = add(s0, s1);
    core = __fadd_rn(pair.x, pair.y);
  } else {
#pragma unroll
    for (int h = 0; h < Paid; h += 4) {
      float2 a0 = make_float2(fabsf(acc[h + 2]), fabsf(acc[h + 3]));
      if constexpr (!Direct) a0 = add(make_float2(acc[h + 2], acc[h + 3]), a0);
      s0 = fma(a0, make_float2(meta[h + 2], meta[h + 3]), s0);
      if (h + 2 < Paid) {
        float2 a1 = make_float2(fabsf(acc[h + 4]), fabsf(acc[h + 5]));
        if constexpr (!Direct)
          a1 = add(make_float2(acc[h + 4], acc[h + 5]), a1);
        s1 = fma(a1, make_float2(meta[h + 4], meta[h + 5]), s1);
      }
    }
    const float2 pair = add(s0, s1);
    const float paid = __fadd_rn(pair.x, pair.y);
    const float negative =
        __fadd_rn(0.f, __fmaf_rn(acc[1], lo1, __fmul_rn(acc[0], lo0)));
    if constexpr (Direct)
      core = __fmaf_rn(paid, .5f, -negative);
    else
      core = __fadd_rn(__fmul_rn(paid, .5f), -fmaxf(negative, 0.f));
  }
  const float rho = decode(uint8_t(code)), guard = meta[Width];
  float upper;
  if constexpr (Direct)
    upper = __fmaf_ru(scale, __fmaf_ru(rho, guard, __fadd_ru(core, a.raw[row])),
                      a.out[row]);
  else
    upper = __fadd_rn(__fmul_rn(scale, __fadd_rn(__fadd_rn(core, a.raw[row]),
                                                 __fmul_rn(rho, guard))),
                      a.out[row]);
  if (!isfinite(upper)) {
    atomicOr(a.error + row, 1);
    upper = INFINITY;
  }
  a.upper[i] = upper;
  a.keep[i] = !(upper < a.tau[row]);
}

void launch(int64_t paid, bool direct, std::vector<Tensor> p, Tensor keys,
            Tensor scales, Tensor rho, Tensor tau, Tensor starts, Tensor ends,
            Tensor error, Tensor upper, Tensor keep) {
  STD_TORCH_CHECK(p.size() == 6, "Expected original six-field plan");
  const int64_t rows = p[0].size(0), nk = keys.size(0), width = paid + 2;
  STD_TORCH_CHECK(rows > 0 && nk > 0 && rows * nk <= INT_MAX,
                  "Invalid SM90 reference extents");
  for (const auto& tensor : p)
    STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                        tensor.get_device_index() == keys.get_device_index(),
                    "Invalid plan storage");
  for (const auto& tensor :
       {keys, scales, rho, tau, starts, ends, error, upper, keep})
    STD_TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                        tensor.get_device_index() == keys.get_device_index(),
                    "Invalid reference storage");
  using ST = torch::headeronly::ScalarType;
  STD_TORCH_CHECK(p[0].scalar_type() == ST::Byte &&
                      keys.scalar_type() == ST::Byte &&
                      keep.scalar_type() == ST::Byte,
                  "Expected FP8 byte storage");
  for (const auto& tensor : {p[1], p[2], p[3], p[4], scales, tau, upper})
    STD_TORCH_CHECK(tensor.scalar_type() == ST::Float, "Expected FP32 fields");
  for (const auto& tensor : {p[5], rho, starts, ends, error})
    STD_TORCH_CHECK(tensor.scalar_type() == ST::Int, "Expected I32 fields");
  STD_TORCH_CHECK(
      p[0].dim() == 3 && p[0].size(1) == width && p[0].size(2) == 128 &&
          p[1].numel() == rows * width &&
          p[2].numel() == rows * (paid <= 16 ? 64 : 96) &&
          p[3].numel() == rows && p[4].numel() == rows &&
          p[5].numel() == rows && keys.dim() == 2 && keys.size(1) == 128 &&
          scales.numel() == nk && rho.numel() == nk && tau.numel() == rows &&
          starts.numel() == rows && ends.numel() == rows &&
          error.numel() == rows && upper.dim() == 2 && upper.size(0) == rows &&
          upper.size(1) == nk && keep.sizes() == upper.sizes(),
      "SM90 reference shape mismatch");
  DeviceGuard device(keys.get_device_index());
  Args a{p[0].const_data_ptr<uint8_t>(),
         keys.const_data_ptr<uint8_t>(),
         p[1].const_data_ptr<float>(),
         p[2].const_data_ptr<float>(),
         p[3].const_data_ptr<float>(),
         p[4].const_data_ptr<float>(),
         scales.const_data_ptr<float>(),
         tau.const_data_ptr<float>(),
         p[5].const_data_ptr<int>(),
         rho.const_data_ptr<int>(),
         starts.const_data_ptr<int>(),
         ends.const_data_ptr<int>(),
         error.mutable_data_ptr<int>(),
         upper.mutable_data_ptr<float>(),
         keep.mutable_data_ptr<uint8_t>(),
         int(rows),
         int(nk)};
  const int blocks = int((rows * nk + 127) / 128);
  if (paid == 14 && direct)
    screen<14, true><<<blocks, 128, 0, get_current_cuda_stream()>>>(a);
  else if (paid == 16 && direct)
    screen<16, true><<<blocks, 128, 0, get_current_cuda_stream()>>>(a);
  else if (paid == 20 && !direct && dosa_geometry::Heads == 32)
    screen<20, false><<<blocks, 128, 0, get_current_cuda_stream()>>>(a);
  else if (paid == 28 && !direct && dosa_geometry::Heads == 64)
    screen<28, false><<<blocks, 128, 0, get_current_cuda_stream()>>>(a);
  else
    STD_TORCH_CHECK(false, "Unsupported original certificate family");
  DOSA_CUDA_CHECK(cudaGetLastError());
}
}  // namespace sm90_screen_reference
}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("screen") "(int paid, bool direct, Tensor[] plan, Tensor keys, Tensor scales, Tensor rho, Tensor tau, Tensor starts, Tensor ends, Tensor! error, Tensor! upper, Tensor! keep) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("screen"),
           TORCH_BOX(&DOSA_NAMESPACE::sm90_screen_reference::launch));
}
