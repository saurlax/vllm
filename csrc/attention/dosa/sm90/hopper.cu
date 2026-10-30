// Hopper FP8 WGMMA transport. The scoring lane map and epilogue follow the
// pinned DeepGEMM sm90_fp8_mqa_logits implementation (MIT, DeepSeek 2025).
#include "../common.cuh"
#include "../geometry.cuh"
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include "bound.cuh"

namespace DOSA_NAMESPACE::sm90_hopper {
constexpr int Heads = dosa_geometry::Heads;
namespace dg = deep_gemm;

// TMA-compatible 128-byte swizzle, implemented with ordinary cooperative loads.
__device__ __forceinline__ int swizzle(int row, int column) {
  return row * 128 + (column ^ ((row & 7) * 16));
}
__device__ __forceinline__ float decode(uint8_t v) {
  return float(*reinterpret_cast<cutlass::float_e4m3_t*>(&v));
}
template <int H>
__device__ __forceinline__ void mma(uint8_t* k, uint8_t* q, float* accum) {
  using M = typename dg::mma::sm90::FP8MMASelector<H>::type;
#pragma unroll
  for (int i = 0; i < M::kNumAccum; ++i)
    dg::ptx::warpgroup_fence_operand(accum[i]);
  // Cooperative shared-memory writes must be visible to the async MMA proxy.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  dg::ptx::warpgroup_arrive();
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    auto a = dg::mma::sm90::make_smem_desc(
        k + d * 32, dg::mma::sm90::to_swizzle_cute_type<128>(), 0, 1024);
    auto b = dg::mma::sm90::make_smem_desc(
        q + d * 32, dg::mma::sm90::to_swizzle_cute_type<128>(), 0, 1024);
    M::wgmma(a, b, accum, d != 0);
  }
  dg::ptx::warpgroup_commit_batch();
#pragma unroll
  for (int i = 0; i < M::kNumAccum; ++i)
    dg::ptx::warpgroup_fence_operand(accum[i]);
  dg::ptx::warpgroup_wait<0>();
}

template <bool Compact>
__global__ __launch_bounds__(128) void score_kernel(
    const uint8_t* q, const uint8_t* keys, const float* w, const float* scales,
    const int* ids, const int* counts, float* scores, int* capture,
    const int* starts, const int* ends, int nk, int cap, int ctas) {
  const int row = blockIdx.x, t = threadIdx.x, lane = t % 32, warp = t / 32;
  __shared__ __align__(1024) uint8_t sq[Heads * 128];
  __shared__ __align__(1024) uint8_t sk[64 * 128];
  __shared__ int key_ids[64];
  __shared__ float scale[64];
  for (int v = t; v < Heads * 8; v += 128) {
    int h = v / 8, d = v % 8 * 16;
    *reinterpret_cast<uint4*>(sq + swizzle(h, d)) =
        *reinterpret_cast<const uint4*>(q + (int64_t(row) * Heads + h) * 128 +
                                        d);
  }
  __syncthreads();
  const int count = Compact ? counts[row] : cap;
  const int begin = Compact ? 2048 : 0;
  for (int first = begin + int(blockIdx.y) * 64; first < count;
       first += ctas * 64) {
    if (t < 64) {
      int slot = first + t;
      int id =
          slot < count ? (Compact ? ids[int64_t(row) * cap + slot] : slot) : -1;
      key_ids[t] = id;
      scale[t] = id >= 0 && id < nk ? scales[id] : 0.f;
    }
    __syncthreads();
    for (int v = t; v < 64 * 8; v += 128) {
      int k = v / 8, d = v % 8 * 16, id = key_ids[k];
      uint4 value = make_uint4(0, 0, 0, 0);
      if (id >= 0 && id < nk)
        value = *reinterpret_cast<const uint4*>(keys + int64_t(id) * 128 + d);
      *reinterpret_cast<uint4*>(sk + swizzle(k, d)) = value;
    }
    __syncthreads();
    float accum[Heads / 2];
    mma<Heads>(sk, sq, accum);
    float weights[Heads / 4];
#pragma unroll
    for (int j = 0; j < Heads / 4; ++j)
      weights[j] = w[row * Heads + (j / 2) * 8 + (j & 1) + (lane % 4) * 2];
    if constexpr (!Compact) {
      // Same sampled sign masks as the author: first 128 physical seed keys.
#pragma unroll
      for (int j = 0; j < Heads / 2; ++j) {
        int k = warp * 16 + lane / 4 + ((j & 2) ? 8 : 0);
        int h = (j / 4) * 8 + (j & 1) + (lane % 4) * 2;
        int id = key_ids[k];
        if (id >= 0 && id < 128 && accum[j] > 0.f) {
          if constexpr (Heads == 32)
            atomicOr(
                reinterpret_cast<unsigned*>(capture + int64_t(row) * 129 + id),
                1u << h);
          else
            atomicOr(reinterpret_cast<unsigned long long*>(capture) +
                         int64_t(row) * 129 + id,
                     1ull << h);
        }
      }
    }
    const auto term = [&](int j) {
      return fmaxf(accum[j], 0.f) * weights[(j / 4) * 2 + (j & 1)];
    };
    float sums[4] = {term(0), term(1), term(2), term(3)};
#pragma unroll
    for (int j = 1; j < Heads / 8; ++j)
#pragma unroll
      for (int p = 0; p < 4; ++p) sums[p] += term(j * 4 + p);
    int k0 = warp * 16 + lane / 4, k1 = k0 + 8;
    float v0 = (sums[0] + sums[1]) * scale[k0],
          v1 = (sums[2] + sums[3]) * scale[k1];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      v0 += __shfl_xor_sync(0xffffffffu, v0, 1 << j);
      v1 += __shfl_xor_sync(0xffffffffu, v1, 1 << j);
    }
    if ((lane % 4) == 0) {
      int id0 = key_ids[k0], id1 = key_ids[k1];
      if (first + k0 < count)
        scores[int64_t(row) * cap + first + k0] =
            id0 >= starts[row] && id0 < ends[row] ? v0 : -INFINITY;
      if (first + k1 < count)
        scores[int64_t(row) * cap + first + k1] =
            id1 >= starts[row] && id1 < ends[row] ? v1 : -INFINITY;
    }
    __syncthreads();
  }
}

template <int Paid, bool Direct, int Batch = 3>
__global__ __launch_bounds__(128) void screen_kernel(
    const uint8_t* fine, const float* dd, const float* meta, const float* raw,
    const float* out, const int* hint, const uint8_t* keys, const float* scales,
    const int* rho, const float* tau, const int* starts, const int* ends,
    int* ids, int* counts, int* errors, float* debug_upper, int nk, int cap,
    int rows, int seed, int ctas, bool debug) {
  constexpr int Width = Paid + 2, Padded = (Width + 7) / 8 * 8;
  constexpr int Stride = Paid <= 16 ? 64 : 96;
  const int first_row = blockIdx.x * Batch, t = threadIdx.x, lane = t % 32,
            warp = t / 32;
  __shared__ __align__(1024) uint8_t sq[Batch * Padded * 128];
  __shared__ __align__(1024) uint8_t sk[64 * 128];
  __shared__ float dots[Batch * Padded * 64];
  __shared__ float metadata[Batch * Stride];
  for (int v = t; v < Batch * Padded * 8; v += 128) {
    int full_h = v / 8, h = full_h % Padded, row = first_row + full_h / Padded,
        d = v % 8 * 16;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (h < Width && row < rows)
      value = *reinterpret_cast<const uint4*>(
          fine + (int64_t(row) * Width + h) * 128 + d);
    *reinterpret_cast<uint4*>(sq + swizzle(full_h, d)) = value;
  }
  for (int i = t; i < Batch * Stride; i += 128) {
    int row = first_row + i / Stride;
    metadata[i] = row < rows ? meta[row * Stride + i % Stride] : 0.f;
  }
  __syncthreads();
  int last = 0;
#pragma unroll
  for (int b = 0; b < Batch; ++b)
    if (first_row + b < rows) last = max(last, min(ends[first_row + b], nk));
  for (int first = seed + int(blockIdx.y) * 64; first < last;
       first += ctas * 64) {
    for (int v = t; v < 64 * 8; v += 128) {
      int key = first + v / 8, d = v % 8 * 16;
      uint4 value = make_uint4(0, 0, 0, 0);
      if (key < last)
        value = *reinterpret_cast<const uint4*>(keys + int64_t(key) * 128 + d);
      *reinterpret_cast<uint4*>(sk + swizzle(v / 8, d)) = value;
    }
    __syncthreads();
    float accum[Batch * Padded / 2];
    mma<Batch * Padded>(sk, sq, accum);
#pragma unroll
    for (int j = 0; j < Batch * Padded / 2; ++j) {
      int k = warp * 16 + lane / 4 + ((j & 2) ? 8 : 0);
      int h = (j / 4) * 8 + (j & 1) + (lane % 4) * 2;
      dots[h * 64 + k] = accum[j];
    }
    __syncthreads();
    if (t < 64) {
#pragma unroll
      for (int b = 0; b < Batch; ++b) {
        const int row = first_row + b;
        const int key = first + t;
        const bool visible =
            row < rows && key < min(ends[row], nk) && key >= starts[row];
        float a[Width];
#pragma unroll
        for (int h = 0; h < Width; ++h) a[h] = dots[(b * Padded + h) * 64 + t];
        float upper = INFINITY;
        if (visible && hint[row] == 2)
          upper = sm90_bound::evaluate<Paid, Direct>(
              a, metadata + b * Stride, dd[row * Width], dd[row * Width + 1],
              raw[row], out[row], scales[key], decode(uint8_t(rho[key])));
        bool live = visible && (!isfinite(upper) || !(upper < tau[row]));
        if (visible && (hint[row] != 2 || !isfinite(upper)))
          atomicOr(errors + row, 1);
        if (debug && key < nk && row < rows)
          debug_upper[int64_t(row) * nk + key] = visible ? upper : -INFINITY;
        unsigned mask = __ballot_sync(0xffffffffu, live);
        int total = __popc(mask), base = 0;
        if (lane == 0 && total) base = atomicAdd(counts + row, total);
        base = __shfl_sync(0xffffffffu, base, 0);
        if (live) {
          int slot = base + __popc(mask & ((1u << lane) - 1u));
          if (slot < cap)
            ids[int64_t(row) * cap + slot] = key;
          else
            atomicOr(errors + row, 2);
        }
      }
    }
    __syncthreads();
  }
}

void storage(const std::vector<Tensor>& tensors, int device) {
  for (const auto& x : tensors)
    STD_TORCH_CHECK(
        x.is_cuda() && x.is_contiguous() && x.get_device_index() == device,
        "Hopper tensors must be contiguous on one CUDA device");
}
void score(Tensor q, Tensor keys, Tensor weights, Tensor scales, Tensor ids,
           Tensor counts, Tensor scores, Tensor capture, Tensor starts,
           Tensor ends, bool compact, int64_t ctas) {
  storage(
      {q, keys, weights, scales, ids, counts, scores, capture, starts, ends},
      q.get_device_index());
  STD_TORCH_CHECK(q.dim() == 3 && q.size(1) == Heads && q.size(2) == 128 &&
                      keys.dim() == 2 && keys.size(1) == 128,
                  "Invalid FP8 Q/K shapes");
  const int rows = q.size(0), nk = keys.size(0), cap = scores.size(1);
  STD_TORCH_CHECK(ctas > 0 && ctas <= 1024 && cap > 0 &&
                      scores.size(0) == rows && rows * int64_t(cap) <= INT_MAX,
                  "Invalid Hopper launch shape");
  STD_TORCH_CHECK(weights.numel() == int64_t(rows) * Heads &&
                      scales.numel() == nk && starts.numel() == rows &&
                      ends.numel() == rows && counts.numel() == rows &&
                      ids.numel() >= int64_t(rows) * cap &&
                      capture.numel() >= rows * 129,
                  "Invalid Hopper metadata shapes");
  STD_TORCH_CHECK(
      q.element_size() == 1 && keys.element_size() == 1 &&
          weights.scalar_type() == torch::headeronly::ScalarType::Float &&
          scores.scalar_type() == torch::headeronly::ScalarType::Float &&
          scales.scalar_type() == torch::headeronly::ScalarType::Float,
      "Invalid Hopper numerical dtypes");
  for (const auto& x : {ids, counts, starts, ends})
    STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Int,
                    "Expected I32 Hopper metadata");
  STD_TORCH_CHECK(capture.element_size() == (Heads == 32 ? 4 : 8),
                  "Invalid capture mask dtype");
  DeviceGuard guard(q.get_device_index());
  dim3 grid(rows, ctas);
#define SCORE_CALL(C)                                                  \
  score_kernel<C><<<grid, 128, 0, get_current_cuda_stream()>>>(        \
      static_cast<const uint8_t*>(q.data_ptr()),                       \
      static_cast<const uint8_t*>(keys.data_ptr()),                    \
      weights.const_data_ptr<float>(), scales.const_data_ptr<float>(), \
      ids.const_data_ptr<int>(), counts.const_data_ptr<int>(),         \
      scores.mutable_data_ptr<float>(),                                \
      reinterpret_cast<int*>(capture.data_ptr()),                      \
      starts.const_data_ptr<int>(), ends.const_data_ptr<int>(), nk, cap, ctas)
  if (compact) {
    SCORE_CALL(true);
  } else {
    STD_TORCH_CHECK(cap <= nk, "Dense capacity exceeds keys");
    SCORE_CALL(false);
  }
#undef SCORE_CALL
  DOSA_CUDA_CHECK(cudaGetLastError());
}
void screen(int64_t paid, std::vector<Tensor> p, Tensor keys, Tensor scales,
            Tensor rho, Tensor tau, Tensor starts, Tensor ends, Tensor ids,
            Tensor counts, Tensor errors, Tensor upper, int64_t seed,
            int64_t ctas, bool debug) {
  STD_TORCH_CHECK(p.size() == 6, "Expected six-field original plan");
  storage(p, keys.get_device_index());
  storage({keys, scales, rho, tau, starts, ends, ids, counts, errors, upper},
          keys.get_device_index());
  const int rows = p[0].size(0), nk = keys.size(0), cap = ids.size(1);
  STD_TORCH_CHECK(seed >= 0 && seed <= nk && seed % 64 == 0 && ctas > 0 &&
                      ctas <= 1024 && cap >= nk && ids.size(0) == rows,
                  "Invalid Hopper screen launch");
  STD_TORCH_CHECK(!debug || upper.numel() == int64_t(rows) * nk,
                  "Invalid debug upper buffer");
  DeviceGuard guard(keys.get_device_index());
  dim3 grid((rows + 2) / 3, ctas);
#define SCREEN_CALL(P, D)                                             \
  screen_kernel<P, D><<<grid, 128, 0, get_current_cuda_stream()>>>(   \
      p[0].const_data_ptr<uint8_t>(), p[1].const_data_ptr<float>(),   \
      p[2].const_data_ptr<float>(), p[3].const_data_ptr<float>(),     \
      p[4].const_data_ptr<float>(), p[5].const_data_ptr<int>(),       \
      static_cast<const uint8_t*>(keys.data_ptr()),                   \
      scales.const_data_ptr<float>(), rho.const_data_ptr<int>(),      \
      tau.const_data_ptr<float>(), starts.const_data_ptr<int>(),      \
      ends.const_data_ptr<int>(), ids.mutable_data_ptr<int>(),        \
      counts.mutable_data_ptr<int>(), errors.mutable_data_ptr<int>(), \
      upper.mutable_data_ptr<float>(), nk, cap, rows, seed, ctas, debug)
  if (paid == 14) {
    SCREEN_CALL(14, true);
  } else if (paid == 16) {
    SCREEN_CALL(16, true);
  } else if (paid == 20 && Heads == 32) {
    SCREEN_CALL(20, false);
  } else if (paid == 28 && Heads == 64) {
    SCREEN_CALL(28, false);
  } else
    STD_TORCH_CHECK(false, "Unsupported Hopper paid-head budget");
#undef SCREEN_CALL
  DOSA_CUDA_CHECK(cudaGetLastError());
}
}  // namespace DOSA_NAMESPACE::sm90_hopper

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP(
      "score") "(Tensor q, Tensor keys, Tensor weights, Tensor scales, Tensor "
               "ids, Tensor counts, Tensor! scores, Tensor! capture, Tensor "
               "starts, Tensor ends, bool compact, int ctas) -> ()");
  ops.def(DOSA_OP("screen") "(int paid, Tensor[] plan, Tensor keys, Tensor scales, Tensor rho, Tensor tau, Tensor starts, Tensor ends, Tensor! ids, Tensor! counts, Tensor! errors, Tensor! upper, int seed, int ctas, bool debug) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("score"), TORCH_BOX(&DOSA_NAMESPACE::sm90_hopper::score));
  ops.impl(DOSA_OP("screen"), TORCH_BOX(&DOSA_NAMESPACE::sm90_hopper::screen));
}
