// Register-resident Hopper screening. Four queries occupy exactly 4*(paid+2)
// WGMMA N columns, matching the author's packed certificate width.
#include "../common.cuh"
#include "../geometry.cuh"
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace DOSA_NAMESPACE::sm90_screen_fast {
namespace dg = deep_gemm;
constexpr int Heads = dosa_geometry::Heads;
struct QueryFields {
  float dd[2], raw, out, tau;
  int hint, start, end;
};

__device__ __forceinline__ int swizzle(int row, int d) {
  return row * 128 + (d ^ ((row & 7) * 16));
}
__device__ __forceinline__ float decode(uint8_t v) {
  return float(*reinterpret_cast<cutlass::float_e4m3_t*>(&v));
}
__device__ __forceinline__ void copy16(uint8_t* dst, const uint8_t* src,
                                       bool valid) {
  const unsigned address = unsigned(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(address),
               "l"(src), "r"(valid ? 16 : 0)
               : "memory");
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void mma(uint8_t* k, uint8_t* q, float* a) {
  using M = typename dg::mma::sm90::FP8MMASelector<N>::type;
#pragma unroll
  for (int i = 0; i < N / 2; ++i) dg::ptx::warpgroup_fence_operand(a[i]);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  dg::ptx::warpgroup_arrive();
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    auto x = dg::mma::sm90::make_smem_desc(
        k + d * 32, dg::mma::sm90::to_swizzle_cute_type<128>(), 0, 1024);
    auto y = dg::mma::sm90::make_smem_desc(
        q + d * 32, dg::mma::sm90::to_swizzle_cute_type<128>(), 0, 1024);
    M::wgmma(x, y, a, d != 0);
  }
  dg::ptx::warpgroup_commit_batch();
#pragma unroll
  for (int i = 0; i < N / 2; ++i) dg::ptx::warpgroup_fence_operand(a[i]);
  dg::ptx::warpgroup_wait<0>();
}

template <int Paid, bool Direct, int B>
__device__ __forceinline__ void emit(const float* a, const float* metadata,
                                     const QueryFields* query,
                                     const float* scale, const float* rho,
                                     uint8_t* masks, int* errors, float* upper,
                                     int first_row, int first_key, int rows,
                                     int nk, int mask_stride, bool debug) {
  constexpr int Width = Paid + 2, Stride = Paid <= 16 ? 64 : 96;
  const int t = threadIdx.x, lane = t % 32, owner = lane % 4;
  const int row = first_row + B;
  float s0 = 0.f, s1 = 0.f;
  // Each four-lane group owns eight heads for the same two keys. Use upward
  // rounding for arbitrary head signs, then upward-rounded shuffle sums.
#pragma unroll
  for (int g = (B * Width) / 8; g <= ((B + 1) * Width - 1) / 8; ++g) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      int h = g * 8 + owner * 2 + p - B * Width;
      if (h >= 0 && h < Width) {
        float w = 0.f, x0 = a[g * 4 + p], x1 = a[g * 4 + p + 2];
        if constexpr (Direct) {
          if (h < 2)
            w = -query[B].dd[h];
          else {
            w = __fmul_rn(.5f, metadata[B * Stride + h]);
            x0 = fabsf(x0);
            x1 = fabsf(x1);
          }
        } else {
          if (h >= 2) {
            w = metadata[B * Stride + h];
            x0 = fmaxf(x0, 0.f);
            x1 = fmaxf(x1, 0.f);
          } else {
            x0 = 0.f;
            x1 = 0.f;
          }
        }
        s0 = __fmaf_ru(x0, w, s0);
        s1 = __fmaf_ru(x1, w, s1);
      }
    }
  }
#pragma unroll
  for (int offset = 1; offset <= 2; offset *= 2) {
    s0 = __fadd_ru(s0, __shfl_xor_sync(0xffffffffu, s0, offset));
    s1 = __fadd_ru(s1, __shfl_xor_sync(0xffffffffu, s1, offset));
  }
  constexpr int Owner = ((B * Width) % 8) / 2, Base = (B * Width / 8) * 4;
  // The owner of the DD pair rotates with each tightly packed query.
  if (row < rows && owner == Owner) {
    float core[2] = {s0, s1};
    if constexpr (!Direct) {
      core[0] = __fadd_ru(core[0],
                          -fmaxf(__fmaf_rd(a[Base + 1], query[B].dd[1],
                                           __fmul_rd(a[Base], query[B].dd[0])),
                                 0.f));
      core[1] = __fadd_ru(
          core[1], -fmaxf(__fmaf_rd(a[Base + 3], query[B].dd[1],
                                    __fmul_rd(a[Base + 2], query[B].dd[0])),
                          0.f));
    }
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      int local_key = t / 32 * 16 + lane / 4 + p * 8;
      int key = first_key + local_key;
      bool visible = key < nk && key >= query[B].start && key < query[B].end;
      float bound = INFINITY;
      if (visible && query[B].hint == 2 && isfinite(rho[local_key])) {
        bound =
            __fmaf_ru(scale[local_key],
                      __fmaf_ru(rho[local_key], metadata[B * Stride + Width],
                                __fadd_ru(core[p], query[B].raw)),
                      query[B].out);
      }
      const bool uncertain = !isfinite(bound);
      if (uncertain) bound = INFINITY;
      bool live = visible && !(bound < query[B].tau);
      if (visible && query[B].hint == 2 && isfinite(rho[local_key]) &&
          uncertain)
        atomicOr(errors + row, 1);
      if (debug && key < nk)
        upper[int64_t(row) * nk + key] = visible ? bound : -INFINITY;
      // Eight owners per warp. They execute the ballot under their active
      // mask; every owner writes at most one unique key per half-tile.
      unsigned active = __activemask();
      unsigned mask = __ballot_sync(active, live);
      if (lane == Owner) {
        unsigned packed = 0;
#pragma unroll
        for (int bit = 0; bit < 8; ++bit)
          packed |= ((mask >> (Owner + 4 * bit)) & 1u) << bit;
        int base_key = first_key + t / 32 * 16 + p * 8;
        if (base_key < nk)
          masks[int64_t(row) * mask_stride + base_key / 8] = uint8_t(packed);
      }
    }
  }
}

template <int Paid, bool Direct>
__global__ __launch_bounds__(128) void kernel(
    const uint8_t* fine, const float* dd, const float* meta, const float* raw,
    const float* out, const int* hint, const uint8_t* keys, const float* scales,
    const int* rho, const float* tau, const int* starts, const int* ends,
    uint8_t* masks, int* errors, float* upper, int nk, int mask_stride,
    int rows, int seed, int ctas, bool debug) {
  constexpr int Width = Paid + 2, N = 4 * Width, Stride = Paid <= 16 ? 64 : 96;
  const int first_row = blockIdx.x * 4, t = threadIdx.x;
  bool any_certified = false;
#pragma unroll
  for (int b = 0; b < 4; ++b)
    if (first_row + b < rows && hint[first_row + b] == 2) any_certified = true;
  if (!any_certified) {
    // Exact recovery: uncertified query groups keep all visible nonseed keys.
    // Skip the matrix screen instead of paying for a bound that cannot prune.
    for (int byte = seed / 8 + int(blockIdx.y) * 128 + t; byte < mask_stride;
         byte += ctas * 128) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        int row = first_row + b;
        if (row < rows) {
          int lo = min(8, max(0, starts[row] - byte * 8));
          int hi = min(8, max(0, min(ends[row], nk) - byte * 8));
          masks[int64_t(row) * mask_stride + byte] =
              uint8_t(((1u << hi) - 1u) & ~((1u << lo) - 1u));
        }
      }
    }
    if (t < 4 && first_row + t < rows) atomicOr(errors + first_row + t, 1);
    return;
  }
  __shared__ __align__(1024) uint8_t sq[N * 128];
  __shared__ __align__(1024) uint8_t sk[2 * 64 * 128];
  __shared__ float metadata[4 * Stride];
  __shared__ QueryFields query[4];
  __shared__ float key_scale[128], key_rho[128];
  if (t < 4) {
    int row = first_row + t;
    QueryFields value{};
    if (row < rows) {
      value = {{dd[row * Width], dd[row * Width + 1]},
               raw[row],
               out[row],
               tau[row],
               hint[row],
               starts[row],
               ends[row]};
      if (value.hint != 2) atomicOr(errors + row, 1);
    }
    query[t] = value;
  }
  for (int v = t; v < N * 8; v += 128) {
    int h = v / 8, row = first_row + h / Width, d = v % 8 * 16;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (row < rows)
      value = *reinterpret_cast<const uint4*>(
          fine + (int64_t(row) * Width + h % Width) * 128 + d);
    *reinterpret_cast<uint4*>(sq + swizzle(h, d)) = value;
  }
  for (int i = t; i < 4 * Stride; i += 128) {
    int row = first_row + i / Stride;
    metadata[i] = row < rows ? meta[row * Stride + i % Stride] : 0.f;
  }
  int last = 0;
#pragma unroll
  for (int b = 0; b < 4; ++b)
    if (first_row + b < rows) last = max(last, min(nk, ends[first_row + b]));
  int first = seed + int(blockIdx.y) * 64;
  if (first >= last) return;
  for (int v = t; v < 512; v += 128) {
    int key = first + v / 8, d = v % 8 * 16;
    copy16(sk + swizzle(v / 8, d), keys + int64_t(min(key, nk - 1)) * 128 + d,
           key < last);
  }
  if (t < 64) {
    int key = first + t;
    key_scale[t] = key < last ? scales[key] : 1.f;
    key_rho[t] = key < last ? decode(uint8_t(rho[key])) : 0.f;
  }
  commit();
  wait();
  __syncthreads();
  int stage = 0;
  for (; first < last; first += ctas * 64, stage ^= 1) {
    int next = first + ctas * 64;
    if (next < last) {
      for (int v = t; v < 512; v += 128) {
        int key = next + v / 8, d = v % 8 * 16;
        copy16(sk + (stage ^ 1) * 8192 + swizzle(v / 8, d),
               keys + int64_t(min(key, nk - 1)) * 128 + d, key < last);
      }
      if (t < 64) {
        int key = next + t, slot = (stage ^ 1) * 64 + t;
        key_scale[slot] = key < last ? scales[key] : 1.f;
        key_rho[slot] = key < last ? decode(uint8_t(rho[key])) : 0.f;
      }
    }
    commit();
    float a[N / 2];
    mma<N>(sk + stage * 8192, sq, a);
#define EMIT(B)                                                                \
  emit<Paid, Direct, B>(a, metadata, query, key_scale + stage * 64,            \
                        key_rho + stage * 64, masks, errors, upper, first_row, \
                        first, rows, nk, mask_stride, debug)
    EMIT(0);
    EMIT(1);
    EMIT(2);
    EMIT(3);
#undef EMIT
    wait();
    __syncthreads();
  }
}

void launch(int64_t paid, std::vector<Tensor> p, Tensor keys, Tensor scales,
            Tensor rho, Tensor tau, Tensor starts, Tensor ends, Tensor masks,
            Tensor errors, Tensor upper, int64_t seed, int64_t ctas,
            bool debug) {
  STD_TORCH_CHECK(
      p.size() == 6 && ctas > 0 && ctas <= 1024 && seed >= 0 && seed % 64 == 0,
      "Invalid Hopper screen metadata");
  const int rows = p[0].size(0), nk = keys.size(0), mask_stride = masks.size(1);
  STD_TORCH_CHECK(masks.size(0) == rows && int64_t(mask_stride) * 8 >= nk &&
                      seed <= nk &&
                      (!debug || upper.numel() == int64_t(rows) * nk),
                  "Invalid Hopper screen capacity");
  for (const auto& x : p)
    STD_TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                        x.get_device_index() == keys.get_device_index(),
                    "Invalid Hopper plan storage");
  for (const auto& x :
       {keys, scales, rho, tau, starts, ends, masks, errors, upper})
    STD_TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                        x.get_device_index() == keys.get_device_index(),
                    "Invalid Hopper screen storage");
  DeviceGuard guard(keys.get_device_index());
  dim3 grid((rows + 3) / 4, ctas);
#define LAUNCH(P, D)                                                       \
  kernel<P, D><<<grid, 128, 0, get_current_cuda_stream()>>>(               \
      p[0].const_data_ptr<uint8_t>(), p[1].const_data_ptr<float>(),        \
      p[2].const_data_ptr<float>(), p[3].const_data_ptr<float>(),          \
      p[4].const_data_ptr<float>(), p[5].const_data_ptr<int>(),            \
      static_cast<const uint8_t*>(keys.data_ptr()),                        \
      scales.const_data_ptr<float>(), rho.const_data_ptr<int>(),           \
      tau.const_data_ptr<float>(), starts.const_data_ptr<int>(),           \
      ends.const_data_ptr<int>(), masks.mutable_data_ptr<uint8_t>(),       \
      errors.mutable_data_ptr<int>(), upper.mutable_data_ptr<float>(), nk, \
      mask_stride, rows, seed, ctas, debug)
  if (paid == 14) {
    LAUNCH(14, true);
  } else if (paid == 16) {
    LAUNCH(16, true);
  } else if (paid == 20 && Heads == 32) {
    LAUNCH(20, false);
  } else if (paid == 28 && Heads == 64) {
    LAUNCH(28, false);
  } else
    STD_TORCH_CHECK(false, "Unsupported Hopper certificate budget");
#undef LAUNCH
  DOSA_CUDA_CHECK(cudaGetLastError());
}
}  // namespace DOSA_NAMESPACE::sm90_screen_fast

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("screen") "(int paid, Tensor[] plan, Tensor keys, Tensor scales, Tensor rho, Tensor tau, Tensor starts, Tensor ends, Tensor! masks, Tensor! errors, Tensor! upper, int seed, int ctas, bool debug) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("screen"),
           TORCH_BOX(&DOSA_NAMESPACE::sm90_screen_fast::launch));
}
