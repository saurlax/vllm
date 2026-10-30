// Shared-key native refinement with exact threshold filtering and sparse
// writes.
#include "../common.cuh"
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace DOSA_NAMESPACE::sm90_refine_shared {
namespace dg = deep_gemm;
__device__ __forceinline__ void async_copy(void* dst, const void* src, bool valid) {
  unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
               "r"(address), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void commit_copy() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void wait_copy() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

constexpr int H = 32, Batch = 4, N = H * Batch;
constexpr int Groups = 4, Threads = 128 * Groups;
constexpr int Bytes = N * 128 + 2 * Groups * 8192 + Batch * H * 4 + 2 * Groups * 64 * 8;
__device__ __forceinline__ int swizzle(int row, int d) {
  return row * 128 + (d ^ ((row & 7) * 16));
}

__global__ __launch_bounds__(Threads) void kernel(
    const uint8_t* q, const float* w, const uint8_t* keys, const float* scales,
    const int* union_ids, const int* union_counts, const uint8_t* masks,
    const float* tau, const int* starts, const int* ends, int* ids,
    float* scores, int* counts, int rows, int nk, int mask_stride, int ctas) {
  const int group = blockIdx.x, first_row = group * Batch;
  const int t = threadIdx.x % 128, lane = t % 32, warp = t / 32, wg = threadIdx.x / 128;
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sq = smem;
  uint8_t* sk = sq + N * 128;
  float* weights = reinterpret_cast<float*>(sk + 2 * Groups * 8192);
  float* scale = weights + Batch * H;
  int* key_ids = reinterpret_cast<int*>(scale + 2 * Groups * 64);
  for (int i = threadIdx.x; i < N * 8; i += Threads) {
    const int h = i / 8, row = first_row + h / H, d = i % 8 * 16;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (row < rows)
      value = *reinterpret_cast<const uint4*>(
          q + (int64_t(row) * H + h % H) * 128 + d);
    *reinterpret_cast<uint4*>(sq + swizzle(h, d)) = value;
  }
  for (int i = threadIdx.x; i < N; i += Threads) {
    const int row = first_row + i / H;
    weights[i] = row < rows ? w[row * H + i % H] : 0.f;
  }
  __syncthreads();
  // These weights are invariant for the complete candidate stream. Keep
  // native lane ownership in registers instead of reloading shared memory
  // for every key tile.
  float register_weights[Batch][H / 4];
#pragma unroll
  for (int b = 0; b < Batch; ++b) {
#pragma unroll
    for (int j = 0; j < H / 4; ++j)
      register_weights[b][j] =
          weights[b * H + (j / 2) * 8 + (j & 1) + (lane % 4) * 2];
  }
  const int count = union_counts[group];
  const int begin = int(blockIdx.y) * Groups * 64;
  const auto load = [&](int first, int buffer) {
    for (int i = t; i < 512; i += 128) {
      int at = first + wg * 64 + i / 8;
      int id = at < count ? union_ids[int64_t(group) * nk + at] : -1;
      async_copy(sk + (buffer * Groups + wg) * 8192 + swizzle(i / 8, i % 8 * 16),
                 keys + int64_t(max(id, 0)) * 128 + i % 8 * 16, id >= 0);
      if (i % 8 == 0) {
        key_ids[(buffer * Groups + wg) * 64 + i / 8] = id;
        scale[(buffer * Groups + wg) * 64 + i / 8] = id >= 0 ? scales[id] : 0.f;
      }
    }
    commit_copy();
  };
  if (begin < count) load(begin, 0);
  wait_copy();
  __syncthreads();
  int buffer = 0;
  for (int first = begin; first < count; first += ctas * Groups * 64, buffer ^= 1) {
    if (first + ctas * Groups * 64 < count) load(first + ctas * Groups * 64, buffer ^ 1);
    using M = typename dg::mma::sm90::FP8MMASelector<N>::type;
    float a[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; ++i) dg::ptx::warpgroup_fence_operand(a[i]);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    dg::ptx::warpgroup_arrive();
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      auto x = dg::mma::sm90::make_smem_desc(
          sk + (buffer * Groups + wg) * 8192 + d * 32, dg::mma::sm90::to_swizzle_cute_type<128>(), 0, 1024);
      auto y = dg::mma::sm90::make_smem_desc(
          sq + d * 32, dg::mma::sm90::to_swizzle_cute_type<128>(), 0, 1024);
      M::wgmma(x, y, a, d != 0);
    }
    dg::ptx::warpgroup_commit_batch();
#pragma unroll
    for (int i = 0; i < N / 2; ++i) dg::ptx::warpgroup_fence_operand(a[i]);
    dg::ptx::warpgroup_wait<0>();
#pragma unroll
    for (int b = 0; b < Batch; ++b) {
      const int row = first_row + b;
      const auto product = [&](int j) {
        return __fmul_rn(fmaxf(a[b * 16 + j], 0.f),
                         register_weights[b][(j / 4) * 2 + (j & 1)]);
      };
      const auto fma = [&](int j, float acc) {
        return __fmaf_rn(fmaxf(a[b * 16 + j], 0.f),
                         register_weights[b][(j / 4) * 2 + (j & 1)], acc);
      };
      float sum[4];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        sum[p] = fma(p, product(4 + p));
        sum[p] = fma(8 + p, sum[p]);
        sum[p] = fma(12 + p, sum[p]);
      }
      int k0 = warp * 16 + lane / 4, k1 = k0 + 8;
      float v0 = __fmul_rn(__fadd_rn(sum[0], sum[1]), scale[(buffer * Groups + wg) * 64 + k0]);
      float v1 = __fmul_rn(__fadd_rn(sum[2], sum[3]), scale[(buffer * Groups + wg) * 64 + k1]);
#pragma unroll
      for (int d = 1; d <= 2; d *= 2) {
        v0 = __fadd_rn(v0, __shfl_xor_sync(0xffffffffu, v0, d));
        v1 = __fadd_rn(v1, __shfl_xor_sync(0xffffffffu, v1, d));
      }
      if ((lane % 4) == 0 && row < rows) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          int key = p ? k1 : k0, id = key_ids[(buffer * Groups + wg) * 64 + key];
          float value = p ? v1 : v0;
          bool eligible = id >= starts[row] && id < ends[row];
          if (eligible)
            eligible =
                ((masks[int64_t(row) * mask_stride + id / 8] >> (id % 8)) &
                 1u) != 0;
          bool hit = eligible && !(value < tau[row]);
          unsigned active = __activemask(), bits = __ballot_sync(active, hit);
          int base = 0;
          if (lane == 0 && bits) base = atomicAdd(counts + row, __popc(bits));
          base = __shfl_sync(active, base, 0);
          if (hit) {
            int position = base + __popc(bits & ((1u << lane) - 1u));
            ids[int64_t(row) * nk + position] = id;
            scores[int64_t(row) * nk + position] = value;
          }
        }
      }
    }
    wait_copy();
    __syncthreads();
  }
}

void run(Tensor q, Tensor w, Tensor keys, Tensor scales, Tensor union_ids,
         Tensor union_counts, Tensor masks, Tensor tau, Tensor starts,
         Tensor ends, Tensor ids, Tensor scores, Tensor counts, int64_t ctas) {
  DeviceGuard guard(q.get_device_index());
  STD_TORCH_CHECK(q.size(1) == 32, "Shared native refinement requires H32");
  DOSA_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes));
  kernel<<<dim3((q.size(0) + Batch - 1) / Batch, ctas), Threads, Bytes,
           get_current_cuda_stream()>>>(
      static_cast<const uint8_t*>(q.data_ptr()), w.const_data_ptr<float>(),
      static_cast<const uint8_t*>(keys.data_ptr()),
      scales.const_data_ptr<float>(), union_ids.const_data_ptr<int>(),
      union_counts.const_data_ptr<int>(), masks.const_data_ptr<uint8_t>(),
      tau.const_data_ptr<float>(), starts.const_data_ptr<int>(),
      ends.const_data_ptr<int>(), ids.mutable_data_ptr<int>(),
      scores.mutable_data_ptr<float>(), counts.mutable_data_ptr<int>(),
      q.size(0), keys.size(0), masks.size(1), ctas);
  DOSA_CUDA_CHECK(cudaGetLastError());
}
}  // namespace DOSA_NAMESPACE::sm90_refine_shared

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP(
      "run") "(Tensor q, Tensor weights, Tensor keys, Tensor scales, Tensor "
             "union_ids, Tensor union_counts, Tensor masks, Tensor tau, Tensor "
             "starts, Tensor ends, Tensor! ids, Tensor! scores, Tensor! "
             "counts, int ctas) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("run"), TORCH_BOX(&DOSA_NAMESPACE::sm90_refine_shared::run));
}
