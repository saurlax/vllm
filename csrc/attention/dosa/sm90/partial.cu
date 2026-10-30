// Native-compatible partial-head upper bounds. Only negative contributions
// are omitted; every retained raw dot keeps native order. The FP32 epilogue
// uses a conservative directed-rounding upper bound in registers.
#include "../common.cuh"
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include "../geometry.cuh"
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace DOSA_NAMESPACE::sm90_partial {
namespace dg = deep_gemm;
constexpr int H = dosa_geometry::Heads;
constexpr int Batch = 4;
constexpr int Groups = 4, Threads = 128 * Groups, Stages = 3;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
__device__ __forceinline__ int swizzle(int row, int d) {
  return row * 128 + (d ^ ((row & 7) * 16));
}

// Each retained raw dot is the native WGMMA dot. Omitted weights are
// negative, so native masked scoring dominates full scoring by monotonicity.
// Packed FMA chains and their warp tree have at most ten rounding steps per
// term; native reduction has at most seven. The positive magnitude reduction
// has at most ten steps too. 2^-18 = 64u exceeds
// (gamma_10 + gamma_7) / (1 - gamma_10). The final envelope is rounded upwards.
// Admitted FP8/weight/
// power-of-two-scale domains exclude overflow and FP32 subnormal intermediates.
// This bound covers ONLY the FP32 epilogue, never a speculative raw-dot error.
template <int P, int B>
__device__ __forceinline__ void emit(
    const float* a, const float* wk, const float* ks, const float* tau,
    const int* starts, const int* ends, uint8_t* masks, float* upper,
    int first_row, int first, int rows, int nk, int stride, bool debug) {
  const int lane = threadIdx.x % 32, warp = (threadIdx.x % 128) / 32;
  float sums[2] = {}, magnitude[2] = {};
#pragma unroll
  for (int block = B * P / 8; block < ((B + 1) * P + 7) / 8; ++block) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      int head = block * 8 + (lane % 4) * 2 + pair;
      float w = (head >= B * P && head < (B + 1) * P) ? wk[block * 2 + pair] : 0.f;
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        float raw = fmaxf(a[block * 4 + pair + k * 2], 0.f);
        sums[k] = __fmaf_rn(raw, w, sums[k]);
        magnitude[k] = __fmaf_rn(raw, fabsf(w), magnitude[k]);
      }
    }
  }
  const int row = first_row + B;
#pragma unroll
  for (int k = 0; k < 2; ++k) {
#pragma unroll
    for (int d = 1; d <= 2; d *= 2) {
      sums[k] = __fadd_rn(sums[k], __shfl_xor_sync(0xffffffffu, sums[k], d));
      magnitude[k] = __fadd_rn(magnitude[k], __shfl_xor_sync(0xffffffffu, magnitude[k], d));
    }
    int key = warp * 16 + lane / 4 + k * 8, logical = first + key;
    float value = __fmul_ru(__fadd_ru(sums[k], __fmul_ru(magnitude[k], 0x1p-18f)), ks[key]);
    bool visible = row < rows && logical < nk && logical >= starts[row] && logical < ends[row];
    bool keep = visible && !(value < tau[row]) && (lane % 4 == 0);
    if (debug && row < rows && logical < nk && lane % 4 == 0)
      upper[int64_t(row) * nk + logical] = visible ? value : -INFINITY;
    unsigned bits = __ballot_sync(0xffffffffu, keep);
    bits = (bits | (bits >> 3)) & 0x03030303u;
    bits = (bits | (bits >> 6)) & 0x000f000fu;
    bits = (bits | (bits >> 12)) & 0xffu;
    if (lane == 0 && row < rows && first + warp * 16 + k * 8 < nk)
      masks[int64_t(row) * stride + (first + warp * 16 + k * 8) / 8] = uint8_t(bits);
  }
}

__device__ __forceinline__ void trace(volatile int* progress, int field, int value) {
  if (progress && blockIdx.x == 0 && blockIdx.y == 0) {
    progress[field] = value;
    __threadfence_system();
  }
}

template <int P>
__device__ __forceinline__ void screen(
    const uint8_t* packed_q, const int* head_ids, const int* budgets,
    const float* weights, const float* tau, const int* starts,
    const int* ends, uint8_t* masks, float* upper,
    int rows, int nk, int stride, int ctas, bool debug,
    const CUtensorMap& key_map, const CUtensorMap& scale_map, volatile int* progress) {
  const int first_row = int(blockIdx.x) * Batch;
  constexpr int N = Batch * P, Tile = Groups * 64;
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sq = smem;
  uint8_t* sk = sq + N * 128;
  float* wk = reinterpret_cast<float*>(sk + Stages * Tile * 128);
  // TMA shared destinations require 128-byte alignment, including the
  // non-swizzled scale tile (P14/P20 weights occupy partial 128-byte units).
  float* ks = wk + ((Batch * P + 31) / 32) * 32;
  Barrier* full = reinterpret_cast<Barrier*>(ks + Stages * Tile);
  Barrier* empty = full + Stages;
  const int thread = threadIdx.x, t = thread % 128, wg = thread / 128;
  for (int i = thread; i < Batch * P; i += Threads + 128) {
    int b = i / P, slot = i % P, row = first_row + b;
    wk[i] = row < rows ? weights[row * H + head_ids[row * H + slot]] : 0.f;
  }
  for (int i = thread; i < N * 8; i += Threads + 128) {
    int h = i / 8, row = first_row + h / P, d = i % 8 * 16;
    uint4 value = make_uint4(0, 0, 0, 0);
    if (row < rows)
      value = *reinterpret_cast<const uint4*>(packed_q + (int64_t(row) * H + h % P) * 128 + d);
    *reinterpret_cast<uint4*>(sq + swizzle(h, d)) = value;
  }
  if (thread == Threads) {
#pragma unroll
    for (int i = 0; i < Stages; ++i) {
      full[i].init(1);
      empty[i].init(Threads);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if (thread == 0) trace(progress, 0, 2);
  if (thread == Threads) trace(progress, 1, 2);
  int last = 0;
#pragma unroll
  for (int b = 0; b < Batch; ++b)
    if (first_row + b < rows) last = max(last, min(nk, ends[first_row + b]));
  const int begin = 12288 + int(blockIdx.y) * Tile;
  if (thread >= Threads) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    if (thread != Threads) return;
    trace(progress, 1, 3);
    int iteration = 0;
    for (int first = begin; first < last; first += ctas * Tile, ++iteration) {
      int stage = iteration % Stages, phase = (iteration / Stages) & 1;
      trace(progress, 1, 100 + iteration * 10 + 1);
      empty[stage].wait(phase ^ 1);
      trace(progress, 1, 100 + iteration * 10 + 2);
      dg::tma::copy<128, Tile, 128>(&key_map, full + stage, sk + stage * Tile * 128, 0, first);
      dg::tma::copy<Tile, 1, 0>(&scale_map, full + stage, ks + stage * Tile, first, 0);
      full[stage].arrive_and_expect_tx(Tile * 128 + Tile * 4);
      trace(progress, 1, 100 + iteration * 10 + 3);
    }
    return;
  }
  if (thread == 0) trace(progress, 0, 3);
  cutlass::arch::warpgroup_reg_alloc<112>();
  if (thread == 0) trace(progress, 0, 4);
  float register_weights[N / 4];
#pragma unroll
  for (int block = 0; block < N / 8; ++block) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair)
      register_weights[block * 2 + pair] = wk[block * 8 + (t % 4) * 2 + pair];
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  int iteration = 0;
  for (int first = begin; first < last; first += ctas * Tile, ++iteration) {
    int stage = iteration % Stages, phase = (iteration / Stages) & 1;
    if (thread == 0) trace(progress, 0, 100 + iteration * 10 + 1);
    full[stage].wait(phase);
    if (thread == 0) trace(progress, 0, 100 + iteration * 10 + 2);
    using M = typename dg::mma::sm90::FP8MMASelector<N>::type;
    float a[N / 2];
#pragma unroll
    for (int i = 0; i < N / 2; ++i) dg::ptx::warpgroup_fence_operand(a[i]);
    dg::ptx::warpgroup_arrive();
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      auto x = dg::mma::sm90::make_smem_desc(sk + stage * Tile * 128 + wg * 8192 + d * 32,
          dg::mma::sm90::to_swizzle_cute_type<128>(), 0, 1024);
      auto y = dg::mma::sm90::make_smem_desc(sq + d * 32,
          dg::mma::sm90::to_swizzle_cute_type<128>(), 0, 1024);
      M::wgmma(x, y, a, d != 0);
    }
    dg::ptx::warpgroup_commit_batch();
#pragma unroll
    for (int i = 0; i < N / 2; ++i) dg::ptx::warpgroup_fence_operand(a[i]);
    dg::ptx::warpgroup_wait<0>();
    if (thread == 0) trace(progress, 0, 100 + iteration * 10 + 3);
    emit<P, 0>(a, register_weights, ks + stage * Tile + wg * 64, tau, starts, ends, masks, upper, first_row, first + wg * 64, rows, nk, stride, debug);
    emit<P, 1>(a, register_weights, ks + stage * Tile + wg * 64, tau, starts, ends, masks, upper, first_row, first + wg * 64, rows, nk, stride, debug);
    emit<P, 2>(a, register_weights, ks + stage * Tile + wg * 64, tau, starts, ends, masks, upper, first_row, first + wg * 64, rows, nk, stride, debug);
    emit<P, 3>(a, register_weights, ks + stage * Tile + wg * 64, tau, starts, ends, masks, upper, first_row, first + wg * 64, rows, nk, stride, debug);
    empty[stage].arrive();
    if (thread == 0) trace(progress, 0, 100 + iteration * 10 + 4);
  }
}

__global__ __launch_bounds__(Threads + 128, 1) void dispatch(
    const uint8_t* q, const int* ids, const int* budgets, const float* w,
    const uint8_t* keys, const float* scales, const float* tau,
    const int* starts, const int* ends, uint8_t* masks, float* upper,
    int rows, int nk, int stride, int ctas, bool debug,
    const __grid_constant__ CUtensorMap key_map, const __grid_constant__ CUtensorMap scale_map, volatile int* progress) {
  const int paid = budgets[blockIdx.x / 2];
#define SCREEN(P) screen<P>(q, ids, budgets, w, tau, starts, ends, masks, upper, rows, nk, stride, ctas, debug, key_map, scale_map, progress)
  if (paid == 14) { SCREEN(14); }
  else if (paid == 16) { SCREEN(16); }
  else if (paid == 20) { SCREEN(20); }
  else if (paid == 24) { SCREEN(24); }
  else { SCREEN(32); }
#undef SCREEN
}

CUtensorMap make_map(void* ptr, CUtensorMapDataType type,
    std::initializer_list<cuuint64_t> dims, std::initializer_list<cuuint64_t> strides,
    std::initializer_list<cuuint32_t> box, CUtensorMapSwizzle swizzle) {
  CUtensorMap map;
  cuuint32_t element_strides[2] = {1, 1};
  CUresult result = cuTensorMapEncodeTiled(&map, type, 2, ptr, dims.begin(), strides.begin(),
      box.begin(), element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(result == CUDA_SUCCESS, "TMA descriptor failed: ", result);
  return map;
}

void run(Tensor q, Tensor ids, Tensor budgets, Tensor w, Tensor k,
         Tensor scales, Tensor tau, Tensor starts, Tensor ends, Tensor masks,
         Tensor upper, int64_t ctas, bool debug) {
  STD_TORCH_CHECK(H == 32, "Native partial screen currently requires H32");
  DeviceGuard guard(q.get_device_index());
  constexpr int Tile = Groups * 64;
  constexpr int Bytes = Batch * 32 * 128 + Stages * Tile * 128 + Batch * 32 * 4 + Stages * Tile * 4 + 2 * Stages * sizeof(Barrier);
  auto key_map = make_map(k.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
      {128, cuuint64_t(k.size(0))}, {128}, {128, Tile}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto scale_map = make_map(scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
      {cuuint64_t(k.size(0)), 1}, {cuuint64_t((k.size(0) * 4 + 15) & ~15)}, {Tile, 1}, CU_TENSOR_MAP_SWIZZLE_NONE);
  DOSA_CUDA_CHECK(cudaFuncSetAttribute(dispatch, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes));
  int* progress = nullptr;
  const bool tracing = std::getenv("DOSA_TMA_TRACE") != nullptr;
  if (tracing) {
    DOSA_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&progress), 64 * sizeof(int), cudaHostAllocMapped));
    for (int i = 0; i < 64; ++i) progress[i] = 0;
  }
  dispatch<<<dim3((q.size(0) + Batch - 1) / Batch, ctas), Threads + 128, Bytes, get_current_cuda_stream()>>>(
      q.const_data_ptr<uint8_t>(), ids.const_data_ptr<int>(), budgets.const_data_ptr<int>(),
      w.const_data_ptr<float>(), static_cast<const uint8_t*>(k.data_ptr()), scales.const_data_ptr<float>(),
      tau.const_data_ptr<float>(), starts.const_data_ptr<int>(), ends.const_data_ptr<int>(),
      masks.mutable_data_ptr<uint8_t>(), upper.mutable_data_ptr<float>(), q.size(0), k.size(0), masks.size(1), ctas, debug, key_map, scale_map, progress);
  if (tracing) {
    cudaEvent_t event;
    DOSA_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    DOSA_CUDA_CHECK(cudaEventRecord(event, get_current_cuda_stream()));
    for (int second = 0; second < 8; ++second) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      auto data = static_cast<volatile int*>(progress);
      auto status = cudaEventQuery(event);
      std::fprintf(stderr, "TMA_TRACE second=%d math=%d producer=%d event=%d\n", second, data[0], data[1], int(status));
      std::fflush(stderr);
      if (status == cudaSuccess) break;
      if (second == 7) std::_Exit(73);  // Diagnostic child only: release a stuck GPU context.
    }
    DOSA_CUDA_CHECK(cudaEventDestroy(event));
    DOSA_CUDA_CHECK(cudaFreeHost(progress));
  }
  DOSA_CUDA_CHECK(cudaGetLastError());
}

}  // namespace DOSA_NAMESPACE::sm90_partial

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("screen") "(Tensor packed_q, Tensor ids, Tensor budgets, Tensor weights, Tensor keys, Tensor scales, Tensor tau, Tensor starts, Tensor ends, Tensor! masks, Tensor! upper, int ctas, bool debug) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("screen"), TORCH_BOX(&DOSA_NAMESPACE::sm90_partial::run));
}
