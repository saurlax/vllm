// Qualification only: reuse the author's BF16 epilogue without a TMEM/MMA
// dependency at execution. This is not the SM90 native FP32 scoring backend.
#include "../score.cuh"

namespace DOSA_NAMESPACE {
namespace sm90_reference {
constexpr int Heads = dosa_geometry::Heads;

__global__ void reduce_kernel(const float* dots, const float* weights,
                              const float* scales, float* output, int rows,
                              int keys) {
  const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * keys) return;
  const int row = i / keys, key = i % keys;
  float acc[Heads], w[Heads];
#pragma unroll
  for (int h = 0; h < Heads; ++h) {
    acc[h] = dots[(int64_t(row) * Heads + h) * keys + key];
    w[h] = weights[row * Heads + h];
  }
  output[i] =
      certicascade_streaming_exact::native_bf16_reduce(acc, w, scales[key]);
}

void reduce(Tensor dots, Tensor weights, Tensor scales, Tensor output) {
  for (const auto& tensor : {dots, weights, scales, output}) {
    STD_TORCH_CHECK(
        tensor.is_cuda() && tensor.is_contiguous(),
        "SM90 qualification tensors must be contiguous CUDA tensors");
    STD_TORCH_CHECK(
        tensor.scalar_type() == torch::headeronly::ScalarType::Float,
        "SM90 qualification reducer requires FP32 tensors");
    STD_TORCH_CHECK(tensor.get_device_index() == dots.get_device_index(),
                    "SM90 qualification tensors must share a device");
  }
  STD_TORCH_CHECK(dots.dim() == 3 && dots.size(1) == Heads,
                  "Expected raw dots [Q,H,K]");
  const int64_t rows = dots.size(0), keys = dots.size(2);
  STD_TORCH_CHECK(rows > 0 && keys > 0 && rows * keys <= INT_MAX,
                  "Invalid qualification dimensions");
  STD_TORCH_CHECK(weights.dim() == 2 && weights.size(0) == rows &&
                      weights.size(1) == Heads && scales.dim() == 1 &&
                      scales.size(0) == keys && output.dim() == 2 &&
                      output.size(0) == rows && output.size(1) == keys,
                  "Qualification tensor shape mismatch");
  DeviceGuard guard(dots.get_device_index());
  reduce_kernel<<<(rows * keys + 127) / 128, 128, 0,
                  get_current_cuda_stream()>>>(
      dots.const_data_ptr<float>(), weights.const_data_ptr<float>(),
      scales.const_data_ptr<float>(), output.mutable_data_ptr<float>(),
      int(rows), int(keys));
  DOSA_CUDA_CHECK(cudaGetLastError());
}
}  // namespace sm90_reference
}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("reduce") "(Tensor dots, Tensor weights, Tensor scales, Tensor! output) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("reduce"),
           TORCH_BOX(&DOSA_NAMESPACE::sm90_reference::reduce));
}
