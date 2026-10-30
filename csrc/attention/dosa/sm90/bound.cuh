#pragma once

// Same four RN chains and outward direct-ABS tail as screen_reference.cu.
namespace DOSA_NAMESPACE::sm90_bound {
__device__ __forceinline__ float2 add(float2 a, float2 b) {
  return make_float2(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
}
__device__ __forceinline__ float2 fma(float2 a, float2 b, float2 c) {
  return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
}
template <int Paid, bool Direct>
__device__ __forceinline__ float evaluate(const float* acc, const float* meta,
                                          float lo0, float lo1, float raw,
                                          float out, float scale, float rho) {
  float2 s0 = make_float2(0, 0), s1 = make_float2(0, 0);
  float core;
  if constexpr (Direct && Paid == 14) {
#pragma unroll
    for (int h = 0; h < 16; h += 4) {
      float2 a = h == 0 ? make_float2(acc[0], acc[1])
                        : make_float2(fabsf(acc[h]), fabsf(acc[h + 1]));
      float2 w = h == 0 ? make_float2(-lo0, -lo1)
                        : make_float2(__fmul_rn(.5f, meta[h]),
                                      __fmul_rn(.5f, meta[h + 1]));
      s0 = fma(a, w, s0);
      s1 = fma(
          make_float2(fabsf(acc[h + 2]), fabsf(acc[h + 3])),
          make_float2(__fmul_rn(.5f, meta[h + 2]), __fmul_rn(.5f, meta[h + 3])),
          s1);
    }
    float2 pair = add(s0, s1);
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
    float2 pair = add(s0, s1);
    float paid = __fadd_rn(pair.x, pair.y);
    float negative =
        __fadd_rn(0.f, __fmaf_rn(acc[1], lo1, __fmul_rn(acc[0], lo0)));
    if constexpr (Direct)
      core = __fmaf_rn(paid, .5f, -negative);
    else
      core = __fadd_rn(__fmul_rn(paid, .5f), -fmaxf(negative, 0.f));
  }
  if constexpr (Direct)
    return __fmaf_ru(scale,
                     __fmaf_ru(rho, meta[Paid + 2], __fadd_ru(core, raw)), out);
  return __fadd_rn(__fmul_rn(scale, __fadd_rn(__fadd_rn(core, raw),
                                              __fmul_rn(rho, meta[Paid + 2]))),
                   out);
}
}  // namespace DOSA_NAMESPACE::sm90_bound
