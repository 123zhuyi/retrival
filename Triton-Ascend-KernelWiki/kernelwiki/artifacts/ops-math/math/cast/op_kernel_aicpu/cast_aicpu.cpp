/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_aicpu.h"

#include <memory.h>
#include <cfloat>
#include <cmath>
#include <ctime>

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *const kCast = "Cast";
constexpr size_t k2Dimensions = 2U;
constexpr uint32_t kMinCoreNum = 1U;
constexpr uint64_t kMinDataSize = 1U;

// FP32 constants
constexpr uint32_t kFp32SignIndex = 31U;
constexpr uint32_t kFp32ExpMask = 0x7F800000U;
constexpr uint32_t kFp32ManMask = 0x007FFFFFU;
constexpr uint32_t kFp32AbsMask = 0x7FFFFFFFU;
constexpr int8_t kFp32ManMaxLen = 23;
// FP16 constants
constexpr uint16_t kFp16SignIndex = 15U;
constexpr uint16_t kFp16ExpMask = 0x7C00U;
constexpr uint16_t kFp16ManMask = 0x03FFU;
constexpr uint16_t kFp16AbsMask = 0x7FFFU;
constexpr int8_t kFp16ManMaxLen = 10;
// BF16 constants
constexpr uint16_t kBf16SignIndex = 15U;
constexpr uint16_t kBf16ExpMask = 0x7F80U;
constexpr uint16_t kBf16ManMask = 0x007FU;
constexpr uint16_t kBf16AbsMask = 0x7FFFU;
constexpr int8_t kBf16ManMaxLen = 7;
// HIF8 constants
constexpr int8_t kHif8NegMax = 0xEF;
constexpr int8_t kHif8PosMax = 0x6F;
constexpr int8_t kHif8Nan = 0x80;
constexpr int8_t kHif8Zero = 0x00;
constexpr float kHif8OverValue = 49152.0f;
constexpr int8_t kHif8ExpMin = -23;
constexpr int8_t kHif8DenormalExpMin = -22;
constexpr int8_t kHif8DenormalExpMax = -15;
constexpr int8_t kHif8D1Exp = 1;
constexpr int8_t kHif8D2ExpMin = 2;
constexpr int8_t kHif8D2ExpMax = 3;
constexpr int8_t kHif8D3ExpMin = 4;
constexpr int8_t kHif8D3ExpMax = 7;
constexpr int8_t kHif8D4ExpMin = 8;
constexpr int8_t kHif8D4ExpMax = 15;
constexpr int8_t kHif8DenormalDotValue = 0;
constexpr int8_t kHif8D0DotValue = 1;
constexpr int8_t kHif8D1DotValue = 2;
constexpr int8_t kHif8D2DotValue = 4;
constexpr int8_t kHif8D3DotValue = 8;
constexpr int8_t kHif8D4DotValue = 12;
constexpr int8_t kHif8ExpBitsOne = 1;
constexpr int8_t kHif8ExpBitsTwo = 2;
constexpr int8_t kHif8ExpBitsThree = 3;
constexpr int8_t kHif8ExpBitsFour = 4;
constexpr int8_t kHif8ManBitsOne = 1;
constexpr int8_t kHif8ManBitsTwo = 2;
constexpr int8_t kHif8ManBitsThree = 3;
constexpr uint8_t kHif8NegSignValue = 128;
constexpr int8_t kHif8DotLeftShift = 3;

inline bool Fp32IsNan(uint32_t x) {
  return ((x & kFp32ExpMask) == kFp32ExpMask) && ((x & kFp32ManMask) != 0);
}
inline bool Fp32IsInf(uint32_t x) {
  return ((x & kFp32ExpMask) == kFp32ExpMask) && ((x & kFp32ManMask) == 0);
}
inline bool Fp32IsZero(uint32_t x) { return (x & kFp32AbsMask) == 0; }
inline bool Fp32ExtractSign(uint32_t x) {
  return ((x >> kFp32SignIndex) & 0x1) == 0x1;
}
inline bool Fp16IsNan(uint16_t x) {
  return ((x & kFp16ExpMask) == kFp16ExpMask) && ((x & kFp16ManMask) != 0);
}
inline bool Fp16IsInf(uint16_t x) {
  return ((x & kFp16ExpMask) == kFp16ExpMask) && ((x & kFp16ManMask) == 0);
}
inline bool Fp16IsZero(uint16_t x) { return (x & kFp16AbsMask) == 0; }
inline bool Fp16ExtractSign(uint16_t x) {
  return ((x >> kFp16SignIndex) & 0x1) == 0x1;
}
inline bool Bf16IsNan(uint16_t x) {
  return ((x & kBf16ExpMask) == kBf16ExpMask) && ((x & kBf16ManMask) != 0);
}
inline bool Bf16IsInf(uint16_t x) {
  return ((x & kBf16ExpMask) == kBf16ExpMask) && ((x & kBf16ManMask) == 0);
}
inline bool Bf16IsZero(uint16_t x) { return (x & kBf16AbsMask) == 0; }
inline bool Bf16ExtractSign(uint16_t x) {
  return ((x >> kBf16SignIndex) & 0x1) == 0x1;
}

struct HiF8Info {
  int8_t dot_value;
  int8_t exponent_bits;
  int8_t man_bits;
};

void GetHif8Info(int8_t exponent, HiF8Info &info) {
  if (exponent < kHif8DenormalExpMin) {
    info.dot_value = -1;
    info.exponent_bits = kHif8ExpBitsThree;
    info.man_bits = 0;
  } else if (exponent >= kHif8DenormalExpMin && exponent < kHif8DenormalExpMax) {
    info.dot_value = kHif8DenormalDotValue;
    info.exponent_bits = kHif8ExpBitsThree;
    info.man_bits = 0;
  } else if (exponent == 0) {
    info.dot_value = kHif8D0DotValue;
    info.exponent_bits = 0;
    info.man_bits = kHif8ManBitsThree;
  } else if (abs(exponent) == kHif8D1Exp) {
    info.dot_value = kHif8D1DotValue;
    info.exponent_bits = kHif8ExpBitsOne;
    info.man_bits = kHif8ManBitsThree;
  } else if (abs(exponent) >= kHif8D2ExpMin && abs(exponent) <= kHif8D2ExpMax) {
    info.dot_value = kHif8D2DotValue;
    info.exponent_bits = kHif8ExpBitsTwo;
    info.man_bits = kHif8ManBitsThree;
  } else if (abs(exponent) >= kHif8D3ExpMin && abs(exponent) <= kHif8D3ExpMax) {
    info.dot_value = kHif8D3DotValue;
    info.exponent_bits = kHif8ExpBitsThree;
    info.man_bits = kHif8ManBitsTwo;
  } else if (abs(exponent) >= kHif8D4ExpMin && abs(exponent) <= kHif8D4ExpMax) {
    info.dot_value = kHif8D4DotValue;
    info.exponent_bits = kHif8ExpBitsFour;
    info.man_bits = kHif8ManBitsOne;
  } else if (exponent > kHif8D4ExpMax) {
    info.dot_value = kHif8D4DotValue;
    info.exponent_bits = kHif8ExpBitsFour;
    info.man_bits = -1;
  }
}

bool FpTaRoundToHif8(int32_t fraction_int, int8_t man_bits, int8_t exponent,
                     uint32_t mantissa_len, int8_t &hif8_fraction) {
  if (exponent == kHif8ExpMin) {
    hif8_fraction = 0;
    return true;
  }
  int8_t tmp = fraction_int >> (mantissa_len - (man_bits + 1));
  if (tmp == (1 << (man_bits + 1)) - 1) {
    hif8_fraction = 0;
    return true;
  }
  if (tmp == 0) {
    hif8_fraction = 0;
    return false;
  }
  hif8_fraction = (tmp + (tmp & 1)) >> 1;
  return false;
}

int8_t FpToHif8Proc(bool sign, int8_t exponent, int8_t fraction, HiF8Info &info) {
  if (info.man_bits == -1) return sign ? kHif8NegMax : kHif8PosMax;
  if (exponent <= -kFp32ManMaxLen) return kHif8Zero;
  uint8_t sign_val = sign ? kHif8NegSignValue : 0;
  if (info.dot_value == 0) return sign_val + exponent + kFp32ManMaxLen;
  if (info.dot_value == 1) return sign_val + (info.dot_value << kHif8DotLeftShift) + fraction;
  int8_t abs_exp = abs(exponent) - (1 << (info.exponent_bits - 1));
  int8_t sign_exp = (exponent < 0 ? 1 : 0) << (info.exponent_bits - 1 + info.man_bits);
  return sign_val + (info.dot_value << kHif8DotLeftShift) + sign_exp + (abs_exp << info.man_bits) + fraction;
}
} // namespace

namespace aicpu {

hif8_impl::Hif8Raw::Hif8Raw() : val(0) {}
hif8_impl::Hif8Base::Hif8Base() : Hif8Raw() {}
hif8_impl::Hif8Base::Hif8Base(const Hif8Raw &v) : Hif8Raw(v) {}

hif8_impl::Hif8Raw hif8_impl::Fp32ToHif8Rtne(float f) {
  Hif8Raw ret;
  uint32_t x = *reinterpret_cast<uint32_t *>(&f);
  bool sign = Fp32ExtractSign(x);
  float f_abs = fabs(f);
  if (Fp32IsNan(x)) { ret.val = kHif8Nan; return ret; }
  if (Fp32IsInf(x) || f_abs >= kHif8OverValue) { ret.val = sign ? kHif8NegMax : kHif8PosMax; return ret; }
  if (Fp32IsZero(x)) { ret.val = kHif8Zero; return ret; }
  int8_t exponent = static_cast<int8_t>(floor(log2(f_abs)));
  HiF8Info info;
  GetHif8Info(exponent, info);
  int32_t fraction = static_cast<int32_t>(f_abs * pow(2.0, kFp32ManMaxLen - exponent) - pow(2.0, kFp32ManMaxLen));
  int8_t hif8_frac = 0;
  if (FpTaRoundToHif8(fraction, info.man_bits, exponent, kFp32ManMaxLen, hif8_frac)) {
    exponent++;
    GetHif8Info(exponent, info);
  }
  ret.val = FpToHif8Proc(sign, exponent, hif8_frac, info);
  return ret;
}

hif8_impl::Hif8Raw hif8_impl::Fp16ToHif8Rtne(Eigen::half f) {
  Hif8Raw ret;
  uint16_t x = *reinterpret_cast<uint16_t *>(&f);
  bool sign = Fp16ExtractSign(x);
  Eigen::half f_abs = sign ? Eigen::half(-static_cast<float>(f)) : f;
  if (Fp16IsNan(x)) { ret.val = kHif8Nan; return ret; }
  if (Fp16IsInf(x) || static_cast<float>(f_abs) >= kHif8OverValue) { ret.val = sign ? kHif8NegMax : kHif8PosMax; return ret; }
  if (Fp16IsZero(x)) { ret.val = kHif8Zero; return ret; }
  int8_t exponent = static_cast<int8_t>(floor(log2(static_cast<float>(f_abs))));
  HiF8Info info;
  GetHif8Info(exponent, info);
  int32_t fraction = static_cast<int32_t>(static_cast<float>(f_abs) * pow(2.0, kFp16ManMaxLen - exponent) - pow(2.0, kFp16ManMaxLen));
  int8_t hif8_frac = 0;
  if (FpTaRoundToHif8(fraction, info.man_bits, exponent, kFp16ManMaxLen, hif8_frac)) {
    exponent++;
    GetHif8Info(exponent, info);
  }
  ret.val = FpToHif8Proc(sign, exponent, hif8_frac, info);
  return ret;
}

hif8_impl::Hif8Raw hif8_impl::Bf16ToHif8Rtne(Eigen::bfloat16 f) {
  Hif8Raw ret;
  uint16_t x = *reinterpret_cast<uint16_t *>(&f);
  bool sign = Bf16ExtractSign(x);
  Eigen::bfloat16 f_abs = sign ? Eigen::bfloat16(-static_cast<float>(f)) : f;
  if (Bf16IsNan(x)) { ret.val = kHif8Nan; return ret; }
  if (Bf16IsInf(x) || static_cast<float>(f_abs) >= kHif8OverValue) { ret.val = sign ? kHif8NegMax : kHif8PosMax; return ret; }
  if (Bf16IsZero(x)) { ret.val = kHif8Zero; return ret; }
  int8_t exponent = static_cast<int8_t>(floor(log2(static_cast<float>(f_abs))));
  HiF8Info info;
  GetHif8Info(exponent, info);
  int32_t fraction = static_cast<int32_t>(static_cast<float>(f_abs) * pow(2.0, kBf16ManMaxLen - exponent) - pow(2.0, kBf16ManMaxLen));
  int8_t hif8_frac = 0;
  if (FpTaRoundToHif8(fraction, info.man_bits, exponent, kBf16ManMaxLen, hif8_frac)) {
    exponent++;
    GetHif8Info(exponent, info);
  }
  ret.val = FpToHif8Proc(sign, exponent, hif8_frac, info);
  return ret;
}

hif8::hif8(float f) : hif8_impl::Hif8Base(hif8_impl::Fp32ToHif8Rtne(f)) {}
hif8::hif8(Eigen::half f) : hif8_impl::Hif8Base(hif8_impl::Fp16ToHif8Rtne(f)) {}
hif8::hif8(Eigen::bfloat16 f) : hif8_impl::Hif8Base(hif8_impl::Bf16ToHif8Rtne(f)) {}

CastCpuKernel::CastCpuKernel() : calls_({}), x_tensor_(nullptr), y_tensor_(nullptr),
                                  x_data_type_(DT_INT64), y_data_type_(DT_INT64),
                                  x_data_size_(0), y_data_size_(0) {}

template <typename T, typename S>
void CastFp16ToComplex(Tensor *&x_tensor, Tensor *&y_tensor, const int64_t &start,
    const int64_t &end) {
  T *inptr = static_cast<T *>(x_tensor->GetData());
  S *outptr = static_cast<S *>(y_tensor->GetData());
  Eigen::TensorMap<Eigen::Tensor<T, k2Dimensions, Eigen::RowMajor>> input_map(
      (inptr + start), 1, (end - start));
  const auto &input = Eigen::Tensor<T, k2Dimensions, Eigen::RowMajor>(input_map);
  Eigen::TensorMap<Eigen::Tensor<S, k2Dimensions, Eigen::RowMajor>> output(
      (outptr + start), 1, (end - start));
  auto tmp = input.template cast<float>();
  output = tmp.template cast<S>();
}

template <typename T, typename S>
void CastTask(Tensor *&x_tensor, Tensor *&y_tensor, const int64_t &start,
    const int64_t &end) {
  T *inptr = static_cast<T *>(x_tensor->GetData());
  S *outptr = static_cast<S *>(y_tensor->GetData());
  for (int64_t i = start; i < end; i++) {
    outptr[i] = static_cast<S>(inptr[i]);
  }
}

void CastCpuKernel::SetMap() {
  SetInt8Map();
  SetInt16Map();
  SetInt32Map();
  SetInt64Map();
  SetFloat16Map();
  SetFloatMap();
  SetDoubleMap();
  SetUInt8Map();
  SetUInt16Map();
  SetUInt32Map();
  SetUInt64Map();
  SetBoolMap();
  SetComplexMap();
  SetBfloat16Map();
  SetHif8Map();
}

void CastCpuKernel::SetInt8Map() {
  calls_[DT_INT8][DT_INT8] = CastTask<int8_t, int8_t>;
  calls_[DT_INT8][DT_INT16] = CastTask<int8_t, int16_t>;
  calls_[DT_INT8][DT_INT32] = CastTask<int8_t, int32_t>;
  calls_[DT_INT8][DT_INT64] = CastTask<int8_t, int64_t>;
  calls_[DT_INT8][DT_FLOAT16] = CastTask<int8_t, Eigen::half>;
  calls_[DT_INT8][DT_FLOAT] = CastTask<int8_t, float>;
  calls_[DT_INT8][DT_DOUBLE] = CastTask<int8_t, double>;
  calls_[DT_INT8][DT_UINT8] = CastTask<int8_t, uint8_t>;
  calls_[DT_INT8][DT_UINT16] = CastTask<int8_t, uint16_t>;
  calls_[DT_INT8][DT_UINT32] = CastTask<int8_t, uint32_t>;
  calls_[DT_INT8][DT_UINT64] = CastTask<int8_t, uint64_t>;
  calls_[DT_INT8][DT_BOOL] = CastTask<int8_t, bool>;
  calls_[DT_INT8][DT_COMPLEX64] = CastTask<int8_t, std::complex<float>>;
  calls_[DT_INT8][DT_COMPLEX128] = CastTask<int8_t, std::complex<double>>;
  calls_[DT_INT8][DT_BFLOAT16] = CastTask<int8_t, Eigen::bfloat16>;
}

void CastCpuKernel::SetInt16Map() {
  calls_[DT_INT16][DT_INT8] = CastTask<int16_t, int8_t>;
  calls_[DT_INT16][DT_INT16] = CastTask<int16_t, int16_t>;
  calls_[DT_INT16][DT_INT32] = CastTask<int16_t, int32_t>;
  calls_[DT_INT16][DT_INT64] = CastTask<int16_t, int64_t>;
  calls_[DT_INT16][DT_FLOAT16] = CastTask<int16_t, Eigen::half>;
  calls_[DT_INT16][DT_FLOAT] = CastTask<int16_t, float>;
  calls_[DT_INT16][DT_DOUBLE] = CastTask<int16_t, double>;
  calls_[DT_INT16][DT_UINT8] = CastTask<int16_t, uint8_t>;
  calls_[DT_INT16][DT_UINT16] = CastTask<int16_t, uint16_t>;
  calls_[DT_INT16][DT_UINT32] = CastTask<int16_t, uint32_t>;
  calls_[DT_INT16][DT_UINT64] = CastTask<int16_t, uint64_t>;
  calls_[DT_INT16][DT_BOOL] = CastTask<int16_t, bool>;
  calls_[DT_INT16][DT_COMPLEX64] = CastTask<int16_t, std::complex<float>>;
  calls_[DT_INT16][DT_COMPLEX128] = CastTask<int16_t, std::complex<double>>;
  calls_[DT_INT16][DT_BFLOAT16] = CastTask<int16_t, Eigen::bfloat16>;
}

void CastCpuKernel::SetInt32Map() {
  calls_[DT_INT32][DT_INT8] = CastTask<int32_t, int8_t>;
  calls_[DT_INT32][DT_INT16] = CastTask<int32_t, int16_t>;
  calls_[DT_INT32][DT_INT32] = CastTask<int32_t, int32_t>;
  calls_[DT_INT32][DT_INT64] = CastTask<int32_t, int64_t>;
  calls_[DT_INT32][DT_FLOAT16] = CastTask<int32_t, Eigen::half>;
  calls_[DT_INT32][DT_FLOAT] = CastTask<int32_t, float>;
  calls_[DT_INT32][DT_DOUBLE] = CastTask<int32_t, double>;
  calls_[DT_INT32][DT_UINT8] = CastTask<int32_t, uint8_t>;
  calls_[DT_INT32][DT_UINT16] = CastTask<int32_t, uint16_t>;
  calls_[DT_INT32][DT_UINT32] = CastTask<int32_t, uint32_t>;
  calls_[DT_INT32][DT_UINT64] = CastTask<int32_t, uint64_t>;
  calls_[DT_INT32][DT_BOOL] = CastTask<int32_t, bool>;
  calls_[DT_INT32][DT_COMPLEX64] = CastTask<int32_t, std::complex<float>>;
  calls_[DT_INT32][DT_COMPLEX128] = CastTask<int32_t, std::complex<double>>;
  calls_[DT_INT32][DT_BFLOAT16] = CastTask<int32_t, Eigen::bfloat16>;
}

void CastCpuKernel::SetInt64Map() {
  calls_[DT_INT64][DT_INT8] = CastTask<int64_t, int8_t>;
  calls_[DT_INT64][DT_INT16] = CastTask<int64_t, int16_t>;
  calls_[DT_INT64][DT_INT32] = CastTask<int64_t, int32_t>;
  calls_[DT_INT64][DT_INT64] = CastTask<int64_t, int64_t>;
  calls_[DT_INT64][DT_FLOAT16] = CastTask<int64_t, Eigen::half>;
  calls_[DT_INT64][DT_FLOAT] = CastTask<int64_t, float>;
  calls_[DT_INT64][DT_DOUBLE] = CastTask<int64_t, double>;
  calls_[DT_INT64][DT_UINT8] = CastTask<int64_t, uint8_t>;
  calls_[DT_INT64][DT_UINT16] = CastTask<int64_t, uint16_t>;
  calls_[DT_INT64][DT_UINT32] = CastTask<int64_t, uint32_t>;
  calls_[DT_INT64][DT_UINT64] = CastTask<int64_t, uint64_t>;
  calls_[DT_INT64][DT_BOOL] = CastTask<int64_t, bool>;
  calls_[DT_INT64][DT_COMPLEX64] = CastTask<int64_t, std::complex<float>>;
  calls_[DT_INT64][DT_COMPLEX128] = CastTask<int64_t, std::complex<double>>;
  calls_[DT_INT64][DT_BFLOAT16] = CastTask<int64_t, Eigen::bfloat16>;
}

void CastCpuKernel::SetFloat16Map() {
  calls_[DT_FLOAT16][DT_INT8] = CastTask<Eigen::half, int8_t>;
  calls_[DT_FLOAT16][DT_INT16] = CastTask<Eigen::half, int16_t>;
  calls_[DT_FLOAT16][DT_INT32] = CastTask<Eigen::half, int32_t>;
  calls_[DT_FLOAT16][DT_INT64] = CastTask<Eigen::half, int64_t>;
  calls_[DT_FLOAT16][DT_FLOAT16] = CastTask<Eigen::half, Eigen::half>;
  calls_[DT_FLOAT16][DT_FLOAT] = CastTask<Eigen::half, float>;
  calls_[DT_FLOAT16][DT_DOUBLE] = CastTask<Eigen::half, double>;
  calls_[DT_FLOAT16][DT_UINT8] = CastTask<Eigen::half, uint8_t>;
  calls_[DT_FLOAT16][DT_UINT16] = CastTask<Eigen::half, uint16_t>;
  calls_[DT_FLOAT16][DT_UINT32] = CastTask<Eigen::half, uint32_t>;
  calls_[DT_FLOAT16][DT_UINT64] = CastTask<Eigen::half, uint64_t>;
  calls_[DT_FLOAT16][DT_BOOL] = CastTask<Eigen::half, bool>;
  calls_[DT_FLOAT16][DT_COMPLEX64] = CastFp16ToComplex<Eigen::half, std::complex<float>>;
  calls_[DT_FLOAT16][DT_COMPLEX128] = CastFp16ToComplex<Eigen::half, std::complex<double>>;
  calls_[DT_FLOAT16][DT_BFLOAT16] = CastTask<Eigen::half, Eigen::bfloat16>;
  calls_[DT_FLOAT16][DT_HIFLOAT8] = CastTask<Eigen::half, hif8>;
}

void CastCpuKernel::SetFloatMap() {
  calls_[DT_FLOAT][DT_INT8] = CastTask<float, int8_t>;
  calls_[DT_FLOAT][DT_INT16] = CastTask<float, int16_t>;
  calls_[DT_FLOAT][DT_INT32] = CastTask<float, int32_t>;
  calls_[DT_FLOAT][DT_INT64] = CastTask<float, int64_t>;
  calls_[DT_FLOAT][DT_FLOAT16] = CastTask<float, Eigen::half>;
  calls_[DT_FLOAT][DT_FLOAT] = CastTask<float, float>;
  calls_[DT_FLOAT][DT_DOUBLE] = CastTask<float, double>;
  calls_[DT_FLOAT][DT_UINT8] = CastTask<float, uint8_t>;
  calls_[DT_FLOAT][DT_UINT16] = CastTask<float, uint16_t>;
  calls_[DT_FLOAT][DT_UINT32] = CastTask<float, uint32_t>;
  calls_[DT_FLOAT][DT_UINT64] = CastTask<float, uint64_t>;
  calls_[DT_FLOAT][DT_BOOL] = CastTask<float, bool>;
  calls_[DT_FLOAT][DT_COMPLEX64] = CastTask<float, std::complex<float>>;
  calls_[DT_FLOAT][DT_COMPLEX128] = CastTask<float, std::complex<double>>;
  calls_[DT_FLOAT][DT_BFLOAT16] = CastTask<float, Eigen::bfloat16>;
  calls_[DT_FLOAT][DT_HIFLOAT8] = CastTask<float, hif8>;
}

void CastCpuKernel::SetDoubleMap() {
  calls_[DT_DOUBLE][DT_INT8] = CastTask<double, int8_t>;
  calls_[DT_DOUBLE][DT_INT16] = CastTask<double, int16_t>;
  calls_[DT_DOUBLE][DT_INT32] = CastTask<double, int32_t>;
  calls_[DT_DOUBLE][DT_INT64] = CastTask<double, int64_t>;
  calls_[DT_DOUBLE][DT_FLOAT16] = CastTask<double, Eigen::half>;
  calls_[DT_DOUBLE][DT_FLOAT] = CastTask<double, float>;
  calls_[DT_DOUBLE][DT_DOUBLE] = CastTask<double, double>;
  calls_[DT_DOUBLE][DT_UINT8] = CastTask<double, uint8_t>;
  calls_[DT_DOUBLE][DT_UINT16] = CastTask<double, uint16_t>;
  calls_[DT_DOUBLE][DT_UINT32] = CastTask<double, uint32_t>;
  calls_[DT_DOUBLE][DT_UINT64] = CastTask<double, uint64_t>;
  calls_[DT_DOUBLE][DT_BOOL] = CastTask<double, bool>;
  calls_[DT_DOUBLE][DT_COMPLEX64] = CastTask<double, std::complex<float>>;
  calls_[DT_DOUBLE][DT_COMPLEX128] = CastTask<double, std::complex<double>>;
  calls_[DT_DOUBLE][DT_BFLOAT16] = CastTask<double, Eigen::bfloat16>;
}

void CastCpuKernel::SetUInt8Map() {
  calls_[DT_UINT8][DT_INT8] = CastTask<uint8_t, int8_t>;
  calls_[DT_UINT8][DT_INT16] = CastTask<uint8_t, int16_t>;
  calls_[DT_UINT8][DT_INT32] = CastTask<uint8_t, int32_t>;
  calls_[DT_UINT8][DT_INT64] = CastTask<uint8_t, int64_t>;
  calls_[DT_UINT8][DT_FLOAT16] = CastTask<uint8_t, Eigen::half>;
  calls_[DT_UINT8][DT_FLOAT] = CastTask<uint8_t, float>;
  calls_[DT_UINT8][DT_DOUBLE] = CastTask<uint8_t, double>;
  calls_[DT_UINT8][DT_UINT8] = CastTask<uint8_t, uint8_t>;
  calls_[DT_UINT8][DT_UINT16] = CastTask<uint8_t, uint16_t>;
  calls_[DT_UINT8][DT_UINT32] = CastTask<uint8_t, uint32_t>;
  calls_[DT_UINT8][DT_UINT64] = CastTask<uint8_t, uint64_t>;
  calls_[DT_UINT8][DT_BOOL] = CastTask<uint8_t, bool>;
  calls_[DT_UINT8][DT_COMPLEX64] = CastTask<uint8_t, std::complex<float>>;
  calls_[DT_UINT8][DT_COMPLEX128] = CastTask<uint8_t, std::complex<double>>;
  calls_[DT_UINT8][DT_BFLOAT16] = CastTask<uint8_t, Eigen::bfloat16>;
}

void CastCpuKernel::SetUInt16Map() {
  calls_[DT_UINT16][DT_INT8] = CastTask<uint16_t, int8_t>;
  calls_[DT_UINT16][DT_INT16] = CastTask<uint16_t, int16_t>;
  calls_[DT_UINT16][DT_INT32] = CastTask<uint16_t, int32_t>;
  calls_[DT_UINT16][DT_INT64] = CastTask<uint16_t, int64_t>;
  calls_[DT_UINT16][DT_FLOAT16] = CastTask<uint16_t, Eigen::half>;
  calls_[DT_UINT16][DT_FLOAT] = CastTask<uint16_t, float>;
  calls_[DT_UINT16][DT_DOUBLE] = CastTask<uint16_t, double>;
  calls_[DT_UINT16][DT_UINT8] = CastTask<uint16_t, uint8_t>;
  calls_[DT_UINT16][DT_UINT16] = CastTask<uint16_t, uint16_t>;
  calls_[DT_UINT16][DT_UINT32] = CastTask<uint16_t, uint32_t>;
  calls_[DT_UINT16][DT_UINT64] = CastTask<uint16_t, uint64_t>;
  calls_[DT_UINT16][DT_BOOL] = CastTask<uint16_t, bool>;
  calls_[DT_UINT16][DT_COMPLEX64] = CastTask<uint16_t, std::complex<float>>;
  calls_[DT_UINT16][DT_COMPLEX128] = CastTask<uint16_t, std::complex<double>>;
  calls_[DT_UINT16][DT_BFLOAT16] = CastTask<uint16_t, Eigen::bfloat16>;
}

void CastCpuKernel::SetUInt32Map() {
  calls_[DT_UINT32][DT_INT8] = CastTask<uint32_t, int8_t>;
  calls_[DT_UINT32][DT_INT16] = CastTask<uint32_t, int16_t>;
  calls_[DT_UINT32][DT_INT32] = CastTask<uint32_t, int32_t>;
  calls_[DT_UINT32][DT_INT64] = CastTask<uint32_t, int64_t>;
  calls_[DT_UINT32][DT_FLOAT16] = CastTask<uint32_t, Eigen::half>;
  calls_[DT_UINT32][DT_FLOAT] = CastTask<uint32_t, float>;
  calls_[DT_UINT32][DT_DOUBLE] = CastTask<uint32_t, double>;
  calls_[DT_UINT32][DT_UINT8] = CastTask<uint32_t, uint8_t>;
  calls_[DT_UINT32][DT_UINT16] = CastTask<uint32_t, uint16_t>;
  calls_[DT_UINT32][DT_UINT32] = CastTask<uint32_t, uint32_t>;
  calls_[DT_UINT32][DT_UINT64] = CastTask<uint32_t, uint64_t>;
  calls_[DT_UINT32][DT_BOOL] = CastTask<uint32_t, bool>;
  calls_[DT_UINT32][DT_COMPLEX64] = CastTask<uint32_t, std::complex<float>>;
  calls_[DT_UINT32][DT_COMPLEX128] = CastTask<uint32_t, std::complex<double>>;
  calls_[DT_UINT32][DT_BFLOAT16] = CastTask<uint32_t, Eigen::bfloat16>;
}

void CastCpuKernel::SetUInt64Map() {
  calls_[DT_UINT64][DT_INT8] = CastTask<uint64_t, int8_t>;
  calls_[DT_UINT64][DT_INT16] = CastTask<uint64_t, int16_t>;
  calls_[DT_UINT64][DT_INT32] = CastTask<uint64_t, int32_t>;
  calls_[DT_UINT64][DT_INT64] = CastTask<uint64_t, int64_t>;
  calls_[DT_UINT64][DT_FLOAT16] = CastTask<uint64_t, Eigen::half>;
  calls_[DT_UINT64][DT_FLOAT] = CastTask<uint64_t, float>;
  calls_[DT_UINT64][DT_DOUBLE] = CastTask<uint64_t, double>;
  calls_[DT_UINT64][DT_UINT8] = CastTask<uint64_t, uint8_t>;
  calls_[DT_UINT64][DT_UINT16] = CastTask<uint64_t, uint16_t>;
  calls_[DT_UINT64][DT_UINT32] = CastTask<uint64_t, uint32_t>;
  calls_[DT_UINT64][DT_UINT64] = CastTask<uint64_t, uint64_t>;
  calls_[DT_UINT64][DT_BOOL] = CastTask<uint64_t, bool>;
  calls_[DT_UINT64][DT_COMPLEX64] = CastTask<uint64_t, std::complex<float>>;
  calls_[DT_UINT64][DT_COMPLEX128] = CastTask<uint64_t, std::complex<double>>;
  calls_[DT_UINT64][DT_BFLOAT16] = CastTask<uint64_t, Eigen::bfloat16>;
}

void CastCpuKernel::SetBoolMap() {
  calls_[DT_BOOL][DT_INT8] = CastTask<bool, int8_t>;
  calls_[DT_BOOL][DT_INT16] = CastTask<bool, int16_t>;
  calls_[DT_BOOL][DT_INT32] = CastTask<bool, int32_t>;
  calls_[DT_BOOL][DT_INT64] = CastTask<bool, int64_t>;
  calls_[DT_BOOL][DT_FLOAT16] = CastTask<bool, Eigen::half>;
  calls_[DT_BOOL][DT_FLOAT] = CastTask<bool, float>;
  calls_[DT_BOOL][DT_DOUBLE] = CastTask<bool, double>;
  calls_[DT_BOOL][DT_UINT8] = CastTask<bool, uint8_t>;
  calls_[DT_BOOL][DT_UINT16] = CastTask<bool, uint16_t>;
  calls_[DT_BOOL][DT_UINT32] = CastTask<bool, uint32_t>;
  calls_[DT_BOOL][DT_UINT64] = CastTask<bool, uint64_t>;
  calls_[DT_BOOL][DT_BOOL] = CastTask<bool, bool>;
  calls_[DT_BOOL][DT_COMPLEX64] = CastTask<bool, std::complex<float>>;
  calls_[DT_BOOL][DT_COMPLEX128] = CastTask<bool, std::complex<double>>;
  calls_[DT_BOOL][DT_BFLOAT16] = CastTask<bool, Eigen::bfloat16>;
}

void CastCpuKernel::SetComplexMap() {
  calls_[DT_COMPLEX64][DT_COMPLEX128] = CastTask<std::complex<float>, std::complex<double>>;
  calls_[DT_COMPLEX128][DT_COMPLEX64] = CastTask<std::complex<double>, std::complex<float>>;
  calls_[DT_COMPLEX64][DT_BFLOAT16] = CastTask<std::complex<float>, Eigen::bfloat16>;
  calls_[DT_COMPLEX128][DT_BFLOAT16] = CastTask<std::complex<double>, Eigen::bfloat16>;
}

void CastCpuKernel::SetBfloat16Map() {
  calls_[DT_BFLOAT16][DT_BFLOAT16] = CastTask<Eigen::bfloat16, Eigen::bfloat16>;
  calls_[DT_BFLOAT16][DT_BOOL] = CastTask<Eigen::bfloat16, bool>;
  calls_[DT_BFLOAT16][DT_UINT8] = CastTask<Eigen::bfloat16, uint8_t>;
  calls_[DT_BFLOAT16][DT_UINT16] = CastTask<Eigen::bfloat16, uint16_t>;
  calls_[DT_BFLOAT16][DT_UINT32] = CastTask<Eigen::bfloat16, uint32_t>;
  calls_[DT_BFLOAT16][DT_UINT64] = CastTask<Eigen::bfloat16, uint64_t>;
  calls_[DT_BFLOAT16][DT_INT8] = CastTask<Eigen::bfloat16, int8_t>;
  calls_[DT_BFLOAT16][DT_INT16] = CastTask<Eigen::bfloat16, int16_t>;
  calls_[DT_BFLOAT16][DT_INT32] = CastTask<Eigen::bfloat16, int32_t>;
  calls_[DT_BFLOAT16][DT_INT64] = CastTask<Eigen::bfloat16, int64_t>;
  calls_[DT_BFLOAT16][DT_FLOAT] = CastTask<Eigen::bfloat16, float>;
  calls_[DT_BFLOAT16][DT_DOUBLE] = CastTask<Eigen::bfloat16, double>;
  calls_[DT_BFLOAT16][DT_COMPLEX64] = CastTask<Eigen::bfloat16, std::complex<float>>;
  calls_[DT_BFLOAT16][DT_COMPLEX128] = CastTask<Eigen::bfloat16, std::complex<double>>;
  calls_[DT_BFLOAT16][DT_FLOAT16] = CastTask<Eigen::bfloat16, Eigen::half>;
  calls_[DT_BFLOAT16][DT_HIFLOAT8] = CastTask<Eigen::bfloat16, hif8>;
}

void CastCpuKernel::SetHif8Map() {
  calls_[DT_FLOAT16][DT_HIFLOAT8] = CastTask<Eigen::half, hif8>;
  calls_[DT_FLOAT][DT_HIFLOAT8] = CastTask<float, hif8>;
  calls_[DT_BFLOAT16][DT_HIFLOAT8] = CastTask<Eigen::bfloat16, hif8>;
}

uint32_t CastCpuKernel::ValidateTensors(CpuKernelContext &ctx) {
  x_tensor_ = ctx.Input(0);
  if (x_tensor_ == nullptr) {
    KERNEL_LOG_ERROR("Get input tensor failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  y_tensor_ = ctx.Output(0);
  if (y_tensor_ == nullptr) {
    KERNEL_LOG_ERROR("Get output tensor failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  x_data_size_ = x_tensor_->GetDataSize();
  y_data_size_ = y_tensor_->GetDataSize();
  if (x_data_size_ == 0) {
    KERNEL_LOG_INFO("Input data is empty, input size: [%lu]", x_data_size_);
    return KERNEL_STATUS_OK;
  }
  return KERNEL_STATUS_OK;
}

uint32_t CastCpuKernel::ValidateDataType() {
  x_data_type_ = DataType(x_tensor_->GetDataType());
  y_data_type_ = DataType(y_tensor_->GetDataType());
  KERNEL_LOG_INFO("Cast input type: [%u], output type: [%u]", x_data_type_, y_data_type_);
  int x_type_size = GetSizeByDataType(x_data_type_);
  int y_type_size = GetSizeByDataType(y_data_type_);
  if (x_type_size <= 0 || y_type_size <= 0) {
    KERNEL_LOG_ERROR("Input type size and output type size should greater than 0, "
                     "input data type: [%s], input data size: [%d], "
                     "output data type: [%s], output data size: [%d]",
                     DTypeStr(x_data_type_).c_str(), x_type_size,
                     DTypeStr(y_data_type_).c_str(), y_type_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  SetMap();
  if (calls_.find(x_data_type_) == calls_.end()) {
    KERNEL_LOG_WARN("Cast kernel input types:[%u] not support", x_data_type_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (calls_[x_data_type_].find(y_data_type_) == calls_[x_data_type_].end()) {
    KERNEL_LOG_WARN("Cast kernel output types:[%u] not support", y_data_type_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  x_data_size_ = x_data_size_ / static_cast<uint64_t>(x_type_size);
  y_data_size_ = y_data_size_ / static_cast<uint64_t>(y_type_size);
  if (x_data_size_ > y_data_size_) {
    x_data_size_ = y_data_size_;
  }
  return KERNEL_STATUS_OK;
}

uint32_t CastCpuKernel::ExecuteCast(CpuKernelContext &ctx) {
  uint64_t max_core_num = std::max(kMinCoreNum,
      aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
  if (max_core_num > x_data_size_) {
    max_core_num = x_data_size_;
  }
  if (max_core_num == 0) {
    max_core_num = kMinDataSize;
  }
  uint32_t result = aicpu::CpuKernelUtils::ParallelFor(
      ctx, static_cast<int64_t>(x_data_size_),
      static_cast<int64_t>(x_data_size_ / max_core_num),
      [this](int64_t start, int64_t end) {
        calls_[x_data_type_][y_data_type_](x_tensor_, y_tensor_, start, end);
      });
  calls_.clear();
  y_data_size_ = y_tensor_->GetDataSize();
  return result;
}

uint32_t CastCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t ret = ValidateTensors(ctx);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  ret = ValidateDataType();
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  return ExecuteCast(ctx);
}

REGISTER_CPU_KERNEL(kCast, CastCpuKernel);
}  // namespace aicpu