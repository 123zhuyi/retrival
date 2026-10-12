/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "pow_aicpu.h"
#include <cmath>
#include <stdint.h>
#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace aicpu {
namespace {
constexpr uint32_t kOutputNum = 1U;
constexpr uint32_t kInputNum = 2U;
const char *const kPow = "Pow";
constexpr int64_t kParallelDataNum = 25 * 1024;
constexpr int64_t kParallelDataNumSameShape = 30 * 1024;

// 关于数据类型提升的逻辑，torch和cann内部有所不同
// 优先级上：complex > float > int, x1 > x2，torch会考虑x1.x2的优先级
// cann内部优先直接按照类型提升，同时会存在 uint8/int8 -> int16, double/DT_COMPLEX64 ->DT_COMPLEX128 这种提升
// 且cann对int16的类型提升不支持
// kernel在这里对上述几种提升类型都进行兼容
std::unordered_map<int32_t, std::unordered_map<int32_t, std::unordered_map<int32_t,
    std::function<uint32_t(CpuKernelContext &)>>>> kcalls {
  {DT_INT8, {{DT_INT8, {{DT_INT8, PowCpuKernel::PowCompute<int8_t, int8_t, int8_t>}}},
      {DT_INT16, {{DT_INT8, PowCpuKernel::PowCompute<int8_t, int16_t, int8_t>},
                  {DT_INT16, PowCpuKernel::PowCompute<int8_t, int16_t, int16_t>}}},
      {DT_INT32, {{DT_INT8, PowCpuKernel::PowCompute<int8_t, int32_t, int8_t>},
                  {DT_INT32, PowCpuKernel::PowCompute<int8_t, int32_t, int32_t>}}},
      {DT_INT64, {{DT_INT8, PowCpuKernel::PowCompute<int8_t, int64_t, int8_t>},
                  {DT_INT64, PowCpuKernel::PowCompute<int8_t, int64_t, int64_t>}}},
      {DT_FLOAT16, {{DT_FLOAT16, PowCpuKernel::PowCompute<int8_t, Eigen::half, Eigen::half>}}},
      {DT_FLOAT, {{DT_FLOAT, PowCpuKernel::PowCompute<int8_t, float, float>}}},
      {DT_DOUBLE, {{DT_DOUBLE, PowCpuKernel::PowCompute<int8_t, double, double>}}},
      {DT_UINT8, {{DT_INT8, PowCpuKernel::PowCompute<int8_t, uint8_t, int8_t>},
                  {DT_INT16, PowCpuKernel::PowCompute<int8_t, uint8_t, int16_t>}}},
      {DT_COMPLEX64, {{DT_COMPLEX64, PowCpuKernel::PowCompute<int8_t, std::complex<float>, std::complex<float>>}}},
      {DT_COMPLEX128, {{DT_COMPLEX128,
          PowCpuKernel::PowCompute<int8_t, std::complex<double>, std::complex<double>>}}}}},
  {DT_INT16, {{DT_INT8, {{DT_INT16, PowCpuKernel::PowCompute<int16_t, int8_t, int16_t>}}},
      {DT_INT16, {{DT_INT16, PowCpuKernel::PowCompute<int16_t, int16_t, int16_t>}}},
      {DT_INT32, {{DT_INT16, PowCpuKernel::PowCompute<int16_t, int32_t, int16_t>},
                  {DT_INT32, PowCpuKernel::PowCompute<int16_t, int32_t, int32_t>}}},
      {DT_INT64, {{DT_INT16, PowCpuKernel::PowCompute<int16_t, int64_t, int16_t>},
                  {DT_INT64, PowCpuKernel::PowCompute<int16_t, int64_t, int64_t>}}},
      {DT_FLOAT16, {{DT_FLOAT16, PowCpuKernel::PowCompute<int16_t, Eigen::half, Eigen::half>},
                    {DT_INT16, PowCpuKernel::PowCompute<int16_t, Eigen::half, int16_t>}}},
      {DT_FLOAT, {{DT_FLOAT, PowCpuKernel::PowCompute<int16_t, float, float>},
                  {DT_INT16, PowCpuKernel::PowCompute<int16_t, float, int16_t>}}},
      {DT_DOUBLE, {{DT_DOUBLE, PowCpuKernel::PowCompute<int16_t, double, double>},
                   {DT_INT16, PowCpuKernel::PowCompute<int16_t, double, int16_t>}}},
      {DT_UINT8, {{DT_INT16, PowCpuKernel::PowCompute<int16_t, uint8_t, int16_t>}}},
      {DT_COMPLEX64, {{DT_COMPLEX64, PowCpuKernel::PowCompute<int16_t, std::complex<float>, std::complex<float>>},
                      {DT_INT16, PowCpuKernel::PowCompute<int16_t, std::complex<float>, int16_t>}}},
      {DT_COMPLEX128, {{DT_COMPLEX128, PowCpuKernel::PowCompute<int16_t, std::complex<double>, std::complex<double>>},
                       {DT_INT16, PowCpuKernel::PowCompute<int16_t, std::complex<double>, int16_t>}}}}},
  {DT_INT32, {{DT_INT8, {{DT_INT32, PowCpuKernel::PowCompute<int32_t, int8_t, int32_t>}}},
      {DT_INT16, {{DT_INT32, PowCpuKernel::PowCompute<int32_t, int16_t, int32_t>}}},
      {DT_INT32, {{DT_INT32, PowCpuKernel::PowCompute<int32_t, int32_t, int32_t>}}},
      {DT_INT64, {{DT_INT32, PowCpuKernel::PowCompute<int32_t, int64_t, int32_t>},
                  {DT_INT64, PowCpuKernel::PowCompute<int32_t, int64_t, int64_t>}}},
      {DT_FLOAT16, {{DT_FLOAT16, PowCpuKernel::PowCompute<int32_t, Eigen::half, Eigen::half>}}},
      {DT_FLOAT, {{DT_FLOAT, PowCpuKernel::PowCompute<int32_t, float, float>}}},
      {DT_DOUBLE, {{DT_DOUBLE, PowCpuKernel::PowCompute<int32_t, double, double>}}},
      {DT_UINT8, {{DT_INT32, PowCpuKernel::PowCompute<int32_t, uint8_t, int32_t>}}},
      {DT_COMPLEX64, {{DT_COMPLEX64, PowCpuKernel::PowCompute<int32_t, std::complex<float>, std::complex<float>>}}},
      {DT_COMPLEX128, {{DT_COMPLEX128,
          PowCpuKernel::PowCompute<int32_t, std::complex<double>, std::complex<double>>}}}}},
  {DT_INT64, {{DT_INT8, {{DT_INT64, PowCpuKernel::PowCompute<int64_t, int8_t, int64_t>}}},
      {DT_INT16, {{DT_INT64, PowCpuKernel::PowCompute<int64_t, int16_t, int64_t>}}},
      {DT_INT32, {{DT_INT64, PowCpuKernel::PowCompute<int64_t, int32_t, int64_t>}}},
      {DT_INT64, {{DT_INT64, PowCpuKernel::PowCompute<int64_t, int64_t, int64_t>}}},
      {DT_FLOAT16, {{DT_FLOAT16, PowCpuKernel::PowCompute<int64_t, Eigen::half, Eigen::half>}}},
      {DT_FLOAT, {{DT_FLOAT, PowCpuKernel::PowCompute<int64_t, float, float>}}},
      {DT_DOUBLE, {{DT_DOUBLE, PowCpuKernel::PowCompute<int64_t, double, double>}}},
      {DT_UINT8, {{DT_INT64, PowCpuKernel::PowCompute<int64_t, uint8_t, int64_t>}}},
      {DT_COMPLEX64, {{DT_COMPLEX64, PowCpuKernel::PowCompute<int64_t, std::complex<float>, std::complex<float>>}}},
      {DT_COMPLEX128, {{DT_COMPLEX128,
          PowCpuKernel::PowCompute<int64_t, std::complex<double>, std::complex<double>>}}}}},
  {DT_FLOAT16, {{DT_INT8, {{DT_FLOAT16, PowCpuKernel::PowCompute<Eigen::half, int8_t, Eigen::half>}}},
      {DT_INT16, {{DT_FLOAT16, PowCpuKernel::PowCompute<Eigen::half, int16_t, Eigen::half>}}},
      {DT_INT32, {{DT_FLOAT16, PowCpuKernel::PowCompute<Eigen::half, int32_t, Eigen::half>}}},
      {DT_INT64, {{DT_FLOAT16, PowCpuKernel::PowCompute<Eigen::half, int64_t, Eigen::half>}}},
      {DT_FLOAT16, {{DT_FLOAT16, PowCpuKernel::PowCompute<Eigen::half, Eigen::half, Eigen::half>}}},
      {DT_FLOAT, {{DT_FLOAT16, PowCpuKernel::PowCompute<Eigen::half, float, Eigen::half>},
                  {DT_FLOAT, PowCpuKernel::PowCompute<Eigen::half, float, float>}}},
      {DT_DOUBLE, {{DT_FLOAT16, PowCpuKernel::PowCompute<Eigen::half, double, Eigen::half>},
                   {DT_DOUBLE, PowCpuKernel::PowCompute<Eigen::half, double, double>}}},
      {DT_UINT8, {{DT_FLOAT16, PowCpuKernel::PowCompute<Eigen::half, uint8_t, Eigen::half>}}},
      {DT_COMPLEX64, {{DT_COMPLEX64, PowCpuKernel::PowCompute<Eigen::half, std::complex<float>, std::complex<float>>}}},
      {DT_COMPLEX128, {{DT_COMPLEX128,
          PowCpuKernel::PowCompute<Eigen::half, std::complex<double>, std::complex<double>>}}}}},
  {DT_FLOAT, {{DT_INT8, {{DT_FLOAT, PowCpuKernel::PowCompute<float, int8_t, float>}}},
      {DT_INT16, {{DT_FLOAT, PowCpuKernel::PowCompute<float, int16_t, float>}}},
      {DT_INT32, {{DT_FLOAT, PowCpuKernel::PowCompute<float, int32_t, float>}}},
      {DT_INT64, {{DT_FLOAT, PowCpuKernel::PowCompute<float, int64_t, float>}}},
      {DT_FLOAT16, {{DT_FLOAT, PowCpuKernel::PowCompute<float, Eigen::half, float>}}},
      {DT_FLOAT, {{DT_FLOAT, PowCpuKernel::PowCompute<float, float, float>}}},
      {DT_DOUBLE, {{DT_FLOAT, PowCpuKernel::PowCompute<float, double, float>},
                   {DT_DOUBLE, PowCpuKernel::PowCompute<float, double, double>}}},
      {DT_UINT8, {{DT_FLOAT, PowCpuKernel::PowCompute<float, uint8_t, float>}}},
      {DT_COMPLEX64, {{DT_COMPLEX64, PowCpuKernel::PowCompute<float, std::complex<float>, std::complex<float>>}}},
      {DT_COMPLEX128, {{DT_COMPLEX128, PowCpuKernel::PowCompute<float, std::complex<double>, std::complex<double>>}}}}},
  {DT_DOUBLE, {{DT_INT8, {{DT_DOUBLE, PowCpuKernel::PowCompute<double, int8_t, double>}}},
      {DT_INT16, {{DT_DOUBLE, PowCpuKernel::PowCompute<double, int16_t, double>}}},
      {DT_INT32, {{DT_DOUBLE, PowCpuKernel::PowCompute<double, int32_t, double>}}},
      {DT_INT64, {{DT_DOUBLE, PowCpuKernel::PowCompute<double, int64_t, double>}}},
      {DT_FLOAT16, {{DT_DOUBLE, PowCpuKernel::PowCompute<double, Eigen::half, double>}}},
      {DT_FLOAT, {{DT_DOUBLE, PowCpuKernel::PowCompute<double, float, double>}}},
      {DT_DOUBLE, {{DT_DOUBLE, PowCpuKernel::PowCompute<double, double, double>}}},
      {DT_UINT8, {{DT_DOUBLE, PowCpuKernel::PowCompute<double, uint8_t, double>}}},
      {DT_COMPLEX64, {{DT_COMPLEX64, PowCpuKernel::PowCompute<double, std::complex<float>, std::complex<float>>},
                      {DT_COMPLEX128, PowCpuKernel::PowCompute<double, std::complex<float>, std::complex<double>>}}},
      {DT_COMPLEX128, {{DT_COMPLEX128,
          PowCpuKernel::PowCompute<double, std::complex<double>, std::complex<double>>}}}}},
  {DT_UINT8, {{DT_INT8, {{DT_UINT8, PowCpuKernel::PowCompute<uint8_t, int8_t, uint8_t>},
                         {DT_INT16, PowCpuKernel::PowCompute<uint8_t, int8_t, int16_t>}}},
      {DT_INT16, {{DT_UINT8, PowCpuKernel::PowCompute<uint8_t, int16_t, uint8_t>},
                  {DT_INT16, PowCpuKernel::PowCompute<uint8_t, int16_t, int16_t>}}},
      {DT_INT32, {{DT_UINT8, PowCpuKernel::PowCompute<uint8_t, int32_t, uint8_t>},
                  {DT_INT32, PowCpuKernel::PowCompute<uint8_t, int32_t, int32_t>}}},
      {DT_INT64, {{DT_UINT8, PowCpuKernel::PowCompute<uint8_t, int64_t, uint8_t>},
                  {DT_INT64, PowCpuKernel::PowCompute<uint8_t, int64_t, int64_t>}}},
      {DT_FLOAT16, {{DT_FLOAT16, PowCpuKernel::PowCompute<uint8_t, Eigen::half, Eigen::half>}}},
      {DT_FLOAT, {{DT_FLOAT, PowCpuKernel::PowCompute<uint8_t, float, float>}}},
      {DT_DOUBLE, {{DT_DOUBLE, PowCpuKernel::PowCompute<uint8_t, double, double>}}},
      {DT_UINT8, {{DT_UINT8, PowCpuKernel::PowCompute<uint8_t, uint8_t, uint8_t>}}},
      {DT_COMPLEX64, {{DT_COMPLEX64, PowCpuKernel::PowCompute<uint8_t, std::complex<float>, std::complex<float>>}}},
      {DT_COMPLEX128, {{DT_COMPLEX128,
          PowCpuKernel::PowCompute<uint8_t, std::complex<double>, std::complex<double>>}}}}},
  {DT_COMPLEX64, {{DT_INT8, {{DT_COMPLEX64,
                      PowCpuKernel::PowCompute<std::complex<float>, int8_t, std::complex<float>>}}},
      {DT_INT16, {{DT_COMPLEX64, PowCpuKernel::PowCompute<std::complex<float>, int16_t, std::complex<float>>}}},
      {DT_INT32, {{DT_COMPLEX64, PowCpuKernel::PowCompute<std::complex<float>, int32_t, std::complex<float>>}}},
      {DT_INT64, {{DT_COMPLEX64, PowCpuKernel::PowCompute<std::complex<float>, int64_t, std::complex<float>>}}},
      {DT_FLOAT16, {{DT_COMPLEX64, PowCpuKernel::PowCompute<std::complex<float>, Eigen::half, std::complex<float>>}}},
      {DT_FLOAT, {{DT_COMPLEX64, PowCpuKernel::PowCompute<std::complex<float>, float, std::complex<float>>}}},
      {DT_DOUBLE, {{DT_COMPLEX64, PowCpuKernel::PowCompute<std::complex<float>, double, std::complex<float>>},
                   {DT_COMPLEX128, PowCpuKernel::PowCompute<std::complex<float>, double, std::complex<double>>}}},
      {DT_UINT8, {{DT_COMPLEX64, PowCpuKernel::PowCompute<std::complex<float>, uint8_t, std::complex<float>>}}},
      {DT_COMPLEX64, {{DT_COMPLEX64,
          PowCpuKernel::PowCompute<std::complex<float>, std::complex<float>, std::complex<float>>}}},
      {DT_COMPLEX128,
          {{DT_COMPLEX64, PowCpuKernel::PowCompute<std::complex<float>, std::complex<double>, std::complex<float>>},
           {DT_COMPLEX128,
               PowCpuKernel::PowCompute<std::complex<float>, std::complex<double>, std::complex<double>>}}}}},
  {DT_COMPLEX128,
      {{DT_INT8, {{DT_COMPLEX128, PowCpuKernel::PowCompute<std::complex<double>, int8_t, std::complex<double>>}}},
       {DT_INT16, {{DT_COMPLEX128, PowCpuKernel::PowCompute<std::complex<double>, int16_t, std::complex<double>>}}},
       {DT_INT32, {{DT_COMPLEX128, PowCpuKernel::PowCompute<std::complex<double>, int32_t, std::complex<double>>}}},
       {DT_INT64, {{DT_COMPLEX128, PowCpuKernel::PowCompute<std::complex<double>, int64_t, std::complex<double>>}}},
       {DT_FLOAT16, {{DT_COMPLEX128,
           PowCpuKernel::PowCompute<std::complex<double>, Eigen::half, std::complex<double>>}}},
       {DT_FLOAT, {{DT_COMPLEX128, PowCpuKernel::PowCompute<std::complex<double>, float, std::complex<double>>}}},
       {DT_DOUBLE, {{DT_COMPLEX128, PowCpuKernel::PowCompute<std::complex<double>, double, std::complex<double>>}}},
       {DT_UINT8, {{DT_COMPLEX128, PowCpuKernel::PowCompute<std::complex<double>, uint8_t, std::complex<double>>}}},
       {DT_COMPLEX64, {{DT_COMPLEX128,
           PowCpuKernel::PowCompute<std::complex<double>, std::complex<float>, std::complex<double>>}}},
       {DT_COMPLEX128, {{DT_COMPLEX128,
           PowCpuKernel::PowCompute<std::complex<double>, std::complex<double>, std::complex<double>>}}}}}
};
} // namespace

uint32_t PowCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Pow check input and output number failed.");
  auto dtype_in1 = ctx.Input(0)->GetDataType();
  auto dtype_in2 = ctx.Input(1)->GetDataType();
  auto dtype_out = ctx.Output(0)->GetDataType();
  KERNEL_LOG_DEBUG("Pow kernel get input1 dtype[%s], input2 dtype[%s], output dtype[%s].",
                   DTypeStr(dtype_in1).c_str(), DTypeStr(dtype_in2).c_str(), DTypeStr(dtype_out).c_str());
  const auto &func_map = kcalls.find(dtype_in1);
  if (func_map != kcalls.end()) {
    const auto &funcs = func_map->second.find(dtype_in2);
    if (funcs != func_map->second.end()) {
      const auto &func = funcs->second.find(dtype_out);
      if (func != funcs->second.end()) {
        return (func->second)(ctx);
      }
    }
  }

  KERNEL_LOG_ERROR("Pow kernel input1 dtype[%s], input2 dtype[%s], output dtype[%s] not support.",
                   DTypeStr(dtype_in1).c_str(), DTypeStr(dtype_in2).c_str(), DTypeStr(dtype_out).c_str());
  return KERNEL_STATUS_PARAM_INVALID;
}

// pow不支持Eigen::half和别的类型混调
template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<(!std::is_integral<TIn1>::value || !std::is_integral<TIn2>::value) &&
    (std::is_same<TIn1, Eigen::half>::value && !std::is_same<TIn2, Eigen::half>::value), void>::type
inline PowImpl(TIn1 a, TIn2 b, TOut& output) {
  const float tmp = a;
  output = static_cast<TOut>(std::pow(tmp, b));
}

template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<(!std::is_integral<TIn1>::value || !std::is_integral<TIn2>::value) &&
    (!std::is_same<TIn1, Eigen::half>::value && std::is_same<TIn2, Eigen::half>::value), void>::type
inline PowImpl(TIn1 a, TIn2 b, TOut& output) {
  const float tmp = b;
  output = static_cast<TOut>(std::pow(a, tmp));
}

// 复数类型不支持直接转换成int
template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<(!std::is_integral<TIn1>::value || !std::is_integral<TIn2>::value) &&
    ((!std::is_same<TIn1, Eigen::half>::value && !std::is_same<TIn2, Eigen::half>::value) ||
     (std::is_same<TIn1, Eigen::half>::value && std::is_same<TIn2, Eigen::half>::value)) &&
    (std::is_same<TOut, int16_t>::value && (std::is_same<TIn2, std::complex<float>>::value ||
                                            std::is_same<TIn2, std::complex<double>>::value)), void>::type
inline PowImpl(TIn1 a, TIn2 b, TOut& output) {
  const auto tmp = std::pow(a, b).real();
  output = static_cast<TOut>(tmp);
}

template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<(!std::is_integral<TIn1>::value || !std::is_integral<TIn2>::value) &&
    ((!std::is_same<TIn1, Eigen::half>::value && !std::is_same<TIn2, Eigen::half>::value) ||
     (std::is_same<TIn1, Eigen::half>::value && std::is_same<TIn2, Eigen::half>::value)) &&
    (!(std::is_same<TOut, int16_t>::value && (std::is_same<TIn2, std::complex<float>>::value ||
                                              std::is_same<TIn2, std::complex<double>>::value))), void>::type
inline PowImpl(TIn1 a, TIn2 b, TOut& output) {
  output = static_cast<TOut>(std::pow(a, b));
}

// 事实上只有pow_impl TIn1, TIn2整形才会走这个分支
template <typename TIn1, typename TIn2, typename TOut>
inline void PowiImpl(TIn1 a, TIn2 b, TOut& output) {
  output = 1;
  while (b) {
    if (b & 1) {
      output *= a;
    }
    b = b >> 1;
    a *= a;
  }
}

template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<std::is_integral<TIn1>::value && std::is_integral<TIn2>::value, void>::type
inline PowImpl(TIn1 a, TIn2 b, TOut& output) {
  // TF 不允许有负数，pytorch允许
  if (b < 0) {
    if (a == 1) {
      output = 1;
    } else if (a == -1) {
      auto negative = (-b) % static_cast<TIn2>(2); // 2 表示偶数次幂为正
      output = negative ? -1 : 1;
    } else {
      output = 0; // 输入都为整形，out只能推导出整形，1/a 向下取整为0
    }
  } else {
    PowiImpl(a, b, output);
  }
}

template <typename TIn1, typename TIn2, typename TOut>
void PowCpuKernel::SpecialCompute(BcastShapeType type, int64_t start,
                                  int64_t end, CpuKernelContext &ctx) {
  auto input1 = reinterpret_cast<TIn1 *>(ctx.Input(0)->GetData());
  auto input2 = reinterpret_cast<TIn2 *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<TOut *>(ctx.Output(0)->GetData());
  switch (type) {
    case BcastShapeType::SAME_SHAPE:
      for (int64_t i = start; i < end; ++i) {
        PowImpl(*(input1+i), *(input2+i), *(output+i));
      }
      break;
    case BcastShapeType::X_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        PowImpl(*(input1), *(input2 + i), *(output + i));
      }
      break;
    case BcastShapeType::Y_ONE_ELEMENT:
      for (int64_t i = start; i < end; ++i) {
        PowImpl(*(input1 + i), *(input2), *(output + i));
      }
      break;
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
      break;
  }
}

template <typename TIn1, typename TIn2, typename TOut>
uint32_t PowCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  int64_t in0_elements_nums = ctx.Input(0)->NumElements();
  int64_t in1_elements_nums = ctx.Input(1)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type =
      in0_elements_nums == in1_elements_nums
          ? BcastShapeType::SAME_SHAPE
          : (in0_elements_nums == 1 ? BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT);

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));

    auto sharder_pow = [&](size_t start, size_t end) {
      SpecialCompute<TIn1, TIn2, TOut>(type, start, end, ctx);
    };

    auto per_unit_size = CeilMultiple(data_num, max_core_num);
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size,
                                    sharder_pow),
        "Pow Compute failed.")
  } else {
    SpecialCompute<TIn1, TIn2, TOut>(type, 0, data_num, ctx);
  }

  return KERNEL_STATUS_OK;
}

template <typename TIn1, typename TIn2, typename TOut>
uint32_t PowCpuKernel::BcastCompute(CpuKernelContext &ctx, Bcast &bcast) {
  auto in0 = reinterpret_cast<TIn1 *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<TIn2 *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<TOut *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    auto sharder_pow = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        auto input1 =
            in0 + bcast.GetBroadcastXIndex(i);  // i-th value of input0
        auto input2 =
            in1 + bcast.GetBroadcastYIndex(i);  // i-th value of input1
        PowImpl((*input1), (*input2), *(out + i));
      }
    };

    auto per_unit_size = CeilMultiple(data_num, max_core_num);
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size,
                                    sharder_pow),
        "Pow Compute failed.")
  } else {
    for (int64_t i = 0; i < data_num; i++) {
      auto input1 = in0 + bcast.GetBroadcastXIndex(i);  // i-th value of input0
      auto input2 = in1 + bcast.GetBroadcastYIndex(i);  // i-th value of input1
      PowImpl((*input1), (*input2), *(out + i));
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename TIn1, typename TIn2, typename TOut>
uint32_t PowCpuKernel::PowCompute(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  int64_t input0_elements_nums = input0_tensor->NumElements();

  Tensor *input1_tensor = ctx.Input(1);
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
  int64_t input1_elements_nums = input1_tensor->NumElements();
  if ((input0_tensor->GetDataSize() == 0) || (input1_tensor->GetDataSize() == 0)) {
      KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_OK;
  }
  bool no_need_bcast = (input0_shape == input1_shape) || (input0_elements_nums == 1) ||
                       (input1_elements_nums == 1);
  if (no_need_bcast) {
    return NoBcastCompute<TIn1, TIn2, TOut>(ctx);
  } else {
    Bcast bcast(input0_shape, input1_shape);
    return BcastCompute<TIn1, TIn2, TOut>(ctx, bcast);
  }
}

REGISTER_CPU_KERNEL(kPow, PowCpuKernel);
}  // namespace aicpu