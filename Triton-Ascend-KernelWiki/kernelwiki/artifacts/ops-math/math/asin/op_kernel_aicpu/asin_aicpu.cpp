/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asin_aicpu.h"

#include <algorithm>
#include <cmath>

#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kAsinInputNum{1};
const std::uint32_t kAsinOutputNum{1};
const char *const kAsin{"Asin"};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline T ScalarAsin(const T x) {
  return std::asin(x);
}

template <>
inline Eigen::half ScalarAsin(const Eigen::half x) {
  const Eigen::half val{static_cast<Eigen::half>(std::asin(static_cast<std::float_t>(x)))};
  return val;
}

template <typename T>
inline std::uint32_t ComputeAsinKernel(const CpuKernelContext &ctx) {
  using i64 = std::int64_t;
  const auto parallel_for = aicpu::CpuKernelUtils::ParallelFor;
  const auto scalar_asin = ScalarAsin<T>;
  auto input = static_cast<T *>(ctx.Input(0)->GetData());
  auto output = static_cast<T *>(ctx.Output(0)->GetData());
  i64 total = ctx.Input(0)->NumElements();
  std::uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  i64 per_unit_size{total / std::min(std::max(1L, static_cast<int64_t>(cores) - 2L), total)};
  return parallel_for(ctx, total, per_unit_size, [&](i64 begin, i64 end) {
    std::transform(input + begin, input + end, output + begin, scalar_asin);
  });
}

template <typename T>
inline std::uint32_t ComputeAsin(const CpuKernelContext &ctx) {
  std::uint32_t result = ComputeAsinKernel<T>(ctx);
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Asin compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckAsin(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the output [%s].",
                     DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
                     DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    KERNEL_LOG_ERROR("The data size of the input [%lu] need be the same as the output [%lu].",
                     ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckAsin(CpuKernelContext &ctx, std::uint32_t inputs_num, std::uint32_t outputs_num) {
  return NormalCheck(ctx, inputs_num, outputs_num) ? KERNEL_STATUS_PARAM_INVALID : ExtraCheckAsin(ctx);
}

inline std::uint32_t ComputeAsinDispatch(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataSize() == 0UL) {
    KERNEL_LOG_DEBUG("The tensor x is empty.");
    return KERNEL_STATUS_OK;
  }
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeAsin<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAsin<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeAsin<std::double_t>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t AsinCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckAsin(ctx, kAsinInputNum, kAsinOutputNum) ? KERNEL_STATUS_PARAM_INVALID
                                                               : detail::ComputeAsinDispatch(ctx);
}

REGISTER_CPU_KERNEL(kAsin, AsinCpuKernel);
}  // namespace aicpu
