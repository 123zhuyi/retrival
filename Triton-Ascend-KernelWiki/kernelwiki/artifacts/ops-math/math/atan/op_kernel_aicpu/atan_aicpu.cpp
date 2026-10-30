/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atan_aicpu.h"

#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kAtanInputNum{1};
const std::uint32_t kAtanOutputNum{1};
const char *const kAtan{"Atan"};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline auto ScalarAtan(const T x) -> T {
  return std::atan(x);
}

template <>
inline Eigen::half ScalarAtan(const Eigen::half x) {
  const Eigen::half val{
      static_cast<Eigen::half>(std::atan(static_cast<std::float_t>(x)))};
  return Eigen::half_impl::isnan(val) ? Eigen::half{0.0f} : val;
}

inline std::uint32_t ParallelFor(
    const CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
    const std::function<void(std::int64_t, std::int64_t)> &work) {
  constexpr std::int64_t KParallelNum = 64 * 1024;
  if (total > KParallelNum) {
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  } else {
    work(0, total);
    return KERNEL_STATUS_OK;
  }
}
template <typename T>
inline std::uint32_t ComputeAtanKernel(const CpuKernelContext &ctx) {
  T *input0{static_cast<T *>(ctx.Input(0)->GetData())};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Input(0)->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, cores - 2L), total)};
  return ParallelFor(ctx, total, per_unit_size,
                     [&](std::int64_t begin, std::int64_t end) {
                       std::transform(input0 + begin, input0 + end,
                                      output + begin, ScalarAtan<T>);
                     });
}
template <typename T>
inline std::uint32_t ComputeAtan(const CpuKernelContext &ctx) {
  std::uint32_t result{ComputeAtanKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Atan compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckAtan(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Output(0)->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR(
        "The data type of the input [%s] need be the same as the output [%s].",
        DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
        DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    KERNEL_LOG_ERROR(
        "The data size of the input [%lu] need be the same as the output [%lu].",
        ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckAtan(CpuKernelContext &ctx, std::uint32_t inputs_num,
                               std::uint32_t outputs_num) {
  return NormalCheck(ctx, inputs_num, outputs_num)
             ? KERNEL_STATUS_PARAM_INVALID
             : ExtraCheckAtan(ctx);
}

inline std::uint32_t ComputeAtan(const CpuKernelContext &ctx) {
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeAtan<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAtan<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeAtan<std::double_t>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t AtanCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckAtan(ctx, kAtanInputNum, kAtanOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::ComputeAtan(ctx);
}

REGISTER_CPU_KERNEL(kAtan, AtanCpuKernel);
}  // namespace aicpu