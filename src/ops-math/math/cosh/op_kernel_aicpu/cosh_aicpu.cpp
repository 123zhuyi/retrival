/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cosh_aicpu.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>

#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const std::uint32_t kCoshInputNum{1};
const std::uint32_t kCoshOutputNum{1};
const char *const kCosh{"Cosh"};
const std::int64_t kCoshParallelNum{64 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline T ScalarCosh(const T x) {
  return std::cosh(x);
}

template <>
inline Eigen::half ScalarCosh(const Eigen::half x) {
  const Eigen::half val{
      static_cast<Eigen::half>(std::cosh(static_cast<std::float_t>(x)))};
  return Eigen::half_impl::isnan(val) ? Eigen::half{0.0f} : val;
}

inline std::uint32_t ParallelForCosh(
    const CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
    const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kCoshParallelNum) {
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  }
  work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeCoshKernel(const CpuKernelContext &ctx) {
  auto *input = static_cast<T *>(ctx.Input(0)->GetData());
  auto *output = static_cast<T *>(ctx.Output(0)->GetData());
  const std::int64_t total = ctx.Input(0)->NumElements();
  const std::uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  KERNEL_CHECK_FALSE((total != 0), KERNEL_STATUS_PARAM_INVALID,
                     "The total size of input data is zero, please check");
  const std::int64_t per_unit_size{
      total / std::min(std::max(1L, static_cast<int64_t>(cores) - 2L), total)};
  return ParallelForCosh(ctx, total, per_unit_size,
                         [&](std::int64_t begin, std::int64_t end) {
                           std::transform(input + begin, input + end,
                                          output + begin, ScalarCosh<T>);
                         });
}

template <typename T>
inline std::uint32_t ComputeCosh(const CpuKernelContext &ctx) {
  const std::uint32_t result{ComputeCoshKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Cosh compute failed.");
  }
  return result;
}

inline std::uint32_t ExtraCheckCosh(const CpuKernelContext &ctx) {
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
        "The data size of the input [%zu] need be the same as the output [%zu].",
        ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const std::vector<int64_t> input_dims_value = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> output_dims_value = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if (input_dims_value.size() != output_dims_value.size()) {
    KERNEL_LOG_ERROR("The data dim size of the input [%zu] need be the same as the output [%zu].",
                     input_dims_value.size(), output_dims_value.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t i = 0; i < input_dims_value.size(); ++i) {
    if (input_dims_value[i] != output_dims_value[i]) {
      KERNEL_LOG_ERROR("The data dim of the input need be the same as the output.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t CheckCosh(CpuKernelContext &ctx, std::uint32_t inputs_num,
                               std::uint32_t outputs_num) {
  return NormalCheck(ctx, inputs_num, outputs_num)
             ? KERNEL_STATUS_PARAM_INVALID
             : ExtraCheckCosh(ctx);
}

inline std::uint32_t CoshCompute(const CpuKernelContext &ctx) {
  const DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeCosh<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeCosh<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeCosh<std::double_t>(ctx);
    case DT_COMPLEX64:
      return ComputeCosh<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return ComputeCosh<std::complex<std::double_t>>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t CoshCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckCosh(ctx, kCoshInputNum, kCoshOutputNum)
             ? KERNEL_STATUS_PARAM_INVALID
             : detail::CoshCompute(ctx);
}

REGISTER_CPU_KERNEL(kCosh, CoshCpuKernel);
}  // namespace aicpu
