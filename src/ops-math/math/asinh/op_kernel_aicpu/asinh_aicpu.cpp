/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asinh_aicpu.h"

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
const std::uint32_t kAsinhInputNum{1};
const std::uint32_t kAsinhOutputNum{1};
const char *const kAsinh{"Asinh"};
const std::int64_t kAsinhParallelNum{64 * 1024};
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline T ScalarAsinh(const T x) {
  return std::asinh(x);
}

template <>
inline Eigen::half ScalarAsinh(const Eigen::half x) {
  const Eigen::half val{static_cast<Eigen::half>(std::asinh(static_cast<std::float_t>(x)))};
  return Eigen::half_impl::isnan(val) ? Eigen::half{0.0f} : val;
}

inline std::uint32_t ParallelForAsinh(const CpuKernelContext &ctx, std::int64_t total, std::int64_t per_unit_size,
                                      const std::function<void(std::int64_t, std::int64_t)> &work) {
  if (total > kAsinhParallelNum) {
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, work);
  }
  work(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline std::uint32_t ComputeAsinhKernel(const CpuKernelContext &ctx) {
  T *input0{static_cast<T *>(ctx.Input(0)->GetData())};
  T *output{static_cast<T *>(ctx.Output(0)->GetData())};
  std::int64_t total{ctx.Input(0)->NumElements()};
  std::uint32_t cores{aicpu::CpuKernelUtils::GetCPUNum(ctx)};
  std::int64_t per_unit_size{total / std::min(std::max(1L, static_cast<int64_t>(cores) - 2L), total)};
  return ParallelForAsinh(ctx, total, per_unit_size, [&](std::int64_t begin, std::int64_t end) {
    std::transform(input0 + begin, input0 + end, output + begin, ScalarAsinh<T>);
  });
}

template <typename T>
inline std::uint32_t ComputeAsinh(const CpuKernelContext &ctx) {
  std::uint32_t result{ComputeAsinhKernel<T>(ctx)};
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Asinh compute failed.");
  }
  return result;
}

inline std::uint32_t CheckAsinhDataPtr(const Tensor *tensor, const char *error_msg) {
  if (tensor->GetData() != nullptr) {
    return KERNEL_STATUS_OK;
  }
  KERNEL_LOG_ERROR("%s", error_msg);
  return KERNEL_STATUS_PARAM_INVALID;
}

inline std::uint32_t CheckAsinhOutput(const Tensor *input, const Tensor *output) {
  if (input->GetDataType() != output->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the output [%s].",
                     DTypeStr(input->GetDataType()).c_str(), DTypeStr(output->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (input->GetDataSize() != output->GetDataSize()) {
    KERNEL_LOG_ERROR("The data size of the input [%lu] need be the same as the output [%lu].",
                     input->GetDataSize(), output->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

inline std::uint32_t ExtraCheckAsinh(const CpuKernelContext &ctx) {
  Tensor *input{ctx.Input(0)};
  Tensor *output{ctx.Output(0)};
  if (CheckAsinhDataPtr(input, "Get input data failed.") != KERNEL_STATUS_OK ||
      CheckAsinhDataPtr(output, "Get output data failed.") != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return CheckAsinhOutput(input, output);
}

inline std::uint32_t CheckAsinh(CpuKernelContext &ctx, std::uint32_t inputs_num, std::uint32_t outputs_num) {
  return NormalCheck(ctx, inputs_num, outputs_num) ? KERNEL_STATUS_PARAM_INVALID : ExtraCheckAsinh(ctx);
}

inline std::uint32_t ComputeAsinhDispatch(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataSize() == 0UL) {
    KERNEL_LOG_DEBUG("The tensor x is empty.");
    return KERNEL_STATUS_OK;
  }
  DataType input_type{ctx.Input(0)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeAsinh<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAsinh<std::float_t>(ctx);
    case DT_DOUBLE:
      return ComputeAsinh<std::double_t>(ctx);
    case DT_COMPLEX64:
      return ComputeAsinh<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return ComputeAsinh<std::complex<std::double_t>>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

std::uint32_t AsinhCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::CheckAsinh(ctx, kAsinhInputNum, kAsinhOutputNum) ? KERNEL_STATUS_PARAM_INVALID
                                                                  : detail::ComputeAsinhDispatch(ctx);
}

REGISTER_CPU_KERNEL(kAsinh, AsinhCpuKernel);
}  // namespace aicpu
