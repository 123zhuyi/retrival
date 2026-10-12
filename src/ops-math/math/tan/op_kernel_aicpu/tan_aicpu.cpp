/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tan_aicpu.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kTanInputNum = 1;
const uint32_t kTanOutputNum = 1;
const int64_t kParallelNum = 1024;
const char *const kTan = "Tan";
}  // namespace

namespace aicpu {
namespace detail {
template <typename T>
inline uint32_t ComputeTanKernel(const CpuKernelContext &ctx) {
  auto input = static_cast<T *>(ctx.Input(0)->GetData());
  auto output = static_cast<T *>(ctx.Output(0)->GetData());
  const int64_t total = ctx.Input(0)->NumElements();
  const uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  if (cores == 0U) {
    KERNEL_LOG_ERROR("CPU core count should not be equal to zero.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (total > kParallelNum) {
    const int64_t per_unit_size = total / std::min(std::max(1L, static_cast<int64_t>(cores) - 2L), total);
    return aicpu::CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, [&](int64_t begin, int64_t end) {
      std::transform(input + begin, input + end, output + begin, Eigen::numext::tan<T>);
    });
  }
  std::transform(input, input + total, output, Eigen::numext::tan<T>);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline uint32_t ComputeTan(const CpuKernelContext &ctx) {
  const uint32_t ret = ComputeTanKernel<T>(ctx);
  if (ret != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Tan compute failed.");
  }
  return ret;
}

inline uint32_t TanExtraCheck(const CpuKernelContext &ctx) {
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the output [%s].",
                     DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
                     DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    KERNEL_LOG_ERROR("The data size of the input [%zu] need be the same as the output [%zu].",
                     ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")

  const std::vector<int64_t> input_dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> output_dims = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if (input_dims.size() != output_dims.size()) {
    KERNEL_LOG_ERROR("The data dim size of the input [%zu] need be the same as the output [%zu].",
                     input_dims.size(), output_dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t i = 0; i < input_dims.size(); ++i) {
    if (input_dims[i] != output_dims[i]) {
      KERNEL_LOG_ERROR("The data dim of the input need be the same as the output.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

inline uint32_t TanCheck(CpuKernelContext &ctx, uint32_t inputs_num, uint32_t outputs_num) {
  return NormalCheck(ctx, inputs_num, outputs_num) ? KERNEL_STATUS_PARAM_INVALID : TanExtraCheck(ctx);
}

inline uint32_t TanCompute(const CpuKernelContext &ctx) {
  const DataType input_type = ctx.Input(0)->GetDataType();
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeTan<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeTan<float>(ctx);
    case DT_DOUBLE:
      return ComputeTan<double>(ctx);
    case DT_COMPLEX64:
      return ComputeTan<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return ComputeTan<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
}  // namespace detail

uint32_t TanCpuKernel::Compute(CpuKernelContext &ctx) {
  return detail::TanCheck(ctx, kTanInputNum, kTanOutputNum) ? KERNEL_STATUS_PARAM_INVALID : detail::TanCompute(ctx);
}

REGISTER_CPU_KERNEL(kTan, TanCpuKernel);
}  // namespace aicpu
