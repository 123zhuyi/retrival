/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "exp_aicpu.h"

#include <cfloat>
#include <complex>

#include <unsupported/Eigen/CXX11/Tensor>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *const kExp = "Exp";
const size_t kExpInputNum = 1;
const size_t kExpOutputNum = 1;
constexpr int64_t kParallelComplexDataNums = 4 * 1024;
}  // namespace

namespace aicpu {
uint32_t ExpCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kExpInputNum, kExpOutputNum), "Check Exp params failed.");
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the output [%s]",
                     DTypeStr(ctx.Input(0)->GetDataType()).c_str(),
                     DTypeStr(ctx.Output(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (ctx.Input(0)->GetDataSize() != ctx.Output(0)->GetDataSize()) {
    KERNEL_LOG_ERROR("The data size of the input [%lu] need be the same as the output [%lu]",
                     ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (ctx.Output(0)->NumElements() == 0) {
    KERNEL_LOG_DEBUG("Exp op output shape element number is zero.");
    return KERNEL_STATUS_OK;
  }

  DataType datatype = ctx.Input(0)->GetDataType();
  if (datatype == DT_COMPLEX64) {
    return ExpComputeComplex<std::complex<float>>(ctx);
  }

  KERNEL_LOG_ERROR("Exp input type [%s] not supported by AICPU.", DTypeStr(datatype).c_str());
  return KERNEL_STATUS_PARAM_INVALID;
}

template <typename T>
uint32_t ExpCpuKernel::ExpComputeComplex(const CpuKernelContext &ctx) const {
  auto input_x = PtrToPtr<void, T>(ctx.Input(0)->GetData());
  auto output_y = PtrToPtr<void, T>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Input(0)->NumElements();
  if (data_num <= kParallelComplexDataNums) {
    for (int64_t i = 0; i < data_num; i++) {
      output_y[i] = Eigen::internal::scalar_exp_op<T>()(input_x[i]);
    }
    return KERNEL_STATUS_OK;
  }

  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(
      min_core_num,
      aicpu::CpuKernelUtils::GetCPUNum(ctx) >= kResvCpuNum ? (aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum)
                                                            : aicpu::CpuKernelUtils::GetCPUNum(ctx));
  auto shard_exp = [&input_x, &output_y](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output_y[i] = Eigen::internal::scalar_exp_op<T>()(input_x[i]);
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_exp),
                      "Exp Compute failed.");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kExp, ExpCpuKernel);
}  // namespace aicpu
