/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tanh_grad_aicpu.h"

#include <algorithm>
#include <complex>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *const kTanhGrad = "TanhGrad";
const int64_t kDataFloat16ParallelNum = 4096;
const int64_t kDataComplexParallelNum = 8192;
const int64_t kDataDefaultParallelNum = 32768;
}  // namespace

namespace aicpu {
uint32_t TanhGradCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalMathCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // Empty tensor: short-circuit before any dispatch / data access (preserve canndev semantics).
  Tensor *input_y = ctx.Input(kFirstInputIndex);
  Tensor *input_dy = ctx.Input(kSecondInputIndex);
  if ((input_y->NumElements() == 0) || (input_dy->NumElements() == 0)) {
    KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_OK;
  }
  uint32_t check_ret = CheckInputs(ctx);
  return (check_ret != KERNEL_STATUS_OK) ? check_ret : DispatchCompute(ctx);
}

uint32_t TanhGradCpuKernel::CheckInputs(const CpuKernelContext &ctx) const {
  Tensor *input_y = ctx.Input(kFirstInputIndex);
  Tensor *input_dy = ctx.Input(kSecondInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);
  if (input_dy->GetDataType() != output->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of input y [%s], input dy [%s], output [%s] must be the same type.",
                     DTypeStr(input_y->GetDataType()).c_str(),
                     DTypeStr(input_dy->GetDataType()).c_str(),
                     DTypeStr(output->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // Preserve canndev semantics: compute uses total=min(y,dy). Guard only against output OOB.
  int64_t compute_total = std::min(input_y->NumElements(), input_dy->NumElements());
  if (output->NumElements() < compute_total) {
    KERNEL_LOG_ERROR(
        "Output element count [%ld] is smaller than compute range [%ld] (min of input y [%ld] / dy [%ld]).",
        output->NumElements(), compute_total, input_y->NumElements(), input_dy->NumElements());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (input_y->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input y data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (input_dy->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get input dy data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (output->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Get output data failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t TanhGradCpuKernel::DispatchCompute(const CpuKernelContext &ctx) {
  Tensor *input_y = ctx.Input(kFirstInputIndex);
  AttrValue *complex_conj_attr = ctx.GetAttr("complex_conj");
  bool complex_conj = (complex_conj_attr == nullptr) ? false : complex_conj_attr->GetBool();
  auto data_type = input_y->GetDataType();
  switch (data_type) {
    case DT_FLOAT16:
      return TanhGradCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return TanhGradCompute<float>(ctx);
    case DT_DOUBLE:
      return TanhGradCompute<double>(ctx);
    case DT_COMPLEX64:
      return complex_conj ? TanhGradComputeConj<std::complex<std::float_t>>(ctx)
                          : TanhGradCompute<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return complex_conj ? TanhGradComputeConj<std::complex<std::double_t>>(ctx)
                          : TanhGradCompute<std::complex<std::double_t>>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not supported, input data type is [%s].",
                       ctx.GetOpType().c_str(), DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t TanhGradCpuKernel::TanhGradComputeConj(const CpuKernelContext &ctx) {
  int64_t input_y_total = ctx.Input(kFirstInputIndex)->NumElements();
  int64_t input_dy_total = ctx.Input(kSecondInputIndex)->NumElements();
  int64_t total = std::min(input_y_total, input_dy_total);
  T *input_y = static_cast<T *>(ctx.Input(kFirstInputIndex)->GetData());
  T *input_dy = static_cast<T *>(ctx.Input(kSecondInputIndex)->GetData());
  T *output = static_cast<T *>(ctx.Output(kFirstOutputIndex)->GetData());

  auto shard_tanhgrad = [&](size_t begin, size_t end) {
    T one_trans = static_cast<T>(1.0);
    for (size_t i = begin; i < end; i++) {
      output[i] = input_dy[i] * (std::conj(one_trans - input_y[i] * input_y[i]));
    }
  };

  if (total > kDataComplexParallelNum) {
    const auto parallel_for = aicpu::CpuKernelUtils::ParallelFor;
    int64_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
    std::int64_t per_unit_size = total / std::min(std::max(1L, cores - kResvCpuNum), total);
    KERNEL_HANDLE_ERROR(parallel_for(ctx, total, per_unit_size, shard_tanhgrad), "TanhGrad Compute failed.")
    return KERNEL_STATUS_OK;
  }
  shard_tanhgrad(0, total);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TanhGradCpuKernel::TanhGradCompute(const CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("[%s] Input[0] data size is [%lu], input[1] data size is [%lu], output data size is [%lu].",
                  ctx.GetOpType().c_str(), ctx.Input(kFirstInputIndex)->GetDataSize(),
                  ctx.Input(kSecondInputIndex)->GetDataSize(), ctx.Output(kFirstOutputIndex)->GetDataSize());

  int64_t input_y_total = ctx.Input(kFirstInputIndex)->NumElements();
  int64_t input_dy_total = ctx.Input(kSecondInputIndex)->NumElements();
  int64_t total = std::min(input_y_total, input_dy_total);
  ctx.Output(kFirstOutputIndex)->GetTensorShape()->SetDimSizes(
      ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDimSizes());
  T *input_y = static_cast<T *>(ctx.Input(kFirstInputIndex)->GetData());
  T *input_dy = static_cast<T *>(ctx.Input(kSecondInputIndex)->GetData());
  T *output = static_cast<T *>(ctx.Output(kFirstOutputIndex)->GetData());
  bool multi_core_flag = false;
  DataType input_type{ctx.Output(kFirstOutputIndex)->GetDataType()};
  switch (input_type) {
    case DT_FLOAT16:
      multi_core_flag = total > kDataFloat16ParallelNum;
      break;
    case DT_COMPLEX64:
    case DT_COMPLEX128:
      multi_core_flag = total > kDataComplexParallelNum;
      break;
    default:
      multi_core_flag = total > kDataDefaultParallelNum;
      break;
  }
  if (multi_core_flag) {
    const auto parallel_for = aicpu::CpuKernelUtils::ParallelFor;
    int64_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
    std::int64_t per_unit_size = total / std::min(std::max(1L, cores - kResvCpuNum), total);
    auto shard_tanhgrad = [&](std::int64_t begin, std::int64_t end) {
      std::int64_t length = end - begin;
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_y(input_y + begin, length, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_dy(input_dy + begin, length, 1);
      Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_z(output + begin, length, 1);
      array_z = array_dy * (T(1.0) - array_y * array_y);
    };
    KERNEL_HANDLE_ERROR(parallel_for(ctx, total, per_unit_size, shard_tanhgrad), "TanhGrad Compute failed.")
    return KERNEL_STATUS_OK;
  }
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_y(input_y, total, 1);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_dy(input_dy, total, 1);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> array_z(output, total, 1);
  array_z = array_dy * (T(1.0) - array_y * array_y);
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kTanhGrad, TanhGradCpuKernel);
}  // namespace aicpu
