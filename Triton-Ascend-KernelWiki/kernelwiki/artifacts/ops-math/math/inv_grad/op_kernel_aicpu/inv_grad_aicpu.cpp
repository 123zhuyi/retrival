/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inv_grad_aicpu.h"

#include <complex>
#include <cstdint>
#include "Eigen/Dense"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *const kInvGrad = "InvGrad";
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;
}  // namespace

namespace aicpu {
uint32_t InvGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "InvGrad check input and output number failed.");
  KERNEL_HANDLE_ERROR(InvGradParamCheck(ctx), "InvGrad check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  uint32_t result;
  switch (data_type) {
    case (DT_FLOAT16):
      result = InvGradCompute<Eigen::half>(ctx);
      break;
    case (DT_FLOAT):
      result = InvGradCompute<float>(ctx);
      break;
    case (DT_DOUBLE):
      result = InvGradCompute<double>(ctx);
      break;
    case (DT_COMPLEX64):
      result = InvGradCompute<std::complex<float>>(ctx);
      break;
    case (DT_COMPLEX128):
      result = InvGradCompute<std::complex<double>>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("InvGrad kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("InvGrad kernel compute failed.");
  }
  return result;
}

uint32_t InvGradCpuKernel::InvGradParamCheck(const CpuKernelContext &ctx) const {
  // the non null of input_0, input_1, output has been verified in NormalCheck
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "input1 [%s].",
                     DTypeStr(input0_type).c_str(),
                     DTypeStr(input1_type).c_str())
  KERNEL_LOG_DEBUG(
      "InvGradCpuKernel[%s], input0: size[%lu];"
      "input1: size[%lu], output: size[%lu].",
      ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(),
      output->GetDataSize());

  return KERNEL_STATUS_OK;
}

// special compute is used in the following situations.
// 1. the shapes of input1 and input2 are the same
// 2. input1 is a 1D tensor with only one element or input1 is scalar
// 3. input2 is a 1D tensor with only one element or input2 is scalar
// 4. the shapes of input1 and input2 are different
template <typename T>
void InvGradCpuKernel::SpecialCompute(int64_t start, int64_t end, const T *input1,
                                      const T *input2, T *output) const {
  for (int64_t i = start; i < end; ++i) {
    *(output + i) =
        *(input1 + i) * *(input1 + i) * *(input2 + i) * static_cast<T>(-1);
  }
}

template <typename T>
uint32_t InvGradCpuKernel::NoBcastCompute(const CpuKernelContext &ctx) const {
  auto in0 = static_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = static_cast<T *>(ctx.Input(1)->GetData());
  auto out = static_cast<T *>(ctx.Output(0)->GetData());
  int64_t in0_elements_nums = ctx.Input(0)->NumElements();
  int64_t data_num = in0_elements_nums;

  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(
        min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, static_cast<int64_t>(4));  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_invgrad = [this, &in0, &in1, &out](size_t start, size_t end) {
      SpecialCompute<T>(start, end, in0, in1, out);
    };

    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num,
                                    sharder_invgrad),
        "InvGrad Compute failed.")
  } else {
    SpecialCompute<T>(0, data_num, in0, in1, out);
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t InvGradCpuKernel::InvGradCompute(const CpuKernelContext &ctx) const {
  Tensor *input0_tensor = ctx.Input(0);
  int64_t input0_elements_nums = input0_tensor->NumElements();

  Tensor *input1_tensor = ctx.Input(1);
  int64_t input1_elements_nums = input1_tensor->NumElements();
  // elements numbers check
  if (input0_elements_nums != input1_elements_nums) {
    KERNEL_LOG_WARN("Invalid element numbers, got[%d] and [%d]",
                    static_cast<int32_t>(input0_elements_nums),
                    static_cast<int32_t>(input1_elements_nums));
    return KERNEL_STATUS_PARAM_INVALID;
  } else {
    return NoBcastCompute<T>(ctx);
  }
}

REGISTER_CPU_KERNEL(kInvGrad, InvGradCpuKernel);
}  // namespace aicpu
