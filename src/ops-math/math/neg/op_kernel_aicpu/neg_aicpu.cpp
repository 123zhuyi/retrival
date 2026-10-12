/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "neg_aicpu.h"

#include <complex>

#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

using namespace std;

namespace {
const char *const kNeg = "Neg";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;

template <typename T>
void RangeNeg(int64_t start, int64_t end, const T *input, T *out) {
  for (int64_t i = start; i < end; ++i) {
    out[i] = -input[i];
  }
}
}  // namespace

namespace aicpu {
template <typename T>
uint32_t NegCpuKernel::DoCompute(const CpuKernelContext &ctx) {
  auto input_tensor = ctx.Input(0);
  auto output_tensor = ctx.Output(0);
  DataType input_type = input_tensor->GetDataType();
  DataType output_type = output_tensor->GetDataType();
  KERNEL_CHECK_FALSE((input_type == output_type), KERNEL_STATUS_INNER_ERROR,
                     "Input data type[%s], output data type[%s] "
                     "must be same",
                     DTypeStr(input_type).c_str(),
                     DTypeStr(output_type).c_str());
  auto shard_copy = [&input_tensor, &output_tensor](int64_t start, int64_t end) {
    RangeNeg(start, end, static_cast<T *>(input_tensor->GetData()),
             static_cast<T *>(output_tensor->GetData()));
  };
  uint32_t ret = CpuKernelUtils::ParallelFor(ctx, input_tensor->NumElements(),
                                             1, shard_copy);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  return KERNEL_STATUS_OK;
}

uint32_t NegCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check Neg params failed.");
  DataType input0_data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[x] data type is [%s].", kNeg,
                   DTypeStr(input0_data_type).c_str());
  switch (input0_data_type) {
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_DOUBLE:
      return DoCompute<double>(ctx);
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    case DT_COMPLEX64:
      return DoCompute<complex<float>>(ctx);
    case DT_COMPLEX128:
      return DoCompute<complex<double>>(ctx);
    default:
      return KERNEL_STATUS_PARAM_INVALID;
  }
}
REGISTER_CPU_KERNEL(kNeg, NegCpuKernel);
}  // namespace aicpu
