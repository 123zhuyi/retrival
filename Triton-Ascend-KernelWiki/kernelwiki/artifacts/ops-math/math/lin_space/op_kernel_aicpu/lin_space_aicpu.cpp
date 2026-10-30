/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lin_space_aicpu.h"

#include <iostream>

#include "cpu_kernel_utils.h"
#include "securec.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *const kLinSpace = "LinSpace";
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
}

namespace aicpu {
uint32_t LinSpaceParaCheck(const CpuKernelContext &ctx, int64_t &num_value) {
  Tensor *tensor_start = ctx.Input(kFirstInputIndex);
  Tensor *tensor_stop = ctx.Input(kSecondInputIndex);
  Tensor *tensor_num = ctx.Input(kThirdInputIndex);
  Tensor *tensor_output = ctx.Output(kFirstOutputIndex);

  auto start_shape = tensor_start->GetTensorShape();
  KERNEL_CHECK_FALSE((IsScalar(start_shape->GetDimSizes()) ||
                     ((start_shape->GetDimSizes().size() == 1) &&
                     (start_shape->GetDimSize(0) == 1))), KERNEL_STATUS_PARAM_INVALID,
                     "Input[start] must be a scalar")
  auto stop_shape = tensor_stop->GetTensorShape();
  KERNEL_CHECK_FALSE((IsScalar(stop_shape->GetDimSizes()) ||
                     ((stop_shape->GetDimSizes().size() == 1) &&
                     (stop_shape->GetDimSize(0) == 1))), KERNEL_STATUS_PARAM_INVALID,
                     "Input[stop] must be a scalar")
  auto num_shape = tensor_num->GetTensorShape();
  KERNEL_CHECK_FALSE((IsScalar(num_shape->GetDimSizes()) ||
                     ((num_shape->GetDimSizes().size() == 1) &&
                     (num_shape->GetDimSize(0) == 1))), KERNEL_STATUS_PARAM_INVALID,
                     "Input[num] must be a scalar")
  KERNEL_CHECK_FALSE((tensor_start->GetDataType() == tensor_stop->GetDataType()), KERNEL_STATUS_PARAM_INVALID,
                     "start datatype != stop datatype fail.")
  KERNEL_CHECK_FALSE((tensor_start->GetDataType() == tensor_output->GetDataType()), KERNEL_STATUS_PARAM_INVALID,
                     "start datatype != output datatype fail.")

  auto num_type = static_cast<DataType>(tensor_num->GetDataType());
  switch (num_type) {
    case DT_INT32:
    {
        int32_t *num32 = static_cast<int32_t *>(tensor_num->GetData());
        num_value = static_cast<int64_t>(*num32);
        break;
    }
    case DT_INT64:
    {
        int64_t *num64 = static_cast<int64_t *>(tensor_num->GetData());
        num_value = *num64;
        break;
    }
    default:
      KERNEL_LOG_ERROR("num datatype[%d] must be DT_INT32 or DT_INT64 fail.", num_type);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((num_value > 0), KERNEL_STATUS_PARAM_INVALID, "Input[num] <= 0 fail.")
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LinSpaceCompute(const CpuKernelContext &ctx, int64_t num_value) {
    T *start_value = static_cast<T *>(ctx.Input(kFirstInputIndex)->GetData());
    T *stop_value = static_cast<T *>(ctx.Input(kSecondInputIndex)->GetData());
    T *output_value = static_cast<T *>(ctx.Output(kFirstOutputIndex)->GetData());

    output_value[0] = *start_value;
    if (num_value > 1) {
        T interval = (*stop_value - *start_value) / (num_value - 1);
        for (int64_t i = 1; i < num_value - 1; i++) {
            output_value[i] = *start_value + interval * i;
        }
        output_value[num_value - 1] = *stop_value;
    }

    return KERNEL_STATUS_OK;
}

uint32_t LinSpaceCpuKernel::Compute(CpuKernelContext &ctx) {
  int64_t num_value = 0;
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "LinSpace NormalCheck fail.");
  KERNEL_HANDLE_ERROR(LinSpaceParaCheck(ctx, num_value), "LinSpace LinSpaceParaCheck fail.");

  auto data_type = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  switch (data_type) {
    case DT_FLOAT:
      return LinSpaceCompute<float>(ctx, num_value);
    case DT_DOUBLE:
      return LinSpaceCompute<double>(ctx, num_value);
    default:
      KERNEL_LOG_ERROR("LinSpace dtype[%d] is invalid.", data_type);
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(kLinSpace, LinSpaceCpuKernel);
}  // namespace aicpu