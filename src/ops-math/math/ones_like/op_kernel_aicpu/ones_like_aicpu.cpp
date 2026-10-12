/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ones_like_aicpu.h"

#include <atomic>
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "log.h"
#include "utils/kernel_util.h"

namespace {
const char* const kOnesLike = "OnesLike";
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;

template <typename T>
void RangeOnesLike(int64_t start, int64_t end, T *out) {
  for (int64_t i = start; i < end; ++i) {
    out[i] = T(1);
  }
}
}  // namespace

namespace aicpu {
uint32_t OnesLikeCpuKernel::SetOnesByType(DataType input_type, void *output_data,
                                          int64_t start, int64_t end) {
  switch (input_type) {
    case DT_FLOAT16:
      RangeOnesLike(start, end, static_cast<Eigen::half *>(output_data));
      return KERNEL_STATUS_OK;
    case DT_FLOAT:
      RangeOnesLike(start, end, static_cast<float *>(output_data));
      return KERNEL_STATUS_OK;
    case DT_DOUBLE:
      RangeOnesLike(start, end, static_cast<double *>(output_data));
      return KERNEL_STATUS_OK;
    case DT_INT8:
      RangeOnesLike(start, end, static_cast<int8_t *>(output_data));
      return KERNEL_STATUS_OK;
    case DT_INT16:
      RangeOnesLike(start, end, static_cast<int16_t *>(output_data));
      return KERNEL_STATUS_OK;
    case DT_INT32:
      RangeOnesLike(start, end, static_cast<int32_t *>(output_data));
      return KERNEL_STATUS_OK;
    case DT_INT64:
      RangeOnesLike(start, end, static_cast<int64_t *>(output_data));
      return KERNEL_STATUS_OK;
    case DT_UINT8:
      RangeOnesLike(start, end, static_cast<uint8_t *>(output_data));
      return KERNEL_STATUS_OK;
    case DT_UINT16:
      RangeOnesLike(start, end, static_cast<uint16_t *>(output_data));
      return KERNEL_STATUS_OK;
    case DT_BOOL:
      RangeOnesLike(start, end, static_cast<bool *>(output_data));
      return KERNEL_STATUS_OK;
    default:
      KERNEL_LOG_ERROR("Unsupported input data type[%s]",
                       DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t OnesLikeCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check OnesLike params failed.");
  Tensor *input_tensor = ctx.Input(0);
  Tensor *output_tensor = ctx.Output(0);
  void *output_data = output_tensor->GetData();
  DataType input_type = input_tensor->GetDataType();
  std::atomic<bool> shard_ret(true);
  auto shard = [&](int64_t start, int64_t end) {
    uint32_t ret = SetOnesByType(input_type, output_data, start, end);
    if (ret != KERNEL_STATUS_OK) {
      shard_ret.store(false);
    }
  };

  uint32_t ret =
      CpuKernelUtils::ParallelFor(ctx, input_tensor->NumElements(), 1, shard);
  if ((ret != KERNEL_STATUS_OK) || (!shard_ret.load())) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kOnesLike, OnesLikeCpuKernel);
}  // namespace aicpu