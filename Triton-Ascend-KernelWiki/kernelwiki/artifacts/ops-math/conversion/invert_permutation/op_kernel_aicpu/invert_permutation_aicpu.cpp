/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "invert_permutation_aicpu.h"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const kInvertPermutation = "InvertPermutation";
const int64_t kParallelDataNums = 64 * 1024;
}  // namespace

namespace aicpu {
uint32_t InvertPermutation::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(
      NormalCheck(ctx, kInputNum, kOutputNum),
      "InvertPermutation check input and output number failed.");
  Tensor *x = ctx.Input(0);
  int32_t dims = x->GetTensorShape()->GetDims();
  KERNEL_CHECK_FALSE((dims == 1), KERNEL_STATUS_PARAM_INVALID,
                     "InvertPermutation input dim need 1D, but get %d.", dims);
  int64_t num = x->NumElements();
  KERNEL_CHECK_FALSE((num >= 0), KERNEL_STATUS_INNER_ERROR,
                     "InvertPermutation get the num of input elements faild.");

  auto data_type = x->GetDataType();
  switch (data_type) {
    case (DT_INT32):
      return InvertPermutationCompute<int32_t>(x, num, ctx);
    case (DT_INT64):
      return InvertPermutationCompute<int64_t>(x, num, ctx);
    default:
      KERNEL_LOG_ERROR("InvertPermutation kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t InvertPermutation::InvertPermutationCompute(Tensor *x, int64_t num,
                                                     CpuKernelContext &ctx) {
  T *x_addrs = reinterpret_cast<T *>(x->GetData());
  T *y_addrs = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  size_t total = static_cast<uint64_t>(num);
  for (size_t i = 0; i < total; i++) {
    y_addrs[i] = -1;
  }
  if (total < kParallelDataNums) {
    for (size_t i = 0; i < total; i++) {
      const T data_ith = x_addrs[i];
      KERNEL_CHECK_FALSE((data_ith >= 0 && data_ith < static_cast<T>(total)), KERNEL_STATUS_PARAM_INVALID, "%ld is not between 0 and %zu.", static_cast<int64_t>(data_ith), total);
      KERNEL_CHECK_FALSE((y_addrs[data_ith] == -1), KERNEL_STATUS_PARAM_INVALID, "%ld is duplicated in the input.", static_cast<int64_t>(data_ith));
      y_addrs[data_ith] = i;
    }
  } else {
    int64_t max_core_num = 1;
    int64_t cpu_num = aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum;
    max_core_num = std::max(max_core_num, cpu_num);
    auto shard_invert_permutation = [&x_addrs, &y_addrs, &num](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        const T data_ith = x_addrs[i];
        KERNEL_CHECK_FALSE((data_ith >= 0 && data_ith < static_cast<T>(num)), KERNEL_STATUS_PARAM_INVALID, "%ld is not between 0 and %ld.", static_cast<int64_t>(data_ith), num);
        KERNEL_CHECK_FALSE((y_addrs[data_ith] == -1), KERNEL_STATUS_PARAM_INVALID, "%ld is duplicated in the input.", static_cast<int64_t>(data_ith));
        y_addrs[data_ith] = i;
      }
      return KERNEL_STATUS_OK;
    };
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, num, num / max_core_num,
                                    shard_invert_permutation),
        "InvertPermutation Compute failed.");
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kInvertPermutation, InvertPermutation);
}  // namespace aicpu