/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "reduce_mean_aicpu.h"
#include "reduce_op.h"
#include "utils/kernel_util.h"

namespace aicpu {
namespace {
const char *const kReduceMean = "ReduceMean";
const uint32_t kReduceMeanInputNum = 2;
const uint32_t kReduceMeanOutputNum = 1;
constexpr size_t kAxesIdx = 1U;

template <typename T>
void GetValueFromData(const T *tensor_data, size_t size, QuickVector &vec) {
  vec.SetDimNum(size);
  for (size_t i = 0U; i < size; i++) {
    vec[i] = tensor_data[i];
  }
}

static std::map<
    DataType, std::function<uint32_t(CpuKernelContext *, const QuickVector &)>>
    kreduce_mean_calls = {
        {DT_FLOAT16,
         ReductionOp<Eigen::half, Eigen::internal::MeanReducer<Eigen::half>>},
        {DT_FLOAT, ReductionOp<float, Eigen::internal::MeanReducer<float>>},
        {DT_DOUBLE,
         ReductionOp<double, Eigen::internal::MeanReducer<double>>},
        {DT_INT8,
         ReductionOp<int8_t, Eigen::internal::MeanReducer<int8_t>>},
        {DT_INT16,
         ReductionOp<int16_t, Eigen::internal::MeanReducer<int16_t>>},
        {DT_INT32,
         ReductionOp<int32_t, Eigen::internal::MeanReducer<int32_t>>},
        {DT_INT64,
         ReductionOp<int64_t, Eigen::internal::MeanReducer<int64_t>>},
        {DT_UINT8,
         ReductionOp<uint8_t, Eigen::internal::MeanReducer<uint8_t>>},
        {DT_UINT16,
         ReductionOp<uint16_t, Eigen::internal::MeanReducer<uint16_t>>},
        {DT_UINT32,
         ReductionOp<uint32_t, Eigen::internal::MeanReducer<uint32_t>>},
        {DT_UINT64,
         ReductionOp<uint64_t, Eigen::internal::MeanReducer<uint64_t>>},
        {DT_COMPLEX64,
         ReductionOp<std::complex<float>, ComplexFloatMeanReducer>},
        {DT_COMPLEX128,
         ReductionOp<std::complex<double>, ComplexDoubleMeanReducer>}
};
}  // namespace

uint32_t ReduceMeanCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kReduceMeanInputNum, kReduceMeanOutputNum),
                      "[%s] check input and output failed.", kReduceMean);

  auto axes_tensor = ctx.Input(kAxesIdx);
  auto axes_type = axes_tensor->GetDataType();
  auto axes_shape = axes_tensor->GetTensorShape();
  KERNEL_CHECK_NULLPTR(axes_shape, KERNEL_STATUS_PARAM_INVALID, "get axes_shape failed.");
  QuickVector axes;
  if (axes_type == DT_INT32) {
    auto axes_data = static_cast<int32_t *>(axes_tensor->GetData());
    GetValueFromData<int32_t>(axes_data, axes_tensor->NumElements(), axes);
  } else if (axes_type == DT_INT64) {
    auto axes_data = static_cast<int64_t *>(axes_tensor->GetData());
    GetValueFromData<int64_t>(axes_data, axes_tensor->NumElements(), axes);
  } else {
    KERNEL_LOG_ERROR("axes type[%d] not support, only support DT_INT32 or DT_INT64.", axes_type);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto input_dtype = ctx.Input(kXIdx)->GetDataType();
  const auto &func = kreduce_mean_calls.find(input_dtype);
  if (func != kreduce_mean_calls.end()) {
    return (func->second)(&ctx, axes);
  } else {
    KERNEL_LOG_ERROR("input[0] type[%d] not support", input_dtype);
    return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(kReduceMean, ReduceMeanCpuKernel);
}  // namespace aicpu