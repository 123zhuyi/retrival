/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "greater_aicpu.h"

#include <complex>
#include <iostream>

#include "cpu_kernel_utils.h"
#include "kernel_util.h"
#include "log.h"
#include "status.h"

using namespace std;

namespace {
const char* const kGreater = "Greater";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
constexpr int32_t kDim1 = 1;
constexpr int32_t kDim2 = 2;
constexpr int32_t kDim3 = 3;
constexpr int32_t kDim4 = 4;
constexpr int32_t kDim5 = 5;
constexpr int32_t kDim6 = 6;
constexpr int32_t kDim7 = 7;
constexpr int32_t kDim8 = 8;
}  // namespace

namespace aicpu {
template <typename T, int32_t RANK>
uint32_t GreaterCpuKernel::BroadcastCompute(TensorMap<T> &x, TensorMap<T> &y,
                                            TensorMap<bool> &out,
                                            const Bcast &bcast) {
  Eigen::DSizes<Eigen::DenseIndex, RANK> x_reshape;
  Eigen::DSizes<Eigen::DenseIndex, RANK> y_reshape;
  Eigen::DSizes<Eigen::DenseIndex, RANK> result_shape;
  Eigen::array<Eigen::DenseIndex, RANK> x_bcast;
  Eigen::array<Eigen::DenseIndex, RANK> y_bcast;

  for (int32_t i = 0; i < RANK; i++) {
    x_reshape[i] = bcast.XReshape()[i];
    y_reshape[i] = bcast.YReshape()[i];
    result_shape[i] = bcast.ResultShape()[i];
    x_bcast[i] = bcast.XBcast()[i];
    y_bcast[i] = bcast.YBcast()[i];
  }
  out.reshape(result_shape) = x.reshape(x_reshape).broadcast(x_bcast) >
                              y.reshape(y_reshape).broadcast(y_bcast);
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t GreaterCpuKernel::DoCompute(const CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(kFirstInputIndex);
  auto input1_tensor = ctx.Input(kSecondInputIndex);
  DataType input0_data_type = input0_tensor->GetDataType();
  DataType input1_data_type = input1_tensor->GetDataType();
  KERNEL_CHECK_FALSE(
      (input0_data_type == input1_data_type),
      KERNEL_STATUS_PARAM_INVALID,
      "Input[x1] data type[%s] and input[x2] data type[%s] must be same",
      DTypeStr(input0_data_type).c_str(), DTypeStr(input1_data_type).c_str());
  auto input0_shape_sizes = input0_tensor->GetTensorShape()->GetDimSizes();
  auto input0_elements_num = input0_tensor->NumElements();
  TensorMap<T> input0(reinterpret_cast<T *>(input0_tensor->GetData()),
                      input0_elements_num);

  auto input1_shape_sizes = input1_tensor->GetTensorShape()->GetDimSizes();
  auto input1_elements_num = input1_tensor->NumElements();
  TensorMap<T> input1(reinterpret_cast<T *>(input1_tensor->GetData()),
                      input1_elements_num);

  auto output_tensor = ctx.Output(kFirstOutputIndex);
  auto output_elements_num = output_tensor->NumElements();
  TensorMap<bool> output(reinterpret_cast<bool *>(output_tensor->GetData()),
                         output_elements_num);

  Bcast bcast(input0_shape_sizes, input1_shape_sizes);
  if (!bcast.IsValid()) {
    KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int32_t rank = static_cast<int32_t>(bcast.XReshape().size());
  switch (rank) {
    case kDim1:
      return BroadcastCompute<T, kDim1>(input0, input1, output, bcast);
    case kDim2:
      return BroadcastCompute<T, kDim2>(input0, input1, output, bcast);
    case kDim3:
      return BroadcastCompute<T, kDim3>(input0, input1, output, bcast);
    case kDim4:
      return BroadcastCompute<T, kDim4>(input0, input1, output, bcast);
    case kDim5:
      return BroadcastCompute<T, kDim5>(input0, input1, output, bcast);
    case kDim6:
      return BroadcastCompute<T, kDim6>(input0, input1, output, bcast);
    case kDim7:
      return BroadcastCompute<T, kDim7>(input0, input1, output, bcast);
    case kDim8:
      return BroadcastCompute<T, kDim8>(input0, input1, output, bcast);
    default:
      KERNEL_LOG_ERROR("Rank[%d] broadcast Compute not support.", rank);
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t GreaterCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check Greater params failed.");
  DataType input0_data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_INFO("%s op input[x1] data type is [%s].", kGreater,
                  DTypeStr(input0_data_type).c_str());
  uint32_t ret = KERNEL_STATUS_OK;
  switch (input0_data_type) {
    case DT_FLOAT:
      ret = DoCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      ret = DoCompute<double>(ctx);
      break;
    case DT_FLOAT16:
      ret = DoCompute<Eigen::half>(ctx);
      break;
    case DT_INT16:
      ret = DoCompute<int16_t>(ctx);
      break;
    case DT_INT32:
      ret = DoCompute<int32_t>(ctx);
      break;
    case DT_INT64:
      ret = DoCompute<int64_t>(ctx);
      break;
    case DT_INT8:
      ret = DoCompute<int8_t>(ctx);
      break;
    case DT_UINT16:
      ret = DoCompute<uint16_t>(ctx);
      break;
    case DT_UINT32:
      ret = DoCompute<uint32_t>(ctx);
      break;
    case DT_UINT64:
      ret = DoCompute<uint64_t>(ctx);
      break;
    case DT_UINT8:
      ret = DoCompute<uint8_t>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input[x1] data type[%s]",
                       DTypeStr(input0_data_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kGreater, GreaterCpuKernel);
}  // namespace aicpu
