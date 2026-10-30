/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sub_aicpu.h"

#include <complex>
#include <iostream>

#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

using namespace std;

namespace {
const char *const kSub = "Sub";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
}  // namespace
namespace aicpu {
template <typename T, int32_t RANK>
uint32_t SubCpuKernel::BroadcastCompute(TensorMap<T> &x, TensorMap<T> &y,
                                        TensorMap<T> &out, const Bcast &bcast) {
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
  out.reshape(result_shape) = x.reshape(x_reshape).broadcast(x_bcast) -
                              y.reshape(y_reshape).broadcast(y_bcast);
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t SubCpuKernel::DoCompute(const CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(0);
  auto input1_tensor = ctx.Input(1);
  DataType input0_dt = input0_tensor->GetDataType();
  DataType input1_dt = input1_tensor->GetDataType();
  KERNEL_CHECK_FALSE((input0_dt == input1_dt), KERNEL_STATUS_INNER_ERROR,
                     "Input[x1] data type[%s] and input[x2] data type[%s] "
                     "must be same.",
                     DTypeStr(input0_dt).c_str(), DTypeStr(input1_dt).c_str());
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  auto input0_elements_num = input0_tensor->NumElements();
  TensorMap<T> input0(reinterpret_cast<T *>(input0_tensor->GetData()),
                      input0_elements_num);
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
  auto input1_elements_num = input1_tensor->NumElements();
  TensorMap<T> input1(reinterpret_cast<T *>(input1_tensor->GetData()),
                      input1_elements_num);
  auto output_tensor = ctx.Output(kFirstOutputIndex);
  auto output_elements_num = output_tensor->NumElements();
  TensorMap<T> output(reinterpret_cast<T *>(output_tensor->GetData()),
                      output_elements_num);

  Bcast bcast(input0_shape, input1_shape);
  if (!bcast.IsValid()) {
    KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int32_t rank = static_cast<int32_t>(bcast.XReshape().size());
  switch (rank) {
    case kRank1:
      return BroadcastCompute<T, kRank1>(input0, input1, output, bcast);
    case kRank2:
      return BroadcastCompute<T, kRank2>(input0, input1, output, bcast);
    case kRank3:
      return BroadcastCompute<T, kRank3>(input0, input1, output, bcast);
    case kRank4:
      return BroadcastCompute<T, kRank4>(input0, input1, output, bcast);
    case kRank5:
      return BroadcastCompute<T, kRank5>(input0, input1, output, bcast);
    case kRank6:
      return BroadcastCompute<T, kRank6>(input0, input1, output, bcast);
    case kRank7:
      return BroadcastCompute<T, kRank7>(input0, input1, output, bcast);
    case kRank8:
      return BroadcastCompute<T, kRank8>(input0, input1, output, bcast);
    default:
      KERNEL_LOG_ERROR("sub kernel rank exceed %d.", rank);
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t SubCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check Sub params failed.");
  DataType input0_data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[x1] data type is [%s].", kSub, DTypeStr(input0_data_type).c_str());
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
    case DT_UINT8:
      ret = DoCompute<uint8_t>(ctx);
      break;
    case DT_INT8:
      ret = DoCompute<int8_t>(ctx);
      break;
    case DT_UINT16:
      ret = DoCompute<uint16_t>(ctx);
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
    case DT_COMPLEX64:
      ret = DoCompute<complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      ret = DoCompute<complex<double>>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input[x1] data type[%s]", DTypeStr(input0_data_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kSub, SubCpuKernel);
}  // namespace aicpu
