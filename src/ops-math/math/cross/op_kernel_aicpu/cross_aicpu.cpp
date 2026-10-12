/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cross_aicpu.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const int64_t KNum2 = 2;
const int64_t kParallelNum = static_cast<int64_t>(2 * 1024);
const char *const kCross = "Cross";
const int64_t kThreeNum = 3;
const int64_t kDimDef = static_cast<int64_t>(-65530);
#define CROSS_COMPUTE_CASE(DTYPE, TYPE, CTX)                 \
  case (DTYPE): {                                            \
    uint32_t result = CrossBcastCompute<TYPE>(CTX);               \
    if (result != static_cast<uint32_t>(KERNEL_STATUS_OK)) { \
      KERNEL_LOG_ERROR("Cross kernel compute failed.");      \
      return result;                                         \
    }                                                        \
    break;                                                   \
  }
}  // namespace

namespace aicpu {
KernelStatus CrossCpuKernel::GetDimAndCheck(const CpuKernelContext &ctx, const BCalcInfo &calc_info) {
  auto &shape_out = calc_info.shape_out;
  int64_t rank = shape_out.size();
  AttrValue *dim_attr = ctx.GetAttr("dim");
  dim_ = (dim_attr == nullptr) ? kDimDef : dim_attr->GetInt();
  if (dim_ == kDimDef) {
    for (int64_t i = 0; i < rank; i++) {
      if (shape_out[i] == kThreeNum) {
        dim_ = i;
        break;
      }
      if ((i == (rank - 1)) && (shape_out[i] != kThreeNum)) {
        KERNEL_LOG_ERROR("The size of inputs dim should be 3,but got [%ld].",
                         shape_out[i]);
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  }
  if ((dim_ < (-rank)) || (dim_ > (rank - 1))) {
    KERNEL_LOG_ERROR("dim should between [%ld] and [%ld],but got [%ld].",
                     -rank,
                     rank - 1, dim_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (dim_ < 0) {
    dim_ = rank + dim_;
  }
  if (shape_out[dim_] != kThreeNum) {
    KERNEL_LOG_ERROR("The size of inputs dim should be 3,but got [%ld].",
                     shape_out[dim_]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t CrossCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Cross check input and output number failed.");
  auto output_type = ctx.Output(0)->GetDataType();
  switch (output_type) {
    CROSS_COMPUTE_CASE(DT_DOUBLE, double, ctx);
    CROSS_COMPUTE_CASE(DT_FLOAT, float, ctx);
    CROSS_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx);
    CROSS_COMPUTE_CASE(DT_INT8, int8_t, ctx);
    CROSS_COMPUTE_CASE(DT_INT16, int16_t, ctx);
    CROSS_COMPUTE_CASE(DT_INT32, int32_t, ctx);
    CROSS_COMPUTE_CASE(DT_INT64, int64_t, ctx);
    CROSS_COMPUTE_CASE(DT_UINT8, uint8_t, ctx);
    CROSS_COMPUTE_CASE(DT_UINT16, uint16_t, ctx);
    CROSS_COMPUTE_CASE(DT_UINT32, uint32_t, ctx);
    CROSS_COMPUTE_CASE(DT_UINT64, uint64_t, ctx);
    CROSS_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx);
    CROSS_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx);
    default:
      KERNEL_LOG_ERROR("Cross kernel data type [%s] not support.",
                       DTypeStr(output_type).c_str());
      return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

std::vector<int64_t> CrossCpuKernel::StridesCompute(
    int64_t stride_length, std::vector<int64_t> data_dims) {
  std::vector<int64_t> stride;
  int64_t stride_tmp = 1;
  for (int64_t i = stride_length - static_cast<int64_t>(1); i > static_cast<int64_t>(-1); i--) {
    (void)stride.insert(stride.begin(), stride_tmp);
    stride_tmp *= data_dims[i];
  }

  return stride;
}

void CrossCpuKernel::InitStride(const BCalcInfo &calc_info) {
  const int64_t n = calc_info.shape_out.size();
  stride_ = StridesCompute(n, calc_info.shape_out);
  data_stride_ = stride_[dim_];
}

template <typename T1>
void CrossCpuKernel::SetResult(const BCalcInfo &calc_info,
                               int64_t data_start_bcast) const {
  auto input1_data_addr = static_cast<const T1 *>(calc_info.input_0->GetData());
  auto input2_data_addr = static_cast<const T1 *>(calc_info.input_1->GetData());
  auto output_data_addr = static_cast<T1 *>(calc_info.output->GetData());
  
  int64_t input1_data_start = calc_info.x_indexes[data_start_bcast];
  int64_t input2_data_start = calc_info.y_indexes[data_start_bcast];
  int64_t input1_data_stride1 = calc_info.x_indexes[data_start_bcast + data_stride_];
  int64_t input1_data_stride2 = calc_info.x_indexes[data_start_bcast + KNum2 * data_stride_];
  int64_t input2_data_stride1 = calc_info.y_indexes[data_start_bcast + data_stride_];
  int64_t input2_data_stride2 = calc_info.y_indexes[data_start_bcast + KNum2 * data_stride_];

  output_data_addr[data_start_bcast] =
      input1_data_addr[input1_data_stride1] * input2_data_addr[input2_data_stride2] -
      input1_data_addr[input1_data_stride2] * input2_data_addr[input2_data_stride1];
  output_data_addr[data_start_bcast + data_stride_] =
      input1_data_addr[input1_data_stride2] * input2_data_addr[input2_data_start] -
      input1_data_addr[input1_data_start] * input2_data_addr[input2_data_stride2];
  output_data_addr[data_start_bcast + KNum2 * data_stride_] =
      input1_data_addr[input1_data_start] * input2_data_addr[input2_data_stride1] -
      input1_data_addr[input1_data_stride1] * input2_data_addr[input2_data_start];
}

template <typename T1>
void CrossCpuKernel::ParallelCompute(const BCalcInfo &calc_info, int64_t start, int64_t end) {
  auto &data_shape = calc_info.shape_out;
  const int64_t data_dim = data_shape.size();
  std::vector<int64_t> position_in_dims(data_dim);
  int64_t index_in_curr_dim = start;
  int64_t data_start = 0;

  for (int64_t i = 0; i < data_dim; i++) {
    if (i == dim_) {
      continue;
    }
    position_in_dims[i] = index_in_curr_dim % data_shape[i];
    data_start += (index_in_curr_dim % data_shape[i]) * stride_[i];
    index_in_curr_dim = index_in_curr_dim / data_shape[i];
  }
  while (start < end) {
    SetResult<T1>(calc_info, data_start);
    start++;
    for (int64_t i = 0; i < data_dim; i++) {
      if (i == dim_) {
        continue;
      }
      position_in_dims[i]++;
      data_start += stride_[i];
      if ((position_in_dims[i] == data_shape[i]) && (i != data_dim - 1)) {
        data_start -= position_in_dims[i] * stride_[i];
        position_in_dims[i] = 0;
      } else {
        break;
      }
    }
  }
}

template <typename T1>
uint32_t CrossCpuKernel::CrossCompute(const CpuKernelContext &ctx, const BCalcInfo &calc_info) {
  KERNEL_HANDLE_ERROR(static_cast<uint32_t>(GetDimAndCheck(ctx, calc_info)),
                      "[%s] check params failed.", kCross);
  int64_t total = ctx.Output(0)->NumElements() / 3;
  InitStride(calc_info);
  auto cross_shard = [this, &calc_info](int64_t start, int64_t end) {
    ParallelCompute<T1>(calc_info, start, end);
  };
  if (total > kParallelNum) {
    const int64_t max_core_num = std::max(
        static_cast<int64_t>(1),
        static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2));
    const int64_t per_unit_size = total / std::min(total, max_core_num);
    KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, total, per_unit_size, cross_shard),
        "Cross compute failed");
  } else {
    cross_shard(0, total);
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

template <typename T1>
uint32_t CrossCpuKernel::CrossBcastCompute(const CpuKernelContext &ctx) {
  BCalcInfo calc_info;
  calc_info.input_0 = ctx.Input(0);
  calc_info.input_1 = ctx.Input(1);
  calc_info.output = ctx.Output(0);
  DataType input0_type = calc_info.input_0->GetDataType();
  DataType input1_type = calc_info.input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "DataType of x1 [%d] should be same as x2 [%d].",
                     static_cast<int32_t>(input0_type), static_cast<int32_t>(input1_type))
  KERNEL_LOG_INFO("CpuKernel[%s], input x1 : addr[%p], size[%lu];"
      "input x2: addr[%p], size[%lu];"
      "output: addr[%p], size[%lu].",
      ctx.GetOpType().c_str(), calc_info.input_0->GetData(),
      calc_info.input_0->GetDataSize(), calc_info.input_1->GetData(),
      calc_info.input_1->GetDataSize(), calc_info.output->GetData(),
      calc_info.output->GetDataSize());

  Bcast bcast;
  KERNEL_HANDLE_ERROR(bcast.GenerateBcastInfo(calc_info),
                      "Generate broadcast info failed.")
  bcast.BCastIndexes(calc_info.x_indexes, calc_info.y_indexes);
  bcast.GetBcastVec(calc_info);

  return CrossCompute<T1>(ctx, calc_info);
}

REGISTER_CPU_KERNEL(kCross, CrossCpuKernel);
}  // namespace aicpu
