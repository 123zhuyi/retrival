/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "select_v2_aicpu.h"

#include <algorithm>
#include <unordered_set>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"

namespace {
const char* const kSelectV2 = "SelectV2";
const uint32_t kInputNum = 3;
const int64_t kNoBroadcastValue = 1;
const int64_t kNoRepeatElements = 2;

#define SELECTV2_COMPUTE_CASE(DTYPE, TYPE)                 \
  case (DTYPE): {                                          \
    KernelStatus result = Selectv2BuildBcast<TYPE>(ctx);   \
    if (result != KERNEL_STATUS_OK) {                      \
      KERNEL_LOG_ERROR("SelectV2 kernel compute failed."); \
      return static_cast<uint32_t>(result);                \
    }                                                      \
    break;                                                 \
  }

#define SELECTV2_DIM_CASE(RANK)                               \
  case (RANK): {                                              \
    KernelStatus result =                                     \
        SelectV2CalculateWithAlignedCheck<RANK, T>(calc_info); \
    if (result != KERNEL_STATUS_OK) {                         \
      KERNEL_LOG_ERROR("SelectV2 kernel compute failed.");    \
      return result;                                          \
    }                                                         \
    break;                                                    \
  }
}  // namespace

namespace aicpu {
uint32_t Selectv2CpuKernel::Compute(CpuKernelContext& ctx) {
  // check if input1 and input2 are of the same type
  if (Selectv2ParamCheck(ctx) != KERNEL_STATUS_OK) {
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  // choose compute function depend on dataType
  auto data_type =
      static_cast<DataType>(ctx.Input(kSecondInputIndex)->GetDataType());
  switch (data_type) {
    SELECTV2_COMPUTE_CASE(DT_FLOAT16, Eigen::half);
    SELECTV2_COMPUTE_CASE(DT_FLOAT, float);
    SELECTV2_COMPUTE_CASE(DT_DOUBLE, double);
    SELECTV2_COMPUTE_CASE(DT_INT8, int8_t);
    SELECTV2_COMPUTE_CASE(DT_INT16, int16_t);
    SELECTV2_COMPUTE_CASE(DT_INT32, int32_t);
    SELECTV2_COMPUTE_CASE(DT_INT64, int64_t);
    SELECTV2_COMPUTE_CASE(DT_UINT8, uint8_t);
    SELECTV2_COMPUTE_CASE(DT_UINT16, uint16_t);
    SELECTV2_COMPUTE_CASE(DT_UINT32, uint32_t);
    SELECTV2_COMPUTE_CASE(DT_UINT64, uint64_t);
    SELECTV2_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>);
    SELECTV2_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>);
    SELECTV2_COMPUTE_CASE(DT_BOOL, bool);
    default:
      KERNEL_LOG_ERROR(
          "[%s] Data type of input is not support, input data type is [%s].",
          ctx.GetOpType().c_str(), DTypeStr(data_type).c_str());
      return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

KernelStatus Selectv2CpuKernel::Selectv2ParamCheck(
    const CpuKernelContext& ctx) const {
  KERNEL_CHECK_FALSE((ctx.GetInputsSize() >= kInputNum),
                     KERNEL_STATUS_PARAM_INVALID,
                     "[%s] need [%u] inputs, but got [%u].",
                     ctx.GetOpType().c_str(), kInputNum, ctx.GetInputsSize());
  DataType input1_type = DT_INT8;
  DataType input2_type = DT_UINT8;
  for (uint32_t i = 0; i < kInputNum; ++i) {
    Tensor* input = ctx.Input(i);
    KERNEL_CHECK_NULLPTR(input, KERNEL_STATUS_INNER_ERROR,
                         "[%s] get input[%u] failed.", ctx.GetOpType().c_str(),
                         i);
    if (i == kSecondInputIndex) {
      input1_type = input->GetDataType();
    }
    if (i == kThirdInputIndex) {
      input2_type = input->GetDataType();
    }
    // After confirming that it is a non-empty tensor, perform this step of
    // verification
    if (!IsEmptyTensor(input)) {
      auto input_data = input->GetData();
      KERNEL_CHECK_NULLPTR(input_data, KERNEL_STATUS_PARAM_INVALID,
                           "[%s] get input[%u] tensor data is nullptr.",
                           ctx.GetOpType().c_str(), i);
    }
  }
  KERNEL_CHECK_FALSE((input1_type == input2_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input1 [%s] need be same with "
                     "input2 [%s].",
                     DTypeStr(input1_type).c_str(),
                     DTypeStr(input2_type).c_str())
  Tensor* output = ctx.Output(kFirstOutputIndex);
  KERNEL_CHECK_NULLPTR(output, KERNEL_STATUS_INNER_ERROR,
                       "[%s] get output failed.", ctx.GetOpType().c_str());
  if (!IsEmptyTensor(output)) {
    auto output_data = output->GetData();
    KERNEL_CHECK_NULLPTR(output_data, KERNEL_STATUS_PARAM_INVALID,
                         "[%s] get output tensor data is nullptr.",
                         ctx.GetOpType().c_str());
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
KernelStatus Selectv2CpuKernel::Selectv2BuildBcast(
    const CpuKernelContext& ctx) {
  Tensor* input_0 = ctx.Input(kFirstInputIndex);
  Tensor* input_1 = ctx.Input(kSecondInputIndex);
  Tensor* input_2 = ctx.Input(kThirdInputIndex);
  Tensor* output = ctx.Output(kFirstOutputIndex);

  if (input_0->GetDataSize() == 0 || input_1->GetDataSize() == 0 ||
      input_2->GetDataSize() == 0) {
    KERNEL_LOG_WARN("SelectV2 kernel input tensor is empty.");
    return KERNEL_STATUS_OK;
  }

  KERNEL_LOG_DEBUG(
      "Selectv2CpuKernel[%s], input0: size[%lu];"
      "input1: size[%lu], input2: size[%lu], output: size[%lu].",
      ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(),
      input_2->GetDataSize(), output->GetDataSize());

  SelectV2BCalcInfo calc_info;
  calc_info.input_0 = input_0;
  calc_info.input_1 = input_1;
  calc_info.input_2 = input_2;
  calc_info.output = output;

  // broadcast input
  if (Selectv2GenerateBcastInfo(calc_info) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("[%s] Generate broadcast info failed.", kSelectV2);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  SelectV2GetBcastVec(calc_info);
  int32_t rank = static_cast<int32_t>(calc_info.shape_out.size());
  switch (rank) {
    case SCALAR: {
      bool v0 = *(reinterpret_cast<const bool*>(calc_info.input_0->GetData()));
      T v1 = *(reinterpret_cast<const T*>(calc_info.input_1->GetData()));
      T v2 = *(reinterpret_cast<const T*>(calc_info.input_2->GetData()));
      T* value_out = reinterpret_cast<T*>(calc_info.output->GetData());
      *(value_out) = (v0 == true) ? v1 : v2;
      return KERNEL_STATUS_OK;
    }
    SELECTV2_DIM_CASE(ONE_DIM);
    SELECTV2_DIM_CASE(TWO_DIM);
    SELECTV2_DIM_CASE(THREE_DIM);
    SELECTV2_DIM_CASE(FOUR_DIM);
    SELECTV2_DIM_CASE(FIVE_DIM);
    SELECTV2_DIM_CASE(SIX_DIM);
    SELECTV2_DIM_CASE(SEVEN_DIM);
    SELECTV2_DIM_CASE(EIGHT_DIM);
    default:
      KERNEL_LOG_ERROR("[%s] Rank of output should less than 9 but get [%zu].",
                       ctx.GetOpType().c_str(), calc_info.shape_out.size());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <int32_t RANK, typename T>
KernelStatus Selectv2CpuKernel::SelectV2CalculateWithAlignedCheck(
    SelectV2BCalcInfo& calc_info) {
  if (AlignedCheck(calc_info)) {
    return SelectV2Calculate<RANK, T, Eigen::Aligned>(calc_info);
  }
  return SelectV2Calculate<RANK, T, Eigen::Unaligned>(calc_info);
}

bool Selectv2CpuKernel::AlignedCheck(const SelectV2BCalcInfo& calc_info) const {
  return AddrAlignedCheck(calc_info.input_0->GetData()) &&
         AddrAlignedCheck(calc_info.input_1->GetData()) &&
         AddrAlignedCheck(calc_info.input_2->GetData()) &&
         AddrAlignedCheck(calc_info.output->GetData());
}

template <int32_t RANK, typename T, int32_t OPTION>
KernelStatus Selectv2CpuKernel::SelectV2Calculate(SelectV2BCalcInfo& calc_info) {
  Eigen::TensorMap<Eigen::Tensor<bool, 1>, OPTION> input0(
      static_cast<bool*>(calc_info.input_0->GetData()),
      calc_info.input_0->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input1(
      static_cast<T*>(calc_info.input_1->GetData()),
      calc_info.input_1->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input2(
      static_cast<T*>(calc_info.input_2->GetData()),
      calc_info.input_2->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> output(
      static_cast<T*>(calc_info.output->GetData()),
      calc_info.output->GetTensorShape()->NumElements());

  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape0;
  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape1;
  Eigen::DSizes<Eigen::DenseIndex, RANK> reshape2;
  Eigen::DSizes<Eigen::DenseIndex, RANK> shape_out;
  Eigen::array<Eigen::DenseIndex, RANK> bcast0;
  Eigen::array<Eigen::DenseIndex, RANK> bcast1;
  Eigen::array<Eigen::DenseIndex, RANK> bcast2;

  for (size_t i = 0; i < static_cast<size_t>(RANK); ++i) {
    size_t index = (static_cast<size_t>(RANK) - i) - 1UL;
    reshape0[index] = calc_info.reshape_0[i];
    reshape1[index] = calc_info.reshape_1[i];
    reshape2[index] = calc_info.reshape_2[i];
    shape_out[index] = calc_info.shape_out[i];
    bcast0[index] = calc_info.bcast_0[i];
    bcast1[index] = calc_info.bcast_1[i];
    bcast2[index] = calc_info.bcast_2[i];
  }

  output.reshape(shape_out) = input0.reshape(reshape0).broadcast(bcast0).select(
      input1.reshape(reshape1).broadcast(bcast1),
      input2.reshape(reshape2).broadcast(bcast2));
  return KERNEL_STATUS_OK;
}

KernelStatus Selectv2CpuKernel::Selectv2GenerateBcastInfo(
    const SelectV2BCalcInfo& calc_info) {
  x_reshape_ = calc_info.input_0->GetTensorShape()->GetDimSizes();
  y_reshape_ = calc_info.input_1->GetTensorShape()->GetDimSizes();
  z_reshape_ = calc_info.input_2->GetTensorShape()->GetDimSizes();
  shape_out_ = calc_info.output->GetTensorShape()->GetDimSizes();

  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());
  std::reverse(z_reshape_.begin(), z_reshape_.end());

  size_t dim_num_x = x_reshape_.size();
  size_t dim_num_y = y_reshape_.size();
  size_t dim_num_z = z_reshape_.size();

  size_t max_size = std::max({dim_num_x, dim_num_y, dim_num_z});
  if (dim_num_x != max_size) {
    x_reshape_.resize(max_size, kNoBroadcastValue);
  }
  if (dim_num_y != max_size) {
    y_reshape_.resize(max_size, kNoBroadcastValue);
  }
  if (dim_num_z != max_size) {
    z_reshape_.resize(max_size, kNoBroadcastValue);
  }
  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());
  std::reverse(z_reshape_.begin(), z_reshape_.end());
  // Check if shape match
  if (shape_out_.size() != max_size) {
    KERNEL_LOG_ERROR("shape mismatch, max_dim_in=%zu, dim_out=%zu.", max_size,
                     shape_out_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t i = 0; i < max_size; ++i) {
    if (shape_out_[i] !=
        std::max({x_reshape_[i], y_reshape_[i], z_reshape_[i]})) {
      KERNEL_LOG_ERROR(
          "shape mismatch, index=%zu, dim_x=%ld, dim_y=%ld, dim_z=%ld"
          "dim_out=%ld.",
          i, x_reshape_[i], y_reshape_[i], z_reshape_[i], shape_out_[i]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  // genarate broarcast info
  if (SelectV2BcastInfo(max_size) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("SelectV2 genarate broarcast info failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

KernelStatus Selectv2CpuKernel::SelectV2BcastInfo(size_t max_size) {
  // genarate broarcast info
  x_bcast_.resize(max_size, kNoBroadcastValue);
  y_bcast_.resize(max_size, kNoBroadcastValue);
  z_bcast_.resize(max_size, kNoBroadcastValue);
  for (size_t i = 0; i < max_size; ++i) {
    // no need broadcast
    if ((x_reshape_[i] == y_reshape_[i]) && (y_reshape_[i] == z_reshape_[i])) {
      continue;
    }

    if (SelectV2BcastCheck(x_reshape_[i], y_reshape_[i], z_reshape_[i]) !=
        KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("Broadcast not support, dim_x[%zu]=%ld, dim_y[%zu]=%ld.",
                       i, x_reshape_[i], i, y_reshape_[i]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (x_reshape_[i] == kNoBroadcastValue) {
      x_bcast_[i] = std::max(y_reshape_[i], z_reshape_[i]);
    }
    if (y_reshape_[i] == kNoBroadcastValue) {
      y_bcast_[i] = std::max(x_reshape_[i], z_reshape_[i]);
    }
    if (z_reshape_[i] == kNoBroadcastValue) {
      z_bcast_[i] = std::max(x_reshape_[i], y_reshape_[i]);
    }
  }
  return KERNEL_STATUS_OK;
}

KernelStatus Selectv2CpuKernel::SelectV2BcastCheck(const int64_t x,
                                                   const int64_t y,
                                                   const int64_t z) const {
  std::unordered_set<int64_t> set_tmp{x, y, z};
  if (set_tmp.size() != kNoRepeatElements) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (x != 1 && y != 1 && z != 1) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

void Selectv2CpuKernel::SelectV2GetBcastVec(SelectV2BCalcInfo& calc_info) const {
  calc_info.reshape_0 = std::move(x_reshape_);
  calc_info.reshape_1 = std::move(y_reshape_);
  calc_info.reshape_2 = std::move(z_reshape_);
  calc_info.shape_out = std::move(shape_out_);
  calc_info.bcast_0 = std::move(x_bcast_);
  calc_info.bcast_1 = std::move(y_bcast_);
  calc_info.bcast_2 = std::move(z_bcast_);
}
REGISTER_CPU_KERNEL(kSelectV2, Selectv2CpuKernel);
}  // namespace aicpu