/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "split_d_aicpu.h"
#include "utils/kernel_util.h"

namespace {
const char *const kSplitD = "SplitD";
}

namespace aicpu {
uint32_t SplitDCpuKernel::CheckAndInitParams(const CpuKernelContext &ctx) {
  // get input value
  Tensor *value_ptr = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(value_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input value failed.");
  value_data_ptr_ = value_ptr->GetData();
  KERNEL_CHECK_NULLPTR(value_data_ptr_, KERNEL_STATUS_PARAM_INVALID,
                       "Get input value data failed.");
  auto value_shape_ptr = value_ptr->GetTensorShape();
  KERNEL_CHECK_NULLPTR(value_shape_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input value shape failed.");
  int64_t value_dim = value_shape_ptr->GetDims();
  value_shape_vec_ = value_shape_ptr->GetDimSizes();
  data_type_ = value_ptr->GetDataType();
  value_num_ = value_ptr->NumElements();
  // get Attr num_split
  AttrValue *num_split_ptr = ctx.GetAttr("num_split");
  KERNEL_CHECK_NULLPTR(num_split_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr num_split failed.");
  num_split_ = num_split_ptr->GetInt();
  KERNEL_CHECK_FALSE(
    (num_split_ >= 1), KERNEL_STATUS_PARAM_INVALID, "Attr num_split must >= 1, but got attr num_split[%ld]", num_split_);
  // get input split_dim
  AttrValue *split_dim_ptr = ctx.GetAttr("split_dim");
  KERNEL_CHECK_NULLPTR(split_dim_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr split_dim failed.");
  split_dim_ = split_dim_ptr->GetInt();
  if (split_dim_ < 0) {
    split_dim_ += value_dim;
  }
  KERNEL_CHECK_FALSE(value_dim > split_dim_, KERNEL_STATUS_PARAM_INVALID,
                     "Dim of Input value must greater than split_dim, value dim is [%ld], split_dim is [%d].",
                     value_dim, split_dim_);
  // get input size_splits
  int64_t real_dim = value_shape_ptr->GetDimSize(split_dim_);
  KERNEL_CHECK_FALSE(((num_split_ != 0) && (real_dim % num_split_ == 0)), KERNEL_STATUS_PARAM_INVALID,
    "Split dim of Input value[%ld] must divisible by split_dim[%ld].", real_dim, num_split_);
  size_splits_ = real_dim / num_split_;
  // get output data
  output_ptr_vec_.resize(static_cast<std::size_t>(num_split_));
  for (int64_t i = 0; i < num_split_; i++) {
    Tensor *output_ptr = ctx.Output(static_cast<uint32_t>(i));
    KERNEL_CHECK_NULLPTR(output_ptr, KERNEL_STATUS_PARAM_INVALID,
                         "Get output [%ld] failed.", i);
    auto output_data_ptr = output_ptr->GetData();
    KERNEL_CHECK_NULLPTR(output_data_ptr, KERNEL_STATUS_PARAM_INVALID,
                         "Get output data [%ld] failed.", i);
    output_ptr_vec_[i] = output_data_ptr;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitDCpuKernel::SplitDWithOneOutput(const T *input_data_ptr,
                                              std::vector<T *> output_data_vec) {
  int64_t copy_size = value_num_ * static_cast<int64_t>(sizeof(T));
  auto mem_ret = BiggerMemCpy(output_data_vec[0], copy_size, input_data_ptr, copy_size);
  KERNEL_CHECK_FALSE(mem_ret, KERNEL_STATUS_PARAM_INVALID, "Memcpy size[%ld] from input value to output[0] failed.",
                     copy_size);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitDCpuKernel::SplitDWithDimZero(T *input_data_ptr,
                                            std::vector<T *> output_data_vec) {
  int64_t copy_num = value_num_ / value_shape_vec_[0];
  T *input_copy_ptr = input_data_ptr;
  int64_t copy_size_per = size_splits_ * copy_num;
  for (int64_t i = 0; i < num_split_; i++) {
    int64_t copy_size = copy_size_per * static_cast<int64_t>(sizeof(T));
    auto mem_ret = BiggerMemCpy(output_data_vec[i], copy_size, input_copy_ptr, copy_size);
    KERNEL_CHECK_FALSE(mem_ret, KERNEL_STATUS_PARAM_INVALID, "Memcpy size[%ld] from input value to output[%ld] failed.",
                       copy_size, i);
    input_copy_ptr += copy_size_per;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitDCpuKernel::SplitDCompute(T *input_data_ptr,
                                        std::vector<T *> output_data_vec) {
  // Calculate dimensions before and after split dimension
  int64_t pre_dim_size = 1;
  int32_t dim_idx = 0;
  while (dim_idx < split_dim_) {
    pre_dim_size *= value_shape_vec_[dim_idx];
    dim_idx++;
  }
  const int64_t split_dim_size = value_shape_vec_[split_dim_];
  int64_t post_dim_size = 1;
  const int32_t total_dims = static_cast<int32_t>(value_shape_vec_.size());
  dim_idx = split_dim_ + 1;
  while (dim_idx < total_dims) {
    post_dim_size *= value_shape_vec_[dim_idx];
    dim_idx++;
  }

  const int64_t chunk_elements = post_dim_size * size_splits_;
  const int64_t bytes_per_chunk = chunk_elements * static_cast<int64_t>(sizeof(T));
  const int64_t stride_per_iter = post_dim_size * split_dim_size;
  int64_t cur_offset = 0;

  int64_t out_idx = 0;
  while (out_idx < num_split_) {
    T *cur_output = output_data_vec[out_idx];
    T *cur_input = input_data_ptr + cur_offset;
    int64_t iter_count = 0;
    while (iter_count < pre_dim_size) {
      auto mem_ret = BiggerMemCpy(cur_output, bytes_per_chunk, cur_input, bytes_per_chunk);
      KERNEL_CHECK_FALSE(mem_ret, KERNEL_STATUS_PARAM_INVALID,
                         "Memcpy size[%ld] from input value to output[%ld] failed.",
                         bytes_per_chunk, out_idx);
      cur_input += stride_per_iter;
      cur_output += chunk_elements;
      iter_count++;
    }
    cur_offset += chunk_elements;
    out_idx++;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SplitDCpuKernel::DoCompute() {
  T *input_data_ptr = reinterpret_cast<T *>(value_data_ptr_);
  std::vector<T *> output_data_vec;
  output_data_vec.resize(static_cast<std::size_t>(num_split_));
  for (int64_t i = 0; i < num_split_; i++) {
    output_data_vec[i] = reinterpret_cast<T *>(output_ptr_vec_[i]);
  }

  if (num_split_ == 1) {
    KERNEL_CHECK_FALSE((SplitDWithOneOutput<T>(input_data_ptr, output_data_vec) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "SplitDWithOneOutput failed.");
    return KERNEL_STATUS_OK;
  }
  if (split_dim_ == 0) {
    KERNEL_CHECK_FALSE((SplitDWithDimZero<T>(input_data_ptr, output_data_vec) == KERNEL_STATUS_OK),
                       KERNEL_STATUS_PARAM_INVALID, "SplitDWithDimZero failed.");
    return KERNEL_STATUS_OK;
  }
  KERNEL_CHECK_FALSE((SplitDCompute<T>(input_data_ptr, output_data_vec) == KERNEL_STATUS_OK),
                     KERNEL_STATUS_PARAM_INVALID,
                     "SplitD Compute failed.");
  return KERNEL_STATUS_OK;
}

uint32_t SplitDCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("Enter SplitD Compute.");
  auto input_tensor = ctx.Input(0);
  if (input_tensor == nullptr) {
    KERNEL_LOG_ERROR("Get input value failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const uint64_t data_bytes = input_tensor->GetDataSize();
  if (data_bytes == 0) {
    KERNEL_LOG_DEBUG("Self data size is 0.");
    return KERNEL_STATUS_OK;
  }
  KERNEL_CHECK_FALSE((CheckAndInitParams(ctx) == KERNEL_STATUS_OK),
                     KERNEL_STATUS_PARAM_INVALID, "CheckAndInitParams failed.");
  uint32_t ret = KERNEL_STATUS_OK;
  if (data_type_ == DT_FLOAT16) {
    ret = DoCompute<Eigen::half>();
  } else if (data_type_ == DT_FLOAT) {
    ret = DoCompute<float>();
  } else if (data_type_ == DT_DOUBLE) {
    ret = DoCompute<double>();
  } else if (data_type_ == DT_BOOL) {
    ret = DoCompute<bool>();
  } else if (data_type_ == DT_INT8) {
    ret = DoCompute<int8_t>();
  } else if (data_type_ == DT_INT16) {
    ret = DoCompute<int16_t>();
  } else if (data_type_ == DT_INT32) {
    ret = DoCompute<int32_t>();
  } else if (data_type_ == DT_INT64) {
    ret = DoCompute<int64_t>();
  } else if (data_type_ == DT_UINT8) {
    ret = DoCompute<uint8_t>();
  } else if (data_type_ == DT_UINT16) {
    ret = DoCompute<uint16_t>();
  } else if (data_type_ == DT_UINT32) {
    ret = DoCompute<uint32_t>();
  } else if (data_type_ == DT_UINT64) {
    ret = DoCompute<uint64_t>();
  } else {
    KERNEL_LOG_WARN("Unsupport datatype[%s]", DTypeStr(data_type_).c_str());
    ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kSplitD, SplitDCpuKernel);
}  // namespace aicpu