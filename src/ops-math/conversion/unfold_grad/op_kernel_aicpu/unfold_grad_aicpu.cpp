/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file unfold_grad_aicpu.cpp
 * \brief
 */
 
#include "unfold_grad_aicpu.h"
#include "cpu_kernel_utils.h"
#include "log.h"
#include "Eigen/Core"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *const kUnfoldGrad = "UnfoldGrad";
const int64_t kParallelSize = 1024 * 512;
constexpr size_t gradOutIndex = 0;
constexpr size_t inputSizesIndex = 1;
constexpr size_t gradInIndex = 0;
constexpr int64_t kDimSizeMax = 8;
constexpr int64_t kDimSizeMin = 1;
}  // namespace

namespace aicpu {

#define UNFOLDGRAD_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                                    \
    uint32_t result = DoCompute<TYPE>(CTX);                          \
    if (result != KERNEL_STATUS_OK) {                                \
      KERNEL_LOG_ERROR("UnfoldGrad kernel doCompute failed."); \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }

uint32_t UnfoldGradCpuKernel::CheckParam(CpuKernelContext &ctx) {
  auto output_data_temp = ctx.Output(gradInIndex)->GetData();
  Tensor *input_tensor = ctx.Input(gradOutIndex);
  Tensor *input_sizes_tensor = ctx.Input(inputSizesIndex);
  KERNEL_CHECK_NULLPTR(output_data_temp, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get output data failed.", kUnfoldGrad);
  auto input_sizes = reinterpret_cast<int64_t *>(input_sizes_tensor->GetData());
  auto input_sizes_shape = input_sizes_tensor->GetTensorShape();
  std::vector<int64_t> input_sizes_dims = input_sizes_shape->GetDimSizes();
  KERNEL_CHECK_FALSE((input_sizes_dims.size() == 1), KERNEL_STATUS_PARAM_INVALID,
                     "%s input_sizes_dims size [%zu] must be equal to 1.", kUnfoldGrad,
                     input_sizes_dims.size());
  auto input_shape = input_tensor->GetTensorShape();
  std::vector<int64_t> input_shape_dims = input_shape->GetDimSizes();
  KERNEL_CHECK_FALSE((input_sizes_dims[0] == input_shape_dims.size() - 1), KERNEL_STATUS_PARAM_INVALID,
                     "%s input_sizes_dims size [%zu] must be equal to input_shape_dims.size() - 1.", kUnfoldGrad,
                     input_sizes_dims.size());
  for (int64_t idx = 0; idx < input_sizes_dims[0]; idx++) {
    KERNEL_CHECK_FALSE((input_sizes[idx] > 0), KERNEL_STATUS_PARAM_INVALID,
                      "%s input_sizes[%ld] must be large than 0, but is [%ld].", kUnfoldGrad,
                      idx, input_sizes[idx]);
  }
  KERNEL_CHECK_FALSE((input_shape_dims.size() <= kDimSizeMax && input_shape_dims.size() >= kDimSizeMin), KERNEL_STATUS_PARAM_INVALID,
                     "%s input_shape_dims size [%zu] must be in [1, 8].", kUnfoldGrad,
                     input_shape_dims.size());
  AttrValue *dim_ptr = ctx.GetAttr("dim");
  KERNEL_CHECK_NULLPTR(dim_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get attr dim fail.", kUnfoldGrad);
  int64_t dim = dim_ptr->GetInt();
  KERNEL_CHECK_FALSE((dim >= 0 && dim < input_shape_dims.size()),
                     KERNEL_STATUS_PARAM_INVALID, "dim should be in [0, len(input_tensor.shape)).");
  AttrValue *step_ptr = ctx.GetAttr("step");
  KERNEL_CHECK_NULLPTR(step_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get attr step fail.", kUnfoldGrad);
  int64_t step = step_ptr->GetInt();
  KERNEL_CHECK_FALSE((step > 0),
                     KERNEL_STATUS_PARAM_INVALID, "step should be large than 0.");
  AttrValue *size_ptr = ctx.GetAttr("size");
  KERNEL_CHECK_NULLPTR(size_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get attr size fail.", kUnfoldGrad);
  int64_t size = size_ptr->GetInt();
  KERNEL_CHECK_FALSE((size > 0 && size <= input_sizes[dim]),
                     KERNEL_STATUS_PARAM_INVALID, "size should be in (0, input_sizes[dim]].");
  return KERNEL_STATUS_OK;
}

template <typename T>
static void Process(T *output_data, T *input_data, int64_t *input_sizes, int64_t idx, int64_t iter_num, int64_t dim, int64_t size, int64_t step, int64_t grad_out_dim_size) {
    int64_t j = idx / (iter_num * input_sizes[dim]);
    int64_t k = (idx % (iter_num * input_sizes[dim])) % iter_num;
    int64_t idx_dim = (idx % (iter_num * input_sizes[dim])) / iter_num;

    int64_t left_fold_idx = idx_dim > size ? (idx_dim - size) / step : 0;
    if (!(left_fold_idx * step <= idx_dim && idx_dim < left_fold_idx * step + size)) {
      left_fold_idx += 1;
    }

    int64_t right_fold_idx = idx_dim / step;
    right_fold_idx = right_fold_idx >= grad_out_dim_size ? grad_out_dim_size - 1 : right_fold_idx;
    
    output_data[idx] = static_cast<T>(0.0f);
    for (int64_t fold_idx = left_fold_idx; fold_idx <= right_fold_idx; fold_idx++) {
      int64_t idx_last_dim = idx_dim - fold_idx * step;
      output_data[idx] += input_data[j * (grad_out_dim_size * iter_num * size) + (k + fold_idx * iter_num) * size + idx_last_dim];
    }
}

template <typename T>
uint32_t UnfoldGradCpuKernel::DoCompute(CpuKernelContext &ctx) {
  T *output_data = reinterpret_cast<T *>(ctx.Output(gradInIndex)->GetData());
  Tensor *input_tensor = ctx.Input(gradOutIndex);
  Tensor *input_sizes_tensor = ctx.Input(inputSizesIndex);
  auto input_data = reinterpret_cast<T *>(input_tensor->GetData());
  auto input_sizes = reinterpret_cast<int64_t *>(input_sizes_tensor->GetData());
  auto input_shape = input_tensor->GetTensorShape();
  std::vector<int64_t> input_shape_dims = input_shape->GetDimSizes();
  int64_t dim_len = input_shape->GetDims() - 1;
  AttrValue *dim_ptr = ctx.GetAttr("dim");
  AttrValue *size_ptr = ctx.GetAttr("size");
  AttrValue *step_ptr = ctx.GetAttr("step");
  int64_t dim = dim_ptr->GetInt();
  int64_t size = size_ptr->GetInt();
  int64_t step = step_ptr->GetInt();
  int64_t element_num = 1;
  int64_t iter_num = 1;
  int64_t batch_size = 1;
  int64_t grad_out_dim_size = input_shape_dims[dim];
  KERNEL_CHECK_FALSE(grad_out_dim_size == (input_sizes[dim] - size) / step + 1,
                     KERNEL_STATUS_PARAM_INVALID,
                     "grad_out.shape[dim] should be equal to ((input_sizes[dim] - size) / step + 1)");
  for (int64_t i = 0; i < dim_len; i++) {
    if (i < dim) batch_size *= input_sizes[i];
    else if (i > dim) iter_num *= input_sizes[i];
  }
  element_num = batch_size * iter_num * input_sizes[dim];

  if (element_num <= kParallelSize) {
    for (int64_t idx = 0; idx < element_num; idx++) {
      Process<T>(output_data, input_data, input_sizes, idx, iter_num, dim, size, step, grad_out_dim_size);
    }
  } else {
    auto shard_cummax = [&](int64_t start, int64_t end) {
      for (int64_t idx = start; idx < end; idx++) {
        Process<T>(output_data, input_data, input_sizes, idx, iter_num, dim, size, step, grad_out_dim_size);
      }
    };
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(
        min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > element_num) {
      max_core_num = element_num;
    }
    KERNEL_CHECK_FALSE(max_core_num > 0,
                      KERNEL_STATUS_PARAM_INVALID,
                      "max_core_num should be large than 0.");
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(
        ctx, element_num, element_num / max_core_num, shard_cummax),
        "UnfoldGrad Compute failed.")
  }

  return KERNEL_STATUS_OK;
}

uint32_t UnfoldGradCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *grad_out_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(grad_out_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get grad_out_tensor fail.", kUnfoldGrad);
  Tensor *input_sizes_tensor = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(input_sizes_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get input_sizes_tensor fail.", kUnfoldGrad);
  Tensor *grad_in_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(grad_in_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get grad_in_tensor fail.", kUnfoldGrad);
  KERNEL_CHECK_FALSE((CheckParam(ctx) == KERNEL_STATUS_OK),
                     KERNEL_STATUS_PARAM_INVALID, "CheckParam failed.");
  DataType dt = static_cast<DataType>(grad_out_tensor->GetDataType());
  switch (dt) {
    UNFOLDGRAD_COMPUTE_CASE(DT_BFLOAT16, Eigen::bfloat16, ctx)
    UNFOLDGRAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    UNFOLDGRAD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    default:
      KERNEL_LOG_WARN(
          "UnfoldGrad kernels does not support this data type [%d].", dt);
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kUnfoldGrad, UnfoldGradCpuKernel);

} // namespace aicpu
