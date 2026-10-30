/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "search_sorted_aicpu.h"

#include <algorithm>
#include <atomic>
#include <numeric>
#include <utility>

#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "utils/kernel_util.h"

namespace {
const char *const kSearchSorted = "SearchSorted";
constexpr size_t kOutputSize = 1;
constexpr int64_t kParallelDataNum = 8 * 1024;
}  // namespace

namespace aicpu {
template <typename S>
int64_t CustomizedLowerBound(int64_t seq_start, int64_t seq_end, const S key,
                             const S *sequence, int64_t *sort) {
  const int64_t orig_start = seq_start;
  while (seq_start < seq_end) {
    const int64_t mid = seq_start + ((seq_end - seq_start) >> 1);
    // 因为sorter 表征的是ND
    // tensor的相对排序，即最后一维的排序，因此需要在orig_start的位置上进行偏移
    const S mid_val = sort ? sequence[sort[mid] + orig_start] : sequence[mid];
    if (!(mid_val >= key)) {
      seq_start = mid + 1;
    } else {
      seq_end = mid;
    }
  }
  return seq_start;
}

template <typename S>
int64_t CustomizedUpperBound(int64_t seq_start, int64_t seq_end, const S key,
                             const S *sequence, int64_t *sort) {
  const int64_t orig_start = seq_start;
  while (seq_start < seq_end) {
    const int64_t mid = seq_start + ((seq_end - seq_start) >> 1);
    // 因为sorter 表征的是ND
    // tensor的相对排序，即最后一维的排序，因此需要在orig_start的位置上进行偏移
    const S mid_val = sort ? sequence[sort[mid] + orig_start] : sequence[mid];
    if (!(mid_val > key)) {
      seq_start = mid + 1;
    } else {
      seq_end = mid;
    }
  }
  return seq_start;
}

inline bool matched_before_last_dim(const std::vector<int64_t> &sequence_dims,
                                    const std::vector<int64_t> &values_dims) {
  if (sequence_dims.size() != values_dims.size()) {
    return false;
  }
  for (size_t i = 0; (i + 1) < sequence_dims.size(); ++i) {
    if (sequence_dims[i] != values_dims[i]) {
      return false;
    }
  }
  return true;
}

inline KernelStatus CheckShape(const std::vector<int64_t> &sequence_shape,
                               const std::vector<int64_t> &values_shape) {
  size_t dim_num = sequence_shape.size();
  // 除过最后一个维度的dims可以不相等，其它的应该都相等
  if (!(dim_num == 1 ||
        matched_before_last_dim(sequence_shape, values_shape))) {
    KERNEL_LOG_ERROR(
        "sequence should be 1D tensor or the first N-1 dims of sequence and "
        "value must match, but we got sequence shape is [%s], value shape is "
        "[%s].",
        VectorToString(sequence_shape).c_str(),
        VectorToString(values_shape).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

KernelStatus SearchSortedKernel::GetInputAndCheck(const CpuKernelContext &ctx) {
  AttrValue *right = ctx.GetAttr("right");
  KERNEL_CHECK_NULLPTR(right, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[right] failed.");
  right_ = right->GetBool();
  sequence_t_ = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(sequence_t_, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[0] failed");
  sequence_dtype_ = static_cast<DataType>(sequence_t_->GetDataType());
  auto sequence_shape = sequence_t_->GetTensorShape();
  auto sequence_dims = sequence_shape->GetDimSizes();

  values_t_ = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(values_t_, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[1] failed");
  values_dtype_ = static_cast<DataType>(values_t_->GetDataType());
  auto values_shape = values_t_->GetTensorShape();
  auto values_dims = values_shape->GetDimSizes();
  KERNEL_LOG_DEBUG("sequence shape is [%s], value shape is [%s].",
                   VectorToString(sequence_dims).c_str(),
                   VectorToString(values_dims).c_str());
  sorter_t_ = ctx.Input(2);  // 2 is for sorter
  if (sorter_t_ != nullptr) {
    if (sorter_t_->GetDataType() != DT_INT64) {
      KERNEL_LOG_ERROR("sorter type muse be DT_INT64, but got %s",
                       DTypeStr(sorter_t_->GetDataType()).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (sorter_t_->NumElements() != sequence_t_->NumElements()) {
      KERNEL_LOG_ERROR(
          "sorted_sequence and sorter must have the same size, but got "
          "sorted_sequence size is [%ld], sorter size is [%ld]",
          sequence_t_->NumElements(), sorter_t_->NumElements());
      return KERNEL_STATUS_PARAM_INVALID;
    }

    int64_t *sorter = static_cast<int64_t *>(sorter_t_->GetData());
    int64_t sorter_max =
        *(std::max_element(sorter, sorter + sorter_t_->NumElements()));
    int64_t sorter_min =
        *(std::min_element(sorter, sorter + sorter_t_->NumElements()));
    int64_t last_dim = sequence_dims.back();
    if (!(sorter_min >= 0 && sorter_max < last_dim)) {
      KERNEL_LOG_ERROR(
          "sorter index out of range, sorter_max is [%ld], sorter_min is "
          "[%ld], sequence last dim is [%ld]",
          sorter_max, sorter_min, last_dim);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    KERNEL_LOG_DEBUG("optional input sorter is nullptr.");
  }

  output_t_ = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_t_, KERNEL_STATUS_PARAM_INVALID,
                       "Get output:[1] failed");
  output_dtype_ = static_cast<DataType>(output_t_->GetDataType());
  // outputs: positions
  if (ctx.GetOutputsSize() != kOutputSize) {
    KERNEL_LOG_ERROR(
        "Output number is: [%d], but SearchSorted needs [%zu] outputs.",
        ctx.GetOutputsSize(), kOutputSize);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (output_t_->NumElements() != values_t_->NumElements()) {
    KERNEL_LOG_ERROR(
        "The output size should be same with values size, but got output "
        "numelements is [%ld], values output numelements is [%ld]",
        output_t_->NumElements(), values_t_->NumElements());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return CheckShape(sequence_dims, values_dims);
}

template <typename S, typename T>
KernelStatus CalSearchSorted(bool right, const Tensor *sequence_t,
                             const Tensor *values_t, const Tensor *sort_t,
                             const Tensor *output_t,
                             const CpuKernelContext &ctx) {
  // Empty tensor
  if (sequence_t->NumElements() == 0) {
    KERNEL_LOG_DEBUG("sequence size is zero.");
    return KERNEL_STATUS_OK;
  }
  if (values_t->NumElements() == 0) {
    KERNEL_LOG_DEBUG("values size is zero.");
    return KERNEL_STATUS_OK;
  }
  auto sequence = PtrToPtr<void, S>(sequence_t->GetData());
  auto values = PtrToPtr<void, S>(values_t->GetData());
  auto output = PtrToPtr<void, T>(output_t->GetData());
  auto values_shape = values_t->GetTensorShape();
  auto values_dims = values_shape->GetDimSizes();
  int64_t search_repeat = values_dims.back();
  if (search_repeat == 0) {
    KERNEL_LOG_ERROR("values last dim can't be zero");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto sequence_shape = sequence_t->GetTensorShape();
  auto sequence_dims = sequence_shape->GetDimSizes();
  size_t seq_dim = sequence_dims.size();
  int64_t search_len = sequence_dims.back();
  if (search_len <= 0) {
    KERNEL_LOG_ERROR(
        "sequence last dim should be larger than zero, but got [%ld]",
        search_len);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t *sort = nullptr;
  if (sort_t != nullptr) {
    sort = static_cast<int64_t *>(sort_t->GetData());
  }
  auto task = [&seq_dim, &search_repeat, &search_len, &values, &sequence, &sort, &output, &right](
                  int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
          int64_t seq_start = (seq_dim == 1) ? 0 : (i / search_repeat) * search_len;
          int64_t pos = right ? CustomizedUpperBound(seq_start, seq_start + search_len, values[i], sequence, sort) :
                                CustomizedLowerBound(seq_start, seq_start + search_len, values[i], sequence, sort);
          output[i] = static_cast<T>(pos - seq_start);
      }
  };
  int64_t elem_num = values_t->NumElements();
  if (elem_num >= kParallelDataNum) {
    int64_t max_core_num = std::min(
        elem_num, static_cast<int64_t>(CpuKernelUtils::GetCPUNum(ctx)));
    auto per_unit_size = CeilMultiple(elem_num, max_core_num);
    auto ret = CpuKernelUtils::ParallelFor(ctx, elem_num, per_unit_size, task);
    if (ret != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor failed.");
      return ret;
    }
  } else {
    task(0, elem_num);
  }
  return KERNEL_STATUS_OK;
}

uint32_t SearchSortedKernel::Compute(CpuKernelContext &ctx) {
  KernelStatus res = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), static_cast<uint32_t>(res),
                     "GetInputAndCheck failed, result = [%u].", res);

  std::map<DataType, std::map<DataType, std::function<KernelStatus(
                                            bool, Tensor *, Tensor *, Tensor *,
                                            Tensor *, CpuKernelContext &)>>>
      calls;
  calls[DT_FLOAT16][DT_INT32] = CalSearchSorted<Eigen::half, int>;
  calls[DT_FLOAT][DT_INT32] = CalSearchSorted<float, int>;
  calls[DT_DOUBLE][DT_INT32] = CalSearchSorted<double, int>;
  calls[DT_UINT8][DT_INT32] = CalSearchSorted<uint8_t, int>;
  calls[DT_INT8][DT_INT32] = CalSearchSorted<int8_t, int>;
  calls[DT_INT16][DT_INT32] = CalSearchSorted<int16_t, int>;
  calls[DT_INT32][DT_INT32] = CalSearchSorted<int32_t, int>;
  calls[DT_INT64][DT_INT32] = CalSearchSorted<int64_t, int>;
  calls[DT_FLOAT16][DT_INT64] = CalSearchSorted<Eigen::half, int64_t>;
  calls[DT_FLOAT][DT_INT64] = CalSearchSorted<float, int64_t>;
  calls[DT_DOUBLE][DT_INT64] = CalSearchSorted<double, int64_t>;
  calls[DT_UINT8][DT_INT64] = CalSearchSorted<uint8_t, int64_t>;
  calls[DT_INT8][DT_INT64] = CalSearchSorted<int8_t, int64_t>;
  calls[DT_INT16][DT_INT64] = CalSearchSorted<int16_t, int64_t>;
  calls[DT_INT32][DT_INT64] = CalSearchSorted<int32_t, int64_t>;
  calls[DT_INT64][DT_INT64] = CalSearchSorted<int64_t, int64_t>;

  auto iter = calls.find(sequence_dtype_);
  if (iter == calls.end()) {
    KERNEL_LOG_ERROR(
        "SearchSorted op doesn't support input[0] and input[1] tensor types: "
        "[%s]",
        DTypeStr(sequence_dtype_).c_str());
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  } else {
    if (iter->second.find(output_dtype_) == iter->second.end()) {
      KERNEL_LOG_ERROR(
          "SearchSorted op doesn't support output[0] tensor types: [%s]",
          DTypeStr(output_dtype_).c_str());
    }
  }
  return iter->second[output_dtype_](right_, sequence_t_, values_t_, sorter_t_,
                                     output_t_, ctx);
}

REGISTER_CPU_KERNEL(kSearchSorted, SearchSortedKernel);
}  // namespace aicpu
