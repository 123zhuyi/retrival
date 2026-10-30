/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "slice_write_aicpu.h"

#include <atomic>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

namespace {
const char *const kSliceWrite = "SliceWrite";
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const int32_t kMinDimNum = 1;
const int32_t kDimNum = 2;

template <typename T>
void RangeSliceWrite(int64_t start, int64_t end, const aicpu::Tensor *x,
                     const aicpu::Tensor *value, int64_t row_offset,
                     int64_t col_offset) {
  auto value_shape = value->GetTensorShape();
  int32_t col_index = value_shape->GetDims() - 1;
  int64_t value_col_num = value_shape->GetDimSize(col_index);
  T *value_data = static_cast<T *>(value->GetData());
  int64_t value_offset = start * value_col_num;
  T *value_start = value_data + value_offset;

  int64_t x_col_num = x->GetTensorShape()->GetDimSize(col_index);
  T *x_data = static_cast<T *>(x->GetData());
  int64_t x_offset = (row_offset + start) * x_col_num + col_offset;
  T *x_start = x_data + x_offset;
  int64_t x_left_size = static_cast<int64_t>(x->GetDataSize()) -
                        x_offset * static_cast<int64_t>(sizeof(T));
  KERNEL_LOG_INFO(
      "Slice write begin, x data offset[%ld], x col num[%ld],"
      "value offset[%ld], value col num[%ld], start[%ld], end[%ld]",
      x_offset, x_col_num, value_offset, value_col_num, start, end);
  for (int64_t i = start; i < end; ++i) {
    auto ret = memcpy_s(x_start, static_cast<size_t>(x_left_size), value_start,
                        static_cast<size_t>(value_col_num) * sizeof(T));
    KERNEL_CHECK_FALSE_VOID(
        (ret == EOK), "[%s] copy to output failed, output left size [%ld], copy size [%ld].",
        kSliceWrite, x_left_size, value_col_num * static_cast<int64_t>(sizeof(T)));
    value_start += value_col_num;
    x_start += x_col_num;
    x_left_size -= x_col_num * static_cast<int64_t>(sizeof(T));
  }
}
}  // namespace

namespace aicpu {
bool SliceWriteCpuKernel::CheckValueSupported(
    const DataType input_x_type) const {
  switch (input_x_type) {
    case DT_FLOAT16:
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_INT32:
    case DT_INT64:
      return true;
    default:
      KERNEL_LOG_ERROR("Unsupported input x data type[%s]", DTypeStr(input_x_type).c_str());
      return false;
  }
}

KernelStatus SliceWriteCpuKernel::Check(const Tensor *x, const Tensor *value,
                                        int64_t row_offset, int64_t col_offset) {
  if (!CheckValueSupported(x->GetDataType())) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto x_shape = x->GetTensorShape();
  KERNEL_CHECK_FALSE((x_shape->GetDims() >= kMinDimNum && x_shape->GetDims() <= kDimNum), KERNEL_STATUS_PARAM_INVALID,
                     "Input[x] dims value must be in range [1, 2], but got[%d].", x_shape->GetDims());

  auto value_shape = value->GetTensorShape();
  KERNEL_CHECK_FALSE(
      (value_shape->GetDims() == x_shape->GetDims()), KERNEL_STATUS_PARAM_INVALID,
      "Input [x] dims value[%d] must be equal to input [value] dims value[%d].",
      x_shape->GetDims(), value_shape->GetDims());

  int32_t col_index = x_shape->GetDims() - 1;
  KERNEL_CHECK_FALSE(
      (value_shape->GetDimSize(col_index) + col_offset <= x_shape->GetDimSize(col_index)),
      KERNEL_STATUS_PARAM_INVALID, "Input [begin] col offset value[%ld] error, must be <= [%ld].",
      col_offset, x_shape->GetDimSize(col_index) - value_shape->GetDimSize(col_index));

  if (x_shape->GetDims() == kDimNum) {
    KERNEL_CHECK_FALSE((value_shape->GetDimSize(0) + row_offset <= x_shape->GetDimSize(0)),
        KERNEL_STATUS_PARAM_INVALID,
        "Input [begin] row offset value[%ld] error, must be <= [%ld].",
        row_offset, x_shape->GetDimSize(0) - value_shape->GetDimSize(0));
  }
  return KERNEL_STATUS_OK;
}

KernelStatus SliceWriteCpuKernel::GetBeginValue(const Tensor *begin, int64_t &row_offset, int64_t &col_offset) {
  auto begin_data = begin->GetData();
  auto begin_data_type = begin->GetDataType();
  auto shape = begin->GetTensorShape();
  KERNEL_CHECK_FALSE((shape->GetDims() == 1), KERNEL_STATUS_INNER_ERROR,
                     "Input [begin] dims value must be 1, but got[%d].", shape->GetDims());
  auto begin_value_num = shape->NumElements();
  KERNEL_CHECK_FALSE((begin_value_num <= static_cast<int64_t>(kDimNum)), KERNEL_STATUS_INNER_ERROR,
      "Input [begin] dim[0] value must be not greater than 2, but got[%ld].", begin_value_num);
  if (begin_data_type == DT_INT32) {
    col_offset = static_cast<int32_t *>(begin_data)[begin_value_num - 1];
    if (begin_value_num > 1) {
      row_offset = static_cast<int32_t *>(begin_data)[0];
    }
  } else if (begin_data_type == DT_INT64) {
    col_offset = static_cast<int64_t *>(begin_data)[begin_value_num - 1];
    if (begin_value_num > 1) {
      row_offset = static_cast<int64_t *>(begin_data)[0];
    }
  } else {
    KERNEL_LOG_ERROR("Unsupported input begin data type[%s]", DTypeStr(begin_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t SliceWriteCpuKernel::ValidateInputOutput(CpuKernelContext &ctx, Tensor *&x, Tensor *&begin,
                                                   const Tensor *&value, Tensor *&output) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check SliceWrite params failed.");
  x = ctx.Input(0);
  begin = ctx.Input(1);
  value = ctx.Input(2);
  output = ctx.Output(0);
  auto x_data = x->GetData();
  auto output_data = output->GetData();
  if (x_data != output_data) {
    KERNEL_LOG_ERROR("Input x and output x must be same tensor");
    return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
  }
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

uint32_t SliceWriteCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *x = nullptr;
  Tensor *begin = nullptr;
  const Tensor *value = nullptr;
  Tensor *output = nullptr;
  uint32_t ret = ValidateInputOutput(ctx, x, begin, value, output);
  KERNEL_CHECK_FALSE((ret == static_cast<uint32_t>(KERNEL_STATUS_OK)), ret,
      "Validate input output failed, ret=[%u].", ret);

  int64_t row_offset = 0;
  int64_t col_offset = 0;
  ret = static_cast<uint32_t>(GetBeginValue(begin, row_offset, col_offset));
  KERNEL_CHECK_FALSE((ret == static_cast<uint32_t>(KERNEL_STATUS_OK)), ret,
      "Get begin value failed, ret=[%u].", ret);

  ret = static_cast<uint32_t>(Check(x, value, row_offset, col_offset));
  KERNEL_CHECK_FALSE((ret == static_cast<uint32_t>(KERNEL_STATUS_OK)), ret,
      "Check input param failed, ret=[%u].", ret);

  std::atomic<bool> shard_ret(true);
  auto shardCopy = [&](int64_t start, int64_t end) {
    switch (x->GetDataType()) {
      case DT_FLOAT16:
        RangeSliceWrite<Eigen::half>(start, end, x, value, row_offset, col_offset);
        break;
      case DT_FLOAT:
        RangeSliceWrite<float>(start, end, x, value, row_offset, col_offset);
        break;
      case DT_DOUBLE:
        RangeSliceWrite<double>(start, end, x, value, row_offset, col_offset);
        break;
      case DT_INT32:
        RangeSliceWrite<int32_t>(start, end, x, value, row_offset, col_offset);
        break;
      case DT_INT64:
        RangeSliceWrite<int64_t>(start, end, x, value, row_offset, col_offset);
        break;
      default:
        KERNEL_LOG_ERROR("Unsupported input x data type[%s]", DTypeStr(x->GetDataType()).c_str());
        shard_ret.store(false);
        return;
    }
  };

  int64_t row_num = 1;
  auto value_shape = value->GetTensorShape();
  if (value_shape->GetDims() == kDimNum) {
    row_num = value_shape->GetDimSize(0);
  }

  ret = CpuKernelUtils::ParallelFor(ctx, row_num, 1, shardCopy);
  KERNEL_CHECK_FALSE((ret == static_cast<uint32_t>(KERNEL_STATUS_OK) && shard_ret.load()),
                     static_cast<uint32_t>(KERNEL_STATUS_INNER_ERROR),
                     "ParallelFor failed, ret=[%u].", ret);
  return static_cast<uint32_t>(KERNEL_STATUS_OK);
}

REGISTER_CPU_KERNEL(kSliceWrite, SliceWriteCpuKernel);
}  // namespace aicpu