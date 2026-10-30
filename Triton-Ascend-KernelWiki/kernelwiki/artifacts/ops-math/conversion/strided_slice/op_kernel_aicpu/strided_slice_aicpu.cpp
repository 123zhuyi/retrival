/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "strided_slice_aicpu.h"

#include <algorithm>
#include "securec.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_types.h"

namespace {
constexpr uint32_t kStridedSliceInputNum = 4;
constexpr uint32_t kStridedSliceOutputNum = 1;
constexpr const char *kStridedSlice = "StridedSlice";
}

namespace aicpu {
template <typename T>
static inline void DataLeftShift(T &data) {  data = data << 1;  }

static uint32_t ProcessEllipsisMask(
    const std::vector<int64_t> &begin,
    const std::vector<int64_t> &end,
    const std::vector<int64_t> &strides,
    const std::vector<int64_t> &x_shape,
    int64_t ellipsis_mask, int64_t new_axis_mask,
    size_t &i, size_t &j, int64_t &bit_mask, bool &has_ellipsis,
    int64_t &begin_j, int64_t &end_j, int64_t &strides_j,
    std::vector<int64_t> &begin_res,
    std::vector<int64_t> &end_res,
    std::vector<int64_t> &strides_res) {
  if (ellipsis_mask & bit_mask) {
    if (has_ellipsis) {
      KERNEL_LOG_ERROR("[%s] multiple ellipses in slice spec not allowed.",
                       kStridedSlice);
      return KERNEL_STATUS_INNER_ERROR;
    }

    j++;
    DataLeftShift(bit_mask);
    int64_t ellipsis_bits = static_cast<int64_t>(x_shape.size()) - static_cast<int64_t>(strides.size());
    int64_t bit_mask_tmp = 1;
    for (size_t k = 0; k < strides.size(); ++k) {
      if ((new_axis_mask & bit_mask_tmp) && !(ellipsis_mask & bit_mask_tmp)) {
        ++ellipsis_bits;
      }
      bit_mask_tmp <<= 1;
    }

    for (int64_t k = 0; k <= ellipsis_bits; ++k) {
      begin_res.push_back(0);
      end_res.push_back(x_shape[i]);
      strides_res.push_back(1);
      i++;
    }

    begin_j = begin[j];
    end_j = end[j];
    strides_j = strides[j];
    has_ellipsis = true;
  }

  return KERNEL_STATUS_OK;
}

inline void ProcessEndMask(const std::vector<int64_t> &strides,
                           const std::vector<int64_t> &x_shape,
                           int64_t end_mask, int64_t shrink_axis_mask,
                           size_t i, size_t j, int64_t bit_mask,
                           int64_t &end_j) {
  if ((end_mask & bit_mask) && !(shrink_axis_mask & bit_mask)) {
    end_j = (strides[j] > 0) ? x_shape[i] : -(x_shape[i] + 1);
  }
}

inline bool ProcessNewAxisMask(int64_t new_axis_mask,
                               size_t &i, const int64_t &bit_mask) {
  bool result = (new_axis_mask & bit_mask) != 0;
  i -= result ? 1 : 0;
  return result;
}

inline uint32_t ProcessShrinkAxisMask(const std::vector<int64_t> &x_shape,
                                      int64_t shrink_axis_mask,
                                      size_t i, int64_t bit_mask,
                                      int64_t begin_j, int64_t strides_j,
                                      int64_t &end_j) {
  if (shrink_axis_mask & bit_mask) {
    if ((begin_j < -x_shape[i]) || (begin_j >= x_shape[i]) || (strides_j < 0)) {
      KERNEL_LOG_ERROR("[%s] process shrink axis mask failed.", kStridedSlice);
      return KERNEL_STATUS_INNER_ERROR;
    }
    end_j = begin_j + 1;
  }
  return KERNEL_STATUS_OK;
}

uint32_t ProcessMasks(const std::vector<int64_t> &begin,
                      const std::vector<int64_t> &end,
                      const std::vector<int64_t> &strides,
                      const std::vector<int64_t> &x_shape,
                      int64_t begin_mask, int64_t end_mask,
                      int64_t ellipsis_mask, int64_t new_axis_mask,
                      int64_t shrink_axis_mask,
                      size_t &i, size_t &j,
                      int64_t &bit_mask, bool &has_ellipsis,
                      std::vector<int64_t> &begin_res,
                      std::vector<int64_t> &end_res,
                      std::vector<int64_t> &strides_res) {
  int64_t begin_j = begin[j];
  int64_t end_j = end[j];
  int64_t strides_j = strides[j];
  if (j < strides.size()) {
    if (ProcessEllipsisMask(begin, end, strides, x_shape, ellipsis_mask,
        new_axis_mask, i, j, bit_mask, has_ellipsis,
        begin_j, end_j, strides_j, begin_res, end_res, strides_res) ==
        KERNEL_STATUS_INNER_ERROR) {
      return KERNEL_STATUS_INNER_ERROR;
    }

    if ((begin_mask & bit_mask) && (!(shrink_axis_mask & bit_mask))) {
      begin_j = (strides[j] > 0) ? 0 : (x_shape[i] - 1);
    }

    ProcessEndMask(strides, x_shape, end_mask, shrink_axis_mask,
                   i, j, bit_mask, end_j);
    if (ProcessNewAxisMask(new_axis_mask, i, bit_mask)) {
      return KERNEL_STATUS_OK;
    }
    if (ProcessShrinkAxisMask(x_shape, shrink_axis_mask, i, bit_mask,
        begin_j, strides_j, end_j) == KERNEL_STATUS_INNER_ERROR) {
      return KERNEL_STATUS_INNER_ERROR;
    }
  } else {
    begin_j = 0;
    end_j = x_shape[i];
    strides_j = 1;
  }

  begin_res.push_back(begin_j);
  end_res.push_back(end_j);
  strides_res.push_back(strides_j);
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceCpuKernel::InitParamsWithMasks(
    const std::vector<int64_t> &x_shape,
    int64_t begin_mask, int64_t end_mask,
    int64_t ellipsis_mask, int64_t new_axis_mask,
    int64_t shrink_axis_mask,
    std::vector<int64_t> &begin,
    std::vector<int64_t> &end,
    std::vector<int64_t> &strides) {
  size_t i = 0;
  size_t j = 0;
  int64_t bit_mask = 1;
  bool has_ellipsis = false;
  std::vector<int64_t> begin_res;
  std::vector<int64_t> end_res;
  std::vector<int64_t> strides_res;
  while (i < x_shape.size()) {
    KERNEL_HANDLE_ERROR(ProcessMasks(begin, end, strides, x_shape,
        begin_mask, end_mask, ellipsis_mask, new_axis_mask,
        shrink_axis_mask, i, j, bit_mask, has_ellipsis,
        begin_res, end_res, strides_res),
        "[%s] process masks failed.", kStridedSlice);
    i++;
    j++;
    DataLeftShift(bit_mask);
  }

  auto remove_zero = [](int stride) { return stride == 0; };
  auto new_end_strides = std::remove_if(strides_res.begin(), strides_res.end(), remove_zero);
  auto new_end_begin = begin_res.begin() + std::distance(strides_res.begin(), new_end_strides);
  auto new_end_end = end_res.begin() + std::distance(strides_res.begin(), new_end_strides);
  strides_res.erase(new_end_strides, strides_res.end());
  begin_res.erase(new_end_begin, begin_res.end());
  end_res.erase(new_end_end, end_res.end());

  if (begin_res.empty() || end_res.empty() || strides_res.empty()) {
    KERNEL_LOG_ERROR("[%s] init params with masks failed.", kStridedSlice);
    return KERNEL_STATUS_INNER_ERROR;
  }

  begin = begin_res;
  end = end_res;
  strides = strides_res;
  KERNEL_LOG_INFO("[%s] begin with masks: [%s].", kStridedSlice,
                  VectorToString(begin).c_str());
  KERNEL_LOG_INFO("[%s] end with masks: [%s].", kStridedSlice,
                  VectorToString(end).c_str());
  KERNEL_LOG_INFO("[%s] strides with masks: [%s].", kStridedSlice,
                  VectorToString(strides).c_str());
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kStridedSliceInputNum,
                                  kStridedSliceOutputNum),
                      "[%s] check params failed.", kStridedSlice);

  // parse params
  KERNEL_HANDLE_ERROR(ParseKernelParams(ctx),
                      "[%s] parse kernel params failed.", kStridedSlice);

  // init params with masks
  KERNEL_HANDLE_ERROR(InitParamsWithMasks(x_shape_, begin_mask_, end_mask_,
      ellipsis_mask_, new_axis_mask_, shrink_axis_mask_,
      begin_, end_, strides_),
      "[%s] init params with masks failed.", kStridedSlice);

  // cal strided slice
  Tensor *x_tensor = ctx.Input(0);
  Tensor *y_tensor = ctx.Output(0);
  DataType data_type = x_tensor->GetDataType();

#define STRIDED_SLICE_CASE(dtype, T)                                        \
  case dtype:                                                               \
    return CalStridedSlice<T>(ctx, begin_, end_, strides_,                  \
                              x_tensor, y_tensor)

  switch (data_type) {
    STRIDED_SLICE_CASE(DT_INT8, int8_t);
    STRIDED_SLICE_CASE(DT_INT16, int16_t);
    STRIDED_SLICE_CASE(DT_INT32, int32_t);
    STRIDED_SLICE_CASE(DT_INT64, int64_t);
    STRIDED_SLICE_CASE(DT_UINT8, uint8_t);
    STRIDED_SLICE_CASE(DT_UINT16, uint16_t);
    STRIDED_SLICE_CASE(DT_UINT32, uint32_t);
    STRIDED_SLICE_CASE(DT_UINT64, uint64_t);
    STRIDED_SLICE_CASE(DT_FLOAT16, Eigen::half);
    STRIDED_SLICE_CASE(DT_FLOAT, float);
    STRIDED_SLICE_CASE(DT_DOUBLE, double);
    STRIDED_SLICE_CASE(DT_BOOL, bool);
    default:
      KERNEL_LOG_ERROR("[%s] doesn't support input[0] data_type [%s].",
                       kStridedSlice, DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

#undef STRIDED_SLICE_CASE
}

uint32_t StridedSliceCpuKernel::ParseKernelParams(const CpuKernelContext &ctx) {
  // get inputs
  x_shape_ = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  KERNEL_LOG_INFO("[%s] get input[0] shape: [%s].",
                  kStridedSlice, VectorToString(x_shape_).c_str());

  KERNEL_HANDLE_ERROR(ParseIndexInput(ctx, 1, begin_),
                      "[%s] parse index input failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(ParseIndexInput(ctx, 2, end_),
                      "[%s] parse index input failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(ParseIndexInput(ctx, 3, strides_),
                      "[%s] parse index input failed.", kStridedSlice);

  // get masks
  KERNEL_HANDLE_ERROR(GetMaskAttr(ctx, "begin_mask", begin_mask_),
                      "[%s] get mask attr failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(GetMaskAttr(ctx, "end_mask", end_mask_),
                      "[%s] get mask attr failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(GetMaskAttr(ctx, "ellipsis_mask", ellipsis_mask_),
                      "[%s] get mask attr failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(GetMaskAttr(ctx, "new_axis_mask", new_axis_mask_),
                      "[%s] get mask attr failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(GetMaskAttr(ctx, "shrink_axis_mask", shrink_axis_mask_),
                      "[%s] get mask attr failed.", kStridedSlice);

  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceCpuKernel::ParseIndexInput(const CpuKernelContext &ctx,
                                                uint32_t index,
                                                std::vector<int64_t> &vec) {
  Tensor *index_tensor = ctx.Input(index);
  int64_t tensor_size = index_tensor->NumElements();
  switch (index_tensor->GetDataType()) {
    case DT_INT32: {
      int32_t *tensor_data = static_cast<int32_t *>(index_tensor->GetData());
      vec.insert(vec.begin(), tensor_data, tensor_data + tensor_size);
      break;
    }
    case DT_INT64: {
      int64_t *tensor_data = static_cast<int64_t *>(index_tensor->GetData());
      vec.insert(vec.begin(), tensor_data, tensor_data + tensor_size);
      break;
    }
    default:
      KERNEL_LOG_ERROR("[%s] input[%u] data_tpye must be in {int32 int64}.",
                       kStridedSlice, index);
      return KERNEL_STATUS_PARAM_INVALID;
  }

  KERNEL_LOG_INFO("[%s] get input[%u]: [%s].", kStridedSlice, index,
                  VectorToString(vec).c_str());
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceCpuKernel::GetMaskAttr(const CpuKernelContext &ctx,
                                            const std::string attr,
                                            int64_t &mask) const {
  AttrValue *mask_attr = ctx.GetAttr(attr);
  if (mask_attr != nullptr) {
    mask = mask_attr->GetInt();
  } else {
    KERNEL_LOG_WARN("[%s] can not get attr [%s].", kStridedSlice, attr.c_str());
    mask = 0;
  }
  KERNEL_LOG_INFO("[%s] get attr [%s]: [%ld].",
                  kStridedSlice, attr.c_str(), mask);
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kStridedSlice, StridedSliceCpuKernel);
}  // namespace aicpu
