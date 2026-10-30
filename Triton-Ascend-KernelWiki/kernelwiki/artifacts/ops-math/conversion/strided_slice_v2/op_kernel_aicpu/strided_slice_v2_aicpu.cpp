/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "strided_slice_v2_aicpu.h"

#include <algorithm>
#include <numeric>

#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "../../strided_slice/op_kernel_aicpu/strided_slice_aicpu.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

namespace {
const char *const kStridedSliceV2 = "StridedSliceV2";
const char *const kAxes = "axes";
const char *const kStrides = "strides";
const char *const kBeginMask = "begin_mask";
const char *const kEndMask = "end_mask";
const char *const kEllipsisMask = "ellipsis_mask";
const char *const kNewAxisMask = "new_axis_mask";
const char *const kShrinkAxisMask = "shrink_axis_mask";
constexpr uint32_t kStridedSliceV2AllInputNum = 5;
}  // namespace

namespace aicpu {
uint32_t StridedSliceV2CpuKernel::CheckBeginEndDataType(const Tensor *begin,
                                                          const Tensor *end) {
  DataType begin_type = begin->GetDataType();
  KERNEL_CHECK_FALSE((begin_type == end->GetDataType()),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Expect begin and end to be same data type, but got begin "
                     "data type[%d], end data type[%d]",
                     static_cast<int32_t>(begin_type),
                     static_cast<int32_t>(end->GetDataType()));
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceV2CpuKernel::CheckBeginEndShape(const Tensor *begin,
                                                       const Tensor *end) {
  auto begin_shape = begin->GetTensorShape();
  KERNEL_CHECK_NULLPTR(begin_shape, KERNEL_STATUS_PARAM_INVALID,
                       "Get input begin shape failed")
  auto end_shape = end->GetTensorShape();
  KERNEL_CHECK_NULLPTR(end_shape, KERNEL_STATUS_PARAM_INVALID,
                       "Get input end shape failed")
  KERNEL_CHECK_FALSE(
      ((begin_shape->GetDims() <= 1) && (end_shape->GetDims() <= 1) &&
       (begin_shape->NumElements() == end_shape->NumElements())),
      KERNEL_STATUS_PARAM_INVALID,
      "Expect begin and end to be 1d or scalar equal size tensor, but got begin "
      "dims[%u] and element number[%ld], end dim size[%u] and element "
      "number[%ld].",
      begin_shape->GetDims(), begin_shape->NumElements(), end_shape->GetDims(),
      end_shape->NumElements());
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceV2CpuKernel::CheckStridesDataTypeAndShape(
    const Tensor *begin, const Tensor *strides,
    const std::shared_ptr<TensorShape> &begin_shape) {
  if (strides == nullptr) {
    return KERNEL_STATUS_OK;
  }
  DataType begin_type = begin->GetDataType();
  KERNEL_CHECK_FALSE(
      (begin_type == strides->GetDataType()), KERNEL_STATUS_PARAM_INVALID,
      "Expect begin and strides to be same data type, but got begin "
      "data type[%d], strides data type[%d]",
      static_cast<int32_t>(begin_type),
      static_cast<int32_t>(strides->GetDataType()));
  auto strides_shape = strides->GetTensorShape();
  KERNEL_CHECK_NULLPTR(strides_shape, KERNEL_STATUS_PARAM_INVALID,
                       "Get input strides shape failed")
  KERNEL_CHECK_FALSE(
      ((strides_shape->GetDims() <= 1) &&
       (begin_shape->NumElements() == strides_shape->NumElements())),
      KERNEL_STATUS_PARAM_INVALID,
      "Expect begin and strides to be 1d or scalar equal size tensor, but got begin "
      "dims[%u] and element number[%ld], strides dim size[%u] and element "
      "number[%ld].",
      begin_shape->GetDims(), begin_shape->NumElements(),
      strides_shape->GetDims(), strides_shape->NumElements());
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceV2CpuKernel::CheckAxesDataType(const Tensor *begin,
                                                      const Tensor *axes) {
  if (axes == nullptr) {
    return KERNEL_STATUS_OK;
  }
  DataType begin_type = begin->GetDataType();
  KERNEL_CHECK_FALSE(
      (begin_type == axes->GetDataType()), KERNEL_STATUS_PARAM_INVALID,
      "Expect begin and axes to be same data type, but got begin "
      "data type[%d], axes data type[%d]",
      static_cast<int32_t>(begin_type),
      static_cast<int32_t>(axes->GetDataType()));
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceV2CpuKernel::CheckParam(const Tensor *begin,
                                             const Tensor *end,
                                             const Tensor *axes,
                                             const Tensor *strides) {
  uint32_t ret = CheckBeginEndDataType(begin, end);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  ret = CheckBeginEndShape(begin, end);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  auto begin_shape = begin->GetTensorShape();
  KERNEL_CHECK_NULLPTR(begin_shape, KERNEL_STATUS_PARAM_INVALID,
                       "Get input begin shape failed")
  ret = CheckStridesDataTypeAndShape(begin, strides, begin_shape);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  return CheckAxesDataType(begin, axes);
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::BuildBeginParam(
    const std::shared_ptr<TensorShape> &x_shape, const Tensor *begin,
    std::vector<int64_t> &begin_vec) {
  (void)x_shape;
  T *begin_data = static_cast<T *>(begin->GetData());
  KERNEL_CHECK_NULLPTR(begin_data, KERNEL_STATUS_PARAM_INVALID,
                       "Get input begin data failed")
  for (int64_t i = 0; i < begin->NumElements(); ++i) {
    begin_vec.push_back(static_cast<int64_t>(begin_data[i]));
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::BuildEndParam(
    const std::shared_ptr<TensorShape> &x_shape, const Tensor *end,
    std::vector<int64_t> &end_vec) {
  (void)x_shape;
  T *end_data = static_cast<T *>(end->GetData());
  KERNEL_CHECK_NULLPTR(end_data, KERNEL_STATUS_PARAM_INVALID,
                       "Get input end data failed")
  for (int64_t i = 0; i < end->NumElements(); ++i) {
    end_vec.push_back(static_cast<int64_t>(end_data[i]));
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::BuildStridesParam(
    const std::shared_ptr<TensorShape> &x_shape, const Tensor *strides,
    std::vector<int64_t> &strides_vec) {
  const int32_t x_dims = x_shape->GetDims();
  if (strides == nullptr) {
    for (int32_t i = 0; i < x_dims; ++i) {
      strides_vec.push_back(1);
    }
  } else {
    T *strides_data = static_cast<T *>(strides->GetData());
    KERNEL_CHECK_NULLPTR(strides_data, KERNEL_STATUS_PARAM_INVALID,
                         "Get input strides data failed")
    for (int64_t i = 0; i < strides->NumElements(); ++i) {
      strides_vec.push_back(static_cast<int64_t>(strides_data[i]));
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::BuildAxesParam(
    const std::shared_ptr<TensorShape> &x_shape, const Tensor *axes,
    std::vector<int64_t> &axes_vec) {
  const int32_t x_dims = x_shape->GetDims();
  if (axes == nullptr) {
    axes_vec.resize(x_dims);
    std::iota(axes_vec.begin(), axes_vec.end(), 0);
  } else {
    T *axes_data = static_cast<T *>(axes->GetData());
    KERNEL_CHECK_NULLPTR(axes_data, KERNEL_STATUS_PARAM_INVALID,
                         "Get input axes data failed")
    for (int64_t i = 0; i < axes->NumElements(); ++i) {
      T axes_value = axes_data[i] < 0 ? axes_data[i] + x_dims : axes_data[i];
      KERNEL_CHECK_FALSE(
          ((axes_value >= 0) && (axes_value < x_dims)), KERNEL_STATUS_PARAM_INVALID, "Check axes[%ld] value[%ld] failed, must be in range [-%d, %d]", i,
          static_cast<int64_t>(axes_value), x_dims, x_dims - 1);
      KERNEL_CHECK_FALSE(
          (std::find(axes_vec.begin(), axes_vec.end(), axes_value) ==
           axes_vec.end()), KERNEL_STATUS_PARAM_INVALID, "Check value failed, axes[%ld] value[%ld] is repeated", i,
          static_cast<int64_t>(axes_value));
      axes_vec.push_back(static_cast<int64_t>(axes_value));
    }

    for (int32_t i = 0; i < x_dims; ++i) {
      if (std::find(axes_vec.begin(), axes_vec.end(), i) == axes_vec.end()) {
        axes_vec.push_back(static_cast<int64_t>(i));
      }
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::BuildParam(
    const Tensor *x, const Tensor *begin, const Tensor *end, const Tensor *axes,
    const Tensor *strides, std::vector<int64_t> &begin_vec,
    std::vector<int64_t> &end_vec, std::vector<int64_t> &strides_vec) {
  auto x_shape = x->GetTensorShape();

  std::vector<int64_t> begin_ret{};
  uint32_t ret = BuildBeginParam<T>(x_shape, begin, begin_ret);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "Build begin parameter failed, ret[%u]", ret);

  std::vector<int64_t> end_ret{};
  ret = BuildEndParam<T>(x_shape, end, end_ret);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "Build end parameter failed, ret[%u]", ret);

  std::vector<int64_t> strides_ret{};
  ret = BuildStridesParam<T>(x_shape, strides, strides_ret);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "Build strides parameter failed, ret[%u]", ret);

  std::vector<int64_t> axes_ret{};
  ret = BuildAxesParam<T>(x_shape, axes, axes_ret);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "Build axes parameter failed, ret[%u]", ret);

  begin_vec.resize(axes_ret.size());
  end_vec.resize(axes_ret.size());
  strides_vec.resize(axes_ret.size());
  for (size_t i = 0; i < axes_ret.size(); ++i) {
    T axes_value = static_cast<T>(axes_ret[i]);
    if (i < begin_ret.size()) {
      begin_vec[axes_value] = begin_ret[i];
      end_vec[axes_value] = end_ret[i];
      strides_vec[axes_value] = strides_ret[i];
    }
    else {
      begin_vec[axes_value] = 0;
      end_vec[axes_value] = x_shape->GetDimSize(axes_value);
      strides_vec[axes_value] = 1;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t StridedSliceV2CpuKernel::CheckAndBuildParam(
    const Tensor *x, const Tensor *begin, const Tensor *end, const Tensor *axes,
    const Tensor *strides, std::vector<int64_t> &begin_vec,
    std::vector<int64_t> &end_vec, std::vector<int64_t> &strides_vec) {
  uint32_t ret = CheckParam(begin, end, axes, strides);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  return BuildParam<T>(x, begin, end, axes, strides, begin_vec, end_vec,
                       strides_vec);
}

uint32_t StridedSliceV2CpuKernel::DoStridedSliceV2(
    const CpuKernelContext &ctx, const std::vector<int64_t> &begin_vec,
    const std::vector<int64_t> &end_vec,
    const std::vector<int64_t> &strides_vec) {
  uint32_t ret = KERNEL_STATUS_OK;
#define STRIDED_SLICE_V2_CASE(DT, T)                                  \
  case (DT): {                                                        \
    ret = StridedSliceCpuKernel::CalStridedSlice<T>(                  \
        ctx, begin_vec, end_vec, strides_vec,                         \
        ctx.Input(0), ctx.Output(0));                                 \
    break;                                                            \
  }

  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    STRIDED_SLICE_V2_CASE(DT_INT8, int8_t)
    STRIDED_SLICE_V2_CASE(DT_INT16, int16_t)
    STRIDED_SLICE_V2_CASE(DT_INT32, int32_t)
    STRIDED_SLICE_V2_CASE(DT_INT64, int64_t)
    STRIDED_SLICE_V2_CASE(DT_UINT8, uint8_t)
    STRIDED_SLICE_V2_CASE(DT_UINT16, uint16_t)
    STRIDED_SLICE_V2_CASE(DT_UINT32, uint32_t)
    STRIDED_SLICE_V2_CASE(DT_UINT64, uint64_t)
    STRIDED_SLICE_V2_CASE(DT_FLOAT16, Eigen::half)
    STRIDED_SLICE_V2_CASE(DT_FLOAT, float)
    STRIDED_SLICE_V2_CASE(DT_DOUBLE, double)
    STRIDED_SLICE_V2_CASE(DT_BOOL,  bool)
    STRIDED_SLICE_V2_CASE(DT_BFLOAT16, Eigen::bfloat16)
    STRIDED_SLICE_V2_CASE(DT_COMPLEX64, std::complex<float>)
    STRIDED_SLICE_V2_CASE(DT_COMPLEX128, std::complex<double>)
    STRIDED_SLICE_V2_CASE(DT_QINT8, int8_t)
    STRIDED_SLICE_V2_CASE(DT_QINT16, int16_t)
    STRIDED_SLICE_V2_CASE(DT_QINT32, int32_t)
    STRIDED_SLICE_V2_CASE(DT_QUINT8, uint8_t)
    STRIDED_SLICE_V2_CASE(DT_QUINT16, uint16_t)
    default:
      KERNEL_LOG_ERROR("%s kernel data type [%s] not support.", kStridedSliceV2,
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
#undef STRIDED_SLICE_V2_CASE
  return ret;
}

uint32_t StridedSliceV2CpuKernel::GetInputTensors(CpuKernelContext &ctx, Tensor *&x,
                                                    Tensor *&begin, Tensor *&end,
                                                    Tensor *&axes, Tensor *&strides) {
  x = ctx.Input(kFirstInputIndex);
  KERNEL_CHECK_NULLPTR(x, KERNEL_STATUS_PARAM_INVALID, "Get input x failed")
  auto x_shape = x->GetTensorShape();
  KERNEL_CHECK_NULLPTR(x_shape, KERNEL_STATUS_PARAM_INVALID,
                       "Get input x shape failed")
  begin = ctx.Input(kSecondInputIndex);
  KERNEL_CHECK_NULLPTR(begin, KERNEL_STATUS_PARAM_INVALID,
                       "Get input begin failed")
  end = ctx.Input(kThirdInputIndex);
  KERNEL_CHECK_NULLPTR(end, KERNEL_STATUS_PARAM_INVALID, "Get input end failed")

  axes = nullptr;
  strides = nullptr;
  uint32_t input_size = ctx.GetInputsSize();
  KERNEL_LOG_INFO("Input size[%u]", input_size);

  if (input_size == kStridedSliceV2AllInputNum) {
    axes = ctx.Input(kFourthInputIndex);
    KERNEL_CHECK_NULLPTR(axes, KERNEL_STATUS_PARAM_INVALID, "Get input axes failed")
    strides = ctx.Input(kFifthInputIndex);
    KERNEL_CHECK_NULLPTR(strides, KERNEL_STATUS_PARAM_INVALID, "Get input strides failed")
  } else {
    for (uint32_t i = 3; i < input_size; ++i) {
      Tensor *tmp = ctx.Input(i);
      KERNEL_CHECK_NULLPTR(tmp, KERNEL_STATUS_PARAM_INVALID, "Get input[%u] failed", i)
      std::string name = CpuKernelUtils::GetTensorName(tmp);
      if (name == kAxes) {
        axes = tmp;
      } else if (name == kStrides) {
        strides = tmp;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceV2CpuKernel::GetMaskAttrs(CpuKernelContext &ctx,
                                                  int64_t &begin_mask_value,
                                                  int64_t &end_mask_value,
                                                  int64_t &ellipsis_mask_value,
                                                  int64_t &new_axis_mask_value,
                                                  int64_t &shrink_axis_mask_value) {
  begin_mask_value = 0;
  AttrValue *begin_mask = ctx.GetAttr(kBeginMask);
  if (begin_mask != nullptr) {
    begin_mask_value = begin_mask->GetInt();
  }

  end_mask_value = 0;
  AttrValue *end_mask = ctx.GetAttr(kEndMask);
  if (end_mask != nullptr) {
    end_mask_value = end_mask->GetInt();
  }

  ellipsis_mask_value = 0;
  AttrValue *ellipsis_mask = ctx.GetAttr(kEllipsisMask);
  if (ellipsis_mask != nullptr) {
    ellipsis_mask_value = ellipsis_mask->GetInt();
  }

  new_axis_mask_value = 0;
  AttrValue *new_axis_mask = ctx.GetAttr(kNewAxisMask);
  if (new_axis_mask != nullptr) {
    new_axis_mask_value = new_axis_mask->GetInt();
  }

  shrink_axis_mask_value = 0;
  AttrValue *shrink_axis_mask = ctx.GetAttr(kShrinkAxisMask);
  if (shrink_axis_mask != nullptr) {
    shrink_axis_mask_value = shrink_axis_mask->GetInt();
  }
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceV2CpuKernel::CheckAndBuildParamsByType(
    const Tensor *x, const Tensor *begin, const Tensor *end, const Tensor *axes,
    const Tensor *strides, std::vector<int64_t> &begin_vec,
    std::vector<int64_t> &end_vec, std::vector<int64_t> &strides_vec) {
  DataType type = begin->GetDataType();
  uint32_t ret = KERNEL_STATUS_OK;
  if (type == DT_INT32) {
    ret = CheckAndBuildParam<int32_t>(x, begin, end, axes, strides, begin_vec,
                                      end_vec, strides_vec);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret,
                       "Check and build parameter failed, ret[%u]", ret);
  } else if (type == DT_INT64) {
    ret = CheckAndBuildParam<int64_t>(x, begin, end, axes, strides, begin_vec,
                                      end_vec, strides_vec);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret,
                       "Check and build parameter failed, ret[%u]", ret);
  } else {
    KERNEL_LOG_ERROR(
        "Unsupported input begin data_type[%u], only support DT_INT32 and "
        "DT_INT64.",
        static_cast<uint32_t>(type));
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_LOG_INFO("Check and build parameters success.");
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceV2CpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *x = nullptr;
  Tensor *begin = nullptr;
  Tensor *end = nullptr;
  Tensor *axes = nullptr;
  Tensor *strides = nullptr;
  uint32_t ret = GetInputTensors(ctx, x, begin, end, axes, strides);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  int64_t begin_mask_value = 0;
  int64_t end_mask_value = 0;
  int64_t ellipsis_mask_value = 0;
  int64_t new_axis_mask_value = 0;
  int64_t shrink_axis_mask_value = 0;
  ret = GetMaskAttrs(ctx, begin_mask_value, end_mask_value, ellipsis_mask_value,
                    new_axis_mask_value, shrink_axis_mask_value);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  std::vector<int64_t> begin_vec{};
  std::vector<int64_t> end_vec{};
  std::vector<int64_t> strides_vec{};
  auto x_shape = x->GetTensorShape();
  std::vector<int64_t> x_shape_value = x_shape->GetDimSizes();

  ret = CheckAndBuildParamsByType(x, begin, end, axes, strides, begin_vec,
                                  end_vec, strides_vec);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  ret = StridedSliceCpuKernel::InitParamsWithMasks(
      x_shape_value, begin_mask_value, end_mask_value, ellipsis_mask_value,
      new_axis_mask_value, shrink_axis_mask_value, begin_vec, end_vec,
      strides_vec);
  KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret,
                     "Init parameters with masks failed, ret[%u]", ret);

  return DoStridedSliceV2(ctx, begin_vec, end_vec, strides_vec);
}

REGISTER_CPU_KERNEL(kStridedSliceV2, StridedSliceV2CpuKernel);
}  // namespace aicpu
