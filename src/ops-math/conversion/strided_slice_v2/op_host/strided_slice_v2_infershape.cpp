/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file strided_slice_v2.cc
 * \brief
 */
#include <numeric>
#include <cmath>
#include "register/op_impl_registry.h"
#include "../../strided_slice/op_host/strided_slice_util.h"
#include "log/log.h"
#include "util/const_util.h"
#include "util/shape_util.h"

namespace ops {
using namespace ge;

static const size_t IDX_X = 0;
static const size_t IDX_BEGIN = 1;
static const size_t IDX_END = 2;
static const size_t IDX_AXES = 3;
static const size_t IDX_STRIDES = 4;
static const size_t IDX_Y = 0;
static const size_t IDX_MASK_BEGIN = 0;
static const size_t IDX_MASK_END = 1;
static const size_t IDX_MASK_ELLIPSIS = 2;
static const size_t IDX_MASK_NEW_AXIS = 3;
static const size_t IDX_MASK_SHRINK_AXIS = 4;

static bool GetValueList(QuickVector& valueList)
{
    return (valueList.GetDimNum() != 0);
}

static int64_t CalcMaxShapeSize(int64_t begin_shape_size, int64_t end_shape_size)
{
    int64_t shape_max = static_cast<int64_t>(-1);

    shape_max = std::max(begin_shape_size, shape_max);
    shape_max = std::max(end_shape_size, shape_max);
    OP_LOGD(
        OP_NAME, "begin_shape_size:%ld, end_shape_size:%ld, shape_max:%ld.", begin_shape_size, end_shape_size,
        shape_max);
    return shape_max;
}

template <typename T>
static void PositiveAxisImpl(int32_t inputDims, const gert::Tensor* axesTensor, std::vector<int64_t>& newAxes)
{
    int32_t axesSize = static_cast<int32_t>(axesTensor->GetShapeSize());
    const T* data = axesTensor->GetData<T>();
    if (data == nullptr) {
        OP_LOGE(OP_NAME, "Failed to get tensor data, data is null.");
        return;
    }
    for (int32_t i = 0; i < axesSize; i++) {
        int32_t value = static_cast<int32_t>(data[i]);
        if (value >= 0 && value < inputDims) {
            newAxes.push_back(value);
            OP_LOGD(OP_NAME, "add new axes value:%d", value);
        } else if (value < 0 && value >= -inputDims) {
            newAxes.push_back(value + inputDims);
            OP_LOGD(OP_NAME, "add new axes value plus:%d", value + inputDims);
        }
    }
}

static std::vector<int64_t> ConstructValidAxis(const gert::Tensor* axesTensor, int32_t inputDims)
{
    std::vector<int64_t> newAxes;
    if (!axesTensor || axesTensor->GetShapeSize() == 0) {
        newAxes.resize(inputDims);
        std::iota(newAxes.begin(), newAxes.end(), 0);
        return newAxes;
    }

    if (axesTensor->GetDataType() == ge::DT_INT32) {
        PositiveAxisImpl<int32_t>(inputDims, axesTensor, newAxes);
    } else {
        PositiveAxisImpl<int64_t>(inputDims, axesTensor, newAxes);
    }
    return newAxes;
}

static int64_t GetConstIndexValue(const gert::Tensor* tensor, int32_t idx, int64_t defaultValue = 0)
{
    // idx must be valid
    if (!tensor) {
        OP_LOGE(OP_NAME, "Tensor is null.");
        return defaultValue;
    }

    int64_t value = defaultValue;
    const auto dataType = tensor->GetDataType();
    if (dataType == ge::DT_INT32) {
        const int32_t* data = tensor->GetData<int32_t>();
        if (data == nullptr) {
            OP_LOGE(OP_NAME, "Failed to get tensor data, data is null.");
            return defaultValue;
        }
        value = static_cast<int64_t>(data[idx]);
    } else if (dataType == ge::DT_INT64) {
        const int64_t* data = tensor->GetData<int64_t>();
        if (data == nullptr) {
            OP_LOGE(OP_NAME, "Failed to get tensor data, data is null.");
            return defaultValue;
        }
        value = data[idx];
    } else {
        OP_LOGE(OP_NAME, "Unsupported data type: %d", static_cast<int>(dataType));
        return defaultValue;
    }

    OP_LOGD(OP_NAME, "const tensor[%d] is %ld.", idx, value);
    return value;
}

static void InitListWithDimNum(QuickVector& list, int32_t dimNum, int64_t initValue = 0)
{
    for (int32_t i = 0; i < dimNum; i++) {
        list.AppendDim(initValue);
    }
}

static void ConstructStrideList(
    const gert::Tensor* strideTensor, int32_t dimNum, const std::vector<int64_t>& axes, QuickVector& list_strides)
{
    // Initialize all strides to 1
    InitListWithDimNum(list_strides, dimNum, 1);

    if (!strideTensor) {
        OP_LOGD(OP_NAME, "Stride tensor is null. Set stride as 1.");
        return;
    }

    // Update list_strides with const value of strideTensor
    const int32_t strideSize = static_cast<int32_t>(strideTensor->GetShapeSize());
    const int32_t axesSize = static_cast<int32_t>(axes.size());

    for (int32_t i = 0; i < axesSize && i < strideSize; i++) {
        int64_t axesValue = axes[i];
        list_strides.SetDim(axesValue, GetConstIndexValue(strideTensor, i));
    }
    OP_LOGD(OP_NAME, "strideSize:%d, axesSize:%d.", strideSize, axesSize);
}

static void ConstructBeginList(
    const gert::Tensor* beginTensor, const QuickVector* xShape, const std::vector<int64_t>& axes,
    QuickVector& beginList)
{
    // Initialize beginList with 0
    const int32_t dimNum = static_cast<int32_t>(xShape->GetDimNum());
    InitListWithDimNum(beginList, dimNum, 0);

    // Update beginList with const value of beginTensor
    const int32_t beginsSize = static_cast<int32_t>(beginTensor->GetShapeSize());
    const int32_t axesSize = static_cast<int32_t>(axes.size());

    for (int32_t i = 0; i < axesSize && i < beginsSize; i++) {
        int64_t axesValue = axes[i];
        int64_t inputDim = xShape->GetDim(axesValue);
        beginList.SetDim(axesValue, GetConstIndexValue(beginTensor, i, inputDim));
    }
    OP_LOGD(OP_NAME, "dimNum:%d, beginsSize:%d, axesSize:%d.", dimNum, beginsSize, axesSize);
}

static void ConstructEndList(
    const gert::Tensor* endTensor, const QuickVector* xShape, const std::vector<int64_t>& axes, QuickVector& endList)
{
    // Initialize endList with input_shape
    const int32_t dimNum = static_cast<int32_t>(xShape->GetDimNum());
    for (int32_t i = 0; i < dimNum; i++) {
        endList.AppendDim(xShape->GetDim(i));
    }

    // Update endList with const value of endTensor
    const int32_t endSize = static_cast<int32_t>(endTensor->GetShapeSize());
    const int32_t axesSize = static_cast<int32_t>(axes.size());

    for (int32_t i = 0; i < axesSize && i < endSize; i++) {
        int64_t axesValue = axes[i];
        int64_t inputDim = xShape->GetDim(axesValue);
        endList.SetDim(axesValue, GetConstIndexValue(endTensor, i, inputDim));
    }
    OP_LOGD(OP_NAME, "dimNum:%d, endSize:%d, axesSize:%d.", dimNum, endSize, axesSize);
}

static ge::graphStatus InferShape4StridedSliceV2(gert::InferShapeContext* context)
{
    StridedSliceParams input_params;

    const auto shape_x = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_x);
    auto shape_y = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_y);
    auto shape_begin = context->GetInputShape(IDX_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_begin);
    auto shape_end = context->GetInputShape(IDX_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_end);
    auto tensor_begin = context->GetInputTensor(IDX_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensor_begin);
    auto tensor_end = context->GetInputTensor(IDX_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensor_end);

    if (Ops::Base::IsUnknownRank(*shape_x)) {
        Ops::Base::SetUnknownRank(*shape_y);
        return GRAPH_SUCCESS;
    }

    int32_t input_dim_num = static_cast<int32_t>(shape_x->GetDimNum());
    std::vector<int64_t> new_axis = ConstructValidAxis(context->GetOptionalInputTensor(IDX_AXES), input_dim_num);
    const gert::Tensor* tensor_strides = context->GetOptionalInputTensor(IDX_STRIDES);
    ConstructStrideList(tensor_strides, input_dim_num, new_axis, input_params.strides);

    OP_LOGI(
        context, "shape_x:%s, shape_begin:%s, shape_end:%s.",
        Ops::Base::ToString(*shape_x).c_str(), Ops::Base::ToString(*shape_begin).c_str(),
        Ops::Base::ToString(*shape_end).c_str());

    // Calculate max shape of (begin, end, strides)
    int64_t shape_max = CalcMaxShapeSize(shape_begin->GetDim(0), shape_end->GetDim(0));
    // Necessary input valid check
    if (shape_max == static_cast<int64_t>(-1)) {
        OP_LOGD(OP_NAME, "max shape is -1.");
        shape_y->SetDimNum(0);
        shape_y->AppendDim(UNKNOWN_DIM_NUM);
        return GRAPH_SUCCESS;
    }

    ConstructBeginList(tensor_begin, shape_x, new_axis, input_params.begin);
    ConstructEndList(tensor_end, shape_x, new_axis, input_params.end);

    bool valid_begin = GetValueList(input_params.begin);
    bool valid_end = GetValueList(input_params.end);
    bool valid_strides = GetValueList(input_params.strides);

    OP_LOGD(OP_NAME, "begin_list:%s, valid_begin:%d.", Ops::Base::ToString(input_params.begin).c_str(), valid_begin);
    OP_LOGD(OP_NAME, "end_list:%s, valid_end:%d.", Ops::Base::ToString(input_params.end).c_str(), valid_end);
    OP_LOGD(
        OP_NAME, "stride_list:%s, valid_strides:%d.", Ops::Base::ToString(input_params.strides).c_str(), valid_strides);

    // Check (begin, end) shape size same
    if (input_params.end.GetDimNum() != input_params.begin.GetDimNum()) {
        OP_LOGE(OP_NAME, "end shape, begin shape length mismatch!");
        return GRAPH_FAILED;
    }

    // Get relevant masks from const node
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

// Helper macro to safely get mask values
#define GET_MASK_VALUE(index, mask_name)                                     \
    const int64_t* mask_##mask_name = attrs->GetAttrPointer<int64_t>(index); \
    OP_CHECK_NULL_WITH_CONTEXT(context, mask_##mask_name);                   \
    input_params.mask_name##_mask = static_cast<uint64_t>(*mask_##mask_name)

    GET_MASK_VALUE(IDX_MASK_BEGIN, begin);
    GET_MASK_VALUE(IDX_MASK_END, end);
    GET_MASK_VALUE(IDX_MASK_ELLIPSIS, ellipsis);
    GET_MASK_VALUE(IDX_MASK_NEW_AXIS, new_axis);
    GET_MASK_VALUE(IDX_MASK_SHRINK_AXIS, shrink_axis);

#undef GET_MASK_VALUE

    // infer shape
    input_params.input_shape = *shape_x;
    input_params.begin_valid = valid_begin;
    input_params.end_valid = valid_end;
    input_params.stride_valid = valid_strides;
    if (!InferShape(input_params, shape_y)) {
        OP_LOGE(OP_NAME, "StridedSliceV2 inferShape fail.");
        return GRAPH_FAILED;
    }

    OP_LOGD(context, "output_shape:%s", Ops::Base::ToString(*shape_y).c_str());
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StridedSliceV2)
    .InferShape(InferShape4StridedSliceV2)
    .InputsDataDependency({IDX_BEGIN, IDX_END, IDX_AXES, IDX_STRIDES});
} // namespace ops