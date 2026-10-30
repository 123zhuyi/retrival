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
 * \file strided_slice_v3_infershape.cpp
 * \brief
 */

#include <numeric>
#include "register/op_impl_registry.h"
#include "../../strided_slice/op_host/strided_slice_util.h"
#include "log/log.h"
#include "op_api/op_util.h"
#include "util/const_util.h"
#include "util/math_util.h"
#include "util/shape_util.h"

using namespace std;
namespace {
const std::string OP_NAME = "StridedSliceV3";
const int INDEX_X = 0;
const int INDEX_BEGIN = 1;
const int INDEX_END = 2;
const int INDEX_AXES = 3;
const int INDEX_STRIDES = 4;
const int INDEX_Y = 0;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1L;
} // namespace

namespace ops {
static int64_t GetConstIndexValue(const gert::Tensor* tensor, size_t idx)
{
    // idx must be valid
    int64_t value = 0;
    if (tensor->GetDataType() == ge::DT_INT32) {
        const int32_t* data = tensor->GetData<int32_t>();
        value = static_cast<int64_t>(data[idx]);
    } else {
        const int64_t* data = tensor->GetData<int64_t>();
        value = data[idx];
    }
    OP_LOGD(OP_NAME, "const tensor[%ld] is %ld.", idx, value);
    return value;
}

static int64_t GetConstIndexValue(
    const gert::Tensor* tensor, size_t idx, int64_t inputSize, int64_t clipLower, int64_t clipUpper)
{
    // idx must be valid
    int64_t value = 0;
    if (tensor->GetDataType() == ge::DT_INT32) {
        const int32_t* data = tensor->GetData<int32_t>();
        value = static_cast<int64_t>(data[idx]);
    } else {
        const int64_t* data = tensor->GetData<int64_t>();
        value = data[idx];
    }
    if (value < 0) {
        value += inputSize;
    }

    // clamp value
    if (value < clipLower) {
        value = clipLower;
    } else if (value > clipUpper) {
        value = clipUpper;
    }
    OP_LOGD(OP_NAME, "const tensor[%ld] is %ld.", idx, value);
    return value;
}

template <typename T>
static void PositiveAxisImpl(int32_t inputDims, const gert::Tensor* axisTensor, vector<int32_t>& newAxis)
{
    const int64_t axisSize = axisTensor->GetShapeSize();
    const T* data = axisTensor->GetData<T>();
    for (int i = 0; i < axisSize; i++) {
        int64_t value = static_cast<int64_t>(data[i]);
        if (value >= 0 && value < inputDims) {
            newAxis.push_back(value);
            OP_LOGD(OP_NAME, "add new axes value:%ld", value);
        } else if (value < 0 && value >= -inputDims) {
            newAxis.push_back(value + inputDims);
            OP_LOGD(OP_NAME, "add new axes value plus:%ld", value + inputDims);
        } else {
            OP_LOGI(OP_NAME, "idx:%d axes value:%ld invalid, inputDims:%d", i, value, inputDims);
        }
    }
    return;
}

static bool ConstructValidAxis(int32_t inputDims, const gert::Tensor* axisTensor, std::vector<int32_t>& newAxis)
{
    if (!axisTensor || axisTensor->GetShapeSize() == 0) {
        newAxis.resize(inputDims);
        std::iota(newAxis.begin(), newAxis.end(), 0);
        return true;
    }
    if (axisTensor->GetDataType() == ge::DT_INT32) {
        PositiveAxisImpl<int32_t>(inputDims, axisTensor, newAxis);
    } else if (axisTensor->GetDataType() == ge::DT_INT64) {
        PositiveAxisImpl<int64_t>(inputDims, axisTensor, newAxis);
    } else {
        OP_LOGE(
            OP_NAME, "axesTensor dtype:%s invalid, only support DT_INT32 or DT_INT64",
            Ops::Base::ToString(axisTensor->GetDataType()).c_str());
        return false;
    }
    return true;
}

static ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* outputShape)
{
    outputShape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        outputShape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD(OP_NAME, "set all dim = -1, output = %s", Ops::Base::ToString(*outputShape).c_str());

    return ge::GRAPH_SUCCESS;
}

static bool GetBeginValue(
    int64_t curAxisInputSize, int64_t stepValue, const gert::Tensor* beginTensor, int32_t idx, int64_t& beginValue)
{
    if (!IsConstTensor(beginTensor)) {
        OP_LOGD(OP_NAME, "beginTensor is unconst");
        return false;
    }

    int64_t clipUpper = curAxisInputSize;
    if (stepValue < 0) {
        clipUpper -= 1; // if step < 0, start from last valid_index
    }
    beginValue = GetConstIndexValue(beginTensor, idx, curAxisInputSize, 0, clipUpper);
    return true;
}

static bool GetEndValue(
    int64_t curAxisInputSize, int64_t stepValue, const gert::Tensor* endTensor, int32_t idx, int64_t& endValue)
{
    if (!IsConstTensor(endTensor)) {
        OP_LOGD(OP_NAME, "endTensor is unconst");
        return false;
    }
    int64_t clipLower = 0;
    if (stepValue < 0) {
        clipLower = -1; // if step < 0, end with first valid_index
    }
    endValue = GetConstIndexValue(endTensor, idx, curAxisInputSize, clipLower, curAxisInputSize);
    return true;
}

static bool GetStepValue(const gert::Tensor* stridesTensor, int32_t idx, int64_t& stepValue)
{
    if (!IsConstTensor(stridesTensor)) {
        OP_LOGD(OP_NAME, "stridesTensor is unconst");
        return false;
    }
    stepValue = GetConstIndexValue(stridesTensor, idx);
    return true;
}

static bool DoInferShape(
    const gert::Shape* xShape, gert::Shape* yShape, const gert::Tensor* stridesTensor, const gert::Tensor* beginTensor,
    const gert::Tensor* endTensor, const std::vector<int32_t>& newAxis)
{
    const int32_t stridesSize = (stridesTensor) ? static_cast<int32_t>(stridesTensor->GetShapeSize()) : 0;
    const int32_t beginsSize = static_cast<int32_t>(beginTensor->GetShapeSize());
    const int32_t endsSize = static_cast<int32_t>(endTensor->GetShapeSize());

    const int32_t axisSize = static_cast<int32_t>(newAxis.size());
    for (int32_t i = 0; i < axisSize; i++) {
        const int32_t axisValue = newAxis[i];
        int64_t stepValue = 1;
        if (i < stridesSize && !GetStepValue(stridesTensor, i, stepValue)) {
            yShape->SetDim(axisValue, UNKNOWN_DIM_VALUE_);
            continue;
        }

        if (stepValue == 0) {
            OP_LOGE(OP_NAME, "idx:%d stepValue[%ld] must be non-zero", i, stepValue);
            return false;
        }

        int64_t curAxisInputSize = xShape->GetDim(axisValue);
        if (curAxisInputSize == UNKNOWN_DIM_VALUE_) {
            yShape->SetDim(axisValue, UNKNOWN_DIM_VALUE_);
            continue;
        }

        int64_t beginValue = 0;
        if (i < beginsSize && !GetBeginValue(curAxisInputSize, stepValue, beginTensor, i, beginValue)) {
            yShape->SetDim(axisValue, UNKNOWN_DIM_VALUE_);
            continue;
        }
        int64_t endValue = curAxisInputSize;
        if (i < endsSize && !GetEndValue(curAxisInputSize, stepValue, endTensor, i, endValue)) {
            yShape->SetDim(axisValue, UNKNOWN_DIM_VALUE_);
            continue;
        }
        int64_t curOutSize = Ops::Base::CeilDiv((endValue - beginValue), stepValue);
        if (curOutSize < 0) {
            curOutSize = 0;
        }
        yShape->SetDim(axisValue, curOutSize);
    }
    return true;
}

static ge::graphStatus StridedSliceV3InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Tensor* beginTensor = context->GetInputTensor(INDEX_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, beginTensor);
    const gert::Tensor* endTensor = context->GetInputTensor(INDEX_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, endTensor);
    const gert::Tensor* axesTensor = context->GetOptionalInputTensor(INDEX_AXES);
    const gert::Tensor* stridesTensor = context->GetOptionalInputTensor(INDEX_STRIDES);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }
    if (axesTensor && !IsConstTensor(axesTensor)) {
        OP_LOGD(OP_NAME, "axes is not const tensor");
        return SetAllUnknownDim(xShape->GetDimNum(), yShape);
    }

    *yShape = *xShape; // init output_shape with input_shape

    int32_t inputDimNum = static_cast<int32_t>(xShape->GetDimNum());
    std::vector<int32_t> newAxis;
    if (!ConstructValidAxis(inputDimNum, axesTensor, newAxis)) {
        return ge::GRAPH_FAILED;
    }
    const int32_t axisSize = static_cast<int32_t>(newAxis.size());
    if (axisSize == 0) {
        OP_LOGE(OP_NAME, "axisSize is 0. Please check.");
        return ge::GRAPH_FAILED;
    }

    if (!DoInferShape(xShape, yShape, stridesTensor, beginTensor, endTensor, newAxis)) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(OP_NAME, "out shape: %s", Ops::Base::ToString(*yShape).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StridedSliceV3).InferShape(StridedSliceV3InferShape).InputsDataDependency({1, 2, 3, 4});
} // namespace ops
