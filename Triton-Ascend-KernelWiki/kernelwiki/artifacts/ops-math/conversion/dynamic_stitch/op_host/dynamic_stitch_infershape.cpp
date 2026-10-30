/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_stitch_infershape.cpp
 * \brief infershape func of DynamicStitch
 */
#include <algorithm>
#include <limits>

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static constexpr size_t DYNAMICSTITCH_IDX_ZERO = 0;
static constexpr size_t DYNAMICSTITCH_IDX_IN_INDICES = 0;
static constexpr size_t DYNAMICSTITCH_IDX_IN_X = 1;
static constexpr size_t DYNAMICSTITCH_IDX_OUT_Y = 0;
static constexpr size_t DYNAMICSTITCH_IDX_ATTR_N = 0;
static constexpr int64_t DYNAMICSTITCH_UNKOWNSHAPE = -2;
static constexpr int64_t DYNAMICSTITCH_UNKNOWNDIM = -1;

static bool IsUnknownRankShape(const std::vector<int64_t>& shapeVec)
{
    if (shapeVec.size() == 1 && shapeVec[0] == DYNAMICSTITCH_UNKOWNSHAPE) {
        return true;
    }
    return false;
}

static bool IsUnKnownShape(const std::vector<int64_t>& shapeVec)
{
    auto found = std::find(shapeVec.begin(), shapeVec.end(), DYNAMICSTITCH_UNKNOWNDIM);
    return found != shapeVec.end();
}

static bool IsUnknown(const std::vector<int64_t>& shapeVec)
{
    return (IsUnKnownShape(shapeVec) || IsUnknownRankShape(shapeVec));
}

static graphStatus MergeDim(int64_t dimA, int64_t dimB, int64_t& result)
{
    if (dimA == dimB) {
        result = dimA;
        return GRAPH_SUCCESS;
    }
    if (dimA == DYNAMICSTITCH_UNKNOWNDIM) {
        result = dimB;
        return GRAPH_SUCCESS;
    }
    if (dimB == DYNAMICSTITCH_UNKNOWNDIM) {
        result = dimA;
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

static graphStatus GetNattrValue(gert::InferShapeContext* context, int64_t& numIndices)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* attrN = attrs->GetAttrPointer<int64_t>(DYNAMICSTITCH_IDX_ATTR_N);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrN);
    numIndices = *attrN;
    OP_CHECK_IF(
        numIndices < 1,
        OP_LOGE(context->GetNodeName(),
            "invalid value [%ld] of attr[N], it should be not less than 1.", numIndices),
        return GRAPH_FAILED);
    const auto indicesInfoIndices = context->GetIrInputInstanceInfo(DYNAMICSTITCH_IDX_IN_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesInfoIndices);
    const auto instanceNumIndices = indicesInfoIndices->GetInstanceNum();
    const auto indicesInfoX = context->GetIrInputInstanceInfo(DYNAMICSTITCH_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesInfoX);
    const auto instanceNumX = indicesInfoX->GetInstanceNum();
    bool isInvalid = (static_cast<size_t>(numIndices) != instanceNumIndices ||
                      static_cast<size_t>(numIndices) != instanceNumX || instanceNumIndices != instanceNumX);
    OP_CHECK_IF(
        isInvalid,
        OP_LOGE(context->GetNodeName(),
            "the tensorList size of indices and x must be same of the value of N, now indices'size is: %zu, "
            "x's size is: %zu, N's value is: %ld", instanceNumIndices, instanceNumX, numIndices),
        return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static bool CheckValidShapeSize(const gert::Shape* indicesShape, const gert::Shape* xShape, int32_t& zeroShapeNum)
{
    const int64_t indicesShapeSize = indicesShape->GetShapeSize();
    const int64_t xShapeSize = xShape->GetShapeSize();
    if (indicesShapeSize == DYNAMICSTITCH_UNKOWNSHAPE || xShapeSize == DYNAMICSTITCH_UNKOWNSHAPE) {
        return false;
    }
    if (*indicesShape == gert::Shape({DYNAMICSTITCH_IDX_ZERO})) {
        zeroShapeNum++;
        return false;
    }
    return true;
}

static graphStatus StartWith(gert::InferShapeContext* context, const gert::Shape* indicesShape,
    const gert::Shape* xShape, std::vector<int64_t>& dims, const int64_t i)
{
    const size_t indicesDimNum = indicesShape->GetDimNum();
    const size_t xDimNum = xShape->GetDimNum();
    OP_CHECK_IF(
        indicesDimNum > xDimNum,
        OP_LOGE(context->GetNodeName(),
            "the %ldth input[indices]'s dimNum must <= the %ldth input[x]'s dimNum, but now is: %zu and %zu",
            i, i, indicesDimNum, xDimNum),
        return GRAPH_FAILED);
    dims.reserve(xDimNum);
    dims.resize(indicesDimNum);
    for (size_t j = 0; j < indicesDimNum; ++j) {
        if (MergeDim(xShape->GetDim(j), indicesShape->GetDim(j), dims[j]) != GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(),
                "failed to call MergeDim function to merge the %ldth input[indices]'s dim[%ld] and the %ldth "
                "input[x]'s dim[%ld]", i, indicesShape->GetDim(j), i, xShape->GetDim(j));
            return GRAPH_FAILED;
        }
    }
    dims.resize(xDimNum - indicesDimNum);
    for (size_t j = 0; j < xDimNum - indicesDimNum; j++) {
        dims[j] = xShape->GetDim(j + indicesDimNum);
    }
    return GRAPH_SUCCESS;
}

static graphStatus SameExtraShape(gert::InferShapeContext* context, gert::Shape& commonShape,
    const std::vector<int64_t>& dims, const int64_t i)
{
    if (commonShape == gert::Shape({DYNAMICSTITCH_UNKOWNSHAPE})) {
        commonShape.SetDimNum(dims.size());
        for (size_t j = 0; j < dims.size(); j++) {
            commonShape.SetDim(j, dims[j]);
        }
    } else {
        gert::Shape currSuffixShape;
        currSuffixShape.SetDimNum(dims.size());
        for (size_t j = 0; j < dims.size(); j++) {
            currSuffixShape.SetDim(j, dims[j]);
        }
        OP_CHECK_IF(
            currSuffixShape != commonShape,
            OP_LOGE(context->GetNodeName(),
                "the suffixShape must be same, but now %ldth suffixShape is does not same.", i),
            return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static graphStatus GetInputConstData(gert::InferShapeContext* context, const gert::Shape* indicesShape,
    int32_t& maxIndex, bool& getAllIndicesData, const int64_t i)
{
    auto indicesTensor = context->GetDynamicInputTensor(DYNAMICSTITCH_IDX_IN_INDICES, i);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesTensor);
    auto indicesDtype = indicesTensor->GetDataType();
    OP_CHECK_IF(
        indicesDtype != DT_INT32,
        OP_LOGE(context->GetNodeName(),
            "the indices dtype only support int32, but now get a %s", Ops::Base::ToString(indicesDtype).c_str()),
        return GRAPH_FAILED);
    const int32_t* indicesTensorData = indicesTensor->GetData<int32_t>();
    if (indicesTensorData) {
        for (int64_t j = 0; j < indicesShape->GetShapeSize(); j++) {
            int32_t currValue = static_cast<int32_t>(*(indicesTensorData + j));
            OP_CHECK_IF(
                currValue < 0,
                OP_LOGE(context->GetNodeName(),
                    "the indices'value must be a positive value, but got %d", currValue),
                return GRAPH_FAILED);
            maxIndex = std::max(maxIndex, currValue);
        }
    } else {
        getAllIndicesData = false;
    }
    return GRAPH_SUCCESS;
}

static void SetOuputShape(gert::Shape* yShape, const bool& getAllIndicesData, const int32_t& maxIndex,
    const gert::Shape& commonShape)
{
    int64_t outputDim0 = getAllIndicesData ? (static_cast<int64_t>(maxIndex) + 1) : (DYNAMICSTITCH_UNKNOWNDIM);
    size_t outputDimNum = commonShape.GetDimNum() + 1;
    yShape->SetDimNum(outputDimNum);
    yShape->SetDim(DYNAMICSTITCH_IDX_ZERO, outputDim0);
    for (size_t j = 1; j < outputDimNum; j++) {
        yShape->SetDim(j, commonShape.GetDim(j - 1));
    }
}

struct DynamicStitchState {
    gert::Shape commonShape;
    bool getAllIndicesData;
    int32_t maxIndex;
    int32_t zeroShapeNum;
};

static graphStatus InferAllStitchIndices(gert::InferShapeContext* context, const int64_t numIndices,
    DynamicStitchState& state, gert::Shape* yShape, bool& earlyReturn)
{
    earlyReturn = false;
    for (int64_t i = 0; i < numIndices; i++) {
        auto indicesShape = context->GetDynamicInputShape(DYNAMICSTITCH_IDX_IN_INDICES, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
        auto xShape = context->GetDynamicInputShape(DYNAMICSTITCH_IDX_IN_X, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
        if (!CheckValidShapeSize(indicesShape, xShape, state.zeroShapeNum)) {
            continue;
        }
        std::vector<int64_t> dims;
        OP_CHECK_IF(
            StartWith(context, indicesShape, xShape, dims, i) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "StartWith failed."),
            return GRAPH_FAILED);
        if (IsUnknown(dims)) {
            yShape->SetDimNum(1);
            yShape->SetDim(0, DYNAMICSTITCH_UNKOWNSHAPE);
            earlyReturn = true;
            return GRAPH_SUCCESS;
        }
        OP_CHECK_IF(
            SameExtraShape(context, state.commonShape, dims, i) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "SameExtraShape failed."),
            return GRAPH_FAILED);
        OP_CHECK_IF(
            GetInputConstData(context, indicesShape, state.maxIndex, state.getAllIndicesData, i) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "GetInputConstData failed."),
            return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferShape4DynamicStitch(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4DynamicStitch enter");
    int64_t numIndices = 0;
    OP_CHECK_IF(
        GetNattrValue(context, numIndices) != GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "GetNattrValue failed."),
        return GRAPH_FAILED);
    DynamicStitchState state = {gert::Shape({DYNAMICSTITCH_UNKOWNSHAPE}), true, -1, 0};
    auto yShape = context->GetOutputShape(DYNAMICSTITCH_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    bool earlyReturn = false;
    OP_CHECK_IF(
        InferAllStitchIndices(context, numIndices, state, yShape, earlyReturn) != GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "InferAllStitchIndices failed."),
        return GRAPH_FAILED);
    if (earlyReturn) {
        return GRAPH_SUCCESS;
    }
    if (state.zeroShapeNum == numIndices) {
        *yShape = gert::Shape({DYNAMICSTITCH_IDX_ZERO});
        return GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        state.maxIndex >= std::numeric_limits<int32_t>::max(),
        OP_LOGE(context->GetNodeName(),
            "the maxIndex must be less than the maximum value of int32, now got %d", state.maxIndex),
        return GRAPH_FAILED);
    SetOuputShape(yShape, state.getAllIndicesData, state.maxIndex, state.commonShape);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4DynamicStitch(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4DynamicStitch enter");
    auto xDtype = context->GetDynamicInputDataType(DYNAMICSTITCH_IDX_IN_X, DYNAMICSTITCH_IDX_ZERO);
    context->SetOutputDataType(DYNAMICSTITCH_IDX_OUT_Y, xDtype);
    OP_LOGD(context->GetNodeName(), "InferDataType4DynamicStitch end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicStitch)
    .InputsDataDependency({DYNAMICSTITCH_IDX_IN_INDICES})
    .InferShape(InferShape4DynamicStitch)
    .InferDataType(InferDataType4DynamicStitch);
}  // namespace ops
