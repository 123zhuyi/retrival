/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file strided_slice_grad_tiling.cc
 * \brief
 */

#include <vector>
#include <cmath>
#include <tuple>
#include "strided_slice_grad_tiling_arch35.h"

using namespace std;
using namespace ge;

namespace optiling {
// base param
constexpr int64_t WORKSPACE_SIZE = 16777216; // 16 * 1024 * 1024
constexpr int64_t RESERVED_SPACE = 40960;    // 40 * 1024
constexpr int64_t DOUBLE_UB_NUM = 2;
constexpr int64_t ONE_BLK_BYTES = 32;
constexpr int64_t BYTE_THRESHOLD_TAIL_AXIS = 128;
constexpr size_t MAX_DIM_NUM = 8;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t DIGIT_HUNDRED = 100;
constexpr int64_t ALL_CLEAR_MODE = 1;
constexpr int64_t NO_CLEAR_MODE = 2;
constexpr int64_t LARGE_BLOCK_FOR_MOVE = 1024;
constexpr int64_t MAX_NDDMA_DIM = 5;

// byte size
constexpr int64_t BYTE_PER_DATA_1 = 1;
constexpr int64_t BYTE_PER_DATA_4 = 4;
constexpr int64_t BYTE_PER_DATA_2 = 2;
constexpr int64_t BYTE_PER_DATA_8 = 8;

// define calcu mode
constexpr int64_t MODE_EMPTY_TENSOR = 0;
constexpr int64_t MODE_SIMT = 1;
constexpr int64_t MODE_SPLIT_FRONT_MOVE_ALIGN = 2;
constexpr int64_t MODE_SPLIT_ALL_MOVE_ALIGN = 3;
constexpr int64_t MODE_SPLIT_FRONT_NDDMA = 4;
constexpr int64_t MODE_SPLIT_ALL_NDDMA = 5;

// define input & attr & output idx
constexpr int64_t IN_SHAPE_IDX = 0;
constexpr int64_t IN_BEGIN_IDX = 1;
constexpr int64_t IN_END_IDX = 2;
constexpr int64_t IN_STRIDES_IDX = 3;
constexpr int64_t IN_DY_IDX = 4;

constexpr int64_t ATTR_BEGIN_IDX = 0;
constexpr int64_t ATTR_END_IDX = 1;
constexpr int64_t ATTR_ELLIPSIS_IDX = 2;
constexpr int64_t ATTR_NEW_AXIS_IDX = 3;
constexpr int64_t ATTR_SHRINK_AXIS_IDX = 4;
constexpr int64_t OUT_PUT_IDX = 0;

constexpr int64_t TUPLE_INDEX_0 = 0;
constexpr int64_t TUPLE_INDEX_1 = 1;
constexpr int64_t TUPLE_INDEX_2 = 2;

constexpr uint32_t BATCH_MODE = 1;
constexpr int64_t SIMT_DCACHE_SIZE = static_cast<int64_t>(32 * 1024);

template <typename T>
void GetIntValue(const gert::Tensor* constTensor, gert::Shape& constShape)
{
    const T* constValue = constTensor->GetData<T>();
    const size_t constNum = constTensor->GetShapeSize();
    constShape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        constShape.AppendDim(constValue[i]);
    }
}

template <typename T>
bool GetIntToShape(T* context, const int64_t constIdx, gert::Shape& constShape)
{
    auto constTensor = context->GetInputTensor(constIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, constTensor);

    auto inputDescPtr = context->GetInputDesc(constIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescPtr);
    auto constDtype = inputDescPtr->GetDataType();

    switch (constDtype) {
        case ge::DT_INT32: {
            GetIntValue<int32_t>(constTensor, constShape);
            break;
        }
        case ge::DT_INT64: {
            GetIntValue<int64_t>(constTensor, constShape);
            break;
        }
        case ge::DT_UINT64: {
            GetIntValue<uint64_t>(constTensor, constShape);
            break;
        }
        case ge::DT_UINT32: {
            GetIntValue<uint32_t>(constTensor, constShape);
            break;
        }
        default:
            OP_LOGW(
                context->GetNodeName(), "GetConstIntToShape only support [int32, int64, uint64, uint32]. but is %s",
                Ops::Base::ToString(constDtype).c_str());
            return false;
    }
    OP_LOGI(context->GetNodeName(), "GetConstIntToShape: output shape is %s", Ops::Base::ToString(constShape).c_str());
    return true;
}

static inline ge::graphStatus StridedSliceGradSaveTilingData(
    gert::TilingContext* context, StridedSliceGradTilingData& tilingData)
{
    if (tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static inline void TilingDataToLogging(StridedSliceGradTilingData& tilingData)
{
    OP_LOGI(
        "[StridedSliceGrad]",
        "totalCoreNum: %ld, usedCoreNum: %ld, bufferSize: %ld, tailAxisOuter: %ld, \
tailAxisInner: %ld, tailAxisTail: %ld, normalCoreProcessNum: %ld, tailCoreProcessNum: %ld,  \
usedCoreNumForClear: %ld, normalCoreProcessNumForClear: %ld, tailCoreProcessNumForClear: %ld, splitUbAxisNum: %ld,  \
bytesForOneData: %ld, outputShape: %s, begin: %s, end: %s, strides: %s, inputShape: %s, fusedOutputInnerShape: %s,  \
fusedSliceInnerShape: %s, tilingKey: %ld, workspaceSize: %ld, inputDimNum: %ld",
        tilingData.get_totalCoreNum(), tilingData.get_usedCoreNum(), tilingData.get_bufferSize(),
        tilingData.get_tailAxisOuter(), tilingData.get_tailAxisInner(), tilingData.get_tailAxisTail(),
        tilingData.get_normalCoreProcessNum(), tilingData.get_tailCoreProcessNum(),
        tilingData.get_usedCoreNumForClear(), tilingData.get_normalCoreProcessNumForClear(),
        tilingData.get_tailCoreProcessNumForClear(), tilingData.get_splitUbAxisNum(), tilingData.get_bytesForOneData(),
        ops::ToStringWithSize(tilingData.get_outputShape(), TILING_ARRAY_LEN_EIGHT).c_str(),
        ops::ToStringWithSize(tilingData.get_begin(), TILING_ARRAY_LEN_EIGHT).c_str(),
        ops::ToStringWithSize(tilingData.get_end(), TILING_ARRAY_LEN_EIGHT).c_str(),
        ops::ToStringWithSize(tilingData.get_strides(), TILING_ARRAY_LEN_EIGHT).c_str(),
        ops::ToStringWithSize(tilingData.get_inputShape(), TILING_ARRAY_LEN_EIGHT).c_str(),
        ops::ToStringWithSize(tilingData.get_fusedOutputInnerShape(), TILING_ARRAY_LEN_EIGHT).c_str(),
        ops::ToStringWithSize(tilingData.get_fusedSliceInnerShape(), TILING_ARRAY_LEN_EIGHT).c_str(),
        tilingData.get_tilingKey(), tilingData.get_workspaceSize(), tilingData.get_inputDimNum());
}

static inline int64_t GetBytePerData(const ge::DataType& dtype)
{
    return static_cast<int64_t>(ge::GetSizeByDataType(dtype));
}

static inline bool CheckTypeIsInvalid(
    ge::DataType& inShape, ge::DataType& begin, ge::DataType& end, ge::DataType& stride, ge::DataType& dy)
{
    std::set<ge::DataType> supportedIndexDtype = {ge::DT_INT32, ge::DT_INT64};
    std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64, ge::DT_UINT64, ge::DT_INT32,    ge::DT_UINT32,
        ge::DT_INT16, ge::DT_UINT16,  ge::DT_INT8, ge::DT_UINT8, ge::DT_DOUBLE, ge::DT_COMPLEX64};

    bool inShapeInValid = (supportedIndexDtype.count(inShape) == 0);
    bool beginInValid = (supportedIndexDtype.count(begin) == 0);
    bool endInValid = (supportedIndexDtype.count(end) == 0);
    bool strideInValid = (supportedIndexDtype.count(stride) == 0);
    bool dyInValid = (supportedDtype.count(dy) == 0);

    return inShapeInValid || beginInValid || endInValid || strideInValid || dyInValid;
}

static void RevertSliceParamByInferShape(
    const ops::StridedSliceParams& inputParams, const gert::Shape& shapeOutput, SliceParametersRuntime& sliceParam)
{
    sliceParam.beginList = inputParams.begin;
    sliceParam.endList = inputParams.end;
    sliceParam.strideList = inputParams.strides;
    sliceParam.inputShape = inputParams.input_shape;
    sliceParam.outputShape = shapeOutput;
}

static void MakeInputParamSameDims(SliceParametersRuntime& parameters)
{
    bool sameSize = parameters.inputShape.GetDimNum() == parameters.beginList.GetDimNum() &&
                    parameters.inputShape.GetDimNum() == parameters.endList.GetDimNum() &&
                    parameters.inputShape.GetDimNum() == parameters.strideList.GetDimNum();
    if (!sameSize) {
        return;
    }
    parameters.outputShape.SetDimNum(0);
    for (size_t i = 0; i < parameters.inputShape.GetDimNum(); i++) {
        auto interval = parameters.endList[i] - parameters.beginList[i];
        auto strideI = parameters.strideList[i];
        if (strideI == 0) {
            strideI = 1;
        }
        int64_t outputSize = interval / strideI + (interval % strideI != 0 ? 1 : 0);
        parameters.outputShape.AppendDim(outputSize);
    }
}

static void MakePerformanceSliceParams(SliceParametersRuntime& param)
{
    if (param.outputShape.GetDimNum() == 0) {
        return;
    }
    size_t th = param.inputShape.GetDimNum();
    const auto inputLastDim = param.inputShape[th - 1];
    const auto beginLastDim = param.beginList[th - 1];
    const auto strideLastDim = param.strideList[th - 1];
    const auto outputLastDim = param.outputShape[th - 1];
    if (strideLastDim > 1 && inputLastDim % strideLastDim == 0 && beginLastDim / strideLastDim == 0 &&
        outputLastDim == inputLastDim / strideLastDim) {
        param.inputShape[th - 1] = inputLastDim / strideLastDim;
        param.inputShape.AppendDim(strideLastDim);
        param.beginList[th - 1] = beginLastDim / strideLastDim;
        param.beginList.AppendDim(beginLastDim % strideLastDim);
        param.strideList[th - 1] = 1;
        param.strideList.AppendDim(1);
        param.outputShape.AppendDim(1);
        param.endList[th - 1] = param.beginList[th - 1] + outputLastDim;
        param.endList.AppendDim(param.beginList[th] + 1);
    }

    optiling::SliceParametersRuntime perfParams;
    size_t perfSize = 0;
    for (size_t i = 0; i < param.inputShape.GetDimNum(); i++) {
        const auto inputShapeI = param.inputShape[i];
        const auto outputShapeI = param.outputShape[i];
        const auto beginI = param.beginList[i];
        const auto endI = param.endList[i];
        const auto strideI = endI > beginI ? std::min(param.strideList[i], endI - beginI) : param.strideList[i];
        if (i == 0 || inputShapeI != outputShapeI || strideI != 1 || perfParams.strideList[perfSize - 1] != 1) {
            perfParams.inputShape.AppendDim(inputShapeI);
            perfParams.outputShape.AppendDim(outputShapeI);
            perfParams.beginList.AppendDim(beginI);
            perfParams.endList.AppendDim(endI);
            perfParams.strideList.AppendDim(strideI);
            perfSize++;
            continue;
        }

        const auto perfIndex = perfSize - 1;
        perfParams.inputShape[perfIndex] *= inputShapeI;
        perfParams.outputShape[perfIndex] *= outputShapeI;
        perfParams.beginList[perfIndex] *= inputShapeI;
        perfParams.endList[perfIndex] *= inputShapeI;
        perfParams.strideList[perfIndex] = 1;
    }
    param = perfParams;
}

static void UpdateInitParams(ops::StridedSliceParams& initParam, const SliceParametersRuntime& param)
{
    initParam.input_shape = param.inputShape;
    initParam.begin = param.beginList;
    initParam.end = param.endList;
    initParam.strides = param.strideList;
    initParam.dy_shape = param.outputShape;
}

static inline ge::graphStatus GetInputParamAndAttrValue(
    const gert::TilingContext* context, ops::StridedSliceParams& initParam)
{
    initParam.begin_valid = true;
    initParam.end_valid = true;
    initParam.stride_valid = true;
    OP_CHECK_IF(
        !GetIntToShape(context, IN_SHAPE_IDX, initParam.input_shape),
        OP_LOGE(context->GetNodeName(), "Get const value of shape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !GetIntToShape(context, IN_BEGIN_IDX, initParam.begin),
        OP_LOGE(context->GetNodeName(), "Get const value of begin failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !GetIntToShape(context, IN_END_IDX, initParam.end),
        OP_LOGE(context->GetNodeName(), "Get const value of end failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !GetIntToShape(context, IN_STRIDES_IDX, initParam.strides),
        OP_LOGE(context->GetNodeName(), "Get const value of strides failed"), return ge::GRAPH_FAILED);

    OP_LOGD(
        context->GetNodeName(), "before infershape the shape = %s", Ops::Base::ToString(initParam.input_shape).c_str());
    OP_LOGD(context->GetNodeName(), "before infershape the begin = %s", Ops::Base::ToString(initParam.begin).c_str());
    OP_LOGD(context->GetNodeName(), "before infershape the end = %s", Ops::Base::ToString(initParam.end).c_str());
    OP_LOGD(
        context->GetNodeName(), "before infershape the strides = %s", Ops::Base::ToString(initParam.strides).c_str());

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* maskBegin = attrs->GetAttrPointer<int64_t>(ATTR_BEGIN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskBegin);
    const int64_t* maskEnd = attrs->GetAttrPointer<int64_t>(ATTR_END_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskEnd);
    const int64_t* maskEllipsis = attrs->GetAttrPointer<int64_t>(ATTR_ELLIPSIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskEllipsis);
    const int64_t* maskNewAxis = attrs->GetAttrPointer<int64_t>(ATTR_NEW_AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskNewAxis);
    const int64_t* maskShrinkAxis = attrs->GetAttrPointer<int64_t>(ATTR_SHRINK_AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShrinkAxis);
    initParam.begin_mask = static_cast<uint64_t>(*maskBegin);
    initParam.end_mask = static_cast<uint64_t>(*maskEnd);
    initParam.ellipsis_mask = static_cast<uint64_t>(*maskEllipsis);
    initParam.new_axis_mask = static_cast<uint64_t>(*maskNewAxis);
    initParam.shrink_axis_mask = static_cast<uint64_t>(*maskShrinkAxis);

    gert::Shape shapeY;
    OP_CHECK_IF(
        !ops::InferShape(initParam, &shapeY), OP_LOGE(context->GetNodeName(), "do strided slice infershape failed"),
        return ge::GRAPH_FAILED);

    // revert sliceParam by infer shape
    optiling::SliceParametersRuntime sliceParam;
    RevertSliceParamByInferShape(initParam, shapeY, sliceParam);

    // align slice param dims.
    MakeInputParamSameDims(sliceParam);
    OP_LOGD(context->GetNodeName(), "align slice params: %s", sliceParam.to_string().c_str());

    // optimize formance slice param
    MakePerformanceSliceParams(sliceParam);
    OP_LOGD(context->GetNodeName(), "performance slice params: %s", sliceParam.to_string().c_str());

    // update init param
    UpdateInitParams(initParam, sliceParam);
    OP_LOGD(
        context->GetNodeName(), "after infershape the shape = %s", Ops::Base::ToString(initParam.input_shape).c_str());
    OP_LOGD(context->GetNodeName(), "after infershape the begin = %s", Ops::Base::ToString(initParam.begin).c_str());
    OP_LOGD(context->GetNodeName(), "after infershape the end = %s", Ops::Base::ToString(initParam.end).c_str());
    OP_LOGD(
        context->GetNodeName(), "after infershape the strides = %s", Ops::Base::ToString(initParam.strides).c_str());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateBeginEndStrides(
    const gert::TilingContext* context, int64_t beginI, int64_t endI, int64_t strideI)
{
    if (strideI != 0) {
        OP_CHECK_IF(
            (beginI < endI && strideI < 0),
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(),
                "begin, end, strides",
                (std::to_string(beginI) + ", " + std::to_string(endI) + ", " + std::to_string(strideI)).c_str(),
                "If the value of strides is less than 0, the value of begin must be greater than or equal to that of end"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (beginI > endI && strideI > 0),
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(),
                "begin, end, strides",
                (std::to_string(beginI) + ", " + std::to_string(endI) + ", " + std::to_string(strideI)).c_str(),
                "If the value of strides is greater than 0, the value of begin must be less than or equal to that of end."),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (strideI == 0),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(),
                "strides",
                std::to_string(strideI).c_str(),
                "The value of strides must be non-zero."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateMasks(const gert::TilingContext* context, const StridedSliceGradParamList& inputParams)
{
    OP_CHECK_IF(
        (inputParams.beginMask < 0 || inputParams.endMask < 0 || inputParams.ellipsisMask < 0 ||
         inputParams.newAxisMask < 0 || inputParams.shrinkAxisMask < 0),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(),
            "begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask",
            (std::to_string(inputParams.beginMask) + ", " + std::to_string(inputParams.endMask) + ", " +
             std::to_string(inputParams.ellipsisMask) + ", " + std::to_string(inputParams.newAxisMask) + ", " +
             std::to_string(inputParams.shrinkAxisMask)).c_str(),
            "The values of begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask must be greater than or equal to 0."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputParamIsValid(
    const gert::TilingContext* context, StridedSliceGradParamList& inputParams)
{
    const size_t rank = inputParams.begin.GetDimNum();
    for (size_t i = 0; i < rank; i++) {
        int64_t beginI = inputParams.begin[i];
        int64_t endI = inputParams.end[i];
        int64_t strideI = inputParams.strides[i];
        
        if (ValidateBeginEndStrides(context, beginI, endI, strideI) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        
        if (ValidateMasks(context, inputParams) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    auto dimNum = inputParams.inShape.GetDimNum();
    OP_CHECK_IF(
        inputParams.begin.GetDimNum() != dimNum,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(),
            "begin",
            std::to_string(inputParams.begin.GetDimNum()).c_str(),
            "The shape dim of begin must be equal to the shape dim of input_shape."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "dimNum = %ld", dimNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus UpdateStructInputParam(
    const gert::TilingContext* context, const ops::StridedSliceParams& initParam,
    StridedSliceGradParamList& inputParams)
{
    size_t rank = initParam.input_shape.GetDimNum();
    int64_t inputDimNum = 0;
    for (size_t i = 0; i < rank; i++) {
        if (initParam.input_shape[i] != 1) {
            inputDimNum += 1;
        }
    }

    inputParams.inputDimNum = (inputDimNum == 0) ? rank : inputDimNum;
    inputParams.inShape.SetDimNum(inputParams.inputDimNum);
    inputParams.begin.SetDimNum(inputParams.inputDimNum);
    inputParams.end.SetDimNum(inputParams.inputDimNum);
    inputParams.strides.SetDimNum(inputParams.inputDimNum);
    inputParams.dyShape.SetDimNum(inputParams.inputDimNum);
    int32_t inputIdx = 0;
    for (size_t i = 0; i < rank; i++) {
        if (initParam.input_shape[i] == 1 && inputDimNum != 0) {
            continue;
        }
        inputParams.inShape.SetDim(inputIdx, initParam.input_shape[i]);
        inputParams.begin.SetDim(inputIdx, initParam.begin[i]);
        inputParams.end.SetDim(inputIdx, initParam.end[i]);
        inputParams.strides.SetDim(inputIdx, initParam.strides[i]);
        inputParams.dyShape.SetDim(inputIdx, initParam.dy_shape[i]);
        inputIdx += 1;
    }

    inputParams.beginMask = initParam.begin_mask;
    inputParams.endMask = initParam.end_mask;
    inputParams.ellipsisMask = initParam.ellipsis_mask;
    inputParams.newAxisMask = initParam.new_axis_mask;
    inputParams.shrinkAxisMask = initParam.shrink_axis_mask;
    inputParams.beginValid = initParam.begin_valid;
    inputParams.endValid = initParam.end_valid;
    inputParams.strideValid = initParam.stride_valid;

    OP_LOGD(context->GetNodeName(), "after reverse the shape = %s", Ops::Base::ToString(inputParams.inShape).c_str());
    OP_LOGD(context->GetNodeName(), "after reverse the begin = %s", Ops::Base::ToString(inputParams.begin).c_str());
    OP_LOGD(context->GetNodeName(), "after reverse the end = %s", Ops::Base::ToString(inputParams.end).c_str());
    OP_LOGD(context->GetNodeName(), "after reverse the strides = %s", Ops::Base::ToString(inputParams.strides).c_str());
    OP_LOGD(context->GetNodeName(), "after reverse the dyShape = %s", Ops::Base::ToString(inputParams.dyShape).c_str());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetOfInputParams(
    const gert::TilingContext* context, StridedSliceGradParamList& inputParams, ops::StridedSliceParams& initParam)
{
    auto inputDescShapePtr = context->GetInputDesc(IN_SHAPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescShapePtr);
    auto inputDescBeginPtr = context->GetInputDesc(IN_BEGIN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescBeginPtr);
    auto inputDescEndPtr = context->GetInputDesc(IN_END_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescEndPtr);
    auto inputDescStridedPtr = context->GetInputDesc(IN_STRIDES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescStridedPtr);
    auto inputDescDyPtr = context->GetInputDesc(IN_DY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescDyPtr);

    auto dataTypeInShape = inputDescShapePtr->GetDataType();
    auto dataTypeInBegin = inputDescBeginPtr->GetDataType();
    auto dataTypeInEnd = inputDescEndPtr->GetDataType();
    auto dataTypeInStrides = inputDescStridedPtr->GetDataType();
    auto dataTypeInDy = inputDescDyPtr->GetDataType();
    inputParams.IndexDtype = dataTypeInShape;
    inputParams.dtype = dataTypeInDy;

    // 检查输入参数类型
    if (CheckTypeIsInvalid(dataTypeInShape, dataTypeInBegin, dataTypeInEnd, dataTypeInStrides, dataTypeInDy)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context->GetNodeName(),
            "input_shape, begin, end, strides",
            (Ops::Base::ToString(dataTypeInShape) + ", " + Ops::Base::ToString(dataTypeInBegin) + ", " +
             Ops::Base::ToString(dataTypeInEnd) + ", " + Ops::Base::ToString(dataTypeInStrides)).c_str(),
            "The dtypes of input_shape, begin, end, strides must be within the range [DT_INT32, DT_INT64].");
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(),
            "dy",
            Ops::Base::ToString(dataTypeInDy).c_str(),
            "The dtype of dy must be within the range [DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT64, DT_UINT64, "
            "DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX64].");
        return ge::GRAPH_FAILED;
    }

    // 设置StridedSliceGrad算子中的输入值 && 属性值
    OP_CHECK_IF(
        GetInputParamAndAttrValue(context, initParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "fail to get tiling input param or attr."), return ge::GRAPH_FAILED);

    // 获取StridedSliceParams -> StridedSliceGradParamList算子中的传递输入值, 并去除shape中含1的项
    OP_CHECK_IF(
        UpdateStructInputParam(context, initParam, inputParams) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "fail to correct tiling input param."), return ge::GRAPH_FAILED);

    // 校验StridedSliceGrad算子中的输入值
    OP_CHECK_IF(
        CheckInputParamIsValid(context, inputParams) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "check input tiling param."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static inline void GetTilingKey(StridedSliceGradParamList& inputParams)
{
    // tilingkey使用3位数表示，个位数由数据类型所占字节决定(1/2/4/8)，十位数由计算模式决定(1/2/3)，百位数由清零模式决定(1/2)
    int64_t hundredDigit =
        (inputParams.caluMode == MODE_SPLIT_FRONT_NDDMA || inputParams.caluMode == MODE_SPLIT_ALL_NDDMA) ?
            NO_CLEAR_MODE :
            ALL_CLEAR_MODE;
    int64_t tenDigit = inputParams.caluMode;
    int64_t digit = inputParams.bytesForOneData;
    inputParams.tilingKey = hundredDigit * DIGIT_HUNDRED + tenDigit * DIGIT_TEN + digit;
}

static inline void SetTilingParam(StridedSliceGradTilingData& tilingData, StridedSliceGradParamList& inputParams)
{
    tilingData.set_normalCoreProcessNum(inputParams.normalCoreProcessNum);
    tilingData.set_tailCoreProcessNum(inputParams.tailCoreProcessNum);
    tilingData.set_tailAxisOuter(inputParams.tailAxisOuter);
    tilingData.set_tailAxisInner(inputParams.tailAxisInner);
    tilingData.set_tailAxisTail(inputParams.tailAxisTail);
    tilingData.set_inputDimNum(inputParams.inputDimNum);
    tilingData.set_usedCoreNumForClear(inputParams.usedCoreNumForClear);
    tilingData.set_normalCoreProcessNumForClear(inputParams.normalCoreProcessNumForClear);
    tilingData.set_tailCoreProcessNumForClear(inputParams.tailCoreProcessNumForClear);
    tilingData.set_splitUbAxisNum(inputParams.splitUbAxisNum);
    tilingData.set_usedCoreNum(inputParams.usedCoreNum);
    tilingData.set_totalCoreNum(inputParams.totalCoreNum);
    tilingData.set_bufferSize(inputParams.bufferSize);
    tilingData.set_bytesForOneData(inputParams.bytesForOneData);
    tilingData.set_tilingKey(inputParams.tilingKey);
    tilingData.set_workspaceSize(inputParams.workspaceSize);

    inputParams.inShapeVec = ops::ToVector(inputParams.inShape);
    inputParams.beginVec = ops::ToVector(inputParams.begin);
    inputParams.endVec = ops::ToVector(inputParams.end);
    inputParams.stridesVec = ops::ToVector(inputParams.strides);
    inputParams.dyShapeVec = ops::ToVector(inputParams.dyShape);

    int64_t lengthIsEightArray[TILING_ARRAY_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::copy(inputParams.inShapeVec.begin(), inputParams.inShapeVec.end(), lengthIsEightArray);
    tilingData.set_outputShape(lengthIsEightArray);
    std::copy(inputParams.dyShapeVec.begin(), inputParams.dyShapeVec.end(), lengthIsEightArray);
    tilingData.set_inputShape(lengthIsEightArray);
    std::copy(inputParams.fusedInShape.begin(), inputParams.fusedInShape.end(), lengthIsEightArray);
    tilingData.set_fusedOutputInnerShape(lengthIsEightArray);
    std::copy(inputParams.fusedDyShape.begin(), inputParams.fusedDyShape.end(), lengthIsEightArray);
    tilingData.set_fusedSliceInnerShape(lengthIsEightArray);
    std::copy(inputParams.beginVec.begin(), inputParams.beginVec.end(), lengthIsEightArray);
    tilingData.set_begin(lengthIsEightArray);
    std::copy(inputParams.endVec.begin(), inputParams.endVec.end(), lengthIsEightArray);
    tilingData.set_end(lengthIsEightArray);
    std::copy(inputParams.stridesVec.begin(), inputParams.stridesVec.end(), lengthIsEightArray);
    tilingData.set_strides(lengthIsEightArray);
}

static void CaluBaseParams(StridedSliceGradParamList& inputParams)
{
    inputParams.bytesForOneData = GetBytePerData(inputParams.dtype);
    inputParams.bufferSize =
        Ops::Base::FloorAlign((inputParams.hardwareUbSize - RESERVED_SPACE) / DOUBLE_UB_NUM, ONE_BLK_BYTES);
    inputParams.maxUbAvailable = inputParams.bufferSize / inputParams.bytesForOneData;

    inputParams.inShapeOutSize = 1;
    inputParams.dyShapeOutSize = 1;
    for (int64_t i = 0; i < inputParams.inputDimNum; i++) {
        inputParams.inShapeOutSize *= inputParams.inShape[i];
        inputParams.dyShapeOutSize *= inputParams.dyShape[i];
    }
    inputParams.dyTailAxisLen = inputParams.dyShape[inputParams.inputDimNum - 1];
    inputParams.dyFusedFront = Ops::Base::FloorDiv(inputParams.dyShapeOutSize, inputParams.dyTailAxisLen);
    inputParams.inTailAxisLen = inputParams.inShape[inputParams.inputDimNum - 1];
    inputParams.inFusedFront = Ops::Base::FloorDiv(inputParams.inShapeOutSize, inputParams.inTailAxisLen);
}

static inline bool JudgeIsNddmaMode(StridedSliceGradParamList& inputParams)
{
    if (inputParams.strides[inputParams.inputDimNum - 1] == 1) {
        return false;
    }
    // 判断是否走nddma分支
    bool isNddma = true;
    size_t rank = inputParams.strides.GetDimNum();
    for (size_t i = 0; i < rank; i++) {
        if (inputParams.strides[i] < 0) {
            isNddma = false;
            break;
        }
    }
    return isNddma;
}

static inline bool JudgeIsMoveAlignMode(StridedSliceGradParamList& inputParams)
{
    // 判断是否走move_align分支
    bool isMoveAlign = false;
    size_t rank = inputParams.strides.GetDimNum();
    if (inputParams.strides[rank - 1] == 1) {
        isMoveAlign = true;
    }
    return isMoveAlign;
}

static void CaluModeParam(StridedSliceGradParamList& inputParams)
{
    if (inputParams.dyShape.GetShapeSize() == 0) {
        inputParams.caluMode = MODE_EMPTY_TENSOR;
        return;
    }
    size_t shapeDim = inputParams.dyShape.GetDimNum();
    if (JudgeIsMoveAlignMode(inputParams)) {
        if (inputParams.dyShape[shapeDim - 1] <= inputParams.maxUbAvailable) {
            inputParams.caluMode = MODE_SPLIT_FRONT_MOVE_ALIGN;
        } else {
            inputParams.caluMode = MODE_SPLIT_ALL_MOVE_ALIGN;
        }
        return;
    }

    if (JudgeIsNddmaMode(inputParams)) {
        if (inputParams.inShape[shapeDim - 1] <= inputParams.maxUbAvailable) {
            inputParams.caluMode = MODE_SPLIT_FRONT_NDDMA;
        } else {
            inputParams.caluMode = MODE_SPLIT_ALL_NDDMA;
        }
        return;
    }

    inputParams.caluMode = MODE_SIMT;
    return;
}

static void SearchBestAxisNumInUb(StridedSliceGradParamList& inputParams)
{
    int64_t bestAxisNum = 1;
    int64_t innerSize = inputParams.inTailAxisLen;
    int64_t outerSize = inputParams.inShapeOutSize / inputParams.inTailAxisLen;
    size_t maxSplitAxis = std::min(inputParams.inputDimNum, MAX_NDDMA_DIM);

    for (size_t i = 1; i < maxSplitAxis; i++) {
        innerSize *= inputParams.inShape[inputParams.inputDimNum - 1 - i];
        if (innerSize <= inputParams.maxUbAvailable) {
            outerSize /= inputParams.inShape[inputParams.inputDimNum - 1 - i];
            if (outerSize >= inputParams.totalCoreNum) {
                bestAxisNum++;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    inputParams.splitUbAxisNum = bestAxisNum;
}

static int64_t ComputeMaxUsedCoreNum(StridedSliceGradParamList& inputParams)
{
    int64_t maxUsedCoreNum = 0;
    if (inputParams.caluMode == MODE_EMPTY_TENSOR) {
        maxUsedCoreNum = inputParams.usedCoreNumForClear;
    } else if (inputParams.caluMode == MODE_SPLIT_FRONT_NDDMA || inputParams.caluMode == MODE_SPLIT_ALL_NDDMA) {
        maxUsedCoreNum = inputParams.usedCoreNum;
    } else {
        maxUsedCoreNum = std::max(inputParams.usedCoreNumForClear, inputParams.usedCoreNum);
    }
    return maxUsedCoreNum;
}

std::tuple<int64_t, int64_t, int64_t> AverageTilingLargeBlock(int64_t fusedNum, int64_t totalCoreNum, int64_t bitWidth)
{
    int64_t coreNum = 0;
    int64_t normalCoreProcessNum = 0;
    int64_t tailCoreProcessNum = 0;
    if (fusedNum * bitWidth < LARGE_BLOCK_FOR_MOVE) { // 大块搬运，按照单次搬运不小于1024Byte计算
        coreNum = 1;
        normalCoreProcessNum = fusedNum;
        tailCoreProcessNum = fusedNum;
    } else {
        normalCoreProcessNum = Ops::Base::CeilDiv(fusedNum, totalCoreNum);
        normalCoreProcessNum = (normalCoreProcessNum < Ops::Base::FloorDiv(LARGE_BLOCK_FOR_MOVE, bitWidth)) ?
                                   Ops::Base::FloorDiv(LARGE_BLOCK_FOR_MOVE, bitWidth) :
                                   normalCoreProcessNum;
        coreNum = Ops::Base::CeilDiv(fusedNum, normalCoreProcessNum);
        tailCoreProcessNum = fusedNum - normalCoreProcessNum * (coreNum - 1);
    }
    return std::make_tuple(coreNum, normalCoreProcessNum, tailCoreProcessNum);
}

std::tuple<int64_t, int64_t, int64_t> AverageTiling(int64_t fusedNum, int64_t totalCoreNum)
{
    int64_t normalCore = Ops::Base::CeilDiv(fusedNum, totalCoreNum);
    int64_t coreNum = Ops::Base::CeilDiv(fusedNum, normalCore);
    int64_t tailCore = fusedNum - normalCore * (coreNum - 1);
    return std::make_tuple(coreNum, normalCore, tailCore);
}

std::pair<int64_t, int64_t> SplitInputTailAxis(StridedSliceGradParamList& inputParams)
{
    int64_t tailAxisInner = Ops::Base::FloorDiv(LARGE_BLOCK_FOR_MOVE, inputParams.bytesForOneData);
    int64_t tailAxisOuter = Ops::Base::CeilDiv(inputParams.dyTailAxisLen, tailAxisInner);
    int64_t fusedOuter = tailAxisOuter * inputParams.dyFusedFront;
    while (fusedOuter > inputParams.totalCoreNum && tailAxisInner < inputParams.maxUbAvailable) {
        tailAxisInner += Ops::Base::FloorDiv(ONE_BLK_BYTES, inputParams.bytesForOneData);
        tailAxisOuter = Ops::Base::CeilDiv(inputParams.dyTailAxisLen, tailAxisInner);
        fusedOuter = tailAxisOuter * inputParams.dyFusedFront;
    }
    return std::make_pair(tailAxisInner, tailAxisOuter);
}

std::pair<int64_t, int64_t> SplitOutputTailAxis(StridedSliceGradParamList& inputParams)
{
    int64_t tailAxisInner = Ops::Base::FloorDiv(LARGE_BLOCK_FOR_MOVE, inputParams.bytesForOneData);
    int64_t tailAxisOuter = Ops::Base::CeilDiv(inputParams.inTailAxisLen, tailAxisInner);
    int64_t fusedOuter = tailAxisOuter * inputParams.inFusedFront;
    while (fusedOuter > inputParams.totalCoreNum && tailAxisInner < inputParams.maxUbAvailable) {
        tailAxisInner += Ops::Base::FloorDiv(ONE_BLK_BYTES, inputParams.bytesForOneData);
        tailAxisOuter = Ops::Base::CeilDiv(inputParams.inTailAxisLen, tailAxisInner);
        fusedOuter = tailAxisOuter * inputParams.inFusedFront;
    }
    return std::make_pair(tailAxisInner, tailAxisOuter);
}

static inline void CaluSplitCoreParam(StridedSliceGradParamList& inputParams)
{
    // 初始化公共参数 & 清零分核 & 计算分核
    inputParams.tailAxisOuter = 0;
    inputParams.tailAxisInner = 0;
    inputParams.tailAxisTail = 0;

    std::tuple<int64_t, int64_t, int64_t> resultInit =
    AverageTilingLargeBlock(inputParams.inShapeOutSize, inputParams.totalCoreNum, inputParams.bytesForOneData);
    inputParams.usedCoreNumForClear = std::get<TUPLE_INDEX_0>(resultInit);
    inputParams.normalCoreProcessNumForClear = std::get<TUPLE_INDEX_1>(resultInit);
    inputParams.tailCoreProcessNumForClear = std::get<TUPLE_INDEX_2>(resultInit);

    if (inputParams.caluMode == MODE_SIMT) { // simt
        std::tuple<int64_t, int64_t, int64_t> result =
            AverageTiling(inputParams.dyShapeOutSize, inputParams.totalCoreNum);
        inputParams.usedCoreNum = std::get<TUPLE_INDEX_0>(result);
        inputParams.normalCoreProcessNum = std::get<TUPLE_INDEX_1>(result);
        inputParams.tailCoreProcessNum = std::get<TUPLE_INDEX_2>(result);
    } else if (inputParams.caluMode == MODE_SPLIT_FRONT_MOVE_ALIGN) { // move_align完整尾轴
        int64_t dyShapeSizeNoTailAxis = 1;
        for (size_t i = 0; i < inputParams.inShape.GetDimNum() - 1; i++) {
            dyShapeSizeNoTailAxis *= inputParams.dyShape[i];
        }
        std::tuple<int64_t, int64_t, int64_t> result = AverageTiling(dyShapeSizeNoTailAxis, inputParams.totalCoreNum);
        inputParams.usedCoreNum = std::get<TUPLE_INDEX_0>(result);
        inputParams.normalCoreProcessNum = std::get<TUPLE_INDEX_1>(result);
        inputParams.tailCoreProcessNum = std::get<TUPLE_INDEX_2>(result);
    } else if (inputParams.caluMode == MODE_SPLIT_ALL_MOVE_ALIGN) { // move_align切尾轴
        std::pair<int64_t, int64_t> result = SplitInputTailAxis(inputParams);
        inputParams.tailAxisInner = result.first;
        inputParams.tailAxisOuter = result.second;
        inputParams.tailAxisTail =
            inputParams.dyTailAxisLen - inputParams.tailAxisInner * (inputParams.tailAxisOuter - 1);
        std::tuple<int64_t, int64_t, int64_t> resultSplitCore =
            AverageTiling(inputParams.dyFusedFront * inputParams.tailAxisOuter, inputParams.totalCoreNum);
        inputParams.usedCoreNum = std::get<TUPLE_INDEX_0>(resultSplitCore);
        inputParams.normalCoreProcessNum = std::get<TUPLE_INDEX_1>(resultSplitCore);
        inputParams.tailCoreProcessNum = std::get<TUPLE_INDEX_2>(resultSplitCore);
    } else if (inputParams.caluMode == MODE_SPLIT_FRONT_NDDMA) {
        SearchBestAxisNumInUb(inputParams);
        int64_t fusedRemainAxis = 1;
        size_t dimNum = inputParams.dyShape.GetDimNum();
        for (size_t i = 0; i < (dimNum - inputParams.splitUbAxisNum); i++) {
            fusedRemainAxis *= inputParams.inShape[i];
        }
        std::tuple<int64_t, int64_t, int64_t> result = AverageTiling(fusedRemainAxis, inputParams.totalCoreNum);
        inputParams.usedCoreNum = std::get<TUPLE_INDEX_0>(result);
        inputParams.normalCoreProcessNum = std::get<TUPLE_INDEX_1>(result);
        inputParams.tailCoreProcessNum = std::get<TUPLE_INDEX_2>(result);
    } else if (inputParams.caluMode == MODE_SPLIT_ALL_NDDMA) {
        std::pair<int64_t, int64_t> result = SplitOutputTailAxis(inputParams);
        inputParams.tailAxisInner = result.first;
        inputParams.tailAxisOuter = result.second;
        inputParams.tailAxisTail =
            inputParams.inTailAxisLen - inputParams.tailAxisInner * (inputParams.tailAxisOuter - 1);
        std::tuple<int64_t, int64_t, int64_t> resultSplitCore =
            AverageTiling(inputParams.inFusedFront * inputParams.tailAxisOuter, inputParams.totalCoreNum);
        inputParams.usedCoreNum = std::get<TUPLE_INDEX_0>(resultSplitCore);
        inputParams.normalCoreProcessNum = std::get<TUPLE_INDEX_1>(resultSplitCore);
        inputParams.tailCoreProcessNum = std::get<TUPLE_INDEX_2>(resultSplitCore);
    }
}

static inline void UpdateParaAlign8Dim(
    const gert::TilingContext* context, StridedSliceGradParamList& inputParams, ops::StridedSliceParams& initParam)
{
    auto rank = inputParams.dyShape.GetDimNum();
    initParam.input_shape = inputParams.inShape;
    initParam.begin = inputParams.begin;
    initParam.end = inputParams.end;
    initParam.strides = inputParams.strides;
    initParam.dy_shape = inputParams.dyShape;

    inputParams.inShape.SetDimNum(MAX_DIM_NUM);
    inputParams.begin.SetDimNum(MAX_DIM_NUM);
    inputParams.end.SetDimNum(MAX_DIM_NUM);
    inputParams.strides.SetDimNum(MAX_DIM_NUM);
    inputParams.dyShape.SetDimNum(MAX_DIM_NUM);

    for (size_t i = 0; i < MAX_DIM_NUM; i++) {
        if (i < (MAX_DIM_NUM - rank)) {
            inputParams.inShape.SetDim(i, 1);
            inputParams.begin.SetDim(i, 0);
            inputParams.end.SetDim(i, 1);
            inputParams.strides.SetDim(i, 1);
            inputParams.dyShape.SetDim(i, 1);
        } else {
            inputParams.inShape.SetDim(i, initParam.input_shape[i + rank - MAX_DIM_NUM]);
            inputParams.begin.SetDim(i, initParam.begin[i + rank - MAX_DIM_NUM]);
            inputParams.end.SetDim(i, initParam.end[i + rank - MAX_DIM_NUM]);
            inputParams.strides.SetDim(i, initParam.strides[i + rank - MAX_DIM_NUM]);
            inputParams.dyShape.SetDim(i, initParam.dy_shape[i + rank - MAX_DIM_NUM]);
        }
    }

    for (size_t index = 0; index < MAX_DIM_NUM; index++) {
        int64_t fuseInRes = 1;
        int64_t fuseDyRes = 1;
        for (size_t j = index + 1; j < MAX_DIM_NUM; ++j) {
            fuseInRes *= inputParams.inShape[j];
            fuseDyRes *= inputParams.dyShape[j];
        }
        inputParams.fusedInShape.push_back(fuseInRes);
        inputParams.fusedDyShape.push_back(fuseDyRes);
    }

    OP_LOGD(
        context->GetNodeName(), "after inShape extend to dim8 = %s", Ops::Base::ToString(inputParams.inShape).c_str());
    OP_LOGD(context->GetNodeName(), "after begin extend to dim8 = %s", Ops::Base::ToString(inputParams.begin).c_str());
    OP_LOGD(context->GetNodeName(), "after end extend to dim8 = %s", Ops::Base::ToString(inputParams.end).c_str());
    OP_LOGD(
        context->GetNodeName(), "after strides extend to dim8 = %s", Ops::Base::ToString(inputParams.strides).c_str());
    OP_LOGD(
        context->GetNodeName(), "after dyShape extend to dim8 = %s", Ops::Base::ToString(inputParams.dyShape).c_str());
}

ge::graphStatus TilingPrepareForStridedSlice(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init TransposeNddmaTiling.");
    auto ci = context->GetCompiledInfo<StridedSliceGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((ci->coreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((ci->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForStridedSliceGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(
        context == nullptr, OP_LOGE("[StridedSliceGradTilingForAscendC]", "Context should not be nullptr."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "StridedSliceGradTilingForAscendC running begin.");

    auto compileInfo = reinterpret_cast<const StridedSliceGradCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    // 获取 totalCore && hardUbSize
    StridedSliceGradParamList inputParams;
    inputParams.totalCoreNum = static_cast<int64_t>(compileInfo->coreNum);
    inputParams.hardwareUbSize = static_cast<int64_t>(compileInfo->ubSize);

    OP_CHECK_IF(
        inputParams.totalCoreNum == 0, OP_LOGE(context->GetNodeName(), "total_core_num is 0, please check."),
        return ge::GRAPH_FAILED);

    // 校验和设置的输入值、属性值和类型校验
    ops::StridedSliceParams initParam;
    OP_CHECK_IF(
        CheckAndSetOfInputParams(context, inputParams, initParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "tilingStridedSliceGrad fail to check and set input params."),
        return ge::GRAPH_FAILED);

    // 计算基础参数param
    CaluBaseParams(inputParams);

    // 计算分支mode
    CaluModeParam(inputParams);
    OP_LOGD("CaluModeParam", "current calu mode: %ld", inputParams.caluMode);

    if (inputParams.caluMode == MODE_SIMT) {
        context->SetLocalMemorySize(inputParams.hardwareUbSize - SIMT_DCACHE_SIZE);
    }

    // 计算分核参数
    CaluSplitCoreParam(inputParams);

    // 输出补充到8维
    UpdateParaAlign8Dim(context, inputParams, initParam);

    // 计算tilingKey参数
    GetTilingKey(inputParams);

    // 更新tilingData
    StridedSliceGradTilingData tilingData;
    SetTilingParam(tilingData, inputParams);
    OP_CHECK_IF(
        StridedSliceGradSaveTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context->GetNodeName(), "[StridedSliceGradSaveTilingData] TilingStridedSliceGrad fail to set tiling data."),
        return ge::GRAPH_FAILED);

    // 设置userWorkspace
    size_t* userWorkspaceSize = context->GetWorkspaceSizes(1);
    userWorkspaceSize[0] = WORKSPACE_SIZE;
    tilingData.set_workspaceSize(userWorkspaceSize[0]);
    int64_t maxUsedCoreNum = ComputeMaxUsedCoreNum(inputParams);
    if (!(inputParams.caluMode == MODE_SPLIT_FRONT_NDDMA || inputParams.caluMode == MODE_SPLIT_ALL_NDDMA)) {
        context->SetScheduleMode(BATCH_MODE);
    }
    
    context->SetBlockDim(maxUsedCoreNum);
    context->SetTilingKey(tilingData.get_tilingKey());
    TilingDataToLogging(tilingData);

    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the StridedSliceGrad op.
IMPL_OP_OPTILING(StridedSliceGrad)
    .TilingInputsDataDependency({IN_SHAPE_IDX, IN_BEGIN_IDX, IN_END_IDX, IN_STRIDES_IDX})
    .Tiling(TilingForStridedSliceGrad)
    .TilingParse<StridedSliceGradCompileInfo>(TilingPrepareForStridedSlice);

} // namespace optiling