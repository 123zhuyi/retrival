/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file strided_slice_assign_v2_tiling.cpp
 * \brief
 */

#include <vector>
#include "util/math_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "strided_slice_assign_v2_tiling.h"


namespace optiling {
static const size_t INDEX_BEGIN = 2;
static const size_t INDEX_END = 3;
static const size_t INDEX_STRIDES = 4;
static const size_t INDEX_AXES = 5;

constexpr int32_t INPUT_VAR_INDEX = 0;
constexpr int32_t INPUT_INPUTVAL_INDEX = 1;

template <typename T>
static ge::graphStatus CopyData2Array(gert::TilingContext *context, const gert::Tensor *listTensor, int64_t listSize,
    int64_t dataList[])
{
    const T *listDataPtr = listTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, listDataPtr);
    for (int64_t i = 0; i < listSize; i++) {
        dataList[i] = listDataPtr[i];
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus CopyData2Array(gert::TilingContext *context, const gert::Tensor *listTensor, int64_t listSize,
    std::vector<int64_t> &dataList)
{
    const T *listDataPtr = listTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, listDataPtr);
    for (int64_t i = 0; i < listSize; i++) {
        dataList.emplace_back(listDataPtr[i]);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus GetConstInputData(gert::TilingContext *context, const size_t idxInput, T &dataList)
{
    auto listTensor = context->GetInputTensor(idxInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, listTensor);
    auto listDataType = listTensor->GetDataType();
    int64_t listSize = listTensor->GetShapeSize();
    if (listDataType == ge::DT_INT32) {
        return CopyData2Array<int32_t>(context, listTensor, listSize, dataList);
    }
    if (listDataType == ge::DT_INT64) {
        return CopyData2Array<int64_t>(context, listTensor, listSize, dataList);
    }

    OP_LOGE(context->GetNodeName(), "Input begin/end/strides/axes only support int32/int64.");
    return ge::GRAPH_FAILED;
}

static ge::graphStatus AdjustWithAxes(gert::TilingContext *context, size_t dimNum, int64_t varDim[], int64_t begin[],
    int64_t end[], int64_t strides[])
{
    std::vector<int64_t> axes;
    OP_CHECK_IF(GetConstInputData(context, INDEX_AXES, axes) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Get const input axes data failed."), return ge::GRAPH_FAILED);
    // adjust begin/end/strides
    int64_t *beginAdjust = new int64_t[dimNum];
    int64_t *endAdjust = new int64_t[dimNum];
    int64_t *stridesAdjust = new int64_t[dimNum];
    for (size_t i = 0; i < dimNum; i++) {
        beginAdjust[i] = 0;
        endAdjust[i] = varDim[i];
        stridesAdjust[i] = 1;
    }
    for (size_t i = 0; i < axes.size(); i++) {
        beginAdjust[axes[i]] = begin[i];
        endAdjust[axes[i]] = end[i];
        stridesAdjust[axes[i]] = strides[i];
    }
    for (size_t i = 0; i < dimNum; i++) {
        begin[i] = beginAdjust[i];
        end[i] = endAdjust[i];
        strides[i] = stridesAdjust[i];
    }
    delete[] beginAdjust;
    delete[] endAdjust;
    delete[] stridesAdjust;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AdjustBegin(size_t dimNum, int64_t varDim[], int64_t begin[])
{
    for (size_t i = 0; i < dimNum; i++) {
        if (begin[i] < 0) {
            begin[i] = varDim[i] + begin[i];
        }
        if (begin[i] < 0) {
            begin[i] = 0;
        }
        if (begin[i] > varDim[i]) {
            begin[i] = varDim[i];
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AdjustEnd(size_t dimNum, int64_t varDim[], int64_t end[])
{
    for (size_t i = 0; i < dimNum; i++) {
        if (end[i] < 0) {
            end[i] = varDim[i] + end[i];
        }
        if (end[i] < 0) {
            end[i] = 0;
        }
        if (end[i] > varDim[i]) {
            end[i] = varDim[i];
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape(const gert::TilingContext *context, size_t dimNum,
    const int64_t inputValueDim[], const int64_t begin[], const int64_t end[], const int64_t strides[])
{
    for (size_t i = 0; i < dimNum; i++) {
        int64_t calcValue = Ops::Base::CeilDiv(end[i] - begin[i], strides[i]);
        calcValue = calcValue < 0 ? 0 : calcValue;
        OP_CHECK_IF(calcValue != inputValueDim[i],
            OP_LOGE(context->GetNodeName(), "Input shape does not match, please check!"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4StridedSliceAssignV2(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), " Tiling4StridedSliceAssignV2 is running.");
    StridedSliceAssignV2TilingData tiling;
    auto compileInfo = reinterpret_cast<const StridedSliceAssignV2CompileInfo *>(context->GetCompileInfo());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t totalCoreNum = (compileInfo == nullptr) ? ascendcPlatform.GetCoreNumAiv() : compileInfo->totalCoreNum;

    const gert::StorageShape *varShape = context->GetInputShape(INPUT_VAR_INDEX);
    const gert::StorageShape *inputValueShape = context->GetInputShape(INPUT_INPUTVAL_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputValueShape);

    size_t dimNum = varShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(dimNum != inputValueShape->GetStorageShape().GetDimNum(),
        OP_LOGE(context->GetNodeName(), "Var's dim num must equal to input_value's"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimNum <= 0, OP_LOGE(context->GetNodeName(), "Var's dim num should not be smaller or equal to zero."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(dimNum > 8, OP_LOGE(context->GetNodeName(), "Var's dim num should not be greater than eight."),
        return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < dimNum; i++) {
        OP_CHECK_IF(varShape->GetStorageShape().GetDim(i) <= 0,
            OP_LOGE(context->GetNodeName(), "Input var shape can not less or equal 0."), return false);
    }

    int64_t varDim[MAX_DIM_NUM] = {0};
    int64_t inputValueDim[MAX_DIM_NUM] = {0};

    for (size_t i = 0; i < dimNum; i++) {
        varDim[i] = varShape->GetStorageShape().GetDim(i);
        inputValueDim[i] = inputValueShape->GetStorageShape().GetDim(i);
    }

    auto axesTensor = context->GetOptionalInputTensor(5);
    bool isHasAxes = (nullptr == axesTensor) ? 0 : 1;

    int64_t begin[MAX_DIM_NUM] = {0};
    int64_t end[MAX_DIM_NUM] = {0};
    int64_t strides[MAX_DIM_NUM] = {0};
    OP_CHECK_IF(GetConstInputData(context, INDEX_BEGIN, begin) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Get const input begin data failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetConstInputData(context, INDEX_END, end) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Get const input end data failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetConstInputData(context, INDEX_STRIDES, strides) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Get const input strides data failed."), return ge::GRAPH_FAILED);
    if (isHasAxes) {
        OP_CHECK_IF(AdjustWithAxes(context, dimNum, varDim, begin, end, strides) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "Adjust begin, end and strides failed."), return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(AdjustBegin(dimNum, varDim, begin) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Adjust begin data failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(AdjustEnd(dimNum, varDim, end) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Adjust end data failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckInputShape(context, dimNum, inputValueDim, begin, end, strides) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Get const input strides data failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(strides[dimNum - 1] != 1, OP_LOGE(context->GetNodeName(), "The stride of last dim must be 1."),
        return ge::GRAPH_FAILED);

    int64_t varCumShape[MAX_DIM_NUM] = {0};
    int64_t inputCumShape[MAX_DIM_NUM] = {0};

    varCumShape[dimNum - 1U] = varDim[dimNum - 1U];
    inputCumShape[dimNum - 1U] = inputValueDim[dimNum - 1U];

    for (int32_t i = static_cast<int32_t>(dimNum) - 2; i > 0; i--) {
        varCumShape[i] = varDim[i] * varCumShape[i + 1];
        inputCumShape[i] = inputValueDim[i] * inputCumShape[i + 1];
    }
    uint32_t useCoreNum = totalCoreNum;

    context->SetBlockDim(useCoreNum);

    uint32_t tilingKey = 1;
    context->SetTilingKey(tilingKey);

    tiling.set_dimNum(static_cast<int64_t>(dimNum));
    tiling.set_varDim(varDim);
    tiling.set_inputValueDim(inputValueDim);
    tiling.set_begin(begin);
    tiling.set_strides(strides);
    tiling.set_varCumShape(varCumShape);
    tiling.set_inputCumShape(inputCumShape);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t sysWorkspaceSize = 16777216U; // 16777216U = 16U * 1024U * 1024U;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4StridedSliceAssignV2(gert::TilingParseContext *context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4StridedSliceAssignV2 running.");
    auto compileInfo = context->GetCompiledInfo<StridedSliceAssignV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    // no vector core enabled
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0),
        OP_LOGE(context->GetNodeName(), "TilingPrepare4StridedSliceAssignV2 fail to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
        OP_LOGE(context->GetNodeName(), "TilingPrepare4StridedSliceAssignV2 fail to get ub size."),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "TilingPrepare4StridedSliceAssignV2 exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StridedSliceAssignV2)
    .Tiling(Tiling4StridedSliceAssignV2)
    .TilingParse<StridedSliceAssignV2CompileInfo>(TilingPrepare4StridedSliceAssignV2)
    .TilingInputsDataDependency({ 2, 3, 4, 5 });
} // namespace optiling