/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file one_hot_tiling_arch35.cpp
 * \brief
 */

#include <cmath>
#include "graph/utils/type_utils.h"
#include "one_hot_tiling.h"
#include "one_hot_tiling_arch35.h"

using namespace ge;

namespace optiling {

constexpr uint64_t TILING_KEY_WITHOUT_UB = 1000L;
constexpr uint64_t TILING_KEY_WITH_UB = 1001L;
constexpr uint64_t SYSTEM_WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);
constexpr int64_t SIMT_DCACHE_SIZE = static_cast<int64_t>(58 * 1024);
constexpr uint32_t MAX_DIM_CNT = 8;
constexpr uint32_t X_MAX_DIM_CNT = 7;

ge::graphStatus OneHotTilingBase::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const OneHotCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aivNum_ = static_cast<int32_t>(compileInfoPtr->core_num);
        ubSize_ = compileInfoPtr->ub_size;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        aivNum_ = static_cast<int32_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<int64_t>(ubSizePlatform);
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus OneHotTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    OP_CHECK_IF(
        !AnalyzeDtypeAndFormat() || !AnalyzeInputs() || !AnalyzeAttrs(),
        OP_LOGE(opName_, "fail to analyze context_ info"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OneHotTilingBase::CalUbTilingData(void)
{
    uint64_t totalNeedSize = inputTotalDimNum_ * static_cast<uint64_t>(depth_);
    uint64_t totalMinUbLoop = Ops::Base::CeilDiv(totalNeedSize, static_cast<uint64_t>(MIN_UB_CAL_SIZE));
    if (totalMinUbLoop < static_cast<uint64_t>(aivNum_)) {
        tilingData_.set_offValueUsedCoreNum(totalMinUbLoop);
        tilingData_.set_offValuePerCoreCalSize(MIN_UB_CAL_SIZE);
        tilingData_.set_offValueHasTail(totalNeedSize % MIN_UB_CAL_SIZE != 0);
    } else {
        tilingData_.set_offValueUsedCoreNum(aivNum_);
        int64_t singleCoreCalSize = Ops::Base::FloorDiv(totalNeedSize, static_cast<uint64_t>(aivNum_));
        tilingData_.set_offValuePerCoreCalSize(singleCoreCalSize);
        tilingData_.set_offValueHasTail(totalNeedSize % aivNum_ != 0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OneHotTilingBase::GetTilingParam(void)
{
    uint32_t blockFactor = MIN_FACTOR;
    uint32_t aivCoreNum = 1;
    uint16_t hasTail = 0;
    if (inputTotalDimNum_ <= MIN_FACTOR) {
        // use one core
        blockFactor = inputTotalDimNum_;
        aivCoreNum = 1U;
        hasTail = 0U;
    } else if (inputTotalDimNum_ <= static_cast<uint64_t>(aivNum_ * MIN_FACTOR)) {
        // use part of cores
        blockFactor = MIN_FACTOR;
        aivCoreNum = Ops::Base::CeilDiv(inputTotalDimNum_, static_cast<uint64_t>(MIN_FACTOR));
        if (inputTotalDimNum_ % MIN_FACTOR == 0) {
            hasTail = 0U;
        } else {
            hasTail = 1U;
        }
    } else {
        // use all cores
        aivCoreNum = static_cast<uint32_t>(aivNum_);
        blockFactor = Ops::Base::FloorDiv(inputTotalDimNum_, static_cast<uint64_t>(aivNum_));
        hasTail = (inputTotalDimNum_ % aivNum_ != 0) ? 1U : 0U;
    }
    tilingData_.set_usedCoreNum(aivCoreNum);
    tilingData_.set_perCoreCalSize(blockFactor);
    tilingData_.set_hasTail(hasTail);
    tilingData_.set_coreNum(aivNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetShapes()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OneHotTilingBase::MergeDims()
{
    int32_t dimNum = inputOriginShape.GetDimNum();
    int32_t tmpAbsAxis = axis_ == -1 ? dimNum : axis_;
    if (tmpAbsAxis < 0 || tmpAbsAxis > dimNum) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "axis",
            std::to_string(axis_).c_str(), "within the range [-1, the shape dim of x]");
        return ge::GRAPH_FAILED;
    }
    // Merge to 2D Shape(prefixdim, suffixdim)
    int64_t prefixDim = 1;
    int64_t suffixDim = 1;
    int32_t idx = 0;
    while (idx < tmpAbsAxis) {
        if (inputOriginShape[idx] != 0) {
            prefixDim *= inputOriginShape[idx];
        }
        idx++;
    }
    while (idx < dimNum) {
        if (inputOriginShape[idx] != 0) {
            suffixDim *= inputOriginShape[idx];
        }
        idx++;
    }
    tilingData_.set_prefixDim(prefixDim);
    tilingData_.set_suffixDim(suffixDim);
    inputTotalDimNum_ = static_cast<uint64_t>(prefixDim * suffixDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OneHotTilingBase::DoOpTiling()
{
    if (MergeDims() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "MergeDims failed.");
        return ge::GRAPH_FAILED;
    }
    if (depth_ > DEPTH_THRESHHOLD && inputTotalDimNum_ * static_cast<uint64_t>(depth_) > TOTALSIZE_THRESHHOLD) {
        if (CalUbTilingData() != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "CalUbTiling failed.");
            return ge::GRAPH_FAILED;
        }
        isUsedUbInit_ = true;
    } else {
        tilingData_.set_offValueUsedCoreNum(0);
        isUsedUbInit_ = false;
    }
    if (GetTilingParam() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Get TilingParam failed!.");
        return ge::GRAPH_FAILED;
    }
    tilingData_.set_ubSize(ubSize_ - SIMT_DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus OneHotTilingBase::PostTiling()
{
    context_->SetBlockDim(std::max(tilingData_.get_usedCoreNum(), tilingData_.get_offValueUsedCoreNum()));
    context_->SetTilingKey(GetTilingKey());
    if (GetTilingKey() == TILING_KEY_WITH_UB) {
        context_->SetLocalMemorySize(ubSize_ - SIMT_DCACHE_SIZE);
        context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    }
    OP_LOGD(context_->GetNodeName(), "Get TilingParam TilingKey:%ld.", GetTilingKey());
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus OneHotTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus OneHotTilingBase::GetWorkspaceSize()
{
    // set workspace
    workspaceSize_ = SYSTEM_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}
uint64_t OneHotTilingBase::GetTilingKey() const
{
    return isUsedUbInit_ ? TILING_KEY_WITH_UB : TILING_KEY_WITHOUT_UB;
}
bool OneHotTilingBase::AnalyzeDtypeAndFormat()
{
    auto indicesTypePtr = context_->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_IF(indicesTypePtr == nullptr, OP_LOGE(context_, "Input x's desc is nullptr."), return ge::GRAPH_FAILED);
    indicesType = indicesTypePtr->GetDataType();
    auto depthTypePtr = context_->GetInputDesc(INPUT_DEPTH_IDX);
    OP_CHECK_IF(depthTypePtr == nullptr, OP_LOGE(context_, "Input depth's desc is nullptr."), return ge::GRAPH_FAILED);
    depthType = depthTypePtr->GetDataType();
    auto onValueTypePtr = context_->GetInputDesc(INPUT_ON_VALUE_IDX);
    OP_CHECK_IF(
        onValueTypePtr == nullptr, OP_LOGE(context_, "Input on_value's desc is nullptr."), return ge::GRAPH_FAILED);
    onValueType = onValueTypePtr->GetDataType();
    auto offValueTypePtr = context_->GetInputDesc(INPUT_OFF_VALUE_IDX);
    OP_CHECK_IF(
        offValueTypePtr == nullptr, OP_LOGE(context_, "Input off_value's desc is nullptr."), return ge::GRAPH_FAILED);
    offValueType = offValueTypePtr->GetDataType();
    auto outputPtr = context_->GetOutputDesc(OUTPUT_IDX);
    OP_CHECK_IF(outputPtr == nullptr, OP_LOGE(context_, "Output's desc is nullptr."), return ge::GRAPH_FAILED);
    outputType = outputPtr->GetDataType();
    OP_CHECK_IF(
        indicesType != ge::DT_INT32 && indicesType != ge::DT_INT64 && indicesType != ge::DT_UINT8,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x",
            Ops::Base::ToString(indicesType).c_str(), "uint8, int32 or int64"),
        return false);
    OP_CHECK_IF(
        depthType != ge::DT_INT32 && depthType != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "depth",
            Ops::Base::ToString(depthType).c_str(), "int32 or int64"),
        return false);
    OP_CHECK_IF(
        ((onValueType != offValueType) ||
         (onValueType != ge::DT_FLOAT16 && onValueType != ge::DT_FLOAT && onValueType != ge::DT_INT32 &&
          onValueType != ge::DT_INT64 && onValueType != ge::DT_INT8 && onValueType != ge::DT_UINT8)),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "on_value and off_value",
            (Ops::Base::ToString(onValueType) + " and " + Ops::Base::ToString(offValueType)).c_str(),
            "The dtypes of on_value and off_value must be the same, and must be float16, float, int32, int64, int8 or uint8"),
        return false);
    OP_CHECK_IF(
        (outputType != onValueType),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "y and on_value",
            (Ops::Base::ToString(outputType) + " and " + Ops::Base::ToString(onValueType)).c_str(),
            "The dtypes of y and on_value must be the same"),
        return false);
    return true;
}
bool OneHotTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const auto axis = attrs->GetAttrPointer<int32_t>(ATTR_AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, axis);
    axis_ = *axis;

    if (axis_ > static_cast<int32_t>(inputOriginShape.GetDimNum()) || (axis_ < 0 && axis_ != -1)) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "axis",
            std::to_string(axis_).c_str(), "within the range [-1, input x's dim size]");
        return false;
    }
    if (axis_ < 0 && axis_ != -1) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "axis",
            std::to_string(axis_).c_str(), "greater than or equal to -1");
        return false;
    }
    return true;
}

bool OneHotTilingBase::AnalyzeInputs()
{
    auto indicesShapePtr = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_IF(indicesShapePtr == nullptr, OP_LOGE(context_, "Input x's shape is nullptr."), return ge::GRAPH_FAILED);
    auto depthTypePtr = context_->GetInputDesc(INPUT_DEPTH_IDX);
    auto depthDType = depthTypePtr->GetDataType();
    auto depthTensor = context_->GetInputTensor(INPUT_DEPTH_IDX);
    if (depthDType == ge::DT_INT32) {
        if (depthTensor->GetData<int32_t>() == nullptr) {
            OP_LOGE(context_->GetNodeName(), "depth data is nullptr");
            return false;
        }
        depth_ = static_cast<int64_t>(*(depthTensor->GetData<int32_t>()));
    } else if (depthDType == ge::DT_INT64) {
        if (depthTensor->GetData<int64_t>() == nullptr) {
            OP_LOGE(context_->GetNodeName(), "depth data is nullptr");
            return false;
        }
        depth_ = static_cast<int64_t>(*(depthTensor->GetData<int64_t>()));
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "depth",
            Ops::Base::ToString(depthDType).c_str(), "int32 or int64");
        return false;
    }
    auto depthShapePtr = context_->GetInputShape(INPUT_DEPTH_IDX);
    OP_CHECK_IF(
        depthShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "Input depth's shape is nullptr."),
        return ge::GRAPH_FAILED);
    auto onValueShapePtr = context_->GetInputShape(INPUT_ON_VALUE_IDX);
    OP_CHECK_IF(
        onValueShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "Input on_value's shape is nullptr."),
        return ge::GRAPH_FAILED);
    auto offValueShapePtr = context_->GetInputShape(INPUT_OFF_VALUE_IDX);
    OP_CHECK_IF(
        offValueShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "Input off_value's shape is nullptr."),
        return ge::GRAPH_FAILED);
    auto outShapePtr = context_->GetOutputShape(OUTPUT_IDX);
    OP_CHECK_IF(
        outShapePtr == nullptr, OP_LOGE(context_->GetNodeName(), "Output's shape is nullptr."),
        return ge::GRAPH_FAILED);
    inputOriginShape = Ops::Base::EnsureNotScalar(indicesShapePtr->GetOriginShape());
    OP_CHECK_IF(
        (inputOriginShape.GetDimNum() > X_MAX_DIM_CNT),
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "x", std::to_string(inputOriginShape.GetDimNum()).c_str(),
            "less than or equal to 7"),
        return false);
    auto outShape = Ops::Base::EnsureNotScalar(outShapePtr->GetOriginShape());

    auto depthShape = Ops::Base::EnsureNotScalar(depthShapePtr->GetOriginShape());

    auto onValueShape = Ops::Base::EnsureNotScalar(onValueShapePtr->GetOriginShape());

    auto offValueShape = Ops::Base::EnsureNotScalar(offValueShapePtr->GetOriginShape());

    return CheckSingleShape(outShape, "Output") && CheckSingleShape(depthShape, "Input depth") &&
           CheckSingleShape(onValueShape, "Input on_value") && CheckSingleShape(offValueShape, "Input off_value");
}

bool OneHotTilingBase::CheckSingleShape(const gert::Shape& shape, const string name) const
{
    OP_CHECK_IF((shape.GetDimNum() > MAX_DIM_CNT),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->GetNodeName(), name, std::to_string(shape.GetDimNum()),
                    "The shape dims of " + name + " must be less than or equal to " + std::to_string(MAX_DIM_CNT)),
                return false);
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(
            shape.GetDim(i) <= 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), name, Ops::Base::ToString(shape),
                                                  "All axes of " + name + " must be a positive number"), return false);
    }
    return true;
}

ge::graphStatus OneHotTilingForAscendC(gert::TilingContext* context)
{
    optiling::OneHotTilingBase tilingInstance(context);
    return tilingInstance.DoTiling();
}
} // namespace optiling