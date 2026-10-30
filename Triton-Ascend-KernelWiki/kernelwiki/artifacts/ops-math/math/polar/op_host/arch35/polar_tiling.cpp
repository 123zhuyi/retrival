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
 * \file    polar_tiling.cpp
 * \brief   polar tiling
 */

#include "polar_tiling.h"
#include <graph/utils/type_utils.h>

using namespace ge;

namespace optiling {
static constexpr uint64_t INPUT_ABS = 0;
static constexpr uint64_t INPUT_ANGLE = 1;
static constexpr uint64_t OUTPUT_Y = 0;

ge::graphStatus PolarTiling::GetPlatformInfo()
{
    OP_LOGD(context_, "PolarTiling GetPlatformInfo.");
    compileInfo_ = static_cast<const PolarCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PolarTiling::CheckDtype()
{
    OP_LOGD(context_, "PolarTiling CheckDtype.");
    auto input0Desc = context_->GetInputDesc(INPUT_ABS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0Dtype = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(INPUT_ANGLE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1Dtype = input1Desc->GetDataType();
    auto outputDesc = context_->GetOutputDesc(OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    if (input0Dtype != ge::DT_FLOAT || input1Dtype != ge::DT_FLOAT || outputDtype != ge::DT_COMPLEX64) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(input0Dtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(input1Dtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "abs, angle and y", dtypesStr.c_str(),
            "The dtypes of abs and angle must be float, and the dtype of y must be complex64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PolarTiling::CheckBroadcastAndMergeShape()
{
    OP_LOGD(context_, "PolarTiling CheckBroadcastAndMergeShape.");
    const gert::StorageShape* absStorageShape = context_->GetInputShape(INPUT_ABS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, absStorageShape);
    const gert::StorageShape* angleStorageShape = context_->GetInputShape(INPUT_ANGLE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, angleStorageShape);

    auto absShape = absStorageShape->GetStorageShape();
    auto angleShape = angleStorageShape->GetStorageShape();

    int64_t absDimNum = static_cast<int64_t>(absShape.GetDimNum());
    int64_t angleDimNum = static_cast<int64_t>(angleShape.GetDimNum());
    dimNum_ = std::max(absDimNum, angleDimNum);
    OP_CHECK_IF(dimNum_ > POLAR_MAX_DIM,
                OP_LOGE(context_, "dimNum %ld exceeds POLAR_MAX_DIM %ld", dimNum_, POLAR_MAX_DIM),
                return ge::GRAPH_FAILED);

    for (int64_t i = 0; i < dimNum_; i++) {
        int64_t absOffset = i - (dimNum_ - absDimNum);
        int64_t angleOffset = i - (dimNum_ - angleDimNum);
        int64_t absDim = (absOffset >= 0) ? absShape.GetDim(absOffset) : 1;
        int64_t angleDim = (angleOffset >= 0) ? angleShape.GetDim(angleOffset) : 1;
        absDims_[i] = absDim;
        angleDims_[i] = angleDim;
        OP_CHECK_IF(absDim != angleDim && absDim != 1 && angleDim != 1,
                    OP_LOGE(context_, "Shapes not broadcastable at dim %ld: %ld vs %ld", i, absDim, angleDim),
                    return ge::GRAPH_FAILED);
        mergedShape_[i] = std::max(absDim, angleDim);
    }

    totalElements_ = 1;
    for (int64_t i = 0; i < dimNum_; i++) {
        totalElements_ *= mergedShape_[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PolarTiling::CalcStride()
{
    OP_LOGD(context_, "PolarTiling CalcStride.");
    int64_t strideAbs = 1;
    int64_t strideAngle = 1;
    int64_t strideMerged = 1;
    int64_t strideY = 1;
    for (int64_t i = dimNum_ - 1; i >= 0; i--) {
        absStride_[i] = (absDims_[i] == 1) ? 0 : strideAbs;
        angleStride_[i] = (angleDims_[i] == 1) ? 0 : strideAngle;
        mergedStride_[i] = strideMerged;
        yStride_[i] = strideY;
        strideAbs *= absDims_[i];
        strideAngle *= angleDims_[i];
        strideMerged *= mergedShape_[i];
        strideY *= mergedShape_[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PolarTiling::GetShapeAttrsInfo()
{
    if (CheckDtype() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CheckBroadcastAndMergeShape() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (CalcStride() != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PolarTiling::DoOpTiling()
{
    OP_LOGD(context_, "PolarTiling DoOpTiling.");

    int64_t coreNum = compileInfo_->coreNum;
    int64_t elementsPerCore = totalElements_ / coreNum;
    int64_t formerCore = totalElements_ % coreNum;

    tilingData_.totalElements = totalElements_;
    tilingData_.elementsPerCore = elementsPerCore;
    tilingData_.coreNum = coreNum;
    tilingData_.formerCore = formerCore;
    tilingData_.dimNum = dimNum_;

    for (int64_t i = 0; i < POLAR_MAX_DIM; i++) {
        if (i < dimNum_) {
            tilingData_.mergedStride[i] = mergedStride_[i];
            tilingData_.absStride[i] = absStride_[i];
            tilingData_.angleStride[i] = angleStride_[i];
            tilingData_.yStride[i] = yStride_[i];
        } else {
            tilingData_.mergedStride[i] = 1;
            tilingData_.absStride[i] = 0;
            tilingData_.angleStride[i] = 0;
            tilingData_.yStride[i] = 0;
        }
    }

    blockDim_ = (totalElements_ < coreNum) ? totalElements_ : coreNum;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PolarTiling::PostTiling()
{
    OP_LOGD(context_, "PolarTiling PostTiling.");

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0;

    auto res = context_->SetBlockDim(static_cast<uint32_t>(blockDim_));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context_, "SetBlockDim failed."), return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           &tilingData_, sizeof(PolarTilingData));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(PolarTilingData));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Polar(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4Polar start.");

    PolarTiling polarTiling(context);
    auto ret = polarTiling.DoTiling();
    OP_CHECK_IF((ret == ge::GRAPH_FAILED), OP_LOGD(context, "Tiling4Polar failed!"), return ge::GRAPH_FAILED);
    OP_LOGD(context, "Tiling4Polar end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4PolarAscendc(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4PolarAscendc.");

    auto compileInfo = context->GetCompiledInfo<PolarCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "core num is negative."),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Exit TilingPrepare4PolarAscendc.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Polar(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<PolarCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD("TilingPrepare4Polar", "Ascend C TilingPrepare4Polar success.");
    return TilingPrepare4PolarAscendc(context);
}

IMPL_OP_OPTILING(Polar).Tiling(Tiling4Polar).TilingParse<PolarCompileInfo>(TilingPrepare4Polar);

} // namespace optiling
