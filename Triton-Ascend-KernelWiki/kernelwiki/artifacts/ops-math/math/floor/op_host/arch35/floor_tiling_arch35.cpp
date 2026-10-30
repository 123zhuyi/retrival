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
 * \file floor_tiling_arch35.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "floor_tiling_arch35.h"

using namespace ge;
using namespace Ops::Base;

namespace optiling {

static constexpr uint64_t OP_KEY_INVALID = 0;
static constexpr uint64_t OP_KEY_1 = 1;
static constexpr uint64_t OP_KEY_2 = 2;
static constexpr uint64_t OP_KEY_3 = 3;
static constexpr uint64_t INDEX_0 = 0;
static constexpr uint64_t WORKSPACE_SIZE = 32;

ge::graphStatus FloorTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FloorCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "compile_info", "nullptr", "not nullptr"), return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize = ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t FloorTiling::GetOpKey(ge::DataType xDtype, ge::DataType yDtype)
{
    bool opKey1Flag = xDtype == DT_BF16 && yDtype == DT_BF16;
    if (opKey1Flag) {
        return OP_KEY_1;
    }
    bool opKey2Flag = xDtype == DT_FLOAT16 && yDtype == DT_FLOAT16;
    if (opKey2Flag) {
        return OP_KEY_2;
    }
    bool opKey3Flag = xDtype == DT_FLOAT && yDtype == DT_FLOAT;
    if (opKey3Flag) {
        return OP_KEY_3;
    }

    return OP_KEY_INVALID;
}

uint64_t FloorTiling::GenerateTilingKey(uint64_t innerKey)
{
    return opKey * OP_KEY_OFFSET + innerKey;
}

std::map<uint64_t, ComputeParams> FloorTiling::GetComputeMap(uint64_t paramOpKey)
{
    ComputeParams computeParams0;
    switch (paramOpKey) {
        case OP_KEY_1:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS16_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0};
            computeParams0.bufferDivisor = {64};
            return {{0, computeParams0}};
        case OP_KEY_2:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS16_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0};
            computeParams0.bufferDivisor = {64};
            return {{0, computeParams0}};
        case OP_KEY_3:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS32_SIZE);
            computeParams0.extraSize = {0};
            computeParams0.bufferDivisor = {128};
            return {{0, computeParams0}};
        default:
            return {};
    }
}

ge::graphStatus FloorTiling::GetShapeAttrsInfo()
{
    auto x = context_->GetInputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    auto xDtype = x->GetDataType();
    auto y = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y);
    auto yDtype = y->GetDataType();

    opKey = GetOpKey(xDtype, yDtype);
    OP_CHECK_IF(
        (opKey == OP_KEY_INVALID), OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "opKey", std::to_string(opKey), "not 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool FloorTiling::IsCapable()
{
    return true;
}

ge::graphStatus FloorTiling::DoOpTiling()
{
    auto xShape = context_->GetInputShape(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);

    ElewiseTilingParams elewiseTilingParams;
    elewiseTilingParams.shape = xShape->GetStorageShape();
    elewiseTilingParams.computeMap = GetComputeMap(opKey);
    elewiseTilingParams.coreNum = coreNum;
    elewiseTilingParams.ubSize = ubSize;

    ElewiseTilingData elewiseTilingData;
    auto status = ElewiseTiling(elewiseTilingParams, elewiseTilingData);
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED), OP_LOGE(context_->GetNodeName(), "elewise tiling failed"),
        return ge::GRAPH_FAILED);

    tilingKey_ = GenerateTilingKey(elewiseTilingData.innerKey);
    blockNum = elewiseTilingData.blockNum;
    tilingData.set_dim0(elewiseTilingData.dim0);
    tilingData.set_blockFormer(elewiseTilingData.blockFormer);
    tilingData.set_ubFormer(elewiseTilingData.ubFormer);
    tilingData.set_ubLoopOfFormerBlock(elewiseTilingData.ubLoopOfFormerBlock);
    tilingData.set_ubLoopOfTailBlock(elewiseTilingData.ubLoopOfTailBlock);
    tilingData.set_ubTailOfFormerBlock(elewiseTilingData.ubTailOfFormerBlock);
    tilingData.set_ubTailOfTailBlock(elewiseTilingData.ubTailOfTailBlock);
    tilingData.set_elemNum(elewiseTilingData.elemNum);

    return ge::GRAPH_SUCCESS;
}

std::string FloorTiling::ToString(FloorTilingData& paramTilingData)
{
    std::string str;
    str += " dim0:" + std::to_string(paramTilingData.get_dim0());
    str += " blockFormer:" + std::to_string(paramTilingData.get_blockFormer());
    str += " ubFormer:" + std::to_string(paramTilingData.get_ubFormer());
    str += " ubLoopOfFormerBlock:" + std::to_string(paramTilingData.get_ubLoopOfFormerBlock());
    str += " ubLoopOfTailBlock:" + std::to_string(paramTilingData.get_ubLoopOfTailBlock());
    str += " ubTailOfFormerBlock:" + std::to_string(paramTilingData.get_ubTailOfFormerBlock());
    str += " ubTailOfTailBlock:" + std::to_string(paramTilingData.get_ubTailOfTailBlock());
    str += " elemNum:" + std::to_string(paramTilingData.get_elemNum());
    return str;
}

ge::graphStatus FloorTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t FloorTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus FloorTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FloorTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "workspace", "nullptr", "not nullptr"), return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGI(context_, "TilingInfo: %s.", ToString(tilingData).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForFloor(gert::TilingContext* context)
{
    auto compileInfo = context->GetCompileInfo<FloorCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    FloorTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForFloor(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<FloorCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Floor).Tiling(TilingForFloor).TilingParse<FloorCompileInfo>(TilingPrepareForFloor);

} // namespace optiling
