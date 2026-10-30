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
 * \file lerp_tiling_arch35.cpp
 * \brief
 */

#include "lerp_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_info.h"


using namespace AscendC;
using namespace ge;

namespace optiling {

static constexpr uint64_t OP_KEY_INVALID = 0;
static constexpr uint64_t OP_KEY_1 = 1;
static constexpr uint64_t OP_KEY_2 = 2;
static constexpr uint64_t OP_KEY_3 = 3;
static constexpr uint64_t OP_KEY_4 = 4;
static constexpr uint64_t OP_KEY_5 = 5;
static constexpr uint64_t INDEX_0 = 0;
static constexpr uint64_t INDEX_1 = 1;
static constexpr uint64_t INDEX_2 = 2;
static constexpr uint64_t INDEX_3 = 3;
static constexpr uint64_t WORKSPACE_SIZE = 32;

ge::graphStatus LerpTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = static_cast<const LerpCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
                        return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize = ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t LerpTiling::GetOpKey(ge::DataType startDtype, ge::DataType endDtype, ge::DataType weightDtype,
                              ge::DataType yDtype)
{
    bool opKey1Flag =
        startDtype == DT_FLOAT16 && endDtype == DT_FLOAT16 && weightDtype == DT_FLOAT16 && yDtype == DT_FLOAT16;
    if (opKey1Flag) {
        return OP_KEY_1;
    }
    bool opKey2Flag = startDtype == DT_FLOAT && endDtype == DT_FLOAT && weightDtype == DT_FLOAT && yDtype == DT_FLOAT;
    if (opKey2Flag) {
        return OP_KEY_2;
    }
    bool opKey3Flag = startDtype == DT_BF16 && endDtype == DT_BF16 && weightDtype == DT_BF16 && yDtype == DT_BF16;
    if (opKey3Flag) {
        return OP_KEY_3;
    }
    bool opKey4Flag = startDtype == DT_BF16 && endDtype == DT_BF16 && weightDtype == DT_FLOAT && yDtype == DT_BF16;
    if (opKey4Flag) {
        return OP_KEY_4;
    }
    bool opKey5Flag =
        startDtype == DT_FLOAT16 && endDtype == DT_FLOAT16 && weightDtype == DT_FLOAT && yDtype == DT_FLOAT16;
    if (opKey5Flag) {
        return OP_KEY_5;
    }

    return OP_KEY_INVALID;
}

uint64_t LerpTiling::GenerateTilingKey(uint64_t innerKey)
{
    return opKey * Ops::Base::BROADCAST_OP_KEY_OFFSET + innerKey;
}

std::map<uint64_t, Ops::Base::BroadcastComputeParams> LerpTiling::GetComputeMap(uint64_t inputOpKey)
{
    Ops::Base::BroadcastComputeParams computeParams0;
    switch (inputOpKey) {
        case OP_KEY_1:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS1_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {128, 128};
            return {{1, computeParams0}};
        case OP_KEY_2:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS1_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {256, 256};
            return {{1, computeParams0}};
        case OP_KEY_3:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS1_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {128, 128};
            return {{1, computeParams0}};
        case OP_KEY_4:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS1_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {160, 160};
            return {{1, computeParams0}};
        case OP_KEY_5:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS1_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {160, 160};
            return {{1, computeParams0}};
        default:
            return {};
    }
}

ge::graphStatus LerpTiling::GetShapeAttrsInfo()
{
    auto start = context_->GetInputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, start);
    auto startDtype = start->GetDataType();
    auto end = context_->GetInputDesc(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, end);
    auto endDtype = end->GetDataType();
    auto weight = context_->GetInputDesc(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weight);
    auto weightDtype = weight->GetDataType();
    auto y = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y);
    auto yDtype = y->GetDataType();

    opKey = GetOpKey(startDtype, endDtype, weightDtype, yDtype);
    OP_CHECK_IF((opKey == OP_KEY_INVALID),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "start, end, weight and y",
                        (Ops::Base::ToString(startDtype) + ", " + Ops::Base::ToString(endDtype) + ", " +
                         Ops::Base::ToString(weightDtype) + " and " + Ops::Base::ToString(yDtype)).c_str(),
                        "The dtypes of start, end, weight and y must be one of: all float16, all float, all bf16, "
                        "(bf16, bf16, float, bf16), or (float16, float16, float, float16)"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool LerpTiling::IsCapable()
{
    return true;
}

ge::graphStatus LerpTiling::DoOpTiling()
{
    Ops::Base::BroadcastTilingParams broadcastTilingParams;
    for (uint64_t i = 0; i < context_->GetComputeNodeInputNum(); i++) {
        auto shape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
        broadcastTilingParams.inShape.push_back(Ops::Base::EnsureNotScalar(shape->GetStorageShape()));
    }
    auto outputShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    broadcastTilingParams.outShape = Ops::Base::EnsureNotScalar(outputShape->GetStorageShape());
    broadcastTilingParams.computeMap = GetComputeMap(opKey);
    broadcastTilingParams.coreNum = coreNum;
    broadcastTilingParams.ubSize = ubSize;

    Ops::Base::BroadcastTilingData broadcastTilingData;
    ge::graphStatus status = BroadcastTiling(broadcastTilingParams, broadcastTilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "broadcast tiling failed.");
        return ge::GRAPH_FAILED;
    }

    tilingKey_ = GenerateTilingKey(broadcastTilingData.innerKey);
    blockNum = broadcastTilingData.blockNum;
    tilingData.set_blockFormer(broadcastTilingData.blockFormer);
    tilingData.set_ubFormer(broadcastTilingData.ubFormer);
    tilingData.set_ubOuter(broadcastTilingData.ubOuter);
    tilingData.set_ubTail(broadcastTilingData.ubTail);
    tilingData.set_blockTail(broadcastTilingData.blockTail);
    tilingData.set_shapeLen(broadcastTilingData.shapeLen);
    tilingData.set_ubSplitAxis(broadcastTilingData.ubSplitAxis);
    tilingData.set_dimProductBeforeUbInner(broadcastTilingData.dimProductBeforeUbInner);
    tilingData.set_elemNum(broadcastTilingData.elemNum);

    std::copy(broadcastTilingData.dims[INDEX_0].begin(), broadcastTilingData.dims[INDEX_0].end(), input0Dims);
    tilingData.set_input0Dims(input0Dims);
    std::copy(broadcastTilingData.dims[INDEX_1].begin(), broadcastTilingData.dims[INDEX_1].end(), input1Dims);
    tilingData.set_input1Dims(input1Dims);
    std::copy(broadcastTilingData.dims[INDEX_2].begin(), broadcastTilingData.dims[INDEX_2].end(), input2Dims);
    tilingData.set_input2Dims(input2Dims);
    std::copy(broadcastTilingData.dims[INDEX_3].begin(), broadcastTilingData.dims[INDEX_3].end(), outputDims);
    tilingData.set_outputDims(outputDims);
    std::copy(broadcastTilingData.strides[INDEX_0].begin(), broadcastTilingData.strides[INDEX_0].end(), input0Strides);
    tilingData.set_input0Strides(input0Strides);
    std::copy(broadcastTilingData.strides[INDEX_1].begin(), broadcastTilingData.strides[INDEX_1].end(), input1Strides);
    tilingData.set_input1Strides(input1Strides);
    std::copy(broadcastTilingData.strides[INDEX_2].begin(), broadcastTilingData.strides[INDEX_2].end(), input2Strides);
    tilingData.set_input2Strides(input2Strides);
    std::copy(broadcastTilingData.strides[INDEX_3].begin(), broadcastTilingData.strides[INDEX_3].end(), outputStrides);
    tilingData.set_outputStrides(outputStrides);

    return ge::GRAPH_SUCCESS;
}

std::string LerpTiling::ToString(LerpTilingData& inputTilingData)
{
    std::string str;
    str += " blockFormer:" + std::to_string(inputTilingData.get_blockFormer());
    str += " ubFormer:" + std::to_string(inputTilingData.get_ubFormer());
    str += " ubOuter:" + std::to_string(inputTilingData.get_ubOuter());
    str += " ubTail:" + std::to_string(inputTilingData.get_ubTail());
    str += " blockTail:" + std::to_string(inputTilingData.get_blockTail());
    str += " shapeLen:" + std::to_string(inputTilingData.get_shapeLen());
    str += " ubSplitAxis:" + std::to_string(inputTilingData.get_ubSplitAxis());
    str += " dimProductBeforeUbInner:" + std::to_string(inputTilingData.get_dimProductBeforeUbInner());
    str += " elemNum:" + std::to_string(inputTilingData.get_elemNum());
    return str;
}

ge::graphStatus LerpTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LerpTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus LerpTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LerpTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGI(context_, "TilingInfo: %s.", ToString(tilingData).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForLerp(gert::TilingContext* context)
{
    LerpTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForLerp(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<LerpCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Lerp).Tiling(TilingForLerp).TilingParse<LerpCompileInfo>(TilingPrepareForLerp);
}  // namespace optiling
