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
 * \file fused_mul_add_n_tiling_arch35.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "fused_mul_add_n_tiling_arch35.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"

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

ge::graphStatus FusedMulAddNTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FusedMulAddNCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
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

void FusedMulAddNTiling::SetOpKey()
{
    opKeys[{DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT}] = OP_KEY_1;
    opKeys[{DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16}] = OP_KEY_2;
    opKeys[{DT_INT32, DT_INT32, DT_INT32, DT_INT32}] = OP_KEY_3;
    opKeys[{DT_INT16, DT_INT16, DT_INT16, DT_INT16}] = OP_KEY_4;
    opKeys[{DT_BF16, DT_BF16, DT_BF16, DT_BF16}] = OP_KEY_5;
}

uint64_t FusedMulAddNTiling::GetOpKey(
    ge::DataType inputX1Dtype, ge::DataType inputX2Dtype, ge::DataType inputX3Dtype, ge::DataType outputYDtype)
{
    std::tuple<ge::DataType, ge::DataType, ge::DataType, ge::DataType> condition = {
        inputX1Dtype, inputX2Dtype, inputX3Dtype, outputYDtype};
    if (opKeys.find(condition) != opKeys.end()) {
        return opKeys[condition];
    }
    return OP_KEY_INVALID;
}

uint64_t FusedMulAddNTiling::GenerateTilingKey(uint64_t innerKey)
{
    return opKey * OP_KEY_OFFSET + innerKey;
}

std::map<uint64_t, ComputeParams> FusedMulAddNTiling::GetComputeMap(uint64_t paramOpKey)
{
    ComputeParams computeParams0;
    switch (paramOpKey) {
        case OP_KEY_1:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS32_SIZE);
            computeParams0.extraSize = {0};
            computeParams0.bufferDivisor = {192};
            return {{0, computeParams0}};
        case OP_KEY_2:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0};
            computeParams0.bufferDivisor = {96};
            return {{0, computeParams0}};
        case OP_KEY_3:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS32_SIZE);
            computeParams0.extraSize = {0};
            computeParams0.bufferDivisor = {192};
            return {{0, computeParams0}};
        case OP_KEY_4:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS16_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0};
            computeParams0.bufferDivisor = {96};
            return {{0, computeParams0}};
        case OP_KEY_5:
            computeParams0.maxDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0};
            computeParams0.bufferDivisor = {96};
            return {{0, computeParams0}};
        default:
            return {};
    }
}

ge::graphStatus FusedMulAddNTiling::GetShapeAttrsInfo()
{
    auto inputX1 = context_->GetInputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX1);
    auto inputX1Dtype = inputX1->GetDataType();
    auto inputX2 = context_->GetInputDesc(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX2);
    auto inputX2Dtype = inputX2->GetDataType();
    auto inputX3 = context_->GetInputDesc(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX3);
    auto inputX3Dtype = inputX3->GetDataType();
    auto outputY = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputY);
    auto outputYDtype = outputY->GetDataType();

    SetOpKey();
    opKey = GetOpKey(inputX1Dtype, inputX2Dtype, inputX3Dtype, outputYDtype);
    if (opKey == OP_KEY_INVALID) {
        std::string dtypeMsg = Ops::Base::ToString(inputX1Dtype) + ", " + Ops::Base::ToString(inputX2Dtype) + ", " +
                               Ops::Base::ToString(inputX3Dtype) + " and " + Ops::Base::ToString(outputYDtype);
        std::string reasonMsg =
            "The dtypes of all input and output parameters x1, x2, x3 and y must be the same, "
            "and be in the range of FLOAT, FLOAT16, INT32, INT16 and BF16";
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1, x2, x3 and y", dtypeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FusedMulAddNTiling::IsCapable()
{
    return true;
}

ge::graphStatus FusedMulAddNTiling::DoOpTiling()
{
    auto inputX1Shape = context_->GetInputShape(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX1Shape);
    auto shapeX1 = inputX1Shape->GetStorageShape();
    auto inputX2Shape = context_->GetInputShape(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX2Shape);
    auto shapeX2 = inputX2Shape->GetStorageShape();
    auto inputX3Shape = context_->GetInputShape(INDEX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX3Shape);
    auto shapeX3 = inputX3Shape->GetStorageShape();
    if (shapeX1 != shapeX2) {
        std::string shapeMsg = Ops::Base::ToString(shapeX1) + " and " + Ops::Base::ToString(shapeX2);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "x1 and x2", shapeMsg.c_str(), "The shapes of x1 and x2 must be the same");
        return ge::GRAPH_FAILED;
    }

    if (shapeX3.GetShapeSize() != 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(
            context_->GetNodeName(), "x3", std::to_string(shapeX3.GetShapeSize()).c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    ElewiseTilingParams elewiseTilingParams;
    elewiseTilingParams.shape = inputX1Shape->GetStorageShape();
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

std::string FusedMulAddNTiling::ToString(FusedMulAddNTilingData& paramTilingData)
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

ge::graphStatus FusedMulAddNTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t FusedMulAddNTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus FusedMulAddNTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedMulAddNTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGI(context_, "TilingInfo: %s.", ToString(tilingData).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForFusedMulAddN(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const FusedMulAddNCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    FusedMulAddNTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForFusedMulAddN(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<FusedMulAddNCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedMulAddN)
    .Tiling(TilingForFusedMulAddN)
    .TilingParse<FusedMulAddNCompileInfo>(TilingPrepareForFusedMulAddN);

} // namespace optiling
