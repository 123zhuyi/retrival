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
 * \file stateless_random_normal_v2_tiling_arch35.cpp
 * \brief
 */
#include "stateless_random_normal_v2_tiling_arch35.h"
#include <string>
#include "platform/platform_info.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace optiling {

static const std::unordered_map<ge::DataType, uint32_t> OUTPUT_DATA_TYPE_TO_INT{
    {ge::DataType::DT_FLOAT, 1}, {ge::DataType::DT_FLOAT16, 2}, {ge::DataType::DT_BF16, 3}};

static constexpr uint16_t INPUT_IDX_SHAPE = 0;
static constexpr uint16_t INPUT_IDX_KEY = 1;
static constexpr uint16_t INPUT_IDX_COUNTER = 2;
static constexpr uint16_t INPUT_IDX_ALG = 3;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t SIZE_OF_FLOAT = 4;
static constexpr uint16_t SPLIT_UB_NUM = 5;
static constexpr int64_t COUNTER_NUMBER_LOW_BOUND = 1;

ge::graphStatus StatelessRandomNormalV2Tiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const StatelessRandomNormalV2CompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->aivNum;
        ubSize_ = compileInfoPtr->ubSize - REGBASE_CCEC_CACHE_SIZE;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        auto aivNum = ascendcPlatform.GetCoreNumAiv();
        OP_CHECK_IF(
            (aivNum <= 0), OP_LOGE(opName, "StatelessRandomNormalV2Tiling fail to get coreNum."),
            return ge::GRAPH_FAILED);
        coreNum_ = aivNum;
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        OP_CHECK_IF(
            (ubSizePlatForm <= REGBASE_CCEC_CACHE_SIZE),
            OP_LOGE(opName, "ub size less than REGBASE_CCEC_CACHE_SIZE Size. please check"), return ge::GRAPH_FAILED);
        ubSize_ = ubSizePlatForm - REGBASE_CCEC_CACHE_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomNormalV2Tiling::GetShapeAttrsInfo()
{
    auto res = GetInputInfo();
    if (res != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    res = GetOutputInfo();
    if (res != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomNormalV2Tiling::GetInputInfo()
{
    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto shapeValue = outputShape->GetStorageShape();
    uint32_t shapeRank = shapeValue.GetDimNum();
    for (uint32_t idx = 0; idx < shapeRank; idx++) {
        outputSize_ *= shapeValue.GetDim(idx);
    }

    auto algTensor = context_->GetInputTensor(INPUT_IDX_ALG);
    OP_CHECK_NULL_WITH_CONTEXT(context_, algTensor);
    if (algTensor->GetShapeSize() != 1) {
        std::string valueStr = std::to_string(algTensor->GetShapeSize());
        std::string reasonMsg = "alg data must be 1 tensor scalar";
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "input alg", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    const int32_t* algVal = algTensor->GetData<int32_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, algVal);
    alg_ = Algorithm(algVal[0]);
    if (alg_ == Algorithm::RNG_ALG_AUTO_SELECT) {
        alg_ = Algorithm::RNG_ALG_PHILOX;
    }
    if (alg_ != Algorithm::RNG_ALG_PHILOX) {
        std::string valueStr = std::to_string(static_cast<int32_t>(alg_));
        std::string reasonMsg = "alg only support RNG_ALG_PHILOX";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "input alg", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomNormalV2Tiling::GetOutputInfo()
{
    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    outputDtype_ = outputDesc->GetDataType();
    auto iter = OUTPUT_DATA_TYPE_TO_INT.find(outputDtype_);
    if (iter != OUTPUT_DATA_TYPE_TO_INT.end()) {
        outputDtypeVal_ = iter->second;
    } else {
        std::string valueStr = Ops::Base::ToString(outputDtype_);
        std::string reasonMsg = "output dtype not supported";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "output tensor y", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T1, typename T2>
inline T1 StatelessRandomNormalV2Tiling::CeilDiv(const T1 a, const T2 b) const
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

void StatelessRandomNormalV2Tiling::BlockTiling()
{
    outputDtypeSize_ = SIZE_OF_FLOAT;
    auto coreAlignFactor = CORE_ALIGN_SIZE / outputDtypeSize_;
    auto blockFactor = CeilDiv(outputSize_, coreNum_);
    auto blockAlignFactor = CeilDiv(blockFactor, coreAlignFactor) * coreAlignFactor;
    auto minTilingSize = MIN_TILING_SIZE;
    blockTilingSize_ = std::max(static_cast<uint32_t>(blockAlignFactor), minTilingSize);
    blockNum_ = CeilDiv(outputSize_, blockTilingSize_);
    tailBlockTilingSize_ = outputSize_ - blockTilingSize_ * (blockNum_ - 1);
    OP_LOGD(
        opName,
        "outputSize = %lld, blockFactor = %lld, blockAlignFactor = %lld,"
        "blockTilingSize = %d, tailBlockTilingSize = %d",
        outputSize_, blockFactor, blockAlignFactor, blockTilingSize_, tailBlockTilingSize_);
    return;
}

ge::graphStatus StatelessRandomNormalV2Tiling::UbTiling()
{
    // splitUbSize: 2 for double buffer; 3 for data converse
    auto splitUbSize = ubSize_ / SPLIT_UB_NUM;
    auto alignFactor = BLOCK_SIZE_BYTES / outputDtypeSize_;
    ubTilingSize_ = CeilDiv(splitUbSize / outputDtypeSize_, alignFactor) * alignFactor;
    OP_LOGD(opName, "splitUbSize = %u, ubTilingSize = %u", splitUbSize, ubTilingSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomNormalV2Tiling::DoOpTiling()
{
    BlockTiling();
    ge::graphStatus res = UbTiling();
    if (res == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomNormalV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t StatelessRandomNormalV2Tiling::GetTilingKey() const
{
    uint64_t tilingKey = 100;
    tilingKey += outputDtypeVal_;
    OP_LOGD(opName, "tilingKey = %lld.", tilingKey);
    return tilingKey;
}

ge::graphStatus StatelessRandomNormalV2Tiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomNormalV2Tiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum_);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void StatelessRandomNormalV2Tiling::SetTilingData()
{
    tilingData.set_blockNum(blockNum_);
    tilingData.set_blockTilingSize(blockTilingSize_);
    tilingData.set_tailBlockTilingSize(tailBlockTilingSize_);
    tilingData.set_ubTilingSize(ubTilingSize_);
    tilingData.set_alg(static_cast<uint32_t>(alg_));
    tilingData.set_key(key_);
    tilingData.set_counter(counter_);
    return;
}

ge::graphStatus Tiling4StatelessRandomNormalV2(gert::TilingContext* context)
{
    StatelessRandomNormalV2Tiling tilingObj(context);
    return tilingObj.DoTiling();
}

static ge::graphStatus TilingPrepare4StatelessRandomNormalV2(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<StatelessRandomNormalV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StatelessRandomNormalV2)
    .Tiling(Tiling4StatelessRandomNormalV2)
    .TilingParse<StatelessRandomNormalV2CompileInfo>(TilingPrepare4StatelessRandomNormalV2)
    .TilingInputsDataDependency({INPUT_IDX_ALG});

} // namespace optiling