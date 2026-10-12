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
 * \file stateless_random_choice_with_mask_simt_tiling.cpp
 * \brief
 */

#include "stateless_random_choice_with_mask_simt_tiling.h"
#include <string>
#include "random/stateless_random_choice_with_mask/op_kernel/arch35/stateless_random_choice_with_mask_struct.h"

namespace optiling {

const std::set<ge::DataType> SUPPORT_DTYPE = {ge::DT_BOOL};

bool StatelessRandomChoiceWithMaskSimtTiling::IsCapable()
{
    return true;
}

ge::graphStatus StatelessRandomChoiceWithMaskSimtTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const StatelessRandomChoiceWithMaskCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<uint64_t>(ubSizePlatform);
    }
    ubSize_ = ubSize_ - DCACHE_SIZE;
    OP_CHECK_IF(
        (coreNum_ <= 0 || ubSize_ <= 0),
        OP_LOGE(
            context_, "coreNum and ubSize should not be samller than 0, but got coreNum [%ld] and ubSize [%ld]",
            coreNum_, ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomChoiceWithMaskSimtTiling::GetShapeAttrsInfo()
{
    auto x = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    xShape_ = x->GetStorageShape();
    inputDim_ = xShape_.GetDimNum();
    if (inputDim_ > INPUT_X_MAX_DIM_NUM || inputDim_ < INPUT_X_MIN_DIM_NUM) {
        std::string valueStr = std::to_string(xShape_.GetDimNum());
        std::string reasonMsg = "xDimNum should be greater than 1 and smaller than 5";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "input tensor x", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    inputSize_ = xShape_.GetShapeSize();
    auto seed = context_->GetInputTensor(INPUT_SEED_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seed);
    const int64_t* seedValue = seed->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedValue);
    if (seed->GetShapeSize() <= 0) {
        std::string valueStr = std::to_string(seed->GetShapeSize());
        std::string reasonMsg = "inputSeed shapeSize need be greater than 0";
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context_->GetNodeName(), "input seed", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    seed_ = seed->GetData<int64_t>()[0];
    auto offset = context_->GetInputTensor(INPUT_OFFSET_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offset);
    const int64_t* offsetValue = offset->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetValue);
    if (offset->GetShapeSize() <= 0) {
        std::string valueStr = std::to_string(offset->GetShapeSize());
        std::string reasonMsg = "inputOffset shapeSize need be greater than 0";
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context_->GetNodeName(), "input offset", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_NULL_WITH_CONTEXT(context_, offset);
    offset_ = offset->GetData<int64_t>()[0];
    auto xDesc = context_->GetInputDesc(INPUT_X_IDX);
    ge::DataType xDtype = xDesc->GetDataType();
    if (SUPPORT_DTYPE.count(xDtype) == 0) {
        std::string valueStr = Ops::Base::ToString(xDtype);
        std::string reasonMsg = "input x dtype only support BOOL currently";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "input tensor x", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    auto y = context_->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, y);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomChoiceWithMaskSimtTiling::ComputeCoreNum()
{
    blockNum_ = Ops::Base::CeilDiv(inputSize_, THREAD_NUM);
    if (blockNum_ > coreNum_) {
        blockNum_ = coreNum_;
        normalCoreProNum_ = Ops::Base::CeilDiv(inputSize_, blockNum_);
        return ge::GRAPH_SUCCESS;
    }
    blockNum_ = !blockNum_ ? 1 : blockNum_;
    normalCoreProNum_ = THREAD_NUM;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomChoiceWithMaskSimtTiling::SetTilingData()
{
    StatelessRandomChoiceWithMaskSimtTilingData* tilingData =
        context_->GetTilingData<StatelessRandomChoiceWithMaskSimtTilingData>();
    for (int64_t i = 0; i < static_cast<int64_t>(inputDim_); i++) {
        tilingData->inputShape[i] = xShape_.GetDim(i);
    }
    tilingData->blockNum = blockNum_;
    tilingData->normalCoreProNum = normalCoreProNum_;
    tilingData->m = m_;
    tilingData->n = n_;
    tilingData->seed = seed_;
    tilingData->offset = offset_;
    tilingData->inputSize = inputSize_;
    tilingData->noZeroCalcCount = noZeroCalcCount_;
    tilingData->randomWorkspaceSize = randomWorkspaceSize_;
    tilingData->noZeroWorkspaceSize = noZeroWorkspaceSize_;
    tilingData->ubSize = ubSize_;
    tilingData->inputDim = inputDim_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomChoiceWithMaskSimtTiling::DoOpTiling()
{
    ComputeCoreNum();
    m_ = ROWS_NUM;
    int64_t total = blockNum_ * THREAD_NUM + 1;
    int64_t cols = (total + ALIGNMENT_32 - 1) / ALIGNMENT_32;
    n_ = Ops::Base::CeilAlign(cols, ALIGNMENT_32);
    noZeroCalcCount_ = blockNum_ * THREAD_NUM;
    noZeroWorkspaceSize_ = m_ * n_;
    randomWorkspaceSize_ = Ops::Base::CeilAlign(inputSize_, ALIGNMENT_256);

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomChoiceWithMaskSimtTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t StatelessRandomChoiceWithMaskSimtTiling::GetTilingKey() const
{
    uint64_t tilingKey = GET_TPL_TILING_KEY(TPL_SCH_MODE_0);
    return tilingKey;
}

ge::graphStatus StatelessRandomChoiceWithMaskSimtTiling::GetWorkspaceSize()
{
    size_t* workspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspace);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    workspace[0] = noZeroWorkspaceSize_ * INT64_SIZE + randomWorkspaceSize_ * TWO * INT32_SIZE +
                   ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomChoiceWithMaskSimtTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    context_->SetTilingKey(GetTilingKey());
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void StatelessRandomChoiceWithMaskSimtTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "blockNum: " << blockNum_;
    info << "normalCoreProNum: " << normalCoreProNum_;
    info << "m: " << m_;
    info << "n: " << n_;
    info << "seed: " << seed_;
    info << "offset: " << offset_;
    info << "inputSize: " << inputSize_;
    info << "noZeroCalcCount: " << noZeroCalcCount_;
    info << "noZeroWorkspaceSize: " << noZeroWorkspaceSize_;
    info << "randomWorkspaceSize: " << randomWorkspaceSize_;
    info << "ubSize: " << ubSize_;
    info << "inputDim: " << inputDim_;

    OP_LOGI(opName_, "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(StatelessRandomChoiceWithMask, StatelessRandomChoiceWithMaskSimtTiling, 0);

ge::graphStatus Tiling4StatelessRandomChoiceWithMask(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling for StatelessRandomChoiceWithMask is running.");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4StatelessRandomChoiceWithMask(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<StatelessRandomChoiceWithMaskCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StatelessRandomChoiceWithMask)
    .TilingInputsDataDependency({INPUT_SEED_IDX, INPUT_OFFSET_IDX})
    .Tiling(Tiling4StatelessRandomChoiceWithMask)
    .TilingParse<StatelessRandomChoiceWithMaskCompileInfo>(TilingPrepare4StatelessRandomChoiceWithMask);
} // namespace optiling