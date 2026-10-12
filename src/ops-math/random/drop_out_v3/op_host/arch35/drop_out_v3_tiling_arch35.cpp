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
 * \file drop_out_v3_tiling_arch35.cpp
 * \brief
 */

#include "drop_out_v3_tiling_arch35.h"
#include <string>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/fp16.h"
#include "util/bfloat16.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {

static constexpr uint16_t INPUT_IDX_X = 0;
static constexpr uint16_t INPUT_IDX_P = 2;
static constexpr uint16_t INPUT_IDX_SEED = 3;
static constexpr uint16_t INPUT_IDX_OFFSET = 4;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t OUTPUT_IDX_MASK = 1;
static constexpr int64_t DCACHE_SIZE = 32768;
static constexpr int64_t CORE_ALIGN_SIZE = 256;
static constexpr int64_t ALIGNMENT_32 = 32;
static constexpr int64_t OFFSET_LIMIT = 4;
static constexpr int64_t MASK_ALIGN_SIZE = 128;
static constexpr int64_t UINT8_BIT_SIZE = 8;
static constexpr int64_t VEC_INIT = 8;
static constexpr int64_t NUM_2 = 2;
static constexpr int64_t NUM_4 = 4;
static constexpr int64_t NUM_8 = 8;
static constexpr int64_t NUM_16 = 16;
static constexpr int64_t DROPOUT_CORE_GRANULARITY = 32;

OpTilingConfig DropOutV3Tiling::BuildOpConfig(gert::TilingContext* context)
{
    OpTilingConfig config;

    int64_t xSize = -1;
    int64_t maskSize = -1;
    if (context != nullptr) {
        auto inputShape = context->GetRequiredInputShape(INPUT_IDX_X);
        if (inputShape != nullptr) {
            auto storageShape = inputShape->GetStorageShape();
            xSize = storageShape.IsScalar() ? 1 : storageShape.GetShapeSize();
            maskSize = Ops::Base::CeilAlign(xSize, MASK_ALIGN_SIZE) / UINT8_BIT_SIZE;
        }
    }

    config.inputCheckRules = {
        {INPUT_IDX_X, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}},
        {INPUT_IDX_P, {{ge::DT_DOUBLE, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}},
        {INPUT_IDX_SEED, {{ge::DT_INT32, ge::DT_INT64}, 1, {}, nullptr}},
        {INPUT_IDX_OFFSET, {{ge::DT_INT64}, 2, {}, nullptr}}};
    config.outputCheckRules = {{OUTPUT_IDX_Y, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, xSize, {}, nullptr}},
                               {OUTPUT_IDX_MASK, {{ge::DT_UINT8}, maskSize, {}, nullptr}}};

    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& size) {
        auto inputShape = ctx->GetRequiredInputShape(INPUT_IDX_X);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, inputShape);
        auto storageShape = inputShape->GetStorageShape();
        size = storageShape.IsScalar() ? 1 : storageShape.GetShapeSize();
        return ge::GRAPH_SUCCESS;
    };

    config.getSeedAndOffset = [](gert::TilingContext* ctx, int64_t& seed, int64_t& offset) {
        gert::Shape seedShape;
        auto ret = ExtractTensorValue(ctx, INPUT_IDX_SEED, seedShape);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(ctx->GetNodeName(), "get seed value failed"),
                    return ge::GRAPH_FAILED);
        seed = static_cast<int64_t>(seedShape.GetDim(0));
        gert::Shape offsetShape;
        ret = ExtractTensorValue(ctx, INPUT_IDX_OFFSET, offsetShape);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(ctx->GetNodeName(), "get offset value failed"),
                    return ge::GRAPH_FAILED);
        offset = static_cast<int64_t>(offsetShape.GetDim(1));
        if (offset % OFFSET_LIMIT != 0) {
            std::string valueStr = std::to_string(offset);
            std::string reasonMsg = "The offset must be a multiple of 4";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "input offset", valueStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    config.kernelMode = RandomKernelMode::SIMT;
    config.DcacheSize = DCACHE_SIZE;
    config.isNeedSyncAll = true;
    config.coreAlignSize = CORE_ALIGN_SIZE;
    return config;
}

ge::graphStatus DropOutV3Tiling::UniqueProcess()
{
    dropOutV3TilingData_.usedCoreNum = simtTilingData_.usedCoreNum;
    dropOutV3TilingData_.outputSize = simtTilingData_.outputSize;
    dropOutV3TilingData_.seed = simtTilingData_.seed;
    dropOutV3TilingData_.offset = simtTilingData_.offset;
    dropOutV3TilingData_.ubSize = ubSize_;

    auto pTensor = context_->GetRequiredInputTensor(INPUT_IDX_P);
    OP_CHECK_NULL_WITH_CONTEXT(context_, pTensor);
    if (pTensor->GetShapeSize() <= 0) {
        std::string valueStr = std::to_string(pTensor->GetShapeSize());
        std::string reasonMsg = "shape size of prob tensor must be greater than 0";
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "shape size of prob tensor",
                                                  valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    auto pDescPtr = context_->GetRequiredInputDesc(INPUT_IDX_P);
    OP_CHECK_NULL_WITH_CONTEXT(context_, pDescPtr);
    float prob = 0.0f;
    switch (pDescPtr->GetDataType()) {
        case ge::DT_DOUBLE: {
            prob = static_cast<float>(double(1) - pTensor->GetData<double>()[0]);
            break;
        }
        case ge::DT_FLOAT16: {
            auto srcP = pTensor->GetData<Ops::Base::fp16_t>()[0];
            prob = 1.0f - srcP.toFloat();
            break;
        }
        case ge::DT_BF16: {
            float srcP = pTensor->GetData<Ops::Base::bfloat16>()[0];
            prob = 1.0f - srcP;
            break;
        }
        case ge::DT_FLOAT: {
            prob = 1.0f - pTensor->GetData<float>()[0];
            break;
        }
        default: {
            std::string valueStr = Ops::Base::ToString(pDescPtr->GetDataType());
            std::string reasonMsg = "Unsupported p dtype";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "input p", valueStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (prob < 0.0f || prob > 1.0f) {
        std::string valueStr = std::to_string(1.0f - prob);
        std::string reasonMsg = "The value of p has to be between 0 and 1";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "input p", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    dropOutV3TilingData_.prob = prob;

    auto xDescPtr = context_->GetRequiredInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDescPtr);
    auto xDtype = xDescPtr->GetDataType();
    auto sizeofT = static_cast<uint32_t>(ge::GetSizeByDataType(xDtype));

    int64_t outputSize = dropOutV3TilingData_.outputSize;
    int64_t totalCoreNum = totalCoreNum_;

    // 计算 VEC
    uint32_t vec = static_cast<uint32_t>(VEC_INIT);
    if (outputSize % NUM_2 != 0) {
        vec = 1;
    } else {
        uint32_t optimalVec = static_cast<uint32_t>(NUM_16 / sizeofT);
        vec = std::min(static_cast<uint32_t>(VEC_INIT), optimalVec);
        while (vec > 1 && static_cast<uint64_t>(outputSize) % vec != 0) {
            vec /= NUM_2;
        }
    }
    dropOutV3TilingData_.vec = vec;
    dropOutV3TilingData_.transportMode = (vec == 1) ? 1 : 0;

    // 计算 grid 和 totalThreads
    int64_t blockSize = SIMT_THREAD_GROUP_SIZE;
    int64_t maxThreadsPerMultiProcessor = MAX_THREADS_PER_AIC;
    int64_t blocksPerSM = maxThreadsPerMultiProcessor / blockSize;
    int64_t multiProcessorCount = AIC_CLUSTER_COUNT;
    int64_t blocksCount = multiProcessorCount * blocksPerSM;
    int64_t grid = (outputSize + blockSize - 1) / blockSize;
    grid = (blocksCount < grid) ? blocksCount : grid;
    dropOutV3TilingData_.totalThreads = static_cast<uint64_t>(grid * blockSize);

    // 分核
    int64_t coreGranularity = DROPOUT_CORE_GRANULARITY;
    int64_t avgPerCore = Ops::Base::CeilDiv(outputSize, totalCoreNum);
    int64_t perCoreElements = Ops::Base::CeilAlign(avgPerCore, coreGranularity);
    perCoreElements = std::max(perCoreElements, coreGranularity);
    int64_t usedCoreNum = std::min(totalCoreNum, Ops::Base::CeilDiv(outputSize, perCoreElements));
    usedCoreNum = std::max(1L, usedCoreNum);
    int64_t tailCoreElements = outputSize - (perCoreElements * (usedCoreNum - 1));

    dropOutV3TilingData_.usedCoreNum = usedCoreNum;
    dropOutV3TilingData_.perCoreElements = perCoreElements;
    dropOutV3TilingData_.tailCoreElements = tailCoreElements;

    // 分 UB (双buffer: 2*input + 2*output + 2*maskBit + randomFloatBuf)
    int64_t perBlockBytes = (NUM_2 * coreGranularity * sizeofT) + (NUM_2 * coreGranularity * sizeofT) +
                            (NUM_2 * coreGranularity / NUM_8) + (coreGranularity * NUM_4);
    int64_t ubFactorElements = Ops::Base::FloorDiv(ubSize_, perBlockBytes) * coreGranularity;
    while (NUM_2 * Ops::Base::CeilAlign(ubFactorElements * sizeofT, ALIGNMENT_32) +
               NUM_2 * Ops::Base::CeilAlign(ubFactorElements * sizeofT, ALIGNMENT_32) +
               NUM_2 * Ops::Base::CeilAlign(ubFactorElements / NUM_8, ALIGNMENT_32) +
               Ops::Base::CeilAlign(ubFactorElements * NUM_4, ALIGNMENT_32) >
           ubSize_) {
        ubFactorElements -= coreGranularity;
    }
    ubFactorElements = std::max(ubFactorElements, coreGranularity);
    dropOutV3TilingData_.ubFactorElements = ubFactorElements;

    dropOutV3TilingData_.ubLoopCount = Ops::Base::CeilDiv(perCoreElements, ubFactorElements);
    dropOutV3TilingData_.tailUbFactorElements = perCoreElements -
                                                (dropOutV3TilingData_.ubLoopCount - 1) * ubFactorElements;
    dropOutV3TilingData_.tailUbLoopCount = Ops::Base::CeilDiv(tailCoreElements, ubFactorElements);
    dropOutV3TilingData_.tailCoreTailUbFactorElements = tailCoreElements -
                                                        (dropOutV3TilingData_.tailUbLoopCount - 1) * ubFactorElements;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    workspaceSize_ = Ops::Base::CeilAlign(outputSize, ALIGNMENT_32) * sizeof(uint8_t) +
                     ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DropOutV3Tiling::WriteBackToContext()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    context_->SetBlockDim(dropOutV3TilingData_.usedCoreNum);
    context_->SetScheduleMode(config_.isNeedSyncAll);
    context_->SetTilingKey(tilingKey_);
    if (config_.DcacheSize != 0) {
        auto res = context_->SetLocalMemorySize(ubSize_);
        OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
                    OP_LOGE(opName_, "SetLocalMemorySize ubSize = %ld failed.", static_cast<int64_t>(ubSize_)),
                    return ge::GRAPH_FAILED);
    }

    auto* tilingData = context_->GetTilingData<DropOutV3TilingDataStruct>();
    OP_CHECK_IF(tilingData == nullptr, OP_LOGE(opName_, "DropOutV3 tilingData ptr is null"), return ge::GRAPH_FAILED);
    *tilingData = dropOutV3TilingData_;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4DropOutV3(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4DropOutV3 running DropOutV3 tiling.");
    DropOutV3Tiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4DropOutV3(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "DropOutV3");
}

IMPL_OP_OPTILING(DropOutV3)
    .Tiling(Tiling4DropOutV3)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4DropOutV3)
    .TilingInputsDataDependency({INPUT_IDX_P, INPUT_IDX_SEED, INPUT_IDX_OFFSET});
} // namespace optiling
