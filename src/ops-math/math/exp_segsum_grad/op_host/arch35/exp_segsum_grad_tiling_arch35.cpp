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
 * \file exp_segsum_grad_tiling_arch35.cpp
 * \brief arch35 / Ascend950 tiling implementation for ExpSegsumGrad.
 */
#include <algorithm>
#include "exp_segsum_grad_tiling_arch35.h"
#include "log/log.h"
#include "tiling/tiling_api.h"

namespace optiling {

namespace {
constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint64_t REDUCE_SUM_SIZE = 20 * 1024;
constexpr uint8_t BYTE_LEN_4 = 4;
constexpr uint8_t BYTE_LEN_2 = 2;
constexpr int32_t BLOCK = 32;
constexpr int32_t TAILNUM = 2;
constexpr uint32_t COMMON_TILING_KEY = 0;
constexpr uint32_t SMALL_SIZE_TILING_KEY = 1;
constexpr uint32_t TENSOR_NUM = 6;
constexpr uint32_t INPUT_OUTPUT_TENSOR_NUM = 3;
constexpr uint32_t FP32_TENSOR_NUM = 3;
constexpr uint8_t SCHEDULE_MODE = 1; // batchmode模式，核间同步算子需要设置该模式
} // namespace

uint8_t ExpSegsumGradTilingArch35::GetDataTypeSize() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return BYTE_LEN_4;
    }
}

ge::graphStatus ExpSegsumGradTilingArch35::ParseInputAttrs()
{
    auto srcGradOutShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, srcGradOutShape);
    int32_t gradOutDim = srcGradOutShape->GetStorageShape().GetDimNum();
    auto gradOutShape = srcGradOutShape->GetOriginShape();
    for (int8_t i = 0; i < gradOutDim - TAILNUM; i++) {
        batches *= gradOutShape.GetDim(i);
    }
    tailDimLength = gradOutShape.GetDim(gradOutDim - 1);
    auto temp = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, temp);
    dataType = temp->GetDataType();
    return ge::GRAPH_SUCCESS;
}

void ExpSegsumGradTilingArch35::GetTilingKey(uint64_t ubSizePlatform)
{
    int32_t blockSizeReal = BLOCK / GetDataTypeSize();
    int64_t calNumAlign = CeilA2B(tailDimLength, blockSizeReal) * blockSizeReal;
    int64_t bytesPerElement = INPUT_OUTPUT_TENSOR_NUM * GetDataTypeSize() + FP32_TENSOR_NUM * BYTE_LEN_4;
    int64_t fullLoadSize = ubSizePlatform / bytesPerElement / blockSizeReal * blockSizeReal;
    if (tailDimLength <= fullLoadSize) {
        int64_t rowNum = std::max<int64_t>(fullLoadSize / calNumAlign, 1);
        while (rowNum > 1) {
            auto shape = ge::Shape({rowNum, calNumAlign});
            uint32_t maxValue = 0;
            uint32_t minValue = 0;
            AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DataType::DT_FLOAT, AscendC::ReducePattern::AR, true, false,
                                               maxValue, minValue);
            int64_t slideBytes = rowNum * calNumAlign * bytesPerElement;
            if (slideBytes + static_cast<int64_t>(minValue) <= static_cast<int64_t>(ubSizePlatform)) {
                break;
            }
            --rowNum;
        }
        tilingKey = SMALL_SIZE_TILING_KEY;
        slideSize = rowNum * calNumAlign;
        return;
    }

    // The COMMON fallback still uses the advanced ReduceSum API and reserves its temporary UB.
    int64_t eachTensorSize = (ubSizePlatform - REDUCE_SUM_SIZE) / TENSOR_NUM;
    int64_t maxSlideSizeUnalign = eachTensorSize / BYTE_LEN_4;
    int64_t maxSlideSize = maxSlideSizeUnalign / (BLOCK / BYTE_LEN_4) * (BLOCK / BYTE_LEN_4);

    int64_t rowNum = 1;
    int64_t colNum = maxSlideSize;
    auto shape = ge::Shape({rowNum, colNum});
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DataType::DT_FLOAT, AscendC::ReducePattern::AR, true, false, maxValue,
                                       minValue);
    if (minValue > REDUCE_SUM_SIZE) {
        int64_t size = CeilA2B(CeilA2B(minValue - REDUCE_SUM_SIZE, TENSOR_NUM), BLOCK) * BLOCK;
        maxSlideSize = maxSlideSize - size / BYTE_LEN_4;
    }
    tilingKey = COMMON_TILING_KEY;
    slideSize = maxSlideSize;
}

void ExpSegsumGradTilingArch35::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    int64_t averageBatches = CeilA2B(batches, coreNumPlatform);
    needCoreNum = CeilA2B(batches, averageBatches);
    for (int64_t coreIndex = 0; coreIndex < needCoreNum; coreIndex++) {
        batchStart[coreIndex] = coreIndex * averageBatches;
        batchEnd[coreIndex] = std::min((coreIndex + 1) * averageBatches, batches);
    }
}
void ExpSegsumGradTilingArch35::FillTilingData(ExpSegsumGradTilingDataArch35* tiling)
{
    tiling->needCoreNum = static_cast<int64_t>(needCoreNum);
    tiling->batches = batches;
    tiling->tailDimLength = tailDimLength;
    tiling->slideSize = slideSize;
    for (uint16_t i = 0; i < EXP_SEGSUM_GRAD_MAX_CORE_ARCH35; i++) {
        tiling->batchStart[i] = batchStart[i];
        tiling->batchEnd[i] = batchEnd[i];
    }
}

ge::graphStatus ExpSegsumGradTilingArch35::RunTiling()
{
    OP_CHECK_IF(ParseInputAttrs() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "ExpSegsumGrad arch35 ParseInputAttrs failed"),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfoPtr = tilingContext->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t coreNumPlatform = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((static_cast<int32_t>(coreNumPlatform) <= 0),
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "ExpSegsumGrad arch35 get coreNum failed: %u",
                                       coreNumPlatform),
                return ge::GRAPH_FAILED);
    // POD batchStart/batchEnd arrays are sized EXP_SEGSUM_GRAD_MAX_CORE_ARCH35; clamp defensively.
    if (coreNumPlatform > EXP_SEGSUM_GRAD_MAX_CORE_ARCH35) {
        coreNumPlatform = EXP_SEGSUM_GRAD_MAX_CORE_ARCH35;
    }
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    OP_CHECK_IF((static_cast<int64_t>(ubSizePlatform) <= 0),
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "ExpSegsumGrad arch35 get ubSize failed: %lu",
                                       ubSizePlatform),
                return ge::GRAPH_FAILED);

    if (batches == 0 || tailDimLength == 0) {
        // Runtime requires a non-zero block dimension even when there is no output to compute.
        needCoreNum = 1;
        tilingKey = SMALL_SIZE_TILING_KEY;
        slideSize = 0;
    } else {
        GetNeedCoreNum(coreNumPlatform);
        GetTilingKey(ubSizePlatform);
    }

    auto tiling = tilingContext->GetTilingData<ExpSegsumGradTilingDataArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    *tiling = ExpSegsumGradTilingDataArch35{};
    FillTilingData(tiling);

    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(tilingKey);
    if (tilingKey == COMMON_TILING_KEY) {
        OP_CHECK_IF(tilingContext->SetScheduleMode(SCHEDULE_MODE) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "ExpSegsumGrad arch35 SetScheduleMode failed"),
                    return ge::GRAPH_FAILED);
    }

    size_t sysWsSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, workspaces);
    workspaces[0] = WORK_SPACE_SIZE + sysWsSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ExpSegsumGradArch35(gert::TilingContext* context)
{
    ExpSegsumGradTilingArch35 tilingObject(context);
    return tilingObject.RunTiling();
}

static ge::graphStatus TilingPrepare4ExpSegsumGradArch35([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ExpSegsumGrad)
    .Tiling(Tiling4ExpSegsumGradArch35)
    .TilingParse<ExpSegsumGradCompileInfoArch35>(TilingPrepare4ExpSegsumGradArch35);
} // namespace optiling
