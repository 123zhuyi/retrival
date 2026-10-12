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
 * \file elewise_tiling.cpp
 * \brief atvoss elewise template tiling
 */

#include "op_common/atvoss/elewise/elewise_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace Ops {
namespace Base {

static constexpr int64_t REPEAT_BYTES = 256;
static constexpr uint64_t COMPUTE_KEY_OFFSET = 1000000;
static constexpr uint64_t COMPUTE_KEY = 0;
static constexpr uint64_t SCHEDULE_KEY = 100;
static constexpr int64_t MIN_TILING_BITS_SIZE_PER_CORE = 32768; // 4KB
static constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;

static uint64_t GetComputeKey() { return COMPUTE_KEY; }

static uint64_t GetScheduleKey() { return SCHEDULE_KEY; }

static int64_t GetMaxElemNum(int64_t ubSize, const ComputeParams& computeParams)
{
    int64_t minDtypeBits = computeParams.minDtypeBits;
    int64_t extraSize = computeParams.extraSize[0];
    int64_t bufferDivisor = computeParams.bufferDivisor[0];
    int64_t maxElemNums = (ubSize - extraSize) * BITS_NUM / bufferDivisor;
    int64_t alignFactor = REPEAT_BYTES * BITS_NUM / minDtypeBits;
    int64_t maxElemNumAlign = maxElemNums / alignFactor * alignFactor;
    return maxElemNumAlign;
}

ge::graphStatus ElewiseTiling(const ElewiseTilingParams& elewiseTilingParams, ElewiseTilingData& elewiseTilingData)
{
    OP_LOGD("ElewiseTiling", "Enter ElewiseTiling.");
    int64_t dim0 = 1;
    for (uint64_t i = 0; i < elewiseTilingParams.shape.GetDimNum(); i++) {
        OP_CHECK_IF(elewiseTilingParams.shape.GetDim(i) == 0,
                    OP_LOGE("ElewiseTiling", "elewiseTiling dim can not be 0"), return ge::GRAPH_FAILED);
        dim0 = dim0 * elewiseTilingParams.shape.GetDim(i);
    }

    uint64_t computeKey = GetComputeKey();

    auto iter = elewiseTilingParams.computeMap.find(computeKey);
    ComputeParams computeParams;
    if (iter != elewiseTilingParams.computeMap.end()) {
        computeParams = iter->second;
    } else {
        OP_LOGE("ElewiseTiling", "can not find computeKey");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(elewiseTilingParams.coreNum == 0, OP_LOGE("ElewiseTiling", "coreNum can not be 0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(elewiseTilingParams.ubSize < computeParams.extraSize[0],
                OP_LOGE("ElewiseTiling", "ubSize is smaller than extra size"), return ge::GRAPH_FAILED);

    int64_t coreNum = (dim0 * computeParams.minDtypeBits + MIN_TILING_BITS_SIZE_PER_CORE - 1) /
                      MIN_TILING_BITS_SIZE_PER_CORE;
    if (coreNum > elewiseTilingParams.coreNum) {
        coreNum = elewiseTilingParams.coreNum;
    }
    int64_t blockFormer = ((dim0 + coreNum - 1) / coreNum + CACHE_LINE_BYTE_LENGTH - 1) / CACHE_LINE_BYTE_LENGTH *
                          CACHE_LINE_BYTE_LENGTH;
    int64_t blockNum = (dim0 + blockFormer - 1) / blockFormer;
    int64_t blockTail = dim0 - (blockNum - 1) * blockFormer;

    int64_t maxElemNum = GetMaxElemNum(elewiseTilingParams.ubSize, computeParams);
    int64_t ubFormer = maxElemNum;
    OP_CHECK_IF(ubFormer == 0, OP_LOGE("ElewiseTiling", "ubFormer can not be 0"), return ge::GRAPH_FAILED);
    int64_t ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
    int64_t ubLoopOfTailBlock = (blockTail + ubFormer - 1) / ubFormer;
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;

    uint64_t scheduleKey = GetScheduleKey();

    elewiseTilingData.innerKey = computeKey * COMPUTE_KEY_OFFSET + scheduleKey;
    elewiseTilingData.dim0 = dim0;
    elewiseTilingData.blockFormer = blockFormer;
    elewiseTilingData.blockNum = blockNum;
    elewiseTilingData.ubFormer = ubFormer;
    elewiseTilingData.ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    elewiseTilingData.ubLoopOfTailBlock = ubLoopOfTailBlock;
    elewiseTilingData.ubTailOfFormerBlock = ubTailOfFormerBlock;
    elewiseTilingData.ubTailOfTailBlock = ubTailOfTailBlock;
    elewiseTilingData.elemNum = maxElemNum;
    elewiseTilingData.coreNum = coreNum;
    return ge::GRAPH_SUCCESS;
}

bool IsSameElewiseShape(const gert::Shape& shape1, const gert::Shape& shape2)
{
    if (shape1.IsScalar() || shape2.IsScalar() || shape1.GetShapeSize() == 1 || shape2.GetShapeSize() == 1) {
        return true;
    }

    if (shape1.GetDimNum() != shape2.GetDimNum()) {
        OP_LOGE("ElewiseTiling", "elewise shape dim num is not equal, shape1 is %zu, shape2 is %zu", shape1.GetDimNum(),
                shape2.GetDimNum());
        return false;
    }

    for (size_t i = 0; i < shape1.GetDimNum(); i++) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            OP_LOGE("ElewiseTiling", "elewise shape dim %zu not equal, shape1 is %ld, shape2 is %ld", i,
                    shape1.GetDim(i), shape2.GetDim(i));
            return false;
        }
    }

    return true;
}

ge::graphStatus ElewiseBaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize = ubSizePlatForm;

    return ge::GRAPH_SUCCESS;
}

void ElewiseBaseTiling::AdaptEleBaseTilingData(const ElewiseTilingData& elewiseTilingData,
                                               EleBaseTilingData& eleBaseTilingData) const
{
    OP_LOGD("ElewiseTiling", "Enter AdaptEleBaseTilingData.");
    eleBaseTilingData.scheMode = GetScheMode();
    eleBaseTilingData.dim0 = elewiseTilingData.dim0;
    eleBaseTilingData.blockFormer = elewiseTilingData.blockFormer;
    eleBaseTilingData.blockNum = elewiseTilingData.blockNum;
    eleBaseTilingData.ubFormer = elewiseTilingData.ubFormer;
    eleBaseTilingData.ubLoopOfFormerBlock = elewiseTilingData.ubLoopOfFormerBlock;
    eleBaseTilingData.ubLoopOfTailBlock = elewiseTilingData.ubLoopOfTailBlock;
    eleBaseTilingData.ubTailOfFormerBlock = elewiseTilingData.ubTailOfFormerBlock;
    eleBaseTilingData.ubTailOfTailBlock = elewiseTilingData.ubTailOfTailBlock;
    eleBaseTilingData.elemNum = elewiseTilingData.elemNum;
    eleBaseTilingData.coreNum = static_cast<int32_t>(elewiseTilingData.coreNum);
}

void ElewiseBaseTiling::AdaptEleBaseTilingData16B(const ElewiseTilingData& elewiseTilingData,
                                                  EleBaseTilingData16B& eleBaseTilingData16B)
{
    OP_LOGD("ElewiseTiling", "Enter AdaptEleBaseTilingData16B.");

    eleBaseTilingData16B.dim0 = elewiseTilingData.dim0;
    eleBaseTilingData16B.ubFormer = elewiseTilingData.ubFormer;
    eleBaseTilingData16B.coreNum = elewiseTilingData.coreNum;

    numBlocks = elewiseTilingData.blockNum;
}

ge::graphStatus ElewiseBaseTiling::AdaptEleBaseTilingData24B(const ElewiseTilingData& elewiseTilingData)
{
    OP_LOGD("ElewiseTiling", "Enter AdaptEleBaseTilingData24B.");

    Ele24BTilingData = context_->GetTilingData<EleBaseTilingData24B>();
    OP_CHECK_IF((Ele24BTilingData == nullptr),
                OP_LOGD(context_->GetNodeName(), "Get Ele24BTilingData from GE context failed"),
                return ge::GRAPH_FAILED);

    Ele24BTilingData->dim0 = elewiseTilingData.dim0;
    Ele24BTilingData->ubFormer = elewiseTilingData.ubFormer;
    Ele24BTilingData->coreNum = elewiseTilingData.coreNum;

    numBlocks = elewiseTilingData.blockNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ElewiseBaseTiling::AdaptEleBaseTilingData32B(const ElewiseTilingData& elewiseTilingData)
{
    OP_LOGD("ElewiseTiling", "Enter AdaptEleBaseTilingData32B.");

    Ele32BTilingData = context_->GetTilingData<EleBaseTilingData32B>();
    OP_CHECK_IF((Ele32BTilingData == nullptr),
                OP_LOGD(context_->GetNodeName(), "Get Ele32BTilingData from GE context failed"),
                return ge::GRAPH_FAILED);

    Ele32BTilingData->dim0 = elewiseTilingData.dim0;
    Ele32BTilingData->ubFormer = elewiseTilingData.ubFormer;
    Ele32BTilingData->coreNum = elewiseTilingData.coreNum;

    numBlocks = elewiseTilingData.blockNum;
    return ge::GRAPH_SUCCESS;
}

int64_t ElewiseBaseTiling::GetScheMode() const { return DEFAULT_SCHEMODE; }

int64_t ElewiseBaseTiling::GetBlockDim() const { return numBlocks; }

} // namespace Base
} // namespace Ops
