/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please
 * refer to the License for details. You may not use this file except in compliance with the License. THIS SOFTWARE IS
 * PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
 * repository for the full text of the License.
 */

/*!
 * \file split_v_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <vector>

#include "../op_kernel/split_v_tiling_data.h"
#include "../op_kernel/split_v_tiling_key.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"

namespace optiling {
static constexpr uint32_t GM_ALIGN_SIZE = 512;
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t QUEUE_NUM = 1; // TQueBind
static constexpr uint32_t UB_RESERVED_BYTES = 1024;
static constexpr uint32_t MAX_COPY_BYTES = 65535;
static constexpr uint32_t MAX_COPY_BYTES_8BIT = MAX_COPY_BYTES * 2;
static constexpr uint32_t MAX_DATA_COPY_EXT_BLOCK_LEN = 2097151;
static constexpr uint32_t MAX_DATA_COPY_BLOCK_COUNT = 4095;
static constexpr uint32_t VNCHW_TILE_SIDE = 16;
static constexpr uint32_t MAX_TRANS_SPLIT_TILE = 128;
static constexpr uint32_t LARGE_SAME_LEN_FULL_ROW_TILE = 48;
static constexpr uint32_t SAME_LEN_COMPACT_OUTER_TILE = 256;
static constexpr uint32_t SAME_LEN_COMPACT_LARGE_OUTER_TILE = 2048;
static constexpr uint32_t SAME_LEN_COMPACT_B16_ADDR_COUNT = 16;
static constexpr bool ENABLE_SAME_LEN_INNER_FULL_ROW_PACK = true;

struct SplitVCompileInfo {};

static constexpr size_t MaxSizeT(size_t lhs, size_t rhs) { return lhs > rhs ? lhs : rhs; }

static constexpr size_t SPLIT_V_TILING_DATA_SIZE =
    MaxSizeT(
        sizeof(SplitVTilingData),
        MaxSizeT(
            sizeof(SplitVTilingDataPureCopy),
            MaxSizeT(
                sizeof(SplitVTilingDataOneRowPureCopy),
                MaxSizeT(
                    sizeof(SplitVTilingDataSameLen),
                    MaxSizeT(sizeof(SplitVTilingDataSameLenCompact),
                             MaxSizeT(sizeof(SplitVTilingDataSameLenPureCopy8Bit),
                                      MaxSizeT(sizeof(SplitVTilingDataSameLenInnerCopy),
                                               MaxSizeT(sizeof(SplitVTilingDataUnevenInnerAlignedMid),
                                                        MaxSizeT(sizeof(SplitVTilingDataUnevenCompact),
                                                                 sizeof(SplitVTilingDataUnevenPureCopy16Bit)))))))))) +
    10;

struct RuntimeInfo {
    uint64_t totalLength = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint32_t dataTypeSize = 0;
    uint32_t blockSize = 0;
    uint32_t alignedNum = 0;
    uint32_t alignedGm = 0;
    uint32_t coreNum = 1;
    uint64_t ubNum = 0;
    uint64_t effectiveUbNum = 0;
    uint32_t maxTileLength = 0;
    uint32_t maxUbTileLength = 0;
    int64_t splitNum = 0;
    int64_t splitDim = 0;
    uint64_t outerLength = 1;
    uint64_t midLength = 0;
    uint64_t innerLength = 1;
    std::vector<int64_t> sizeSplits;
};

struct CoreSplitPlan {
    uint64_t needCore = 1;
    uint64_t formerNum = 0;
    uint64_t formerTaskNum = 0;
    uint64_t tailTaskNum = 0;
};

static CoreSplitPlan CalcCoreSplitPlanByCore(uint64_t totalTaskNum, uint64_t needCore)
{
    CoreSplitPlan plan;
    plan.needCore = std::max<uint64_t>(needCore, 1);
    plan.formerNum = totalTaskNum % plan.needCore;
    plan.tailTaskNum = totalTaskNum / plan.needCore;
    plan.formerTaskNum = plan.tailTaskNum + (plan.formerNum > 0 ? 1 : 0);
    return plan;
}

static CoreSplitPlan CalcCoreSplitPlan(uint64_t totalTaskNum, uint64_t coreNum)
{
    return CalcCoreSplitPlanByCore(totalTaskNum, std::min<uint64_t>(totalTaskNum, coreNum));
}

static ge::graphStatus SetWorkspaceZero(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static void FillSplitVTilingBasic(const RuntimeInfo& runtimeInfo, SplitVTilingData* tiling)
{
    tiling->totalLength = runtimeInfo.totalLength;
    tiling->outerLength = runtimeInfo.outerLength;
    tiling->midLength = runtimeInfo.midLength;
    tiling->innerLength = runtimeInfo.innerLength;
    tiling->splitDim = runtimeInfo.splitDim;
    tiling->splitNum = runtimeInfo.splitNum;
    for (int64_t i = 0; i < runtimeInfo.splitNum; ++i) {
        tiling->sizeSplits[i] = runtimeInfo.sizeSplits[i];
    }
    for (int64_t i = runtimeInfo.splitNum; i < maxSplitNum; ++i) {
        tiling->sizeSplits[i] = 0;
    }
}

static void FillSplitVTilingLoop(const CoreSplitPlan& corePlan, uint32_t tileLength, uint32_t tileNum,
                                 uint32_t lastTileLength, SplitVTilingData* tiling)
{
    tiling->formerLoop = corePlan.formerTaskNum;
    tiling->tailLoop = corePlan.tailTaskNum;
    tiling->formerNum = corePlan.formerNum;
    tiling->innerTileLength = tileLength;
    tiling->innerTileNum = tileNum;
    tiling->innerLastTileLength = lastTileLength;
}

template <typename TilingData>
static void FillTaskSplitFields(TilingData* tiling, uint64_t totalTaskNum, const CoreSplitPlan& corePlan)
{
    tiling->totalTaskNum = totalTaskNum;
    tiling->formerTaskNum = corePlan.formerTaskNum;
    tiling->tailTaskNum = corePlan.tailTaskNum;
    tiling->formerNum = corePlan.formerNum;
}

static uint64_t AlignUpTo(uint64_t value, uint64_t align);
static uint64_t CeilDiv(uint64_t value, uint64_t divisor);

struct OuterTilePlan {
    uint32_t outerTileNum = 0;
    uint32_t outerTail = 0;
};

struct OuterTaskSplitPlan {
    uint32_t needCore = 1;
    uint32_t formerOuterTileNum = 0;
    uint32_t tailOuterTileNum = 0;
    uint32_t formerNum = 0;
};

struct CoreRowsPlan {
    uint32_t realCoreNum = 0;
    uint32_t formerCoreRows = 0;
    uint32_t tailCoreRows = 0;
    uint32_t formerNum = 0;
    uint32_t maxCoreRows = 0;
};

struct UnevenPureCopyTilePlan {
    uint32_t mode = 0;
    uint32_t outerTile = 0;
    uint32_t colTileLength = 0;
    uint32_t colTilePitch = 0;
};

static bool CalcOuterTilePlan(uint64_t outerLength, uint32_t outerTile, OuterTilePlan& plan)
{
    if (outerLength == 0 || outerTile == 0) {
        return false;
    }
    const uint64_t outerTileNum64 = CeilDiv(outerLength, outerTile);
    if (outerTileNum64 == 0 || outerTileNum64 > UINT32_MAX) {
        return false;
    }
    plan.outerTileNum = static_cast<uint32_t>(outerTileNum64);
    plan.outerTail = static_cast<uint32_t>(outerLength - static_cast<uint64_t>(plan.outerTileNum - 1) * outerTile);
    return plan.outerTail != 0;
}

static bool CalcOuterTaskSplitPlan(uint64_t logicalTaskNum, uint32_t coreNum, OuterTaskSplitPlan& plan)
{
    if (logicalTaskNum == 0 || logicalTaskNum > UINT32_MAX) {
        return false;
    }
    const CoreSplitPlan corePlan = CalcCoreSplitPlan(logicalTaskNum, coreNum);
    if (corePlan.needCore > UINT32_MAX || corePlan.formerTaskNum > UINT32_MAX || corePlan.tailTaskNum > UINT32_MAX ||
        corePlan.formerNum > UINT32_MAX) {
        return false;
    }
    plan.needCore = static_cast<uint32_t>(corePlan.needCore);
    plan.formerOuterTileNum = static_cast<uint32_t>(corePlan.formerTaskNum);
    plan.tailOuterTileNum = static_cast<uint32_t>(corePlan.tailTaskNum);
    plan.formerNum = static_cast<uint32_t>(corePlan.formerNum);
    return true;
}

static bool CalcOuterTileTaskSplitPlan(uint64_t outerLength, uint32_t outerTile, uint32_t taskFactor, uint32_t coreNum,
                                       OuterTilePlan& outerPlan, OuterTaskSplitPlan& taskPlan)
{
    if (taskFactor == 0 || !CalcOuterTilePlan(outerLength, outerTile, outerPlan)) {
        return false;
    }
    if (outerPlan.outerTileNum > UINT32_MAX / taskFactor) {
        return false;
    }
    return CalcOuterTaskSplitPlan(static_cast<uint64_t>(outerPlan.outerTileNum) * taskFactor, coreNum, taskPlan);
}

static bool CalcCoreRowsPlan(uint64_t outerLength, uint32_t coreNum, CoreRowsPlan& plan)
{
    if (outerLength == 0) {
        return false;
    }
    plan.realCoreNum = static_cast<uint32_t>(std::min<uint64_t>(coreNum, outerLength));
    if (plan.realCoreNum == 0) {
        return false;
    }
    plan.tailCoreRows = static_cast<uint32_t>(outerLength / plan.realCoreNum);
    plan.formerNum = static_cast<uint32_t>(outerLength % plan.realCoreNum);
    plan.formerCoreRows = plan.tailCoreRows + (plan.formerNum == 0 ? 0 : 1);
    plan.maxCoreRows = plan.formerNum == 0 ? plan.tailCoreRows : plan.formerCoreRows;
    return true;
}

template <typename TilingData>
static void FillOuterTaskFields(TilingData* tiling, uint32_t outerTile, const OuterTilePlan& outerPlan,
                                const OuterTaskSplitPlan& taskPlan)
{
    tiling->outerTile = outerTile;
    tiling->outerTail = outerPlan.outerTail;
    tiling->outerTileNum = outerPlan.outerTileNum;
    tiling->formerOuterTileNum = taskPlan.formerOuterTileNum;
    tiling->tailOuterTileNum = taskPlan.tailOuterTileNum;
    tiling->formerNum = taskPlan.formerNum;
}

template <typename TilingData>
static void FillUnevenSplitLists(TilingData* tiling, int64_t splitNum, const uint32_t* sizeSplits)
{
    uint32_t splitStart = 0;
    for (int64_t i = 0; i < splitNum; ++i) {
        tiling->sizeSplits[i] = sizeSplits[i];
        tiling->splitStarts[i] = splitStart;
        splitStart += sizeSplits[i];
    }
    for (int64_t i = splitNum; i < maxSplitNum; ++i) {
        tiling->sizeSplits[i] = 0;
        tiling->splitStarts[i] = 0;
    }
}

template <typename TilingData>
static void FillUnevenSplitListsFromRuntime(TilingData* tiling, const RuntimeInfo& runtimeInfo)
{
    uint32_t sizeSplits[maxSplitNum] = {};
    for (int64_t i = 0; i < runtimeInfo.splitNum; ++i) {
        sizeSplits[i] = static_cast<uint32_t>(runtimeInfo.sizeSplits[i]);
    }
    FillUnevenSplitLists(tiling, runtimeInfo.splitNum, sizeSplits);
}

struct UnevenSplitValidationResult {
    uint64_t splitSum = 0;
    uint32_t maxSplitSize = 0;
    uint32_t minSplitSize = UINT32_MAX;
};

// Validates sizeSplits entries and computes splitSum / maxSplitSize (and optionally minSplitSize).
// Returns false when any split is out of range, bytes exceed the block limit, or the sum/max check fails.
static bool ValidateUnevenSplits(const RuntimeInfo& runtimeInfo, UnevenSplitValidationResult& result,
                                 bool trackMin = false)
{
    for (int64_t i = 0; i < runtimeInfo.splitNum; ++i) {
        const int64_t split = runtimeInfo.sizeSplits[i];
        if (split <= 0 || static_cast<uint64_t>(split) > UINT32_MAX) {
            return false;
        }
        const uint64_t splitBytes = static_cast<uint64_t>(split) * runtimeInfo.dataTypeSize;
        if (splitBytes == 0 || splitBytes > MAX_DATA_COPY_EXT_BLOCK_LEN) {
            return false;
        }
        const uint32_t splitU32 = static_cast<uint32_t>(split);
        result.splitSum += static_cast<uint64_t>(split);
        result.maxSplitSize = std::max<uint32_t>(result.maxSplitSize, splitU32);
        if (trackMin) {
            result.minSplitSize = std::min<uint32_t>(result.minSplitSize, splitU32);
        }
    }
    if (result.splitSum != runtimeInfo.midLength || result.maxSplitSize == 0) {
        return false;
    }
    if (trackMin && result.minSplitSize == 0) {
        return false;
    }
    return true;
}

template <typename TilingData>
static void FillUnevenCompactRuntimeFields(TilingData* tiling, uint64_t totalLength, uint64_t outerLength,
                                           uint32_t rowLength, int64_t splitNum, uint32_t maxSplitSize)
{
    tiling->totalLength = totalLength;
    tiling->outerLength = outerLength;
    tiling->rowLength = rowLength;
    tiling->splitNum = static_cast<uint32_t>(splitNum);
    tiling->maxSplitSize = maxSplitSize;
}

static void FillUnevenCompactModeFields(SplitVTilingDataUnevenCompact* tiling, uint32_t mode, uint32_t rowTransLen,
                                        uint32_t splitTransLen, uint32_t virtualSplitSize, uint32_t virtualSplitNum,
                                        uint32_t colChunkSize, uint32_t colChunkNum)
{
    tiling->mode = mode;
    tiling->rowTransLen = rowTransLen;
    tiling->splitTransLen = splitTransLen;
    tiling->virtualSplitSize = virtualSplitSize;
    tiling->virtualSplitNum = virtualSplitNum;
    tiling->colChunkSize = colChunkSize;
    tiling->colChunkNum = colChunkNum;
}

// Bundles the common runtime-fill sequence shared by uneven-compact tiling paths:
// FillUnevenCompactRuntimeFields + FillUnevenSplitListsFromRuntime + FillOuterTaskFields.
static void FillUnevenCompactTilingBase(SplitVTilingDataUnevenCompact* tiling, const RuntimeInfo& runtimeInfo,
                                        uint32_t outerTile, const OuterTilePlan& outerPlan,
                                        const OuterTaskSplitPlan& taskPlan, uint32_t maxSplitSize)
{
    FillUnevenCompactRuntimeFields(tiling, runtimeInfo.totalLength, runtimeInfo.outerLength,
                                   static_cast<uint32_t>(runtimeInfo.midLength), runtimeInfo.splitNum, maxSplitSize);
    FillUnevenSplitListsFromRuntime(tiling, runtimeInfo);
    FillOuterTaskFields(tiling, outerTile, outerPlan, taskPlan);
}

template <typename TilingData>
static void FillSameLenCompactRuntimeFields(TilingData* tiling, uint64_t totalLength, uint64_t outerLength,
                                            uint32_t rowLength, uint32_t splitSize, uint32_t tailSplitSize,
                                            int64_t splitNum)
{
    tiling->totalLength = totalLength;
    tiling->outerLength = outerLength;
    tiling->rowLength = rowLength;
    tiling->splitSize = splitSize;
    tiling->tailSplitSize = tailSplitSize;
    tiling->splitNum = static_cast<uint32_t>(splitNum);
}

static void FillSameLenCompactFields(SplitVTilingDataSameLenCompact* tiling, uint64_t totalLength, uint64_t outerLength,
                                     uint32_t rowLength, uint32_t splitSize, uint32_t tailSplitSize, int64_t splitNum,
                                     uint32_t outerTile, const OuterTilePlan& outerPlan,
                                     const OuterTaskSplitPlan& taskPlan, uint32_t rowTransLen, uint32_t splitTransLen,
                                     uint32_t chunkSplitNum, uint32_t colChunkNum)
{
    FillSameLenCompactRuntimeFields(tiling, totalLength, outerLength, rowLength, splitSize, tailSplitSize, splitNum);
    FillOuterTaskFields(tiling, outerTile, outerPlan, taskPlan);
    tiling->rowTransLen = rowTransLen;
    tiling->splitTransLen = splitTransLen;
    tiling->chunkSplitNum = chunkSplitNum;
    tiling->colChunkNum = colChunkNum;
}

static void FillSameLenFields(SplitVTilingDataSameLen* tiling, uint64_t totalLength, uint64_t outerLength,
                              uint64_t midLength, uint64_t innerLength, uint32_t splitSize, uint32_t tailSplitSize,
                              int64_t splitNum, uint32_t outerTile, const OuterTilePlan& outerPlan,
                              const OuterTaskSplitPlan& taskPlan, uint32_t splitTileLength)
{
    tiling->totalLength = totalLength;
    tiling->outerLength = outerLength;
    tiling->midLength = midLength;
    tiling->innerLength = innerLength;
    tiling->splitSize = splitSize;
    tiling->tailSplitSize = tailSplitSize;
    tiling->splitNum = static_cast<uint32_t>(splitNum);
    tiling->formerLoop = taskPlan.formerOuterTileNum;
    tiling->tailLoop = taskPlan.tailOuterTileNum;
    FillOuterTaskFields(tiling, outerTile, outerPlan, taskPlan);
    tiling->splitTileLength = splitTileLength;
}

template <typename TilingData>
static void FillCoreRowsFields(TilingData* tiling, const CoreRowsPlan& coreRows)
{
    tiling->realCoreNum = coreRows.realCoreNum;
    tiling->formerCoreRows = coreRows.formerCoreRows;
    tiling->tailCoreRows = coreRows.tailCoreRows;
    tiling->formerNum = coreRows.formerNum;
}

template <typename TilingData>
static void FillPureCopyTileFields(TilingData* tiling, uint32_t mode, uint32_t outerTile, uint32_t colTileLength,
                                   uint32_t colTilePitch)
{
    tiling->mode = mode;
    tiling->outerTile = outerTile;
    tiling->colTileLength = colTileLength;
    tiling->colTilePitch = colTilePitch;
}

template <typename TilingData>
static void FillPureCopyTileFields(TilingData* tiling, const UnevenPureCopyTilePlan& tilePlan)
{
    FillPureCopyTileFields(tiling, tilePlan.mode, tilePlan.outerTile, tilePlan.colTileLength, tilePlan.colTilePitch);
}

static ge::graphStatus CalcUnevenPureCopyTilePlan(const RuntimeInfo& runtimeInfo, uint32_t dataTypeSize,
                                                  uint32_t rowLength, uint32_t maxSplitSize, uint32_t minSplitSize,
                                                  uint64_t maxSplitPitchBytes, uint64_t splitPitch64,
                                                  const CoreRowsPlan& coreRows, bool checkSplitMajorCopyLen,
                                                  uint32_t splitMajorMode, uint32_t rowLengthChunkMode,
                                                  UnevenPureCopyTilePlan& plan)
{
    if (dataTypeSize == 0) {
        return ge::GRAPH_FAILED;
    }
    plan.mode = splitMajorMode;
    plan.outerTile = 0;
    plan.colTileLength = maxSplitSize;
    plan.colTilePitch = static_cast<uint32_t>(splitPitch64);

    const uint64_t maxRowsByUb = runtimeInfo.effectiveUbNum / (BUFFER_NUM * maxSplitPitchBytes);
    const uint64_t maxSrcStrideBytes = static_cast<uint64_t>(rowLength - minSplitSize) * dataTypeSize;
    const uint64_t maxSplitBytes = static_cast<uint64_t>(maxSplitSize) * dataTypeSize;
    const bool splitMajorCopyOk = !checkSplitMajorCopyLen || maxSplitBytes <= MAX_DATA_COPY_EXT_BLOCK_LEN;
    const bool enableSplitMajor = maxRowsByUb != 0 && splitMajorCopyOk && maxSrcStrideBytes <= UINT32_MAX &&
                                  coreRows.maxCoreRows != 0;

    if (enableSplitMajor) {
        uint64_t outerTile64 = std::min<uint64_t>(maxRowsByUb, coreRows.maxCoreRows);
        outerTile64 = std::min<uint64_t>(outerTile64, MAX_DATA_COPY_BLOCK_COUNT);
        plan.outerTile = static_cast<uint32_t>(outerTile64);
        return plan.outerTile == 0 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
    }

    plan.mode = rowLengthChunkMode;
    const uint64_t maxTileBytesByUb = runtimeInfo.effectiveUbNum / BUFFER_NUM;
    uint64_t colTileBytes = std::min<uint64_t>(maxTileBytesByUb, MAX_DATA_COPY_EXT_BLOCK_LEN);
    colTileBytes = (colTileBytes / runtimeInfo.blockSize) * runtimeInfo.blockSize;
    if (colTileBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    plan.colTileLength = static_cast<uint32_t>(std::min<uint64_t>(maxSplitSize, colTileBytes / dataTypeSize));
    if (plan.colTileLength == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t colTilePitchBytes = AlignUpTo(static_cast<uint64_t>(plan.colTileLength) * dataTypeSize,
                                                 runtimeInfo.blockSize);
    const uint64_t colTilePitchElems = colTilePitchBytes / dataTypeSize;
    if (colTilePitchElems > UINT32_MAX || colTilePitchBytes > maxTileBytesByUb ||
        colTilePitchElems - plan.colTileLength > UINT8_MAX) {
        return ge::GRAPH_FAILED;
    }
    plan.colTilePitch = static_cast<uint32_t>(colTilePitchElems);
    plan.outerTile = 1;
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetMaxDataCopyParamBlockLenBytes(const RuntimeInfo& runtimeInfo)
{
    return static_cast<uint64_t>(MAX_COPY_BYTES) * runtimeInfo.blockSize;
}

static ge::graphStatus GetRuntimeInfo(gert::TilingContext* context, RuntimeInfo& runtimeInfo)
{
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    const auto* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const auto inputShape = xShape->GetStorageShape();
    const int64_t xDim = inputShape.GetDimNum();
    OP_CHECK_IF(xDim <= 0, OP_LOGE(context, "x dim num must be greater than 0"), return ge::GRAPH_FAILED);

    const auto* sizeSplitTensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeSplitTensor);
    const auto* splitDimTensor = context->GetInputTensor(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, splitDimTensor);
    const auto* splitDimData = splitDimTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, splitDimData);

    int64_t splitDim = splitDimData[0];
    if (splitDim < 0) {
        splitDim += xDim;
    }
    OP_CHECK_IF(splitDim < 0 || splitDim >= xDim,
                OP_LOGE(context, "splitDim out of range, splitDim=%ld, xDim=%ld", splitDim, xDim),
                return ge::GRAPH_FAILED);

    const auto* sizeSplitShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeSplitShape);
    const int64_t splitNum = sizeSplitShape->GetStorageShape().GetDim(0);
    OP_CHECK_IF(splitNum <= 0 || splitNum > maxSplitNum,
                OP_LOGE(context, "splitNum invalid, splitNum=%ld, maxSplitNum=%d", splitNum, maxSplitNum),
                return ge::GRAPH_FAILED);

    uint64_t outerLength = 1;
    uint64_t innerLength = 1;
    for (int64_t i = 0; i < splitDim; ++i) {
        OP_CHECK_IF(inputShape.GetDim(i) < 0, OP_LOGE(context, "dynamic shape is not supported in tiling"),
                    return ge::GRAPH_FAILED);
        outerLength *= static_cast<uint64_t>(inputShape.GetDim(i));
    }
    for (int64_t i = splitDim + 1; i < xDim; ++i) {
        OP_CHECK_IF(inputShape.GetDim(i) < 0, OP_LOGE(context, "dynamic shape is not supported in tiling"),
                    return ge::GRAPH_FAILED);
        innerLength *= static_cast<uint64_t>(inputShape.GetDim(i));
    }

    const int64_t splitDimLength = inputShape.GetDim(splitDim);
    OP_CHECK_IF(splitDimLength < 0, OP_LOGE(context, "dynamic split dim is not supported in tiling"),
                return ge::GRAPH_FAILED);

    int32_t negativeOneIdx = -1;
    uint64_t midLength = 0;
    std::vector<int64_t> sizeSplits(splitNum);
    if (sizeSplitTensor->GetDataType() == ge::DT_INT64) {
        const auto* sizeSplitsData = sizeSplitTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, sizeSplitsData);
        for (int64_t i = 0; i < splitNum; ++i) {
            sizeSplits[i] = sizeSplitsData[i];
        }
    } else if (sizeSplitTensor->GetDataType() == ge::DT_INT32) {
        const auto* sizeSplitsData = sizeSplitTensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, sizeSplitsData);
        for (int64_t i = 0; i < splitNum; ++i) {
            sizeSplits[i] = static_cast<int64_t>(sizeSplitsData[i]);
        }
    } else {
        OP_LOGE(context, "size_splits only supports int32/int64");
        return ge::GRAPH_FAILED;
    }

    for (int64_t i = 0; i < splitNum; ++i) {
        if (sizeSplits[i] == -1) {
            OP_CHECK_IF(negativeOneIdx != -1, OP_LOGE(context, "size_splits has more than one -1"),
                        return ge::GRAPH_FAILED);
            negativeOneIdx = static_cast<int32_t>(i);
            continue;
        }
        OP_CHECK_IF(sizeSplits[i] < 0, OP_LOGE(context, "size_splits[%ld] must be >= -1", i), return ge::GRAPH_FAILED);
        midLength += static_cast<uint64_t>(sizeSplits[i]);
    }

    if (negativeOneIdx != -1) {
        OP_CHECK_IF(midLength > static_cast<uint64_t>(splitDimLength),
                    OP_LOGE(context, "sum(size_splits except -1) exceeds split dim length"), return ge::GRAPH_FAILED);
        sizeSplits[negativeOneIdx] = splitDimLength - static_cast<int64_t>(midLength);
        midLength = static_cast<uint64_t>(splitDimLength);
    } else {
        OP_CHECK_IF(midLength != static_cast<uint64_t>(splitDimLength),
                    OP_LOGE(context, "sum(size_splits) must equal split dim length"), return ge::GRAPH_FAILED);
    }

    const auto* xTensor = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    ge::DataType dataType = xTensor->GetDataType();
    uint32_t dataTypeSize = ge::GetSizeByDataType(dataType);
    if (dataTypeSize == 0) {
        OP_LOGE(context, "unsupported input dtype");
        return ge::GRAPH_FAILED;
    }

    uint32_t blockSize = Ops::Base::GetUbBlockSize(context);
    if (blockSize == 0) {
        OP_LOGE(context, "GetUbBlockSize returned 0");
        return ge::GRAPH_FAILED;
    }

    uint32_t alignedNum = std::max<uint32_t>(1, blockSize / dataTypeSize);
    uint32_t alignedGm = std::max<uint32_t>(1, GM_ALIGN_SIZE / dataTypeSize);
    uint32_t coreNum = std::max<uint32_t>(1, ascendPlatform.GetCoreNumAiv());
    uint64_t ubNum = 0;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubNum);
    uint64_t effectiveUbNum = ubNum > UB_RESERVED_BYTES ? ubNum - UB_RESERVED_BYTES : 0;

    uint32_t maxTileLength = 0;
    if (effectiveUbNum > 0) {
        maxTileLength = static_cast<uint32_t>(effectiveUbNum / (QUEUE_NUM * BUFFER_NUM * dataTypeSize));
    }
    maxTileLength = std::max<uint32_t>(alignedNum, (maxTileLength / alignedNum) * alignedNum);
    const uint32_t maxUbTileLength = maxTileLength;
    uint32_t maxCopyNum = std::max<uint32_t>(alignedNum, MAX_COPY_BYTES / dataTypeSize);
    maxCopyNum = std::max<uint32_t>(alignedNum, (maxCopyNum / alignedNum) * alignedNum);
    maxTileLength = std::min(maxTileLength, maxCopyNum);

    runtimeInfo.totalLength = outerLength * midLength * innerLength;
    runtimeInfo.dataType = dataType;
    runtimeInfo.dataTypeSize = dataTypeSize;
    runtimeInfo.blockSize = blockSize;
    runtimeInfo.alignedNum = alignedNum;
    runtimeInfo.alignedGm = alignedGm;
    runtimeInfo.coreNum = coreNum;
    runtimeInfo.ubNum = ubNum;
    runtimeInfo.effectiveUbNum = effectiveUbNum;
    runtimeInfo.maxTileLength = maxTileLength;
    runtimeInfo.maxUbTileLength = maxUbTileLength;
    runtimeInfo.splitNum = splitNum;
    runtimeInfo.splitDim = splitDim;
    runtimeInfo.outerLength = outerLength;
    runtimeInfo.midLength = midLength;
    runtimeInfo.innerLength = innerLength;
    runtimeInfo.sizeSplits = std::move(sizeSplits);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForPureCopy(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    uint32_t tileLength = runtimeInfo.maxTileLength;
    if (tileLength == 0 || runtimeInfo.alignedGm == 0 || runtimeInfo.alignedNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t needCoreNum = (runtimeInfo.totalLength + tileLength - 1) / tileLength;
    needCoreNum = std::min(needCoreNum, runtimeInfo.coreNum);
    if (needCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t formerLength = runtimeInfo.totalLength / needCoreNum;
    formerLength = (formerLength + runtimeInfo.alignedGm - 1) / runtimeInfo.alignedGm * runtimeInfo.alignedGm;
    needCoreNum = (runtimeInfo.totalLength + formerLength - 1) / formerLength;
    uint64_t tailLength = runtimeInfo.totalLength - (needCoreNum - 1) * formerLength;

    if (formerLength < tileLength) {
        tileLength = (formerLength + runtimeInfo.alignedNum - 1) / runtimeInfo.alignedNum * runtimeInfo.alignedNum;
    }

    uint32_t formerTileNum = (formerLength + tileLength - 1) / tileLength;
    uint32_t formerLastTileLength = formerLength - (formerTileNum - 1) * tileLength;
    uint32_t tailTileNum = (tailLength + tileLength - 1) / tileLength;
    uint32_t tailLastTileLength = tailLength - (tailTileNum - 1) * tileLength;

    SplitVTilingDataPureCopy* tiling = context->GetTilingData<SplitVTilingDataPureCopy>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalLength = runtimeInfo.totalLength;
    tiling->formerLength = formerLength;
    tiling->tailLength = tailLength;
    tiling->tileLength = tileLength;
    tiling->formerTileNum = formerTileNum;
    tiling->formerLastTileLength = formerLastTileLength;
    tiling->tailTileNum = tailTileNum;
    tiling->tailLastTileLength = tailLastTileLength;

    context->SetBlockDim(needCoreNum);
    context->SetTilingKey(TILING_KEY_SPLIT_V_PURE_COPY);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sizeof(SplitVTilingDataPureCopy);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForOneRowPureCopy(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    if (runtimeInfo.outerLength != 1 || runtimeInfo.innerLength != 1 || runtimeInfo.splitNum <= 1 ||
        runtimeInfo.splitNum > maxSplitNum || runtimeInfo.midLength == 0 || runtimeInfo.midLength > UINT32_MAX ||
        runtimeInfo.dataTypeSize == 0 || runtimeInfo.maxTileLength == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t chunkLength64 = std::min<uint64_t>(runtimeInfo.maxTileLength,
                                                MAX_DATA_COPY_EXT_BLOCK_LEN / runtimeInfo.dataTypeSize);
    chunkLength64 = std::min<uint64_t>(chunkLength64, UINT32_MAX);
    if (chunkLength64 == 0) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.alignedNum > 1 && chunkLength64 > runtimeInfo.alignedNum) {
        chunkLength64 = (chunkLength64 / runtimeInfo.alignedNum) * runtimeInfo.alignedNum;
    }
    if (chunkLength64 == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t sizeSplits[maxSplitNum] = {};
    uint64_t splitSum = 0;
    uint64_t totalTaskNum = 0;
    const uint32_t chunkLength = static_cast<uint32_t>(chunkLength64);
    for (int64_t i = 0; i < runtimeInfo.splitNum; ++i) {
        const int64_t split = runtimeInfo.sizeSplits[i];
        if (split <= 0 || static_cast<uint64_t>(split) > UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }
        sizeSplits[i] = static_cast<uint32_t>(split);
        splitSum += static_cast<uint64_t>(split);
        totalTaskNum += (static_cast<uint64_t>(split) + chunkLength - 1) / chunkLength;
    }
    if (splitSum != runtimeInfo.midLength || totalTaskNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const auto corePlan = CalcCoreSplitPlan(totalTaskNum, runtimeInfo.coreNum);

    auto* tiling = context->GetTilingData<SplitVTilingDataOneRowPureCopy>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalLength = runtimeInfo.totalLength;
    tiling->splitNum = static_cast<uint32_t>(runtimeInfo.splitNum);
    FillTaskSplitFields(tiling, totalTaskNum, corePlan);
    tiling->chunkLength = chunkLength;
    for (int64_t i = 0; i < runtimeInfo.splitNum; ++i) {
        tiling->sizeSplits[i] = sizeSplits[i];
    }

    context->SetBlockDim(static_cast<uint32_t>(corePlan.needCore));
    context->SetTilingKey(TILING_KEY_SPLIT_V_ONE_ROW_PURE_COPY);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForOneOuter(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    uint32_t tileLength = std::min<uint64_t>(runtimeInfo.maxTileLength, runtimeInfo.innerLength);
    uint32_t tileNum = (runtimeInfo.innerLength + tileLength - 1) / tileLength;
    uint32_t lastTileLength = runtimeInfo.innerLength - (tileNum - 1) * tileLength;

    uint64_t loopNum = runtimeInfo.midLength * tileNum;
    const auto corePlan = CalcCoreSplitPlan(loopNum, runtimeInfo.coreNum);

    SplitVTilingData* tiling = context->GetTilingData<SplitVTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillSplitVTilingBasic(runtimeInfo, tiling);
    FillSplitVTilingLoop(corePlan, tileLength, tileNum, lastTileLength, tiling);

    context->SetBlockDim(corePlan.needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_ONE_OUTER);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForGeneral(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    uint32_t tileLength = std::min<uint64_t>(runtimeInfo.maxTileLength, runtimeInfo.innerLength);
    uint32_t tileNum = (runtimeInfo.innerLength + tileLength - 1) / tileLength;
    uint32_t lastTileLength = runtimeInfo.innerLength - (tileNum - 1) * tileLength;

    uint64_t loopNum = runtimeInfo.outerLength * runtimeInfo.midLength * tileNum;
    const auto corePlan = CalcCoreSplitPlan(loopNum, runtimeInfo.coreNum);

    SplitVTilingData* tiling = context->GetTilingData<SplitVTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillSplitVTilingBasic(runtimeInfo, tiling);
    FillSplitVTilingLoop(corePlan, tileLength, tileNum, lastTileLength, tiling);

    context->SetBlockDim(corePlan.needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_GENERAL);
    return SetWorkspaceZero(context);
}

struct SameLenSplitInfo {
    uint64_t splitSize = 0;
    uint64_t tailSplitSize = 0;
    uint64_t maxSplitSize = 0;
};

static bool GetSameLenSplitInfo(const RuntimeInfo& runtimeInfo, SameLenSplitInfo& info)
{
    if (runtimeInfo.splitNum <= 1 || runtimeInfo.splitNum > maxSplitNum || runtimeInfo.sizeSplits.empty()) {
        return false;
    }
    const int64_t first = runtimeInfo.sizeSplits[0];
    if (first <= 0) {
        return false;
    }
    uint64_t sum = 0;
    for (int64_t i = 0; i + 1 < runtimeInfo.splitNum; ++i) {
        if (runtimeInfo.sizeSplits[i] != first) {
            return false;
        }
        sum += static_cast<uint64_t>(runtimeInfo.sizeSplits[i]);
    }
    const int64_t tail = runtimeInfo.sizeSplits[runtimeInfo.splitNum - 1];
    if (tail <= 0) {
        return false;
    }
    sum += static_cast<uint64_t>(tail);
    if (sum != runtimeInfo.midLength) {
        return false;
    }
    info.splitSize = static_cast<uint64_t>(first);
    info.tailSplitSize = static_cast<uint64_t>(tail);
    info.maxSplitSize = std::max(info.splitSize, info.tailSplitSize);
    return true;
}

static bool IsSplitSameLenWithTail(const RuntimeInfo& runtimeInfo)
{
    SameLenSplitInfo info;
    return GetSameLenSplitInfo(runtimeInfo, info);
}

static uint64_t AlignUpTo(uint64_t value, uint64_t align)
{
    if (align == 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}

static uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

static uint64_t Gcd64(uint64_t lhs, uint64_t rhs)
{
    while (rhs != 0) {
        const uint64_t tmp = lhs % rhs;
        lhs = rhs;
        rhs = tmp;
    }
    return lhs;
}

static bool IsSameLenCompactDataType(ge::DataType dataType)
{
    return dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16 || dataType == ge::DT_INT16 ||
           dataType == ge::DT_UINT16;
}

static bool IsCompact32BitDataType(ge::DataType dataType)
{
    return dataType == ge::DT_FLOAT || dataType == ge::DT_INT32 || dataType == ge::DT_UINT32 ||
           dataType == ge::DT_INT64 || dataType == ge::DT_UINT64;
}

static uint32_t GetCompact32BitViewFactor(const RuntimeInfo& runtimeInfo)
{
    if (!IsCompact32BitDataType(runtimeInfo.dataType) ||
        (runtimeInfo.dataTypeSize != sizeof(uint32_t) && runtimeInfo.dataTypeSize != sizeof(uint64_t))) {
        return 0;
    }
    return runtimeInfo.dataTypeSize / sizeof(uint16_t);
}

static bool IsCompact8DataType(ge::DataType dataType) { return dataType == ge::DT_INT8 || dataType == ge::DT_UINT8; }

static bool IsSameLenCompact8DataType(ge::DataType dataType)
{
    return IsCompact8DataType(dataType) || dataType == ge::DT_BOOL;
}

struct InnerSplitChunkPlan {
    uint32_t outerTile = 0;
    uint32_t chunkElems = 0;
    uint32_t chunkElemsAligned = 0;
    uint32_t chunkNumMax = 0;
};

static bool CalcInnerSplitChunkPlan(const RuntimeInfo& runtimeInfo, uint64_t maxSplitSize, uint64_t stageBudget,
                                    InnerSplitChunkPlan& plan)
{
    if (runtimeInfo.dataTypeSize == 0 || runtimeInfo.innerLength == 0 || maxSplitSize == 0 ||
        stageBudget < runtimeInfo.blockSize || maxSplitSize > UINT64_MAX / runtimeInfo.innerLength ||
        runtimeInfo.midLength > UINT64_MAX / runtimeInfo.innerLength) {
        return false;
    }

    const uint64_t maxSplitElems = maxSplitSize * runtimeInfo.innerLength;
    const uint64_t rowElems = runtimeInfo.midLength * runtimeInfo.innerLength;
    if (maxSplitElems == 0 || rowElems < maxSplitElems || rowElems > UINT32_MAX / runtimeInfo.dataTypeSize ||
        maxSplitElems > UINT32_MAX / runtimeInfo.dataTypeSize) {
        return false;
    }

    uint64_t candidate = maxSplitElems;
    candidate = std::min<uint64_t>(candidate, MAX_DATA_COPY_EXT_BLOCK_LEN / runtimeInfo.dataTypeSize);
    candidate = std::min<uint64_t>(candidate, runtimeInfo.maxUbTileLength);
    candidate = std::min<uint64_t>(candidate, stageBudget / runtimeInfo.dataTypeSize);
    if (candidate >= runtimeInfo.alignedNum && runtimeInfo.alignedNum != 0) {
        candidate = (candidate / runtimeInfo.alignedNum) * runtimeInfo.alignedNum;
    }
    if (candidate == 0) {
        return false;
    }

    const uint64_t step = runtimeInfo.alignedNum == 0 ? 1 : runtimeInfo.alignedNum;
    while (candidate > 0) {
        const uint64_t chunkBytes = candidate * runtimeInfo.dataTypeSize;
        const uint64_t chunkPitch = AlignUpTo(chunkBytes, runtimeInfo.blockSize);
        const uint64_t chunkPitchElems = chunkPitch / runtimeInfo.dataTypeSize;
        if (chunkBytes <= MAX_DATA_COPY_EXT_BLOCK_LEN && chunkPitch != 0 && chunkPitch <= stageBudget &&
            chunkPitchElems != 0 && chunkPitchElems <= runtimeInfo.maxUbTileLength) {
            uint64_t outerTile64 = CeilDiv(runtimeInfo.outerLength, runtimeInfo.coreNum);
            outerTile64 = std::max<uint64_t>(outerTile64, 1);
            outerTile64 = std::min<uint64_t>(outerTile64, runtimeInfo.outerLength);
            outerTile64 = std::min<uint64_t>(outerTile64, MAX_DATA_COPY_BLOCK_COUNT);
            outerTile64 = std::min<uint64_t>(outerTile64, stageBudget / chunkPitch);
            outerTile64 = std::min<uint64_t>(outerTile64, runtimeInfo.maxUbTileLength / chunkPitchElems);
            if (outerTile64 != 0 && outerTile64 <= UINT32_MAX) {
                const uint64_t chunkNumMax = CeilDiv(maxSplitElems, candidate);
                if (chunkNumMax != 0 && chunkNumMax <= UINT32_MAX && candidate <= UINT32_MAX &&
                    chunkPitchElems <= UINT32_MAX) {
                    plan.outerTile = static_cast<uint32_t>(outerTile64);
                    plan.chunkElems = static_cast<uint32_t>(candidate);
                    plan.chunkElemsAligned = static_cast<uint32_t>(chunkPitchElems);
                    plan.chunkNumMax = static_cast<uint32_t>(chunkNumMax);
                    return true;
                }
            }
        }
        if (candidate <= step) {
            break;
        }
        candidate -= step;
    }
    return false;
}

static ge::graphStatus TilingForSameLenInnerCopy(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    SameLenSplitInfo splitInfo;
    if (runtimeInfo.innerLength <= 1 || !GetSameLenSplitInfo(runtimeInfo, splitInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.outerLength == 0 || runtimeInfo.outerLength > UINT32_MAX || runtimeInfo.midLength == 0 ||
        runtimeInfo.midLength > UINT32_MAX || runtimeInfo.innerLength == 0 || runtimeInfo.innerLength > UINT32_MAX ||
        runtimeInfo.splitNum <= 1 || runtimeInfo.splitNum > maxSplitNum || runtimeInfo.dataTypeSize == 0 ||
        runtimeInfo.effectiveUbNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t splitSize64 = splitInfo.splitSize;
    const uint64_t tailSplitSize64 = splitInfo.tailSplitSize;
    const uint64_t maxSplitSize64 = splitInfo.maxSplitSize;
    if (splitSize64 == 0 || splitSize64 > UINT32_MAX || tailSplitSize64 == 0 || tailSplitSize64 > UINT32_MAX ||
        maxSplitSize64 == 0 || maxSplitSize64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t innerBytes = runtimeInfo.innerLength * runtimeInfo.dataTypeSize;
    if (innerBytes == 0 || maxSplitSize64 > UINT64_MAX / innerBytes ||
        std::min(splitSize64, tailSplitSize64) > UINT64_MAX / innerBytes ||
        runtimeInfo.midLength > UINT64_MAX / innerBytes) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t splitBytes = maxSplitSize64 * innerBytes;
    const uint64_t minSplitBytes = std::min(splitSize64, tailSplitSize64) * innerBytes;
    const uint64_t rowBytes = runtimeInfo.midLength * innerBytes;
    const uint64_t stageBudget = runtimeInfo.effectiveUbNum / BUFFER_NUM;
    if (stageBudget < runtimeInfo.blockSize) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t maxDataCopyParamBlockLenBytes = GetMaxDataCopyParamBlockLenBytes(runtimeInfo);

    auto CalcOuterTile = [&](uint64_t tileBytes) -> uint32_t {
        const uint64_t stagePitch = AlignUpTo(tileBytes, runtimeInfo.blockSize);
        if (stagePitch == 0 || stagePitch > stageBudget) {
            return 0;
        }
        uint64_t outerTile64 = std::min<uint64_t>(runtimeInfo.outerLength, stageBudget / stagePitch);
        outerTile64 = std::min<uint64_t>(outerTile64, MAX_DATA_COPY_BLOCK_COUNT);
        return static_cast<uint32_t>(outerTile64);
    };

    auto CalcSegmentOuterTile = [&](uint64_t tileBytes) -> uint32_t {
        const uint64_t stagePitch = AlignUpTo(tileBytes, runtimeInfo.blockSize);
        if (stagePitch == 0 || stagePitch > stageBudget) {
            return 0;
        }
        const uint64_t stagePitchElems = stagePitch / runtimeInfo.dataTypeSize;
        if (stagePitchElems == 0 || stagePitchElems > runtimeInfo.maxUbTileLength) {
            return 0;
        }
        uint64_t outerTile64 = CeilDiv(runtimeInfo.outerLength, runtimeInfo.coreNum);
        outerTile64 = std::max<uint64_t>(outerTile64, 1);
        outerTile64 = std::min<uint64_t>(outerTile64, runtimeInfo.outerLength);
        outerTile64 = std::min<uint64_t>(outerTile64, runtimeInfo.maxUbTileLength / stagePitchElems);
        outerTile64 = std::min<uint64_t>(outerTile64, stageBudget / stagePitch);
        outerTile64 = std::min<uint64_t>(outerTile64, MAX_DATA_COPY_BLOCK_COUNT);
        return static_cast<uint32_t>(outerTile64);
    };

    uint32_t mode = splitVSameLenInnerCopyInnerTilePack;
    uint32_t outerTile = 0;
    uint32_t midTile = 1;
    uint32_t innerTile = 1;
    InnerSplitChunkPlan chunkPlan;

    if (ENABLE_SAME_LEN_INNER_FULL_ROW_PACK && runtimeInfo.innerLength < runtimeInfo.maxTileLength &&
        rowBytes <= maxDataCopyParamBlockLenBytes && splitBytes <= MAX_COPY_BYTES && rowBytes >= splitBytes &&
        (rowBytes - minSplitBytes) <= maxDataCopyParamBlockLenBytes && (rowBytes % runtimeInfo.blockSize) == 0 &&
        (splitBytes % runtimeInfo.blockSize) == 0) {
        outerTile = CalcOuterTile(rowBytes);
        if (outerTile > 0) {
            mode = splitVSameLenInnerCopyFullRowPack;
            midTile = static_cast<uint32_t>(runtimeInfo.midLength);
            innerTile = static_cast<uint32_t>(runtimeInfo.innerLength);
        }
    }

    if (outerTile == 0 && runtimeInfo.innerLength < runtimeInfo.maxTileLength &&
        (splitBytes % runtimeInfo.blockSize != 0 || rowBytes % runtimeInfo.blockSize != 0) &&
        AlignUpTo(splitBytes, runtimeInfo.blockSize) <=
            static_cast<uint64_t>(runtimeInfo.maxUbTileLength) * runtimeInfo.dataTypeSize &&
        AlignUpTo(splitBytes, runtimeInfo.blockSize) <= MAX_COPY_BYTES && rowBytes >= splitBytes &&
        (rowBytes - minSplitBytes) <= MAX_DATA_COPY_EXT_BLOCK_LEN &&
        AlignUpTo(splitBytes, runtimeInfo.blockSize) <= stageBudget) {
        outerTile = CalcSegmentOuterTile(splitBytes);
        if (outerTile > 0) {
            mode = splitVSameLenInnerCopySegmentInnerPack;
            midTile = static_cast<uint32_t>(splitSize64);
            innerTile = static_cast<uint32_t>(runtimeInfo.innerLength);
        }
    }

    if (outerTile == 0 && runtimeInfo.innerLength < runtimeInfo.maxTileLength) {
        const uint64_t maxMidByCopy = std::min<uint64_t>(maxSplitSize64, MAX_DATA_COPY_EXT_BLOCK_LEN / innerBytes);
        const uint64_t maxMidByUb = stageBudget / innerBytes;
        uint64_t candidate = std::min(maxMidByCopy, maxMidByUb);
        while (candidate > 0 && AlignUpTo(candidate * innerBytes, runtimeInfo.blockSize) > stageBudget) {
            --candidate;
        }
        if (candidate == 0) {
            return ge::GRAPH_FAILED;
        }
        outerTile = CalcOuterTile(candidate * innerBytes);
        if (outerTile > 0) {
            mode = splitVSameLenInnerCopyMidTilePack;
            midTile = static_cast<uint32_t>(candidate);
            innerTile = static_cast<uint32_t>(runtimeInfo.innerLength);
        }
    }

    if (outerTile == 0 && CalcInnerSplitChunkPlan(runtimeInfo, maxSplitSize64, stageBudget, chunkPlan)) {
        mode = splitVSameLenInnerCopySplitChunkPack;
        outerTile = chunkPlan.outerTile;
        midTile = chunkPlan.chunkElems;
        innerTile = 1;
    }

    if (outerTile == 0) {
        uint64_t candidate = std::min<uint64_t>(runtimeInfo.innerLength, runtimeInfo.maxTileLength);
        candidate = std::min<uint64_t>(candidate, MAX_DATA_COPY_EXT_BLOCK_LEN / runtimeInfo.dataTypeSize);
        candidate = std::min<uint64_t>(candidate, stageBudget / runtimeInfo.dataTypeSize);
        if (candidate >= runtimeInfo.alignedNum) {
            candidate = (candidate / runtimeInfo.alignedNum) * runtimeInfo.alignedNum;
        }
        if (candidate == 0) {
            return ge::GRAPH_FAILED;
        }
        outerTile = CalcOuterTile(candidate * runtimeInfo.dataTypeSize);
        if (outerTile == 0) {
            return ge::GRAPH_FAILED;
        }
        mode = splitVSameLenInnerCopyInnerTilePack;
        midTile = 1;
        innerTile = static_cast<uint32_t>(candidate);
    }

    const uint32_t outerTileNum = static_cast<uint32_t>(CeilDiv(runtimeInfo.outerLength, outerTile));
    const uint32_t outerTail = static_cast<uint32_t>(runtimeInfo.outerLength -
                                                     static_cast<uint64_t>(outerTileNum - 1) * outerTile);
    uint32_t midTileNum = 1;
    uint32_t midTail = midTile;
    uint32_t innerTileNum = 1;
    uint32_t innerTail = innerTile;
    uint64_t totalTaskNum = outerTileNum;

    if (mode == splitVSameLenInnerCopyMidTilePack) {
        midTileNum = static_cast<uint32_t>(CeilDiv(maxSplitSize64, midTile));
        midTail = static_cast<uint32_t>(maxSplitSize64 - static_cast<uint64_t>(midTileNum - 1) * midTile);
        totalTaskNum = static_cast<uint64_t>(outerTileNum) * runtimeInfo.splitNum * midTileNum;
    } else if (mode == splitVSameLenInnerCopySplitChunkPack) {
        midTileNum = chunkPlan.chunkNumMax;
        midTail = chunkPlan.chunkElems;
        totalTaskNum = static_cast<uint64_t>(outerTileNum) * runtimeInfo.splitNum * chunkPlan.chunkNumMax;
    } else if (mode == splitVSameLenInnerCopyInnerTilePack) {
        innerTileNum = static_cast<uint32_t>(CeilDiv(runtimeInfo.innerLength, innerTile));
        innerTail = static_cast<uint32_t>(runtimeInfo.innerLength -
                                          static_cast<uint64_t>(innerTileNum - 1) * innerTile);
        midTileNum = static_cast<uint32_t>(maxSplitSize64);
        midTail = 1;
        totalTaskNum = static_cast<uint64_t>(outerTileNum) * runtimeInfo.splitNum * maxSplitSize64 * innerTileNum;
    }

    if (totalTaskNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t needCore = 0;
    uint64_t formerNum = 0;
    uint64_t formerTaskNum = 0;
    uint64_t tailTaskNum = 0;
    if (mode == splitVSameLenInnerCopyFullRowPack || mode == splitVSameLenInnerCopySegmentInnerPack) {
        const uint64_t taskFactor = CeilDiv(totalTaskNum, runtimeInfo.coreNum);
        if (taskFactor == 0) {
            return ge::GRAPH_FAILED;
        }
        needCore = CeilDiv(totalTaskNum, taskFactor);
        if (needCore == 0) {
            return ge::GRAPH_FAILED;
        }
        formerTaskNum = taskFactor;
        tailTaskNum = totalTaskNum - (needCore - 1) * taskFactor;
        formerNum = tailTaskNum == taskFactor ? needCore : needCore - 1;
    } else {
        needCore = std::max<uint64_t>(1, std::min<uint64_t>(runtimeInfo.coreNum, totalTaskNum));
        if (needCore == 0) {
            return ge::GRAPH_FAILED;
        }
        formerNum = totalTaskNum % needCore;
        formerTaskNum = totalTaskNum / needCore;
        tailTaskNum = formerTaskNum;
        if (formerNum > 0) {
            ++formerTaskNum;
        }
    }

    SplitVTilingDataSameLenInnerCopy* tiling = context->GetTilingData<SplitVTilingDataSameLenInnerCopy>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalLength = runtimeInfo.totalLength;
    tiling->outerLength = runtimeInfo.outerLength;
    tiling->midLength = runtimeInfo.midLength;
    tiling->innerLength = runtimeInfo.innerLength;
    tiling->splitSize = static_cast<uint32_t>(splitSize64);
    tiling->tailSplitSize = static_cast<uint32_t>(tailSplitSize64);
    tiling->splitNum = static_cast<uint32_t>(runtimeInfo.splitNum);
    tiling->mode = mode;
    tiling->outerTile = outerTile;
    tiling->outerTileNum = outerTileNum;
    tiling->outerTail = outerTail;
    tiling->midTile = midTile;
    tiling->midTileNum = midTileNum;
    tiling->midTail = midTail;
    tiling->innerTile = innerTile;
    tiling->innerTileNum = innerTileNum;
    tiling->innerTail = innerTail;
    tiling->chunkElems = chunkPlan.chunkElems;
    tiling->chunkElemsAligned = chunkPlan.chunkElemsAligned;
    tiling->chunkNumMax = chunkPlan.chunkNumMax;
    tiling->totalTaskNum = totalTaskNum;
    tiling->formerTaskNum = formerTaskNum;
    tiling->tailTaskNum = tailTaskNum;
    tiling->formerNum = formerNum;

    context->SetBlockDim(static_cast<uint32_t>(needCore));
    context->SetTilingKey(TILING_KEY_SPLIT_V_SAME_LEN_INNER_COPY);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForUnevenInnerAlignedMid(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    if (runtimeInfo.innerLength <= 1 || IsSplitSameLenWithTail(runtimeInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.outerLength == 0 || runtimeInfo.outerLength > UINT32_MAX || runtimeInfo.midLength == 0 ||
        runtimeInfo.midLength > UINT32_MAX || runtimeInfo.innerLength == 0 || runtimeInfo.innerLength > UINT32_MAX ||
        runtimeInfo.splitNum <= 1 || runtimeInfo.splitNum > maxSplitNum || runtimeInfo.dataTypeSize == 0 ||
        runtimeInfo.effectiveUbNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t innerBytes = runtimeInfo.innerLength * runtimeInfo.dataTypeSize;
    const uint64_t stageBudget = runtimeInfo.effectiveUbNum / BUFFER_NUM;
    if (innerBytes == 0 || stageBudget < runtimeInfo.blockSize) {
        return ge::GRAPH_FAILED;
    }

    uint64_t splitOffset = 0;
    uint64_t maxSplitSize = 0;
    bool splitBoundaryAligned = true;
    for (int64_t i = 0; i < runtimeInfo.splitNum; ++i) {
        if (runtimeInfo.sizeSplits[i] <= 0 || runtimeInfo.sizeSplits[i] > UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }
        if ((splitOffset * innerBytes) % runtimeInfo.blockSize != 0) {
            splitBoundaryAligned = false;
        }
        maxSplitSize = std::max<uint64_t>(maxSplitSize, static_cast<uint64_t>(runtimeInfo.sizeSplits[i]));
        splitOffset += static_cast<uint64_t>(runtimeInfo.sizeSplits[i]);
    }
    if (splitOffset != runtimeInfo.midLength || maxSplitSize == 0) {
        return ge::GRAPH_FAILED;
    }

    InnerSplitChunkPlan chunkPlan;
    const bool hasChunkPlan = CalcInnerSplitChunkPlan(runtimeInfo, maxSplitSize, stageBudget, chunkPlan);

    auto FillSplitInfo = [&](SplitVTilingDataUnevenInnerAlignedMid* tiling) {
        uint64_t curOffset = 0;
        for (int64_t i = 0; i < runtimeInfo.splitNum; ++i) {
            tiling->sizeSplits[i] = static_cast<uint32_t>(runtimeInfo.sizeSplits[i]);
            tiling->splitOffsets[i] = static_cast<uint32_t>(curOffset);
            curOffset += static_cast<uint64_t>(runtimeInfo.sizeSplits[i]);
        }
        for (int64_t i = runtimeInfo.splitNum; i < maxSplitNum; ++i) {
            tiling->sizeSplits[i] = 0;
            tiling->splitOffsets[i] = 0;
        }
    };

    auto EmitSplitChunk = [&]() -> ge::graphStatus {
        if (!hasChunkPlan) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t outerTile = chunkPlan.outerTile;
        const uint32_t outerTileNum = static_cast<uint32_t>(CeilDiv(runtimeInfo.outerLength, outerTile));
        const uint32_t outerTail = static_cast<uint32_t>(runtimeInfo.outerLength -
                                                         static_cast<uint64_t>(outerTileNum - 1) * outerTile);
        const uint64_t totalTaskNum = static_cast<uint64_t>(outerTileNum) * runtimeInfo.splitNum *
                                      chunkPlan.chunkNumMax;
        if (totalTaskNum == 0) {
            return ge::GRAPH_FAILED;
        }

        const auto corePlan = CalcCoreSplitPlan(totalTaskNum, runtimeInfo.coreNum);

        SplitVTilingDataUnevenInnerAlignedMid* tiling = context->GetTilingData<SplitVTilingDataUnevenInnerAlignedMid>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        tiling->totalLength = runtimeInfo.totalLength;
        tiling->outerLength = runtimeInfo.outerLength;
        tiling->midLength = runtimeInfo.midLength;
        tiling->innerLength = runtimeInfo.innerLength;
        tiling->splitNum = static_cast<uint32_t>(runtimeInfo.splitNum);
        tiling->mode = splitVUnevenInnerSplitChunkPack;
        FillSplitInfo(tiling);
        tiling->outerTile = outerTile;
        tiling->outerTileNum = outerTileNum;
        tiling->outerTail = outerTail;
        tiling->midTile = chunkPlan.chunkElems;
        tiling->midTileNum = chunkPlan.chunkNumMax;
        tiling->midTail = chunkPlan.chunkElems;
        tiling->chunkElems = chunkPlan.chunkElems;
        tiling->chunkElemsAligned = chunkPlan.chunkElemsAligned;
        tiling->chunkNumMax = chunkPlan.chunkNumMax;
        FillTaskSplitFields(tiling, totalTaskNum, corePlan);

        context->SetBlockDim(static_cast<uint32_t>(corePlan.needCore));
        context->SetTilingKey(TILING_KEY_SPLIT_V_UNEVEN_INNER_ALIGNED_MID);
        return SetWorkspaceZero(context);
    };

    if (!splitBoundaryAligned) {
        const uint64_t maxSplitBytes = maxSplitSize * innerBytes;
        const uint64_t maxSplitPitch = AlignUpTo(maxSplitBytes, runtimeInfo.blockSize);
        const uint64_t maxSplitPitchElems = maxSplitPitch / runtimeInfo.dataTypeSize;
        if (maxSplitBytes != 0 && maxSplitBytes <= MAX_DATA_COPY_EXT_BLOCK_LEN && maxSplitPitch != 0 &&
            maxSplitPitch <= stageBudget && maxSplitPitchElems != 0 &&
            maxSplitPitchElems <= runtimeInfo.maxUbTileLength) {
            uint64_t outerTile64 = CeilDiv(runtimeInfo.outerLength, runtimeInfo.coreNum);
            outerTile64 = std::max<uint64_t>(outerTile64, 1);
            outerTile64 = std::min<uint64_t>(outerTile64, runtimeInfo.outerLength);
            outerTile64 = std::min<uint64_t>(outerTile64, MAX_DATA_COPY_BLOCK_COUNT);
            outerTile64 = std::min<uint64_t>(outerTile64, stageBudget / maxSplitPitch);
            outerTile64 = std::min<uint64_t>(outerTile64, runtimeInfo.maxUbTileLength / maxSplitPitchElems);
            if (outerTile64 == 0 || outerTile64 > UINT32_MAX) {
                return EmitSplitChunk();
            }

            const uint32_t outerTile = static_cast<uint32_t>(outerTile64);
            const uint32_t outerTileNum = static_cast<uint32_t>(CeilDiv(runtimeInfo.outerLength, outerTile));
            const uint32_t outerTail = static_cast<uint32_t>(runtimeInfo.outerLength -
                                                             static_cast<uint64_t>(outerTileNum - 1) * outerTile);
            const uint64_t totalTaskNum = static_cast<uint64_t>(outerTileNum) * runtimeInfo.splitNum;
            if (totalTaskNum == 0) {
                return EmitSplitChunk();
            }

            const auto corePlan = CalcCoreSplitPlanByCore(
                totalTaskNum, std::min<uint64_t>(runtimeInfo.coreNum, static_cast<uint64_t>(outerTileNum)));

            SplitVTilingDataUnevenInnerAlignedMid*
                tiling = context->GetTilingData<SplitVTilingDataUnevenInnerAlignedMid>();
            OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
            tiling->totalLength = runtimeInfo.totalLength;
            tiling->outerLength = runtimeInfo.outerLength;
            tiling->midLength = runtimeInfo.midLength;
            tiling->innerLength = runtimeInfo.innerLength;
            tiling->splitNum = static_cast<uint32_t>(runtimeInfo.splitNum);
            tiling->mode = splitVUnevenInnerSegmentPack;
            FillSplitInfo(tiling);
            tiling->outerTile = outerTile;
            tiling->outerTileNum = outerTileNum;
            tiling->outerTail = outerTail;
            tiling->midTile = static_cast<uint32_t>(maxSplitSize);
            tiling->midTileNum = static_cast<uint32_t>(runtimeInfo.splitNum);
            tiling->midTail = static_cast<uint32_t>(maxSplitSize);
            tiling->chunkElems = 0;
            tiling->chunkElemsAligned = 0;
            tiling->chunkNumMax = 0;
            FillTaskSplitFields(tiling, totalTaskNum, corePlan);

            context->SetBlockDim(static_cast<uint32_t>(corePlan.needCore));
            context->SetTilingKey(TILING_KEY_SPLIT_V_UNEVEN_INNER_ALIGNED_MID);
            return SetWorkspaceZero(context);
        }
        return EmitSplitChunk();
    }

    if (runtimeInfo.innerLength >= runtimeInfo.maxTileLength || innerBytes > MAX_COPY_BYTES) {
        return EmitSplitChunk();
    }

    const uint64_t alignMidUnit = runtimeInfo.blockSize / Gcd64(innerBytes, runtimeInfo.blockSize);
    const uint64_t minMidUnit = std::min<uint64_t>(runtimeInfo.midLength, alignMidUnit);
    const uint64_t minPitch = AlignUpTo(minMidUnit * innerBytes, runtimeInfo.blockSize);
    if (minPitch == 0 || minPitch > stageBudget) {
        return EmitSplitChunk();
    }

    uint64_t outerTile64 = CeilDiv(runtimeInfo.outerLength, runtimeInfo.coreNum);
    outerTile64 = std::max<uint64_t>(outerTile64, 1);
    outerTile64 = std::min<uint64_t>(outerTile64, runtimeInfo.outerLength);
    outerTile64 = std::min<uint64_t>(outerTile64, MAX_DATA_COPY_BLOCK_COUNT);
    if (outerTile64 * minPitch > stageBudget) {
        outerTile64 = std::min<uint64_t>(runtimeInfo.outerLength, stageBudget / minPitch);
        outerTile64 = std::min<uint64_t>(outerTile64, MAX_DATA_COPY_BLOCK_COUNT);
    }
    if (outerTile64 == 0) {
        return EmitSplitChunk();
    }

    auto CalcMidTile = [&](uint64_t outerTile) -> uint32_t {
        uint64_t candidate = runtimeInfo.midLength;
        candidate = std::min<uint64_t>(candidate, MAX_COPY_BYTES / innerBytes);
        candidate = std::min<uint64_t>(candidate, MAX_DATA_COPY_EXT_BLOCK_LEN / innerBytes);
        candidate = std::min<uint64_t>(candidate, runtimeInfo.maxUbTileLength / runtimeInfo.innerLength);
        if (candidate >= alignMidUnit) {
            candidate = (candidate / alignMidUnit) * alignMidUnit;
        } else if (runtimeInfo.midLength > alignMidUnit) {
            return 0U;
        }
        while (candidate > 0) {
            const uint64_t tileBytes = candidate * innerBytes;
            const uint64_t tilePitch = AlignUpTo(tileBytes, runtimeInfo.blockSize);
            const uint64_t tilePitchElems = tilePitch / runtimeInfo.dataTypeSize;
            bool valid = tileBytes <= MAX_COPY_BYTES && tilePitchElems <= runtimeInfo.maxUbTileLength &&
                         outerTile * tilePitch <= stageBudget &&
                         outerTile * tilePitchElems <= runtimeInfo.maxUbTileLength;
            if (valid) {
                return static_cast<uint32_t>(candidate);
            }
            if (candidate <= alignMidUnit) {
                if (runtimeInfo.midLength <= alignMidUnit && candidate == runtimeInfo.midLength) {
                    break;
                }
                return 0U;
            }
            candidate -= alignMidUnit;
        }
        return 0U;
    };

    uint32_t midTile = CalcMidTile(outerTile64);
    if (midTile == 0) {
        outerTile64 = std::min<uint64_t>(runtimeInfo.outerLength, stageBudget / minPitch);
        outerTile64 = std::min<uint64_t>(outerTile64, MAX_DATA_COPY_BLOCK_COUNT);
        if (outerTile64 == 0) {
            return EmitSplitChunk();
        }
        midTile = CalcMidTile(outerTile64);
    }
    if (midTile == 0) {
        return EmitSplitChunk();
    }

    const uint32_t outerTile = static_cast<uint32_t>(outerTile64);
    const uint32_t outerTileNum = static_cast<uint32_t>(CeilDiv(runtimeInfo.outerLength, outerTile));
    const uint32_t outerTail = static_cast<uint32_t>(runtimeInfo.outerLength -
                                                     static_cast<uint64_t>(outerTileNum - 1) * outerTile);
    const uint32_t midTileNum = static_cast<uint32_t>(CeilDiv(runtimeInfo.midLength, midTile));
    const uint32_t midTail = static_cast<uint32_t>(runtimeInfo.midLength -
                                                   static_cast<uint64_t>(midTileNum - 1) * midTile);
    const uint64_t totalTaskNum = static_cast<uint64_t>(outerTileNum) * midTileNum;
    if (totalTaskNum == 0) {
        return EmitSplitChunk();
    }

    if (hasChunkPlan) {
        const uint32_t chunkOuterTileNum = static_cast<uint32_t>(CeilDiv(runtimeInfo.outerLength, chunkPlan.outerTile));
        const uint64_t alignedCost = totalTaskNum * runtimeInfo.splitNum;
        const uint64_t chunkCost = static_cast<uint64_t>(chunkOuterTileNum) * runtimeInfo.splitNum *
                                   chunkPlan.chunkNumMax;
        if (chunkCost <= alignedCost) {
            return EmitSplitChunk();
        }
    }

    const auto corePlan = CalcCoreSplitPlan(totalTaskNum, runtimeInfo.coreNum);

    SplitVTilingDataUnevenInnerAlignedMid* tiling = context->GetTilingData<SplitVTilingDataUnevenInnerAlignedMid>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalLength = runtimeInfo.totalLength;
    tiling->outerLength = runtimeInfo.outerLength;
    tiling->midLength = runtimeInfo.midLength;
    tiling->innerLength = runtimeInfo.innerLength;
    tiling->splitNum = static_cast<uint32_t>(runtimeInfo.splitNum);
    tiling->mode = splitVUnevenInnerAlignedMidTilePack;
    FillSplitInfo(tiling);
    tiling->outerTile = outerTile;
    tiling->outerTileNum = outerTileNum;
    tiling->outerTail = outerTail;
    tiling->midTile = midTile;
    tiling->midTileNum = midTileNum;
    tiling->midTail = midTail;
    tiling->chunkElems = 0;
    tiling->chunkElemsAligned = 0;
    tiling->chunkNumMax = 0;
    FillTaskSplitFields(tiling, totalTaskNum, corePlan);

    context->SetBlockDim(static_cast<uint32_t>(corePlan.needCore));
    context->SetTilingKey(TILING_KEY_SPLIT_V_UNEVEN_INNER_ALIGNED_MID);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForSameLenCompact32Bit(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    static constexpr uint32_t OUTER_TILE_FIXED = 256;
    static constexpr uint32_t B16_ADDR_COUNT = 16;
    static constexpr uint32_t VIEW_DATA_SIZE = sizeof(uint16_t);

    SameLenSplitInfo splitInfo;
    if (runtimeInfo.innerLength != 1 || !GetSameLenSplitInfo(runtimeInfo, splitInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (splitInfo.tailSplitSize > splitInfo.splitSize) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t viewFactor = GetCompact32BitViewFactor(runtimeInfo);
    if (viewFactor == 0 || runtimeInfo.outerLength == 0 || runtimeInfo.midLength == 0 || runtimeInfo.splitNum <= 1 ||
        runtimeInfo.splitNum > maxSplitNum) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t splitSize64 = splitInfo.splitSize;
    const uint64_t tailSplitSize64 = splitInfo.tailSplitSize;
    if (runtimeInfo.midLength > UINT64_MAX / viewFactor || splitSize64 > UINT64_MAX / viewFactor ||
        tailSplitSize64 > UINT64_MAX / viewFactor || runtimeInfo.totalLength > UINT64_MAX / viewFactor) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t viewRowLength64 = runtimeInfo.midLength * viewFactor;
    const uint64_t viewSplitSize64 = splitSize64 * viewFactor;
    const uint64_t viewTailSplitSize64 = tailSplitSize64 * viewFactor;
    const uint64_t viewTotalLength64 = runtimeInfo.totalLength * viewFactor;
    if (splitSize64 == 0 || viewRowLength64 == 0 || viewRowLength64 > UINT8_MAX || viewSplitSize64 == 0 ||
        viewSplitSize64 > UINT8_MAX || viewTailSplitSize64 == 0 || viewTailSplitSize64 > UINT8_MAX ||
        viewTotalLength64 / viewFactor != runtimeInfo.totalLength) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t needUbBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * (2 * viewRowLength64 + 2 * viewSplitSize64) *
                                 VIEW_DATA_SIZE;
    const uint64_t inputTileBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * viewRowLength64 * VIEW_DATA_SIZE;
    const uint64_t outputTileBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * viewSplitSize64 * VIEW_DATA_SIZE;
    if (needUbBytes == 0 || needUbBytes > runtimeInfo.effectiveUbNum || inputTileBytes == 0 || outputTileBytes == 0 ||
        inputTileBytes > MAX_COPY_BYTES || outputTileBytes > MAX_COPY_BYTES) {
        return ge::GRAPH_FAILED;
    }

    OuterTilePlan outerPlan;
    OuterTaskSplitPlan taskPlan;
    if (!CalcOuterTileTaskSplitPlan(runtimeInfo.outerLength, OUTER_TILE_FIXED, 1, runtimeInfo.coreNum, outerPlan,
                                    taskPlan)) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<SplitVTilingDataSameLenCompact>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillSameLenCompactFields(
        tiling, viewTotalLength64, runtimeInfo.outerLength, static_cast<uint32_t>(viewRowLength64),
        static_cast<uint32_t>(viewSplitSize64), static_cast<uint32_t>(viewTailSplitSize64), runtimeInfo.splitNum,
        OUTER_TILE_FIXED, outerPlan, taskPlan, static_cast<uint32_t>(B16_ADDR_COUNT * viewRowLength64),
        static_cast<uint32_t>(B16_ADDR_COUNT * viewSplitSize64), static_cast<uint32_t>(runtimeInfo.splitNum), 1);

    context->SetBlockDim(taskPlan.needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_SAME_LEN_COMPACT_32BIT);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForUnevenCompact32Bit(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    static constexpr uint32_t MODE_FULL_ROW = 0;
    static constexpr uint32_t MODE_ROW_CHUNK = 1;
    static constexpr uint32_t OUTER_TILE_FIXED = 256;
    static constexpr uint32_t B16_ADDR_COUNT = 16;
    static constexpr uint32_t VIEW_DATA_SIZE = sizeof(uint16_t);

    if (runtimeInfo.innerLength != 1 || IsSplitSameLenWithTail(runtimeInfo)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t viewFactor = GetCompact32BitViewFactor(runtimeInfo);
    if (viewFactor == 0 || runtimeInfo.outerLength == 0 || runtimeInfo.midLength == 0 || runtimeInfo.splitNum <= 1 ||
        runtimeInfo.splitNum > maxSplitNum) {
        return ge::GRAPH_FAILED;
    }

    if (runtimeInfo.midLength > UINT64_MAX / viewFactor || runtimeInfo.totalLength > UINT64_MAX / viewFactor) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t viewRowLength64 = runtimeInfo.midLength * viewFactor;
    const uint64_t viewTotalLength64 = runtimeInfo.totalLength * viewFactor;
    if (viewRowLength64 == 0 || viewRowLength64 > UINT32_MAX ||
        viewTotalLength64 / viewFactor != runtimeInfo.totalLength) {
        return ge::GRAPH_FAILED;
    }
    uint64_t splitSum = 0;
    uint32_t maxViewSplitSize = 0;
    uint32_t viewSizeSplits[maxSplitNum] = {};
    for (int64_t i = 0; i < runtimeInfo.splitNum; ++i) {
        const int64_t split = runtimeInfo.sizeSplits[i];
        if (split <= 0 || static_cast<uint64_t>(split) > UINT64_MAX / viewFactor ||
            splitSum > UINT64_MAX / viewFactor) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t viewSplit = static_cast<uint64_t>(split) * viewFactor;
        if (viewSplit == 0 || viewSplit > UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }
        viewSizeSplits[i] = static_cast<uint32_t>(viewSplit);
        splitSum += static_cast<uint64_t>(split);
        maxViewSplitSize = std::max<uint32_t>(maxViewSplitSize, static_cast<uint32_t>(viewSplit));
    }
    if (splitSum != runtimeInfo.midLength || maxViewSplitSize == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t mode = UINT32_MAX;
    uint32_t rowPitch = 0;
    uint32_t colChunkSize = 0;
    uint32_t colChunkNum = 1;
    if (viewRowLength64 <= UINT8_MAX && maxViewSplitSize <= UINT8_MAX) {
        const uint64_t inputTileBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * viewRowLength64 * VIEW_DATA_SIZE;
        const uint64_t outputTileBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * maxViewSplitSize * VIEW_DATA_SIZE;
        const uint64_t needUbBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) *
                                     (2 * viewRowLength64 + 2 * static_cast<uint64_t>(maxViewSplitSize)) *
                                     VIEW_DATA_SIZE;
        if (inputTileBytes != 0 && outputTileBytes != 0 && inputTileBytes <= MAX_COPY_BYTES &&
            outputTileBytes <= MAX_COPY_BYTES && needUbBytes <= runtimeInfo.effectiveUbNum) {
            mode = MODE_FULL_ROW;
            rowPitch = static_cast<uint32_t>(viewRowLength64);
        }
    }
    if (mode == UINT32_MAX && maxViewSplitSize <= UINT8_MAX) {
        const uint64_t splitPitch64 = AlignUpTo(maxViewSplitSize, B16_ADDR_COUNT);
        const uint64_t needUbBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * 3 * splitPitch64 * VIEW_DATA_SIZE;
        const uint64_t chunkCopyBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * splitPitch64 * VIEW_DATA_SIZE;
        if (splitPitch64 <= UINT8_MAX && needUbBytes <= runtimeInfo.effectiveUbNum &&
            chunkCopyBytes <= MAX_DATA_COPY_EXT_BLOCK_LEN) {
            mode = MODE_ROW_CHUNK;
            rowPitch = static_cast<uint32_t>(splitPitch64);
            colChunkSize = 0;
            colChunkNum = static_cast<uint32_t>(runtimeInfo.splitNum);
        }
    }
    if (rowPitch == 0 || rowPitch > UINT8_MAX || rowPitch > UINT32_MAX / B16_ADDR_COUNT) {
        return ge::GRAPH_FAILED;
    }

    OuterTilePlan outerPlan;
    OuterTaskSplitPlan taskPlan;
    const uint32_t taskFactor = mode == MODE_ROW_CHUNK ? colChunkNum : 1U;
    if (!CalcOuterTileTaskSplitPlan(runtimeInfo.outerLength, OUTER_TILE_FIXED, taskFactor, runtimeInfo.coreNum,
                                    outerPlan, taskPlan)) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<SplitVTilingDataUnevenCompact>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillUnevenCompactRuntimeFields(tiling, viewTotalLength64, runtimeInfo.outerLength,
                                   static_cast<uint32_t>(viewRowLength64), runtimeInfo.splitNum, maxViewSplitSize);
    FillUnevenSplitLists(tiling, runtimeInfo.splitNum, viewSizeSplits);
    FillOuterTaskFields(tiling, OUTER_TILE_FIXED, outerPlan, taskPlan);
    FillUnevenCompactModeFields(tiling, mode, B16_ADDR_COUNT * rowPitch, B16_ADDR_COUNT * rowPitch, 0, 0, colChunkSize,
                                colChunkNum);

    context->SetBlockDim(taskPlan.needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_UNEVEN_COMPACT_32BIT);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForSameLenPureCopyWide(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    SameLenSplitInfo splitInfo;
    if (runtimeInfo.innerLength != 1 || !GetSameLenSplitInfo(runtimeInfo, splitInfo) ||
        GetCompact32BitViewFactor(runtimeInfo) == 0 || runtimeInfo.outerLength == 0 || runtimeInfo.splitNum <= 1 ||
        runtimeInfo.splitNum > maxSplitNum) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t b16ViewFactor = runtimeInfo.dataTypeSize / sizeof(uint16_t);
    if (runtimeInfo.totalLength > UINT64_MAX / b16ViewFactor || splitInfo.maxSplitSize > UINT64_MAX / b16ViewFactor) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t rowLength64 = runtimeInfo.midLength * b16ViewFactor;
    const uint64_t splitSize64 = splitInfo.splitSize * b16ViewFactor;
    const uint64_t tailSplitSize64 = splitInfo.tailSplitSize * b16ViewFactor;
    const uint64_t maxSplitSize64 = splitInfo.maxSplitSize * b16ViewFactor;
    const uint64_t totalLength64 = runtimeInfo.totalLength * b16ViewFactor;
    if (rowLength64 == 0 || rowLength64 > UINT32_MAX || splitSize64 == 0 || splitSize64 > UINT32_MAX ||
        tailSplitSize64 == 0 || tailSplitSize64 > UINT32_MAX || maxSplitSize64 == 0 || maxSplitSize64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t b16AlignedNum = runtimeInfo.blockSize / sizeof(uint16_t);
    const uint64_t splitAligned = AlignUpTo(maxSplitSize64, b16AlignedNum);
    const uint64_t splitBytes = maxSplitSize64 * sizeof(uint16_t);
    if (splitBytes == 0 || splitBytes > MAX_COPY_BYTES) {
        return ge::GRAPH_FAILED;
    }
    uint64_t outerTile64 = runtimeInfo.effectiveUbNum / (2 * splitAligned * sizeof(uint16_t));
    outerTile64 = std::min<uint64_t>(outerTile64, runtimeInfo.outerLength);
    outerTile64 = std::min<uint64_t>(outerTile64, MAX_DATA_COPY_BLOCK_COUNT);
    if (outerTile64 == 0 || outerTile64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t outerTile = static_cast<uint32_t>(outerTile64);
    OuterTilePlan outerPlan;
    OuterTaskSplitPlan taskPlan;
    if (!CalcOuterTileTaskSplitPlan(runtimeInfo.outerLength, outerTile, 1, runtimeInfo.coreNum, outerPlan, taskPlan)) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<SplitVTilingDataSameLen>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillSameLenFields(tiling, totalLength64, runtimeInfo.outerLength, rowLength64, 1,
                      static_cast<uint32_t>(splitSize64), static_cast<uint32_t>(tailSplitSize64), runtimeInfo.splitNum,
                      outerTile, outerPlan, taskPlan, 0);

    context->SetBlockDim(taskPlan.needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_SAME_LEN_PURE_COPY_WIDE);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForSameLenCompactLargeOuter(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    SameLenSplitInfo splitInfo;
    if (runtimeInfo.innerLength != 1 || !GetSameLenSplitInfo(runtimeInfo, splitInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!IsSameLenCompactDataType(runtimeInfo.dataType) || splitInfo.tailSplitSize > splitInfo.splitSize) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.outerLength <= static_cast<uint64_t>(SAME_LEN_COMPACT_OUTER_TILE) * runtimeInfo.coreNum ||
        runtimeInfo.outerLength == 0 || runtimeInfo.midLength == 0 || runtimeInfo.midLength > UINT8_MAX ||
        runtimeInfo.splitNum <= 1 || runtimeInfo.splitNum > maxSplitNum) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t splitSize64 = splitInfo.splitSize;
    const uint64_t tailSplitSize64 = splitInfo.tailSplitSize;
    if (splitSize64 == 0 || splitSize64 > UINT8_MAX || tailSplitSize64 == 0 || tailSplitSize64 > UINT8_MAX) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.midLength > UINT16_MAX || splitSize64 > UINT16_MAX || runtimeInfo.midLength < splitSize64) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t inputTileBytes = static_cast<uint64_t>(SAME_LEN_COMPACT_LARGE_OUTER_TILE) * runtimeInfo.midLength *
                                    runtimeInfo.dataTypeSize;
    const uint64_t transTileBytes = static_cast<uint64_t>(SAME_LEN_COMPACT_OUTER_TILE) * runtimeInfo.midLength *
                                    runtimeInfo.dataTypeSize;
    const uint64_t segTileBytes = static_cast<uint64_t>(SAME_LEN_COMPACT_OUTER_TILE) * splitSize64 *
                                  runtimeInfo.dataTypeSize;
    const uint64_t needUbBytes = inputTileBytes + transTileBytes + segTileBytes;
    if (inputTileBytes == 0 || inputTileBytes > MAX_DATA_COPY_EXT_BLOCK_LEN || segTileBytes == 0 ||
        segTileBytes > MAX_DATA_COPY_EXT_BLOCK_LEN || needUbBytes == 0 || needUbBytes > runtimeInfo.effectiveUbNum ||
        needUbBytes > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    OuterTilePlan outerPlan;
    OuterTaskSplitPlan taskPlan;
    if (!CalcOuterTileTaskSplitPlan(runtimeInfo.outerLength, SAME_LEN_COMPACT_LARGE_OUTER_TILE, 1, runtimeInfo.coreNum,
                                    outerPlan, taskPlan)) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<SplitVTilingDataSameLenCompact>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillSameLenCompactFields(tiling, runtimeInfo.totalLength, runtimeInfo.outerLength,
                             static_cast<uint32_t>(runtimeInfo.midLength), static_cast<uint32_t>(splitSize64),
                             static_cast<uint32_t>(tailSplitSize64), runtimeInfo.splitNum,
                             SAME_LEN_COMPACT_LARGE_OUTER_TILE, outerPlan, taskPlan,
                             static_cast<uint32_t>(SAME_LEN_COMPACT_B16_ADDR_COUNT * runtimeInfo.midLength),
                             static_cast<uint32_t>(SAME_LEN_COMPACT_B16_ADDR_COUNT * splitSize64),
                             static_cast<uint32_t>(runtimeInfo.splitNum), 1);

    context->SetBlockDim(taskPlan.needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_SAME_LEN_COMPACT_LARGE_OUTER);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForSameLenCompactDoubleBuffer(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    static constexpr uint32_t OUTER_TILE_FIXED = SAME_LEN_COMPACT_OUTER_TILE;
    static constexpr uint32_t MAX_DOUBLE_BUFFER_ROW_LENGTH = 128;

    SameLenSplitInfo splitInfo;
    if (runtimeInfo.innerLength != 1 || !GetSameLenSplitInfo(runtimeInfo, splitInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!IsSameLenCompactDataType(runtimeInfo.dataType) || splitInfo.tailSplitSize > splitInfo.splitSize) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.splitNum < 2 || runtimeInfo.splitNum > maxSplitNum || runtimeInfo.outerLength == 0 ||
        runtimeInfo.midLength == 0 || runtimeInfo.midLength > MAX_DOUBLE_BUFFER_ROW_LENGTH) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t splitSize64 = splitInfo.splitSize;
    const uint64_t tailSplitSize64 = splitInfo.tailSplitSize;
    if (splitSize64 == 0 || splitSize64 > UINT8_MAX || tailSplitSize64 == 0 || tailSplitSize64 > splitSize64 ||
        runtimeInfo.midLength < splitSize64) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t inputTileBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * runtimeInfo.midLength *
                                    runtimeInfo.dataTypeSize;
    const uint64_t outputTileBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * splitSize64 * runtimeInfo.dataTypeSize;
    const uint64_t needUbBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) *
                                 (2 * runtimeInfo.midLength + 4 * splitSize64) * runtimeInfo.dataTypeSize;
    if (inputTileBytes == 0 || outputTileBytes == 0 || inputTileBytes > MAX_DATA_COPY_EXT_BLOCK_LEN ||
        outputTileBytes > MAX_DATA_COPY_EXT_BLOCK_LEN || needUbBytes == 0 || needUbBytes > runtimeInfo.effectiveUbNum ||
        needUbBytes > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t outerTileNum64 = CeilDiv(runtimeInfo.outerLength, OUTER_TILE_FIXED);
    if (outerTileNum64 == 0 || outerTileNum64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t outerTileNum = static_cast<uint32_t>(outerTileNum64);
    const uint32_t outerTail = static_cast<uint32_t>(runtimeInfo.outerLength -
                                                     static_cast<uint64_t>(outerTileNum - 1) * OUTER_TILE_FIXED);
    if (outerTail == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t maxCoreNum = std::max<uint32_t>(runtimeInfo.coreNum, 1);
    const uint32_t blockFactor = static_cast<uint32_t>(CeilDiv(outerTileNum, maxCoreNum));
    const uint32_t needCore = static_cast<uint32_t>(CeilDiv(outerTileNum, blockFactor));
    const uint32_t tailOuterTileNum = outerTileNum -
                                      static_cast<uint32_t>(static_cast<uint64_t>(needCore - 1) * blockFactor);
    const uint32_t formerNum = tailOuterTileNum == blockFactor ? needCore : needCore - 1;
    const OuterTilePlan outerPlan{outerTileNum, outerTail};
    const OuterTaskSplitPlan taskPlan{needCore, blockFactor, tailOuterTileNum, formerNum};

    auto* tiling = context->GetTilingData<SplitVTilingDataSameLenCompact>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillSameLenCompactFields(tiling, runtimeInfo.totalLength, runtimeInfo.outerLength,
                             static_cast<uint32_t>(runtimeInfo.midLength), static_cast<uint32_t>(splitSize64),
                             static_cast<uint32_t>(tailSplitSize64), runtimeInfo.splitNum, OUTER_TILE_FIXED, outerPlan,
                             taskPlan, static_cast<uint32_t>(SAME_LEN_COMPACT_B16_ADDR_COUNT * runtimeInfo.midLength),
                             static_cast<uint32_t>(SAME_LEN_COMPACT_B16_ADDR_COUNT * splitSize64),
                             static_cast<uint32_t>(runtimeInfo.splitNum), 1);

    context->SetBlockDim(needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_SAME_LEN_COMPACT_DOUBLE_BUFFER);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForSameLenCompact(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    static constexpr uint32_t OUTER_TILE_FIXED = SAME_LEN_COMPACT_OUTER_TILE;

    SameLenSplitInfo splitInfo;
    if (runtimeInfo.innerLength != 1 || !GetSameLenSplitInfo(runtimeInfo, splitInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (splitInfo.tailSplitSize > splitInfo.splitSize) {
        return ge::GRAPH_FAILED;
    }
    if (!IsSameLenCompactDataType(runtimeInfo.dataType)) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t splitSize64 = splitInfo.splitSize;
    const uint64_t tailSplitSize64 = splitInfo.tailSplitSize;
    if (splitSize64 == 0 || splitSize64 > UINT8_MAX) {
        return ge::GRAPH_FAILED;
    }
    if (tailSplitSize64 == 0 || tailSplitSize64 > UINT8_MAX) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.midLength == 0 || runtimeInfo.midLength > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.splitNum <= 1 || runtimeInfo.splitNum > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    uint32_t chunkSplitNum = static_cast<uint32_t>(runtimeInfo.splitNum);
    uint32_t colChunkNum = 1;
    uint64_t rowPitch = runtimeInfo.midLength;
    bool enableCompact = false;

    auto IsUbEnough = [&](uint64_t pitch) -> bool {
        const uint64_t needUbBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * (2 * pitch + splitSize64) *
                                     runtimeInfo.dataTypeSize;
        return needUbBytes != 0 && needUbBytes <= runtimeInfo.effectiveUbNum && needUbBytes <= UINT32_MAX;
    };

    if (runtimeInfo.midLength <= UINT8_MAX && IsUbEnough(runtimeInfo.midLength)) {
        enableCompact = true;
    } else {
        const uint64_t maxChunkSplitNum = std::min<uint64_t>(static_cast<uint64_t>(runtimeInfo.splitNum),
                                                             UINT8_MAX / splitSize64);
        for (uint64_t candidate = maxChunkSplitNum; candidate >= 1; --candidate) {
            const uint64_t chunkCols = candidate * splitSize64;
            const uint64_t chunkPitch = AlignUpTo(chunkCols, SAME_LEN_COMPACT_B16_ADDR_COUNT);
            if (chunkPitch > UINT8_MAX) {
                continue;
            }
            if (IsUbEnough(chunkPitch)) {
                chunkSplitNum = static_cast<uint32_t>(candidate);
                colChunkNum = static_cast<uint32_t>((runtimeInfo.splitNum + candidate - 1) / candidate);
                rowPitch = chunkPitch;
                enableCompact = true;
                break;
            }
        }
    }
    if (!enableCompact || rowPitch == 0 || rowPitch > UINT8_MAX) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t inputTileBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * rowPitch * runtimeInfo.dataTypeSize;
    const uint64_t outputTileBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * splitSize64 * runtimeInfo.dataTypeSize;
    if (inputTileBytes == 0 || outputTileBytes == 0 || inputTileBytes > MAX_DATA_COPY_EXT_BLOCK_LEN ||
        outputTileBytes > MAX_DATA_COPY_EXT_BLOCK_LEN) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.midLength > UINT16_MAX || rowPitch > UINT16_MAX || splitSize64 > UINT16_MAX ||
        rowPitch < splitSize64) {
        return ge::GRAPH_FAILED;
    }

    OuterTilePlan outerPlan;
    if (!CalcOuterTilePlan(runtimeInfo.outerLength, OUTER_TILE_FIXED, outerPlan)) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t logicalTaskNum64 = static_cast<uint64_t>(outerPlan.outerTileNum) * colChunkNum;
    if (logicalTaskNum64 == 0 || logicalTaskNum64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t logicalTaskNum = static_cast<uint32_t>(logicalTaskNum64);

    const uint32_t maxCoreNum = std::max<uint32_t>(runtimeInfo.coreNum, 1);
    const uint32_t blockFactor = static_cast<uint32_t>(CeilDiv(logicalTaskNum, maxCoreNum));
    const uint32_t needCore = static_cast<uint32_t>(CeilDiv(logicalTaskNum, blockFactor));
    const uint32_t tailOuterTileNum = logicalTaskNum -
                                      static_cast<uint32_t>(static_cast<uint64_t>(needCore - 1) * blockFactor);
    const uint32_t formerNum = tailOuterTileNum == blockFactor ? needCore : needCore - 1;
    const OuterTaskSplitPlan taskPlan{needCore, blockFactor, tailOuterTileNum, formerNum};

    SplitVTilingDataSameLenCompact* tiling = context->GetTilingData<SplitVTilingDataSameLenCompact>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillSameLenCompactFields(
        tiling, runtimeInfo.totalLength, runtimeInfo.outerLength, static_cast<uint32_t>(runtimeInfo.midLength),
        static_cast<uint32_t>(splitSize64), static_cast<uint32_t>(tailSplitSize64), runtimeInfo.splitNum,
        OUTER_TILE_FIXED, outerPlan, taskPlan, static_cast<uint32_t>(SAME_LEN_COMPACT_B16_ADDR_COUNT * rowPitch),
        static_cast<uint32_t>(SAME_LEN_COMPACT_B16_ADDR_COUNT * splitSize64), chunkSplitNum, colChunkNum);

    context->SetBlockDim(needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_SAME_LEN_COMPACT);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForSameLenCompact8Bit(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    static constexpr uint32_t B8_COMPACT_OUTER_TILE = 512;
    static constexpr uint32_t B8_FULL_ROW_256_OUTER_TILE = 256;
    static constexpr uint32_t B8_FULL_ROW_256_GROUP_ROWS = 8;
    static constexpr uint32_t B8_FULL_ROW_512_GROUP_ROWS = 16;
    static constexpr uint32_t B8_ADDR_COUNT = 16;
    static constexpr uint32_t B8_ELEMS_PER_DB = 32;
    static constexpr uint32_t B8_ROW_LENGTH_LIMIT = 256;
    static constexpr uint32_t B8_CHUNK_COL_LIMIT = 128;

    SameLenSplitInfo splitInfo;
    if (runtimeInfo.innerLength != 1 || !GetSameLenSplitInfo(runtimeInfo, splitInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!IsSameLenCompact8DataType(runtimeInfo.dataType)) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t splitSize64 = static_cast<uint64_t>(splitInfo.splitSize);
    const uint64_t tailSplitSize64 = static_cast<uint64_t>(splitInfo.tailSplitSize);
    if (splitSize64 == 0 || splitSize64 > runtimeInfo.midLength || splitSize64 > UINT32_MAX ||
        runtimeInfo.midLength == 0 || runtimeInfo.midLength > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    if (tailSplitSize64 == 0 || tailSplitSize64 > splitSize64 || tailSplitSize64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    if (splitSize64 > MAX_TRANS_SPLIT_TILE) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.splitNum <= 1 || runtimeInfo.splitNum > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    if (splitSize64 > UINT16_MAX || runtimeInfo.midLength - splitSize64 > UINT16_MAX) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t rowCopyBytes = runtimeInfo.midLength * runtimeInfo.dataTypeSize;
    const uint64_t splitCopyBytes = splitSize64 * runtimeInfo.dataTypeSize;
    if (rowCopyBytes == 0 || splitCopyBytes == 0 || rowCopyBytes > MAX_COPY_BYTES_8BIT ||
        splitCopyBytes > MAX_COPY_BYTES_8BIT) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t fullRowTransLen64 = AlignUpTo(B8_ADDR_COUNT * runtimeInfo.midLength, B8_ELEMS_PER_DB);
    const uint64_t fullSplitTransLen64 = AlignUpTo(B8_ADDR_COUNT * splitSize64, B8_ELEMS_PER_DB);
    const uint64_t fullInputStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * fullRowTransLen64 *
                                         runtimeInfo.dataTypeSize;
    const uint64_t fullTransStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * fullRowTransLen64 *
                                         runtimeInfo.dataTypeSize;
    const uint64_t fullOutputStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * fullSplitTransLen64 *
                                          runtimeInfo.dataTypeSize;
    const uint64_t fullNeedBytes = fullInputStageBytes + fullTransStageBytes + fullOutputStageBytes;
    const uint64_t fullCopyInBytes = static_cast<uint64_t>(B8_COMPACT_OUTER_TILE) * runtimeInfo.midLength *
                                     runtimeInfo.dataTypeSize;
    const uint64_t copyOutBytes = static_cast<uint64_t>(B8_COMPACT_OUTER_TILE) * splitCopyBytes;
    const bool enableFullRow = fullNeedBytes != 0 && fullNeedBytes <= runtimeInfo.effectiveUbNum &&
                               fullInputStageBytes <= UINT32_MAX && fullTransStageBytes <= UINT32_MAX &&
                               fullOutputStageBytes <= UINT32_MAX && fullCopyInBytes <= MAX_COPY_BYTES_8BIT &&
                               copyOutBytes <= MAX_COPY_BYTES_8BIT &&
                               fullRowTransLen64 / B8_ELEMS_PER_DB <= UINT8_MAX &&
                               fullSplitTransLen64 / B8_ELEMS_PER_DB <= UINT8_MAX && fullRowTransLen64 <= UINT32_MAX &&
                               fullSplitTransLen64 <= UINT32_MAX;

    const uint64_t full256RowTransLen64 = B8_FULL_ROW_256_GROUP_ROWS * runtimeInfo.midLength;
    const uint64_t full256SplitTransLen64 = B8_FULL_ROW_256_GROUP_ROWS * splitSize64;
    const uint64_t full256InputStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * full256RowTransLen64 *
                                            runtimeInfo.dataTypeSize;
    const uint64_t full256TransStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * full256RowTransLen64 *
                                            runtimeInfo.dataTypeSize;
    const uint64_t full256OutputStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * full256SplitTransLen64 *
                                             runtimeInfo.dataTypeSize;
    const uint64_t full256NeedBytes = full256InputStageBytes + full256TransStageBytes + full256OutputStageBytes;
    const uint64_t full256CopyInBytes = static_cast<uint64_t>(B8_FULL_ROW_256_OUTER_TILE) * runtimeInfo.midLength *
                                        runtimeInfo.dataTypeSize;
    const uint64_t full256CopyOutBytes = static_cast<uint64_t>(B8_FULL_ROW_256_OUTER_TILE) * splitCopyBytes;
    const uint64_t full256SplitRepeat = (full256SplitTransLen64 + B8_ELEMS_PER_DB - 1) / B8_ELEMS_PER_DB;
    const bool enableFullRow256 = runtimeInfo.midLength > B8_CHUNK_COL_LIMIT && runtimeInfo.midLength % 4 == 0 &&
                                  splitSize64 % 2 == 0 && full256NeedBytes != 0 &&
                                  full256NeedBytes <= runtimeInfo.effectiveUbNum &&
                                  full256InputStageBytes <= UINT32_MAX && full256TransStageBytes <= UINT32_MAX &&
                                  full256OutputStageBytes <= UINT32_MAX && full256CopyInBytes <= MAX_COPY_BYTES_8BIT &&
                                  full256CopyOutBytes <= MAX_COPY_BYTES_8BIT &&
                                  full256RowTransLen64 / B8_ELEMS_PER_DB <= UINT8_MAX &&
                                  full256SplitRepeat <= UINT8_MAX && full256RowTransLen64 <= UINT32_MAX &&
                                  full256SplitTransLen64 <= UINT32_MAX;

    const uint64_t full512ParityRowTransLen64 = B8_FULL_ROW_512_GROUP_ROWS * runtimeInfo.midLength;
    const uint64_t full512ParitySplitTransLen64 = B8_FULL_ROW_512_GROUP_ROWS * splitSize64;
    const uint64_t full512ParityInputStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * full512ParityRowTransLen64 *
                                                  runtimeInfo.dataTypeSize;
    const uint64_t full512ParityTransStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * full512ParityRowTransLen64 *
                                                  runtimeInfo.dataTypeSize;
    const uint64_t full512ParityOutputStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) *
                                                   full512ParitySplitTransLen64 * runtimeInfo.dataTypeSize;
    const uint64_t full512ParityNeedBytes = full512ParityInputStageBytes + full512ParityTransStageBytes +
                                            full512ParityOutputStageBytes;
    const uint64_t full512ParityRowRepeat = (runtimeInfo.midLength + 1) / 2;
    const uint64_t full512ParitySplitRepeat = (splitSize64 + 1) / 2;
    const bool enableFull512Parity = full512ParityNeedBytes != 0 &&
                                     full512ParityNeedBytes <= runtimeInfo.effectiveUbNum &&
                                     full512ParityInputStageBytes <= UINT32_MAX &&
                                     full512ParityTransStageBytes <= UINT32_MAX &&
                                     full512ParityOutputStageBytes <= UINT32_MAX &&
                                     fullCopyInBytes <= MAX_COPY_BYTES_8BIT && copyOutBytes <= MAX_COPY_BYTES_8BIT &&
                                     full512ParityRowRepeat <= UINT8_MAX && full512ParitySplitRepeat <= UINT8_MAX &&
                                     full512ParityRowTransLen64 <= UINT32_MAX &&
                                     full512ParitySplitTransLen64 <= UINT32_MAX;

    const uint64_t pureCopySplitPitch64 = AlignUpTo(splitSize64, B8_ELEMS_PER_DB);
    const uint64_t pureCopyRowTransLen64 = B8_ADDR_COUNT * splitSize64;
    const uint64_t pureCopySplitTransLen64 = B8_ADDR_COUNT * pureCopySplitPitch64;
    const uint64_t pureCopyInputStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * pureCopySplitTransLen64 *
                                             runtimeInfo.dataTypeSize;
    const uint64_t pureCopyTransStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * pureCopyRowTransLen64 *
                                             runtimeInfo.dataTypeSize;
    const uint64_t pureCopyOutputStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * pureCopySplitTransLen64 *
                                              runtimeInfo.dataTypeSize;
    const uint64_t pureCopyNeedBytes = pureCopyInputStageBytes + pureCopyTransStageBytes + pureCopyOutputStageBytes;
    const uint64_t pureCopyTileCopyBytes = static_cast<uint64_t>(B8_COMPACT_OUTER_TILE) * pureCopySplitPitch64 *
                                           runtimeInfo.dataTypeSize;
    const bool preferOddLargePureCopy = runtimeInfo.midLength > B8_ROW_LENGTH_LIMIT ||
                                        (runtimeInfo.midLength > B8_CHUNK_COL_LIMIT && runtimeInfo.midLength % 2 != 0 &&
                                         splitSize64 < 64);
    const bool enableOddLargePureCopy = false && preferOddLargePureCopy && pureCopyNeedBytes != 0 &&
                                        pureCopyNeedBytes <= runtimeInfo.effectiveUbNum &&
                                        pureCopyInputStageBytes <= UINT32_MAX &&
                                        pureCopyTransStageBytes <= UINT32_MAX &&
                                        pureCopyOutputStageBytes <= UINT32_MAX &&
                                        pureCopyTileCopyBytes <= MAX_COPY_BYTES_8BIT && splitSize64 <= UINT16_MAX &&
                                        runtimeInfo.midLength - splitSize64 <= UINT32_MAX &&
                                        pureCopyRowTransLen64 <= UINT32_MAX && pureCopySplitTransLen64 <= UINT32_MAX;

    uint64_t rowTransLen64 = fullRowTransLen64;
    uint64_t splitTransLen64 = fullSplitTransLen64;
    uint32_t chunkSplitNum = static_cast<uint32_t>(runtimeInfo.splitNum);
    uint32_t colChunkNum = 1;
    uint64_t inputStageBytes = fullInputStageBytes;
    uint64_t transStageBytes = fullTransStageBytes;
    uint64_t outputStageBytes = fullOutputStageBytes;

    if (enableFullRow256) {
        rowTransLen64 = full256RowTransLen64;
        splitTransLen64 = full256SplitTransLen64;
        inputStageBytes = full256InputStageBytes;
        transStageBytes = full256TransStageBytes;
        outputStageBytes = full256OutputStageBytes;
    } else if (enableFull512Parity) {
        rowTransLen64 = full512ParityRowTransLen64;
        splitTransLen64 = full512ParitySplitTransLen64;
        inputStageBytes = full512ParityInputStageBytes;
        transStageBytes = full512ParityTransStageBytes;
        outputStageBytes = full512ParityOutputStageBytes;
    } else if (enableOddLargePureCopy) {
        rowTransLen64 = pureCopyRowTransLen64;
        splitTransLen64 = pureCopySplitTransLen64;
        chunkSplitNum = 1;
        colChunkNum = static_cast<uint32_t>(runtimeInfo.splitNum);
        inputStageBytes = pureCopyInputStageBytes;
        transStageBytes = pureCopyTransStageBytes;
        outputStageBytes = pureCopyOutputStageBytes;
    } else if (!enableFullRow) {
        const uint64_t chunkSplitNum64 = std::max<uint64_t>(1, B8_CHUNK_COL_LIMIT / splitSize64);
        chunkSplitNum = static_cast<uint32_t>(
            std::min<uint64_t>(chunkSplitNum64, static_cast<uint64_t>(runtimeInfo.splitNum)));
        colChunkNum = static_cast<uint32_t>((static_cast<uint64_t>(runtimeInfo.splitNum) + chunkSplitNum - 1) /
                                            chunkSplitNum);
        const uint64_t chunkColsMax = static_cast<uint64_t>(chunkSplitNum) * splitSize64;
        const bool hasFullColChunks = static_cast<uint64_t>(chunkSplitNum) * colChunkNum ==
                                      static_cast<uint64_t>(runtimeInfo.splitNum);
        const uint64_t chunkParityRowTransLen64 = B8_FULL_ROW_512_GROUP_ROWS * chunkColsMax;
        const uint64_t chunkParitySplitTransLen64 = B8_FULL_ROW_512_GROUP_ROWS * splitSize64;
        const uint64_t chunkParityInputStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * chunkParityRowTransLen64 *
                                                    runtimeInfo.dataTypeSize;
        const uint64_t chunkParityTransStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) * chunkParityRowTransLen64 *
                                                    runtimeInfo.dataTypeSize;
        const uint64_t chunkParityOutputStageBytes = static_cast<uint64_t>(B8_ELEMS_PER_DB) *
                                                     chunkParitySplitTransLen64 * runtimeInfo.dataTypeSize;
        const uint64_t chunkParityNeedBytes = chunkParityInputStageBytes + chunkParityTransStageBytes +
                                              chunkParityOutputStageBytes;
        const bool enableChunk512Parity = hasFullColChunks && chunkColsMax % B8_ELEMS_PER_DB == 0 &&
                                          chunkParityNeedBytes != 0 &&
                                          chunkParityNeedBytes <= runtimeInfo.effectiveUbNum &&
                                          chunkParityInputStageBytes <= UINT32_MAX &&
                                          chunkParityTransStageBytes <= UINT32_MAX &&
                                          chunkParityOutputStageBytes <= UINT32_MAX &&
                                          (chunkColsMax + 1) / 2 <= UINT8_MAX && (splitSize64 + 1) / 2 <= UINT8_MAX &&
                                          chunkParityRowTransLen64 <= UINT32_MAX &&
                                          chunkParitySplitTransLen64 <= UINT32_MAX;
        if (enableChunk512Parity) {
            rowTransLen64 = chunkParityRowTransLen64;
            splitTransLen64 = chunkParitySplitTransLen64;
            inputStageBytes = chunkParityInputStageBytes;
            transStageBytes = chunkParityTransStageBytes;
            outputStageBytes = chunkParityOutputStageBytes;
        } else {
            return ge::GRAPH_FAILED;
        }
    }

    const uint64_t rowRepeat = (rowTransLen64 + B8_ELEMS_PER_DB - 1) / B8_ELEMS_PER_DB;
    const uint64_t splitRepeat = (splitTransLen64 + B8_ELEMS_PER_DB - 1) / B8_ELEMS_PER_DB;
    const uint64_t needBytes = inputStageBytes + transStageBytes + outputStageBytes;
    if (rowRepeat == 0 || splitRepeat == 0 || rowRepeat > UINT8_MAX || splitRepeat > UINT8_MAX ||
        rowTransLen64 > UINT32_MAX || splitTransLen64 > UINT32_MAX || needBytes == 0 ||
        needBytes > runtimeInfo.effectiveUbNum || inputStageBytes > UINT32_MAX || transStageBytes > UINT32_MAX ||
        outputStageBytes > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t outerTile = enableFullRow256 ? B8_FULL_ROW_256_OUTER_TILE : B8_COMPACT_OUTER_TILE;
    OuterTilePlan outerPlan;
    OuterTaskSplitPlan taskPlan;
    if (!CalcOuterTileTaskSplitPlan(runtimeInfo.outerLength, outerTile, colChunkNum, runtimeInfo.coreNum, outerPlan,
                                    taskPlan)) {
        return ge::GRAPH_FAILED;
    }

    SplitVTilingDataSameLenCompact* tiling = context->GetTilingData<SplitVTilingDataSameLenCompact>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillSameLenCompactFields(tiling, runtimeInfo.totalLength, runtimeInfo.outerLength,
                             static_cast<uint32_t>(runtimeInfo.midLength), static_cast<uint32_t>(splitSize64),
                             static_cast<uint32_t>(tailSplitSize64), runtimeInfo.splitNum, outerTile, outerPlan,
                             taskPlan, static_cast<uint32_t>(rowTransLen64), static_cast<uint32_t>(splitTransLen64),
                             chunkSplitNum, colChunkNum);

    context->SetBlockDim(taskPlan.needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_SAME_LEN_COMPACT_8BIT);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForSameLenPureCopy8Bit(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    static constexpr uint32_t PURE_COPY_MODE_SPLIT_MAJOR = 0;
    static constexpr uint32_t PURE_COPY_MODE_ROW_LENGTH_CHUNK = 1;

    SameLenSplitInfo splitInfo;
    if (runtimeInfo.innerLength != 1 || !GetSameLenSplitInfo(runtimeInfo, splitInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!IsSameLenCompact8DataType(runtimeInfo.dataType)) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.outerLength == 0 || runtimeInfo.outerLength > UINT32_MAX || runtimeInfo.midLength == 0 ||
        runtimeInfo.midLength > UINT32_MAX || runtimeInfo.splitNum <= 1 || runtimeInfo.splitNum > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t splitSize64 = splitInfo.splitSize;
    const uint64_t tailSplitSize64 = splitInfo.tailSplitSize;
    const uint64_t maxSplitSize64 = splitInfo.maxSplitSize;
    if (splitSize64 == 0 || splitSize64 > runtimeInfo.midLength || splitSize64 > UINT32_MAX || tailSplitSize64 == 0 ||
        tailSplitSize64 > UINT32_MAX || maxSplitSize64 == 0 || maxSplitSize64 > runtimeInfo.midLength ||
        maxSplitSize64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t splitBytes = maxSplitSize64 * runtimeInfo.dataTypeSize;
    const uint64_t splitPitchBytes = AlignUpTo(splitBytes, static_cast<uint64_t>(runtimeInfo.blockSize));
    const uint64_t splitPitch64 = splitPitchBytes / runtimeInfo.dataTypeSize;
    const uint64_t maxRightPadding = splitPitch64 - std::min<uint64_t>(splitSize64, tailSplitSize64);
    if (splitBytes == 0 || splitPitchBytes == 0 || splitPitch64 > UINT32_MAX || splitPitchBytes > UINT32_MAX ||
        maxRightPadding > UINT8_MAX) {
        return ge::GRAPH_FAILED;
    }

    CoreRowsPlan coreRows;
    if (!CalcCoreRowsPlan(runtimeInfo.outerLength, runtimeInfo.coreNum, coreRows)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t mode = PURE_COPY_MODE_SPLIT_MAJOR;
    uint32_t outerTile = 0;
    uint32_t colTileLength = static_cast<uint32_t>(maxSplitSize64);
    uint32_t colTilePitch = static_cast<uint32_t>(splitPitch64);

    const uint64_t maxRowsByUb = runtimeInfo.effectiveUbNum / (BUFFER_NUM * splitPitchBytes);
    const uint64_t srcStrideBytes = (runtimeInfo.midLength - maxSplitSize64) * runtimeInfo.dataTypeSize;
    const bool enableSplitMajor = maxRowsByUb != 0 && splitBytes <= MAX_DATA_COPY_EXT_BLOCK_LEN &&
                                  srcStrideBytes <= UINT32_MAX && coreRows.maxCoreRows != 0;

    if (enableSplitMajor) {
        uint64_t outerTile64 = std::min<uint64_t>(maxRowsByUb, static_cast<uint64_t>(coreRows.maxCoreRows));
        outerTile64 = std::min<uint64_t>(outerTile64, MAX_DATA_COPY_BLOCK_COUNT);
        outerTile = static_cast<uint32_t>(outerTile64);
    } else {
        mode = PURE_COPY_MODE_ROW_LENGTH_CHUNK;
        const uint64_t maxTileBytesByUb = runtimeInfo.effectiveUbNum / BUFFER_NUM;
        uint64_t colTileBytes = std::min<uint64_t>(maxTileBytesByUb, MAX_DATA_COPY_EXT_BLOCK_LEN);
        colTileBytes = (colTileBytes / runtimeInfo.blockSize) * runtimeInfo.blockSize;
        if (colTileBytes == 0) {
            return ge::GRAPH_FAILED;
        }
        colTileLength = static_cast<uint32_t>(
            std::min<uint64_t>(maxSplitSize64, colTileBytes / runtimeInfo.dataTypeSize));
        if (colTileLength == 0) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t colTilePitchBytes = AlignUpTo(static_cast<uint64_t>(colTileLength) * runtimeInfo.dataTypeSize,
                                                     static_cast<uint64_t>(runtimeInfo.blockSize));
        if (colTilePitchBytes / runtimeInfo.dataTypeSize > UINT32_MAX || colTilePitchBytes > maxTileBytesByUb ||
            colTilePitchBytes / runtimeInfo.dataTypeSize - colTileLength > UINT8_MAX) {
            return ge::GRAPH_FAILED;
        }
        colTilePitch = static_cast<uint32_t>(colTilePitchBytes / runtimeInfo.dataTypeSize);
        outerTile = 1;
    }

    if (outerTile == 0) {
        return ge::GRAPH_FAILED;
    }

    SplitVTilingDataSameLenPureCopy8Bit* tiling = context->GetTilingData<SplitVTilingDataSameLenPureCopy8Bit>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillSameLenCompactRuntimeFields(tiling, runtimeInfo.totalLength, runtimeInfo.outerLength,
                                    static_cast<uint32_t>(runtimeInfo.midLength), static_cast<uint32_t>(splitSize64),
                                    static_cast<uint32_t>(tailSplitSize64), runtimeInfo.splitNum);
    tiling->splitPitch = static_cast<uint32_t>(splitPitch64);
    FillCoreRowsFields(tiling, coreRows);
    FillPureCopyTileFields(tiling, mode, outerTile, colTileLength, colTilePitch);

    context->SetBlockDim(coreRows.realCoreNum);
    context->SetTilingKey(TILING_KEY_SPLIT_V_SAME_LEN_PURE_COPY_8BIT);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForUnevenCompact(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    static constexpr uint32_t MODE_FULL_ROW = 0;
    static constexpr uint32_t MODE_ROW_CHUNK = 1;
    static constexpr uint32_t OUTER_TILE_FIXED = 256;
    static constexpr uint32_t B16_ADDR_COUNT = 16;

    if (runtimeInfo.innerLength != 1 || IsSplitSameLenWithTail(runtimeInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!IsSameLenCompactDataType(runtimeInfo.dataType)) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.midLength == 0 || runtimeInfo.midLength > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.splitNum <= 1 || runtimeInfo.splitNum > maxSplitNum) {
        return ge::GRAPH_FAILED;
    }

    UnevenSplitValidationResult splitResult;
    if (!ValidateUnevenSplits(runtimeInfo, splitResult) || runtimeInfo.outerLength == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t maxSplitSize = splitResult.maxSplitSize;

    uint32_t mode = UINT32_MAX;
    uint32_t outerTile = OUTER_TILE_FIXED;
    uint32_t rowPitch = 0;
    uint32_t colChunkSize = 0;
    uint32_t colChunkNum = 1;

    const auto isFullRowUbEnough = [&]() -> bool {
        if (runtimeInfo.midLength > UINT8_MAX || maxSplitSize > UINT8_MAX) {
            return false;
        }
        const uint64_t inputTileBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * runtimeInfo.midLength *
                                        runtimeInfo.dataTypeSize;
        const uint64_t outputTileBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * maxSplitSize *
                                         runtimeInfo.dataTypeSize;
        if (inputTileBytes == 0 || outputTileBytes == 0 || inputTileBytes > MAX_COPY_BYTES ||
            outputTileBytes > MAX_COPY_BYTES) {
            return false;
        }
        const uint64_t needUbBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) *
                                     (2 * runtimeInfo.midLength + 2 * static_cast<uint64_t>(maxSplitSize)) *
                                     runtimeInfo.dataTypeSize;
        return needUbBytes <= runtimeInfo.effectiveUbNum;
    };

    if (isFullRowUbEnough()) {
        mode = MODE_FULL_ROW;
        rowPitch = static_cast<uint32_t>(runtimeInfo.midLength);
    } else if (maxSplitSize <= UINT8_MAX) {
        const uint64_t splitPitch64 = AlignUpTo(maxSplitSize, B16_ADDR_COUNT);
        const uint64_t needUbBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * 3 * splitPitch64 *
                                     runtimeInfo.dataTypeSize;
        const uint64_t chunkCopyBytes = static_cast<uint64_t>(OUTER_TILE_FIXED) * splitPitch64 *
                                        runtimeInfo.dataTypeSize;
        if (splitPitch64 <= UINT8_MAX && needUbBytes <= runtimeInfo.effectiveUbNum &&
            chunkCopyBytes <= MAX_DATA_COPY_EXT_BLOCK_LEN) {
            mode = MODE_ROW_CHUNK;
            rowPitch = static_cast<uint32_t>(splitPitch64);
            colChunkSize = 0;
            colChunkNum = static_cast<uint32_t>(runtimeInfo.splitNum);
        }
    }

    if (rowPitch == 0 || rowPitch > UINT8_MAX) {
        return ge::GRAPH_FAILED;
    }
    if (rowPitch > UINT32_MAX / B16_ADDR_COUNT) {
        return ge::GRAPH_FAILED;
    }

    OuterTilePlan outerPlan;
    OuterTaskSplitPlan taskPlan;
    const uint32_t taskFactor = mode == MODE_ROW_CHUNK ? colChunkNum : 1U;
    if (!CalcOuterTileTaskSplitPlan(runtimeInfo.outerLength, outerTile, taskFactor, runtimeInfo.coreNum, outerPlan,
                                    taskPlan)) {
        return ge::GRAPH_FAILED;
    }

    SplitVTilingDataUnevenCompact* tiling = context->GetTilingData<SplitVTilingDataUnevenCompact>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillUnevenCompactTilingBase(tiling, runtimeInfo, outerTile, outerPlan, taskPlan, maxSplitSize);
    FillUnevenCompactModeFields(tiling, mode, B16_ADDR_COUNT * rowPitch, B16_ADDR_COUNT * rowPitch, 0, 0, colChunkSize,
                                colChunkNum);

    context->SetBlockDim(taskPlan.needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_UNEVEN_COMPACT);
    return SetWorkspaceZero(context);
}
static ge::graphStatus TilingForUnevenPureCopy16Bit(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    static constexpr uint32_t PURE_COPY_MODE_SPLIT_MAJOR = 0;
    static constexpr uint32_t PURE_COPY_MODE_ROW_LENGTH_CHUNK = 1;

    if (runtimeInfo.innerLength != 1 || IsSplitSameLenWithTail(runtimeInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!IsSameLenCompactDataType(runtimeInfo.dataType)) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.outerLength == 0 || runtimeInfo.outerLength > UINT32_MAX || runtimeInfo.midLength == 0 ||
        runtimeInfo.midLength > UINT32_MAX || runtimeInfo.splitNum <= 1 || runtimeInfo.splitNum > maxSplitNum) {
        return ge::GRAPH_FAILED;
    }

    UnevenSplitValidationResult splitResult;
    if (!ValidateUnevenSplits(runtimeInfo, splitResult, true)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t maxSplitSize = splitResult.maxSplitSize;
    const uint32_t minSplitSize = splitResult.minSplitSize;

    const uint64_t maxSplitBytes = static_cast<uint64_t>(maxSplitSize) * runtimeInfo.dataTypeSize;
    const uint64_t maxSplitPitchBytes = AlignUpTo(maxSplitBytes, static_cast<uint64_t>(runtimeInfo.blockSize));
    const uint64_t splitPitch64 = maxSplitPitchBytes / runtimeInfo.dataTypeSize;
    if (maxSplitPitchBytes == 0 || maxSplitPitchBytes > UINT32_MAX || splitPitch64 > UINT32_MAX ||
        splitPitch64 - maxSplitSize > UINT8_MAX) {
        return ge::GRAPH_FAILED;
    }

    CoreRowsPlan coreRows;
    if (!CalcCoreRowsPlan(runtimeInfo.outerLength, runtimeInfo.coreNum, coreRows)) {
        return ge::GRAPH_FAILED;
    }

    UnevenPureCopyTilePlan tilePlan;
    if (CalcUnevenPureCopyTilePlan(runtimeInfo, runtimeInfo.dataTypeSize, static_cast<uint32_t>(runtimeInfo.midLength),
                                   maxSplitSize, minSplitSize, maxSplitPitchBytes, splitPitch64, coreRows, true,
                                   PURE_COPY_MODE_SPLIT_MAJOR, PURE_COPY_MODE_ROW_LENGTH_CHUNK,
                                   tilePlan) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SplitVTilingDataUnevenPureCopy16Bit* tiling = context->GetTilingData<SplitVTilingDataUnevenPureCopy16Bit>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillUnevenCompactRuntimeFields(tiling, runtimeInfo.totalLength, runtimeInfo.outerLength,
                                   static_cast<uint32_t>(runtimeInfo.midLength), runtimeInfo.splitNum, maxSplitSize);
    FillUnevenSplitListsFromRuntime(tiling, runtimeInfo);
    tiling->splitPitch = static_cast<uint32_t>(splitPitch64);
    FillCoreRowsFields(tiling, coreRows);
    FillPureCopyTileFields(tiling, tilePlan);

    context->SetBlockDim(coreRows.realCoreNum);
    context->SetTilingKey(TILING_KEY_SPLIT_V_UNEVEN_PURE_COPY_16BIT);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForUnevenPureCopyWide(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    static constexpr uint32_t PURE_COPY_MODE_SPLIT_MAJOR = 0;
    static constexpr uint32_t PURE_COPY_MODE_ROW_LENGTH_CHUNK = 1;
    static constexpr uint32_t B16_DATA_SIZE = sizeof(uint16_t);

    if (runtimeInfo.innerLength != 1 || IsSplitSameLenWithTail(runtimeInfo) ||
        GetCompact32BitViewFactor(runtimeInfo) == 0 || runtimeInfo.outerLength == 0 ||
        runtimeInfo.outerLength > UINT32_MAX || runtimeInfo.splitNum <= 1 || runtimeInfo.splitNum > maxSplitNum) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t viewFactor = runtimeInfo.dataTypeSize / B16_DATA_SIZE;
    if (runtimeInfo.midLength > UINT32_MAX / viewFactor || runtimeInfo.totalLength > UINT64_MAX / viewFactor) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t rowLength = static_cast<uint32_t>(runtimeInfo.midLength * viewFactor);
    const uint64_t totalLength = runtimeInfo.totalLength * viewFactor;

    uint64_t splitSum = 0;
    uint32_t maxSplitSize = 0;
    uint32_t minSplitSize = UINT32_MAX;
    uint32_t viewSplits[maxSplitNum] = {};
    for (int64_t i = 0; i < runtimeInfo.splitNum; ++i) {
        const int64_t split = runtimeInfo.sizeSplits[i];
        if (split <= 0 || static_cast<uint64_t>(split) > UINT32_MAX / viewFactor) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t viewSplit = static_cast<uint32_t>(split) * viewFactor;
        const uint64_t splitBytes = static_cast<uint64_t>(viewSplit) * B16_DATA_SIZE;
        if (splitBytes == 0 || splitBytes > MAX_DATA_COPY_EXT_BLOCK_LEN) {
            return ge::GRAPH_FAILED;
        }
        viewSplits[i] = viewSplit;
        splitSum += viewSplit;
        maxSplitSize = std::max(maxSplitSize, viewSplit);
        minSplitSize = std::min(minSplitSize, viewSplit);
    }
    if (splitSum != rowLength || maxSplitSize == 0 || minSplitSize == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t maxSplitBytes = static_cast<uint64_t>(maxSplitSize) * B16_DATA_SIZE;
    const uint64_t maxSplitPitchBytes = AlignUpTo(maxSplitBytes, runtimeInfo.blockSize);
    const uint64_t splitPitch64 = maxSplitPitchBytes / B16_DATA_SIZE;
    if (maxSplitPitchBytes == 0 || maxSplitPitchBytes > UINT32_MAX || splitPitch64 > UINT32_MAX ||
        splitPitch64 - maxSplitSize > UINT8_MAX) {
        return ge::GRAPH_FAILED;
    }

    CoreRowsPlan coreRows;
    if (!CalcCoreRowsPlan(runtimeInfo.outerLength, runtimeInfo.coreNum, coreRows)) {
        return ge::GRAPH_FAILED;
    }

    UnevenPureCopyTilePlan tilePlan;
    if (CalcUnevenPureCopyTilePlan(runtimeInfo, B16_DATA_SIZE, rowLength, maxSplitSize, minSplitSize,
                                   maxSplitPitchBytes, splitPitch64, coreRows, false, PURE_COPY_MODE_SPLIT_MAJOR,
                                   PURE_COPY_MODE_ROW_LENGTH_CHUNK, tilePlan) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<SplitVTilingDataUnevenPureCopy16Bit>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillUnevenCompactRuntimeFields(tiling, totalLength, runtimeInfo.outerLength, rowLength, runtimeInfo.splitNum,
                                   maxSplitSize);
    FillUnevenSplitLists(tiling, runtimeInfo.splitNum, viewSplits);
    tiling->splitPitch = static_cast<uint32_t>(splitPitch64);
    FillCoreRowsFields(tiling, coreRows);
    FillPureCopyTileFields(tiling, tilePlan);

    context->SetBlockDim(coreRows.realCoreNum);
    context->SetTilingKey(TILING_KEY_SPLIT_V_UNEVEN_PURE_COPY_WIDE);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForUnevenCompact8Bit(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    static constexpr uint32_t MODE_PURE_COPY = 0;
    static constexpr uint32_t MODE_FULL_512_PARITY = 1;
    static constexpr uint32_t MODE_FULL_256_VIRTUAL = 2;
    static constexpr uint32_t MODE_CHUNK_512_PARITY = 3;
    static constexpr uint32_t OUTER_TILE_512 = 512;
    static constexpr uint32_t OUTER_TILE_256 = 256;
    static constexpr uint32_t GROUP_ROWS_512 = 16;
    static constexpr uint32_t GROUP_ROWS_256 = 8;
    static constexpr uint32_t CHUNK_COL_LIMIT = 128;

    if (runtimeInfo.innerLength != 1 || IsSplitSameLenWithTail(runtimeInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (!IsSameLenCompact8DataType(runtimeInfo.dataType)) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.midLength == 0 || runtimeInfo.midLength > UINT32_MAX || runtimeInfo.outerLength == 0 ||
        runtimeInfo.outerLength > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    if (runtimeInfo.splitNum <= 1 || runtimeInfo.splitNum > maxSplitNum) {
        return ge::GRAPH_FAILED;
    }

    UnevenSplitValidationResult splitResult;
    if (!ValidateUnevenSplits(runtimeInfo, splitResult)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t maxSplitSize = splitResult.maxSplitSize;

    const uint64_t rowCopyBytes = runtimeInfo.midLength * runtimeInfo.dataTypeSize;
    if (rowCopyBytes == 0 || rowCopyBytes > MAX_DATA_COPY_EXT_BLOCK_LEN) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t splitCopyBytes = static_cast<uint64_t>(maxSplitSize) * runtimeInfo.dataTypeSize;
    if (splitCopyBytes == 0 || splitCopyBytes > MAX_DATA_COPY_EXT_BLOCK_LEN) {
        return ge::GRAPH_FAILED;
    }

    uint32_t mode = MODE_PURE_COPY;
    uint32_t outerTile = 0;
    uint32_t rowTransLen = 0;
    uint32_t splitTransLen = 0;
    uint32_t virtualSplitSize = 0;
    uint32_t virtualSplitNum = 0;
    uint32_t colChunkSize = 0;
    uint32_t colChunkNum = 1;

    const uint64_t full512RowTransLen = static_cast<uint64_t>(GROUP_ROWS_512) * runtimeInfo.midLength;
    const uint64_t full512SplitTransLen = static_cast<uint64_t>(GROUP_ROWS_512) * maxSplitSize;
    const uint64_t full512NeedBytes = static_cast<uint64_t>(OUTER_TILE_512) *
                                      (2 * runtimeInfo.midLength + maxSplitSize) * runtimeInfo.dataTypeSize;
    const uint64_t full512CopyInBytes = static_cast<uint64_t>(OUTER_TILE_512) * rowCopyBytes;
    const uint64_t full512CopyOutBytes = static_cast<uint64_t>(OUTER_TILE_512) * splitCopyBytes;
    const bool enableFull512 = full512NeedBytes != 0 && full512NeedBytes <= runtimeInfo.effectiveUbNum &&
                               full512NeedBytes <= UINT32_MAX && full512RowTransLen <= UINT32_MAX &&
                               full512SplitTransLen <= UINT32_MAX && full512CopyInBytes <= MAX_COPY_BYTES_8BIT &&
                               full512CopyOutBytes <= MAX_COPY_BYTES_8BIT &&
                               (runtimeInfo.midLength + 1) / 2 <= UINT8_MAX && (maxSplitSize + 1) / 2 <= UINT8_MAX &&
                               runtimeInfo.midLength <= UINT16_MAX && maxSplitSize <= UINT16_MAX;
    if (enableFull512) {
        mode = MODE_FULL_512_PARITY;
        outerTile = OUTER_TILE_512;
        rowTransLen = static_cast<uint32_t>(full512RowTransLen);
        splitTransLen = static_cast<uint32_t>(full512SplitTransLen);
        colChunkNum = 1;
    }

    if (false && mode == MODE_PURE_COPY && runtimeInfo.midLength > CHUNK_COL_LIMIT && runtimeInfo.midLength % 4 == 0 &&
        runtimeInfo.midLength <= UINT16_MAX) {
        uint32_t bestVirtualSplit = 0;
        uint32_t bestVirtualNum = 0;
        uint32_t bestDiff = UINT32_MAX;
        for (uint32_t candidate = 16; candidate <= CHUNK_COL_LIMIT; candidate += 2) {
            if (runtimeInfo.midLength % candidate != 0) {
                continue;
            }
            const uint32_t candidateNum = static_cast<uint32_t>(runtimeInfo.midLength / candidate);
            if (candidateNum == 0 || candidateNum > 60) {
                continue;
            }
            const uint32_t diff = candidate > 64 ? candidate - 64 : 64 - candidate;
            if (bestVirtualSplit == 0 || diff < bestDiff || (diff == bestDiff && candidateNum < bestVirtualNum)) {
                bestVirtualSplit = candidate;
                bestVirtualNum = candidateNum;
                bestDiff = diff;
            }
        }
        if (bestVirtualSplit != 0) {
            const uint64_t full256RowTransLen = static_cast<uint64_t>(GROUP_ROWS_256) * runtimeInfo.midLength;
            const uint64_t full256SplitTransLen = static_cast<uint64_t>(GROUP_ROWS_256) * bestVirtualSplit;
            const uint64_t full256NeedBytes = static_cast<uint64_t>(OUTER_TILE_256) *
                                              (2 * runtimeInfo.midLength + bestVirtualSplit) * runtimeInfo.dataTypeSize;
            const uint64_t full256CopyInBytes = static_cast<uint64_t>(OUTER_TILE_256) * rowCopyBytes;
            const uint64_t full256CopyOutBytes = static_cast<uint64_t>(OUTER_TILE_256) * bestVirtualSplit *
                                                 runtimeInfo.dataTypeSize;
            if (full256NeedBytes != 0 && full256NeedBytes <= runtimeInfo.effectiveUbNum &&
                full256NeedBytes <= UINT32_MAX && full256RowTransLen <= UINT32_MAX &&
                full256SplitTransLen <= UINT32_MAX && full256CopyInBytes <= MAX_COPY_BYTES_8BIT &&
                full256CopyOutBytes <= MAX_COPY_BYTES_8BIT && runtimeInfo.midLength / 4 <= UINT8_MAX &&
                (bestVirtualSplit + 3) / 4 <= UINT8_MAX) {
                mode = MODE_FULL_256_VIRTUAL;
                outerTile = OUTER_TILE_256;
                rowTransLen = static_cast<uint32_t>(full256RowTransLen);
                splitTransLen = static_cast<uint32_t>(full256SplitTransLen);
                virtualSplitSize = bestVirtualSplit;
                virtualSplitNum = bestVirtualNum;
                colChunkNum = 1;
            }
        }
    }

    if (mode == MODE_PURE_COPY) {
        const uint32_t chunkCandidate = static_cast<uint32_t>(
            AlignUpTo(static_cast<uint64_t>(maxSplitSize) * runtimeInfo.dataTypeSize, runtimeInfo.blockSize) /
            runtimeInfo.dataTypeSize);
        const uint64_t chunk512RowTransLen = static_cast<uint64_t>(GROUP_ROWS_512) * chunkCandidate;
        const uint64_t chunk512SplitTransLen = static_cast<uint64_t>(GROUP_ROWS_512) * maxSplitSize;
        const uint64_t chunk512NeedBytes = static_cast<uint64_t>(OUTER_TILE_512) * (2 * chunkCandidate + maxSplitSize) *
                                           runtimeInfo.dataTypeSize;
        if (chunkCandidate != 0 && chunkCandidate <= CHUNK_COL_LIMIT &&
            chunk512NeedBytes <= runtimeInfo.effectiveUbNum && chunk512NeedBytes <= UINT32_MAX &&
            chunk512RowTransLen <= UINT32_MAX && chunk512SplitTransLen <= UINT32_MAX && chunkCandidate <= UINT16_MAX &&
            maxSplitSize <= UINT16_MAX && (chunkCandidate + 1) / 2 <= UINT8_MAX &&
            (maxSplitSize + 1) / 2 <= UINT8_MAX) {
            mode = MODE_CHUNK_512_PARITY;
            outerTile = OUTER_TILE_512;
            rowTransLen = static_cast<uint32_t>(chunk512RowTransLen);
            splitTransLen = static_cast<uint32_t>(chunk512SplitTransLen);
            colChunkSize = chunkCandidate;
            colChunkNum = static_cast<uint32_t>(runtimeInfo.splitNum);
        }
    }

    if (mode == MODE_PURE_COPY) {
        const uint64_t maxSplitAlignedBytes = AlignUpTo(splitCopyBytes, runtimeInfo.blockSize);
        if (maxSplitAlignedBytes == 0 || maxSplitAlignedBytes > UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }
        if (runtimeInfo.effectiveUbNum <= runtimeInfo.blockSize) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t maxRowsByUb = (runtimeInfo.effectiveUbNum - runtimeInfo.blockSize) / (2 * maxSplitAlignedBytes);
        if (maxRowsByUb == 0) {
            return ge::GRAPH_FAILED;
        }
        uint64_t outerTile64 = std::min<uint64_t>(maxRowsByUb, runtimeInfo.outerLength);
        outerTile64 = std::min<uint64_t>(outerTile64, MAX_DATA_COPY_BLOCK_COUNT);
        if (outerTile64 == 0 || outerTile64 > UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }
        outerTile = static_cast<uint32_t>(outerTile64);
        rowTransLen = 0;
        splitTransLen = 0;
        virtualSplitSize = 0;
        virtualSplitNum = 0;
        colChunkSize = 0;
        colChunkNum = 1;
    }

    if (outerTile == 0 || colChunkNum == 0) {
        return ge::GRAPH_FAILED;
    }

    OuterTilePlan outerPlan;
    OuterTaskSplitPlan taskPlan;
    const uint32_t taskFactor = mode == MODE_CHUNK_512_PARITY ? colChunkNum : 1U;
    if (!CalcOuterTileTaskSplitPlan(runtimeInfo.outerLength, outerTile, taskFactor, runtimeInfo.coreNum, outerPlan,
                                    taskPlan)) {
        return ge::GRAPH_FAILED;
    }

    SplitVTilingDataUnevenCompact* tiling = context->GetTilingData<SplitVTilingDataUnevenCompact>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    FillUnevenCompactTilingBase(tiling, runtimeInfo, outerTile, outerPlan, taskPlan, maxSplitSize);
    FillUnevenCompactModeFields(tiling, mode, rowTransLen, splitTransLen, virtualSplitSize, virtualSplitNum,
                                colChunkSize, colChunkNum);

    context->SetBlockDim(taskPlan.needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_UNEVEN_COMPACT_8BIT);
    return SetWorkspaceZero(context);
}

static ge::graphStatus TilingForSameLen(gert::TilingContext* context, const RuntimeInfo& runtimeInfo)
{
    // Same-len path also accepts a different tail split when all previous splits share one size.
    if (runtimeInfo.innerLength != 1) {
        return ge::GRAPH_FAILED;
    }
    SameLenSplitInfo splitInfo;
    if (!GetSameLenSplitInfo(runtimeInfo, splitInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (splitInfo.splitSize > UINT32_MAX || splitInfo.tailSplitSize > UINT32_MAX ||
        splitInfo.maxSplitSize > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t splitSize = static_cast<uint32_t>(splitInfo.splitSize);
    const uint32_t tailSplitSize = static_cast<uint32_t>(splitInfo.tailSplitSize);
    const uint32_t maxSplitSize = static_cast<uint32_t>(splitInfo.maxSplitSize);
    if (splitSize == 0 || tailSplitSize == 0 || maxSplitSize == 0) {
        return ge::GRAPH_FAILED;
    }
    const bool hasDifferentTail = splitSize != tailSplitSize;

    const uint64_t splitAligned = ((static_cast<uint64_t>(maxSplitSize) + runtimeInfo.alignedNum - 1) /
                                   runtimeInfo.alignedNum) *
                                  runtimeInfo.alignedNum;
    const uint64_t rowAligned = ((runtimeInfo.midLength + runtimeInfo.alignedNum - 1) / runtimeInfo.alignedNum) *
                                runtimeInfo.alignedNum;

    const bool preferLargeDmaB16 = runtimeInfo.dataTypeSize == sizeof(uint16_t) && !hasDifferentTail &&
                                   splitSize > MAX_TRANS_SPLIT_TILE &&
                                   runtimeInfo.midLength % runtimeInfo.alignedNum == 0 &&
                                   splitSize % runtimeInfo.alignedNum == 0;
    bool useFullRowDma = false;

    // b16 same-len path uses TransDataTo5HD, equivalent to TBE scatter_vnchwconv_b16.
    const uint64_t vnchwNeedBytes = (VNCHW_TILE_SIDE * rowAligned + rowAligned * VNCHW_TILE_SIDE +
                                     VNCHW_TILE_SIDE * splitAligned) *
                                    runtimeInfo.dataTypeSize;
    const bool enableVnchwB16 = runtimeInfo.dataTypeSize == sizeof(uint16_t) && !hasDifferentTail &&
                                splitSize <= MAX_TRANS_SPLIT_TILE && rowAligned / VNCHW_TILE_SIDE <= UINT8_MAX &&
                                vnchwNeedBytes <= runtimeInfo.effectiveUbNum;

    uint32_t outerTile = VNCHW_TILE_SIDE;
    if (!enableVnchwB16) {
        // Fallback path compacts one output group at a time:
        // GM [outerTile, midLength] --2D stride copy--> UB [outerTile, splitSize].
        // TQueBind BUFFER_NUM=1 uses 2x UB (VECIN+VECOUT), and DataCopyPad keeps
        // each UB row 32B aligned when splitSize is not block-aligned.
        const uint64_t splitCopyBytes = static_cast<uint64_t>(maxSplitSize) * runtimeInfo.dataTypeSize;
        if (splitCopyBytes == 0 || splitCopyBytes > MAX_COPY_BYTES) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t maxRowsByUb = runtimeInfo.effectiveUbNum / (2 * splitAligned * runtimeInfo.dataTypeSize);
        outerTile = static_cast<uint32_t>(std::min<uint64_t>(maxRowsByUb, UINT32_MAX));
        if (preferLargeDmaB16) {
            const uint32_t preferredOuterTile = static_cast<uint32_t>(
                std::min<uint64_t>(LARGE_SAME_LEN_FULL_ROW_TILE, runtimeInfo.outerLength));
            const uint64_t preferredUbBytes = static_cast<uint64_t>(preferredOuterTile) * runtimeInfo.midLength *
                                              runtimeInfo.dataTypeSize;
            if (preferredOuterTile > 0 && preferredUbBytes <= runtimeInfo.ubNum + UB_RESERVED_BYTES) {
                outerTile = preferredOuterTile;
                useFullRowDma = true;
            }
        }
        outerTile = std::min(outerTile, MAX_DATA_COPY_BLOCK_COUNT);
        outerTile = static_cast<uint32_t>(std::min<uint64_t>(outerTile, runtimeInfo.outerLength));
    }
    if (outerTile == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t outerTileNum = static_cast<uint32_t>((runtimeInfo.outerLength + outerTile - 1) / outerTile);
    const uint32_t outerTail = static_cast<uint32_t>(runtimeInfo.outerLength -
                                                     static_cast<uint64_t>(outerTileNum - 1) * outerTile);

    uint32_t needCore = std::min(outerTileNum, runtimeInfo.coreNum);
    needCore = std::max<uint32_t>(1, needCore);

    uint32_t formerOuterTileNum = 0;
    uint32_t tailOuterTileNum = 0;
    uint32_t formerNum = 0;
    if (useFullRowDma) {
        if (needCore == 0) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t loopPerCore = (outerTileNum + needCore - 1) / needCore;
        if (loopPerCore == 0) {
            return ge::GRAPH_FAILED;
        }
        needCore = (outerTileNum + loopPerCore - 1) / loopPerCore;
        if (needCore == 0) {
            return ge::GRAPH_FAILED;
        }
        formerOuterTileNum = loopPerCore;
        tailOuterTileNum = static_cast<uint32_t>(static_cast<uint64_t>(outerTileNum) -
                                                 static_cast<uint64_t>(needCore - 1) * loopPerCore);
        formerNum = (tailOuterTileNum == formerOuterTileNum) ? needCore : (needCore - 1);
    } else {
        if (needCore == 0) {
            return ge::GRAPH_FAILED;
        }
        formerOuterTileNum = outerTileNum / needCore;
        tailOuterTileNum = outerTileNum / needCore;
        formerNum = outerTileNum % needCore;
        if (formerNum != 0) {
            ++formerOuterTileNum;
        }
    }

    const OuterTilePlan outerPlan{outerTileNum, outerTail};
    const OuterTaskSplitPlan taskPlan{needCore, formerOuterTileNum, tailOuterTileNum, formerNum};

    SplitVTilingDataSameLen* tiling = context->GetTilingData<SplitVTilingDataSameLen>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const uint32_t splitTileLength = useFullRowDma ? splitVSameLenFullRowDma : (enableVnchwB16 ? VNCHW_TILE_SIDE : 0);
    FillSameLenFields(tiling, runtimeInfo.totalLength, runtimeInfo.outerLength, runtimeInfo.midLength,
                      runtimeInfo.innerLength, splitSize, tailSplitSize, runtimeInfo.splitNum, outerTile, outerPlan,
                      taskPlan, splitTileLength);

    context->SetBlockDim(needCore);
    context->SetTilingKey(TILING_KEY_SPLIT_V_SAME_LEN);
    return SetWorkspaceZero(context);
}

// tiling dispatch entry
static ge::graphStatus SplitVTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    RuntimeInfo runtimeInfo;
    OP_CHECK_IF(GetRuntimeInfo(context, runtimeInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetRuntimeInfo failed"),
                return ge::GRAPH_FAILED);

    if (runtimeInfo.splitNum == 1) {
        return TilingForPureCopy(context, runtimeInfo);
    }

    if (TilingForOneRowPureCopy(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    if (runtimeInfo.innerLength == 1 && IsCompact32BitDataType(runtimeInfo.dataType)) {
        if (IsSplitSameLenWithTail(runtimeInfo)) {
            if (TilingForSameLenCompact32Bit(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
                return ge::GRAPH_SUCCESS;
            }
            if (TilingForSameLenPureCopyWide(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
                return ge::GRAPH_SUCCESS;
            }
        } else {
            if (TilingForUnevenCompact32Bit(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
                return ge::GRAPH_SUCCESS;
            }
            if (TilingForUnevenPureCopyWide(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
                return ge::GRAPH_SUCCESS;
            }
        }
        return TilingForGeneral(context, runtimeInfo);
    }

    if (runtimeInfo.outerLength == 1) {
        return TilingForOneOuter(context, runtimeInfo);
    }

    if (runtimeInfo.innerLength == 1) {
        if (IsSameLenCompact8DataType(runtimeInfo.dataType)) {
            if (TilingForSameLenCompact8Bit(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
                return ge::GRAPH_SUCCESS;
            }
            if (TilingForSameLenPureCopy8Bit(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
                return ge::GRAPH_SUCCESS;
            }
            if (TilingForUnevenCompact8Bit(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
                return ge::GRAPH_SUCCESS;
            }
            return TilingForGeneral(context, runtimeInfo);
        }

        // Same-length compact VNCHW path (TransDataTo5HD).
        if (TilingForSameLenCompactLargeOuter(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
        if (TilingForSameLenCompactDoubleBuffer(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
        if (TilingForSameLenCompact(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
        // Same-length DMA fallback.
        if (TilingForSameLen(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
        // Unequal B16 compact VNCHW path before the pure-copy/general fallback.
        if (TilingForUnevenCompact(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
        if (TilingForUnevenPureCopy16Bit(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
    } else {
        // Prefer packed DMA paths for innerLength > 1 before falling back to the general strategy.
        if (TilingForSameLenInnerCopy(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
        if (TilingForUnevenInnerAlignedMid(context, runtimeInfo) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
    }

    return TilingForGeneral(context, runtimeInfo);
}

static ge::graphStatus TilingParseForSplitV([[maybe_unused]] gert::TilingParseContext* context)
{
    // AscendC operators can return SUCCESS directly for better performance.
    return ge::GRAPH_SUCCESS;
}

// tiling register entry.
IMPL_OP_OPTILING(SplitV)
    .Tiling(SplitVTilingFunc, SPLIT_V_TILING_DATA_SIZE)
    .TilingParse<SplitVCompileInfo>(TilingParseForSplitV)
    .TilingInputsDataDependency({1, 2});
} // namespace optiling
