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
 * \file segsum_tiling_arch35.cpp
 * \brief arch35 / Ascend950 tiling implementation for Segsum.
 */
#include <algorithm>
#include "segsum_tiling_arch35.h"
#include "log/log.h"

namespace optiling {

namespace {
constexpr uint8_t BYTE_LEN_4 = 4;
constexpr uint8_t BYTE_LEN_2 = 2;
constexpr int32_t BLOCK = 32;
constexpr uint64_t RESERVED_UB = 1024;
constexpr uint32_t STRIPE_TILING_KEY = 0;
constexpr uint32_t ROW_BLOCK_TILING_KEY = 1;
} // namespace

uint8_t SegsumTilingArch35::GetDataTypeSize() const
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

ge::graphStatus SegsumTilingArch35::ParseInputAttrs()
{
    auto srcInputShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, srcInputShape);
    auto inputShape = srcInputShape->GetOriginShape();
    int32_t inputDim = static_cast<int32_t>(inputShape.GetDimNum());
    for (int32_t i = 0; i < inputDim - 1; i++) {
        batches *= inputShape.GetDim(i);
    }
    tailDimLength = inputShape.GetDim(inputDim - 1);

    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    dataType = inputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

void SegsumTilingArch35::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    averageBatches = CeilA2B(batches, static_cast<int64_t>(coreNumPlatform));
    needCoreNum = static_cast<uint32_t>(CeilA2B(batches, averageBatches));
}

void SegsumTilingArch35::GetTilingKey(uint64_t ubSizePlatform)
{
    const int64_t dataTypeSize = static_cast<int64_t>(GetDataTypeSize());
    const int64_t blockElement = BLOCK / dataTypeSize;
    const bool isFloat = (dataTypeSize == BYTE_LEN_4);
    rowLen = CeilA2B(tailDimLength, blockElement) * blockElement;

    const int64_t ubAvailable = static_cast<int64_t>(ubSizePlatform - RESERVED_UB);
    // TilingKey 1: x row (plus its fp32 view for non-float dtypes) and the fp32 carry are resident,
    // each extra row costs one fp32 accumulator plus its output buffer (for float the exp buffer
    // doubles as the output buffer).
    const int64_t fixedBytes = rowLen * (dataTypeSize + BYTE_LEN_4 + (isFloat ? 0 : BYTE_LEN_4));
    const int64_t bytesPerRow = rowLen * (BYTE_LEN_4 + BYTE_LEN_4 + (isFloat ? 0 : dataTypeSize));
    if (ubAvailable > fixedBytes && bytesPerRow > 0) {
        rowNum = (ubAvailable - fixedBytes) / bytesPerRow;
    } else {
        rowNum = 0;
    }
    if (rowNum >= 1) {
        rowNum = std::min(rowNum, tailDimLength);
        stripeLen = 0;
        tilingKey = ROW_BLOCK_TILING_KEY;
        return;
    }

    // TilingKey 0: a single row does not fit, split the columns into stripes; every stripe owns
    // its own fp32 carry, and x is streamed in fixed chunks because only one scalar is needed.
    const int64_t chunkBytes = SEGSUM_X_CHUNK_ARCH35 * (dataTypeSize + (isFloat ? 0 : BYTE_LEN_4));
    const int64_t bytesPerColumn = BYTE_LEN_4 + BYTE_LEN_4 + BYTE_LEN_4 + (isFloat ? 0 : dataTypeSize);
    stripeLen = (ubAvailable - chunkBytes) / bytesPerColumn / blockElement * blockElement;
    stripeLen = std::max(stripeLen, blockElement);
    stripeLen = std::min(stripeLen, rowLen);
    rowNum = 0;
    tilingKey = STRIPE_TILING_KEY;
}

void SegsumTilingArch35::FillTilingData(SegsumTilingDataArch35* tiling) const
{
    tiling->needCoreNum = static_cast<int64_t>(needCoreNum);
    tiling->batches = batches;
    tiling->tailDimLength = tailDimLength;
    tiling->averageBatches = averageBatches;
    tiling->rowLen = rowLen;
    tiling->rowNum = rowNum;
    tiling->stripeLen = stripeLen;
}

ge::graphStatus SegsumTilingArch35::RunTiling()
{
    OP_CHECK_IF(ParseInputAttrs() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "Segsum arch35 ParseInputAttrs failed"),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos* platformInfoPtr = tilingContext->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t coreNumPlatform = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (static_cast<int32_t>(coreNumPlatform) <= 0),
        OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "Segsum arch35 get coreNum failed: %u", coreNumPlatform),
        return ge::GRAPH_FAILED);
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    OP_CHECK_IF(
        (ubSizePlatform <= RESERVED_UB),
        OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "Segsum arch35 get ubSize failed: %lu", ubSizePlatform),
        return ge::GRAPH_FAILED);

    if (batches == 0 || tailDimLength == 0) {
        // Runtime requires a non-zero block dimension even when there is nothing to compute.
        needCoreNum = 1;
        averageBatches = 0;
        rowLen = 0;
        rowNum = 0;
        stripeLen = 0;
        tilingKey = ROW_BLOCK_TILING_KEY;
    } else {
        GetNeedCoreNum(coreNumPlatform);
        GetTilingKey(ubSizePlatform);
    }

    auto tiling = tilingContext->GetTilingData<SegsumTilingDataArch35>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    *tiling = SegsumTilingDataArch35{};
    FillTilingData(tiling);

    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(tilingKey);

    size_t sysWsSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, workspaces);
    workspaces[0] = sysWsSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4SegsumArch35(gert::TilingContext* context)
{
    SegsumTilingArch35 tilingObject(context);
    return tilingObject.RunTiling();
}

IMPL_OP_OPTILING(Segsum).Tiling(Tiling4SegsumArch35);
} // namespace optiling
