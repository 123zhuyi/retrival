/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file cdist_grad_tiling.cpp
 * \brief CdistGrad Tiling implementation (arch32)
 *
 * Supports all 4 p-value branches (p=1, p=2, p=inf, general p),
 * fp32, 2D/3D input, FullM mode.
 * R_tile computed based on UB budget; multi-core split along B*P.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/common/cdist_grad_tiling_data.h"
#include "../../op_kernel/common/cdist_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t WS_SYS_SIZE = 0U;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static inline int64_t AlignTo32Bytes(int64_t count, int64_t typeSize)
{
    if (typeSize == 0) {
        return count;
    }
    int64_t bytes = count * typeSize;
    int64_t alignedBytes = ((bytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    return alignedBytes / typeSize;
}

static ge::graphStatus CdistGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSizeU64;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSizeU64, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    int64_t ubSize = static_cast<int64_t>(ubSizeU64);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // Get p attribute (index=0, the only attr)
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* pAttrPtr = attrs->GetFloat(0);
    double pValue = 2.0;
    if (pAttrPtr != nullptr) {
        pValue = static_cast<double>(*pAttrPtr);
    }

    // Determine P_MODE
    uint32_t pModeInt = 1; // default p=2
    if (pValue == 1.0) {
        pModeInt = 0;
    } else if (pValue == 2.0) {
        pModeInt = 1;
    } else if (std::isinf(pValue)) {
        pModeInt = 2;
    } else {
        pModeInt = 3;
    }

    // Get input shapes
    // Input 0: grad_output (B, P, R) or (P, R)
    // Input 1: x1 (B, P, M) or (P, M)
    // Input 2: x2 (B, R, M) or (R, M)
    // Input 3: cdist_result (B, P, R) or (P, R)
    auto x1ShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1ShapePtr);
    auto x1Shape = x1ShapePtr->GetStorageShape();

    auto x2ShapePtr = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2ShapePtr);
    auto x2Shape = x2ShapePtr->GetStorageShape();

    int64_t x1Rank = static_cast<int64_t>(x1Shape.GetDimNum());

    int64_t B, P, R, M;
    if (x1Rank == 3) {
        B = x1Shape.GetDim(0);
        P = x1Shape.GetDim(1);
        M = x1Shape.GetDim(2);
        R = x2Shape.GetDim(1);
    } else {
        // 2D: treat as B=1
        B = 1;
        P = x1Shape.GetDim(0);
        M = x1Shape.GetDim(1);
        R = x2Shape.GetDim(0);
    }

    OP_LOGI(context, "CdistGrad Tiling: B=%ld P=%ld R=%ld M=%ld pValue=%.2f pMode=%u", B, P, R, M, pValue, pModeInt);

    // Determine dtype and type size
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType inputDType = inputDesc->GetDataType();
    bool isFp16 = (inputDType == ge::DT_FLOAT16);
    int64_t inputTypeSize = isFp16 ? 2 : static_cast<int64_t>(sizeof(float));

    // Compute aligned sizes (always align to 32 bytes)
    // mAligned is in fp32 element count (computation precision)
    int64_t computeTypeSize = static_cast<int64_t>(sizeof(float));
    int64_t mAligned = AlignTo32Bytes(M, computeTypeSize);

    // Mask buffer: bit-packed, ceil(mAligned/8) rounded up to 32 bytes
    int64_t maskBufSize = ((mAligned / 8 + 31) / 32) * 32;
    if (maskBufSize < 32)
        maskBufSize = 32;

    // tmpReduceBufSize: minimum 4096 bytes
    int64_t tmpReduceBufSize = 4096;

    // Fixed buffer overhead (all compute in fp32):
    // x1RowBuf + accumBuf + diffBuf + localGradBuf + tmpBuf + distBroadcastBuf = 6 buffers
    // For general p (max): 8 buffers (add absDiffBuf + lnBuf)
    // Use 8 buffers to be safe for all p modes
    int64_t numFixedBufs = 8;
    int64_t fixedBytes = mAligned * computeTypeSize * numFixedBufs + maskBufSize + tmpReduceBufSize;

    // For fp16: add a cast buffer for CopyIn/CopyOut (one row at a time)
    // castBuf size = mAligned * sizeof(half), aligned to 32 bytes
    if (isFp16) {
        int64_t castBufBytes = AlignTo32Bytes(mAligned, inputTypeSize) * inputTypeSize;
        if (castBufBytes < BLOCK_SIZE)
            castBufBytes = BLOCK_SIZE;
        fixedBytes += castBufBytes;
    }

    // Per R_tile row: x2 one row in fp32 (M_aligned * 4) + grad one scalar (4) + dist one scalar (4)
    // But grad and dist are allocated as rTileAligned blocks
    int64_t perRowBytes = mAligned * computeTypeSize + 2 * computeTypeSize;

    // Calculate R_tile
    int64_t rTile = 1;
    if (ubSize > fixedBytes) {
        rTile = (ubSize - fixedBytes) / perRowBytes;
    }
    if (rTile < 1)
        rTile = 1;
    if (rTile > R)
        rTile = R;

    // Check if M is too large for UB: even with rTile=1, total buffer must fit in UB
    int64_t minRequiredBytes = fixedBytes + perRowBytes; // minimum: fixed buffers + 1 R-tile row
    if (minRequiredBytes > ubSize) {
        OP_LOGW(
            context,
            "CdistGrad: M=%ld (mAligned=%ld) exceeds UB capacity. "
            "Required %ld bytes > UB %ld bytes. Results may be incorrect.",
            M, mAligned, minRequiredBytes, ubSize);
    }

    // Align rTile for 32-byte alignment (fp32 elements)
    int64_t rTileAligned = AlignTo32Bytes(rTile, computeTypeSize);

    // Number of R chunks
    int64_t numRChunks = CeilDiv(R, rTile);
    int64_t lastRChunkSize = R - (numRChunks - 1) * rTile;
    if (lastRChunkSize <= 0)
        lastRChunkSize = rTile;

    // Multi-core split along B*P
    int64_t totalTasks = B * P;
    int64_t tasksPerCore = CeilDiv(totalTasks, coreNum);
    int64_t usedCoreNum = CeilDiv(totalTasks, tasksPerCore);
    int64_t tailCoreTasks = totalTasks - (usedCoreNum - 1) * tasksPerCore;

    OP_LOGI(
        context, "CdistGrad: mAligned=%ld rTile=%ld rTileAligned=%ld numRChunks=%ld usedCoreNum=%ld", mAligned, rTile,
        rTileAligned, numRChunks, usedCoreNum);

    // Fill TilingData
    CdistGradTilingData* tiling = context->GetTilingData<CdistGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(CdistGradTilingData), 0, sizeof(CdistGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->batchSize = B;
    tiling->pSize = P;
    tiling->rSize = R;
    tiling->mSize = M;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->tailCoreTasks = tailCoreTasks;
    tiling->mAligned = mAligned;
    tiling->rTile = rTile;
    tiling->numRChunks = numRChunks;
    tiling->lastRChunkSize = lastRChunkSize;
    tiling->pModeInt = static_cast<int64_t>(pModeInt);
    tiling->pValue = pValue;
    tiling->pValueF = static_cast<float>(pValue);
    tiling->rTileAligned = rTileAligned;
    tiling->maskBufSize = maskBufSize;
    tiling->tmpReduceBufSize = tmpReduceBufSize;

    context->SetBlockDim(usedCoreNum);

    // SCH_MODE = 0 (FullM)
    uint32_t schMode = 0;
    uint32_t dType = static_cast<uint32_t>(inputDType);
    ASCENDC_TPL_SEL_PARAM(context, dType, pModeInt, schMode);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCdistGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct CdistGradCompileInfo {};

IMPL_OP_OPTILING(CdistGrad).Tiling(CdistGradTilingFunc).TilingParse<CdistGradCompileInfo>(TilingParseForCdistGrad);

} // namespace optiling
