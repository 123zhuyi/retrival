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
 * \file cdist_grad_tiling_arch22.cpp
 * \brief CdistGrad AscendC kernel Tiling for asc910b (arch22)
 *
 * All inputs arrive as broadcast [B, P, Q, M] continuous tensors (prepared by aclnn
 * UnsqueezeNd + BroadcastTo). Supports all p values: p=0, p=1, p=2, p=inf, general p.
 * fp32 and fp16, FullM mode only.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "tiling/tiling_api.h"
#include "math/cdist_grad/op_kernel/cdist_grad_tiling_data_arch22.h"
#include "math/cdist_grad/op_kernel/arch22/cdist_grad_tiling_key_arch22.h"

namespace optiling {

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t FP32_BYTES = 4;
constexpr int64_t BITS_PER_BYTE = 8;
constexpr int64_t COMPARE_ALIGN = 64;
static inline int64_t RowAlign(bool isFp16) { return isFp16 ? BLOCK_SIZE / 2 : BLOCK_SIZE / FP32_BYTES; }
constexpr int64_t NUM_CHUNK_QUEUES = 3;         // x2 / grad / dist
constexpr int64_t NUM_CHUNK_QUEUES_BLOCKED = 4; // x1 / x2 / grad / dist
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t NUM_FIXED_ROWS = 5;
constexpr int64_t NUM_BATCH_ROWS = 7;
constexpr int64_t NUM_FP16_BATCH_ROWS = 3;
constexpr int64_t NUM_POW_SCRATCH_ROWS = 9;
constexpr int64_t NUM_MASKS = 2;
constexpr int64_t MAX_TILE_SOLVE_ROUNDS = 64;
constexpr int64_t Q_SPLIT_MIN_TASK_ELEMS = 16384;
constexpr int64_t MAX_REPEAT_TIMES = 255;
constexpr int64_t MAX_REPEAT_STRIDE_BLOCKS = 255;
constexpr int64_t MAX_MASK_FP32 = 64;
constexpr int64_t VECTOR_FILL_ELEMS = 2048;
constexpr int64_t UB_SAFETY_MARGIN = 16384;

// Attr p → P_MODE
constexpr uint32_t P_MODE_P1 = 0;
constexpr uint32_t P_MODE_P2 = 1;
constexpr uint32_t P_MODE_PINF = 2;
constexpr uint32_t P_MODE_PGENERAL = 3;
constexpr uint32_t P_MODE_P0 = 4;

struct PModeEntry {
    float pValue;
    uint32_t pMode;
};

constexpr PModeEntry P_MODE_TABLE[] = {
    {0.0f, P_MODE_P0},
    {1.0f, P_MODE_P1},
    {2.0f, P_MODE_P2},
};

// Shape after aclnn broadcast: grad storage = [B, P, Q, M].
struct CdistGradShapeInfo {
    int64_t batchSize; // B = product of leading dims
    int64_t pSize;     // P = dim-3
    int64_t rSize;     // Q = dim-2
    int64_t mSize;     // M = dim-1
    int64_t dimNum;    // storage shape rank
};

// Result of the joint mTile/rTile solve (M-tiling).
struct CdistGradTileInfo {
    int64_t mAligned;
    int64_t mTileSize;
    int64_t numMTiles;
    int64_t lastMTileSize;
    int64_t rTile;
    int64_t cTile; // compute batch width; equals rTile, or pTile*Q on the blocked path
    int64_t pTile; // output rows per task; 1 = un-blocked (see PBlockRows)
    int64_t numRChunks;
    int64_t lastRChunkSize;
    int64_t tmpBufSize;
};

// Multi-core split along B*P tasks, plus the optional Q-split.
struct CdistGradCoreInfo {
    int64_t qSplit;
    int64_t qPartSize;
    int64_t totalSubTasks;
    int64_t tasksPerCore;
    int64_t usedCoreNum;
    int64_t tailCoreTasks;
};

// Per-M-segment UB footprint, in bytes.
struct CdistGradSegBytes {
    int64_t tmpBytes;
    int64_t fixedBytes;
    int64_t perTileBytes;
};

static inline int64_t AlignUp(int64_t val, int64_t align) { return ((val + align - 1) / align) * align; }

static inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

static ge::graphStatus GetPlatformLimits(gert::TilingContext* context,
                                         platform_ascendc::PlatformAscendC& ascendcPlatform, int64_t& coreNum,
                                         int64_t& ubSize)
{
    coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSizeU64 = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeU64);
    ubSize = static_cast<int64_t>(ubSizeU64);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context->GetNodeName(), "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint32_t PValueToMode(float pValue)
{
    for (const auto& entry : P_MODE_TABLE) {
        if (pValue == entry.pValue) {
            return entry.pMode;
        }
    }
    return std::isinf(pValue) ? P_MODE_PINF : P_MODE_PGENERAL;
}

static ge::graphStatus GetPValueAttr(gert::TilingContext* context, float& pValue)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    pValue = 2.0f;
    if (attrs->GetAttrNum() > 0) {
        const float* pAttr = attrs->GetAttrPointer<float>(0);
        pValue = (pAttr == nullptr) ? 2.0f : *pAttr;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetInputDataType(gert::TilingContext* context, ge::DataType& inputDType)
{
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    inputDType = inputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseShapeInfo(gert::TilingContext* context, CdistGradShapeInfo& shape)
{
    auto gradShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    auto gradStorageShape = gradShape->GetStorageShape();
    int64_t dimNum = static_cast<int64_t>(gradStorageShape.GetDimNum());
    OP_CHECK_IF(dimNum < 3,
                OP_LOGE(context->GetNodeName(), "CdistGrad requires at least 3D broadcast input, got: %ld", dimNum),
                return ge::GRAPH_FAILED);

    shape.batchSize = 1;
    for (int64_t d = 0; d < dimNum - 3; d++) {
        shape.batchSize *= gradStorageShape.GetDim(d);
    }
    shape.pSize = gradStorageShape.GetDim(dimNum - 3);
    shape.rSize = gradStorageShape.GetDim(dimNum - 2);
    shape.mSize = gradStorageShape.GetDim(dimNum - 1);
    shape.dimNum = dimNum;
    return ge::GRAPH_SUCCESS;
}

// UB footprint of one M-segment of aligned width mTileAligned, split into the part that does
// not depend on the batch width and the part that scales with it.
static CdistGradSegBytes CalcSegBytes(int64_t mTileAligned, bool isFp16, bool isPGeneral)
{
    int64_t mTileBytes = mTileAligned * FP32_BYTES;
    int64_t inTypeBytes = isFp16 ? 2 : FP32_BYTES;
    CdistGradSegBytes bytes;
    int64_t slackBytes = COMPARE_ALIGN * (NUM_CHUNK_QUEUES * DOUBLE_BUFFER * inTypeBytes +
                                          (NUM_BATCH_ROWS + (isFp16 ? NUM_FP16_BATCH_ROWS : 0)) * FP32_BYTES) +
                         COMPARE_ALIGN * NUM_MASKS / BITS_PER_BYTE +
                         (isPGeneral ? NUM_POW_SCRATCH_ROWS * (COMPARE_ALIGN - 1) * FP32_BYTES : 0);
    bytes.fixedBytes = NUM_FIXED_ROWS * mTileBytes + slackBytes;
    // Per batch row: three double-buffered input queue slots (input dtype), the fp32 batch
    // rows, p-general's scratch pool, and the two Compare bitmaps.
    bytes.perTileBytes = NUM_CHUNK_QUEUES * DOUBLE_BUFFER * mTileAligned * inTypeBytes +
                         (NUM_BATCH_ROWS + (isFp16 ? NUM_FP16_BATCH_ROWS : 0)) * mTileBytes +
                         (isPGeneral ? NUM_POW_SCRATCH_ROWS * mTileBytes : 0) +
                         NUM_MASKS * (mTileAligned / BITS_PER_BYTE);
    return bytes;
}

static int64_t PBlockRows(int64_t ubSize, bool isFp16, bool isPGeneral, const CdistGradShapeInfo& shape,
                          const CdistGradTileInfo& tile, int64_t qSplit)
{
    const int64_t mAligned = tile.mAligned;
    const int64_t totalRows = shape.batchSize * shape.pSize;
    if (tile.numMTiles != 1 || qSplit != 1 || totalRows <= 1) {
        return 1;
    }
    if (mAligned > MAX_MASK_FP32 || mAligned % (BLOCK_SIZE / FP32_BYTES) != 0) {
        return 1;
    }
    if (shape.rSize * mAligned / (BLOCK_SIZE / FP32_BYTES) > MAX_REPEAT_STRIDE_BLOCKS) {
        return 1;
    }
    if (shape.rSize * mAligned >= VECTOR_FILL_ELEMS) {
        return 1; // one task already fills an instruction; blocking would only add complexity
    }

    // UB budget for the blocked layout, per (i,j) row of the [pTile*Q, mAligned] slab.
    const int64_t inTypeBytes = isFp16 ? 2 : FP32_BYTES;
    const int64_t mTileBytes = mAligned * FP32_BYTES;
    const int64_t fp16BatchRows = isFp16 ? NUM_FP16_BATCH_ROWS + 1 : 0; // +1: fp32 view of the x1 slab
    const int64_t perRowBytes = NUM_CHUNK_QUEUES_BLOCKED * DOUBLE_BUFFER * mAligned * inTypeBytes +
                                (NUM_BATCH_ROWS + fp16BatchRows) * mTileBytes +
                                (isPGeneral ? NUM_POW_SCRATCH_ROWS * mTileBytes : 0) +
                                NUM_MASKS * (mAligned / BITS_PER_BYTE);
    // Same constant slack as CalcSegBytes: every batch buffer, queue slot and mask carries
    // COMPARE_ALIGN elements past the last row.
    const int64_t slackBytes = COMPARE_ALIGN * (NUM_CHUNK_QUEUES_BLOCKED * DOUBLE_BUFFER * inTypeBytes +
                                                (NUM_BATCH_ROWS + fp16BatchRows) * FP32_BYTES) +
                               COMPARE_ALIGN * NUM_MASKS / BITS_PER_BYTE +
                               (isPGeneral ? NUM_POW_SCRATCH_ROWS * (COMPARE_ALIGN - 1) * FP32_BYTES : 0);
    const int64_t fixedBytes = NUM_FIXED_ROWS * mTileBytes + slackBytes;
    const int64_t avail = ubSize - fixedBytes - UB_SAFETY_MARGIN;
    if (avail <= 0) {
        return 1;
    }
    const int64_t perPRowBytes = shape.rSize * perRowBytes + 3 * mTileBytes;
    int64_t pTile = avail / perPRowBytes;
    if (pTile > MAX_REPEAT_TIMES) {
        pTile = MAX_REPEAT_TIMES;
    }
    if (pTile > totalRows) {
        pTile = totalRows;
    }
    return pTile >= 2 ? pTile : 1;
}

static void SolveMTileAndRTile(int64_t ubSize, bool isFp16, bool isPGeneral, const CdistGradShapeInfo& shape,
                               CdistGradTileInfo& tile)
{
    const int64_t rowAlign = RowAlign(isFp16);
    int64_t mTileSize = shape.mSize;
    int64_t rTile = 1;
    bool solved = false;
    for (int64_t round = 0; round < MAX_TILE_SOLVE_ROUNDS && !solved; round++) {
        CdistGradSegBytes bytes = CalcSegBytes(AlignUp(mTileSize, rowAlign), isFp16, isPGeneral);
        int64_t avail = ubSize - bytes.fixedBytes - UB_SAFETY_MARGIN;
        if (avail >= bytes.perTileBytes) {
            rTile = std::min(avail / bytes.perTileBytes, shape.rSize);
            solved = true;
        } else if (mTileSize <= rowAlign) {
            rTile = 1; // minimum footprint; always fits at the row-stride floor
            solved = true;
        } else {
            mTileSize = std::max(rowAlign, (mTileSize / 2 / rowAlign) * rowAlign);
        }
    }

    // mAligned: full-row element stride (workspace slot stride), aligned to the row stride.
    tile.mAligned = AlignUp(shape.mSize, rowAlign);
    tile.mTileSize = mTileSize;
    tile.rTile = std::max<int64_t>(rTile, 1);
    tile.cTile = tile.rTile;
    tile.pTile = 1;
    tile.tmpBufSize = isPGeneral ? NUM_POW_SCRATCH_ROWS *
                                       AlignUp(tile.cTile * AlignUp(mTileSize, rowAlign), COMPARE_ALIGN) * FP32_BYTES :
                                   BLOCK_SIZE;
    tile.numMTiles = CeilDiv(shape.mSize, tile.mTileSize);
    tile.lastMTileSize = shape.mSize - (tile.numMTiles - 1) * tile.mTileSize;
    tile.numRChunks = CeilDiv(shape.rSize, tile.rTile);
    tile.lastRChunkSize = shape.rSize - (tile.numRChunks - 1) * tile.rTile;
    if (tile.lastRChunkSize <= 0) {
        tile.lastRChunkSize = tile.rTile;
    }
}

static CdistGradCoreInfo SplitCores(const CdistGradShapeInfo& shape, int64_t coreNum, int64_t rowsPerTask)
{
    CdistGradCoreInfo core;
    int64_t totalTasks = CeilDiv(shape.batchSize * shape.pSize, rowsPerTask);
    core.qSplit = 1;
    if (rowsPerTask == 1 && totalTasks > 0 && totalTasks < coreNum && shape.rSize > 1 &&
        shape.rSize * shape.mSize >= Q_SPLIT_MIN_TASK_ELEMS) {
        core.qSplit = std::min(CeilDiv(coreNum, totalTasks), shape.rSize);
    }
    core.qPartSize = CeilDiv(shape.rSize, core.qSplit);
    core.totalSubTasks = totalTasks * core.qSplit;
    if (core.totalSubTasks <= 0) {
        core.tasksPerCore = 0;
        core.usedCoreNum = 1;
        core.tailCoreTasks = 0;
        return core;
    }
    core.tasksPerCore = CeilDiv(core.totalSubTasks, coreNum);
    core.usedCoreNum = CeilDiv(core.totalSubTasks, core.tasksPerCore);
    core.tailCoreTasks = core.totalSubTasks - (core.usedCoreNum - 1) * core.tasksPerCore;
    return core;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context,
                                        platform_ascendc::PlatformAscendC& ascendcPlatform,
                                        const CdistGradTileInfo& tile, const CdistGradCoreInfo& core)
{
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    size_t usrSize = (core.qSplit > 1) ? static_cast<size_t>(core.totalSubTasks * tile.mAligned * sizeof(float)) : 0;
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillTilingData(gert::TilingContext* context, const CdistGradShapeInfo& shape,
                                      const CdistGradTileInfo& tile, const CdistGradCoreInfo& core, float pValue)
{
    auto tiling = context->GetTilingData<CdistGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CdistGradTilingData), 0, sizeof(CdistGradTilingData)) != EOK,
                OP_LOGE(context->GetNodeName(), "memset_s tiling failed"), return ge::GRAPH_FAILED);

    tiling->batchSize = shape.batchSize;
    tiling->pSize = shape.pSize;
    tiling->rSize = shape.rSize;
    tiling->mSize = shape.mSize;
    tiling->mAligned = tile.mAligned;
    tiling->mTileSize = tile.mTileSize;
    tiling->numMTiles = tile.numMTiles;
    tiling->lastMTileSize = tile.lastMTileSize;
    tiling->rTile = tile.rTile;
    tiling->cTile = tile.cTile;
    tiling->pTile = tile.pTile;
    tiling->numRChunks = tile.numRChunks;
    tiling->lastRChunkSize = tile.lastRChunkSize;
    tiling->tasksPerCore = core.tasksPerCore;
    tiling->tailCoreTasks = core.tailCoreTasks;
    tiling->usedCoreNum = core.usedCoreNum;
    tiling->qSplit = core.qSplit;
    tiling->qPartSize = core.qPartSize;
    tiling->tmpBufSize = tile.tmpBufSize;
    tiling->pValueF = pValue;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CdistGradTilingFunc(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    if (GetPlatformLimits(context, ascendcPlatform, coreNum, ubSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    float pValue = 2.0f;
    if (GetPValueAttr(context, pValue) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint32_t pModeInt = PValueToMode(pValue);

    CdistGradShapeInfo shape;
    if (ParseShapeInfo(context, shape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context->GetNodeName(), "CdistGrad Tiling: B=%ld P=%ld Q=%ld M=%ld p=%.4f pMode=%u dimNum=%ld",
            shape.batchSize, shape.pSize, shape.rSize, shape.mSize, pValue, pModeInt, shape.dimNum);

    ge::DataType inputDType = ge::DT_FLOAT;
    if (GetInputDataType(context, inputDType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    CdistGradTileInfo tile;
    const bool isFp16 = (inputDType == ge::DT_FLOAT16);
    const bool isPGeneral = (pModeInt == P_MODE_PGENERAL);
    SolveMTileAndRTile(ubSize, isFp16, isPGeneral, shape, tile);
    CdistGradCoreInfo core = SplitCores(shape, coreNum, 1);

    tile.pTile = PBlockRows(ubSize, isFp16, isPGeneral, shape, tile, core.qSplit);
    if (tile.pTile > 1) {
        tile.rTile = tile.pTile * shape.rSize;
        tile.numRChunks = 1;
        tile.lastRChunkSize = tile.rTile;
        tile.cTile = tile.rTile;
        tile.tmpBufSize = isPGeneral ?
                              NUM_POW_SCRATCH_ROWS * AlignUp(tile.cTile * tile.mAligned, COMPARE_ALIGN) * FP32_BYTES :
                              BLOCK_SIZE;
        core = SplitCores(shape, coreNum, tile.pTile);
    }

    OP_LOGI(context->GetNodeName(),
            "UB=%ld M=%ld mAligned=%ld mTile=%ld numMTiles=%ld rTile=%ld cTile=%ld pTile=%ld numRChunks=%ld "
            "usedCoreNum=%ld tasksPerCore=%ld qSplit=%ld qPartSize=%ld tmpBufSize=%ld",
            ubSize, shape.mSize, tile.mAligned, tile.mTileSize, tile.numMTiles, tile.rTile, tile.cTile, tile.pTile,
            tile.numRChunks, core.usedCoreNum, core.tasksPerCore, core.qSplit, core.qPartSize, tile.tmpBufSize);

    if (SetWorkspaceSize(context, ascendcPlatform, tile, core) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (FillTilingData(context, shape, tile, core, pValue) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Q-split path: all cores participate in SyncAll-based two-phase reduce.
    if (core.qSplit > 1) {
        context->SetScheduleMode(1);
        context->SetBlockDim(coreNum);
    } else {
        context->SetBlockDim(core.usedCoreNum);
    }

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
