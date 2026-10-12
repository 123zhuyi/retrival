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
 * \file truncate_mod_tiling.cpp
 * \brief TruncateMod tiling: core split -> UB split -> tile split.
 *
 * element-wise op. schMode (tiling key) selects the compute dtype
 * (0 half, 1 float, 2 bfloat16). All dtypes are promoted to float on the UB
 * for computation, so每 tile 需要 3 个输入/输出队列 + 3 个 float 计算缓冲 + 1 个
 * mask 缓冲。The not-block-aligned remainder is absorbed by the last core.
 */

#include <string>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/truncate_mod_tiling_data.h"
#include "../op_kernel/truncate_mod_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;

constexpr uint32_t BYTES_PER_BLOCK = 32u;
constexpr uint32_t BYTES_PER_CORE = 4096u;                            // minimal work granularity per core
constexpr uint32_t BLOCK_PER_CORE = BYTES_PER_CORE / BYTES_PER_BLOCK; // 128
constexpr uint32_t FOUR_BYTES = 4u;
constexpr uint32_t TWO_BYTES = 2u;
constexpr uint32_t ONE_BYTE = 1u;
constexpr uint64_t UB_RESERVED_BYTES = 2048u; // 预留给 tiling 结构/栈的 UB 空间

constexpr uint32_t INPUT_X1_IDX = 0u; // 被除数 x1
constexpr uint32_t INPUT_X2_IDX = 1u; // 除数 x2

// UB bytes consumed per processed element (single buffer accounted separately):
//   queues (x1, x2, y)        : 3 * dtypeSize * bufferNum
//   float compute buffers     : 3 * sizeof(float)   (x1_f, x2_f, tmp)
//   compare mask              : 1 byte / element (conservative)
static uint64_t CalcTileLength(uint64_t ubSize, uint32_t dtypeSize, uint32_t bufferNum, uint64_t elemsPerBlock)
{
    constexpr uint64_t QUEUE_NUM = 3u;
    constexpr uint64_t FLOAT_BUF_BYTES = 4u * sizeof(float); // calc0, calc1, tmp, tmp2
    constexpr uint64_t HALF_BUF_BYTES = sizeof(uint16_t);    // int8/uint8 -> half 中转缓冲
    constexpr uint64_t MASK_BYTES = 0u;
    // reserve a little UB for tiling struct / stack.
    uint64_t usable = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : ubSize;
    uint64_t perElem = QUEUE_NUM * dtypeSize * bufferNum + FLOAT_BUF_BYTES + HALF_BUF_BYTES + MASK_BYTES;
    uint64_t maxElems = usable / perElem;
    uint64_t blockPerQue = FloorDiv(maxElems, elemsPerBlock);
    return (blockPerQue == 0u) ? 1u : blockPerQue;
}

static ge::graphStatus TruncateModTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "TruncateMod tiling starts.");

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivCoreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0u;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    // dtype: x1 and x2 must agree; select the tiling key (schMode).
    auto x1Desc = context->GetInputDesc(INPUT_X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    auto x2Desc = context->GetInputDesc(INPUT_X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    ge::DataType dtype = x1Desc->GetDataType();
    OP_CHECK_IF(dtype != x2Desc->GetDataType(), OP_LOGE(context, "x1 and x2 dtype must be consistent."),
                return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0u;
    uint32_t dtypeSize = 0u;
    switch (dtype) {
        case ge::DT_FLOAT16:
            dtypeSize = TWO_BYTES;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_0);
            break;
        case ge::DT_FLOAT:
            dtypeSize = FOUR_BYTES;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_1);
            break;
        case ge::DT_BF16:
            dtypeSize = TWO_BYTES;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_2);
            break;
        case ge::DT_INT32:
            dtypeSize = FOUR_BYTES;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_3);
            break;
        case ge::DT_INT8:
            dtypeSize = ONE_BYTE;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_4);
            break;
        case ge::DT_UINT8:
            dtypeSize = ONE_BYTE;
            tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_5);
            break;
        default:
            OP_LOGE(context, "x1/x2 dtype must be one of float16, float32, bfloat16, int32, int8, uint8.");
            return ge::GRAPH_FAILED;
    }
    uint64_t elemsPerBlock = BYTES_PER_BLOCK / dtypeSize;

    // total element count (scalar shape -> 1).
    auto x1Shape = context->GetInputShape(INPUT_X1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    uint64_t totalLength = static_cast<uint64_t>(x1Shape->GetStorageShape().GetShapeSize());
    OP_CHECK_IF(totalLength == 0u, OP_LOGE(context, "input shape size must not be 0."), return ge::GRAPH_FAILED);

    // core split.
    uint64_t totalBlocks = FloorDiv(totalLength, elemsPerBlock);
    uint64_t tailElems = totalLength % elemsPerBlock;
    uint64_t coreNum = aivCoreNum;
    if (totalBlocks < coreNum * BLOCK_PER_CORE) {
        coreNum = CeilDiv(totalBlocks, static_cast<uint64_t>(BLOCK_PER_CORE));
    }
    if (coreNum == 0u) {
        coreNum = 1u;
    }
    uint64_t blockPerCore = FloorDiv(totalBlocks, coreNum);
    uint64_t tailBlocks = totalBlocks % coreNum;
    uint64_t coreLength = blockPerCore * elemsPerBlock;

    // UB split (enable double buffering only when there is more than one tile).
    uint32_t bufferNum = 1u;
    uint64_t blockPerQue = CalcTileLength(ubSize, dtypeSize, bufferNum, elemsPerBlock);
    if (FloorDiv(blockPerCore, blockPerQue) > 1u) {
        bufferNum = 2u;
        blockPerQue = CalcTileLength(ubSize, dtypeSize, bufferNum, elemsPerBlock);
    }
    uint64_t tileLength = blockPerQue * elemsPerBlock;

    // tile split.
    uint64_t epochs = FloorDiv(blockPerCore, blockPerQue);
    uint64_t tailTileLength = (blockPerCore % blockPerQue) * elemsPerBlock;
    uint64_t blockForLastCore = blockPerCore + tailBlocks;
    uint64_t epochsForLastCore = FloorDiv(blockForLastCore, blockPerQue);
    uint64_t tailTileLengthForLastCore = (blockForLastCore % blockPerQue) * elemsPerBlock;

    // fill tiling data.
    TruncateModTilingData* tiling = context->GetTilingData<TruncateModTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TruncateModTilingData), 0, sizeof(TruncateModTilingData)) != EOK,
                OP_LOGE(context, "memset tiling data error"), return ge::GRAPH_FAILED);
    tiling->coreNum = coreNum;
    tiling->bufferNum = bufferNum;
    tiling->tailElems = tailElems;
    tiling->epochs = epochs;
    tiling->epochsForLastCore = epochsForLastCore;
    tiling->coreLength = coreLength;
    tiling->tileLength = tileLength;
    tiling->tailTileLength = tailTileLength;
    tiling->tailTileLengthForLastCore = tailTileLengthForLastCore;

    context->SetBlockDim(coreNum);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0u;

    context->SetTilingKey(tilingKey);
    OP_LOGD(
        context,
        "TruncateMod tiling: key=%lu coreNum=%lu bufferNum=%u totalLen=%lu coreLength=%lu tileLength=%lu epochs=%lu "
        "epochsLast=%lu tailTile=%lu tailTileLast=%lu tailElems=%lu",
        tilingKey, coreNum, bufferNum, totalLength, coreLength, tileLength, epochs, epochsForLastCore, tailTileLength,
        tailTileLengthForLastCore, tailElems);
    return ge::GRAPH_SUCCESS;
}

struct TruncateModCompileInfo {};

static ge::graphStatus TilingParseForTruncateMod([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TruncateMod)
    .Tiling(TruncateModTilingFunc)
    .TilingParse<TruncateModCompileInfo>(TilingParseForTruncateMod);

} // namespace optiling
