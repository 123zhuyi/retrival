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
 * \file acosh_grad_tiling.cpp
 * \brief AcoshGrad 算子 Tiling 实现
 *
 * UB 布局 (BUFFER_NUM=2 double-buffer):
 *   inQueueY  : 2 × tileLength × sizeof(T)
 *   inQueueDy : 2 × tileLength × sizeof(T)
 *   outQueueZ : 2 × tileLength × sizeof(T)
 *   sinhBuf   : 1 × tileLength × 4  (fp32)
 *   dyBuf     : 1 × tileLength × 4
 *   tmpBuf    : 1 × tileLength × 4
 *
 * 总 UB = tileLength * (3*BUFFER_NUM*sizeof(T) + 3*4) + RESERVE
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "util/platform_util.h"
#include "../op_kernel/acosh_grad_tiling_data.h"
#include "../op_kernel/acosh_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint64_t UB_RESERVE_BYTES = 8192U; // 8 KB for TPipe overhead
constexpr int64_t TILE_GRAN = 64;            // vector-friendly granularity
constexpr int64_t MIN_ELEMS_PER_CORE = 2048; // avoid over-splitting
constexpr int64_t TILING_BUFFER_NUM = 2;     // double-buffer，不依赖 kernel 头文件

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in)
{
    return (in.GetDimNum() == 0) ? g_vec_1_shape : in;
}

static bool IsSameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* ctx, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(ctx == nullptr, OP_LOGE(ctx, "context is nullptr"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfoPtr = ctx->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(ctx, "coreNum must be greater than 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(ctx, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* ctx)
{
    OP_CHECK_IF(ctx == nullptr, OP_LOGE(ctx, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t* ws = ctx->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, ws);
    ws[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDtypeInfo(gert::TilingContext* ctx, int64_t& elemBytes, uint64_t& tilingKey)
{
    auto* inputDesc = ctx->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, inputDesc);
    auto* dyDesc = ctx->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, dyDesc);

    ge::DataType dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != dyDesc->GetDataType(), OP_LOGE(ctx, "dtype of y and dy must be same"),
                return ge::GRAPH_FAILED);
    elemBytes = 4;
    if (dtype == ge::DT_FLOAT16) {
        elemBytes = 2;
        tilingKey = GET_TPL_TILING_KEY(ACOSHGRAD_TPL_SCH_MODE_0);
    } else if (dtype == ge::DT_FLOAT) {
        elemBytes = 4;
        tilingKey = GET_TPL_TILING_KEY(ACOSHGRAD_TPL_SCH_MODE_1);
    } else if (dtype == ge::DT_BF16) {
        elemBytes = 2;
        tilingKey = GET_TPL_TILING_KEY(ACOSHGRAD_TPL_SCH_MODE_2);
    } else {
        OP_LOGE(ctx, "Unsupported dtype %d", static_cast<int>(dtype));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTotalLength(gert::TilingContext* ctx, int64_t& totalLength)
{
    const gert::StorageShape* yStorageShape = ctx->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, yStorageShape);
    const gert::StorageShape* dyStorageShape = ctx->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, dyStorageShape);
    const gert::Shape& yShapeRef = yStorageShape->GetStorageShape();
    const gert::Shape& dyShapeRef = dyStorageShape->GetStorageShape();
    OP_CHECK_IF(!IsSameShape(yShapeRef, dyShapeRef), OP_LOGE(ctx, "shape of y and dy must be same"),
                return ge::GRAPH_FAILED);
    const gert::Shape realShape = EnsureNotScalar(yShapeRef);
    totalLength = 1;
    for (size_t d = 0; d < realShape.GetDimNum(); d++) {
        totalLength *= realShape.GetDim(d);
    }
    if (totalLength <= 0) {
        OP_LOGE(ctx, "totalLength <= 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTileLength(gert::TilingContext* ctx, uint64_t ubSize, int64_t elemBytes, int64_t totalLength,
                                     int64_t& tileLength)
{
    OP_CHECK_IF(elemBytes <= 0, OP_LOGE(ctx, "elemBytes must be greater than 0"), return ge::GRAPH_FAILED);

    int64_t bytesPerElem = static_cast<int64_t>(2 * TILING_BUFFER_NUM) * elemBytes +
                           static_cast<int64_t>(TILING_BUFFER_NUM) * elemBytes + 3 * 4;
    OP_CHECK_IF(bytesPerElem <= 0, OP_LOGE(ctx, "bytesPerElem must be greater than 0"), return ge::GRAPH_FAILED);

    int64_t usableUb = static_cast<int64_t>(ubSize) - static_cast<int64_t>(UB_RESERVE_BYTES);
    if (usableUb <= 0) {
        OP_LOGE(ctx, "UB too small");
        return ge::GRAPH_FAILED;
    }

    const int64_t ubBlockBytes = static_cast<int64_t>(Ops::Base::GetUbBlockSize(ctx));
    OP_CHECK_IF(ubBlockBytes <= 0, OP_LOGE(ctx, "GetUbBlockSize must be greater than 0"), return ge::GRAPH_FAILED);
    const int64_t minAlignElems = CeilDiv(ubBlockBytes, elemBytes);
    OP_CHECK_IF(minAlignElems <= 0, OP_LOGE(ctx, "minAlignElems must be greater than 0"), return ge::GRAPH_FAILED);

    tileLength = FloorAlign(usableUb / bytesPerElem, TILE_GRAN);
    if (tileLength < minAlignElems) {
        tileLength = minAlignElems;
    }
    if (tileLength > totalLength) {
        tileLength = totalLength;
    }
    OP_CHECK_IF(tileLength <= 0, OP_LOGE(ctx, "tileLength must be greater than 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AcoshGradTilingFunc(gert::TilingContext* ctx)
{
    OP_CHECK_IF(ctx == nullptr, OP_LOGE(ctx, "context is nullptr"), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    int64_t maxCoreNum;
    OP_CHECK_IF(GetPlatformInfo(ctx, ubSize, maxCoreNum) != ge::GRAPH_SUCCESS, OP_LOGE(ctx, "GetPlatformInfo failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetWorkspace(ctx) != ge::GRAPH_SUCCESS, OP_LOGE(ctx, "SetWorkspace failed"), return ge::GRAPH_FAILED);

    int64_t elemBytes = 4;
    uint64_t tilingKey = 0;
    OP_CHECK_IF(GetDtypeInfo(ctx, elemBytes, tilingKey) != ge::GRAPH_SUCCESS, OP_LOGE(ctx, "GetDtypeInfo failed"),
                return ge::GRAPH_FAILED);

    int64_t totalLength = 0;
    OP_CHECK_IF(GetTotalLength(ctx, totalLength) != ge::GRAPH_SUCCESS, OP_LOGE(ctx, "GetTotalLength failed"),
                return ge::GRAPH_FAILED);

    int64_t tileLength = 0;
    OP_CHECK_IF(GetTileLength(ctx, ubSize, elemBytes, totalLength, tileLength) != ge::GRAPH_SUCCESS,
                OP_LOGE(ctx, "GetTileLength failed"), return ge::GRAPH_FAILED);

    // Determine core count
    int64_t coreNum = CeilDiv(totalLength, MIN_ELEMS_PER_CORE);
    if (coreNum > maxCoreNum) {
        coreNum = maxCoreNum;
    }
    if (coreNum < 1) {
        coreNum = 1;
    }

    // Distribute elements across cores: first `rem` cores get (base+1), rest get base
    int64_t base = totalLength / coreNum;
    int64_t rem = totalLength % coreNum;

    // "former" cores have blockLength = base + 1  (rem of them)
    // "tail"   cores have blockLength = base       (coreNum - rem of them)
    int64_t blockLength = (rem > 0) ? (base + 1) : base;
    int64_t tailBlockLength = base;
    uint32_t formerCoreNum = static_cast<uint32_t>(rem); // number of "larger" cores

    // Compute tile numbers for a given block length
    auto calcTileInfo = [&](int64_t blen, int64_t& tnum, int64_t& lastLen) {
        if (blen <= 0) {
            tnum = 0;
            lastLen = 0;
            return;
        }
        tnum = CeilDiv(blen, tileLength);
        lastLen = blen - (tnum - 1) * tileLength;
    };

    int64_t tileNum, lastTileLength, tailTileNum, tailLastTileLength;
    calcTileInfo(blockLength, tileNum, lastTileLength);
    calcTileInfo(tailBlockLength, tailTileNum, tailLastTileLength);

    // Fill tiling struct
    AcoshGradTilingData* td = ctx->GetTilingData<AcoshGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, td);

    td->totalLength = static_cast<uint64_t>(totalLength);
    td->blockLength = static_cast<uint64_t>(blockLength);
    td->tailBlockLength = static_cast<uint64_t>(tailBlockLength);
    td->tileLength = static_cast<uint64_t>(tileLength);
    td->tileNum = static_cast<uint64_t>(tileNum);
    td->lastTileLength = static_cast<uint64_t>(lastTileLength);
    td->tailTileNum = static_cast<uint64_t>(tailTileNum);
    td->tailLastTileLength = static_cast<uint64_t>(tailLastTileLength);
    td->coreNum = static_cast<uint32_t>(coreNum);
    td->formerCoreNum = formerCoreNum;

    ctx->SetBlockDim(static_cast<uint32_t>(coreNum));
    ctx->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAcoshGrad([[maybe_unused]] gert::TilingParseContext* ctx)
{
    return ge::GRAPH_SUCCESS;
}

struct AcoshGradCompileInfo {};

IMPL_OP_OPTILING(AcoshGrad).Tiling(AcoshGradTilingFunc).TilingParse<AcoshGradCompileInfo>(TilingParseForAcoshGrad);

} // namespace optiling
