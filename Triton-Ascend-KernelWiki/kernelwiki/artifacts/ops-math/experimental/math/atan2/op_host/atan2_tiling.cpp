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
 * \file atan2_tiling.cpp
 * \brief Tiling for the Atan2 binary elementwise operator.
 *
 * Strategy: identical to celu_v2 big/small core split.
 * UB budget accounts for:
 * - 2 VECIN queues (x1, x2): BUFFER_NUM × tileDataNum × sizeof(T) each
 * - 1 VECOUT queue (y):       BUFFER_NUM × tileDataNum × sizeof(T)
 * - 5 float32 VECCALC scratch: 5 × tileDataNum × sizeof(float)
 * - 1 uint8_t VECCALC scratch for Atan: atanTmpSize bytes
 *
 * atanTmpSize is obtained via GetAtanMaxMinTmpSize (ascendc host-side API).
 * If unavailable, we conservatively allocate 8× the tile float32 size.
 *
 * Supported dtypes: float32, float16, bfloat16.
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/atan2_tiling_data.h"
#include "../op_kernel/atan2_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 512U;

struct Atan2CompileInfo {};

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

static ge::graphStatus GetShapeInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    auto inputX1 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX1);
    totalIdx = inputX1->GetStorageShape().GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Atan2TilingFunc(gert::TilingContext* context)
{
    // 1. Platform info
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Shape / dtype
    int64_t totalIdx = 0;
    ge::DataType dataType = ge::DataType::DT_UNDEFINED;
    OP_CHECK_IF(
        GetShapeInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeInfo error"),
        return ge::GRAPH_FAILED);

    // Handle empty input
    if (totalIdx <= 0) {
        Atan2TilingData* tiling = context->GetTilingData<Atan2TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(Atan2TilingData), 0, sizeof(Atan2TilingData));
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }

    // 3. Workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. Compute tiling parameters
    Atan2TilingData* tiling = context->GetTilingData<Atan2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(Atan2TilingData), 0, sizeof(Atan2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    if (typeLength == 0) {
        OP_LOGE(context, "typeLength is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t inputBytes = static_cast<uint64_t>(typeLength);
    uint64_t inputLengthBytes = static_cast<uint64_t>(totalIdx) * inputBytes;

    // elements per 32-byte block
    uint32_t dtypeAlign = static_cast<uint32_t>(BLOCK_SIZE / inputBytes);

    uint32_t ubTotalFactor = 19U;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        ubTotalFactor = 16U;
    } else if (dataType == ge::DT_FLOAT) {
        ubTotalFactor = 19U;
    }

    uint64_t tileDataNum64 = (ubSize / sizeof(float)) / ubTotalFactor;

    // Round down to nearest dtypeAlign multiple to keep DMA transfers aligned
    tileDataNum64 = (tileDataNum64 / dtypeAlign) * dtypeAlign;
    uint32_t tileDataNum = (tileDataNum64 == 0ULL) ? dtypeAlign : static_cast<uint32_t>(tileDataNum64);

    // tileBlockNum: number of 32-byte BLOCK_SIZE units in one tile (input dtype)
    uint32_t tileBlockNum =
        static_cast<uint32_t>((static_cast<uint64_t>(tileDataNum) * inputBytes + BLOCK_SIZE - 1U) / BLOCK_SIZE);

    uint32_t atanTmpSize = 8U * tileDataNum * sizeof(float);
    tiling->atanTmpSize = atanTmpSize;

    uint64_t blocksTotal = (inputLengthBytes + BLOCK_SIZE - 1ULL) / BLOCK_SIZE;
    uint64_t coreNum64 = static_cast<uint64_t>(coreNum);
    if (coreNum64 > blocksTotal) {
        coreNum64 = blocksTotal;
    }
    if (coreNum64 == 0ULL) {
        coreNum64 = 1ULL;
    }
    uint32_t finalCoreNum = static_cast<uint32_t>(coreNum64);

    uint64_t everyCoreInputBlockNum = blocksTotal / coreNum64;
    uint32_t tailBlockNum = static_cast<uint32_t>(blocksTotal % coreNum64);

    // small-core
    uint64_t smallCoreDataNum_u = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t smallCoreDataNum = static_cast<uint32_t>(smallCoreDataNum_u);
    uint32_t smallTileNum = static_cast<uint32_t>(everyCoreInputBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalSmallTileNum = ((everyCoreInputBlockNum % tileBlockNum) == 0) ? smallTileNum : (smallTileNum + 1);
    int64_t smallTailDataNum_i =
        static_cast<int64_t>(smallCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(smallTileNum);
    uint32_t smallTailDataNum = (smallTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(smallTailDataNum_i);

    // big-core
    uint64_t bigEveryCoreBlockNum = everyCoreInputBlockNum + 1ULL;
    uint64_t bigCoreDataNum_u = bigEveryCoreBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t bigCoreDataNum = static_cast<uint32_t>(bigCoreDataNum_u);
    uint32_t bigTileNum = static_cast<uint32_t>(bigEveryCoreBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalBigTileNum = ((bigEveryCoreBlockNum % tileBlockNum) == 0) ? bigTileNum : (bigTileNum + 1);
    int64_t bigTailDataNum_i =
        static_cast<int64_t>(bigCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(bigTileNum);
    uint32_t bigTailDataNum = (bigTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(bigTailDataNum_i);

    tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<int64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<int64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<int64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);

    context->SetBlockDim(finalCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAtan2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Atan2).Tiling(Atan2TilingFunc).TilingParse<Atan2CompileInfo>(TilingParseForAtan2);
} // namespace optiling