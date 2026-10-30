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
 * \file floor_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "../op_kernel/floor_tiling_data.h"
#include "../op_kernel/floor_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 512U;
constexpr uint32_t UB_DATA_NUM_FLOAT = 4U;
constexpr uint32_t UB_DATA_NUM_OTHER = 6U;
constexpr uint32_t MIN_ELEM_PER_CORE = 512U;

struct FloorCompileInfo {};

static ge::graphStatus TilingParseForFloor([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is %ld", coreNum), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is %lu", ubSize), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t ubSize, uint64_t& inputNum, uint64_t& inputBytes, uint64_t& tileBlockNum,
    uint64_t& tileDataNum, uint64_t& inputLengthAlgin)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    inputNum = inputShape->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;

    OP_CHECK_IF(inputNum == 0, OP_LOGE(context, "inputNum is 0"), return ge::GRAPH_FAILED);
    inputBytes = inputLength / inputNum;

    uint64_t ubDataNumber =
        (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) ? UB_DATA_NUM_FLOAT : UB_DATA_NUM_OTHER;

    OP_CHECK_IF(
        BLOCK_SIZE == 0 || ubDataNumber == 0, OP_LOGE(context, "BLOCK_SIZE or ubDataNumber is 0"),
        return ge::GRAPH_FAILED);
    tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;

    OP_CHECK_IF(inputBytes == 0, OP_LOGE(context, "inputBytes is 0"), return ge::GRAPH_FAILED);
    tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    inputLengthAlgin = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    gert::TilingContext* context, uint64_t inputLengthAlgin, int64_t coreNum, uint64_t tileBlockNum,
    uint64_t inputBytes, uint64_t tileDataNum, FloorTilingData* tiling)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        BLOCK_SIZE == 0 || coreNum <= 0 || tileBlockNum == 0 || inputBytes == 0,
        OP_LOGE(
            context, "invalid params: coreNum %ld, tileBlockNum %lu, inputBytes %lu", coreNum, tileBlockNum,
            inputBytes),
        return ge::GRAPH_FAILED);

    uint64_t everyCoreInputBlockNum = inputLengthAlgin / BLOCK_SIZE / coreNum;
    tiling->tailBlockNum = (inputLengthAlgin / BLOCK_SIZE) % coreNum;
    tiling->smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;

    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    tiling->finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    tiling->smallTailDataNum = tiling->smallCoreDataNum - (tileDataNum * smallTileNum);
    tiling->smallTailDataNum = tiling->smallTailDataNum == 0 ? tileDataNum : tiling->smallTailDataNum;

    everyCoreInputBlockNum += 1;
    tiling->bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    tiling->finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    tiling->bigTailDataNum = tiling->bigCoreDataNum - tileDataNum * bigTileNum;
    tiling->bigTailDataNum = tiling->bigTailDataNum == 0 ? tileDataNum : tiling->bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FloorTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    FloorTilingData* tiling = context->GetTilingData<FloorTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(FloorTilingData), 0, sizeof(FloorTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin;
    ret = GetShapeAttrsInfo(context, ubSize, inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    const uint64_t minElemPerCore = MIN_ELEM_PER_CORE;

    if (inputNum <= minElemPerCore) {
        coreNum = 1;
    } else {
        uint64_t maxAllowedCores = inputNum / minElemPerCore;
        if (static_cast<uint64_t>(coreNum) > maxAllowedCores) {
            coreNum = static_cast<int64_t>(maxAllowedCores);
        }

        uint64_t maxBlockCores = inputLengthAlgin / BLOCK_SIZE;
        if (static_cast<uint64_t>(coreNum) > maxBlockCores) {
            coreNum = static_cast<int64_t>(maxBlockCores);
        }
    }

    ret = CalculateCoreBlockNums(context, inputLengthAlgin, coreNum, tileBlockNum, inputBytes, tileDataNum, tiling);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ge::GRAPH_FAILED);

    tiling->tileDataNum = static_cast<uint64_t>(tileDataNum);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Floor).Tiling(FloorTilingFunc).TilingParse<FloorCompileInfo>(TilingParseForFloor);
} // namespace optiling