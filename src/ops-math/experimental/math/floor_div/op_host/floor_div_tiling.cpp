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
 * \file floor_div_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/floor_div_tiling_data.h"
#include "../op_kernel/floor_div_tiling_key.h"
#include "util/platform_util.h"

namespace optiling {

const uint32_t BUFFER_NUM = 2;

const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
const int32_t DIMS_LIMIT = 4;
#define UBDataNumberForBf16AndHalf 10
#define UBDataNumberForFloat 6
#define UBDataNumberForInt8AndUint8 18
#define UBDataNumberForInt32 8

struct FloorDivCompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    // 获取输入shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    // 如果输入shape 是标量 转换为{1}，否则保持原 shape 不变
    auto inputShapeX = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());
    auto inputY = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto inputShapeY = Ops::Base::EnsureNotScalar(inputY->GetStorageShape());
    auto outZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outZ);
    auto outShapeZ = Ops::Base::EnsureNotScalar(outZ->GetStorageShape());

    // shape校验
    OP_CHECK_IF(
        inputShapeX.GetDimNum() != inputShapeY.GetDimNum() || inputShapeY.GetDimNum() != outShapeZ.GetDimNum(),
        OP_LOGE(context, "FloorDiv: inputx,inputy,outputz shape should equal"), return ge::GRAPH_FAILED);

    totalIdx = 1;
    for (uint32_t i = 0; i < inputShapeX.GetDimNum(); i++) {
        totalIdx *= inputShapeX.GetDim(i);
    }

    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT,   ge::DT_INT32, ge::DT_INT8,
                                                   ge::DT_FLOAT16, ge::DT_UINT8, ge::DT_BF16};
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
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
// 可直接替换你的FloorDivTilingFunc 内部实现（保留函数签名）
static ge::graphStatus FloorDivTilingFunc(gert::TilingContext* context)
{
    uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize(context);
    // 1. platform
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. shapes & dtype
    int64_t totalIdx = 0;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // handle empty input
    if (totalIdx <= 0) {
        FloorDivTilingData* tiling = context->GetTilingData<FloorDivTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(FloorDivTilingData), 0, sizeof(FloorDivTilingData));
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }

    // 3. workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. tiling data
    FloorDivTilingData* tiling = context->GetTilingData<FloorDivTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(FloorDivTilingData), 0, sizeof(FloorDivTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // --- safer numeric types ---
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    if (typeLength == 0) {
        OP_LOGE(context, "typeLength is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t inputBytes = static_cast<uint64_t>(typeLength);
    uint64_t inputLengthBytes = static_cast<uint64_t>(totalIdx) * inputBytes;

    // ub-based tileBlockNum guard (避免为0)
    uint32_t ubDataNumber = 0;
    if (context->GetInputDesc(0)->GetDataType() == ge::DT_BF16 ||
        context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16) {
        ubDataNumber = UBDataNumberForBf16AndHalf;
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) {
        ubDataNumber = UBDataNumberForFloat;
    } else if (
        context->GetInputDesc(0)->GetDataType() == ge::DT_INT8 ||
        context->GetInputDesc(0)->GetDataType() == ge::DT_UINT8) {
        ubDataNumber = UBDataNumberForInt8AndUint8;
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_INT32) {
        ubDataNumber = UBDataNumberForInt32;
    } else {
        OP_LOGE(context, "not support datatype");
        return ge::GRAPH_FAILED;
    }
    uint64_t tmp = (ubSize / BLOCK_SIZE);
    uint32_t tileBlockNum = 1U;
    if (tmp > 0) {
        uint64_t tb = tmp / ubDataNumber;
        tileBlockNum = (tb == 0) ? 1U : static_cast<uint32_t>(tb);
    }

    // 每个 tile 包含的元素数（至少 1）
    uint32_t tileDataNum = static_cast<uint32_t>((static_cast<uint64_t>(tileBlockNum) * BLOCK_SIZE) / inputBytes);

    // 总 block 数（向上取整）
    uint64_t blocksTotal = (inputLengthBytes + BLOCK_SIZE - 1ULL) / BLOCK_SIZE;
    uint64_t coreNum64 = static_cast<uint64_t>(coreNum);
    if (coreNum64 > blocksTotal)
        coreNum64 = blocksTotal;
    if (coreNum64 == 0ULL)
        coreNum64 = 1ULL; // 最少 1 core
    if (tileDataNum >= totalIdx) {
        coreNum64 = 1;
    }
    uint32_t finalCoreNum = static_cast<uint32_t>(coreNum64);

    uint64_t everyCoreInputBlockNum = blocksTotal / coreNum64;              // 基本块数
    uint32_t tailBlockNum = static_cast<uint32_t>(blocksTotal % coreNum64); // 前 tailBlockNum 个核是 big-core

    // small-core 数量（元素）
    uint64_t smallCoreDataNum_u = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t smallCoreDataNum = static_cast<uint32_t>(smallCoreDataNum_u);

    uint32_t smallTileNum = static_cast<uint32_t>(everyCoreInputBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalSmallTileNum = ((everyCoreInputBlockNum % tileBlockNum) == 0) ? smallTileNum : (smallTileNum + 1);
    int64_t smallTailDataNum_i =
        static_cast<int64_t>(smallCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(smallTileNum);
    uint32_t smallTailDataNum = (smallTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(smallTailDataNum_i);

    // big-core（每个多一个 block）
    uint64_t bigEveryCoreBlockNum = everyCoreInputBlockNum + 1ULL;
    uint64_t bigCoreDataNum_u = bigEveryCoreBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t bigCoreDataNum = static_cast<uint32_t>(bigCoreDataNum_u);
    uint32_t bigTileNum = static_cast<uint32_t>(bigEveryCoreBlockNum / static_cast<uint64_t>(tileBlockNum));
    uint32_t finalBigTileNum = ((bigEveryCoreBlockNum % tileBlockNum) == 0) ? bigTileNum : (bigTileNum + 1);
    int64_t bigTailDataNum_i =
        static_cast<int64_t>(bigCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(bigTileNum);
    uint32_t bigTailDataNum = (bigTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(bigTailDataNum_i);

    // write back
    tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<int64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<int64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<int64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);

    context->SetBlockDim(finalCoreNum);

    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    } else if (dataType == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    } else if (dataType == ge::DT_INT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_2);
        context->SetTilingKey(tilingKey);
    } else if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_3);
        context->SetTilingKey(tilingKey);
    } else if (dataType == ge::DT_UINT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_4);
        context->SetTilingKey(tilingKey);
    } else if (dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_5);
        context->SetTilingKey(tilingKey);
    } else {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFloorDiv([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(FloorDiv).Tiling(FloorDivTilingFunc).TilingParse<FloorDivCompileInfo>(TilingParseForFloorDiv);
} // namespace optiling