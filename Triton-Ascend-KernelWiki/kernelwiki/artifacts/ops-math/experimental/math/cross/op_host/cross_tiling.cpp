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
 * \file cross_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/cross_tiling_data.h"
#include "../op_kernel/cross_tiling_key.h"
#include "util/platform_util.h"

namespace optiling {

const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
const uint32_t BLOCK_SIZE = 32U;
const uint32_t UB_DATA_NUM_FP32 = 19U;
const uint32_t UB_DATA_NUM_FP16 = 30U;
const uint32_t UB_DATA_NUM_INT8 = 18U;
const uint32_t UB_DATA_NUM_GROUP_MODE = 6U;

struct CrossCompileInfo {};

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
static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType, int64_t& intervalNum, int64_t& loopTimes)
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
        OP_LOGE(context, "Cross: inputx,inputy,outputz shape should equal"), return ge::GRAPH_FAILED);

    totalIdx = 1;
    for (uint32_t i = 0; i < inputShapeX.GetDimNum(); i++) {
        totalIdx *= inputShapeX.GetDim(i);
    }

    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT,   ge::DT_INT32, ge::DT_INT8,
                                                   ge::DT_FLOAT16, ge::DT_UINT8, ge::DT_INT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }

    // 获取dim属性（API层已处理dim合法性）
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dimPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimPtr);
    int64_t dim = *dimPtr;

    // 计算intervalNum和loopTimes（kernel需要）
    constexpr int64_t kCrossAxisSize = 3;
    int64_t dimProduct = 1;
    for (int64_t i = 0; i <= dim; ++i) {
        dimProduct *= inputShapeX.GetDim(static_cast<uint32_t>(i));
    }
    intervalNum = totalIdx / dimProduct;
    loopTimes = totalIdx / intervalNum / kCrossAxisSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static uint32_t GetUbBufferCountByDtype(ge::DataType dataType)
{
    if (dataType == ge::DT_FLOAT16) {
        return UB_DATA_NUM_FP16;
    }
    if (dataType == ge::DT_FLOAT || dataType == ge::DT_INT32 || dataType == ge::DT_INT16) {
        return UB_DATA_NUM_FP32;
    }
    if (dataType == ge::DT_INT8 || dataType == ge::DT_UINT8) {
        return UB_DATA_NUM_INT8;
    }
    return UB_DATA_NUM_FP16;
}

static bool IsGroupMode(ge::DataType dataType, int64_t intervalNum, int64_t loopTimes)
{
    (void)loopTimes;
    (void)dataType;
    return intervalNum == 1;
}

// tiling 分发入口
// 可直接替换你的CrossTilingFunc 内部实现（保留函数签名）
static ge::graphStatus CrossTilingFunc(gert::TilingContext* context)
{
    uint32_t blockSize = Ops::Base::GetUbBlockSize(context);
    if (blockSize == 0) {
        blockSize = BLOCK_SIZE;
    }

    // 1. platform
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    (void)ubSize;
    (void)coreNum;

    // 2. shapes & dtype
    int64_t totalIdx = 0;
    ge::DataType dataType;
    int64_t intervalNum = 0;
    int64_t loopTimes = 0;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType, intervalNum, loopTimes) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // handle empty input
    if (totalIdx <= 0) {
        CrossTilingData* tiling = context->GetTilingData<CrossTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(CrossTilingData), 0, sizeof(CrossTilingData));
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }

    // 3. workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. tiling data
    CrossTilingData* tiling = context->GetTilingData<CrossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(CrossTilingData), 0, sizeof(CrossTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context, "typeLength is 0"), return ge::GRAPH_FAILED);

    uint64_t tileDataNum = 1;
    if (IsGroupMode(dataType, intervalNum, loopTimes)) {
        uint64_t ubBlockCount = ubSize / blockSize;
        uint64_t usableBlockCount = ubBlockCount / UB_DATA_NUM_GROUP_MODE;
        if (usableBlockCount == 0) {
            usableBlockCount = 1;
        }
        tileDataNum = (usableBlockCount * blockSize) / (typeLength * 3U);
        if (tileDataNum == 0) {
            tileDataNum = 1;
        }
        if (tileDataNum > static_cast<uint64_t>(loopTimes)) {
            tileDataNum = static_cast<uint64_t>(loopTimes);
        }
        if (dataType == ge::DT_FLOAT16) {
            uint64_t minTileCount = static_cast<uint64_t>(std::max<int64_t>(coreNum, 1));
            if (minTileCount > static_cast<uint64_t>(loopTimes)) {
                minTileCount = static_cast<uint64_t>(loopTimes);
            }
            if (minTileCount > 0) {
                uint64_t perCoreLimit = (static_cast<uint64_t>(loopTimes) + minTileCount - 1) / minTileCount;
                if (perCoreLimit == 0) {
                    perCoreLimit = 1;
                }
                if (tileDataNum > perCoreLimit) {
                    tileDataNum = perCoreLimit;
                }
            }
        }
    } else {
        uint32_t ubBufferCount = GetUbBufferCountByDtype(dataType);
        uint64_t ubBlockCount = ubSize / blockSize;
        uint64_t usableBlockCount = ubBlockCount / ubBufferCount;
        if (usableBlockCount == 0) {
            usableBlockCount = 1;
        }
        tileDataNum = (usableBlockCount * blockSize) / typeLength;
        if (tileDataNum == 0) {
            tileDataNum = 1;
        }
        if (tileDataNum > static_cast<uint64_t>(intervalNum)) {
            tileDataNum = static_cast<uint64_t>(intervalNum);
        }
    }

    tiling->intervalNum = intervalNum;
    tiling->loopTimes = loopTimes;
    tiling->tileDataNum = tileDataNum;

    int64_t tilesPerLoop = (intervalNum + static_cast<int64_t>(tileDataNum) - 1) / static_cast<int64_t>(tileDataNum);
    if (IsGroupMode(dataType, intervalNum, loopTimes)) {
        tilesPerLoop = (loopTimes + static_cast<int64_t>(tileDataNum) - 1) / static_cast<int64_t>(tileDataNum);
    }
    if (tilesPerLoop <= 0) {
        tilesPerLoop = 1;
    }
    int64_t totalTileCount = loopTimes * tilesPerLoop;
    if (IsGroupMode(dataType, intervalNum, loopTimes)) {
        totalTileCount = tilesPerLoop;
    }

    uint32_t blockDim = 1;
    if (totalTileCount > 0) {
        int64_t maxBlockNum = std::min<int64_t>(coreNum, totalTileCount);
        blockDim = static_cast<uint32_t>(std::max<int64_t>(maxBlockNum, 1));
    }
    context->SetBlockDim(blockDim);

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
    } else if (dataType == ge::DT_INT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_5);
        context->SetTilingKey(tilingKey);
    } else {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCross([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Cross).Tiling(CrossTilingFunc).TilingParse<CrossCompileInfo>(TilingParseForCross);
} // namespace optiling
