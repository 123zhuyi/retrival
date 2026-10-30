/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua<@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file broadcast_v2_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/broadcast_v2_tiling_data.h"
#include "../op_kernel/broadcast_v2_tiling_key.h"

#include "tiling/tiling_api.h"
namespace optiling {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 512U;

namespace {
struct BroadcastV2CompileInfo {};
struct PlatformInfo {
    uint64_t ubSize{0};
    int64_t coreNum{0};
    uint32_t blockSize{0};
};
} // namespace

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, PlatformInfo& platformInfo)
{
    auto& ubSize = platformInfo.ubSize;
    auto& coreNum = platformInfo.coreNum;
    auto& blockSize = platformInfo.blockSize;
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize <= 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    totalIdx = inputX->GetStorageShape().GetShapeSize();

    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
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

static ge::graphStatus CalculateCoreNum(
    gert::TilingContext* context, BroadcastV2TilingData* tiling, const PlatformInfo& pltInfo, int64_t totalIdx)
{
    auto coreNum = pltInfo.coreNum;
    auto blockSize = pltInfo.blockSize;
    auto ubSize = pltInfo.ubSize;

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    if (typeLength == 0) {
        OP_LOGE(context, "typeLength is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t inputBytes = static_cast<uint64_t>(typeLength);
    uint64_t inputLengthBytes = static_cast<uint64_t>(totalIdx) * inputBytes;

    const gert::RuntimeAttrs* broadcastattrs = context->GetAttrs();
    const uint32_t bufferMode = *(broadcastattrs->GetAttrPointer<uint32_t>(0));
    const uint32_t dim = *(broadcastattrs->GetAttrPointer<uint32_t>(1));
    const uint32_t isReuseSource = *(broadcastattrs->GetAttrPointer<uint32_t>(2));
    const uint32_t axis = *(broadcastattrs->GetAttrPointer<uint32_t>(3));
    const uint32_t num = *(broadcastattrs->GetAttrPointer<uint32_t>(4));
    const gert::StorageShape* src_shape = context->GetInputShape(0);
    uint32_t bLength, sLength;
    ge::Shape inputShapeForTmp, outputShapeForTmp;
    if (dim == 1) {
        bLength = src_shape->GetStorageShape().GetDim(0);
        sLength = 1;
        inputShapeForTmp = ge::Shape({static_cast<int64_t>(bLength)});
        outputShapeForTmp = ge::Shape({static_cast<int64_t>(bLength * num)});
    } else {
        bLength = src_shape->GetStorageShape().GetDim(0);
        sLength = src_shape->GetStorageShape().GetDim(1);
        inputShapeForTmp = ge::Shape({static_cast<int64_t>(bLength), static_cast<int64_t>(sLength)});
        if (axis == 0) {
            outputShapeForTmp = ge::Shape({static_cast<int64_t>(bLength * num), static_cast<int64_t>(sLength)});
        } else {
            outputShapeForTmp = ge::Shape({static_cast<int64_t>(bLength), static_cast<int64_t>(sLength * num)});
        }
    }

    uint32_t maxsize = 0, minsize = 0;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    AscendC::GetBroadCastMaxMinTmpSize(
        ascendcPlatform, inputShapeForTmp, outputShapeForTmp, typeLength, isReuseSource == 1, maxsize, minsize);
    uint32_t tmpSize;
    if (bufferMode == 0) {
        tmpSize = 0;
    } else if (bufferMode == 1) {
        tmpSize = minsize;
    } else if (bufferMode == 2) {
        tmpSize = maxsize;
    } else {
        tmpSize = (maxsize + minsize) / 2;
    }
    if (blockSize == 0) {
        OP_LOGE(context, "blockSize is 0, division by zero error");
        return ge::GRAPH_FAILED;
    }
    uint64_t blocksTotal = (inputLengthBytes + blockSize - 1ULL) / blockSize;
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
    uint32_t smallCoreDataNum = static_cast<uint32_t>(everyCoreInputBlockNum * blockSize / inputBytes);
    uint32_t bigCoreDataNum = static_cast<uint32_t>((everyCoreInputBlockNum + 1) * blockSize / inputBytes);

    uint64_t availableUB = (ubSize > tmpSize) ? (ubSize - tmpSize) : 0;
    uint64_t denom = BUFFER_NUM * typeLength * (1 + num);
    uint32_t tileDataNum = (denom > 0) ? static_cast<uint32_t>(availableUB / denom) : 0;
    if (tileDataNum == 0) {
        OP_LOGE(context, "UB not enough for even 1 element after tmpSize reservation");
        return ge::GRAPH_FAILED;
    }
    if (dim == 2 && sLength > 0) {
        tileDataNum = (tileDataNum / sLength) * sLength;
        if (tileDataNum == 0)
            tileDataNum = sLength;
    }
    uint64_t tileBytes = static_cast<uint64_t>(tileDataNum) * typeLength;
    tileBytes = (tileBytes / 32U) * 32U; // 向下对齐到 32B
    if (tileBytes == 0) {
        tileBytes = 32U;
    }
    tileDataNum = static_cast<uint32_t>(tileBytes / typeLength);
    if (dim == 2 && sLength > 0) {
        tileDataNum = (tileDataNum / sLength) * sLength;
        if (tileDataNum == 0)
            tileDataNum = sLength;
    }
    auto calculate_tiling = [&](uint32_t core_data_num) {
        if (core_data_num == 0)
            return std::make_pair(0U, 0U);
        if (core_data_num <= tileDataNum) {
            return std::make_pair(1U, core_data_num); // 1个tile，尾部大小就是全部
        }
        uint32_t num_tiles = core_data_num / tileDataNum;
        uint32_t tail_data = core_data_num % tileDataNum;
        if (tail_data == 0) {
            tail_data = tileDataNum; // 最后一片是完整的
        } else {
            num_tiles++; // 包含不完整的尾片
        }
        return std::make_pair(num_tiles, tail_data);
    };

    auto small_tiling = calculate_tiling(smallCoreDataNum);
    auto big_tiling = calculate_tiling(bigCoreDataNum);

    tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(small_tiling.first);
    tiling->smallTailDataNum = static_cast<int64_t>(small_tiling.second);
    tiling->finalBigTileNum = static_cast<int64_t>(big_tiling.first);
    tiling->bigTailDataNum = static_cast<int64_t>(big_tiling.second);
    tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);
    tiling->tmpSize = static_cast<int64_t>(tmpSize);
    tiling->totalLength = static_cast<int64_t>(totalIdx);
    tiling->isReuseSource = static_cast<int64_t>(isReuseSource);
    tiling->axis = static_cast<int64_t>(axis);
    tiling->num = static_cast<int64_t>(num);
    tiling->bLength = static_cast<int64_t>(bLength);
    tiling->dim = static_cast<int64_t>(dim);
    tiling->sLength = static_cast<int64_t>(sLength);
    context->SetBlockDim(finalCoreNum);
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus BroadcastV2TilingFunc(gert::TilingContext* context)
{
    // 1. platform
    PlatformInfo platformInfo;
    OP_CHECK_IF(
        GetPlatformInfo(context, platformInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. shapes & dtype
    int64_t totalIdx = 0;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // handle empty input
    if (totalIdx <= 0) {
        BroadcastV2TilingData* tiling = context->GetTilingData<BroadcastV2TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(BroadcastV2TilingData), 0, sizeof(BroadcastV2TilingData));
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }

    // 3. workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    BroadcastV2TilingData* tiling = context->GetTilingData<BroadcastV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(BroadcastV2TilingData), 0, sizeof(BroadcastV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 4. CoreNum & Tiling Calculation
    OP_CHECK_IF(
        CalculateCoreNum(context, tiling, platformInfo, totalIdx) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CalculateCoreNum error"), return ge::GRAPH_FAILED);
    context->GetRawTilingData()->SetDataSize(sizeof(BroadcastV2TilingData));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBroadcastV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(BroadcastV2)
    .Tiling(BroadcastV2TilingFunc)
    .TilingParse<BroadcastV2CompileInfo>(TilingParseForBroadcastV2);
} // namespace optiling
