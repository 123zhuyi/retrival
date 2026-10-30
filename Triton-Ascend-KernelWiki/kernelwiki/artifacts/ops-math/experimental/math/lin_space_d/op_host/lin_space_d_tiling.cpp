/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
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
 * \file lin_space_d_tiling.cpp
 * \brief LinSpaceD operator tiling implementation
 */
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/lin_space_d_tiling_data.h"
#include "../op_kernel/lin_space_d_tiling_key.h"

namespace optiling {

static const size_t INPUT_IDX_START = 0;
static const size_t INPUT_IDX_STOP = 1;
static const size_t INPUT_IDX_NUM = 2;
static const uint32_t BLOCK_SIZE = 32;
static const uint32_t BUFFER_NUM = 2;
static const uint64_t USER_SIZE = 0;

struct LinSpaceDCompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum, uint64_t& wsSysSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    wsSysSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

// 获取shape和数据类型信息
static ge::graphStatus GetShapeAndTypeInfo(
    gert::TilingContext* context, uint32_t& totalLength, ge::DataType& start_type, ge::DataType& end_type)
{
    // 获取num输入张量
    auto tensorN = context->GetInputTensor(INPUT_IDX_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorN);
    auto dataPtr = tensorN->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, dataPtr);
    totalLength = static_cast<uint32_t>(dataPtr[0]);

    // 获取数据类型
    auto inputStartDesc = context->GetInputDesc(INPUT_IDX_START);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStartDesc);
    start_type = inputStartDesc->GetDataType();

    auto inputEndDesc = context->GetInputDesc(INPUT_IDX_STOP);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputEndDesc);
    end_type = inputEndDesc->GetDataType();

    return ge::GRAPH_SUCCESS;
}

static void TilingParamsCalc(
    uint32_t length, uint32_t maxPerCoreElem, uint32_t alignNum, uint32_t& tileNum, uint32_t& lastTileLen)
{
    tileNum = 0;
    lastTileLen = 0;

    if (length == 0 || maxPerCoreElem == 0 || alignNum == 0) {
        return;
    }

    tileNum = length / maxPerCoreElem;

    if ((length % maxPerCoreElem == 0U) || (tileNum == 0U)) {
        if (tileNum == 0U) {
            tileNum = (length + alignNum - 1) / alignNum;
        }
        if (length < maxPerCoreElem) {
            lastTileLen = (tileNum == 1) ? length : (length % alignNum == 0 ? alignNum : length % alignNum);
        } else {
            lastTileLen = length - (tileNum - 1) * maxPerCoreElem;
        }
    } else {
        tileNum++;
        lastTileLen = length - (tileNum - 1) * maxPerCoreElem;
    }
}

// 计算Tiling数据
static ge::graphStatus CalculateTilingData(
    LinSpaceDTilingData* tiling, uint64_t ubSize, int64_t coreNum, uint32_t totalLength, uint32_t dataTypeSize)
{
    if (dataTypeSize == 0 || coreNum == 0)
        return ge::GRAPH_FAILED;
    uint32_t ubDataNumber = 20;
    uint32_t alignNum = BLOCK_SIZE / dataTypeSize;
    if (alignNum == 0)
        return ge::GRAPH_FAILED;
    uint32_t ubBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    uint32_t totalLengthAligned =
        (totalLength % alignNum == 0U) ? totalLength : ((totalLength + alignNum - 1) / alignNum) * alignNum;
    uint32_t totalBlockCount = totalLengthAligned / alignNum;
    uint32_t formerNum = totalBlockCount % coreNum;
    uint32_t formerLength = 0, tailLength = 0;
    uint32_t fTileNum = 0, fLastTileLen = 0;
    uint32_t tTileNum = 0, tLastTileLen = 0;
    uint32_t maxPerCoreElem = alignNum * ubBlockNum;

    if (formerNum != 0) {
        formerLength = (totalBlockCount + coreNum - 1) / coreNum * alignNum;
        TilingParamsCalc(formerLength, maxPerCoreElem, alignNum, fTileNum, fLastTileLen);
    }

    tailLength = totalBlockCount / coreNum * alignNum;
    TilingParamsCalc(tailLength, maxPerCoreElem, alignNum, tTileNum, tLastTileLen);

    tiling->formerNum = formerNum;
    tiling->formerLength = formerLength;
    tiling->formerTileNum = fTileNum;
    tiling->formerLastTileLength = fLastTileLen;
    tiling->tailLength = tailLength;
    tiling->tailTileNum = tTileNum;
    tiling->tailLastTileLength = tLastTileLen;
    tiling->totalLength = totalLength;

    return ge::GRAPH_SUCCESS;
}

// 获取WorkspaceSize信息
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, uint64_t wsSysSize, uint64_t userSize)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = wsSysSize + userSize;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus LinSpaceDTilingFunc(gert::TilingContext* context)
{
    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    uint64_t wsSysSize;
    uint64_t userSize = USER_SIZE;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum, wsSysSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 获取shape和类型信息
    uint32_t totalLength;
    ge::DataType start_type, end_type;
    OP_CHECK_IF(
        GetShapeAndTypeInfo(context, totalLength, start_type, end_type) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAndTypeInfo error"), return ge::GRAPH_FAILED);

    // 获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context, wsSysSize, userSize) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 设置tiling信息
    uint32_t dataTypeSize = 4;
    LinSpaceDTilingData* tiling = context->GetTilingData<LinSpaceDTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(LinSpaceDTilingData), 0, sizeof(LinSpaceDTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalculateTilingData(tiling, ubSize, coreNum, totalLength, dataTypeSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CalculateTilingData error"), return ge::GRAPH_FAILED);

    context->SetBlockDim(coreNum);

    uint64_t tilingKey = 0;
    if ((start_type == ge::DT_UINT8 || start_type == ge::DT_BF16 || start_type == ge::DT_FLOAT16 ||
         start_type == ge::DT_FLOAT || start_type == ge::DT_INT8 || start_type == ge::DT_INT16 ||
         start_type == ge::DT_INT32) &&
        (end_type == ge::DT_UINT8 || end_type == ge::DT_BF16 || end_type == ge::DT_FLOAT16 ||
         end_type == ge::DT_FLOAT || end_type == ge::DT_INT8 || end_type == ge::DT_INT16 || end_type == ge::DT_INT32)) {
        tilingKey = GET_TPL_TILING_KEY(start_type, end_type);
        OP_CHECK_IF(
            context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetTilingKey failed"),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context, "Unsupported data type combination");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLinSpaceD([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(LinSpaceD)
    .Tiling(LinSpaceDTilingFunc)
    .TilingInputsDataDependency({2})
    .TilingParse<LinSpaceDCompileInfo>(TilingParseForLinSpaceD);
} // namespace optiling
