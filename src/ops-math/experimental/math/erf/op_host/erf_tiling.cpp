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
 * \file erf_tiling.cpp
 * \brief
 */
#include <algorithm>
#include <array>
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "../op_kernel/erf_tiling_data.h"
#include "../op_kernel/erf_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0;
constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t DB_TILING_MIN_NUM = 262144; // 开启DoubleBuffer的最小输入shape
constexpr uint32_t UB_DATA_NUM_DB = 6144;      // 开启DoubleBuffer，单次UB可容纳的数据量
constexpr uint32_t UB_DATA_NUM_NDB_MAX = 6656; // 不开启DoubleBuffer，单次UB可容纳的最大数据量
constexpr uint32_t UB_DATA_NUM_NDB_HIGH = 4096;
constexpr uint32_t UB_DATA_NUM_NDB_MEDIUM = 2048;
constexpr uint32_t UB_DATA_NUM_NDB_LOW = 1024;
constexpr uint32_t UB_DATA_NUM_NDB_MIN = 512;
constexpr uint32_t UB_DATA_NUM_NDB_THRESHOLD_ONE = 2048;
constexpr uint32_t UB_DATA_NUM_NDB_THRESHOLD_TWO = 1024 * 20;
constexpr uint32_t UB_DATA_NUM_NDB_THRESHOLD_THREE = 2048 * 30;
constexpr uint32_t UB_DATA_NUM_NDB_THRESHOLD_FOUR = 4096 * 30;

constexpr uint32_t GM_ALIGN_NUM = 512; // 为了提高GM的搬运效率，对GM数据采取512字节对齐
constexpr uint32_t REPEAT_NUM = 256;
struct ErfCompileInfo {};

static ge::graphStatus TilingParseForErf([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t& inputNum, uint64_t& inputBytes)
{
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(INDEXZERO) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);
    inputNum = context->GetInputShape(INDEXZERO)->GetStorageShape().GetShapeSize();
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(INDEXZERO)->GetDataType(), typeLength);
    if (typeLength == 0) {
        return ge::GRAPH_FAILED;
    }
    inputBytes = typeLength;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateTilingDataNoDB(
    uint64_t inputNum, int64_t aivCoreNum, uint64_t inputBytes, uint64_t elemsPer512, uint64_t inputNumAlign256,
    uint64_t& tileDataNum, int64_t& coreNum, uint64_t& bigCoreDataNum, uint64_t& finalBigTileNum,
    uint64_t& bigTailDataNum)
{
    if (0 == BLOCK_SIZE || 0 == inputNum || 0 == aivCoreNum || 0 == inputBytes || 0 == UB_DATA_NUM_NDB_MAX ||
        0 == elemsPer512 || 0 == inputNumAlign256) {
        return ge::GRAPH_FAILED;
    }
    constexpr std::array<std::pair<uint32_t, uint32_t>, 5> thresholdMap = {
        {{UB_DATA_NUM_NDB_THRESHOLD_ONE, UB_DATA_NUM_NDB_MIN},
         {UB_DATA_NUM_NDB_THRESHOLD_TWO, UB_DATA_NUM_NDB_LOW},
         {UB_DATA_NUM_NDB_THRESHOLD_THREE, UB_DATA_NUM_NDB_MEDIUM},
         {UB_DATA_NUM_NDB_THRESHOLD_FOUR, UB_DATA_NUM_NDB_HIGH},
         {UINT32_MAX, UB_DATA_NUM_NDB_MAX}}};

    auto it = std::find_if(
        thresholdMap.begin(), thresholdMap.end(), [inputNum](const auto& pair) { return inputNum <= pair.first; });
    uint32_t suggestedTileDataNum = it->second;

    tileDataNum = suggestedTileDataNum;
    tileDataNum = (tileDataNum + elemsPer512 - 1) / elemsPer512 * elemsPer512;
    if (0 == tileDataNum) {
        return ge::GRAPH_FAILED;
    }
    coreNum = (inputNum + tileDataNum - 1) / tileDataNum;

    tileDataNum = (coreNum == 1) ? inputNumAlign256 : tileDataNum;
    finalBigTileNum = 1;
    bigTailDataNum = inputNumAlign256 - (coreNum - 1) * tileDataNum;
    bigCoreDataNum = tileDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateTilingDataWithDB(
    uint64_t inputNum, int64_t aivCoreNum, uint64_t inputBytes, uint64_t elemsPer512, uint64_t inputNumAlign256,
    uint64_t& tileDataNum, int64_t& coreNum, uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum,
    uint64_t& bigTailDataNum, uint64_t& smallCoreDataNum, uint64_t& finalSmallTileNum, uint64_t& finalBigTileNum)
{
    if (0 == BLOCK_SIZE || 0 == inputNum || 0 == aivCoreNum || 0 == inputBytes || 0 == UB_DATA_NUM_DB ||
        0 == elemsPer512 || 0 == inputNumAlign256) {
        return ge::GRAPH_FAILED;
    }

    tileDataNum = UB_DATA_NUM_DB;
    uint64_t tileDataNum_db = tileDataNum * BUFFER_NUM;
    if (0 == tileDataNum_db || 0 == tileDataNum) {
        return ge::GRAPH_FAILED;
    }
    coreNum = (inputNum + tileDataNum_db - 1) / tileDataNum_db;
    coreNum = (coreNum > aivCoreNum) ? aivCoreNum : coreNum;

    if (0 == coreNum) {
        return ge::GRAPH_FAILED;
    }
    bigCoreDataNum = (inputNum + coreNum - 1) / coreNum;
    bigCoreDataNum = (bigCoreDataNum + elemsPer512 - 1) / elemsPer512 * elemsPer512;
    finalBigTileNum = (bigCoreDataNum + tileDataNum - 1) / tileDataNum;
    if (((finalBigTileNum - 1) * tileDataNum) > bigCoreDataNum) {
        finalBigTileNum = finalBigTileNum - 1;
    }
    bigTailDataNum = bigCoreDataNum - (finalBigTileNum - 1) * tileDataNum;

    smallCoreDataNum = inputNumAlign256 - (coreNum - 1) * bigCoreDataNum;
    finalSmallTileNum = (smallCoreDataNum + tileDataNum - 1) / tileDataNum;
    if (((finalSmallTileNum - 1) * tileDataNum) > smallCoreDataNum) {
        finalSmallTileNum = finalSmallTileNum - 1;
    }
    smallTailDataNum = smallCoreDataNum - (finalSmallTileNum - 1) * tileDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ErfTilingFunc(gert::TilingContext* context)
{
    // ErfTilingData tiling;
    ErfTilingData* tiling = context->GetTilingData<ErfTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ErfTilingData), 0, sizeof(ErfTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    int64_t aivCoreNum = 0;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, aivCoreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    int64_t coreNum = aivCoreNum;

    uint64_t inputNum = 0;
    uint64_t inputBytes = 0;
    uint64_t tileDataNum = 0;
    uint64_t finalBigTileNum = 0;
    uint64_t finalSmallTileNum = 0;
    uint64_t bigTailDataNum = 0;
    uint64_t smallTailDataNum = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t smallCoreDataNum = 0;

    ret = GetShapeAttrsInfo(context, inputNum, inputBytes);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);

    // 为了提高小输入Shape数据的性能，设计2个kernel。
    // ELEMENTWISE_TPL_SCH_MODE_0：对应处理不超过262144数据的kernel，不开启double buffer，尽量减少scale开销；
    // ELEMENTWISE_TPL_SCH_MODE_1：对应处理超过262144数据的kernel，开启double buffer，尽量使用aicore的计算能力；
    OP_CHECK_IF(
        0 == GM_ALIGN_NUM || 0 == REPEAT_NUM || 0 == inputBytes, OP_LOGE(context, "parma error"),
        return ge::GRAPH_FAILED);
    uint64_t elemsPer512 = GM_ALIGN_NUM / inputBytes;
    uint64_t elemsPerRepeat = REPEAT_NUM / inputBytes;
    OP_CHECK_IF(0 == elemsPerRepeat, OP_LOGE(context, "elemsPerRepeat is error"), return ge::GRAPH_FAILED);
    uint64_t inputNumAlign256 = (inputNum + elemsPerRepeat - 1) / elemsPerRepeat * elemsPerRepeat;
    if (inputNum <= DB_TILING_MIN_NUM) //
    {
        uint32_t tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);

        CalculateTilingDataNoDB(
            inputNum, aivCoreNum, inputBytes, elemsPer512, inputNumAlign256, tileDataNum, coreNum, bigCoreDataNum,
            finalBigTileNum, bigTailDataNum);
    } else {
        uint32_t tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);

        CalculateTilingDataWithDB(
            inputNum, aivCoreNum, inputBytes, elemsPer512, inputNumAlign256, tileDataNum, coreNum, bigCoreDataNum,
            smallTailDataNum, bigTailDataNum, smallCoreDataNum, finalSmallTileNum, finalBigTileNum);
    }

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = static_cast<uint32_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<uint32_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint32_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint32_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint32_t>(finalBigTileNum);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Erf).Tiling(ErfTilingFunc).TilingParse<ErfCompileInfo>(TilingParseForErf);
} // namespace optiling
