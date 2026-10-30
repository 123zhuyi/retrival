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
 * \file sign_bits_pack_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/sign_bits_pack_tiling_data.h"
#include "../op_kernel/sign_bits_pack_tiling_key.h"

namespace optiling {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint32_t REPEAT_ALIGN = 256;
constexpr uint32_t GM_ALIGN = 512;
constexpr uint32_t RESERVED_UB_SIZE = 0;     // 有些api需要预留ub空间
constexpr uint32_t MAX_TILEDATA = 18 * 1024; // 最大可以到

struct SignBitsPackCompileInfo {};

static ge::graphStatus TilingParseForSignBitsPack([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();

    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
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

static ge::graphStatus CalcAndSetTilingData(
    uint64_t inputNum, uint32_t typeLength, int64_t realCoreNum, SignBitsPackTilingData* tiling,
    gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context, "typeLength is zero"), return ge::GRAPH_FAILED);
    uint64_t elemsPerGmBlock = (GM_ALIGN / typeLength);
    uint64_t inputLengthAlgin512 = (inputNum + elemsPerGmBlock - 1) / elemsPerGmBlock * elemsPerGmBlock;

    uint64_t tileDataNum = MAX_TILEDATA;
    if (typeLength == 2) {
        tileDataNum = 2 * MAX_TILEDATA;
    }
    int64_t needCoreNum = (inputLengthAlgin512 + tileDataNum * BUFFER_NUM - 1) / (tileDataNum * BUFFER_NUM);
    int64_t coreNum = ((realCoreNum) < needCoreNum) ? realCoreNum : needCoreNum;
    uint64_t needCoreDataNum = ((inputLengthAlgin512 + coreNum - 1) / coreNum);
    if ((coreNum < realCoreNum / 4) && (needCoreDataNum > MAX_TILEDATA / 4)) {
        coreNum = coreNum * 2;
        needCoreDataNum = ((inputLengthAlgin512 + coreNum - 1) / coreNum);
    }
    uint32_t bufferNum = BUFFER_NUM;
    uint32_t usedDb = 1;
    if (needCoreDataNum < (MAX_TILEDATA / 2)) {
        bufferNum = 1;
        usedDb = 0;
    }
    uint64_t needTileDataNum = (needCoreDataNum + bufferNum - 1) / bufferNum;
    needTileDataNum = (needTileDataNum + elemsPerGmBlock - 1) / elemsPerGmBlock * elemsPerGmBlock;
    tileDataNum = (tileDataNum < needTileDataNum) ? tileDataNum : needTileDataNum;
    uint64_t everyCoreInputBlockNum = inputLengthAlgin512 / elemsPerGmBlock / coreNum;
    uint64_t tailBlockNum = (inputLengthAlgin512 / elemsPerGmBlock) % coreNum;
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * elemsPerGmBlock;
    uint64_t finalSmallTileNum = (smallCoreDataNum + tileDataNum - 1) / tileDataNum;
    uint64_t smallTailDataNum = smallCoreDataNum - (finalSmallTileNum - 1) * tileDataNum;
    uint64_t bigCoreDataNum = smallCoreDataNum + elemsPerGmBlock;
    uint64_t finalBigTileNum = (bigCoreDataNum + tileDataNum - 1) / tileDataNum;
    uint64_t bigTailDataNum = bigCoreDataNum - (finalBigTileNum - 1) * tileDataNum;

    uint64_t realLastPackLenth = smallTailDataNum - (inputLengthAlgin512 - inputNum);
    uint64_t elemsPerRepeat = (REPEAT_ALIGN / typeLength);
    uint32_t lastCalcLength = (realLastPackLenth + elemsPerRepeat - 1) / elemsPerRepeat * elemsPerRepeat;
    uint64_t elemsPerUbBlock = (UB_ALIGN / typeLength);
    uint32_t lastCopyLengthAlign = (realLastPackLenth + elemsPerUbBlock - 1) / elemsPerUbBlock * elemsPerUbBlock;
    uint32_t lastCopyLength = realLastPackLenth;
    uint32_t rightPaddingElemNums = lastCopyLengthAlign - realLastPackLenth;

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->usedDb = usedDb;
    tiling->lastCopyLength = lastCopyLength;
    tiling->rightPaddingElemNums = rightPaddingElemNums;
    tiling->lastCalcLength = lastCalcLength;
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus SignBitsPackTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t realCoreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, realCoreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    SignBitsPackTilingData* tiling = context->GetTilingData<SignBitsPackTilingData>();
    uint64_t inputNum;
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    CalcAndSetTilingData(inputNum, typeLength, realCoreNum, tiling, context);
    ret = GetWorkspaceSize(context);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(SignBitsPack)
    .Tiling(SignBitsPackTilingFunc)
    .TilingParse<SignBitsPackCompileInfo>(TilingParseForSignBitsPack);
} // namespace optiling
