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
 * \file sign_bits_unpack_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/sign_bits_unpack_tiling_data.h"
#include "../op_kernel/sign_bits_unpack_tiling_key.h"

namespace optiling {

#define UB_NUM 2U
#define BLOCK_SIZE 64U
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0;

struct SignBitsUnpackCompileInfo {};

static ge::graphStatus TilingParseForSignBitsUnpack([[maybe_unused]] gert::TilingParseContext* context)
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

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t ubSize, int64_t coreNum, uint32_t& typeLengthOut, uint64_t& bufferOpen,
    uint64_t& inputNum, uint64_t& inputBytes, uint64_t& tileBlockNum, uint64_t& tileDataNum, uint64_t& inputLengthAlgin)
{
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    typeLengthOut = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetOutputDesc(0)->GetDataType(), typeLengthOut);
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0) {
        OP_LOGE(context, "inputNum is 0");
        return ge::GRAPH_FAILED;
    }
    inputBytes = inputLength / inputNum;
    uint64_t ubDataNumber = UB_NUM;
    bufferOpen = 1;
    inputLengthAlgin = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    // 重点
    // half uint8 + half * 8
    // 1 + 2 * 8 = 17
    uint64_t ubSizeTmp = ubSize - (((ubSize + 17 - 1) / 17) * 16);
    if (typeLengthOut == 4) {
        // 额外使用一块half
        // 1+ 2*8 + 4*8
        ubSizeTmp = ubSize - (((ubSize + 49 - 1) / 49) * 48);
    }
    if (inputLengthAlgin < coreNum * (((ubSizeTmp / BLOCK_SIZE) * BLOCK_SIZE) / 1U)) {
        ubDataNumber = 1U;
        bufferOpen = 0;
    }

    // double buffer需要重新计算输入所用的UB空间
    if (bufferOpen == 1) {
        ubSizeTmp = ubSize - (((ubSize + 34 - 1) / 34) * 32);
        if (typeLengthOut == 4) {
            // 1*2 + 2*8 + 4*8*2
            ubSizeTmp = ubSize - (((ubSize + 82 - 1) / 82) * 80);
        }
    }
    if (ubDataNumber == 0 || BLOCK_SIZE == 0) {
        OP_LOGE(context, "ubDataNumber or BLOCK_SIZE is 0");
        return ge::GRAPH_FAILED;
    }
    tileBlockNum = (ubSizeTmp / BLOCK_SIZE) / ubDataNumber;
    if (inputBytes == 0) {
        OP_LOGE(context, "inputBytes is 0");
        return ge::GRAPH_FAILED;
    }
    tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    gert::TilingContext* context, uint64_t inputLengthAlgin, int64_t coreNum, uint64_t tileBlockNum,
    uint64_t inputBytes, uint64_t tileDataNum, uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum,
    uint64_t& smallTailDataNum, uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum, uint64_t& finalBigTileNum,
    uint64_t& tailBlockNum)
{
    if (0 == BLOCK_SIZE || 0 == coreNum || 0 == tileBlockNum || 0 == inputBytes) {
        OP_LOGE(context, "BLOCK_SIZE or coreNum or tileBlockNum or inputBytes is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t everyCoreInputBlockNum = inputLengthAlgin / BLOCK_SIZE / coreNum;
    tailBlockNum = (inputLengthAlgin / BLOCK_SIZE) % coreNum;
    smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus SignBitsUnpackTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 2、获取shape、属性信息
    uint64_t bufferOpen;
    uint32_t typeLengthOut;
    uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin;
    ret = GetShapeAttrsInfo(
        context, ubSize, coreNum, typeLengthOut, bufferOpen, inputNum, inputBytes, tileBlockNum, tileDataNum,
        inputLengthAlgin);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    // 4、设置tiling信息
    SignBitsUnpackTilingData* tiling = context->GetTilingData<SignBitsUnpackTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SignBitsUnpackTilingData), 0, sizeof(SignBitsUnpackTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        coreNum =
            (static_cast<uint64_t>(coreNum) < inputLengthAlgin / BLOCK_SIZE) ? coreNum : inputLengthAlgin / BLOCK_SIZE;
    }
    // 计算每个core处理的数据块数
    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum,
        tailBlockNum;
    ret = CalculateCoreBlockNums(
        context, inputLengthAlgin, coreNum, tileBlockNum, inputBytes, tileDataNum, smallCoreDataNum, bigCoreDataNum,
        smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ge::GRAPH_FAILED);
    // 设置tiling数据
    tiling->smallCoreDataNum = static_cast<uint64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint64_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<uint64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<uint64_t>(tailBlockNum);

    tiling->bufferOpen = static_cast<uint64_t>(bufferOpen);

    context->SetBlockDim(coreNum);
    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    if (typeLengthOut == 2) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(SignBitsUnpack)
    .Tiling(SignBitsUnpackTilingFunc)
    .TilingParse<SignBitsUnpackCompileInfo>(TilingParseForSignBitsUnpack);
} // namespace optiling
