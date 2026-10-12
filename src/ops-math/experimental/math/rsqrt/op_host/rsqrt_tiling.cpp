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
 * \file rsqrt_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/rsqrt_tiling_data.h"
#include "../op_kernel/rsqrt_tiling_key.h"

namespace optiling {

#define BLOCK_SIZE 32U
const uint32_t DATA_NUM_INT32 = 2U;
const uint32_t DATA_NUM_FP32 = 3U;
const uint32_t DATA_NUM_FP16 = 3U;
const uint32_t DATA_NUM_BF16 = 6U;
const uint32_t DATA_NUM_INT8 = 6U;
const uint32_t DATA_NUM_BOOL = 2U;
const uint32_t DATA_NUM_INT16 = 2U;
const uint32_t DATA_NUM_UINT8 = 4U;
const uint32_t TILE_SPLIT_NUM = 1024U;
const uint32_t SINGLE_BUFFER_NUM = 1U;
const uint32_t DOUBLE_BUFFER_NUM = 2U;
const uint32_t UB_RESERVED_BYTE = 1024U;
struct RsqrtCompileInfo {};

static ge::graphStatus TilingParseForRsqrt([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDataTypeUbFactor(ge::DataType dataType, uint64_t& ubDataNumber)
{
    switch (dataType) {
        case ge::DT_INT32:
            ubDataNumber = DATA_NUM_INT32;
            break;
        case ge::DT_FLOAT:
            ubDataNumber = DATA_NUM_FP32;
            break;
        case ge::DT_BF16:
            ubDataNumber = DATA_NUM_BF16;
            break;
        case ge::DT_FLOAT16:
            ubDataNumber = DATA_NUM_FP16;
            break;
        case ge::DT_INT8:
            ubDataNumber = DATA_NUM_INT8;
            break;
        case ge::DT_BOOL:
            ubDataNumber = DATA_NUM_BOOL;
            break;
        case ge::DT_INT16:
            ubDataNumber = DATA_NUM_INT16;
            break;
        case ge::DT_UINT8:
            ubDataNumber = DATA_NUM_UINT8;
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static uint64_t AlignToBlockSize(uint64_t length) { return (((length + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE); }

static ge::graphStatus DetermineBufferNum(uint64_t singleBufferNeedSize, uint64_t ubSize, uint64_t coreNum,
                                          uint32_t& bufferNum)
{
    if (singleBufferNeedSize == 0) {
        return ge::GRAPH_FAILED;
    }

    if (singleBufferNeedSize <= coreNum * ubSize) {
        bufferNum = SINGLE_BUFFER_NUM;
    } else {
        bufferNum = DOUBLE_BUFFER_NUM;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateTileParams(uint64_t ubSize, uint32_t bufferNum, uint64_t ubDataNumber,
                                           uint64_t elementBytes, uint64_t& tileBlockNum, uint64_t& tileDataNum)
{
    if (bufferNum == 0 || ubDataNumber == 0 || elementBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    tileBlockNum = (ubSize / bufferNum / BLOCK_SIZE) / ubDataNumber;
    if (tileBlockNum == 0) {
        return ge::GRAPH_FAILED;
    }

    tileDataNum = (tileBlockNum * BLOCK_SIZE) / elementBytes;
    if (tileDataNum == 0) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, uint64_t coreNum,
                                         uint64_t& inputNum, uint64_t& elementBytes, uint64_t& tileBlockNum,
                                         uint64_t& tileDataNum, uint64_t& inputLengthAlgin, uint32_t& bufferNum)
{
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    if (typeLength == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t inputLength = inputNum * typeLength;
    elementBytes = typeLength;
    if (elementBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubDataNumber = 0;
    auto dataType = context->GetInputDesc(0)->GetDataType();
    if (GetDataTypeUbFactor(dataType, ubDataNumber) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    inputLengthAlgin = AlignToBlockSize(inputLength);
    uint64_t singleBufferNeedSize = inputLengthAlgin * ubDataNumber;

    if (DetermineBufferNum(singleBufferNeedSize, ubSize, coreNum, bufferNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CalculateTileParams(ubSize, bufferNum, ubDataNumber, elementBytes, tileBlockNum, tileDataNum) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(uint64_t inputLengthAlgin, int64_t coreNum, uint64_t tileBlockNum,
                                              uint64_t elementBytes, uint64_t tileDataNum, uint64_t& smallCoreDataNum,
                                              uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum,
                                              uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum,
                                              uint64_t& finalBigTileNum, uint64_t& tailBlockNum)
{
    if (0 == BLOCK_SIZE || 0 == coreNum || 0 == tileBlockNum || 0 == elementBytes) {
        return ge::GRAPH_FAILED;
    }

    uint64_t everyCoreInputBlockNum = inputLengthAlgin / BLOCK_SIZE / coreNum;
    tailBlockNum = (inputLengthAlgin / BLOCK_SIZE) % coreNum;

    smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / elementBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / elementBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RsqrtTilingFunc(gert::TilingContext* context)
{
    RsqrtTilingData* tiling = context->GetTilingData<RsqrtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(RsqrtTilingData), 0, sizeof(RsqrtTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ret);

    // 获取输入数据信息和动态决定buffer数量
    uint64_t inputNum, elementBytes, tileBlockNum, tileDataNum, inputLengthAlgin;
    uint32_t bufferNum;
    ubSize -= UB_RESERVED_BYTE;
    ret = GetShapeAttrsInfo(context, ubSize, coreNum, inputNum, elementBytes, tileBlockNum, tileDataNum,
                            inputLengthAlgin, bufferNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ret);

    // 计算coreNum
    uint64_t calcCoreNum = inputNum / TILE_SPLIT_NUM;

    if (inputNum % TILE_SPLIT_NUM)
        calcCoreNum = calcCoreNum + 1;
    coreNum = (calcCoreNum < static_cast<uint64_t>(coreNum)) ? calcCoreNum : coreNum;

    // 计算每个core处理的数据块数
    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
    uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;
    ret = CalculateCoreBlockNums(inputLengthAlgin, coreNum, tileBlockNum, elementBytes, tileDataNum, smallCoreDataNum,
                                 bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum,
                                 tailBlockNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ret);

    // 设置tiling数据
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;

    // 计算workspace大小
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);
    uint64_t tilingKey = 0;
    if (bufferNum == DOUBLE_BUFFER_NUM) {
        tilingKey = GET_TPL_TILING_KEY(0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(1);
    }
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);

    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Rsqrt).Tiling(RsqrtTilingFunc).TilingParse<RsqrtCompileInfo>(TilingParseForRsqrt);
} // namespace optiling
