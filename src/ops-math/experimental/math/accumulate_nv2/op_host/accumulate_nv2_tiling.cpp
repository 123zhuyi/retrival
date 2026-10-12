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
 * \file accumulate_nv2_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/accumulate_nv2_tiling_data.h"
#include "../op_kernel/accumulate_nv2_tiling_key.h"

namespace optiling {

#define BLOCK_SIZE 32U
#define BLOCK_SIZE_LARGE 512U

const uint32_t WS_SYS_SIZE = 0;
const uint32_t DATA_NUM = 5;
const uint32_t DATA_NUM_32B = 2;
const uint32_t TILE_SPLIT_NUM = 1024;
const uint32_t TILE_SPLIT_NUM_8B = 2048;
const uint32_t SINGLE_BUFFER_NUM = 1;
const uint32_t DOUBLE_BUFFER_NUM = 2;
struct AccumulateNv2CompileInfo {};

static ge::graphStatus TilingParseForAccumulateNv2([[maybe_unused]] gert::TilingParseContext* context)
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
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t ubSize, uint64_t coreNum, uint64_t& inputNum, uint64_t& inputBytes,
    uint64_t& tileBlockNum, uint64_t& tileDataNum, uint64_t& inputLengthAlgin, uint32_t& bufferNum,
    uint32_t& typeLength, uint64_t& alginLen)
{
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    alginLen = BLOCK_SIZE;
    uint64_t ubDataNumber = DATA_NUM_32B;
    if (context->GetInputDesc(0)->GetDataType() == ge::DT_UINT8 ||
        context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16 ||
        context->GetInputDesc(0)->GetDataType() == ge::DT_INT8) {
        alginLen = BLOCK_SIZE_LARGE;
        ubDataNumber = DATA_NUM;
    }
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0 || alginLen == 0) {
        OP_LOGE(context, "inputNum is 0 or alginLen is 0");
        return ge::GRAPH_FAILED;
    }
    inputBytes = typeLength;

    inputLengthAlgin = (((inputLength + alginLen - 1) / alginLen) * alginLen);
    // 计算单流水所需的总UB空间
    uint64_t singleBufferNeedSize = inputLengthAlgin * ubDataNumber;

    // 动态决定buffer数量：如果UB空间足够，使用单流水；否则使用双流水
    if (singleBufferNeedSize <= coreNum * ubSize) {
        // UB空间足够，使用单流水以获得更好性能
        bufferNum = SINGLE_BUFFER_NUM;
    } else {
        // UB空间不足，使用双流水
        bufferNum = DOUBLE_BUFFER_NUM;
    }

    bufferNum = SINGLE_BUFFER_NUM;

    if (ubDataNumber == 0 || bufferNum == 0) {
        OP_LOGE(context, "ubDataNumber or bufferNum is 0");
        return ge::GRAPH_FAILED;
    }

    // 根据选择的buffer数量计算tile参数
    tileBlockNum = (ubSize / bufferNum / alginLen) / ubDataNumber;
    if (typeLength == 0) {
        OP_LOGE(context, "typeLength is 0");
        return ge::GRAPH_FAILED;
    }
    tileDataNum = (tileBlockNum * alginLen) / typeLength;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    gert::TilingContext* context, uint64_t inputLengthAlgin, int64_t coreNum, uint64_t tileBlockNum,
    uint64_t inputBytes, uint64_t tileDataNum, uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum,
    uint64_t& smallTailDataNum, uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum, uint64_t& finalBigTileNum,
    uint64_t& tailBlockNum, uint64_t alginLen)
{
    if (0 == alginLen || 0 == coreNum || 0 == tileBlockNum || 0 == inputBytes) {
        OP_LOGE(context, "alginLen, coreNum, tileBlockNum, inputBytes cannot be zero");
        return ge::GRAPH_FAILED;
    }

    uint64_t everyCoreInputBlockNum = inputLengthAlgin / alginLen / coreNum;
    tailBlockNum = (inputLengthAlgin / alginLen) % coreNum;

    smallCoreDataNum = everyCoreInputBlockNum * alginLen / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * alginLen / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AccumulateNv2TilingFunc(gert::TilingContext* context)
{
    AccumulateNv2TilingData* tiling = context->GetTilingData<AccumulateNv2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AccumulateNv2TilingData), 0, sizeof(AccumulateNv2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ret);

    // 获取输入数据信息和动态决定buffer数量
    uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin, alginLen;
    uint32_t bufferNum, typeLength = 0;

    ret = GetShapeAttrsInfo(
        context, ubSize, coreNum, inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin, bufferNum,
        typeLength, alginLen);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ret);

    uint32_t unitNum = typeLength == 1 ? TILE_SPLIT_NUM_8B : TILE_SPLIT_NUM;
    // 计算coreNum
    uint64_t calcCoreNum = inputNum / unitNum;

    if (inputNum % unitNum)
        calcCoreNum = calcCoreNum + 1;
    coreNum = (calcCoreNum < static_cast<uint64_t>(coreNum)) ? calcCoreNum : coreNum;

    // 计算每个core处理的数据块数
    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
    uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;
    ret = CalculateCoreBlockNums(
        context, inputLengthAlgin, coreNum, tileBlockNum, inputBytes, tileDataNum, smallCoreDataNum, bigCoreDataNum,
        smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum, alginLen);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ret);

    // 设置tiling数据
    tiling->smallCoreDataNum = (uint32_t)smallCoreDataNum;
    tiling->bigCoreDataNum = (uint32_t)bigCoreDataNum;
    tiling->tileDataNum = (uint32_t)tileDataNum;
    tiling->smallTailDataNum = (uint32_t)smallTailDataNum;
    tiling->bigTailDataNum = (uint32_t)bigTailDataNum;
    tiling->finalSmallTileNum = (uint32_t)finalSmallTileNum;
    tiling->finalBigTileNum = (uint32_t)finalBigTileNum;
    tiling->tailBlockNum = (uint32_t)tailBlockNum;
    tiling->num = uint32_t(context->GetComputeNodeInputNum());

    // 计算workspace大小
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    uint64_t tilingKey = (tiling->num == 1) ? GET_TPL_TILING_KEY(0) : GET_TPL_TILING_KEY(1);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);

    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(AccumulateNv2)
    .Tiling(AccumulateNv2TilingFunc)
    .TilingParse<AccumulateNv2CompileInfo>(TilingParseForAccumulateNv2);
} // namespace optiling
