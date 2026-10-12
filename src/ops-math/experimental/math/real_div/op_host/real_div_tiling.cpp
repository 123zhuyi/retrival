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
 * \file real_div_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "../op_kernel/real_div_tiling_data.h"
#include "../op_kernel/real_div_tiling_key.h"

namespace optiling {

static uint32_t BLOCK_SIZE = 32;
const uint32_t WS_SYS_SIZE = 0;
const uint32_t DATA_NUM_32B = 3;
const uint32_t DATA_NUM_16B = 5;
const uint32_t DATA_NUM_8B = 12;
const uint32_t TILE_SPLIT_NUM = 1024;
const uint32_t SINGLE_BUFFER_NUM = 1;
const uint32_t DOUBLE_BUFFER_NUM = 2;
struct RealDivCompileInfo {};

static ge::graphStatus TilingParseForRealDiv([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "platformInfo is nullptr"), return ge::GRAPH_FAILED);
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
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
    uint64_t& tileBlockNum, uint64_t& tileDataNum, uint64_t& inputLengthAlgin32, uint32_t& bufferNum)
{
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    inputBytes = inputLength / inputNum;
    auto dataType = context->GetInputDesc(0)->GetDataType();

    // 根据数据类型确定每个元素需要的UB数据量倍数
    uint64_t ubDataNumber = DATA_NUM_32B;
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            ubDataNumber = DATA_NUM_32B; // 输入1 + 输入2 + 输出
            break;
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
            ubDataNumber = DATA_NUM_16B; // 可能需要更多中间缓冲区
            break;
        case ge::DT_BOOL:
        case ge::DT_INT8:
        case ge::DT_UINT8:
            ubDataNumber = DATA_NUM_8B; // 小数据类型可能需要更多处理单元
            break;
        default:
            break;
    }

    inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    // 计算单流水所需的总UB空间
    uint64_t singleBufferNeedSize = inputLengthAlgin32 * ubDataNumber;

    // 动态决定buffer数量：如果UB空间足够，使用单流水；否则使用双流水
    if (singleBufferNeedSize <= coreNum * ubSize) {
        // UB空间足够，使用单流水以获得更好性能
        bufferNum = SINGLE_BUFFER_NUM;
    } else {
        // UB空间不足，使用双流水
        bufferNum = DOUBLE_BUFFER_NUM;
    }

    if (ubDataNumber == 0 || bufferNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // 根据选择的buffer数量计算tile参数
    tileBlockNum = (ubSize / bufferNum / BLOCK_SIZE) / ubDataNumber;
    if (inputBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    uint64_t inputLengthAlgin32, int64_t coreNum, uint64_t tileBlockNum, uint64_t inputBytes, uint64_t tileDataNum,
    uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum, uint64_t& bigTailDataNum,
    uint64_t& finalSmallTileNum, uint64_t& finalBigTileNum, uint64_t& tailBlockNum)
{
    if (0 == BLOCK_SIZE || 0 == coreNum || 0 == tileBlockNum || 0 == inputBytes) {
        return ge::GRAPH_FAILED;
    }

    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;

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

static ge::graphStatus RealDivTilingFunc(gert::TilingContext* context)
{
    RealDivTilingData* tiling = context->GetTilingData<RealDivTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(RealDivTilingData), 0, sizeof(RealDivTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 获取平台运行信息
    // 获取输入数据信息和动态决定buffer数量
    uint64_t ubSize, inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin32;
    int64_t coreNum;
    uint32_t bufferNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ret);

    ret = GetShapeAttrsInfo(
        context, ubSize, coreNum, inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin32, bufferNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ret);

    // 计算coreNum
    uint64_t calcCoreNum = inputNum / TILE_SPLIT_NUM;

    if (inputNum % TILE_SPLIT_NUM)
        calcCoreNum = calcCoreNum + 1;
    coreNum = (calcCoreNum < static_cast<uint64_t>(coreNum)) ? calcCoreNum : coreNum;

    // 计算每个core处理的数据块数
    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum,
        tailBlockNum;
    ret = CalculateCoreBlockNums(
        inputLengthAlgin32, coreNum, tileBlockNum, inputBytes, tileDataNum, smallCoreDataNum, bigCoreDataNum,
        smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);
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
    tiling->bufferNum = (uint64_t)bufferNum; // 添加buffer数量到tiling数据中

    // 计算workspace大小
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    uint64_t tilingKey = 0;
    if (bufferNum == DOUBLE_BUFFER_NUM) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);

    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(RealDiv).Tiling(RealDivTilingFunc).TilingParse<RealDivCompileInfo>(TilingParseForRealDiv);
} // namespace optiling