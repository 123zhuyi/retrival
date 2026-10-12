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
 * \file select_v2_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "../op_kernel/select_v2_tiling_data.h"
#include "../op_kernel/select_v2_tiling_key.h"

namespace optiling {

#define BLOCK_SIZE 32U
#define UB_DATA_NUM_UINT8 16U // 对应DT_FLOAT, DT_INT32, DT_UINT32类型的ub分块数量
#define UB_DATA_NUM_OTHER 8U  // 对应DT_FLOAT16, DT_BFLOAT16, DT_INT16, DT_UINT16其他数据类型的ub分块数量
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0;
constexpr uint32_t RESERVED_UB_SIZE = 8 * 1024; // ub固定预留8k
struct SelectV2CompileInfo {};

static ge::graphStatus TilingParseForSelectV2([[maybe_unused]] gert::TilingParseContext* context)
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

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t ubSize, uint64_t& inputNum, uint64_t& inputBytes, uint64_t& tileBlockNum,
    uint64_t& tileDataNum, uint64_t& inputLengthAlgin32)
{
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(1)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    inputBytes = inputLength / inputNum;
    uint64_t ubDataNumber = (typeLength == sizeof(uint8_t)) ? UB_DATA_NUM_UINT8 : UB_DATA_NUM_OTHER;
    tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    if (inputBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;
    inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
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

static ge::graphStatus SelectV2TilingFunc(gert::TilingContext* context)
{
    // SelectV2TilingData tiling;
    SelectV2TilingData* tiling = context->GetTilingData<SelectV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SelectV2TilingData), 0, sizeof(SelectV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 获取输入数据信息
    ubSize = ubSize - RESERVED_UB_SIZE;
    uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin32;
    ret = GetShapeAttrsInfo(context, ubSize, inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin32);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 计算coreNum
    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number
        // of audits is the actual number of audits
        coreNum = (static_cast<uint64_t>(coreNum) < inputLengthAlgin32 / BLOCK_SIZE) ? coreNum :
                                                                                       inputLengthAlgin32 / BLOCK_SIZE;
    }
    // 计算每个core处理的数据块数
    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
    uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;
    ret = CalculateCoreBlockNums(
        inputLengthAlgin32, coreNum, tileBlockNum, inputBytes, tileDataNum, smallCoreDataNum, bigCoreDataNum,
        smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ge::GRAPH_FAILED);
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
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(SelectV2).Tiling(SelectV2TilingFunc).TilingParse<SelectV2CompileInfo>(TilingParseForSelectV2);
} // namespace optiling
