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
 * \file rsqrt_grad_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/rsqrt_grad_tiling_data.h"
#include "../op_kernel/rsqrt_grad_tiling_key.h"

namespace optiling {

// SINGLE_BUFFER
#define UB_NUM_INT32_ONE 1U
#define UB_NUM_FLOAT_ONE 3U
#define UB_NUM_INT8_ONE 22U
#define UB_NUM_FLOAT16_BF16_ONE 7U
// DOUBLE_BUFFER
#define UB_NUM_INT32_TWO 2U
#define UB_NUM_FLOAT_TWO 6U
#define UB_NUM_INT8_TWO 24U
#define UB_NUM_FLOAT16_BF16_TWO 10U
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0;

struct RsqrtGradCompileInfo {};

static ge::graphStatus TilingParseForRsqrtGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum, uint64_t& BLOCK_SIZE)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    BLOCK_SIZE = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(BLOCK_SIZE == 0, OP_LOGE(context, "BLOCK_SIZE is 0"), return ge::GRAPH_FAILED);
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
    gert::TilingContext* context, uint64_t BLOCK_SIZE, uint64_t ubSize, int64_t coreNum, uint64_t& bufferOpen,
    uint64_t& inputNum, uint64_t& inputBytes, uint64_t& tileBlockNum, uint64_t& tileDataNum, uint64_t& inputLengthAlgin)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context->GetInputShape(0) == nullptr, OP_LOGE(context, "context->GetInputShape(0) is nullptr"),
        return ge::GRAPH_FAILED);
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0 || BLOCK_SIZE == 0) {
        OP_LOGE(context, "inputNum or BLOCK_SIZE is 0");
        return ge::GRAPH_FAILED;
    }
    inputBytes = inputLength / inputNum;
    uint64_t ubDataNumber = 0U;
    inputLengthAlgin = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    // buffer 是否开启，每个核上对齐后的数据量小于ub空间时不开启，默认开启
    bufferOpen = 1;
    if (UB_NUM_INT8_ONE == 0 || UB_NUM_FLOAT16_BF16_ONE == 0) {
        OP_LOGE(context, "UB_NUM_INT8_ONE or UB_NUM_FLOAT16_BF16_ONE is 0");
        return ge::GRAPH_FAILED;
    }
    if (UB_NUM_INT32_ONE == 0 || UB_NUM_FLOAT_ONE == 0) {
        OP_LOGE(context, "UB_NUM_INT32_ONE or UB_NUM_FLOAT_ONE is 0");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDesc(0)->GetDataType() == ge::DT_INT8) {
        if (inputLengthAlgin < coreNum * (((ubSize / BLOCK_SIZE) * BLOCK_SIZE) / UB_NUM_INT8_ONE)) {
            bufferOpen = 0;
            ubDataNumber = UB_NUM_INT8_ONE;
        } else {
            ubDataNumber = UB_NUM_INT8_TWO;
        }
    } else if (
        (context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) ||
        (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16)) {
        if (inputLengthAlgin < coreNum * (((ubSize / BLOCK_SIZE) * BLOCK_SIZE) / UB_NUM_FLOAT16_BF16_ONE)) {
            bufferOpen = 0;
            ubDataNumber = UB_NUM_FLOAT16_BF16_ONE;
        } else {
            ubDataNumber = UB_NUM_FLOAT16_BF16_TWO;
        }
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_INT32) {
        if (inputLengthAlgin < coreNum * (((ubSize / BLOCK_SIZE) * BLOCK_SIZE) / UB_NUM_INT32_ONE)) {
            bufferOpen = 0;
            ubDataNumber = UB_NUM_INT32_ONE;
        } else {
            ubDataNumber = UB_NUM_INT32_TWO;
        }
    } else {
        if (inputLengthAlgin < coreNum * (((ubSize / BLOCK_SIZE) * BLOCK_SIZE) / UB_NUM_FLOAT_ONE)) {
            bufferOpen = 0;
            ubDataNumber = UB_NUM_FLOAT_ONE;
        } else {
            ubDataNumber = UB_NUM_FLOAT_TWO;
        }
    }
    if (ubDataNumber == 0 || inputBytes == 0) {
        OP_LOGE(context, "ubDataNumber or inputBytes is 0");
        return ge::GRAPH_FAILED;
    }
    tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    gert::TilingContext* context, uint64_t BLOCK_SIZE, uint64_t inputLengthAlgin, int64_t coreNum,
    uint64_t tileBlockNum, uint64_t inputBytes, uint64_t tileDataNum, uint64_t& smallCoreDataNum,
    uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum, uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum,
    uint64_t& finalBigTileNum, uint64_t& tailBlockNum)
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
static ge::graphStatus RsqrtGradTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize, BLOCK_SIZE;
    int64_t coreNum;
    uint64_t bufferOpen;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum, BLOCK_SIZE);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 2、获取shape、属性信息
    uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin;
    ret = GetShapeAttrsInfo(
        context, BLOCK_SIZE, ubSize, coreNum, bufferOpen, inputNum, inputBytes, tileBlockNum, tileDataNum,
        inputLengthAlgin);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    // 4、设置tiling信息
    RsqrtGradTilingData* tiling = context->GetTilingData<RsqrtGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(RsqrtGradTilingData), 0, sizeof(RsqrtGradTilingData)) != EOK,
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
        context, BLOCK_SIZE, inputLengthAlgin, coreNum, tileBlockNum, inputBytes, tileDataNum, smallCoreDataNum,
        bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);
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
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(RsqrtGrad).Tiling(RsqrtGradTilingFunc).TilingParse<RsqrtGradCompileInfo>(TilingParseForRsqrtGrad);
} // namespace optiling
