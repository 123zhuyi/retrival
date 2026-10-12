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
 * \file tanh_grad_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "../op_kernel/tanh_grad_tiling_data.h"
#include "../op_kernel/tanh_grad_tiling_key.h"

namespace optiling {

#define UB_DATA_NUM_FLOAT 10U // 对应DT_FLOAT类型的ub分块数量
#define UB_DATA_NUM_OTHER 16U // 对应DT_FLOAT16, DT_BF16类型的ub分块数量
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0;
struct TanhGradCompileInfo {};

static ge::graphStatus TilingParseForTanhGrad([[maybe_unused]] gert::TilingParseContext* context)
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
    // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t ubSize, uint64_t blockSize, uint64_t& inputNum, uint64_t& inputBytes,
    uint64_t& tileBlockNum, uint64_t& tileDataNum, uint64_t& inputLengthAlginBlock)
{
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);

    // 检查 blockSize 的有效性
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);

    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    inputBytes = typeLength; // 修改：直接使用 typeLength 而不是 inputLength/inputNum
    OP_CHECK_IF(inputBytes == 0, OP_LOGE(context, "inputBytes is 0"), return ge::GRAPH_FAILED);

    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubDataNumber =
        (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) ? UB_DATA_NUM_FLOAT : UB_DATA_NUM_OTHER;

    // 检查 ubDataNumber 不为 0
    OP_CHECK_IF(ubDataNumber == 0, OP_LOGE(context, "ubDataNumber is 0"), return ge::GRAPH_FAILED);

    // 计算 tileBlockNum，确保不为 0
    uint64_t blockPerUb = ubSize / blockSize;
    tileBlockNum = blockPerUb / ubDataNumber;
    // 确保 tileBlockNum 至少为 1，避免后续除零错误
    if (tileBlockNum == 0) {
        tileBlockNum = 1;
    }

    // 计算 tileDataNum
    uint64_t blockBytes = tileBlockNum * blockSize;
    tileDataNum = blockBytes / inputBytes;
    // 确保 tileDataNum 至少为 1
    if (tileDataNum == 0) {
        tileDataNum = 1;
    }

    // 计算输入数据总字节数
    uint64_t inputLength = inputNum * inputBytes;

    // 计算对齐长度，确保 blockSize 不为 0
    inputLengthAlginBlock = ((inputLength + blockSize - 1) / blockSize) * blockSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    uint64_t inputLengthAlginBlock, int64_t coreNum, uint64_t tileBlockNum, uint64_t inputBytes, uint64_t tileDataNum,
    uint64_t blockSize, uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum,
    uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum, uint64_t& finalBigTileNum, uint64_t& tailBlockNum)
{
    // 检查参数有效性
    if (blockSize == 0 || coreNum == 0 || tileBlockNum == 0 || inputBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    // 检查 tileDataNum 不为 0
    if (tileDataNum == 0) {
        tileDataNum = 1;
    }

    // 计算总块数
    uint64_t totalBlocks = inputLengthAlginBlock / blockSize;

    // 计算每个核心的基础块数和尾块数
    uint64_t baseBlocksPerCore = totalBlocks / coreNum;
    tailBlockNum = totalBlocks % coreNum;

    // 计算小核心数据
    if (baseBlocksPerCore > 0) {
        smallCoreDataNum = baseBlocksPerCore * blockSize / inputBytes;
        // 确保 smallCoreDataNum 有效
        if (smallCoreDataNum == 0) {
            smallCoreDataNum = 1;
        }

        uint64_t smallTileNum = baseBlocksPerCore / tileBlockNum;
        if (tileBlockNum > 0) {
            finalSmallTileNum = (baseBlocksPerCore % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
            smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
            if (smallTailDataNum == 0) {
                smallTailDataNum = tileDataNum;
            }
        } else {
            finalSmallTileNum = 1;
            smallTailDataNum = smallCoreDataNum;
        }
    } else {
        smallCoreDataNum = 0;
        smallTailDataNum = 0;
        finalSmallTileNum = 0;
    }

    // 计算大核心数据（处理额外一个块的核心）
    if (baseBlocksPerCore + 1 > 0) {
        bigCoreDataNum = (baseBlocksPerCore + 1) * blockSize / inputBytes;
        // 确保 bigCoreDataNum 有效
        if (bigCoreDataNum == 0) {
            bigCoreDataNum = 1;
        }

        uint64_t bigTileNum = (baseBlocksPerCore + 1) / tileBlockNum;
        if (tileBlockNum > 0) {
            finalBigTileNum = ((baseBlocksPerCore + 1) % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
            bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
            if (bigTailDataNum == 0) {
                bigTailDataNum = tileDataNum;
            }
        } else {
            finalBigTileNum = 1;
            bigTailDataNum = bigCoreDataNum;
        }
    } else {
        bigCoreDataNum = 0;
        bigTailDataNum = 0;
        finalBigTileNum = 0;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TanhGradTilingFunc(gert::TilingContext* context)
{
    TanhGradTilingData* tiling = context->GetTilingData<TanhGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(TanhGradTilingData), 0, sizeof(TanhGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 动态获取 blockSize
    uint64_t blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "GetUbBlockSize returned 0"), return ge::GRAPH_FAILED);

    // 获取输入数据信息，传入动态获取的 blockSize
    uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlginBlock;
    ret = GetShapeAttrsInfo(
        context, ubSize, blockSize, inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlginBlock);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 计算coreNum，使用动态获取的 blockSize
    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        coreNum = (static_cast<uint64_t>(coreNum) < inputLengthAlginBlock / blockSize) ?
                      coreNum :
                      inputLengthAlginBlock / blockSize;
    }

    // 计算每个core处理的数据块数，传入动态获取的 blockSize
    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
    uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;
    ret = CalculateCoreBlockNums(
        inputLengthAlginBlock, coreNum, tileBlockNum, inputBytes, tileDataNum, blockSize, smallCoreDataNum,
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

    // 计算workspace大小
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(TanhGrad).Tiling(TanhGradTilingFunc).TilingParse<TanhGradCompileInfo>(TilingParseForTanhGrad);
} // namespace optiling