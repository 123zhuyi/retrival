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
 * \file bitwise_not_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/bitwise_not_tiling_data.h"
#include "../op_kernel/bitwise_not_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;

struct BitwiseNotCompileInfo {};

// 多核切分计算结果：与 BitwiseNotTilingData 字段一一对应，外加 coreNum / tilingKey 由调用方设置到 context。
struct BitwiseNotSplitParams {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t ubPartDataNum;
    uint32_t smallCoreTailDataNum;
    uint32_t bigCoreTailDataNum;
    uint32_t smallCoreLoopNum;
    uint32_t bigCoreLoopNum;
    uint32_t tailBlockNum;
    uint32_t lastCoreId;
    uint32_t lastCoreTailDataNum;
    uint32_t coreNum;
    uint32_t tilingKey;
};

// UB 单份可用数据量：in + out + BOOL 路径 half-tmp 统一保守预算 ubPartNum=3，按 BLOCK_SIZE 对齐。
// 返回 false 表示除数非法（dataTypeLength/BLOCK_SIZE 为 0）或结果为 0（ubPartBlockNum/ubPartDataNum 为 0，
// 二者后续会作除数）。调用方据此返回 GRAPH_FAILED，从而保证后续除法的除数非 0。
static bool ComputeUbPartDataNum(uint64_t ubLength, uint32_t dataTypeLength, uint32_t &ubPartBlockNum,
                                 uint32_t &ubPartDataNum)
{
    ubPartBlockNum = 0;
    ubPartDataNum = 0;
    if (dataTypeLength == 0 || BLOCK_SIZE == 0) {
        return false;
    }
    uint32_t ubPartNum = 3;
    uint32_t ubPartLength = ubLength / ubPartNum / BUFFER_NUM;
    ubPartBlockNum = ubPartLength / BLOCK_SIZE;
    if (ubPartBlockNum == 0) {
        return false;
    }
    ubPartDataNum = (ubPartBlockNum * BLOCK_SIZE) / dataTypeLength;
    return ubPartDataNum != 0;
}

// 多核切分 + 分块循环 + 尾块对齐：根据输入元素数、dtype 宽度、平台核数与 UB 单份能力计算各核切分参数。
// 返回 false 表示除数非法（dataTypeLength/BLOCK_SIZE/ubPartDataNum/ubPartBlockNum/coreNum 为 0），调用方据此失败返回。
static bool ComputeBitwiseNotSplit(uint32_t inputDataNum, uint32_t dataTypeLength, uint32_t platformCoreNum,
                                   uint32_t ubPartBlockNum, uint32_t ubPartDataNum, BitwiseNotSplitParams &p)
{
    // 防御除零：本函数内 dataTypeLength / BLOCK_SIZE / ubPartDataNum / ubPartBlockNum / coreNum 均作除数。
    if (dataTypeLength == 0 || BLOCK_SIZE == 0 || ubPartDataNum == 0 || ubPartBlockNum == 0) {
        return false;
    }

    uint32_t inputLength = inputDataNum * dataTypeLength;
    // 输入数据 32B 对齐
    uint32_t inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    uint32_t coreNum = platformCoreNum;
    if (ubPartDataNum >= inputDataNum) {
        coreNum = 1;
    } else {
        // 每核至少 32B 数据，核数取 min(平台核数, 数据块数)
        coreNum = (coreNum < inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
    }
    if (coreNum == 0) {
        return false;
    }

    uint32_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    p.tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;

    // small core 数据量与分块循环
    p.smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
    p.smallCoreLoopNum = p.smallCoreDataNum / ubPartDataNum;
    p.smallCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? p.smallCoreLoopNum : p.smallCoreLoopNum + 1;
    p.smallCoreTailDataNum = p.smallCoreDataNum - ubPartDataNum * (p.smallCoreLoopNum - 1);
    p.smallCoreTailDataNum = p.smallCoreTailDataNum == 0 ? ubPartDataNum : p.smallCoreTailDataNum;

    p.bigCoreDataNum = 0;
    p.bigCoreLoopNum = 0;
    p.bigCoreTailDataNum = 0;
    if (0 != p.tailBlockNum) {
        everyCoreInputBlockNum += 1;
        p.bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
        p.bigCoreLoopNum = p.bigCoreDataNum / ubPartDataNum;
        p.bigCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? p.bigCoreLoopNum : p.bigCoreLoopNum + 1;
        p.bigCoreTailDataNum = p.bigCoreDataNum - ubPartDataNum * (p.bigCoreLoopNum - 1);
        p.bigCoreTailDataNum = p.bigCoreTailDataNum == 0 ? ubPartDataNum : p.bigCoreTailDataNum;
        p.tilingKey = 1;
    } else {
        p.tilingKey = 0;
    }

    // 尾块对齐处理：32B 块对齐切分使各核名义元素数之和 = alignedTotal（= inputLengthAlign32/dataTypeLength），
    // 比真实 inputDataNum 多 pad(<1 block, 按原 dtype 元素数) 个对齐填充元素。该 pad 只落在 GM 序最后一个核
    // （coreNum-1，必为最后一个 small core）的最后一个 tile：改用真实剩余元素数 lastCoreTailDataNum
    // = smallCoreTailDataNum - pad（可非 32B 对齐 → kernel 走 DataCopyPad，按原 dtype 元素数写回），
    // 既不越界写末核、也不漏写非末核。smallCoreTailDataNum >= 1 block > pad，有真实元素时差值 >= 1，安全非负。
    uint32_t alignedTotalDataNum = inputLengthAlign32 / dataTypeLength;
    uint32_t pad = alignedTotalDataNum - inputDataNum;
    p.lastCoreId = coreNum - 1;
    p.lastCoreTailDataNum = p.smallCoreTailDataNum - pad;

    p.ubPartDataNum = ubPartDataNum;
    p.coreNum = coreNum;
    return true;
}

// 把切分结果写入 TilingData（isBool 由调用方按 dtype 传入）。
static void SetBitwiseNotTilingData(BitwiseNotTilingData *tiling, const BitwiseNotSplitParams &p, uint32_t isBool)
{
    tiling->smallCoreDataNum = p.smallCoreDataNum;
    tiling->bigCoreDataNum = p.bigCoreDataNum;
    tiling->ubPartDataNum = p.ubPartDataNum;
    tiling->smallCoreTailDataNum = p.smallCoreTailDataNum;
    tiling->bigCoreTailDataNum = p.bigCoreTailDataNum;
    tiling->smallCoreLoopNum = p.smallCoreLoopNum;
    tiling->bigCoreLoopNum = p.bigCoreLoopNum;
    tiling->tailBlockNum = p.tailBlockNum;
    tiling->isBool = isBool;
    tiling->lastCoreId = p.lastCoreId;
    tiling->lastCoreTailDataNum = p.lastCoreTailDataNum;
}

static ge::graphStatus BitwiseNotTilingFunc(gert::TilingContext *context)
{
    BitwiseNotTilingData *tiling = context->GetTilingData<BitwiseNotTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BitwiseNotTilingData), 0, sizeof(BitwiseNotTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint64_t ubLength = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto coreNum = ascendcPlatform.GetCoreNum();

    auto inputDtype = context->GetInputDesc(0)->GetDataType();
    uint32_t inputDataNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(inputDtype, dataTypeLength);
    if (coreNum == 0 || BLOCK_SIZE == 0 || dataTypeLength == 0) {
        OP_LOGE(context, "coreNum or BLOCK_SIZE or dataTypeLength is 0");
        return ge::GRAPH_FAILED;
    }

    // BOOL 与 INT8 在 kernel 承载类型同为 int8_t，需显式区分语义。
    uint32_t isBool = (inputDtype == ge::DT_BOOL) ? 1u : 0u;

    // 空 Tensor（0 元素）：清零（已 memset）+ 单核 + key0，kernel loopCount<=0 安全空跑；
    // aclnn 壳对 IsEmpty 已 workspace=0 提前返回，kernel 实际不被拉起，双重保险。
    if (inputDataNum == 0) {
        tiling->isBool = isBool;
        context->SetBlockDim(1);
        context->SetTilingKey(0);
        size_t *currentWorkspaceEmpty = context->GetWorkspaceSizes(1);
        currentWorkspaceEmpty[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    uint32_t ubPartBlockNum = 0;
    uint32_t ubPartDataNum = 0;
    if (!ComputeUbPartDataNum(ubLength, dataTypeLength, ubPartBlockNum, ubPartDataNum)) {
        OP_LOGE(context, "ubPartDataNum is 0");
        return ge::GRAPH_FAILED;
    }

    BitwiseNotSplitParams params;
    if (!ComputeBitwiseNotSplit(inputDataNum, dataTypeLength, coreNum, ubPartBlockNum, ubPartDataNum, params)) {
        OP_LOGE(context, "invalid divisor in tiling split (dataTypeLength/ubPartDataNum/ubPartBlockNum/coreNum is 0)");
        return ge::GRAPH_FAILED;
    }

    SetBitwiseNotTilingData(tiling, params, isBool);
    context->SetTilingKey(params.tilingKey);
    context->SetBlockDim(params.coreNum);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBitwiseNot([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BitwiseNot)
    .Tiling(BitwiseNotTilingFunc)
    .TilingParse<BitwiseNotCompileInfo>(TilingParseForBitwiseNot);
} // namespace optiling
