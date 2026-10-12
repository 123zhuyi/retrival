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
 * \file neg_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/neg_tiling_data.h"
#include "../op_kernel/neg_tiling_key.h"

namespace optiling {

const uint64_t BLOCK_SIZE = 32;
const uint64_t INT64_UB_PART_NUM = 7;
const uint64_t INT64_COMPARE_ALIGN_NUM = 32;
const uint64_t INT64_MAX_REPEAT_TIMES = 255;

struct NegCompileInfo {};

// tiling 分发入口
static ge::graphStatus NegTilingFunc(gert::TilingContext* context)
{
    NegTilingData* tiling = context->GetTilingData<NegTilingData>();
    int64_t ubPartNum = 4;
    uint64_t dataTypeLength = 4;
    auto dt = context->GetInputDesc(0)->GetDataType();
    bool isInt64 = (dt == ge::DT_INT64);
    if (dt == ge::DT_INT8) {
        dataTypeLength = 1;
        ubPartNum += 4;
    } else if (dt == ge::DT_UINT8) {
        dataTypeLength = 1;
        ubPartNum += 6;
    } else if (dt == ge::DT_INT64) {
        dataTypeLength = 8;
        ubPartNum = INT64_UB_PART_NUM;
    } else if (dt == ge::DT_BF16) {
        dataTypeLength = 2;
        ubPartNum += 2;
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_INT16) {
        dataTypeLength = 2;
    }

    uint64_t ubLength = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t bigCoreLoopNum = 0;
    uint64_t bigCoreTailDataNum = 0;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto coreNum = ascendcPlatform.GetCoreNum();

    // Based on the input length and the number of inputs, the number of bytes of the input data type is obtained
    uint64_t inputDataNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint64_t inputLength = inputDataNum * dataTypeLength;
    if (coreNum == 0 || BLOCK_SIZE == 0) {
        OP_LOGE(context, "coreNum or BLOCK_SIZE is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubPartLength = ubLength / ubPartNum;
    // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
    uint64_t ubPartBlockNum = ubPartLength / BLOCK_SIZE;
    uint64_t ubPartDataNum = (ubPartBlockNum * BLOCK_SIZE) / dataTypeLength;
    if (isInt64) {
        ubPartDataNum = (ubPartDataNum / INT64_COMPARE_ALIGN_NUM) * INT64_COMPARE_ALIGN_NUM;
        uint64_t maxInt64TileDataNum = INT64_COMPARE_ALIGN_NUM * INT64_MAX_REPEAT_TIMES;
        ubPartDataNum = (ubPartDataNum > maxInt64TileDataNum) ? maxInt64TileDataNum : ubPartDataNum;
    }
    if (inputDataNum != 0 && ubPartDataNum == 0) {
        OP_LOGE(context, "ubPartDataNum is 0");
        return ge::GRAPH_FAILED;
    }

    if (isInt64) {
        if (inputDataNum == 0) {
            tiling->smallCoreDataNum = 0;
            tiling->bigCoreDataNum = 0;
            tiling->ubPartDataNum = ubPartDataNum;
            tiling->smallCoreTailDataNum = 0;
            tiling->bigCoreTailDataNum = 0;
            tiling->smallCoreLoopNum = 0;
            tiling->bigCoreLoopNum = 0;
            tiling->tailBlockNum = 0;
            context->SetTilingKey(0);
            context->SetBlockDim(1);
            size_t* currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        coreNum = (coreNum < inputDataNum) ? coreNum : inputDataNum;
        uint64_t smallCoreDataNum = inputDataNum / coreNum;
        uint64_t tailBlockNum = inputDataNum % coreNum;
        bigCoreDataNum = smallCoreDataNum + 1;

        uint64_t smallCoreLoopNum = (smallCoreDataNum + ubPartDataNum - 1) / ubPartDataNum;
        bigCoreLoopNum = (bigCoreDataNum + ubPartDataNum - 1) / ubPartDataNum;
        uint64_t smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum - 1);
        bigCoreTailDataNum = bigCoreDataNum - ubPartDataNum * (bigCoreLoopNum - 1);

        context->SetTilingKey((tailBlockNum != 0) ? 1 : 0);

        tiling->smallCoreDataNum = smallCoreDataNum;
        tiling->bigCoreDataNum = bigCoreDataNum;
        tiling->ubPartDataNum = ubPartDataNum;
        tiling->smallCoreTailDataNum = smallCoreTailDataNum;
        tiling->bigCoreTailDataNum = bigCoreTailDataNum;
        tiling->smallCoreLoopNum = smallCoreLoopNum;
        tiling->bigCoreLoopNum = bigCoreLoopNum;
        tiling->tailBlockNum = tailBlockNum;

        context->SetBlockDim(coreNum);
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    // Input data for 32B alignment
    uint64_t inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    if (ubPartDataNum >= inputDataNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number
        // of audits is the actual number of audits
        coreNum = (coreNum < inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
    }

    uint64_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;

    // Small chunks are calculated and sliced several times using the number of data on each core
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
    uint64_t smallCoreLoopNum = smallCoreDataNum / ubPartDataNum;
    smallCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? smallCoreLoopNum : smallCoreLoopNum + 1;
    // Tail block calculation for small chunks of data
    uint64_t smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum - 1);
    smallCoreTailDataNum = smallCoreTailDataNum == 0 ? ubPartDataNum : smallCoreTailDataNum;

    if (0 != tailBlockNum) {
        everyCoreInputBlockNum += 1;
        bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
        bigCoreLoopNum = bigCoreDataNum / ubPartDataNum;
        bigCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? bigCoreLoopNum : bigCoreLoopNum + 1;
        bigCoreTailDataNum = bigCoreDataNum - ubPartDataNum * (bigCoreLoopNum - 1);
        bigCoreTailDataNum = bigCoreTailDataNum == 0 ? ubPartDataNum : bigCoreTailDataNum;
        context->SetTilingKey(1);
    } else {
        context->SetTilingKey(0);
    }

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->ubPartDataNum = ubPartDataNum;
    tiling->smallCoreTailDataNum = smallCoreTailDataNum;
    tiling->bigCoreTailDataNum = bigCoreTailDataNum;
    tiling->smallCoreLoopNum = smallCoreLoopNum;
    tiling->bigCoreLoopNum = bigCoreLoopNum;
    tiling->tailBlockNum = tailBlockNum;

    context->SetBlockDim(coreNum);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForNeg([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Neg).Tiling(NegTilingFunc).TilingParse<NegCompileInfo>(TilingParseForNeg);
} // namespace optiling
