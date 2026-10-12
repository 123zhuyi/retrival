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
 * \file greater_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/greater_tiling_data.h"
#include "../op_kernel/greater_tiling_key.h"

namespace optiling {

static uint64_t BLOCK_SIZE = 32;

struct GreaterCompileInfo {};

// tiling 分发入口
static ge::graphStatus GreaterTilingFunc(gert::TilingContext* context)
{
    GreaterTilingData* tiling = context->GetTilingData<GreaterTilingData>();
    uint64_t ubLength = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t bigCoreLoopNum = 0;
    uint64_t bigCoreTailDataNum = 0;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto coreNum = ascendcPlatform.GetCoreNum();

    // Based on the input length and the number of inputs, the number of bytes of the input data type is obtained
    uint64_t inputDataNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), dataTypeLength);
    uint64_t inputLength = inputDataNum * dataTypeLength;

    // There are a total of 3 shared UB spaces in the input and output. If it's int8, there are 2 more TBUFs
    uint64_t ubPartNum = 3;
    if (context->GetInputDesc(0)->GetDataType() == ge::DT_BF16 ||
        context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16) {
        ubPartNum = 6;
        BLOCK_SIZE = 64;
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) {
        ubPartNum = 7;
        BLOCK_SIZE = 128;
    } else if (
        context->GetInputDesc(0)->GetDataType() == ge::DT_INT8 ||
        context->GetInputDesc(0)->GetDataType() == ge::DT_UINT8) {
        ubPartNum = 10;
        BLOCK_SIZE = 32;
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_INT32) {
        ubPartNum = 9;
        BLOCK_SIZE = 128;
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_INT64) {
        ubPartNum = 8;
        BLOCK_SIZE = 256;
    }
    if (coreNum == 0 || BLOCK_SIZE == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubPartLength = ubLength / ubPartNum;
    // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
    uint64_t ubPartBlockNum = ubPartLength / BLOCK_SIZE;
    uint64_t ubPartDataNum = (ubPartBlockNum * BLOCK_SIZE) / dataTypeLength;

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

static ge::graphStatus TilingParseForGreater([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Greater).Tiling(GreaterTilingFunc).TilingParse<GreaterCompileInfo>(TilingParseForGreater);
} // namespace optiling
