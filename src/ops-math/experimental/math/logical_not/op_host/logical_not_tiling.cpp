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
 * \file logical_not_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/logical_not_tiling_data.h"
#include "../op_kernel/logical_not_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;

struct LogicalNotCompileInfo {};

static ge::graphStatus LogicalNotTilingFunc(gert::TilingContext* context)
{
    LogicalNotTilingData* tiling = context->GetTilingData<LogicalNotTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(LogicalNotTilingData), 0, sizeof(LogicalNotTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    uint64_t ubLength = 0;
    uint32_t bigCoreDataNum = 0;
    uint32_t bigCoreLoopNum = 0;
    uint32_t bigCoreTailDataNum = 0;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    auto coreNum = ascendcPlatform.GetCoreNum();

    // Based on the input length and the number of inputs, the number of bytes of the input data type is obtained
    uint32_t inputDataNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), dataTypeLength);
    uint32_t inputLength = inputDataNum * dataTypeLength;
    if (coreNum == 0 || BLOCK_SIZE == 0) {
        OP_LOGE(context, "coreNum or BLOCK_SIZE is 0");
        return ge::GRAPH_FAILED;
    }

    // There are a total of 3 shared UB spaces in the input and output. If it's int8, there are 2 more TBUFs
    uint32_t ubPartNum = 3;
    uint32_t ubPartLength = ubLength / ubPartNum / BUFFER_NUM;
    // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
    uint32_t ubPartBlockNum = ubPartLength / BLOCK_SIZE;
    uint32_t ubPartDataNum = (ubPartBlockNum * BLOCK_SIZE) / dataTypeLength;

    // Input data for 32B alignment
    uint32_t inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    if (ubPartDataNum >= inputDataNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number
        // of audits is the actual number of audits
        coreNum = (coreNum < inputLengthAlign32 / BLOCK_SIZE) ? coreNum : inputLengthAlign32 / BLOCK_SIZE;
    }

    uint32_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;

    // Small chunks are calculated and sliced several times using the number of data on each core
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dataTypeLength;
    uint32_t smallCoreLoopNum = smallCoreDataNum / ubPartDataNum;
    smallCoreLoopNum = (everyCoreInputBlockNum % ubPartBlockNum) == 0 ? smallCoreLoopNum : smallCoreLoopNum + 1;
    // Tail block calculation for small chunks of data
    uint32_t smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum - 1);
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

static ge::graphStatus TilingParseForLogicalNot([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LogicalNot).Tiling(LogicalNotTilingFunc).TilingParse<LogicalNotCompileInfo>(TilingParseForLogicalNot);
} // namespace optiling
