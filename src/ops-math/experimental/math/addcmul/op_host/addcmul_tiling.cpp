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
 * \file addcmul_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/addcmul_tiling_data.h"
#include "../op_kernel/addcmul_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;

struct AddcmulCompileInfo {};

// tiling 分发入口
static ge::graphStatus AddcmulTilingFunc(gert::TilingContext* context)
{
    AddcmulTilingData* tiling = context->GetTilingData<AddcmulTilingData>();
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    // Based on the input length and the number of inputs, the number of bytes of the input data type is obtained
    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(inputNum == 0, OP_LOGE(context, "inputNum is 0"), return ge::GRAPH_FAILED);
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    uint64_t inputBytes = inputLength / inputNum;
    uint64_t valueSize = context->GetInputShape(3)->GetStorageShape().GetShapeSize();
    if (valueSize == 1) {
        context->SetTilingKey(0);
    } else {
        context->SetTilingKey(1);
    }
    uint64_t ubDataNumber = (context->GetInputDesc(0)->GetDataType() != ge::DT_BF16) ? 4 : 6;
    // The number of 32B data blocks that can be used for each data. DOUBLE BUFFER is already counted here
    uint64_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    // Input data for 32B alignment
    uint64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number
        // of audits is the actual number of audits
        coreNum = (coreNum < inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    }
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputBytes == 0, OP_LOGE(context, "inputBytes is 0"), return ge::GRAPH_FAILED);
    // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of
    // audits is the actual number of audits
    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;

    // Small chunks are calculated and sliced several times using the number of data on each core
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    // Tail block calculation for small chunks of data
    uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // The total length of a large block of data is 32B larger than that of a small block of data
    everyCoreInputBlockNum += 1;
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    // 一个小核数据个数
    tiling->smallCoreDataNum = (uint32_t)smallCoreDataNum;
    // 一个大核数据个数
    tiling->bigCoreDataNum = (uint32_t)bigCoreDataNum;
    // 一次搬运的数据个数
    tiling->tileDataNum = (uint32_t)tileDataNum;
    // 小核尾块数据个数
    tiling->smallTailDataNum = (uint32_t)smallTailDataNum;
    // 大核尾块数据个数
    tiling->bigTailDataNum = (uint32_t)bigTailDataNum;
    // 小核搬运次数
    tiling->finalSmallTileNum = (uint32_t)finalSmallTileNum;
    // 大核搬运次数
    tiling->finalBigTileNum = (uint32_t)finalBigTileNum;
    // 大核数
    tiling->tailBlockNum = (uint32_t)tailBlockNum;

    context->SetBlockDim(coreNum);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAddcmul([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Addcmul).Tiling(AddcmulTilingFunc).TilingParse<AddcmulCompileInfo>(TilingParseForAddcmul);
} // namespace optiling
