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
 * \file equal_tiling.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "util/math_util.h"
#include "../op_kernel/equal_tiling_data.h"
#include "../op_kernel/equal_tiling_key.h"

namespace optiling {

constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t ALIGN_BYTES = 256;
constexpr uint64_t INT32_FP_UB_COMPUTE_RESERVE = 256 * 7; // 256B alignment × 7 slots reserved for int32/fp compute
constexpr uint64_t INT32_FP_UB_SYS_RESERVE = 8 * 1024;    // 8KB system reserve for int32/fp path
constexpr uint64_t UB_DATA_NUMBER_DEFAULT = 6;
constexpr uint64_t UB_DATA_NUMBER_INT8 = 12;
constexpr uint64_t UB_DATA_NUMBER_INT32_FP = 5;

// tiling 分发入口
static ge::graphStatus EqualTilingFunc(gert::TilingContext* context)
{
    EqualTilingData* tiling = context->GetTilingData<EqualTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    uint64_t ubSize;
    uint32_t bigprocessDataNum_computes = 0;
    uint32_t tailbigprocessDataNum_computes = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = ascendcPlatform.GetCoreNum();
    }
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion != platform_ascendc::SocVersion::ASCEND910B &&
        socVersion != platform_ascendc::SocVersion::ASCEND310B &&
        context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) {
        OP_LOGE(context, "bf16 is only supported on ASCEND910B and ASCEND310B");
        return ge::GRAPH_FAILED;
    }

    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    if (inputNum == 0) {
        OP_LOGE(context, "inputNum is 0, invalid number");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubDataNumber = UB_DATA_NUMBER_DEFAULT;
    if (context->GetInputDesc(0)->GetDataType() == ge::DT_INT8 ||
        context->GetInputDesc(0)->GetDataType() == ge::DT_UINT8) {
        ubDataNumber = UB_DATA_NUMBER_INT8;
    }

    if (context->GetInputDesc(0)->GetDataType() == ge::DT_INT32 ||
        context->GetInputDesc(0)->GetDataType() == ge::DT_UINT32 ||
        context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) {
        ubDataNumber = UB_DATA_NUMBER_INT32_FP;
        ubSize = ubSize - INT32_FP_UB_COMPUTE_RESERVE - INT32_FP_UB_SYS_RESERVE;
    }

    ubSize = ubSize / typeLength;
    uint64_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE);

    uint64_t inputLengthAlgin32 = (((inputNum + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number
        // of audits is the actual number of audits
        coreNum = (coreNum < inputLengthAlgin32 / BLOCK_SIZE) ? coreNum : inputLengthAlgin32 / BLOCK_SIZE;
    }

    if (coreNum == 0) {
        OP_LOGE(context, "coreNum is 0, invalid number");
        return ge::GRAPH_FAILED;
    }
    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;

    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
    uint32_t smallprocessDataNum_computes =
        (((tileDataNum * typeLength + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES) / typeLength;
    uint32_t tailsmallprocessDataNum_computes =
        (((smallTailDataNum * typeLength + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES) / typeLength;

    everyCoreInputBlockNum += 1;
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
    bigprocessDataNum_computes =
        (((tileDataNum * typeLength + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES) / typeLength;
    tailbigprocessDataNum_computes =
        (((bigTailDataNum * typeLength + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES) / typeLength;

    tiling->smallCoreDataNum = static_cast<uint32_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint32_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint32_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<uint32_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint32_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint32_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint32_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<uint32_t>(tailBlockNum);
    tiling->bigprocessDataNum_computes = bigprocessDataNum_computes;
    tiling->smallprocessDataNum_computes = smallprocessDataNum_computes;
    tiling->tailbigprocessDataNum_computes = tailbigprocessDataNum_computes;
    tiling->tailsmallprocessDataNum_computes = tailsmallprocessDataNum_computes;

    context->SetBlockDim(coreNum);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Equal).Tiling(EqualTilingFunc);
} // namespace optiling
