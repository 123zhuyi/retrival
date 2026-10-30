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
 * \file pow_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "../op_kernel/pow_tiling_data.h"
#include "../op_kernel/pow_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 256;
const uint32_t BUFFER_NUM = 2;
const uint32_t WS_SYS_SIZE = 0;
const uint64_t UB_DATA_NUMBER_32 = 6;
const uint64_t UB_DATA_NUMBER_16 = 12;
const uint64_t UB_DATA_NUMBER_8 = 25;
struct PowCompileInfo {};
struct PowShapeInfo {
    uint64_t inputNum{0};
    uint64_t inputBytes{0};
    uint64_t tileBlockNum{0};
    uint64_t tileDataNum{0};
    uint64_t inputLengthAlign32{0};
    uint64_t smallCoreDataNum{0};
    uint64_t bigCoreDataNum{0};
    uint64_t smallTailDataNum{0};
    uint64_t bigTailDataNum{0};
    uint64_t finalSmallTileNum{0};
    uint64_t finalBigTileNum{0};
    uint64_t tailBlockNum{0};
};

static ge::graphStatus TilingParseForPow([[maybe_unused]] gert::TilingParseContext* context)
{
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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, PowShapeInfo& info)
{
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);
    info.inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    if (info.inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = info.inputNum * typeLength;
    info.inputBytes = inputLength / info.inputNum;
    if (info.inputBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    auto dataType = context->GetInputDesc(0)->GetDataType();
    uint64_t ubDataNumber = UB_DATA_NUMBER_32;
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            ubDataNumber = UB_DATA_NUMBER_32;
            break;
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_BF16:
            ubDataNumber = UB_DATA_NUMBER_16;
            break;
        case ge::DT_UINT8:
        case ge::DT_INT8:
            ubDataNumber = UB_DATA_NUMBER_8;
            break;
        default:
            break;
    }
    info.tileBlockNum = (ubSize / BUFFER_NUM / BLOCK_SIZE) / ubDataNumber;
    info.tileDataNum = (info.tileBlockNum * BLOCK_SIZE) / info.inputBytes;
    info.inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(int64_t coreNum, PowShapeInfo& info)
{
    if (0 == BLOCK_SIZE || 0 == coreNum || 0 == info.tileBlockNum || 0 == info.inputBytes) {
        return ge::GRAPH_FAILED;
    }
    uint64_t everyCoreInputBlockNum = info.inputLengthAlign32 / BLOCK_SIZE / coreNum;
    info.tailBlockNum = (info.inputLengthAlign32 / BLOCK_SIZE) % coreNum;
    info.smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / info.inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalSmallTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    info.smallTailDataNum = info.smallCoreDataNum - (info.tileDataNum * smallTileNum);
    info.smallTailDataNum = info.smallTailDataNum == 0 ? info.tileDataNum : info.smallTailDataNum;

    everyCoreInputBlockNum += 1;
    info.bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / info.inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalBigTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    info.bigTailDataNum = info.bigCoreDataNum - info.tileDataNum * bigTileNum;
    info.bigTailDataNum = info.bigTailDataNum == 0 ? info.tileDataNum : info.bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PowTilingFunc(gert::TilingContext* context)
{
    // PowTilingData tiling;
    PowTilingData* tiling = context->GetTilingData<PowTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(PowTilingData), 0, sizeof(PowTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 获取输入数据信息
    PowShapeInfo shapeInfo;
    ret = GetShapeAttrsInfo(context, ubSize, shapeInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 计算coreNum
    if (shapeInfo.tileDataNum >= shapeInfo.inputNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number
        // of audits is the actual number of audits
        coreNum = (static_cast<uint64_t>(coreNum) < shapeInfo.inputLengthAlign32 / BLOCK_SIZE) ?
                      coreNum :
                      shapeInfo.inputLengthAlign32 / BLOCK_SIZE;
    }
    // 计算每个core处理的数据块数
    ret = CalculateCoreBlockNums(coreNum, shapeInfo);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // 设置tiling数据
    tiling->smallCoreDataNum = (uint32_t)shapeInfo.smallCoreDataNum;
    tiling->bigCoreDataNum = (uint32_t)shapeInfo.bigCoreDataNum;
    tiling->tileDataNum = (uint32_t)shapeInfo.tileDataNum;
    tiling->smallTailDataNum = (uint32_t)shapeInfo.smallTailDataNum;
    tiling->bigTailDataNum = (uint32_t)shapeInfo.bigTailDataNum;
    tiling->finalSmallTileNum = (uint32_t)shapeInfo.finalSmallTileNum;
    tiling->finalBigTileNum = (uint32_t)shapeInfo.finalBigTileNum;
    tiling->tailBlockNum = (uint32_t)shapeInfo.tailBlockNum;
    // 计算workspace大小
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(coreNum);
    // 设置tilingKey.
    uint32_t tilingKey = 0;
    if (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Pow).Tiling(PowTilingFunc).TilingParse<PowCompileInfo>(TilingParseForPow);
} // namespace optiling