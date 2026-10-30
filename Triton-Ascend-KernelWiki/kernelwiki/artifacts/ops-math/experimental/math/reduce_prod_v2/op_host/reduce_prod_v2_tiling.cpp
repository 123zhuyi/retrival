/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Zhi<@hitLeechi>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file reduce_prod_v2_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <graph/utils/type_utils.h>
#include "../op_kernel/reduce_prod_v2_tiling_data.h"
#include "../op_kernel/reduce_prod_v2_tiling_key.h"
namespace optiling {

const uint32_t BUFFER_NUM = 2;
const uint64_t UB_DATA_NUMBER_32 = 12;
const uint64_t SLOT = 16;
struct ReduceProdV2CompileInfo {};
struct ReduceProdV2ShapeInfo {
    uint32_t blockSize{0};
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
    uint64_t dataTypeId{0};
    uint64_t rows{0};
    uint64_t cols{0};
    uint64_t axes{0};
    uint64_t keepdims{0};
};

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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, ReduceProdV2ShapeInfo& info)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t userSize = 0;
    if (info.axes == 0) {
        userSize = static_cast<size_t>(SLOT * info.cols);
    } else {
        userSize = static_cast<size_t>(SLOT * info.rows);
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = userSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, ReduceProdV2ShapeInfo& info)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    info.blockSize = Ops::Base::GetUbBlockSize(context);
    info.inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    info.dataTypeId = context->GetInputDesc(0)->GetDataType();
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = info.inputNum * typeLength;
    OP_CHECK_IF(info.inputNum == 0, OP_LOGE(context, "inputNum is zero"), return ge::GRAPH_FAILED);
    info.inputBytes = inputLength / info.inputNum;
    auto dataType = context->GetInputDesc(0)->GetDataType();
    uint64_t ubDataNumber = UB_DATA_NUMBER_32;
    if (dataType == ge::DT_FLOAT) {
        ubDataNumber = UB_DATA_NUMBER_32;
    }

    info.tileBlockNum = (ubSize / BUFFER_NUM / info.blockSize) / ubDataNumber;
    OP_CHECK_IF(info.inputBytes == 0, OP_LOGE(context, "inputBytes is zero"), return ge::GRAPH_FAILED);
    info.tileDataNum = (info.tileBlockNum * info.blockSize) / info.inputBytes;
    info.inputLengthAlign32 = (((inputLength + info.blockSize - 1) / info.blockSize) * info.blockSize);
    info.rows = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    info.cols = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    auto attrs = context->GetAttrs();
    info.axes = 0; // 默认值
    info.keepdims = 1;
    if (attrs) {
        if (attrs->GetInt(0)) {
            info.axes = *(attrs->GetInt(0));
        }
        if (attrs->GetInt(1)) {
            info.keepdims = *(attrs->GetInt(1));
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(int64_t coreNum, ReduceProdV2ShapeInfo& info)
{
    if (0 == info.blockSize || 0 == coreNum || 0 == info.tileBlockNum || 0 == info.inputBytes) {
        return ge::GRAPH_FAILED;
    }
    uint64_t everyCoreInputBlockNum = info.inputLengthAlign32 / info.blockSize / coreNum;
    info.tailBlockNum = (info.inputLengthAlign32 / info.blockSize) % coreNum;
    info.smallCoreDataNum = everyCoreInputBlockNum * info.blockSize / info.inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalSmallTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    info.smallTailDataNum = info.smallCoreDataNum - (info.tileDataNum * smallTileNum);
    info.smallTailDataNum = info.smallTailDataNum == 0 ? info.tileDataNum : info.smallTailDataNum;

    everyCoreInputBlockNum += 1;
    info.bigCoreDataNum = everyCoreInputBlockNum * info.blockSize / info.inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalBigTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    info.bigTailDataNum = info.bigCoreDataNum - info.tileDataNum * bigTileNum;
    info.bigTailDataNum = info.bigTailDataNum == 0 ? info.tileDataNum : info.bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}
// tiling 分发入口
static ge::graphStatus ReduceProdV2TilingFunc(gert::TilingContext* context)
{
    // ReduceProdV2TilingData tiling;
    ReduceProdV2TilingData* tiling = context->GetTilingData<ReduceProdV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ReduceProdV2TilingData), 0, sizeof(ReduceProdV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 获取输入数据信息
    ReduceProdV2ShapeInfo shapeInfo;
    ret = GetShapeAttrsInfo(context, ubSize, shapeInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    // 计算coreNum
    if (shapeInfo.tileDataNum >= shapeInfo.inputNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number
        // of audits is the actual number of audits
        coreNum = (static_cast<uint64_t>(coreNum) < shapeInfo.inputLengthAlign32 / shapeInfo.blockSize) ?
                      coreNum :
                      shapeInfo.inputLengthAlign32 / shapeInfo.blockSize;
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
    tiling->dataTypeId = (uint32_t)shapeInfo.dataTypeId;
    tiling->rows = (uint32_t)shapeInfo.rows;
    tiling->cols = (uint32_t)shapeInfo.cols;
    tiling->axes = (uint32_t)shapeInfo.axes;
    tiling->keepdims = (uint32_t)shapeInfo.keepdims;
    // 计算workspace大小
    OP_CHECK_IF(
        GetWorkspaceSize(context, shapeInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReduceProdV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(ReduceProdV2)
    .Tiling(ReduceProdV2TilingFunc)
    .TilingParse<ReduceProdV2CompileInfo>(TilingParseForReduceProdV2);
} // namespace optiling
