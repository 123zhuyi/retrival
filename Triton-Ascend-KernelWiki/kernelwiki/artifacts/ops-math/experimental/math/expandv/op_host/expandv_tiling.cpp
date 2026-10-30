/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
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
 * \file expandv_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/expandv_tiling_data.h"
#include "../op_kernel/expandv_tiling_key.h"

namespace optiling {
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t WS_SYS_SIZE = 0U;
constexpr uint64_t UB_DATA_NUMBER_DEFAULT = 4;
constexpr uint64_t MAX_DIMS_DEFAULT = 10;
struct ExpandvCompileInfo {};
struct ExpandvCompileInfoShapeInfo {
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
    uint64_t blockSize{0};

    uint64_t in_rank{0};
    uint64_t out_rank{0};
    uint64_t inShapeArr[MAX_DIMS_DEFAULT]{0};
    uint64_t outShapeArr[MAX_DIMS_DEFAULT]{0};
    uint64_t inStrideArr[MAX_DIMS_DEFAULT]{0};
    uint64_t outStrideArr[MAX_DIMS_DEFAULT]{0};
};
// 平台信息
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
// 工作空间
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    // 系统workspace大小
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t usrSize = WS_SYS_SIZE; // 用户部分
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
// 形状属性
static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t ubSize, ExpandvCompileInfoShapeInfo& info)
{
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);
    info.inputNum = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = info.inputNum * typeLength;
    if (info.inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    info.inputBytes = inputLength / info.inputNum;

    uint64_t ubDataNumber = UB_DATA_NUMBER_DEFAULT;
    info.tileBlockNum = (ubSize / BUFFER_NUM / info.blockSize) / ubDataNumber;
    if (info.inputBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    info.tileDataNum = (info.tileBlockNum * info.blockSize) / info.inputBytes;
    info.inputLengthAlign32 = (((inputLength + info.blockSize - 1) / info.blockSize) * info.blockSize);

    const gert::Shape x1ShapeObj = context->GetInputShape(0)->GetStorageShape();
    uint64_t dimNum1 = static_cast<uint64_t>(x1ShapeObj.GetDimNum());
    info.in_rank = static_cast<uint64_t>(dimNum1);

    const gert::TypedContinuousVector<int64_t>* shape_list = nullptr;
    auto attrs = context->GetAttrs();
    if (attrs) {
        shape_list = attrs->GetListInt(0);
    }
    if (shape_list == nullptr) {
        OP_LOGE(context, "Failed to get shape attribute");
        return ge::GRAPH_FAILED;
    }
    const int64_t* shape = shape_list->GetData();
    uint64_t dims = static_cast<uint64_t>(shape_list->GetSize());
    info.out_rank = static_cast<uint64_t>(dims);

    for (uint64_t i = 0; i < dims; ++i) {
        info.outShapeArr[i] = static_cast<uint64_t>(shape[i]);
    }
    for (uint64_t i = 0; i < dimNum1; ++i) {
        info.inShapeArr[i] = static_cast<uint64_t>(x1ShapeObj.GetDim(i));
    }
    if (dimNum1 > 0) {
        info.inStrideArr[dimNum1 - 1] = 1;
        for (int i = static_cast<int64_t>(dimNum1) - 2; i >= 0; --i) {
            info.inStrideArr[i] = info.inStrideArr[i + 1] * static_cast<uint64_t>(info.inShapeArr[i + 1]);
        }
    }
    if (dims > 0) {
        info.outStrideArr[dims - 1] = 1;
        for (int i = static_cast<int64_t>(dims) - 2; i >= 0; --i) {
            info.outStrideArr[i] = info.outStrideArr[i + 1] * static_cast<uint64_t>(info.outShapeArr[i + 1]);
        }
    }

    return ge::GRAPH_SUCCESS;
}
// 分块信息
static ge::graphStatus CalculateCoreBlockNums(int64_t coreNum, ExpandvCompileInfoShapeInfo& info)
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
static ge::graphStatus ExpandvTilingFunc(gert::TilingContext* context)
{
    ExpandvTilingData* tiling = context->GetTilingData<ExpandvTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ExpandvTilingData), 0, sizeof(ExpandvTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 获取输入数据信息
    ExpandvCompileInfoShapeInfo shapeInfo;
    shapeInfo.blockSize = Ops::Base::GetUbBlockSize(context);
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
    tiling->smallCoreDataNum = static_cast<uint64_t>(shapeInfo.smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint64_t>(shapeInfo.bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint64_t>(shapeInfo.tileDataNum);
    tiling->smallTailDataNum = static_cast<uint64_t>(shapeInfo.smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint64_t>(shapeInfo.bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint64_t>(shapeInfo.finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint64_t>(shapeInfo.finalBigTileNum);
    tiling->tailBlockNum = static_cast<uint64_t>(shapeInfo.tailBlockNum);

    tiling->in_rank = static_cast<uint64_t>(shapeInfo.in_rank);
    tiling->out_rank = static_cast<uint64_t>(shapeInfo.out_rank);
    for (uint64_t i = 0; i < MAX_DIMS_DEFAULT; i++) {
        tiling->inShapeArr[i] = static_cast<uint64_t>(shapeInfo.inShapeArr[i]);
        tiling->outShapeArr[i] = static_cast<uint64_t>(shapeInfo.outShapeArr[i]);
        tiling->inStrideArr[i] = static_cast<uint64_t>(shapeInfo.inStrideArr[i]);
        tiling->outStrideArr[i] = static_cast<uint64_t>(shapeInfo.outStrideArr[i]);
    }

    // 计算workspace大小
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(coreNum);
    // 设置tilingKey.
    uint64_t tilingKey = 0;
    if (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_2);
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_3);
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_INT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_4);
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_UINT8) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_5);
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_BOOL) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_6);
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_INT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_7);
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_UINT16) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_8);
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_UINT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_9);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForExpandv([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Expandv).Tiling(ExpandvTilingFunc).TilingParse<ExpandvCompileInfo>(TilingParseForExpandv);
} // namespace optiling
