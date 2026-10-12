/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua<@LePenseur>
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
 * \file sort_v2_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sort_v2_tiling_data.h"
#include "../op_kernel/sort_v2_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 32;
const uint32_t ALIGN = 8;
constexpr uint32_t WS_SYS_SIZE = 512U;

struct SortV2CompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    // 获取输入shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    totalIdx = inputX->GetStorageShape().GetShapeSize();
    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreNum(gert::TilingContext* context, SortV2TilingData* tiling, int64_t coreNum)
{
    // 获取输入Shape
    const auto* storageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Shape& shape = storageShape->GetStorageShape();
    const uint32_t rank = shape.GetDimNum();
    OP_CHECK_IF(rank > 2, OP_LOGE(context, "rank must be <= 2"), return ge::GRAPH_FAILED);
    const uint32_t dimH = rank > 0 ? shape.GetDim(0) : 1;
    const uint32_t dimW = rank > 1 ? shape.GetDim(1) : 1;
    const uint32_t dims[2] = {dimH, dimW};
    // 读取属性axis和descending
    int32_t axis = 0;
    bool descending = false;
    auto attrs = context->GetAttrs();
    if (attrs) {
        const int64_t* axisPtr = attrs->GetInt(0);
        if (axisPtr) {
            axis = static_cast<int32_t>(*axisPtr);
        }
        const bool* descPtr = attrs->GetBool(1);
        if (descPtr) {
            descending = *descPtr;
        }
    }
    // 检查指定维度是否在规定范围
    if (axis < -1 * static_cast<int32_t>(rank) || axis >= static_cast<int32_t>(rank)) {
        OP_LOGE(context, "axis value is out of range");
        return ge::GRAPH_FAILED;
    }
    if (axis < 0) {
        axis += static_cast<int32_t>(rank);
    }
    // 定义大小核分配参数以及填充参数
    const uint32_t sliceLen = dims[axis];
    if (coreNum <= 0) {
        OP_LOGE(context, "coreNum must be positive");
        return ge::GRAPH_FAILED;
    }
    const uint32_t workCoreNum = std::min(static_cast<uint32_t>(coreNum), std::max(1U, (axis == 0) ? dimW : dimH));
    const uint32_t smallCoreNum = workCoreNum - (sliceLen % workCoreNum == 0 ? 0 : sliceLen % workCoreNum);
    const uint32_t bigCoreNum = workCoreNum - smallCoreNum;
    const uint32_t smallCoreDataNum = sliceLen / workCoreNum;
    const uint32_t bigCoreDataNum = smallCoreDataNum + (sliceLen % workCoreNum == 0 ? 0 : 1);
    const uint32_t realSortLen = ((sliceLen + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    const uint32_t align8 = ((sliceLen + ALIGN - 1) / ALIGN) * ALIGN;
    const uint32_t padLen = align8 - sliceLen;
    const uint32_t dupCount = BLOCK_SIZE - align8 % BLOCK_SIZE;

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->smallCoreNum = smallCoreNum;
    tiling->bigCoreNum = bigCoreNum;
    tiling->axis = axis;
    tiling->descending = descending;
    tiling->dimH = dimH;
    tiling->dimW = dimW;
    tiling->sliceLen = sliceLen;
    tiling->realSortLen = realSortLen;
    tiling->align8 = align8;
    tiling->padLen = padLen;
    tiling->dupCount = dupCount;
    tiling->startBlockIdx = workCoreNum;
    context->SetBlockDim(workCoreNum);
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus SortV2TilingFunc(gert::TilingContext* context)
{
    // 1. platform
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. shapes & dtype
    int64_t totalIdx = 0;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // handle empty input
    if (totalIdx <= 0) {
        SortV2TilingData* tiling = context->GetTilingData<SortV2TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(SortV2TilingData), 0, sizeof(SortV2TilingData));
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }

    // --- safer numeric types ---
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    if (typeLength == 0) {
        OP_LOGE(context, "typeLength is 0");
        return ge::GRAPH_FAILED;
    }

    // 3. workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    SortV2TilingData* tiling = context->GetTilingData<SortV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SortV2TilingData), 0, sizeof(SortV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 4. CoreNum & Tiling Calculation
    OP_CHECK_IF(
        CalculateCoreNum(context, tiling, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreNum error"),
        return ge::GRAPH_FAILED);
    context->GetRawTilingData()->SetDataSize(sizeof(SortV2TilingData));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSortV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
// tiling注册入口.
IMPL_OP_OPTILING(SortV2).Tiling(SortV2TilingFunc).TilingParse<SortV2CompileInfo>(TilingParseForSortV2);
} // namespace optiling
