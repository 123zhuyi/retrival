/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liu Jun <@kbryantttt>
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
 * \file strided_slice_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/strided_slice_tiling_data.h"
#include "../op_kernel/strided_slice_tiling_key.h"
namespace optiling {
namespace {
const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
uint32_t type_Size = 0;
struct StridedSliceCompileInfo {};
} // namespace
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
    const std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT,  ge::DT_INT32,   ge::DT_UINT32, ge::DT_INT64, // int32对应12种数据类型
        ge::DT_UINT64, ge::DT_FLOAT16, ge::DT_INT16,  ge::DT_UINT16,
        ge::DT_BF16,   ge::DT_INT8,    ge::DT_UINT8,  ge::DT_BOOL};
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
// tiling 分发入口
static ge::graphStatus StridedSliceTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    // 2、获取shape、属性信息
    int64_t totalIdx = 0;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    // handle empty input
    if (totalIdx <= 0) {
        StridedSliceTilingData* tiling = context->GetTilingData<StridedSliceTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(StridedSliceTilingData), 0, sizeof(StridedSliceTilingData));
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }
    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    // 4、设置tiling信息
    StridedSliceTilingData* tiling = context->GetTilingData<StridedSliceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(StridedSliceTilingData), 0, sizeof(StridedSliceTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), type_Size);
    if (type_Size == 0) {
        OP_LOGE(context, "type_Size is 0");
        return ge::GRAPH_FAILED;
    }
    uint32_t start1 = 0;
    uint32_t start2 = 0;
    uint32_t end1 = 0;
    uint32_t end2 = 0;
    uint32_t stride1 = 0;
    uint32_t stride2 = 0;
    const auto xShape = context->GetInputTensor(0)->GetOriginShape();
    int64_t dim = static_cast<int64_t>(xShape.GetDimNum());
    auto attrs = context->GetAttrs();
    if (attrs) {
        const int64_t* start1Ptr = attrs->GetInt(0);
        if (start1Ptr) {
            start1 = static_cast<uint32_t>(*start1Ptr);
        }
        if (dim == 1) {
            start1 = static_cast<uint32_t>(0);
        }
        const int64_t* start2Ptr = attrs->GetInt(1);
        if (start2Ptr) {
            start2 = static_cast<uint32_t>(*start2Ptr);
        }
        const int64_t* end1Ptr = attrs->GetInt(2);
        if (end1Ptr) {
            end1 = static_cast<uint32_t>(*end1Ptr);
        }
        if (dim == 1) {
            end1 = static_cast<uint32_t>(1);
        }
        const int64_t* end2Ptr = attrs->GetInt(3);
        if (end2Ptr) {
            end2 = static_cast<uint32_t>(*end2Ptr);
        }
        const int64_t* stride1Ptr = attrs->GetInt(4);
        if (stride1Ptr) {
            stride1 = static_cast<uint32_t>(*stride1Ptr);
        }
        if (dim == 1) {
            stride1 = static_cast<uint32_t>(1);
        }
        const int64_t* stride2Ptr = attrs->GetInt(5);
        if (stride2Ptr) {
            stride2 = static_cast<uint32_t>(*stride2Ptr);
        }
    }
    uint32_t ubDataNumber = 9;
    auto inputx = context->GetInputShape(0);
    auto inputShapeX = inputx->GetStorageShape();
    uint32_t rows;
    if (dim == 1) {
        rows = 1;
    } else {
        rows = static_cast<uint32_t>(inputShapeX.GetDim(0));
    }
    uint32_t cols = static_cast<uint32_t>(inputShapeX.GetDim(1));
    uint32_t rowBytes = cols * type_Size;
    uint32_t rowBytesAligned = ((rowBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t blocksPerRow = rowBytesAligned / BLOCK_SIZE;
    coreNum = (coreNum > rows) ? rows : coreNum;
    coreNum = (coreNum >= 1) ? coreNum : 1;
    uint32_t baseRowsPerCore = rows / coreNum;
    uint32_t tailRows = rows % coreNum;
    uint32_t bigTotalRows = baseRowsPerCore + 1;
    bigTotalRows = std::min(bigTotalRows, rows);
    uint32_t totalUbBlocks = (ubSize / BLOCK_SIZE) / BUFFER_NUM / ubDataNumber;
    uint32_t tileRows = (blocksPerRow == 0) ? 1 : (totalUbBlocks / blocksPerRow);
    tileRows = (tileRows == 0) ? 1 : tileRows;
    uint32_t smallTileNum = baseRowsPerCore / tileRows;
    uint32_t finalSmallTileNum = (baseRowsPerCore % tileRows == 0) ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailRows = (baseRowsPerCore % tileRows == 0) ? tileRows : (baseRowsPerCore % tileRows);
    uint32_t bigTileNum = bigTotalRows / tileRows;
    uint32_t finalBigTileNum = (bigTotalRows % tileRows == 0) ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailRows = (bigTotalRows % tileRows == 0) ? tileRows : (bigTotalRows % tileRows);
    tiling->cols = cols;
    tiling->rows = rows;
    tiling->start1 = start1;
    tiling->start2 = start2;
    tiling->end1 = end1;
    tiling->end2 = end2;
    tiling->stride1 = stride1;
    tiling->stride2 = stride2;
    tiling->tileRows = tileRows;
    tiling->smallTailRows = smallTailRows;
    tiling->bigTailRows = bigTailRows;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->baseRowsPerCore = baseRowsPerCore;
    tiling->bigTotalRows = bigTotalRows;
    tiling->tailRows = tailRows;
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForStridedSlice([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(StridedSlice)
    .Tiling(StridedSliceTilingFunc)
    .TilingParse<StridedSliceCompileInfo>(TilingParseForStridedSlice);
} // namespace optiling
