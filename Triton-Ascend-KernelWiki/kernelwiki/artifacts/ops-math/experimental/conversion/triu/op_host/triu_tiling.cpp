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
 * \file triu_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/triu_tiling_data.h"
#include "../op_kernel/triu_tiling_key.h"

namespace optiling {
namespace {
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
const uint32_t minNum = 1;

const uint32_t keyOne = 1;
const uint32_t keyTwo = 2;
const uint32_t keyThree = 3;
const uint32_t keyFour = 4;

const uint32_t bufferFour = 4;
const uint32_t BlockSize = 32;
const uint32_t computeBatchSize = 256;
const uint32_t sizeHalf = 2;
const uint32_t VAL_ZRRO = 0;

uint32_t type_Size = VAL_ZRRO;
uint64_t key_value = keyOne;
// buffer for queue
uint64_t ub_Sharing_Num = 2;
int64_t row_Length = VAL_ZRRO;
int64_t column_Length = VAL_ZRRO;
int64_t matrix_Num = 1, matrix_Size = 1;
int64_t diag_Val = VAL_ZRRO;

uint32_t align_Num = VAL_ZRRO;
uint32_t total_Length_Aligned = VAL_ZRRO;
uint64_t loop_Cnt = VAL_ZRRO, full_Tile_Length = VAL_ZRRO, last_Tile_Length = VAL_ZRRO;
int32_t full_Cnt = VAL_ZRRO, last_Cnt = VAL_ZRRO;

struct TriuCompileInfo {};
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
    const std::set<ge::DataType> supportedDtype = {ge::DT_INT32, ge::DT_INT16, ge::DT_FLOAT, ge::DT_FLOAT16};
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
static ge::graphStatus TriuTilingFunc(gert::TilingContext* context)
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
        TriuTilingData* tiling = context->GetTilingData<TriuTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(TriuTilingData), 0, sizeof(TriuTilingData));
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }
    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4、设置tiling信息
    TriuTilingData* tiling = context->GetTilingData<TriuTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(TriuTilingData), 0, sizeof(TriuTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), type_Size);
    if (type_Size == 0) {
        OP_LOGE(context, "type_Size is 0");
        return ge::GRAPH_FAILED;
    }

    auto BLOCK_DIM = 1;
    context->SetBlockDim(BLOCK_DIM);
    const auto inputShape = context->GetInputTensor(0)->GetOriginShape();
    // class Shape: size_t dim_num_; int64_t dims_[];
    int64_t dimSize = inputShape.GetDimNum(), i = 0;
    // The number 2 is to preserve the last two dimensions
    for (i = 0; i < dimSize - 2; i++) {
        matrix_Num *= inputShape.GetDim(i);
    }
    row_Length = inputShape.GetDim(i);
    i++;
    column_Length = inputShape.GetDim(i);
    matrix_Size = row_Length * column_Length;
    const auto runtime_attrs = context->GetAttrs();
    const int64_t* diagPtr = runtime_attrs->GetInt(0);
    diag_Val = *diagPtr;
    if (diag_Val < column_Length - 1 && diag_Val > -row_Length) {
        // Regular
        key_value = keyOne;
    } else if (diag_Val <= -row_Length) {
        // The result is itself, TQueBind is enough
        key_value = keyTwo;
    } else {
        // All zero, just copyIn, Sub and copyOut
        key_value = keyThree;
    }
    align_Num = BlockSize / type_Size;
    total_Length_Aligned = (matrix_Num * matrix_Size + align_Num - 1) / align_Num * align_Num;
    loop_Cnt = VAL_ZRRO;
    full_Tile_Length = VAL_ZRRO;
    last_Tile_Length = VAL_ZRRO;
    full_Cnt = VAL_ZRRO;
    last_Cnt = VAL_ZRRO;
    uint64_t ub_length = ((ubSize / type_Size / ub_Sharing_Num) / align_Num * align_Num) - align_Num;
    if (key_value == keyOne && diag_Val <= 0 && column_Length % (computeBatchSize / type_Size) == 0) {
        // A faster method for aligned processing only
        key_value = keyFour;
        // Double buffer setting
        ub_Sharing_Num = bufferFour;
        // The result would not be the expected
        if (column_Length == 0) {
            column_Length = minNum;
        }
        ub_length = ((ubSize) / type_Size / ub_Sharing_Num) / column_Length * column_Length;
        loop_Cnt = (matrix_Size + ub_length - 1) / ub_length;
        if (loop_Cnt == 1) {
            full_Cnt = 0;
            last_Cnt = row_Length;
        } else {
            // The result would not be the expected
            if (column_Length == 0) {
                column_Length = minNum;
            }
            full_Cnt = ub_length / column_Length;
            last_Cnt = row_Length - full_Cnt * (loop_Cnt - 1);
        }
        // Already aligned
        full_Tile_Length = full_Cnt * column_Length;
        last_Tile_Length = last_Cnt * column_Length;
    } else if (key_value == keyThree) {
        loop_Cnt = (total_Length_Aligned + ub_length - 1) / ub_length;
        ub_Sharing_Num = bufferFour;
        ub_length = ((ubSize / type_Size / ub_Sharing_Num) / align_Num * align_Num) - align_Num;
        full_Tile_Length = ub_length;
        last_Tile_Length =
            (total_Length_Aligned - full_Tile_Length * (loop_Cnt - 1) + align_Num - 1) / align_Num * align_Num;
        if (loop_Cnt == 1) {
            full_Tile_Length = 0;
        }
    } else {
        loop_Cnt = (total_Length_Aligned + ub_length - 1) / ub_length;
        full_Tile_Length = ub_length;
        last_Tile_Length =
            (total_Length_Aligned - full_Tile_Length * (loop_Cnt - 1) + align_Num - 1) / align_Num * align_Num;
        if (loop_Cnt == 1) {
            full_Tile_Length = 0;
        }
    }
    tiling->totalLengthAligned = total_Length_Aligned;
    tiling->matrixNum = matrix_Num;
    tiling->matrixSize = matrix_Size;
    tiling->rowLength = row_Length;
    tiling->columnLength = column_Length;
    tiling->diagVal = diag_Val;

    tiling->loopCnt = loop_Cnt;
    tiling->fullTileLength = full_Tile_Length;
    tiling->lastTileLength = last_Tile_Length;
    tiling->fullCnt = full_Cnt;
    tiling->lastCnt = last_Cnt;

    tiling->alignNum = align_Num;
    tiling->typeSize = type_Size;

    context->SetTilingKey(key_value);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTriu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Triu).Tiling(TriuTilingFunc).TilingParse<TriuCompileInfo>(TilingParseForTriu);
} // namespace optiling
