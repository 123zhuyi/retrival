/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
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
 * \file masked_fill_tiling.cpp
 * \brief MaskedFill operator tiling implementation
 */
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/masked_fill_tiling_data.h"
#include "../op_kernel/masked_fill_tiling_key.h"

namespace optiling {

static const size_t INPUT_IDX_X = 0;     // Input x index
static const size_t INPUT_IDX_MASK = 1;  // Input mask index
static const size_t INPUT_IDX_VALUE = 2; // Input value index
static const uint32_t BLOCK_SIZE = 32;
static const uint32_t BUFFER_NUM = 2;
static const int32_t MY_SHAPE_SIZE = 128;
static const int32_t MY_SIZE = 0; // Workspace size

struct MaskedFillCompileInfo {};

// Get platform information (ubSize, coreNum)
static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum, int32_t& wsSysSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    wsSysSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

// Get shape and data type information
static ge::graphStatus GetShapeAndTypeInfo(
    gert::TilingContext* context, uint32_t& data_x, uint32_t& data_mask, uint32_t sShape[], uint32_t maskShape[],
    uint32_t& xNumShapes, uint32_t& maskNumShapes, ge::DataType& x_type)
{
    // Get x shape and type
    auto x_shape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto x_desc = context->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_desc);
    x_type = x_desc->GetDataType();
    xNumShapes = x_shape->GetStorageShape().GetDimNum();
    data_x = 1;
    for (uint32_t i = 0; i < xNumShapes; i++) {
        sShape[i] = x_shape->GetStorageShape().GetDim(i);
        data_x *= sShape[i];
    }

    // Get mask shape
    auto mask_shape = context->GetInputShape(INPUT_IDX_MASK);
    OP_CHECK_NULL_WITH_CONTEXT(context, mask_shape);
    maskNumShapes = mask_shape->GetStorageShape().GetDimNum();
    data_mask = 1;
    for (uint32_t i = 0; i < maskNumShapes; i++) {
        maskShape[i] = mask_shape->GetStorageShape().GetDim(i);
        data_mask *= maskShape[i];
    }

    // Check shape compatibility between x and mask
    if (maskNumShapes > xNumShapes) {
        OP_LOGE(context, "mask shape dimension exceeds x shape dimension");
        return ge::GRAPH_FAILED;
    } else {
        int align_start = xNumShapes - maskNumShapes;
        for (uint32_t i = 0; i < maskNumShapes; ++i) {
            uint32_t x_dim = sShape[align_start + i];
            uint32_t mask_dim = maskShape[i];
            if (x_dim != mask_dim && x_dim != 1 && mask_dim != 1) {
                OP_LOGE(context, "Incompatible dimensions between x and mask");
                return ge::GRAPH_FAILED;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

// Calculate tiling parameters
static void TilingParamsCalc(
    uint32_t length, uint32_t alignNum, uint32_t tileBlockNum, uint32_t& tileNum, uint32_t& tileLength,
    uint32_t& lastTileLength)
{
    if (length == 0 || alignNum == 0 || tileBlockNum == 0) {
        tileNum = 0;
        tileLength = 0;
        lastTileLength = 0;
        return;
    }
    uint32_t max_ub_num = tileBlockNum / BUFFER_NUM * BUFFER_NUM;
    tileNum = length / (alignNum * max_ub_num);
    if ((static_cast<uint32_t>(length / alignNum) % max_ub_num == 0U) || tileNum == 0U) {
        if (tileNum == 0U) {
            tileNum = 1U;
        }
        if (length < max_ub_num * alignNum) {
            tileLength = ((static_cast<size_t>(length) / alignNum) + 1) / BUFFER_NUM * BUFFER_NUM * alignNum;
            lastTileLength = tileLength;
        } else {
            tileLength = max_ub_num * alignNum;
            lastTileLength = static_cast<uint32_t>(length - (tileNum - 1) * tileLength);
        }
    } else {
        tileNum++;
        tileLength = max_ub_num * alignNum;
        lastTileLength = static_cast<uint32_t>(length - (tileNum - 1) * tileLength);
    }
}

// Calculate tiling data
static ge::graphStatus CalculateTilingData(
    MaskedFillTilingData* tiling, uint64_t ubSize, int64_t coreNum, uint32_t data_x, uint32_t data_mask,
    uint32_t xNumShapes, uint32_t maskNumShapes, uint32_t sShape[], uint32_t maskShape[], ge::DataType x_type)
{
    // Set basic shape info
    tiling->xSize = data_x;
    tiling->maskSize = data_mask;
    tiling->xNumShapes = xNumShapes;
    tiling->maskNumShapes = maskNumShapes;
    // 依次赋值
    for (int i = 0; i < MY_SHAPE_SIZE; i++) {
        tiling->sShape[i] = sShape[i];
        tiling->maskShape[i] = maskShape[i];
    }

    if (coreNum == 0 || ubSize == 0)
        return ge::GRAPH_FAILED;
    // Get input data type size
    uint32_t inputBytes = 0;
    ge::TypeUtils::GetDataTypeLength(x_type, inputBytes);

    // Calculate UB related parameters
    uint32_t ubDataNumber = (inputBytes == 1) ? 5 : 3; // 5 for int8, 3 for others
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;

    uint32_t alignNum_x = BLOCK_SIZE / inputBytes;
    uint32_t alignNum_mask = BLOCK_SIZE / sizeof(bool);

    // Calculate total length and alignment
    uint32_t totalLength = data_x;
    uint32_t totalLengthAligned = (totalLength % alignNum_mask == 0U) ?
                                      totalLength :
                                      ((totalLength + alignNum_mask - 1) / alignNum_mask) * alignNum_mask;

    // Calculate core distribution
    uint32_t formerNum = static_cast<uint32_t>((totalLengthAligned / alignNum_mask) % coreNum);
    uint32_t tailLength = (totalLengthAligned / static_cast<uint32_t>(coreNum) / alignNum_mask) * alignNum_mask;
    uint32_t formerLength =
        (formerNum == 0) ?
            0 :
            static_cast<uint32_t>(((totalLengthAligned + coreNum - 1) / coreNum + alignNum_mask - 1) / alignNum_mask) *
                alignNum_mask;

    // Calculate tiling parameters for former and tail
    uint32_t formerTileNum;
    uint32_t formerTileLength;
    uint32_t formerLastTileLength;
    uint32_t tailTileNum;
    uint32_t tailTileLength;
    uint32_t tailLastTileLength;

    TilingParamsCalc(formerLength, alignNum_x, tileBlockNum, formerTileNum, formerTileLength, formerLastTileLength);
    TilingParamsCalc(tailLength, alignNum_x, tileBlockNum, tailTileNum, tailTileLength, tailLastTileLength);

    // Set tiling data
    tiling->formerNum = formerNum;
    tiling->formerLength = formerLength;
    tiling->formerTileNum = formerTileNum;
    tiling->formerTileLength = formerTileLength;
    tiling->formerLastTileLength = formerLastTileLength;
    tiling->tailLength = tailLength;
    tiling->tailTileNum = tailTileNum;
    tiling->tailTileLength = tailTileLength;
    tiling->tailLastTileLength = tailLastTileLength;

    return ge::GRAPH_SUCCESS;
}

// Get workspace size information
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, int32_t wsSysSize, int32_t mySize)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = wsSysSize + mySize;
    return ge::GRAPH_SUCCESS;
}

// Tiling distribution entry
static ge::graphStatus MaskedFillTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    int32_t wsSysSize;
    int32_t mySize = MY_SIZE;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum, wsSysSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    uint32_t data_x = 0;
    uint32_t data_mask = 0;
    uint32_t sShape[MY_SHAPE_SIZE] = {0};
    uint32_t maskShape[MY_SHAPE_SIZE] = {0};
    uint32_t xNumShapes = 0;
    uint32_t maskNumShapes = 0;
    ge::DataType x_type;
    OP_CHECK_IF(
        GetShapeAndTypeInfo(context, data_x, data_mask, sShape, maskShape, xNumShapes, maskNumShapes, x_type) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAndTypeInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetWorkspaceSize(context, wsSysSize, mySize) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    MaskedFillTilingData* tiling = context->GetTilingData<MaskedFillTilingData>();
    tiling->Init();
    OP_CHECK_IF(
        memset_s(tiling, sizeof(MaskedFillTilingData), 0, sizeof(MaskedFillTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalculateTilingData(
            tiling, ubSize, coreNum, data_x, data_mask, xNumShapes, maskNumShapes, sShape, maskShape, x_type) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CalculateTilingData error"), return ge::GRAPH_FAILED);

    context->SetBlockDim(coreNum);
    uint64_t tilingKey = 0;
    const ge::DataType supportedTypes[] = {ge::DT_BF16,   ge::DT_UINT8,  ge::DT_FLOAT16, ge::DT_FLOAT,
                                           ge::DT_INT8,   ge::DT_INT16,  ge::DT_INT32,   ge::DT_DOUBLE,
                                           ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64,  ge::DT_INT64};
    const size_t supportedTypesCount = sizeof(supportedTypes) / sizeof(supportedTypes[0]);
    bool isSupported = false;
    for (size_t i = 0; i < supportedTypesCount; ++i) {
        if (supportedTypes[i] == x_type) {
            isSupported = true;
            break;
        }
    }

    if (isSupported) {
        tilingKey = GET_TPL_TILING_KEY(x_type);
        OP_CHECK_IF(
            context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetTilingKey failed"),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context, "Unsupported data type: %d", static_cast<int>(x_type));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMaskedFill([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// Tiling registration entry
IMPL_OP_OPTILING(MaskedFill).Tiling(MaskedFillTilingFunc).TilingParse<MaskedFillCompileInfo>(TilingParseForMaskedFill);
} // namespace optiling