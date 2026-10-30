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
 * \file tensor_move_tiling.cpp
 * \brief TensorMove tiling for Ascend910B.
 */

#include <set>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/tensor_move_tiling_data.h"
#include "../op_kernel/tensor_move_tiling_key.h"

namespace optiling {

using namespace Ops::Base;

namespace {
constexpr int64_t INDEX_INPUT_X = 0;
constexpr int64_t INDEX_OUTPUT_Y = 0;
constexpr int64_t RESERVED_UB_SIZE = 8 * 1024;
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t ONE_BLOCK_BYTES = 32;
constexpr int64_t MIN_CORE_CHUNK_BYTES = 2048;
constexpr size_t WORKSPACE_SIZE = 32;
} // namespace

struct TensorMoveCompileInfo {};

static bool IsTensorMoveDtypeSupported(ge::DataType dtype)
{
    static const std::set<ge::DataType> supportedDtypes = {
        ge::DT_BF16,   ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_UINT8, ge::DT_INT8, ge::DT_UINT16, ge::DT_INT16,
        ge::DT_UINT32, ge::DT_INT32,   ge::DT_UINT64, ge::DT_INT64, ge::DT_BOOL, ge::DT_DOUBLE};
    return supportedDtypes.count(dtype) > 0;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "Failed to get valid AIV core count."), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= RESERVED_UB_SIZE, OP_LOGE(context, "UB size is too small."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorMoveIO(gert::TilingContext* context, ge::DataType& dtype, int64_t& numel)
{
    auto inputDesc = context->GetInputDesc(INDEX_INPUT_X);
    auto outputDesc = context->GetOutputDesc(INDEX_OUTPUT_Y);
    auto inputShape = context->GetInputShape(INDEX_INPUT_X);
    auto outputShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    dtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        !IsTensorMoveDtypeSupported(dtype),
        OP_LOGE(context, "TensorMove only supports bf16/f16/f32/u8/i8/u16/i16/u32/i32/u64/i64/bool/double on 910B."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputDesc->GetDataType() != dtype,
                OP_LOGE(context, "The dtype of output y must be the same as input x."), return ge::GRAPH_FAILED);

    const auto& xShape = inputShape->GetStorageShape();
    const auto& yShape = outputShape->GetStorageShape();
    OP_CHECK_IF(xShape != yShape, OP_LOGE(context, "The shape of output y must be the same as input x."),
                return ge::GRAPH_FAILED);

    numel = xShape.GetShapeSize();
    OP_CHECK_IF(numel < 0, OP_LOGE(context, "TensorMove does not support negative runtime shape size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static int64_t GetTypeBytes(gert::TilingContext* context, ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BOOL:
        case ge::DT_UINT8:
        case ge::DT_INT8:
            return 1;
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
        case ge::DT_UINT16:
        case ge::DT_INT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_UINT32:
        case ge::DT_INT32:
            return 4;
        case ge::DT_DOUBLE:
        case ge::DT_UINT64:
        case ge::DT_INT64:
            return 8;
        default:
            OP_LOGE(context, "Failed to get dtype byte size.");
            return 0;
    }
}

static uint64_t GetTilingKeyByTypeBytes(int64_t typeBytes)
{
    switch (typeBytes) {
        case 1:
            return TENSOR_MOVE_TPL_SCH_MODE_0;
        case 2:
            return TENSOR_MOVE_TPL_SCH_MODE_1;
        case 4:
            return TENSOR_MOVE_TPL_SCH_MODE_2;
        case 8:
            return TENSOR_MOVE_TPL_SCH_MODE_3;
        default:
            return TENSOR_MOVE_TPL_SCH_MODE_0;
    }
}

static void CalcBlockFactor(TensorMoveTilingData* tiling, int64_t numel)
{
    const int64_t tileLoopCount = CeilDiv(numel, tiling->ubFactor);
    tiling->tailBlockTailUbFactor = numel % tiling->ubFactor;

    const int64_t coreTileLoops = CeilDiv(tileLoopCount, tiling->totalCoreNum);
    tiling->usedCoreNum = CeilDiv(tileLoopCount, coreTileLoops);
    tiling->blockFactor = CeilDiv(tileLoopCount, tiling->usedCoreNum);
    tiling->tailBlockFactor = tileLoopCount - (tiling->usedCoreNum - 1) * tiling->blockFactor;

    if (tiling->tailBlockTailUbFactor == 0 && numel > 0) {
        tiling->tailBlockTailUbFactor = tiling->ubFactor;
    }
}

static void InitTensorMoveTilingData(TensorMoveTilingData* tiling, int64_t numel, int64_t maxUbFactor,
                                     int64_t totalCoreNum)
{
    tiling->totalCoreNum = totalCoreNum;
    tiling->usedCoreNum = 0;
    tiling->blockFactor = 0;
    tiling->tailBlockFactor = 0;
    tiling->tailBlockTailUbFactor = 0;
    tiling->ubFactor = numel >= maxUbFactor ? maxUbFactor : numel;
    tiling->totalLength = numel;
}

static ge::graphStatus AdjustTensorMoveUbFactor(gert::TilingContext* context, TensorMoveTilingData* tiling,
                                                int64_t numel, int64_t typeBytes, int64_t totalCoreNum,
                                                int64_t blockElemCount, int64_t maxUbFactor)
{
    if (typeBytes <= 0 || totalCoreNum <= 1) {
        OP_LOGE(context, "TensorMove adjusted tiling divisor is invalid.");
        return ge::GRAPH_FAILED;
    }

    const int64_t safeTypeBytes = typeBytes > 0 ? typeBytes : 1;
    const int64_t safeTotalCoreNum = totalCoreNum > 1 ? totalCoreNum : 2;
    if (numel % safeTotalCoreNum == 0) {
        tiling->ubFactor = FloorDiv(numel, safeTotalCoreNum);
    } else {
        tiling->ubFactor = FloorDiv(numel, safeTotalCoreNum - 1);
    }
    tiling->ubFactor = CeilAlign(tiling->ubFactor, blockElemCount);
    const int64_t minUbFactor = MIN_CORE_CHUNK_BYTES / safeTypeBytes;
    if (tiling->ubFactor < minUbFactor) {
        tiling->ubFactor = minUbFactor;
    }
    if (tiling->ubFactor > maxUbFactor) {
        tiling->ubFactor = FloorAlign(maxUbFactor, blockElemCount);
    }
    OP_CHECK_IF(tiling->ubFactor <= 0, OP_LOGE(context, "Adjusted ubFactor is invalid."), return ge::GRAPH_FAILED);

    CalcBlockFactor(tiling, numel);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTensorMoveTiling(gert::TilingContext* context, TensorMoveTilingData* tiling, int64_t numel,
                                          int64_t typeBytes, int64_t totalCoreNum, uint64_t ubSize)
{
    if (typeBytes <= 0) {
        OP_LOGE(context, "TensorMove dtype byte size must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    if (totalCoreNum <= 1) {
        OP_LOGE(context, "TensorMove AIV core count must be greater than 1.");
        return ge::GRAPH_FAILED;
    }

    const int64_t safeTypeBytes = typeBytes > 0 ? typeBytes : 1;
    const int64_t safeTotalCoreNum = totalCoreNum > 1 ? totalCoreNum : 2;
    const int64_t blockElemCount = ONE_BLOCK_BYTES / safeTypeBytes;
    const int64_t ubAvailableBytes = static_cast<int64_t>(ubSize) - RESERVED_UB_SIZE;
    OP_CHECK_IF(ubAvailableBytes <= 0, OP_LOGE(context, "Reserved UB leaves no available space."),
                return ge::GRAPH_FAILED);

    const int64_t maxUbFactor = ubAvailableBytes / DOUBLE_BUFFER_NUM / safeTypeBytes;
    OP_CHECK_IF(maxUbFactor <= 0, OP_LOGE(context, "UB is not enough for one TensorMove tile."),
                return ge::GRAPH_FAILED);

    InitTensorMoveTilingData(tiling, numel, maxUbFactor, safeTotalCoreNum);
    if (tiling->ubFactor > 0) {
        tiling->ubFactor = FloorAlign(tiling->ubFactor, blockElemCount);
    }
    if (tiling->ubFactor <= 0) {
        tiling->ubFactor = blockElemCount;
    }

    CalcBlockFactor(tiling, numel);
    if (tiling->usedCoreNum == tiling->totalCoreNum || tiling->blockFactor > 1) {
        return ge::GRAPH_SUCCESS;
    }

    return AdjustTensorMoveUbFactor(context, tiling, numel, safeTypeBytes, safeTotalCoreNum, blockElemCount,
                                    maxUbFactor);
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TensorMoveTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed."), return ge::GRAPH_FAILED);

    ge::DataType dtype = ge::DT_UNDEFINED;
    int64_t numel = 0;
    OP_CHECK_IF(CheckTensorMoveIO(context, dtype, numel) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "TensorMove input/output validation failed."), return ge::GRAPH_FAILED);

    TensorMoveTilingData* tiling = context->GetTilingData<TensorMoveTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TensorMoveTilingData), 0, sizeof(TensorMoveTilingData)) != EOK,
                OP_LOGE(context, "Failed to reset TensorMove tiling data."), return ge::GRAPH_FAILED);

    const int64_t typeBytes = GetTypeBytes(context, dtype);
    OP_CHECK_IF(typeBytes <= 0, OP_LOGE(context, "Failed to determine TensorMove dtype size."),
                return ge::GRAPH_FAILED);

    context->SetTilingKey(GetTilingKeyByTypeBytes(typeBytes));
    OP_CHECK_IF(SetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetWorkspaceSize failed."),
                return ge::GRAPH_FAILED);

    if (numel == 0) {
        context->SetBlockDim(1);
        context->GetRawTilingData()->SetDataSize(sizeof(TensorMoveTilingData));
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(DoTensorMoveTiling(context, tiling, numel, typeBytes, coreNum, ubSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "DoTensorMoveTiling failed."), return ge::GRAPH_FAILED);

    context->SetBlockDim(static_cast<uint32_t>(tiling->usedCoreNum));
    context->GetRawTilingData()->SetDataSize(sizeof(TensorMoveTilingData));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTensorMove([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TensorMove).Tiling(TensorMoveTilingFunc).TilingParse<TensorMoveCompileInfo>(TilingParseForTensorMove);
} // namespace optiling
