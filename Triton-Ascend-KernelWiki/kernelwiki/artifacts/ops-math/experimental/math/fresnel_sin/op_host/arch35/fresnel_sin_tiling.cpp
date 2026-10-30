/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Disclaimer: This file is generated with the assistance of an AI tool.
 * Please review carefully before use.
 *
 * FresnelSin Tiling implementation (arch35)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/fresnel_sin_tiling_data.h"
#include "../../op_kernel/arch35/fresnel_sin_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t TILE_LENGTH = 4096; // elements per tile (DB chunk)
constexpr uint32_t MIN_TILE = 64;      // Compare/Select 256B align -> 64 fp32

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto outShapeY = EnsureNotScalar(outY->GetStorageShape());

    OP_CHECK_IF(
        inputShapeX.GetShapeSize() != outShapeY.GetShapeSize(),
        OP_LOGE(context, "FresnelSin: input and output shape size mismatch"), return ge::GRAPH_FAILED);

    totalIdx = inputShapeX.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        supportedDtype.count(dataType) == 0,
        OP_LOGE(context, "FresnelSin: invalid dtype %d", static_cast<int>(dataType)), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    // 通过 GetLibApiWorkSpaceSize 获取系统默认 workspace 大小，避免硬编码。
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FresnelSinTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t totalIdx = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    FresnelSinTilingData* tiling = context->GetTilingData<FresnelSinTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(FresnelSinTilingData), 0, sizeof(FresnelSinTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Multi-core split: dynamic GetBlockDim, 32B aligned baseLength
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);

    // Effective core count: do not exceed ceil(totalIdx / MIN_TILE)
    int64_t maxCoreByTask = (totalIdx + MIN_TILE - 1) / MIN_TILE;
    if (maxCoreByTask < 1)
        maxCoreByTask = 1;
    int64_t usedCoreNum = std::min(coreNum, maxCoreByTask);
    if (usedCoreNum < 1)
        usedCoreNum = 1;

    int64_t baseLen = CeilAlign(CeilDiv(totalIdx, usedCoreNum), ubBlockSize);
    // Guard against divide-by-zero: empty tensor (totalIdx==0) or alignment producing baseLen==0
    if (baseLen <= 0) {
        baseLen = (ubBlockSize > 0) ? ubBlockSize : 1;
    }
    // Recompute core count from aligned baseLen
    usedCoreNum = (totalIdx + baseLen - 1) / baseLen;
    if (usedCoreNum < 1)
        usedCoreNum = 1;
    int64_t tailLen = totalIdx - baseLen * (usedCoreNum - 1);
    if (tailLen < 0)
        tailLen = 0;

    tiling->totalLength = static_cast<uint64_t>(totalIdx);
    tiling->blockNum = static_cast<uint32_t>(usedCoreNum);
    tiling->baseLength = static_cast<uint32_t>(baseLen);
    tiling->tailLength = static_cast<uint32_t>(tailLen);
    tiling->tileLength = TILE_LENGTH;
    tiling->tileNum = static_cast<uint32_t>((baseLen + TILE_LENGTH - 1) / TILE_LENGTH);
    tiling->tailTileLen = static_cast<uint32_t>(baseLen % TILE_LENGTH);

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // Set TilingKey based on dtype
    uint32_t schMode = FRESNEL_SIN_KEY_FP32;
    if (dataType == ge::DT_FLOAT) {
        schMode = FRESNEL_SIN_KEY_FP32;
    } else if (dataType == ge::DT_FLOAT16) {
        schMode = FRESNEL_SIN_KEY_FP16;
    } else if (dataType == ge::DT_BF16) {
        schMode = FRESNEL_SIN_KEY_BF16;
    }

    ASCENDC_TPL_SEL_PARAM(context, schMode);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFresnelSin([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct FresnelSinCompileInfo {};

IMPL_OP_OPTILING(FresnelSin).Tiling(FresnelSinTilingFunc).TilingParse<FresnelSinCompileInfo>(TilingParseForFresnelSin);

} // namespace optiling
