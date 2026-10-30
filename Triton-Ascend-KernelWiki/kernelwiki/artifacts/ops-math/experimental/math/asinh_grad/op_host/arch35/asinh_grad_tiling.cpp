/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asinh_grad_tiling.cpp
 * \brief AsinhGrad arch35 tiling implementation
 *
 * Tiling strategy:
 *   - Multi-core: blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), 32B)
 *   - UB: ubFactor = FloorAlign(FloorDiv(ubSize / bytesPerElem), 32B)
 *   - Double buffer threshold: totalNum > 1024
 *
 * FP32 path:
 *   - 3 queue tensors (y, dy, z) x BUFFER_NUM + 2 temp (expY, exp2Y)
 *   - Double buffer: bytesPerElem = (3*2 + 2) * sizeof(float) = 32 bytes
 *   - Single buffer: bytesPerElem = (3*1 + 2) * sizeof(float) = 20 bytes
 *
 * FP16 path (upgrade to FP32):
 *   - 3 queue tensors (y, dy, z) x BUFFER_NUM (sizeof(half)) + 5 FP32 temp (yFp32, dyFp32, expY, exp2Y, zFp32)
 *   - Double buffer: (3*2*2 + 5*4) = 32 bytes/elem
 *   - Single buffer: (3*1*2 + 5*4) = 26 bytes/elem
 *   - Conservative: use 9*sizeof(float) for double, 5*sizeof(float) for single (safe for both paths)
 *
 * Iteration 3: supports FP32 + FP16 + BF16 (full coverage)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/asinh_grad_tiling_data.h"
#include "../../op_kernel/arch35/asinh_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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
    auto inputY = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto inputShapeY = EnsureNotScalar(inputY->GetStorageShape());

    auto inputDy = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDy);
    auto inputShapeDy = EnsureNotScalar(inputDy->GetStorageShape());

    auto outZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outZ);
    auto outShapeZ = EnsureNotScalar(outZ->GetStorageShape());

    OP_CHECK_IF(
        inputShapeY.GetShapeSize() != inputShapeDy.GetShapeSize() ||
            inputShapeY.GetShapeSize() != outShapeZ.GetShapeSize(),
        OP_LOGE(
            context, "AsinhGrad: shape mismatch: y=%ld, dy=%ld, z=%ld", inputShapeY.GetShapeSize(),
            inputShapeDy.GetShapeSize(), outShapeZ.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalIdx = inputShapeY.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    // Iteration 3: support DT_FLOAT, DT_FLOAT16 and DT_BF16
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "AsinhGrad: unsupported dtype %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }

    // Verify y and dy have same dtype
    auto inputDescDy = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescDy);
    OP_CHECK_IF(
        inputDescDy->GetDataType() != dataType, OP_LOGE(context, "AsinhGrad: y/dy dtype mismatch"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AsinhGradTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get shape and dtype info
    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. Calculate tiling parameters
    AsinhGradTilingData* tiling = context->GetTilingData<AsinhGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AsinhGradTilingData), 0, sizeof(AsinhGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    if (totalIdx == 0) {
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

    // UB factor calculation:
    // Conservative estimation using sizeof(float) as base unit to cover both FP32 and FP16 paths:
    //   FP32 double buffer: (3*2 + 2) * 4 = 32 bytes/elem -> divisor 8, use 9 with safety margin
    //   FP16 double buffer: (3*2*2 + 5*4) = 32 bytes/elem -> divisor 8, use 9 with safety margin
    //   FP32 single buffer: (3*1 + 2) * 4 = 20 bytes/elem -> divisor 5
    //   FP16 single buffer: (3*1*2 + 5*4) = 26 bytes/elem -> divisor ~7, use 5*4=20 safe for FP32
    // Use conservative divisor: 9 (double) / 7 (single) in sizeof(float) units for FP16 safety
    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum;
    if (dataType == ge::DT_FLOAT) {
        // FP32: 3 queues * BUFFER_NUM + 2 temp
        bufferNum = useDoubleBuffer ? 8 : 5;
    } else {
        // FP16/BF16: 3 queues (half/bf16) * BUFFER_NUM + 5 FP32 temp
        // Double: (3*2*sizeof(half) + 5*sizeof(float)) / sizeof(float) = (12+20)/4 = 8
        // Single: (3*1*sizeof(half) + 5*sizeof(float)) / sizeof(float) = (6+20)/4 = 6.5 -> 7
        bufferNum = useDoubleBuffer ? 8 : 7;
    }
    tiling->ubFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize) / static_cast<int64_t>(sizeof(float)), bufferNum), ubBlockSize);

    context->SetBlockDim(usedCoreNum);

    // 5. Set TilingKey
    uint32_t dTypeY = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeY, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAsinhGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AsinhGradCompileInfo {};

IMPL_OP_OPTILING(AsinhGrad).Tiling(AsinhGradTilingFunc).TilingParse<AsinhGradCompileInfo>(TilingParseForAsinhGrad);

} // namespace optiling
