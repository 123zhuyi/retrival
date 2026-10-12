/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file acos_grad_tiling.cpp
 * \brief AcosGrad 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "../../op_kernel/arch35/acos_grad_tiling_data.h"
#include "../../op_kernel/arch35/acos_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint32_t ELEM_ALIGN = 512U;
constexpr uint64_t UB_SIZE_BYTES = 184U * 1024U;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint32_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t coreNumI64 = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNumI64 <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    coreNum = static_cast<uint32_t>(coreNumI64);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeInfo(gert::TilingContext* context, uint64_t& totalLength, ge::DataType& dataType)
{
    auto inputYGrad = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputYGrad);
    auto yGradShape = EnsureNotScalar(inputYGrad->GetStorageShape());

    auto inputX = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto xShape = EnsureNotScalar(inputX->GetStorageShape());

    auto outputXGrad = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputXGrad);
    auto xGradShape = EnsureNotScalar(outputXGrad->GetStorageShape());

    OP_CHECK_IF(
        yGradShape.GetShapeSize() != xShape.GetShapeSize() || yGradShape.GetShapeSize() != xGradShape.GetShapeSize(),
        OP_LOGE(
            context, "AcosGrad: shape size mismatch: y_grad=%ld, x=%ld, x_grad=%ld", yGradShape.GetShapeSize(),
            xShape.GetShapeSize(), xGradShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalLength = static_cast<uint64_t>(yGradShape.GetShapeSize());

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "AcosGrad: unsupported dtype %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static void CalcTilingParams(
    uint64_t totalLength, uint32_t availCoreNum, ge::DataType dataType, AcosGradTilingData* tiling)
{
    uint32_t coreNum =
        static_cast<uint32_t>(CeilDiv(static_cast<int64_t>(totalLength), static_cast<int64_t>(ELEM_ALIGN)));
    if (coreNum > availCoreNum) {
        coreNum = availCoreNum;
    }
    if (coreNum < 1U) {
        coreNum = 1U;
    }

    uint32_t blockFormerRaw =
        static_cast<uint32_t>(CeilDiv(static_cast<int64_t>(totalLength), static_cast<int64_t>(coreNum)));
    uint32_t blockFormer = static_cast<uint32_t>(
        CeilDiv(static_cast<int64_t>(blockFormerRaw), static_cast<int64_t>(ELEM_ALIGN)) * ELEM_ALIGN);
    if (blockFormer < ELEM_ALIGN) {
        blockFormer = ELEM_ALIGN;
    }

    uint32_t blockNum =
        static_cast<uint32_t>(CeilDiv(static_cast<int64_t>(totalLength), static_cast<int64_t>(blockFormer)));
    if (blockNum < 1U) {
        blockNum = 1U;
    }

    uint32_t bytesPerElem;
    uint32_t alignFactor;

    if (dataType == ge::DT_FLOAT) {
        bytesPerElem = 32U;
        alignFactor = 64U;
    } else {
        bytesPerElem = 28U;
        alignFactor = 128U;
    }

    uint32_t ubFormerRaw = static_cast<uint32_t>(UB_SIZE_BYTES / bytesPerElem);
    uint32_t ubFormer = static_cast<uint32_t>(
        FloorDiv(static_cast<int64_t>(ubFormerRaw), static_cast<int64_t>(alignFactor)) * alignFactor);
    if (ubFormer < alignFactor) {
        ubFormer = alignFactor;
    }
    if (ubFormer > blockFormer) {
        ubFormer = blockFormer;
    }

    uint64_t tailBlockStart = static_cast<uint64_t>(blockNum - 1) * blockFormer;
    uint32_t tailBlockLen = (tailBlockStart < totalLength) ? static_cast<uint32_t>(totalLength - tailBlockStart) : 0U;

    uint32_t ubLoopOfFormerBlock = (ubFormer > 0U) ? (blockFormer / ubFormer) : 0U;
    uint32_t ubTailOfFormerBlock = (ubFormer > 0U) ? (blockFormer % ubFormer) : blockFormer;

    uint32_t ubLoopOfTailBlock = (ubFormer > 0U && tailBlockLen > 0U) ? (tailBlockLen / ubFormer) : 0U;
    uint32_t ubTailOfTailBlock = (ubFormer > 0U && tailBlockLen > 0U) ? (tailBlockLen % ubFormer) : tailBlockLen;

    tiling->totalLength = totalLength;
    tiling->blockFormer = blockFormer;
    tiling->blockNum = blockNum;
    tiling->ubFormer = ubFormer;
    tiling->ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    tiling->ubTailOfFormerBlock = ubTailOfFormerBlock;
    tiling->ubLoopOfTailBlock = ubLoopOfTailBlock;
    tiling->ubTailOfTailBlock = ubTailOfTailBlock;
}

static ge::graphStatus AcosGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0UL;
    uint32_t coreNum = 0U;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "AcosGrad: GetPlatformInfo error"), return ge::GRAPH_FAILED);

    OP_LOGI(context, "[AcosGrad Tiling] coreNum=%u, ubSize=%lu", coreNum, ubSize);

    uint64_t totalLength = 0UL;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeInfo(context, totalLength, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "AcosGrad: GetShapeInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "AcosGrad: SetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    if (totalLength == 0UL) {
        AcosGradTilingData* tiling = context->GetTilingData<AcosGradTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(AcosGradTilingData), 0, sizeof(AcosGradTilingData));
        context->SetBlockDim(1U);
        uint32_t dTypeX = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dTypeX);
        return ge::GRAPH_SUCCESS;
    }

    AcosGradTilingData* tiling = context->GetTilingData<AcosGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AcosGradTilingData), 0, sizeof(AcosGradTilingData)) != EOK,
        OP_LOGE(context, "AcosGrad: memset_s tiling data error"), return ge::GRAPH_FAILED);

    CalcTilingParams(totalLength, coreNum, dataType, tiling);

    context->SetBlockDim(tiling->blockNum);

    OP_LOGI(
        context,
        "[AcosGrad Tiling] totalLength=%lu, blockFormer=%u, blockNum=%u, ubFormer=%u, "
        "ubLoopFormer=%u, ubTailFormer=%u, ubLoopTail=%u, ubTailTail=%u",
        tiling->totalLength, tiling->blockFormer, tiling->blockNum, tiling->ubFormer, tiling->ubLoopOfFormerBlock,
        tiling->ubTailOfFormerBlock, tiling->ubLoopOfTailBlock, tiling->ubTailOfTailBlock);

    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAcosGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AcosGradCompileInfo {};

IMPL_OP_OPTILING(AcosGrad).Tiling(AcosGradTilingFunc).TilingParse<AcosGradCompileInfo>(TilingParseForAcosGrad);

} // namespace optiling
