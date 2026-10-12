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
 * \file softshrink_grad_tiling.cpp
 * \brief SoftShrinkGrad 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/softshrink_grad_tiling_data.h"
#include "../op_kernel/softshrink_grad_tiling_key.h"

#include <algorithm>
#include <set>

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MAIN_ALIGN_BYTES = 512;
constexpr int64_t VECTOR_ALIGN_BYTES = 256;
constexpr int64_t UB_SYSTEM_OVERHEAD = 16 * 1024;
constexpr int64_t SINGLE_CORE_THRESHOLD = 64 * 1024;
constexpr int64_t TARGET_BYTES_PER_CORE = 16 * 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static bool IsSameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeDtypeAndAttr(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType,
                                            float& lambd)
{
    auto gradShapePtr = context->GetInputShape(0);
    auto xShapePtr = context->GetInputShape(1);
    auto outputShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapePtr);

    const gert::Shape gradShape = EnsureNotScalar(gradShapePtr->GetStorageShape());
    const gert::Shape xShape = EnsureNotScalar(xShapePtr->GetStorageShape());
    const gert::Shape outputShape = EnsureNotScalar(outputShapePtr->GetStorageShape());
    OP_CHECK_IF(!IsSameShape(gradShape, xShape) || !IsSameShape(gradShape, outputShape),
                OP_LOGE(context, "input_grad, input_x and output_y must have the same shape"), return ge::GRAPH_FAILED);
    totalNum = gradShape.GetShapeSize();
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "invalid totalNum: %ld", totalNum), return ge::GRAPH_FAILED);

    auto gradDesc = context->GetInputDesc(0);
    auto xDesc = context->GetInputDesc(1);
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    dataType = gradDesc->GetDataType();
    const std::set<ge::DataType> supportedTypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    OP_CHECK_IF(supportedTypes.count(dataType) == 0 || xDesc->GetDataType() != dataType ||
                    outputDesc->GetDataType() != dataType,
                OP_LOGE(context, "input/output dtype must be identical FP16/FP32/BF16"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* lambdPtr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, lambdPtr);
    lambd = *lambdPtr;
    OP_CHECK_IF(lambd < 0.0f, OP_LOGE(context, "lambd must be non-negative, got %f", lambd), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static int64_t GetTypeSize(ge::DataType dataType)
{
    return dataType == ge::DT_FLOAT ? static_cast<int64_t>(sizeof(float)) : static_cast<int64_t>(sizeof(uint16_t));
}

static int64_t CalcUbFactor(uint64_t ubSize, ge::DataType dataType, int64_t bufferNum, int64_t totalNum)
{
    const int64_t typeSize = GetTypeSize(dataType);
    const int64_t availableUb = static_cast<int64_t>(ubSize) - UB_SYSTEM_OVERHEAD;
    if (availableUb <= 0) {
        return 0;
    }

    int64_t bytesPerElem = 3 * bufferNum * typeSize + 1;
    if (dataType != ge::DT_FLOAT) {
        bytesPerElem += 2 * static_cast<int64_t>(sizeof(float));
    }
    const int64_t mainAlignElems = MAIN_ALIGN_BYTES / typeSize;
    const int64_t vectorAlignElems = VECTOR_ALIGN_BYTES / static_cast<int64_t>(sizeof(float));
    const int64_t alignElems = std::max(mainAlignElems, vectorAlignElems);
    int64_t ubFactor = FloorAlign(FloorDiv(availableUb, bytesPerElem), alignElems);
    if (ubFactor <= 0) {
        return 0;
    }
    if (totalNum > 0 && totalNum < ubFactor) {
        ubFactor = CeilAlign(totalNum, alignElems);
    }
    return ubFactor;
}

static int64_t CalcBlockFactor(int64_t totalNum, int64_t typeSize, int64_t maxCoreNum, int64_t& usedCoreNum)
{
    if (totalNum <= 0) {
        usedCoreNum = 1;
        return 0;
    }

    const int64_t totalBytes = totalNum * typeSize;
    if (totalBytes <= SINGLE_CORE_THRESHOLD) {
        usedCoreNum = 1;
        return totalNum;
    }

    usedCoreNum = std::min(maxCoreNum, CeilDiv(totalBytes, TARGET_BYTES_PER_CORE));
    usedCoreNum = std::max<int64_t>(usedCoreNum, 1);
    const int64_t alignElems = MAIN_ALIGN_BYTES / typeSize;
    const int64_t blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), alignElems);
    usedCoreNum = CeilDiv(totalNum, blockFactor);
    return blockFactor;
}

static uint64_t GetTilingKey(ge::DataType dataType, bool doubleBuffer)
{
    if (dataType == ge::DT_FLOAT16) {
        return doubleBuffer ? GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_1) :
                              GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_0);
    }
    if (dataType == ge::DT_FLOAT) {
        return doubleBuffer ? GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_3) :
                              GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_2);
    }
    return doubleBuffer ? GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_5) :
                          GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_4);
}

static ge::graphStatus SoftShrinkGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t maxCoreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, maxCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    SoftShrinkGradTilingData* tiling = context->GetTilingData<SoftShrinkGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    int64_t totalNum = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    float lambd = 0.5f;
    OP_CHECK_IF(GetShapeDtypeAndAttr(context, totalNum, dataType, lambd) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeDtypeAndAttr error"), return ge::GRAPH_FAILED);

    const int64_t typeSize = GetTypeSize(dataType);
    int64_t usedCoreNum = 1;
    const int64_t blockFactor = CalcBlockFactor(totalNum, typeSize, maxCoreNum, usedCoreNum);
    const int64_t singleUbFactor = CalcUbFactor(ubSize, dataType, 1, totalNum);
    OP_CHECK_IF(singleUbFactor <= 0, OP_LOGE(context, "failed to calculate single-buffer UB factor"),
                return ge::GRAPH_FAILED);

    bool doubleBuffer = false;
    int64_t ubFactor = singleUbFactor;
    if (blockFactor > singleUbFactor) {
        const int64_t doubleUbFactor = CalcUbFactor(ubSize, dataType, 2, totalNum);
        OP_CHECK_IF(doubleUbFactor <= 0, OP_LOGE(context, "failed to calculate double-buffer UB factor"),
                    return ge::GRAPH_FAILED);
        if (CeilDiv(blockFactor, doubleUbFactor) >= 2) {
            doubleBuffer = true;
            ubFactor = doubleUbFactor;
        }
    }

    *tiling = SoftShrinkGradTilingData{};
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = static_cast<int32_t>(ubFactor);
    tiling->lambd = lambd;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(GetTilingKey(dataType, doubleBuffer));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSoftShrinkGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SoftShrinkGradCompileInfo {};

IMPL_OP_OPTILING(SoftshrinkGrad)
    .Tiling(SoftShrinkGradTilingFunc)
    .TilingParse<SoftShrinkGradCompileInfo>(TilingParseForSoftShrinkGrad);

} // namespace optiling
