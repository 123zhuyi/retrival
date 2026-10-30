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

#include <cmath>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/approximate_equal_tiling_data.h"
#include "../op_kernel/approximate_equal_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

constexpr int64_t UB_BYTES_PER_ELEM_FP32 = 20;
constexpr int64_t UB_BYTES_PER_ELEM_FP16 = 20;
constexpr int64_t UB_BYTES_PER_ELEM_BF16 = 20;

constexpr int64_t DTYPE_SIZE_FP32 = 4;
constexpr int64_t DTYPE_SIZE_FP16 = 2;
constexpr int64_t DTYPE_SIZE_BF16 = 2;

constexpr int64_t UB_RESERVED_BYTES = 32 * 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& shape)
{
    if (shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return shape;
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

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType, float& tolerance)
{
    auto inputX1 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX1);
    auto shapeX1 = EnsureNotScalar(inputX1->GetStorageShape());

    auto inputX2 = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX2);
    auto shapeX2 = EnsureNotScalar(inputX2->GetStorageShape());

    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto shapeY = EnsureNotScalar(outY->GetStorageShape());

    OP_CHECK_IF(
        shapeX1.GetShapeSize() != shapeX2.GetShapeSize() || shapeX1.GetShapeSize() != shapeY.GetShapeSize(),
        OP_LOGE(
            context, "ApproximateEqual: shape mismatch: x1=%ld, x2=%ld, y=%ld", shapeX1.GetShapeSize(),
            shapeX2.GetShapeSize(), shapeY.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalNum = shapeX1.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    const std::set<ge::DataType> supported = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    if (supported.count(dataType) == 0) {
        OP_LOGE(
            context, "ApproximateEqual: unsupported dtype=%d (allowed: FLOAT/FLOAT16/BF16)",
            static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }

    auto inputDesc2 = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc2);
    if (inputDesc2->GetDataType() != dataType) {
        OP_LOGE(
            context, "ApproximateEqual: x1 dtype(%d) != x2 dtype(%d)", static_cast<int>(dataType),
            static_cast<int>(inputDesc2->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* tolerancePtr = attrs->GetAttrPointer<float>(0);
    tolerance = (tolerancePtr != nullptr) ? *tolerancePtr : 1e-5f;
    if (!(tolerance >= 0.0f) || std::isnan(tolerance) || std::isinf(tolerance)) {
        OP_LOGE(context, "ApproximateEqual: illegal tolerance=%f", tolerance);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static void PickDtypeParams(ge::DataType dataType, int64_t& inputTypeSize, int64_t& ubBytesPerElem)
{
    if (dataType == ge::DT_FLOAT) {
        inputTypeSize = DTYPE_SIZE_FP32;
        ubBytesPerElem = UB_BYTES_PER_ELEM_FP32;
    } else if (dataType == ge::DT_FLOAT16) {
        inputTypeSize = DTYPE_SIZE_FP16;
        ubBytesPerElem = UB_BYTES_PER_ELEM_FP16;
    } else {
        inputTypeSize = DTYPE_SIZE_BF16;
        ubBytesPerElem = UB_BYTES_PER_ELEM_BF16;
    }
}

static ge::graphStatus InitTilingData(
    gert::TilingContext* context, int64_t totalNum, float tolerance, ApproximateEqualTilingData*& tiling)
{
    tiling = context->GetTilingData<ApproximateEqualTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ApproximateEqualTilingData), 0, sizeof(ApproximateEqualTilingData)) != EOK,
        OP_LOGE(context, "memset tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalNum = totalNum;
    tiling->tolerance = tolerance;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeSplit(
    gert::TilingContext* context, ApproximateEqualTilingData* tiling, uint64_t ubSize, int64_t coreNum,
    ge::DataType dataType)
{
    int64_t inputTypeSize = 0;
    int64_t ubBytesPerElem = 0;
    PickDtypeParams(dataType, inputTypeSize, ubBytesPerElem);
    const int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    const int64_t alignElems = ubBlockSize / inputTypeSize;

    tiling->blockFactor = CeilAlign(CeilDiv(tiling->totalNum, coreNum), alignElems);
    int64_t usedCoreNum = CeilDiv(tiling->totalNum, tiling->blockFactor);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    const int64_t usableUb = static_cast<int64_t>(ubSize) - UB_RESERVED_BYTES;
    int64_t ubFactor = FloorAlign(FloorDiv(usableUb, ubBytesPerElem), alignElems);
    OP_CHECK_IF(
        ubFactor <= 0, OP_LOGE(context, "ApproximateEqual: computed ubFactor<=0 (ubSize=%lu)", ubSize),
        return ge::GRAPH_FAILED);
    tiling->ubFactor = (ubFactor > tiling->blockFactor) ? tiling->blockFactor : ubFactor;

    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApproximateEqualTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    float tolerance = 1e-5f;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalNum, dataType, tolerance) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ApproximateEqualTilingData* tiling = nullptr;
    OP_CHECK_IF(
        InitTilingData(context, totalNum, tolerance, tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "InitTilingData error"), return ge::GRAPH_FAILED);

    if (totalNum == 0) {
        context->SetBlockDim(1);
    } else {
        OP_CHECK_IF(
            ComputeSplit(context, tiling, ubSize, coreNum, dataType) != ge::GRAPH_SUCCESS,
            OP_LOGE(context, "ComputeSplit error"), return ge::GRAPH_FAILED);
    }

    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForApproximateEqual([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApproximateEqualCompileInfo {};

IMPL_OP_OPTILING(ApproximateEqual)
    .Tiling(ApproximateEqualTilingFunc)
    .TilingParse<ApproximateEqualCompileInfo>(TilingParseForApproximateEqual);

} // namespace optiling
