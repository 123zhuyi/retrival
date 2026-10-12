/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <set>
#include "log/log.h"
#include "op_host/math_tiling_templates_registry.h"
#include "op_host/tiling_base_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/const_util.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/cumulative_logsumexp_tiling_data.h"
#include "../../op_kernel/arch35/cumulative_logsumexp_tiling_key.h"

namespace optiling {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_AXIS_INDEX = 1;
constexpr size_t ATTR_EXCLUSIVE_INDEX = 0;
constexpr size_t ATTR_REVERSE_INDEX = 1;
constexpr uint32_t WS_SYS_SIZE = 0U;

struct CumulativeLogsumexpCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum", "0",
                                                      "coreNum must be greater than 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
static ge::graphStatus ReadConstAxis(gert::TilingContext* context, const gert::Tensor* axisTensor, int64_t& axis)
{
    const T* axisData = axisTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, axisData);
    axis = static_cast<int64_t>(axisData[0]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAxisValue(gert::TilingContext* context, int64_t& axis)
{
    const gert::Tensor* axisTensor = context->GetInputTensor(INPUT_AXIS_INDEX);
    OP_CHECK_IF(axisTensor == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis", "unknown",
                                                      "axis must be a compile-time constant scalar."),
                return ge::GRAPH_FAILED);

    ge::DataType axisType = axisTensor->GetDataType();
    switch (axisType) {
        case ge::DT_INT16:
            return ReadConstAxis<int16_t>(context, axisTensor, axis);
        case ge::DT_INT32:
            return ReadConstAxis<int32_t>(context, axisTensor, axis);
        case ge::DT_INT64:
            return ReadConstAxis<int64_t>(context, axisTensor, axis);
        default:
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "axis", Ops::Base::ToString(axisType).c_str(),
                                                  "axis dtype must be int16, int32 or int64.");
            return ge::GRAPH_FAILED;
    }
}

static ge::graphStatus ParseShapeAndAttrs(gert::TilingContext* context, CumulativeLogsumexpTilingData* tiling,
                                          ge::DataType& dataType)
{
    auto inputDesc = context->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    OP_CHECK_IF(
        supportedDtype.count(dataType) == 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(dataType).c_str(),
                                              "x dtype must be float32 or float16."),
        return ge::GRAPH_FAILED);

    auto xShapeInfo = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapeInfo);
    const gert::Shape& xShape = Ops::Base::EnsureNotScalar(xShapeInfo->GetStorageShape());
    size_t dimNum = xShape.GetDimNum();
    OP_CHECK_IF(dimNum == 0, OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "x", "scalar", "x must not be scalar."),
                return ge::GRAPH_FAILED);

    int64_t axis = 0;
    OP_CHECK_IF(GetAxisValue(context, axis) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis", "unknown",
                                                      "axis must be a compile-time constant scalar."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(axis < -static_cast<int64_t>(dimNum) || axis >= static_cast<int64_t>(dimNum),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis", std::to_string(axis).c_str(),
                                                      ("axis must be in range [-" + std::to_string(dimNum) + ", " +
                                                       std::to_string(static_cast<int64_t>(dimNum) - 1) + "].")
                                                          .c_str()),
                return ge::GRAPH_FAILED);
    if (axis < 0) {
        axis += static_cast<int64_t>(dimNum);
    }

    int64_t outerNum = 1;
    int64_t innerNum = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outerNum *= xShape.GetDim(i);
    }
    for (int64_t i = axis + 1; i < static_cast<int64_t>(dimNum); ++i) {
        innerNum *= xShape.GetDim(i);
    }
    int64_t axisNum = xShape.GetDim(axis);
    int64_t totalNum = xShape.GetShapeSize();
    OP_CHECK_IF(axisNum <= 0 || totalNum < 0,
                OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "x", Ops::Base::ToString(xShape).c_str(),
                                          "axis dimension must be positive."),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* exclusive = attrs->GetAttrPointer<bool>(ATTR_EXCLUSIVE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, exclusive);
    const bool* reverse = attrs->GetAttrPointer<bool>(ATTR_REVERSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, reverse);

    tiling->totalNum = totalNum;
    tiling->outerNum = outerNum;
    tiling->axisNum = axisNum;
    tiling->innerNum = innerNum;
    tiling->exclusive = *exclusive ? 1 : 0;
    tiling->reverse = *reverse ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CumulativeLogsumexpTilingFunc(gert::TilingContext* context)
{
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "platformInfo", "invalid",
                                                      "failed to get platform information."),
                return ge::GRAPH_FAILED);

    CumulativeLogsumexpTilingData* tiling = context->GetTilingData<CumulativeLogsumexpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(CumulativeLogsumexpTilingData), 0, sizeof(CumulativeLogsumexpTilingData)) != EOK,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tilingData", "memset_s failed",
                                              "tiling data must be initialized successfully."),
        return ge::GRAPH_FAILED);

    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(ParseShapeAndAttrs(context, tiling, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "shapeOrAttr", "invalid",
                                                      "failed to parse input shape or attributes."),
                return ge::GRAPH_FAILED);

    int64_t usedCoreNum = Ops::Base::CeilDiv(tiling->totalNum, coreNum);
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }
    context->SetBlockDim(usedCoreNum);

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = WS_SYS_SIZE;

    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(CUMULATIVE_LOGSUMEXP_TPL_SCH_MODE_FLOAT);
    } else {
        tilingKey = GET_TPL_TILING_KEY(CUMULATIVE_LOGSUMEXP_TPL_SCH_MODE_FLOAT16);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCumulativeLogsumexp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CumulativeLogsumexp)
    .Tiling(CumulativeLogsumexpTilingFunc)
    .TilingParse<CumulativeLogsumexpCompileInfo>(TilingParseForCumulativeLogsumexp)
    .TilingInputsDataDependency({1});
} // namespace optiling
