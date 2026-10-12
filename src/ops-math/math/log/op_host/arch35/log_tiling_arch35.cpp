/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file log_tiling_arch35.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include "log_tiling_arch35.h"
#include "op_host/tiling_base_util.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "math/log/op_kernel/arch35/log_dag.h"
#include "math/log/op_kernel/arch35/log_struct.h"
#include <cmath>

namespace optiling {
using namespace LogOp;
const size_t ASCEND_WORKSPACE = 0;

ge::graphStatus LogTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "x",
                                  ge::TypeUtils::DataTypeToSerialString(this->inputDtype), "FLOAT16, BF16, FLOAT"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogTiling::CheckShape() const
{
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputYShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputZShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(inputYShape != outputZShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    tilingContext->GetNodeName(), "x, y",
                    (Ops::Base::ToString(inputYShape) + ", " + Ops::Base::ToString(outputZShape)).c_str(),
                    "The shapes of x and y must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y",
                                  ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    tilingContext->GetNodeName(), "x, y",
                    std::string(ge::TypeUtils::DataTypeToSerialString(this->outputDtype)) + ", " +
                        std::string(ge::TypeUtils::DataTypeToSerialString(this->inputDtype)),
                    "The dtypes of x and y must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline bool NearlyEqual(float a, float b, float epsilon = std::numeric_limits<float>::epsilon() * 100)
{
    float diff = std::fabs(a - b);
    if (diff < epsilon) {
        return true;
    }
    float maxVal = std::max(std::fabs(a), std::fabs(b));
    if (maxVal < std::numeric_limits<float>::min()) {
        return diff < epsilon;
    }

    return diff < maxVal * epsilon;
}

ge::graphStatus LogTiling::SetAttr()
{
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const float* baseValueAttr = attrs->GetAttrPointer<float>(0);
    const float* scaleValueAttr = attrs->GetAttrPointer<float>(1);
    const float* shiftValueAttr = attrs->GetAttrPointer<float>(LogDag::PLACEHOLDER_INDEX_2);
    float baseValue = baseValueAttr == nullptr ? -1.0f : *baseValueAttr;
    OP_CHECK_IF(
        baseValue <= 0.0f && !NearlyEqual(baseValue, -1.0f),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "base_value", std::to_string(baseValue),
                                              "The value of base_value must be greater than 0 or equal to -1"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        NearlyEqual(baseValue, 1.0f),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "base_value", std::to_string(baseValue),
                                              "The value of base_value must not be 1"),
        return ge::GRAPH_FAILED);
    float invLnBase = NearlyEqual(baseValue, -1.0f) ? 1.0f : 1.0f / log(baseValue);
    float scale = scaleValueAttr == nullptr ? 1.0f : *scaleValueAttr;
    float shift = shiftValueAttr == nullptr ? 0.0f : *shiftValueAttr;
    attrInvLnBase = invLnBase;
    attrScale = scale;
    attrShift = shift;
    if (NearlyEqual(scale, 1.0f) && NearlyEqual(shift, 0.0f) &&
        std::abs(invLnBase - 1.0f) < std::numeric_limits<float>::epsilon()) {
        attrWork = static_cast<uint64_t>(TPL_SCALE_IS_ONE_SHIFT_IS_ZERO_LNBASE_IS_ONE);
    } else {
        attrWork = static_cast<uint64_t>(TPL_SCALE_NOT_ONE_SHIFT_NOT_ZERO_LNBASE_NOT_ONE);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "check shape failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetAttr() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "set Attr failed"),
                return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        if (attrWork == static_cast<uint64_t>(TPL_SCALE_IS_ONE_SHIFT_IS_ZERO_LNBASE_IS_ONE)) {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<LogDag::LogScaleOneShiftZeroLnbaseOne<half>::OpDag>();
        } else {
            baseTilingResult = elewiseBaseTiling
                                   .DoTiling32B<LogDag::LogScaleNotOneShiftNotZeroLnbaseNotOne<half>::OpDag>();
        }
    } else if (this->outputDtype == ge::DT_BF16) {
        if (attrWork == static_cast<uint64_t>(TPL_SCALE_IS_ONE_SHIFT_IS_ZERO_LNBASE_IS_ONE)) {
            baseTilingResult = elewiseBaseTiling
                                   .DoTiling32B<LogDag::LogScaleOneShiftZeroLnbaseOne<bfloat16_t>::OpDag>();
        } else {
            baseTilingResult = elewiseBaseTiling
                                   .DoTiling32B<LogDag::LogScaleNotOneShiftNotZeroLnbaseNotOne<bfloat16_t>::OpDag>();
        }
    } else if (this->outputDtype == ge::DT_FLOAT) {
        if (attrWork == static_cast<uint64_t>(TPL_SCALE_IS_ONE_SHIFT_IS_ZERO_LNBASE_IS_ONE)) {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<LogDag::LogScaleOneShiftZeroLnbaseOne<float>::OpDag>();
        } else {
            baseTilingResult = elewiseBaseTiling
                                   .DoTiling32B<LogDag::LogScaleNotOneShiftNotZeroLnbaseNotOne<float>::OpDag>();
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y",
                                  ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
                OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed, output dtype: %s.",
                        ge::TypeUtils::DataTypeToSerialString(this->outputDtype).c_str()),
                return ge::GRAPH_FAILED);

    elewiseBaseTiling.SetScalar<float>(attrScale);
    elewiseBaseTiling.SetScalar<float>(attrShift);
    elewiseBaseTiling.SetScalar<float>(attrInvLnBase);
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(1, attrWork);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(elewiseBaseTiling.GetBlockDim());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Log(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Log rt2.0 is running.");
    auto compileInfo = tilingContextGen->GetCompileInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    LogTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForLog(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Log).Tiling(Tiling4Log).TilingParse<ElewiseCompileInfo>(TilingPrepareForLog);
} // namespace optiling
