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
 * \file exp_regbase_optiling.cc
 * \brief
 */
#include <graph/utils/type_utils.h>
#include "exp_tiling_arch35.h"
#include "op_host/tiling_base_util.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "op_host/util/fp16.h"
#include "log/log.h"
#include "math/exp/op_kernel/arch35/exp_dag.h"
#include "math/exp/op_kernel/arch35/exp_struct.h"

namespace optiling {
using namespace ExpOp;
const size_t ASCEND_WORKSPACE = 16777216; // 16 * 1024 * 1024

ge::graphStatus ExpTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ExpTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(this->inputDtype), "FLOAT16, BF16, FLOAT"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "ExpTiling CheckShape enter.");
    auto expInputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, expInputStorageShape);
    const gert::Shape& expInputYShape = Ops::Base::EnsureNotScalar(expInputStorageShape->GetStorageShape());

    auto expOutputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, expOutputStorageShape);
    const gert::Shape& expOutputZShape = Ops::Base::EnsureNotScalar(expOutputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        expInputYShape != expOutputZShape, OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x, y", (Ops::Base::ToString(expInputYShape) + ", " + Ops::Base::ToString(expOutputZShape)).c_str(), "The shapes of x and y must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ExpTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x, y", std::string(ge::TypeUtils::DataTypeToSerialString(this->outputDtype)) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(this->inputDtype)), "The dtypes of x and y must be the same"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpTiling::SetAttr()
{
    OP_LOGD(tilingContext->GetNodeName(), "ExpTiling GetAttrs enter.");
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const float* baseValueAttr = attrs->GetAttrPointer<float>(ExpDag::PLACEHOLDER_INDEX_0);
    const float* scaleValueAttr = attrs->GetAttrPointer<float>(ExpDag::PLACEHOLDER_INDEX_1);
    const float* shiftValueAttr = attrs->GetAttrPointer<float>(ExpDag::PLACEHOLDER_INDEX_2);
    float baseValue = baseValueAttr == nullptr ? -1.0f : *baseValueAttr;
    bool isBaseValue = std::fabs(baseValue - (-1.0f)) <=
                       std::numeric_limits<float>::epsilon() * std::max(std::fabs(baseValue), std::fabs(-1.0f));
    OP_CHECK_IF(
        baseValue <= 1e-6f && !isBaseValue,
        OP_LOGE_FOR_INVALID_VALUE(tilingContext->GetNodeName(), "base_value", std::to_string(baseValue), "greater than 0 or equal to -1"), return ge::GRAPH_FAILED);
    float lnBase = isBaseValue ? 1.0f : log(baseValue);
    float scale = scaleValueAttr == nullptr ? 1.0f : *scaleValueAttr;
    float shift = shiftValueAttr == nullptr ? 0.0f : *shiftValueAttr;

    attrScale = scale;
    attrShift = shift;
    attrlnBase = lnBase;
    bool isScale =
        std::fabs(scale - 1.0f) <= std::numeric_limits<float>::epsilon() * std::max(std::fabs(scale), std::fabs(1.0f));
    bool isShift =
        std::fabs(shift - 0.0f) <= std::numeric_limits<float>::epsilon() * std::max(std::fabs(shift), std::fabs(0.0f));
    if (isScale && isShift && std::abs(lnBase - 1.0f) < std::numeric_limits<float>::epsilon()) {
        attrWork = static_cast<uint64_t>(TPL_SCALE_IS_ONE_SHIFT_IS_ZERO_LNBASE_IS_ONE);
    } else {
        attrWork = static_cast<uint64_t>(TPL_SCALE_NOT_ONE_SHIFT_NOT_ZERO_LNBASE_NOT_ONE);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExpTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "ExpTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);

    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "check shape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetAttr() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "set Attr failed"), return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        if (attrWork == static_cast<uint64_t>(TPL_SCALE_IS_ONE_SHIFT_IS_ZERO_LNBASE_IS_ONE)) {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<ExpDag::ExpScaleOneShiftZeroLnbaseOne<half>::OpDag>();
        } else {
            baseTilingResult =
                elewiseBaseTiling.DoTiling32B<ExpDag::ExpScaleNotOneShiftNotZeroLnbaseNotOne<half>::OpDag>();
        }
    } else if (this->outputDtype == ge::DT_BF16) {
        if (attrWork == static_cast<uint64_t>(TPL_SCALE_IS_ONE_SHIFT_IS_ZERO_LNBASE_IS_ONE)) {
            baseTilingResult =
                elewiseBaseTiling.DoTiling32B<ExpDag::ExpScaleOneShiftZeroLnbaseOne<bfloat16_t>::OpDag>();
        } else {
            baseTilingResult =
                elewiseBaseTiling.DoTiling32B<ExpDag::ExpScaleNotOneShiftNotZeroLnbaseNotOne<bfloat16_t>::OpDag>();
        }
    } else if (this->outputDtype == ge::DT_FLOAT) {
        if (attrWork == static_cast<uint64_t>(TPL_SCALE_IS_ONE_SHIFT_IS_ZERO_LNBASE_IS_ONE)) {
            baseTilingResult = elewiseBaseTiling.DoTiling32B<ExpDag::ExpScaleOneShiftZeroLnbaseOne<float>::OpDag>();
        } else {
            baseTilingResult =
                elewiseBaseTiling.DoTiling32B<ExpDag::ExpScaleNotOneShiftNotZeroLnbaseNotOne<float>::OpDag>();
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    elewiseBaseTiling.SetScalar<float>(attrScale);
    elewiseBaseTiling.SetScalar<float>(attrShift);
    elewiseBaseTiling.SetScalar<float>(attrlnBase);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, attrWork);

    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(elewiseBaseTiling.GetBlockDim());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Exp(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Exp rt2.0 is running.");
    auto compileInfo = tilingContextGen->GetCompileInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    ExpTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForExp(gert::TilingParseContext* context)
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

IMPL_OP_OPTILING(Exp).Tiling(Tiling4Exp).TilingParse<ElewiseCompileInfo>(TilingPrepareForExp);
} // namespace optiling
