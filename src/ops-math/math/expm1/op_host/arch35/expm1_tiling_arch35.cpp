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
 * \file expm1_tiling_arch35.cpp
 * \brief
 */
#include "expm1_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_info.h"
#include "op_host/tiling_base_util.h"
#include "register/tilingdata_base.h"
#include "math/expm1/op_kernel/arch35/expm1_dag.h"
#include "math/expm1/op_kernel/arch35/expm1_struct.h"

#include <iostream>


namespace optiling {
const int64_t ASCEND_WORKSPACE = 16777216; // 16M
const int64_t ASCEND_API_BUFFER = 122880;  // 120K
const int64_t DCACHE_SIZE = 32768;

ge::graphStatus Expm1Tiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "Expm1Tiling SetTilingData enter.");

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = ASCEND_WORKSPACE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(tiling_->scheMode), dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling_->blockNum);

    uint64_t ubSize = 0;
    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const ElewiseCompileInfo*>(tilingContext->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(tilingContext, "compile info is null"), return ge::GRAPH_FAILED);
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatForm = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize = ubSizePlatForm;
    }
    tilingContext->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Expm1Tiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "Expm1Tiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype),
            "FLOAT16, BF16, FLOAT"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Expm1Tiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "Expm1Tiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputXShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputXShape != outputYShape, OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x, y", (Ops::Base::ToString(inputXShape) + ", " + Ops::Base::ToString(outputYShape)).c_str(), "The shapes of x and y must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Expm1Tiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "Expm1Tiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x, y",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype) + ", " + ge::TypeUtils::DataTypeToSerialString(this->inputDtype),
            "The dtypes of x and y must be the same"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Expm1Tiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "Expm1Tiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    tiling_ = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult =
            elewiseBaseTiling.DoTiling<Expm1Op::Expm1DAG<half>::OpDag>(*tiling_, ASCEND_API_BUFFER + DCACHE_SIZE);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult =
            elewiseBaseTiling.DoTiling<Expm1Op::Expm1DAG<bfloat16_t>::OpDag>(*tiling_, ASCEND_API_BUFFER + DCACHE_SIZE);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult =
            elewiseBaseTiling.DoTiling<Expm1Op::Expm1DAG<float>::OpDag>(*tiling_, ASCEND_API_BUFFER + DCACHE_SIZE);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
            "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus Tiling4Expm1(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Expm1 rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    Expm1Tiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForExpm1([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Expm1).Tiling(Tiling4Expm1).TilingParse<ElewiseCompileInfo>(TilingPrepareForExpm1);
} // namespace optiling