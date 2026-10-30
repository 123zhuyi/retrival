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
 * \file sqrt_grad_tiling_arch35.cpp
 * \brief
 */

#include "log/log.h"
#include "sqrt_grad_tiling_arch35.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "math/sqrt_grad/op_kernel/arch35/sqrt_grad_dag.h"
#include "math/sqrt_grad/op_kernel/arch35/sqrt_grad_struct.h"
#include "op_host/tiling_base_util.h"
#include "math/sqrt_grad/op_kernel/arch35/sqrt_grad_tiling_struct.h"
#include <iostream>

using namespace SqrtGradNs;
namespace optiling {
const int64_t ASCEND_WORKSPACE = 16 * 1024 * 1024;
const uint64_t TILING_KEY_FP16 = 101UL;
const uint64_t TILING_KEY_BF16 = 102UL;
const uint64_t TILING_KEY_FP32 = 103UL;

class SqrtGradTiling {
public:
    explicit SqrtGradTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();
    void SetTilingData();

private:
    SqrtGradTilingData* tiling = nullptr;
    gert::TilingContext* tilingContext = nullptr;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    ge::DataType inputDtype = ge::DT_UNDEFINED;
    ge::DataType inputDtype1 = ge::DT_UNDEFINED;
    uint64_t dType = 0;
};

void SqrtGradTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "SqrtGradTiling SetTilingData enter.");

    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(tiling->basetiling.scheMode), dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->basetiling.blockNum);
}

ge::graphStatus SqrtGradTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(this->inputDtype), "FLOAT16, BF16, FLOAT"), return ge::GRAPH_FAILED);
    auto inputDesc1 = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc1);
    this->inputDtype1 = inputDesc1->GetDataType();
    OP_CHECK_IF(
        this->inputDtype1 != ge::DT_FLOAT16 && this->inputDtype1 != ge::DT_BF16 && this->inputDtype1 != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "dy", ge::TypeUtils::DataTypeToSerialString(this->inputDtype1), "FLOAT16, BF16, FLOAT"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        this->inputDtype1 != this->inputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "y, dy", std::string(ge::TypeUtils::DataTypeToSerialString(this->inputDtype)) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(this->inputDtype1)), "The dtypes of y and dy must be the same"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SqrtGradTiling::CheckShape()
{
    auto yStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yStorageShape);
    const gert::Shape& inputYShape = Ops::Base::EnsureNotScalar(yStorageShape->GetStorageShape());
    auto dyStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, dyStorageShape);
    const gert::Shape& inputDYShape = Ops::Base::EnsureNotScalar(dyStorageShape->GetStorageShape());

    auto zStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, zStorageShape);
    const gert::Shape& outputZShape = Ops::Base::EnsureNotScalar(zStorageShape->GetStorageShape());

    OP_CHECK_IF(
        inputYShape != inputDYShape, OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "y, dy", (Ops::Base::ToString(inputYShape) + ", " + Ops::Base::ToString(inputDYShape)).c_str(), "The shapes of y and dy must be the same"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputYShape != outputZShape, OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "y, z", (Ops::Base::ToString(inputYShape) + ", " + Ops::Base::ToString(outputZShape)).c_str(), "The shapes of y and z must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SqrtGradTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "z", ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "y, z", std::string(ge::TypeUtils::DataTypeToSerialString(this->inputDtype)) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(this->outputDtype)), "The dtypes of y and z must be the same"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SqrtGradTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "SqrtGradTiling RunTiling enter.");
    Ops::Base::ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    tiling = tilingContext->GetTilingData<SqrtGradTilingData>();
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "check shape failed"), return ge::GRAPH_FAILED);
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult =
            elewiseBaseTiling.DoTiling<SqrtGradDag::SqrtGradCustom<Ops::Base::half>::OpDag>(tiling->basetiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult =
            elewiseBaseTiling.DoTiling<SqrtGradDag::SqrtGradCustom<Ops::Base::bfloat16_t>::OpDag>(tiling->basetiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<SqrtGradDag::SqrtGradCustom<float>::OpDag>(tiling->basetiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "z", ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);
    SetTilingData();

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4SqrtGrad(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4SqrtGrad rt2.0 is running.");
    auto compileInfo = tilingContextGen->GetCompileInfo<SqrtGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    SqrtGradTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForSqrtGrad(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<SqrtGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SqrtGrad).Tiling(Tiling4SqrtGrad).TilingParse<SqrtGradCompileInfo>(TilingPrepareForSqrtGrad);
} // namespace optiling
