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
 * \file rsqrt_regbase_optiling.cc
 * \brief
 */

#include "rsqrt_tiling_arch35.h"
#include "op_host/tiling_base_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "math/rsqrt/op_kernel/arch35/rsqrt.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"

#include <iostream>


namespace optiling
{

const int64_t ASCEND_WORKSPACE = static_cast<int64_t>(16) * 1024 * 1024;
const uint64_t RSQRT_KEY_FP16 = 101UL;
const uint64_t RSQRT_KEY_BF16 = 102UL;
const uint64_t RSQRT_KEY_FP32 = 103UL; 

ge::graphStatus RsqrtTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "RsqrtTiling SetTilingData enter.");
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingContext->GetRawTilingData());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(ASCEND_WORKSPACE);
    uint64_t tilingKey = RSQRT_KEY_FP16;
    if (this->outputDtype == ge::DT_FLOAT16) {
        tilingKey = RSQRT_KEY_FP16;
    } else if (this->outputDtype == ge::DT_BF16) {
        tilingKey = RSQRT_KEY_BF16;
    } else if (this->outputDtype == ge::DT_FLOAT) {
        tilingKey = RSQRT_KEY_FP32;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling_->blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RsqrtTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(this->inputDtype), "FLOAT16, BF16, FLOAT"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RsqrtTiling::CheckShape()
{
    auto xStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xStorageShape);
    const gert::Shape& inputXShape = Ops::Base::EnsureNotScalar(xStorageShape->GetStorageShape());

    auto yStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yStorageShape);
    const gert::Shape& outputYShape = Ops::Base::EnsureNotScalar(yStorageShape->GetStorageShape());

    OP_CHECK_IF(inputXShape != outputYShape,
               OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x, y", (Ops::Base::ToString(inputXShape) + ", " + Ops::Base::ToString(outputYShape)).c_str(), "The shapes of x and y must be the same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RsqrtTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_FLOAT16 && this->outputDtype != ge::DT_BF16 && this->outputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(this->outputDtype != this->inputDtype,
               OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x, y", std::string(ge::TypeUtils::DataTypeToSerialString(this->outputDtype)) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(this->inputDtype)), "The dtypes of x and y must be the same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RsqrtTiling::RunTiling()
{
    OP_CHECK_IF(tilingContext == nullptr,
               OP_LOGE("RunTiling", "Tiling context is null"),
               return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext->GetNodeName(), "RsqrtTiling RunTiling enter.");
    Ops::Base::ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    tiling_ = tilingContext->GetTilingData<EleBaseTilingDataV2>();
    OP_CHECK_IF(CalcInputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "check shape failed"),
               return ge::GRAPH_FAILED);
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<RsqrtDag::RsqrtOp<half>::OpDag>(*tiling_);
    } else if (this->outputDtype == ge::DT_BF16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<RsqrtDag::RsqrtOp<half>::OpDag>(*tiling_);  // bfloat16类型host没定义
    } else if (this->outputDtype == ge::DT_FLOAT) {
        baseTilingResult = elewiseBaseTiling.DoTiling<RsqrtDag::RsqrtOp<float>::OpDag>(*tiling_);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed"), return ge::GRAPH_FAILED);
    baseTilingResult = SetTilingData();
    return baseTilingResult;
}

static ge::graphStatus TilingPrepareForRsqrt(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<RsqrtCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForRsqrt(gert::TilingContext* tilingContextGen)
{
    OP_CHECK_IF(tilingContextGen == nullptr,
        OP_LOGE("TilingForRsqrt", "Tiling context is null"),
        return ge::GRAPH_FAILED);
    OP_LOGD(tilingContextGen->GetNodeName(), "TilingForRsqrt is running.");
    auto compileInfo = tilingContextGen->GetCompileInfo<RsqrtCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    // 走新的模板tiling
    OP_LOGD("RsqrtTiling", "Enter new RsqrtTiling");
    RsqrtTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

IMPL_OP_OPTILING(Rsqrt).Tiling(TilingForRsqrt)
    .TilingParse<RsqrtCompileInfo>(TilingPrepareForRsqrt);
}  // namespace optiling