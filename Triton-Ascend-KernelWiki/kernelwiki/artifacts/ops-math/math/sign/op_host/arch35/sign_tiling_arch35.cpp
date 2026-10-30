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
 * \file sign_tiling_arch35.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include "sign_tiling_arch35.h"
#include "op_host/tiling_base_util.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "op_host/util/fp16.h"
#include "log/log.h"
#include "math/sign/op_kernel/arch35/sign.h"

namespace optiling {
const uint64_t SIGN_KEY_UNDEFINED = 100UL;
const uint64_t SIGN_KEY_FP16 = 101UL;
const uint64_t SIGN_KEY_BF16 = 102UL;
const uint64_t SIGN_KEY_FP32 = 103UL;
const uint64_t SIGN_KEY_INT32 = 104UL;
const uint64_t SIGN_KEY_INT64 = 105UL;
const int64_t ASCEND_WORKSPACE = 16 * 1024 * 1024;

ge::graphStatus SignTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SignTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "SignTiling SetTilingData enter.");
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingContext->GetRawTilingData());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    uint64_t tilingKey = SIGN_KEY_UNDEFINED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        tilingKey = SIGN_KEY_FP16;
    } else if (this->outputDtype == ge::DT_BF16) {
        tilingKey = SIGN_KEY_BF16;
    } else if (this->outputDtype == ge::DT_FLOAT) {
        tilingKey = SIGN_KEY_FP32;
    } else if (this->outputDtype == ge::DT_INT32) {
        tilingKey = SIGN_KEY_INT32;
    } else if (this->outputDtype == ge::DT_INT64) {
        tilingKey = SIGN_KEY_INT64;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT, INT32, INT64");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SignTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SignTiling::CheckOutputShape()
{
    auto xStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, xStorageShape);
    auto yStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, yStorageShape);
    // get storage shape
    const gert::Shape& inputShape = Ops::Base::EnsureNotScalar(xStorageShape->GetStorageShape());
    const gert::Shape& outputShape = Ops::Base::EnsureNotScalar(yStorageShape->GetStorageShape());

    OP_CHECK_IF(
        (inputShape != outputShape),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            tilingContext->GetNodeName(), "x, y",
            (Ops::Base::ToString(inputShape) + ", " + Ops::Base::ToString(outputShape)).c_str(),
            "The shapes of x and y must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SignTiling::CheckOutputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    OP_CHECK_IF(
        this->outputDtype != inputDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            tilingContext->GetNodeName(), "x, y",
            std::string(ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType())) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(this->outputDtype)),
            "The dtypes of x and y must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SignTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "SignTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    tiling = tilingContext->GetTilingData<SignTilingData>();
    // 获取tiling计算所需参数
    ge::graphStatus baseTilingResult = CalcOutputDtype();
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckOutputShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "check shape failed"), return ge::GRAPH_FAILED);
    if (this->outputDtype == ge::DT_FLOAT16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<SignDag::SignForNumber<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        baseTilingResult = elewiseBaseTiling.DoTiling<SignDag::SignForNumber<float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<SignDag::SignForBf<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        baseTilingResult = elewiseBaseTiling.DoTiling<SignDag::SignForNumber<int32_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT64) {
        baseTilingResult = elewiseBaseTiling.DoTiling<SignDag::SignForInt64<int64_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(this->outputDtype), "FLOAT16, BF16, FLOAT, INT32, INT64");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);
    baseTilingResult = SetTilingData();
    return baseTilingResult;
}

static ge::graphStatus TilingForSign(gert::TilingContext* tilingContextGen)
{
    OP_CHECK_IF(
        tilingContextGen == nullptr, OP_LOGE("TilingForSign", "Tiling context is null"), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContextGen->GetNodeName(), "TilingForSign is running.");
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    OP_LOGD("SignTiling", "Enter new SignTiling");
    SignTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForSign(gert::TilingParseContext* context)
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

IMPL_OP_OPTILING(Sign).Tiling(TilingForSign).TilingParse<ElewiseCompileInfo>(TilingPrepareForSign);
} // namespace optiling
