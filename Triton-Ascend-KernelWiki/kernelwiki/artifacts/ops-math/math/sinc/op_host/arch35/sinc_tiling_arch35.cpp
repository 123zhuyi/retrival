/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sinc_tiling_arch35.cpp
 * \brief
 */
#include "sinc_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/sinc_dag.h"
#include "../../op_kernel/arch35/sinc_struct.h"
#include "log/log.h"
#include "op_host/tiling_base_util.h"

#include <iostream>

using namespace Ops::Base;
namespace optiling {
const int64_t ASCEND_WORKSPACE = 16777216; // 16M
const int64_t ASCEND_API_BUFFER = 122880;  // 120K
const int64_t DCACHE_SIZE = 32768;         // 32K

ge::graphStatus SincTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "SincTiling SetTilingData enter.");

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(ASCEND_WORKSPACE);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(tiling->baseTiling.scheMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);

    uint64_t ubSize = 0;
    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = tilingContext->GetCompileInfo<SincCompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfoPtr);
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

ge::graphStatus SincTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "SincTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            tilingContext->GetNodeName(), "x", ToString(this->inputDtype).c_str(),
            "The dtype of x must be within the range DT_FLOAT16, DT_BF16 and DT_FLOAT"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SincTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "SincTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputYShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputZShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(inputYShape != outputZShape,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext->GetNodeName(), "y", ToString(outputZShape).c_str(),
                                                      "The shape of y must be equal to the shape of x"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SincTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "SincTiling CalcOutputDtype enter.");
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    if (this->outputDtype != this->inputDtype) {
        std::string errorMsg = "The dtype of y must be the same as " + ToString(this->inputDtype) + " of x";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "y", ToString(this->outputDtype).c_str(),
                                              errorMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SincTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "SincTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    tiling = tilingContext->GetTilingData<SincNs::SincTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    if (CalcInputDtype() == ge::GRAPH_FAILED || CalcOutputDtype() == ge::GRAPH_FAILED ||
        CheckShape() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = TPL_FP16;
        baseTilingResult = elewiseBaseTiling.DoTiling<SincOp::SincDAG<Ops::Base::half>::OpDag>(
            tiling->baseTiling, ASCEND_API_BUFFER + DCACHE_SIZE);
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = TPL_BF16;
        baseTilingResult = elewiseBaseTiling.DoTiling<SincOp::SincDAG<bfloat16_t>::OpDag>(
            tiling->baseTiling, ASCEND_API_BUFFER + DCACHE_SIZE);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = TPL_FP32;
        baseTilingResult = elewiseBaseTiling.DoTiling<SincOp::SincDAG<float>::OpDag>(tiling->baseTiling,
                                                                                     ASCEND_API_BUFFER + DCACHE_SIZE);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            tilingContext->GetNodeName(), "y", ToString(this->outputDtype).c_str(),
            "The dtype of y must be within the range DT_FLOAT16, DT_BF16 and DT_FLOAT");
        return ge::GRAPH_FAILED;
    }

    if (baseTilingResult == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    return SetTilingData();
}

static ge::graphStatus Tiling4Sinc(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Sinc rt2.0 is running.");
    auto compileInfo = tilingContextGen->GetCompileInfo<SincCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);

    SincTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForSinc(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<SincCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Sinc).Tiling(Tiling4Sinc).TilingParse<SincCompileInfo>(TilingPrepareForSinc);
} // namespace optiling
