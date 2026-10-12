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
 * \file ones_like_tiling_arch35.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include "ones_like_tiling_arch35.h"
#include "op_host/tiling_base_util.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "op_host/util/fp16.h"
#include "log/log.h"
#include "math/ones_like/op_kernel/arch35/ones_like_dag.h"
#include "math/ones_like/op_kernel/arch35/ones_like_tiling_key.h"

namespace optiling {
const int64_t ASCEND_WORKSPACE = 16777216; // 16 * 1024 * 1024

ge::graphStatus OnesLikeTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "OnesLikeTiling SetTilingData enter.");
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = ONES_LIKE_TPL_FP16;
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = ONES_LIKE_TPL_BF16;
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = ONES_LIKE_TPL_FP32;
    } else if (this->outputDtype == ge::DT_INT8) {
        dType = ONES_LIKE_TPL_INT8;
    } else if (this->outputDtype == ge::DT_BOOL) {
        dType = ONES_LIKE_TPL_BOOL;
    } else if (this->outputDtype == ge::DT_INT32) {
        dType = ONES_LIKE_TPL_INT32;
    } else if (this->outputDtype == ge::DT_UINT8) {
        dType = ONES_LIKE_TPL_UINT8;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y(output)",
            Ops::Base::ToString(this->outputDtype),
            "Float16, BFloat16, Float, Int32, Bool, Int8 and Uint8");
        return ge::GRAPH_FAILED;
    }
    schMode = tiling->baseTiling.scheMode;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OnesLikeTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "OnesLikeTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    static const std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT8, ge::DT_FLOAT,
                                                          ge::DT_BOOL,    ge::DT_INT32, ge::DT_UINT8};
    auto inputTypeCheck = std::find(inputDtypes.begin(), inputDtypes.end(), this->inputDtype);

    OP_CHECK_IF((inputTypeCheck == inputDtypes.end()),
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "x(input)",
            Ops::Base::ToString(this->inputDtype),
            "Float16, BFloat16, Int8, Float, Bool, Int32 and Uint8"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OnesLikeTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "OnesLikeTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputYShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputZShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF((inputYShape.GetShapeSize() != outputZShape.GetShapeSize()),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x(input) and y(output)",
            Ops::Base::ToString(inputYShape) + ", " + Ops::Base::ToString(outputZShape),
            "The shapes of x(input) and y(output) must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OnesLikeTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "OnesLikeTiling CalcOutputDtype enter.");

    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF((this->outputDtype != this->inputDtype),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x(input) and y(output)",
            Ops::Base::ToString(this->inputDtype) + " and " + Ops::Base::ToString(this->outputDtype),
            "The dtypes of x(input) and y(output) must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OnesLikeTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "OnesLikeTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    tiling = tilingContext->GetTilingData<OnesLikeTilingData>();
    if (this->outputDtype == ge::DT_FLOAT16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<OnesLikeDAG<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<OnesLikeDAG<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        baseTilingResult = elewiseBaseTiling.DoTiling<OnesLikeDAG<float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT8 || this->outputDtype == ge::DT_BOOL) {
        baseTilingResult = elewiseBaseTiling.DoTiling<OnesLikeDAG<int8_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        baseTilingResult = elewiseBaseTiling.DoTiling<OnesLikeDAG<int32_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_UINT8) {
        baseTilingResult = elewiseBaseTiling.DoTiling<OnesLikeDAG<uint8_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y(output)",
            Ops::Base::ToString(this->outputDtype),
            "Float16, BFloat16, Int8, Float, Bool, Int32 and Uint8");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus Tiling4OnesLike(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4OnesLike rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    OnesLikeTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForOnesLike(gert::TilingParseContext* context)
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

IMPL_OP_OPTILING(OnesLike).Tiling(Tiling4OnesLike).TilingParse<ElewiseCompileInfo>(TilingPrepareForOnesLike);
} // namespace optiling