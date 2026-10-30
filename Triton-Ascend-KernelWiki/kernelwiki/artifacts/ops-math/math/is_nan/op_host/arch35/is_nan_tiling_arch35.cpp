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
 * \file is_nan_tiling_arch35.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include "is_nan_tiling_arch35.h"
#include "op_host/tiling_base_util.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "op_host/util/fp16.h"
#include "log/log.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "math/is_nan/op_kernel/arch35/is_nan_dag.h"

namespace optiling {
const int64_t ASCEND_WORKSPACE = 16777216; // 16M
const uint64_t TILING_KEY_FP16 = 101UL;
const uint64_t TILING_KEY_BF16 = 102UL;
const uint64_t TILING_KEY_FP32 = 103UL;

ge::graphStatus IsNanTiling::CalcInputDtype()
{
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

ge::graphStatus IsNanTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != ge::DT_BOOL,
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y",
            ge::TypeUtils::DataTypeToSerialString(this->outputDtype),
            "BOOL"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsNanTiling::CheckShape()
{
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

ge::graphStatus IsNanTiling::SetTilingData()
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = static_cast<uint64_t>(ASCEND_WORKSPACE);

    if (this->inputDtype == ge::DT_FLOAT16) {
        tilingContext->SetTilingKey(TILING_KEY_FP16);
    } else if (this->inputDtype == ge::DT_BF16) {
        tilingContext->SetTilingKey(TILING_KEY_BF16);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        tilingContext->SetTilingKey(TILING_KEY_FP32);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            tilingContext->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype),
            "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }

    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsNanTiling::RunTiling()
{
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(),  "check shape failed"), return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    tiling = tilingContext->GetTilingData<IsNanTilingData>();
    if (this->inputDtype == ge::DT_FLOAT16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<IsNanOp::IsNanDAG<half>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_BF16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<IsNanOp::IsNanDAG<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if (this->inputDtype == ge::DT_FLOAT) {
        baseTilingResult = elewiseBaseTiling.DoTiling<IsNanOp::IsNanDAG<float>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            tilingContext->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(this->inputDtype),
            "FLOAT16, BF16, FLOAT");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus TilingPrepare4IsNan([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4IsNanForAscendC(gert::TilingContext* context)
{
    OP_LOGD("IsNanTiling", "Enter new IsNanTiling");

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (static_cast<int32_t>(coreNum) <= 0), OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "core_num", std::to_string(coreNum), "greater than 0"),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        (static_cast<int64_t>(ubSize) <= 0), OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "ub_size", std::to_string(ubSize), "greater than 0"),
        return ge::GRAPH_FAILED);

    IsNanTiling IsNanTiling(context);
    return IsNanTiling.RunTiling();
}

static ge::graphStatus Tiling4IsNan(gert::TilingContext* context)
{
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    auto compile_info = context->GetCompileInfo<IsNanCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    return Tiling4IsNanForAscendC(context);
}
IMPL_OP_OPTILING(IsNan).Tiling(Tiling4IsNan).TilingParse<IsNanCompileInfo>(TilingPrepare4IsNan);
} // namespace optiling