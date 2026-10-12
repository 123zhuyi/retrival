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
 * \file zeros_like_tiling_arch35.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include "zeros_like_tiling_arch35.h"
#include "op_host/tiling_base_util.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "op_host/util/fp16.h"
#include "log/log.h"
#include "conversion/zeros_like/op_kernel/arch35/zeros_like_dag.h"
#include "conversion/zeros_like/op_kernel/arch35/zeros_like_tiling_key.h"

namespace optiling {
static const size_t ASCEND_WORKSPACE = 0;

static constexpr size_t ZL_DTYPE_SIZE_0_5 = 0; // 字宽为0.5，如fp4
static constexpr size_t ZL_DTYPE_SIZE_1 = 1;
static constexpr size_t ZL_DTYPE_SIZE_2 = 2;
static constexpr size_t ZL_DTYPE_SIZE_4 = 3;
static constexpr size_t ZL_DTYPE_SIZE_8 = 4;

std::map<ge::DataType, size_t> ZLDtypeNormalize = {
    {ge::DT_FLOAT4_E1M2, ZL_DTYPE_SIZE_0_5}, {ge::DT_FLOAT4_E2M1, ZL_DTYPE_SIZE_0_5},
    {ge::DT_BOOL, ZL_DTYPE_SIZE_1},          {ge::DT_INT8, ZL_DTYPE_SIZE_1},
    {ge::DT_UINT8, ZL_DTYPE_SIZE_1},         {ge::DT_FLOAT8_E5M2, ZL_DTYPE_SIZE_1},
    {ge::DT_FLOAT8_E4M3FN, ZL_DTYPE_SIZE_1}, {ge::DT_HIFLOAT8, ZL_DTYPE_SIZE_1},
    {ge::DT_FLOAT16, ZL_DTYPE_SIZE_2},       {ge::DT_BF16, ZL_DTYPE_SIZE_2},
    {ge::DT_FLOAT, ZL_DTYPE_SIZE_4},         {ge::DT_INT32, ZL_DTYPE_SIZE_4},
    {ge::DT_INT64, ZL_DTYPE_SIZE_8},
};

ge::graphStatus ZerosLikeTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "ZerosLikeTiling SetTilingData enter.");
    schMode = tiling->baseTiling.scheMode;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = ASCEND_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ZerosLikeTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "ZerosLikeTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputXShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputYShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF((inputXShape.GetShapeSize() != outputYShape.GetShapeSize()),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x(input) and y(output)",
            Ops::Base::ToString(inputXShape) + ", " + Ops::Base::ToString(outputYShape),
            "The shapes of x(input) and y(output) must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ZerosLikeTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "ZerosLikeTiling CalcOutputDtype enter.");

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

ge::graphStatus ZerosLikeTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "ZerosLikeTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    tiling = tilingContext->GetTilingData<ZerosLikeTilingData>();

    auto it = ZLDtypeNormalize.find(this->outputDtype);
    if (it != ZLDtypeNormalize.end()) {
        switch (it->second) {
            case ZL_DTYPE_SIZE_1:
                baseTilingResult =
                    elewiseBaseTiling.DoTiling<ZerosLikeOp::ZerosLikeDAG<int8_t>::OpDag>(tiling->baseTiling);
                break;
            case ZL_DTYPE_SIZE_2:
                baseTilingResult =
                    elewiseBaseTiling.DoTiling<ZerosLikeOp::ZerosLikeDAG<int16_t>::OpDag>(tiling->baseTiling);
                break;
            case ZL_DTYPE_SIZE_4:
                baseTilingResult =
                    elewiseBaseTiling.DoTiling<ZerosLikeOp::ZerosLikeDAG<int32_t>::OpDag>(tiling->baseTiling);
                break;
            case ZL_DTYPE_SIZE_8:
                baseTilingResult =
                    elewiseBaseTiling.DoTiling<ZerosLikeOp::ZerosLikeDAG<int64_t>::OpDag>(tiling->baseTiling);
                break;
            default: // ZL_DTYPE_SIZE_0_5
                baseTilingResult =
                    elewiseBaseTiling.DoTiling4Bits<ZerosLikeOp::ZerosLikeDAG<int8_t>::OpDag>(tiling->baseTiling);
                break;
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y(output)",
            Ops::Base::ToString(this->outputDtype),
            "Float4_E1M2, Float4_E2M1, Bool, Int8, Uint8, Float8_E5M2, Float8_E4M3FN, Hifloat8, Float16, BFloat16, Float, Int32 and Int64");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus Tiling4ZerosLike(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4ZerosLike rt2.0 is running.");
    auto compileInfo = static_cast<const ElewiseCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    ZerosLikeTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

ge::graphStatus TilingPrepareForZerosLike(gert::TilingParseContext* context)
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

IMPL_OP_OPTILING(ZerosLike).Tiling(Tiling4ZerosLike).TilingParse<ElewiseCompileInfo>(TilingPrepareForZerosLike);
} // namespace optiling