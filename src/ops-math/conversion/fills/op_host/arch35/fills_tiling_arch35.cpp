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
 * \file fills_tiling.cc
 * \brief
 */

#include "fills_tiling_arch35.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "util/fp16.h"
#include "util/bfloat16.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "conversion/fills/op_kernel/arch35/fills_dag.h"
#include "conversion/fills/op_kernel/arch35/fills_tiling_key.h"
#include "op_host/tiling_base_util.h"

namespace optiling {
const int64_t ASCEND_WORKSPACE = 16777216; // 16 * 1024 * 1024

union FillsValue {
    uint8_t vU8;
    int8_t vI8;
    int32_t vI32;
    int64_t vI64;
    float vF32;
    Ops::Base::fp16_t vF16;
    Ops::Base::bfloat16 vBf16;

    explicit FillsValue(uint8_t u8) : vU8(u8)
    {}
    explicit FillsValue(int8_t i8) : vI8(i8)
    {}
    explicit FillsValue(int32_t i32) : vI32(i32)
    {}
    explicit FillsValue(int64_t i64) : vI64(i64)
    {}
    explicit FillsValue(float f32) : vF32(f32)
    {}
    explicit FillsValue(Ops::Base::fp16_t f16) : vF16(f16)
    {}
    explicit FillsValue(Ops::Base::bfloat16 bf16) : vBf16(bf16)
    {}
};

ge::graphStatus FillsTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "FillsTiling SetTilingData enter.");

    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = FILLS_TPL_FP16;
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = FILLS_TPL_BF16;
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = FILLS_TPL_FP32;
    } else if (this->outputDtype == ge::DT_INT32) {
        dType = FILLS_TPL_INT32;
    } else if (this->outputDtype == ge::DT_INT64) {
        dType = FILLS_TPL_INT64;
    } else if (this->outputDtype == ge::DT_INT8) {
        dType = FILLS_TPL_INT8;
    } else if (this->outputDtype == ge::DT_UINT8) {
        dType = FILLS_TPL_UINT8;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y(output)",
            Ops::Base::ToString(this->outputDtype),
            "Float16, BFloat16, Float, Int32, Int64, Int8 and Uint8");
        return ge::GRAPH_FAILED;
    }
    schMode = tiling->baseTiling.scheMode;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);

    size_t usrWorkspaceSize = 0;
    size_t sysWorkspaceSize = ASCEND_WORKSPACE;
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillsTiling::CalcOutputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "FillsTiling CalcOutputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF((this->inputDtype != this->outputDtype),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x(input) and y(output)",
            Ops::Base::ToString(this->inputDtype) + " and " + Ops::Base::ToString(this->outputDtype),
            "The dtypes of x(input) and y(output) must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillsTiling::CalcInputDtype()
{
    OP_LOGD(tilingContext->GetNodeName(), "FillsTiling CalcInputDtype enter.");
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF((this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_FLOAT && this->inputDtype != ge::DT_INT8 &&
            this->inputDtype != ge::DT_UINT8 && this->inputDtype != ge::DT_INT32 && this->inputDtype != ge::DT_INT64 &&
            this->inputDtype != ge::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "x(input)",
            Ops::Base::ToString(this->inputDtype),
            "Float16, Float, Int8, Uint8, Int32, Int64 and BFloat16"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillsTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "FillsTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputYShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputZShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF((inputYShape != outputZShape),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x(input) and y(output)",
            Ops::Base::ToString(inputYShape) + ", " + Ops::Base::ToString(outputZShape),
            "The shapes of x(input) and y(output) must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillsTiling::SetAttr()
{
    OP_LOGD(tilingContext->GetNodeName(), "FillsTiling SetAttr enter.");
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const float* valueAttr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, valueAttr);
    float value = *valueAttr;
    FillsValue v(0);
    switch (this->outputDtype) {
        case ge::DT_FLOAT16:
            v.vF16 = Ops::Base::fp16_t(value);
            break;
        case ge::DT_BF16:
            v.vBf16 = Ops::Base::bfloat16(value);
            break;
        case ge::DT_FLOAT:
            v.vF32 = static_cast<float>(value);
            break;
        case ge::DT_INT8:
            v.vI8 = static_cast<int8_t>(value);
            break;
        case ge::DT_UINT8:
            v.vU8 = static_cast<uint8_t>(value);
            break;
        case ge::DT_INT32:
            v.vI32 = static_cast<int32_t>(value);
            break;
        case ge::DT_INT64:
            v.vI64 = static_cast<int64_t>(value);
            break;
        default:
            OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y(output)",
                Ops::Base::ToString(this->outputDtype),
                "Float16, BFloat16, Float, Int32, Int64, Int8 and Uint8");
            return ge::GRAPH_FAILED;
    }

    tiling->value = v.vI64;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillsTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "FillsTiling RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"), return ge::GRAPH_FAILED);

    // get tilingdata address in context
    tiling = tilingContext->GetTilingData<FillsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);

    OP_CHECK_IF(SetAttr() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "set Attr failed"), return ge::GRAPH_FAILED);

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (this->outputDtype == ge::DT_FLOAT16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<FillsDAG<half>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_BF16) {
        baseTilingResult = elewiseBaseTiling.DoTiling<FillsDAG<bfloat16_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        baseTilingResult = elewiseBaseTiling.DoTiling<FillsDAG<float>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT8) {
        baseTilingResult = elewiseBaseTiling.DoTiling<FillsDAG<int8_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_UINT8) {
        baseTilingResult = elewiseBaseTiling.DoTiling<FillsDAG<uint8_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT32) {
        baseTilingResult = elewiseBaseTiling.DoTiling<FillsDAG<int32_t>::OpDag>(tiling->baseTiling);
    } else if (this->outputDtype == ge::DT_INT64) {
        baseTilingResult = elewiseBaseTiling.DoTiling<FillsDAG<int64_t>::OpDag>(tiling->baseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "y(output)",
            Ops::Base::ToString(this->outputDtype),
            "Float16, BFloat16, Float, Int32, Int64, Int8 and Uint8");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingData();
}

static ge::graphStatus Tiling4Fills(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Fills rt2.0 is running.");
    auto compileInfo = reinterpret_cast<const Ops::Base::ElewiseCompileInfo*>(tilingContextGen->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);

    FillsTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

static ge::graphStatus TilingPrepareForFills([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Fills).Tiling(Tiling4Fills).TilingParse<Ops::Base::ElewiseCompileInfo>(TilingPrepareForFills);
} // namespace optiling