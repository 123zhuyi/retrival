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
 * \file div_no_nan_tiling_arch35.cpp
 * \brief div_no_nan_tiling source file
 */

#include "div_no_nan_tiling_arch35.h"

#include <graph/utils/type_utils.h>

#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "math/div_no_nan/op_kernel/arch35/div_no_nan_dag.h"
#include "math/div_no_nan/op_kernel/arch35/div_no_nan_struct.h"
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Base;

namespace optiling {

static constexpr uint64_t DIV_NO_NAN_COMMON_TILING_PRIORITY = 0;

ge::graphStatus DivNoNanTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool DivNoNanTiling::IsCapable()
{
    return true;
}

ge::graphStatus DivNoNanTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();

    auto outputYDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    ge::DataType outputDtype = outputYDesc->GetDataType();
    if ((input0DType != input1DType) || (outputDtype != input1DType)) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(input0DType) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(input1DType) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(outputDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1, x2 and y", dtypeMsg.c_str(), "The dtypes of x1, x2 and y must be the same");
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0DType == ge::DT_INT32) {
        BroadcastBaseTiling<DivNoNanOp::DivNoNan<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<DivNoNanOp::DivNoNanFloatCast<half, float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_BF16) {
        BroadcastBaseTiling<DivNoNanOp::DivNoNanFloatCast<bfloat16_t, float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<DivNoNanOp::DivNoNan<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT8) {
        BroadcastBaseTiling<DivNoNanOp::DivNoNanU8Cast<uint8_t, uint16_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT8) {
        BroadcastBaseTiling<DivNoNanOp::DivNoNanDagS8<int8_t, int16_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0DType).c_str(),
            "int32, float16, bf16, float, uint8 and int8");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus DivNoNanTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t DivNoNanTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus DivNoNanTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DivNoNanTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DivNoNanTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4DivNoNan(gert::TilingContext* context)
{
    OP_LOGD("DivNoNanTiling", "Enter Tiling4DivNoNan");
    if (context == nullptr) {
        OP_LOGE("Tiling4DivNoNan", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc Tiling4DivNoNan");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}
ge::graphStatus TilingPrepareForDivNoNan([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DivNoNan).Tiling(Tiling4DivNoNan).TilingParse<BroadcastCompileInfo>(TilingPrepareForDivNoNan);

REGISTER_OPS_TILING_TEMPLATE(DivNoNan, DivNoNanTiling, DIV_NO_NAN_COMMON_TILING_PRIORITY);
} // namespace optiling
