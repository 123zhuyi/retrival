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
 * \file div_tiling_arch35.cc
 * \brief div_tiling source file
 */

#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/div/op_kernel/arch35/div_dag.h"
#include "math/div/op_kernel/arch35/div_struct.h"
#include "div_tiling_arch35.h"

using namespace Ops::Base;
using namespace ge;

namespace optiling {

constexpr static uint64_t DIV_COMMON_TILING_PRIORITY = 0;

ge::graphStatus DivTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool DivTiling::IsCapable()
{
    return true;
}

ge::graphStatus DivTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();
    if (input0DType != input1DType) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(input0DType) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(input1DType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1 and x2", dtypeMsg.c_str(), "The dtypes of x1 and x2 must be the same");
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0DType == ge::DT_INT32) {
        BroadcastBaseTiling<DivOp::DivIntegerS32<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT16 || input0DType == ge::DT_BF16) {
        BroadcastBaseTiling<DivOp::DivFloatWithCast<half, float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<DivOp::DivFloatWithoutCast<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT8) {
        BroadcastBaseTiling<DivOp::DivIntegerU8<uint8_t, uint16_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT8) {
        BroadcastBaseTiling<DivOp::DivIntegerS8<int8_t, int16_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_COMPLEX32) {
        BroadcastBaseTiling<DivOp::DivComplexWithoutCast<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_COMPLEX64) {
        BroadcastBaseTiling<DivOp::DivComplexWithoutCast<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0DType).c_str(),
            "fp16, bf16, fp32, int32, uint8, int8, complex32 and complex64");
        return ge::GRAPH_FAILED;
    }
    return ret;
}

ge::graphStatus DivTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t DivTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus DivTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DivTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DivTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForDiv(gert::TilingContext* context)
{
    OP_LOGD("DivTiling", "Enter TilingForDiv");
    if (context == nullptr) {
        OP_LOGE("DivTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc DivTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForDiv([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Div).Tiling(TilingForDiv).TilingParse<BroadcastCompileInfo>(TilingPrepareForDiv);

REGISTER_OPS_TILING_TEMPLATE(Div, DivTiling, DIV_COMMON_TILING_PRIORITY);
} // namespace optiling
