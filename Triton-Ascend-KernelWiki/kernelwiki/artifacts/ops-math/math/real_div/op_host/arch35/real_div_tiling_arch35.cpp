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
 * \file real_div_tiling_arch35.cpp
 * \brief real_div_tiling source file
 */

#include <graph/utils/type_utils.h>
#include "real_div_tiling_arch35.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/real_div/op_kernel/arch35/real_div_dag.h"
#include "math/real_div/op_kernel/arch35/real_div_struct.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Base;

namespace optiling
{
constexpr static uint64_t REAL_DIV_COMMON_TILING_PRIORITY = 0;

ge::graphStatus RealDivTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool RealDivTiling::IsCapable()
{
    return true;
}

ge::graphStatus RealDivTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();
    if (input0DType != input1DType) {
        std::string dtypeMsg = Ops::Base::ToString(input0DType) + " and " +
                               Ops::Base::ToString(input1DType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1 and x2", dtypeMsg.c_str(), "The dtypes of x1 and x2 must be the same");
        return ge::GRAPH_FAILED;
    }
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDType = outputDesc->GetDataType();
    if ((input0DType == ge::DT_INT32) && (outputDType != ge::DT_FLOAT && outputDType != ge::DT_INT32)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "y", Ops::Base::ToString(outputDType).c_str(),
            "The dtype of y must be int32 or fp32 when the dtype of x1 is int32");
        return ge::GRAPH_FAILED;
    }

    if ((input0DType == ge::DT_BOOL) && (outputDType != ge::DT_FLOAT)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "y", Ops::Base::ToString(outputDType).c_str(),
            "The dtype of y must be fp32 when the dtype of x1 is bool");
        return ge::GRAPH_FAILED;
    }

    if (input0DType != ge::DT_INT32 && input0DType != ge::DT_BOOL && input0DType != outputDType) {
        std::string dtypeMsg = Ops::Base::ToString(input0DType) + " and " +
                               Ops::Base::ToString(outputDType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1 and y", dtypeMsg.c_str(), "The dtypes of x1 and y must be the same");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0DType == ge::DT_INT32 && outputDType == ge::DT_INT32) {
        BroadcastBaseTiling<RealDivOp::RealDivIntegerWithoutCast<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT32 && outputDType == ge::DT_FLOAT) {
        BroadcastBaseTiling<RealDivOp::RealDivIntegerWithCast<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT16 || input0DType == ge::DT_BF16) {
        BroadcastBaseTiling<RealDivOp::RealDivFloatWithCast<half, float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<RealDivOp::RealDivFloatWithoutCast<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_BOOL) {
        BroadcastBaseTiling<RealDivOp::RealDivWithBool<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT64) {
        BroadcastBaseTiling<RealDivOp::RealDivIntegerWithoutCast<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", Ops::Base::ToString(input0DType).c_str(),
            "fp16, bf16, fp32, int32 or bool");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus RealDivTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t RealDivTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus RealDivTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RealDivTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RealDivTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForRealDiv(gert::TilingContext* context)
{
    OP_LOGD("RealDivTiling", "Enter TilingForRealDiv");
    if (context == nullptr) {
        OP_LOGE("RealDivTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "Enter ascendc RealDivTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForRealDiv([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RealDiv).Tiling(TilingForRealDiv).TilingParse<BroadcastCompileInfo>(TilingPrepareForRealDiv);

REGISTER_OPS_TILING_TEMPLATE(RealDiv, RealDivTiling, REAL_DIV_COMMON_TILING_PRIORITY);
}  // namespace optiling
