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
 * \file    squared_difference_tiling_arch35.cpp
 * \brief   squared_difference_tiling source file
 */

#include "squared_difference_tiling_arch35.h"
#include "log/log.h"
#include "atvoss/util/vec.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/squared_difference_dag.h"
#include "../../op_kernel/arch35/squared_difference_struct.h"
#include "op_host/math_tiling_templates_registry.h"
#include "register/op_impl_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static constexpr uint64_t SQUARED_DIFFERENCE_COMMON_TILING_PRIORITY = 0;

ge::graphStatus SquaredDifferenceTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool SquaredDifferenceTiling::IsCapable()
{
    return true;
}

ge::graphStatus SquaredDifferenceTiling::DoOpTiling()
{
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType input0DType = inputDesc->GetDataType();

    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();

    auto outputYDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    ge::DataType outputDtype = outputYDesc->GetDataType();
    if ((input0DType != input1DType) || (outputDtype != input1DType)) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(input0DType) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(input1DType) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1, x2 and y",
            dtypesStr.c_str(), "The dtypes of x1, x2 and y must be the same");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0DType == ge::DT_INT64) {
        Ops::Base::BroadcastBaseTiling<SquaredDifferenceOp::SquaredDifference<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT32) {
        Ops::Base::BroadcastBaseTiling<SquaredDifferenceOp::SquaredDifference<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT16 || input0DType == ge::DT_BF16) {
        Ops::Base::BroadcastBaseTiling<SquaredDifferenceOp::SquaredDifferenceCast<Ops::Base::half>::OpDag>
            brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        Ops::Base::BroadcastBaseTiling<SquaredDifferenceOp::SquaredDifference<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(input0DType);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", dtypeStr.c_str(),
            "int64, int32, float16, bf16 or fp32");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus SquaredDifferenceTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t SquaredDifferenceTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus SquaredDifferenceTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SquaredDifferenceTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SquaredDifferenceTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSquaredDifference(gert::TilingContext* context)
{
    OP_LOGD("SquaredDifferenceTiling", "Enter TilingForSquaredDifference");
    if (context == nullptr) {
        OP_LOGE("SquaredDifferenceTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const SquaredDifferenceCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context, "Enter ascendc SquaredDifferenceTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForSquaredDifference(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<SquaredDifferenceCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SquaredDifference)
    .Tiling(TilingForSquaredDifference)
    .TilingParse<SquaredDifferenceCompileInfo>(TilingPrepareForSquaredDifference);

REGISTER_OPS_TILING_TEMPLATE(SquaredDifference, SquaredDifferenceTiling, SQUARED_DIFFERENCE_COMMON_TILING_PRIORITY);
} // namespace optiling