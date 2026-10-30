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
 * \file logical_and_tiling_arch35.cpp
 * \brief
 */

#include "log/log.h"
#include "logical_and_tiling_arch35.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/logical_and/op_kernel/arch35/logical_and_dag.h"
#include "math/logical_and/op_kernel/arch35/logical_and_struct.h"
#include "op_host/math_tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

static constexpr uint64_t LOGICAL_AND_COMMON_TILING_PRIORITY = 0;

ge::graphStatus LogicalAndTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool LogicalAndTiling::IsCapable()
{
    return true;
}

ge::graphStatus LogicalAndTiling::DoOpTiling()
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
    if ((input0DType != ge::DT_BOOL) || (input1DType != ge::DT_BOOL) || (outputDtype != ge::DT_BOOL)) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(input0DType) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(input1DType) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1, x2 and y", dtypesStr.c_str(), "The dtypes of x1, x2 and y must be bool");
        return ge::GRAPH_FAILED;
    }

    Ops::Base::BroadcastBaseTiling<LogicalAndOp::LogicalAndCompute<uint8_t>::OpDag> brcBaseTiling(context_);
    ge::graphStatus ret = brcBaseTiling.DoTiling();
    tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    return ret;
}

ge::graphStatus LogicalAndTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LogicalAndTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus LogicalAndTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogicalAndTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogicalAndTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4LogicalAnd(gert::TilingContext* context)
{
    OP_LOGD("LogicalAndTiling", "Enter Tiling4LogicalAnd");
    if (context == nullptr) {
        OP_LOGE("Tiling4LogicalAnd", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const LogicalAndCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context, "Enter ascendc Tiling4LogicalAnd");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForLogicalAnd(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LogicalAnd).Tiling(Tiling4LogicalAnd).TilingParse<LogicalAndCompileInfo>(TilingPrepareForLogicalAnd);

REGISTER_OPS_TILING_TEMPLATE(LogicalAnd, LogicalAndTiling, LOGICAL_AND_COMMON_TILING_PRIORITY);
} // namespace optiling