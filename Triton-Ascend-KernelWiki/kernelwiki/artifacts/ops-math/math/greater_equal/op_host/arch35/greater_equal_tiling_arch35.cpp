/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include "greater_equal_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "math/greater_equal/op_kernel/arch35/greater_equal_dag.h"
#include "math/greater_equal/op_kernel/arch35/greater_equal_struct.h"
#include "atvoss/util/vec.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"

namespace optiling {
using namespace AscendC;
using namespace ge;
using namespace Ops::Base;

constexpr static uint64_t GREATER_EQUAL_COMMON_TILING_PRIORITY = 0;

ge::graphStatus GreaterEqualTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool GreaterEqualTiling::IsCapable()
{
    return true;
}

ge::graphStatus GreaterEqualTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();
    if (input0DType != input1DType) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(input0DType) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(input1DType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1 and x2",
            dtypesStr.c_str(), "The dtypes of x1 and x2 must be the same");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0DType == ge::DT_INT64) {
        BroadcastBaseTiling<GreaterEqualOp::GreaterEqualCompute<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT32) {
        BroadcastBaseTiling<GreaterEqualOp::GreaterEqualCompute<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT16 || input0DType == ge::DT_BF16) {
        BroadcastBaseTiling<GreaterEqualOp::GreaterEqualCompute<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<GreaterEqualOp::GreaterEqualCompute<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT8) {
        BroadcastBaseTiling<GreaterEqualOp::GreaterEqualCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT8 || input0DType == ge::DT_BOOL) {
        BroadcastBaseTiling<GreaterEqualOp::GreaterEqualCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT64) {
        BroadcastBaseTiling<GreaterEqualOp::GreaterEqualCompute<uint64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(input0DType);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", dtypeStr.c_str(),
            "fp16, bf16, fp32, uint64, int64, int32, uint8, int8 or bool");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus GreaterEqualTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GreaterEqualTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus GreaterEqualTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GreaterEqualTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GreaterEqualTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGreaterEqual(gert::TilingContext* context)
{
    OP_LOGD("GreaterEqualTiling", "Enter TilingForGreaterEqual");
    if (context == nullptr) {
        OP_LOGE("GreaterEqualTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "Enter ascendc GreaterEqualTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context); 
}

ge::graphStatus TilingPrepareForGreaterEqual([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GreaterEqual)
    .Tiling(TilingForGreaterEqual)
    .TilingParse<BroadcastCompileInfo>(TilingPrepareForGreaterEqual);

REGISTER_OPS_TILING_TEMPLATE(GreaterEqual, GreaterEqualTiling, GREATER_EQUAL_COMMON_TILING_PRIORITY);
}   // namespace optiling
