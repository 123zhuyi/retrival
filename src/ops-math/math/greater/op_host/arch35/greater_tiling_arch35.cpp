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
#include "greater_tiling_arch35.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/greater/op_kernel/arch35/greater_dag.h"
#include "math/greater/op_kernel/arch35/greater_struct.h"
#include "op_host/math_tiling_templates_registry.h"

namespace optiling {
using namespace AscendC;
using namespace ge;
using namespace Ops::Base;

constexpr static uint64_t GREATER_COMMON_TILING_PRIORITY = 0;

bool GreaterTiling::IsCapable()
{
    return true;
}

ge::graphStatus GreaterTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GreaterTiling::DoOpTiling()
{
    auto greaterInput0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, greaterInput0Desc);
    ge::DataType greaterInput0DType = greaterInput0Desc->GetDataType();
    auto greaterInput1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, greaterInput1Desc);
    ge::DataType greaterInput1DType = greaterInput1Desc->GetDataType();
    if (greaterInput0DType != greaterInput1DType) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(greaterInput0DType) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(greaterInput1DType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1 and x2",
            dtypesStr.c_str(), "The dtypes of x1 and x2 must be the same");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (greaterInput0DType == ge::DT_INT64) {
        BroadcastBaseTiling<GreaterOp::GreaterCompute<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (greaterInput0DType == ge::DT_INT32) {
        BroadcastBaseTiling<GreaterOp::GreaterCompute<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (greaterInput0DType == ge::DT_FLOAT16 || greaterInput0DType == ge::DT_BF16) {
        BroadcastBaseTiling<GreaterOp::GreaterCompute<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (greaterInput0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<GreaterOp::GreaterCompute<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (greaterInput0DType == ge::DT_UINT8) {
        BroadcastBaseTiling<GreaterOp::GreaterCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (greaterInput0DType == ge::DT_INT8 || greaterInput0DType == ge::DT_BOOL) {
        BroadcastBaseTiling<GreaterOp::GreaterCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (greaterInput0DType == ge::DT_UINT64) {
        BroadcastBaseTiling<GreaterOp::GreaterCompute<uint64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(greaterInput0DType);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", dtypeStr.c_str(),
            "fp16, bf16, fp32, uint64, int64, int32, uint8, int8 or bool");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus GreaterTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GreaterTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus GreaterTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GreaterTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GreaterTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGreater(gert::TilingContext* context)
{
    OP_LOGD("GreaterTiling", "Enter TilingForGreater");
    if (context == nullptr) {
        OP_LOGE("GreaterTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "Enter ascendc GreaterTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForGreater([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Greater)
    .Tiling(TilingForGreater)
    .TilingParse<BroadcastCompileInfo>(TilingPrepareForGreater);

REGISTER_OPS_TILING_TEMPLATE(Greater, GreaterTiling, GREATER_COMMON_TILING_PRIORITY);

}   // namespace optiling