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
 * \file maximum_tiling_arch35.cpp
 * \brief maximum_tiling_arch35 source file
 */

#include <vector>
#include <algorithm>
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"

#include "math/maximum/op_kernel/arch35/maximum_dag.h"
#include "math/maximum/op_kernel/arch35/maximum_struct.h"
#include "maximum_tiling_arch35.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Base;

namespace optiling {

constexpr static uint64_t MAXIMUM_COMMON_TILING_PRIORITY = 0;

const std::vector<ge::DataType> DTYPE_LIST = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16,  ge::DataType::DT_FLOAT,
                                              ge::DataType::DT_INT32,   ge::DataType::DT_INT64, ge::DataType::DT_INT8,
                                              ge::DataType::DT_UINT8};

ge::graphStatus MaximumTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool MaximumTiling::IsCapable()
{
    return true;
}

ge::graphStatus MaximumTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType x1DType = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType x2DType = input1Desc->GetDataType();
    if (x1DType != x2DType) {
        std::string dtypesStr = TypeUtils::DataTypeToSerialString(x1DType) + " and " +
                                TypeUtils::DataTypeToSerialString(x2DType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1 and x2",
            dtypesStr.c_str(), "The dtypes of x1 and x2 must be the same");
        return ge::GRAPH_FAILED;
    }

    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType yDType = outputDesc->GetDataType();
    if (x1DType != yDType) {
        std::string dtypesStr =
            TypeUtils::DataTypeToSerialString(x1DType) + " and " + TypeUtils::DataTypeToSerialString(yDType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1 and y", dtypesStr.c_str(), "The dtypes of x1 and y must be the same");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        std::find(DTYPE_LIST.begin(), DTYPE_LIST.end(), x1DType) == DTYPE_LIST.end(),
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(x1DType).c_str(),
            "fp16, bf16, fp32, int64, int32, int8 or uint8"),
        return ge::GRAPH_FAILED);

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (x1DType == ge::DT_INT64) {
        BroadcastBaseTiling<MaximumOp::MaximumCompute<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1DType == ge::DT_INT32) {
        BroadcastBaseTiling<MaximumOp::MaximumCompute<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1DType == ge::DT_UINT8) {
        BroadcastBaseTiling<MaximumOp::MaximumCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1DType == ge::DT_INT8) {
        BroadcastBaseTiling<MaximumOp::MaximumCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1DType == ge::DT_FLOAT16 || x1DType == ge::DT_BF16) {
        BroadcastBaseTiling<MaximumOp::MaximumCompute<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<MaximumOp::MaximumCompute<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    }

    return ret;
}

ge::graphStatus MaximumTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MaximumTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus MaximumTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaximumTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaximumTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMaximum(gert::TilingContext* context)
{
    OP_LOGD("MaximumTiling", "Enter TilingForMaximum");
    if (context == nullptr) {
        OP_LOGE("MaximumTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = static_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc MaximumTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForMaximum(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Enter TilingPrepareForMaximum");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Maximum).Tiling(TilingForMaximum).TilingParse<BroadcastCompileInfo>(TilingPrepareForMaximum);

REGISTER_OPS_TILING_TEMPLATE(Maximum, MaximumTiling, MAXIMUM_COMMON_TILING_PRIORITY);
} // namespace optiling