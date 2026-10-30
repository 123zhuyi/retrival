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
 * \file minimum_tiling_arch35.cpp
 * \brief minimum_tiling_arch35 source file
 */

#include <vector>
#include <algorithm>
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"

#include "math/minimum/op_kernel/arch35/minimum_dag.h"
#include "math/minimum/op_kernel/arch35/minimum_struct.h"
#include "minimum_tiling_arch35.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Base;

namespace optiling {

constexpr static uint64_t MINIMUM_COMMON_TILING_PRIORITY = 0;

const std::vector<ge::DataType> DTYPE_LIST = {ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16,  ge::DataType::DT_FLOAT,
                                              ge::DataType::DT_INT32,   ge::DataType::DT_INT64, ge::DataType::DT_INT8,
                                              ge::DataType::DT_UINT8};

ge::graphStatus MinimumTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool MinimumTiling::IsCapable()
{
    return true;
}

ge::graphStatus MinimumTiling::DoOpTiling()
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
        BroadcastBaseTiling<MinimumOp::MinimumCompute<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1DType == ge::DT_INT32) {
        BroadcastBaseTiling<MinimumOp::MinimumCompute<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1DType == ge::DT_UINT8) {
        BroadcastBaseTiling<MinimumOp::MinimumCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1DType == ge::DT_INT8) {
        BroadcastBaseTiling<MinimumOp::MinimumCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1DType == ge::DT_FLOAT16 || x1DType == ge::DT_BF16) {
        BroadcastBaseTiling<MinimumOp::MinimumCompute<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<MinimumOp::MinimumCompute<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    }

    return ret;
}

ge::graphStatus MinimumTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MinimumTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus MinimumTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimumTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimumTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMinimum(gert::TilingContext* context)
{
    OP_LOGD("MinimumTiling", "Enter TilingForMinimum");
    if (context == nullptr) {
        OP_LOGE("MinimumTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc MinimumTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForMinimum(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Enter TilingPrepareForMaximum");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Minimum).Tiling(TilingForMinimum).TilingParse<BroadcastCompileInfo>(TilingPrepareForMinimum);

REGISTER_OPS_TILING_TEMPLATE(Minimum, MinimumTiling, MINIMUM_COMMON_TILING_PRIORITY);
} // namespace optiling