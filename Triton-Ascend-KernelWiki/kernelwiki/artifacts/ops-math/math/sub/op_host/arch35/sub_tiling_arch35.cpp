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
 * \file sub_tiling_arch35.cpp
 * \brief sub_tiling source file
 */

#include <graph/utils/type_utils.h>
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/sub_dag.h"
#include "../../op_kernel/arch35/sub_struct.h"
#include "sub_tiling_arch35.h"

using namespace Ops::Base;
using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr static uint64_t SUB_COMMON_TILING_PRIORITY = 0;

ge::graphStatus SubTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool SubTiling::IsCapable()
{
    return true;
}

bool SubTiling::CheckDtype(const ge::DataType& input0Dtype, const ge::DataType& input1Dtype,
                           const ge::DataType& outputDtype) const
{
    if (input0Dtype != input1Dtype || input0Dtype != outputDtype) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(input0Dtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(input1Dtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1, x2 and y",
            dtypesStr.c_str(), "The dtypes of x1, x2 and y must be the same");
        return false;
    }
    return true;
}

ge::graphStatus SubTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0Dtype = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1Dtype = input1Desc->GetDataType();
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    if (!CheckDtype(input0Dtype, input1Dtype, outputDtype)) {
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0Dtype == ge::DT_FLOAT16 || input0Dtype == ge::DT_BF16) {
        BroadcastBaseTiling<SubOp::SubWithCastCompute<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<SubOp::SubWithCastCompute<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_BOOL) {
        BroadcastBaseTiling<SubOp::SubBoolCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT64 || input0Dtype == ge::DT_COMPLEX64) {
        BroadcastBaseTiling<SubOp::SubWithoutCastCompute<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_UINT8) {
        BroadcastBaseTiling<SubOp::SubWithoutCastCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT8) {
        BroadcastBaseTiling<SubOp::SubWithoutCastCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT32 || input0Dtype == ge::DT_COMPLEX32) {
        BroadcastBaseTiling<SubOp::SubWithoutCastCompute<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(input0Dtype);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", dtypeStr.c_str(),
            "fp16, bf16, fp32, int64, int32, uint8, int8, bool, complex32 or complex64");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus SubTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t SubTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus SubTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SubTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SubTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSub(gert::TilingContext* context)
{
    OP_LOGD("SubTiling", "Enter TilingForSub");
    if (context == nullptr) {
        OP_LOGE("SubTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const SubCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc SubTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForSub(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<SubCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Sub)
    .Tiling(TilingForSub)
    .TilingParse<SubCompileInfo>(TilingPrepareForSub);

REGISTER_OPS_TILING_TEMPLATE(Sub, SubTiling, SUB_COMMON_TILING_PRIORITY);
}   // namespace optiling
