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
 * \file add_tiling_arch35.cpp
 * \brief add_tiling_arch35 source file
 */
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/add/op_kernel/arch35/add_dag.h"
#include "math/add/op_kernel/arch35/add_struct.h"
#include "add_tiling_arch35.h"

namespace optiling {

constexpr static uint64_t ADD_COMMON_TILING_PRIORITY = 0;

ge::graphStatus AddTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool AddTiling::IsCapable()
{
    return true;
}

bool AddTiling::IsMixedDtype(const ge::DataType& input0Dtype, const ge::DataType& input1Dtype) const
{
    return (input0Dtype == ge::DT_FLOAT16 && input1Dtype == ge::DT_FLOAT) ||
           (input0Dtype == ge::DT_FLOAT && input1Dtype == ge::DT_FLOAT16) ||
           (input0Dtype == ge::DT_BF16 && input1Dtype == ge::DT_FLOAT) ||
           (input0Dtype == ge::DT_FLOAT && input1Dtype == ge::DT_BF16);
}

bool AddTiling::CheckDtype(
    const ge::DataType& input0Dtype, const ge::DataType& input1Dtype, const ge::DataType& outputDtype,
    bool isMixedDtype) const
{
    if (!isMixedDtype && (input0Dtype != input1Dtype || input0Dtype != outputDtype)) {
        std::string reasonMsg = "The dtype of x1 must be the same as the dtypes " +
                                ge::TypeUtils::DataTypeToSerialString(input1Dtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype) + " of x2 and y"
                                ", when the dtypes of x1 and x2 are not within the supported floating-point type combinations: "
                                "x1 is float and x2 is float16 or bfloat16, or x2 is float and x2 is float16 or bfloat16";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str(),
            reasonMsg.c_str());
        return false;
    }
    if (isMixedDtype && outputDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "y",
            ge::TypeUtils::DataTypeToSerialString(outputDtype).c_str(),
            "The dtype of output y must be float, when the dtype of x1 or x2 is float "
            "and the dtype of another input x2 or x1 is float16 or bfloat16");
        return false;
    }
    return true;
}

ge::graphStatus AddTiling::DoOpTiling()
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
    bool isMixedDtype = IsMixedDtype(input0Dtype, input1Dtype);
    if (!CheckDtype(input0Dtype, input1Dtype, outputDtype, isMixedDtype)) {
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (isMixedDtype && input1Dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<AddOp::AddMixDtypeCompute<half, float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (isMixedDtype && input0Dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<AddOp::AddMixDtypeCompute<float, half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_FLOAT16 || input0Dtype == ge::DT_BF16) {
        BroadcastBaseTiling<AddOp::AddWithCastCompute<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<AddOp::AddWithCastCompute<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_BOOL) {
        BroadcastBaseTiling<AddOp::AddBoolCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT64 || input0Dtype == ge::DT_COMPLEX64) {
        BroadcastBaseTiling<AddOp::AddWithoutCastCompute<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_UINT8) {
        BroadcastBaseTiling<AddOp::AddWithoutCastCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT8) {
        BroadcastBaseTiling<AddOp::AddWithoutCastCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT32 || input0Dtype == ge::DT_COMPLEX32) {
        BroadcastBaseTiling<AddOp::AddWithoutCastCompute<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str(),
            "fp16, bf16, fp32, int64, int32, uint8, int8, bool, complex32, complex64");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus AddTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AddTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus AddTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForAdd(gert::TilingContext* context)
{
    OP_LOGD("AddTiling", "Enter TilingForAdd");
    if (context == nullptr) {
        OP_LOGE("AddTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const AddCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc AddTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForAdd(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AddCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Add).Tiling(TilingForAdd).TilingParse<AddCompileInfo>(TilingPrepareForAdd);

REGISTER_OPS_TILING_TEMPLATE(Add, AddTiling, ADD_COMMON_TILING_PRIORITY);
} // namespace optiling
