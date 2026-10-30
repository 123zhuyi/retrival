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
 * \file axpy_tiling_arch35.cpp
 * \brief axpy_tiling source file
 */

#include <cstddef>
#include <type_traits>
#include "../../op_kernel/arch35/axpy_dag.h"
#include "../../op_kernel/arch35/axpy_struct.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "axpy_tiling_arch35.h"
#include "log/log.h"

using namespace Ops::Base;
using namespace AscendC;
using namespace ge;

namespace optiling
{

constexpr static uint64_t AXPY_COMMON_TILING_PRIORITY = 0;

ge::graphStatus AxpyTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool AxpyTiling::IsCapable()
{
    return true;
}

bool AxpyTiling::CheckDtype(const ge::DataType& input0Dtype, const ge::DataType& input1Dtype,
                            const ge::DataType& outputDtype) const
{
    bool isDtypeSame = (input0Dtype == input1Dtype) && (input1Dtype == outputDtype);
    bool isDtypeSupported = (input0Dtype == ge::DT_FLOAT || input0Dtype == ge::DT_FLOAT16 ||
                             input0Dtype == ge::DT_BF16 || input0Dtype == ge::DT_INT32);
    if (!isDtypeSame) {
        std::string reasonMsg = "The dtype of x1 must be the same as the dtypes " +
                                ge::TypeUtils::DataTypeToSerialString(input1Dtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype) + " of x2 and y";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str(),
            reasonMsg.c_str());
        return false;
    }
    if (!isDtypeSupported) {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str(),
            "float32, float16, bfloat16 or int32");
        return false;
    }
    return true;
}

ge::graphStatus AxpyTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType input0Dtype = input0Desc->GetDataType();
    ge::DataType input1Dtype = input1Desc->GetDataType();
    ge::DataType outputDtype = outputDesc->GetDataType();
    if (!CheckDtype(input0Dtype, input1Dtype, outputDtype)) {
        return ge::GRAPH_FAILED;
    }
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    const float* alphaValuePtr = runtimeAttrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, alphaValuePtr);
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0Dtype == ge::DT_INT32) {
        BroadcastBaseTiling<AxpyOp::AxpyCompute<int32_t, AxpyOp::CAST_RINT_MODE, AxpyOp::CAST_TRUNC_MODE>::OpDag>
            brcBaseTiling(context_);
        brcBaseTiling.SetScalar(*alphaValuePtr);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_BF16 || input0Dtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<AxpyOp::AxpyCompute<half, AxpyOp::CAST_NONE_MODE, AxpyOp::CAST_RINT_MODE>::OpDag>
            brcBaseTiling(context_);
        brcBaseTiling.SetScalar(*alphaValuePtr);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<AxpyOp::AxpyComputeFloat::OpDag>
            brcBaseTiling(context_);
        brcBaseTiling.SetScalar(*alphaValuePtr);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    }

    return ret;
}

ge::graphStatus AxpyTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AxpyTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus AxpyTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AxpyTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AxpyTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForAxpy(gert::TilingContext* context)
{
    OP_LOGD("AxpyTiling", "Enter TilingForAxpy");
    if (context == nullptr) {
        OP_LOGE("AxpyTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const AxpyCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc AxpyTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForAxpy(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AxpyCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Axpy).Tiling(TilingForAxpy).TilingParse<AxpyCompileInfo>(TilingPrepareForAxpy);

REGISTER_OPS_TILING_TEMPLATE(Axpy, AxpyTiling, AXPY_COMMON_TILING_PRIORITY);
}  // namespace optiling
