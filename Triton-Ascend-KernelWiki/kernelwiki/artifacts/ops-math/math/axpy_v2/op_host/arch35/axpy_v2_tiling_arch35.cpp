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
 * \file axpy_v2_tiling_arch35.cpp
 * \brief axpy_v2_tiling_arch35 source file
 */

#include <graph/utils/type_utils.h>
#include "op_host/math_tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/axpy_v2_dag.h"
#include "../../op_kernel/arch35/axpy_v2_struct.h"
#include "axpy_v2_tiling_arch35.h"

using namespace Ops::Base;
using namespace AscendC;
using namespace ge;

namespace optiling
{

constexpr static uint64_t AXPY_V2_COMMON_TILING_PRIORITY = 0;
constexpr static uint32_t INPUT_ALPHA_IDX = 2;

ge::graphStatus AxpyV2Tiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool AxpyV2Tiling::IsCapable()
{
    return true;
}

bool AxpyV2Tiling::CheckDtype(const ge::DataType& input0Dtype, const ge::DataType& input1Dtype,
                              const ge::DataType& inputAlphaDtype, const ge::DataType& outputDtype) const
{
    bool isDtypeSame = (input0Dtype == input1Dtype && input1Dtype == inputAlphaDtype && inputAlphaDtype == outputDtype);
    bool isDtypeSupported = (input0Dtype == ge::DT_FLOAT || input0Dtype == ge::DT_FLOAT16 ||
                             input0Dtype == ge::DT_BF16 || input0Dtype == ge::DT_BOOL || input0Dtype == ge::DT_INT64 ||
                             input0Dtype == ge::DT_UINT8 || input0Dtype == ge::DT_INT8 || input0Dtype == ge::DT_INT32);
    if (!isDtypeSame) {
        std::string reasonMsg = "The dtype of x1 must be the same as the dtypes " +
                                ge::TypeUtils::DataTypeToSerialString(input1Dtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(inputAlphaDtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype) + " of x2, alpha and y";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str(),
            reasonMsg.c_str());
        return false;
    }
    if (!isDtypeSupported) {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str(),
            "float32, float16, bfloat16, int32, int64, int8, uint8 or bool");
        return false;
    }
    return true;
}

ge::graphStatus AxpyV2Tiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    auto inputAlphaDesc = context_->GetInputDesc(INPUT_ALPHA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputAlphaDesc);
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType input0Dtype = input0Desc->GetDataType();
    ge::DataType input1Dtype = input1Desc->GetDataType();
    ge::DataType inputAlphaDtype = inputAlphaDesc->GetDataType();
    ge::DataType outputDtype = outputDesc->GetDataType();
    if (!CheckDtype(input0Dtype, input1Dtype, inputAlphaDtype, outputDtype)) {
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0Dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<AxpyV2Op::AxpyV2ComputeFloat::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_BF16 || input0Dtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<
            AxpyV2Op::AxpyV2ComputeFp16Bf16<half, AxpyV2Op::CAST_NONE_MODE, AxpyV2Op::CAST_RINT_MODE>::OpDag>
            brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT32) {
        BroadcastBaseTiling<AxpyV2Op::AxpyV2ComputeInt32Int64<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT64) {
        BroadcastBaseTiling<AxpyV2Op::AxpyV2ComputeInt32Int64<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT8) {
        BroadcastBaseTiling<AxpyV2Op::AxpyV2ComputeUint8Int8<int8_t, int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_UINT8) {
        BroadcastBaseTiling<AxpyV2Op::AxpyV2ComputeUint8Int8<uint8_t, uint32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_BOOL) {
        BroadcastBaseTiling<AxpyV2Op::AxpyV2ComputeBool::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    }

    return ret;
}

ge::graphStatus AxpyV2Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t AxpyV2Tiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus AxpyV2Tiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AxpyV2Tiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AxpyV2Tiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForAxpyV2(gert::TilingContext* context)
{
    OP_LOGD("AxpyV2Tiling", "Enter TilingForAxpyV2");
    if (context == nullptr) {
        OP_LOGE("AxpyV2Tiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const AxpyV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc AxpyV2Tiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForAxpyV2(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AxpyV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AxpyV2).Tiling(TilingForAxpyV2).TilingParse<AxpyV2CompileInfo>(TilingPrepareForAxpyV2);

REGISTER_OPS_TILING_TEMPLATE(AxpyV2, AxpyV2Tiling, AXPY_V2_COMMON_TILING_PRIORITY);
}  // namespace optiling
