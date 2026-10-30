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
 * \file less_tiling_arch35.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/less/op_kernel/arch35/less_dag.h"
#include "math/less/op_kernel/arch35/less_struct.h"
#include "less_tiling_arch35.h"

namespace optiling {

constexpr static uint64_t LESS_COMMON_TILING_PRIORITY = 0;

ge::graphStatus LessTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool LessTiling::IsCapable()
{
    return true;
}

ge::graphStatus LessTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);

    ge::DataType input0DType = input0Desc->GetDataType();
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
        BroadcastBaseTiling<LessOp::LessCompute<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT64) {
        BroadcastBaseTiling<LessOp::LessCompute<uint64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT32) {
        BroadcastBaseTiling<LessOp::LessCompute<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<LessOp::LessCompute<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_BF16 || input0DType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<LessOp::LessCompute<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT8) {
        BroadcastBaseTiling<LessOp::LessCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT8 || input0DType == ge::DT_BOOL) {
        BroadcastBaseTiling<LessOp::LessCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_DOUBLE) {
        BroadcastBaseTiling<LessOp::LessCompute<double>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(input0DType);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", dtypeStr.c_str(),
            "uint64, int64, int32, float16, bf16, float, uint8, int8, bool or double");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus LessTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LessTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus LessTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LessTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LessTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForLess(gert::TilingContext* context)
{
    OP_LOGD("LessTiling", "Enter TilingForLess");
    if (context == nullptr) {
        OP_LOGE("TilingForLess", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const LessCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc TilingForLess");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForLess(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<LessCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Less).Tiling(TilingForLess).TilingParse<LessCompileInfo>(TilingPrepareForLess);

REGISTER_OPS_TILING_TEMPLATE(Less, LessTiling, LESS_COMMON_TILING_PRIORITY);
} // namespace optiling