/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mul_no_nan_tiling_arch35.cpp
 * \brief
 */

#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/mul_no_nan/op_kernel/arch35/mul_no_nan_dag.h"
#include "math/mul_no_nan/op_kernel/arch35/mul_no_nan_struct.h"
#include "mul_no_nan_tiling_arch35.h"
#include "op_host/math_tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr static uint64_t MUL_NO_NAN_COMMON_TILING_PRIORITY = 0;

ge::graphStatus MulNoNanTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool MulNoNanTiling::IsCapable()
{
    return true;
}

bool MulNoNanTiling::CheckDtype(
    const ge::DataType& x1Dtype, const ge::DataType& x2Dtype, const ge::DataType& outputDtype) const
{
    if (x1Dtype != x2Dtype || x1Dtype != outputDtype) {
        std::string reasonMsg = "The dtypes of x1, x2 and y must all be the same. Got " +
                                ge::TypeUtils::DataTypeToSerialString(x1Dtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(x2Dtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype) + ".";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(x1Dtype).c_str(), reasonMsg.c_str());
        return false;
    }
    return true;
}

ge::graphStatus MulNoNanTiling::DoOpTiling()
{
    auto x1Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    ge::DataType x1Dtype = x1Desc->GetDataType();
    auto x2Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    ge::DataType x2Dtype = x2Desc->GetDataType();
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    if (!CheckDtype(x1Dtype, x2Dtype, outputDtype)) {
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (x1Dtype == ge::DT_FLOAT) {
        Ops::Base::BroadcastBaseTiling<MulNoNanOp::MulNoNan<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1Dtype == ge::DT_INT32) {
        Ops::Base::BroadcastBaseTiling<MulNoNanOp::MulNoNan<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1Dtype == ge::DT_FLOAT16) {
        Ops::Base::BroadcastBaseTiling<MulNoNanOp::MulNoNanFloatCast<Ops::Base::half, float>::OpDag> brcBaseTiling(
            context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (x1Dtype == ge::DT_BF16) {
        Ops::Base::BroadcastBaseTiling<MulNoNanOp::MulNoNanFloatCast<Ops::Base::bfloat16_t, float>::OpDag> brcBaseTiling(
            context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(x1Dtype).c_str(),
            "float16, float32, int32, bfloat16");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus MulNoNanTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MulNoNanTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus MulNoNanTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MulNoNanTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MulNoNanTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMulNoNan(gert::TilingContext* context)
{
    OP_LOGD("MulNoNanTiling", "Enter TilingForMulNoNan");
    if (context == nullptr) {
        OP_LOGE("MulNoNanTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const MulNoNanCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context, "Enter ascendc MulNoNanTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForMulNoNan(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<MulNoNanCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MulNoNan)
    .Tiling(TilingForMulNoNan)
    .TilingParse<MulNoNanCompileInfo>(TilingPrepareForMulNoNan);

REGISTER_OPS_TILING_TEMPLATE(MulNoNan, MulNoNanTiling, MUL_NO_NAN_COMMON_TILING_PRIORITY);
} // namespace optiling
