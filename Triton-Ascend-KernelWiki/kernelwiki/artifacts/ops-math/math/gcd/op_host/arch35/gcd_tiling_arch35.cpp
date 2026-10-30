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
 * \file gcd_tiling_arch35.cpp
 * \brief gcd_tiling_arch35
 */

#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/gcd/op_kernel/arch35/gcd_dag.h"
#include "math/gcd/op_kernel/arch35/gcd_struct.h"
#include "gcd_tiling_arch35.h"

namespace optiling {
constexpr static uint64_t GCD_COMMON_TILING_PRIORITY = 0;
constexpr static int64_t DCACHE_SIZE = 32 * 1024;

ge::graphStatus GcdTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool GcdTiling::IsCapable()
{
    return true;
}

ge::graphStatus GcdTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();

    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();
    auto outputYDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    ge::DataType outputDtype = outputYDesc->GetDataType();
    if (input0DType != input1DType || input0DType != outputDtype) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(input0DType) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(input1DType) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(outputDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1, x2 and y", dtypeMsg.c_str(), "The dtypes of x1, x2 and y must be the same");
        return ge::GRAPH_FAILED;
    }

    int64_t extraSize = DCACHE_SIZE;
    int64_t extraBufferNum = 0;

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0DType == ge::DT_UINT8) {
        BroadcastBaseTiling<GcdOp::GcdCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling(extraSize, extraBufferNum);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT8) {
        BroadcastBaseTiling<GcdOp::GcdCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling(extraSize, extraBufferNum);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT16) {
        BroadcastBaseTiling<GcdOp::GcdCompute<int16_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling(extraSize, extraBufferNum);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT32) {
        BroadcastBaseTiling<GcdOp::GcdCompute<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling(extraSize, extraBufferNum);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT64) {
        BroadcastBaseTiling<GcdOp::GcdCompute<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling(extraSize, extraBufferNum);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0DType).c_str(),
            "uint8, int8, int16, int32 and int64");
        return ge::GRAPH_FAILED;
    }
    
    return ret;
}

ge::graphStatus GcdTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GcdTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus GcdTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GcdTiling::PostTiling()
{
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GcdTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const BroadcastCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        ubSize_ = compileInfoPtr->ubSize;
        OP_LOGD(context_->GetNodeName(), "Get ubSize form compileInfo is: %ld", ubSize_);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<int64_t>(ubSizePlatform);
        OP_LOGD(context_->GetNodeName(), "Get ubSize form ascendcPlatform is: %ld", ubSize_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGcd(gert::TilingContext* context)
{
    OP_LOGD("GcdTiling", "Enter TilingForGcd");
    if (context == nullptr) {
        OP_LOGE("GcdTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc GcdTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForGcd(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Gcd).Tiling(TilingForGcd).TilingParse<BroadcastCompileInfo>(TilingPrepareForGcd);
REGISTER_OPS_TILING_TEMPLATE(Gcd, GcdTiling, GCD_COMMON_TILING_PRIORITY);
} // namespace optiling