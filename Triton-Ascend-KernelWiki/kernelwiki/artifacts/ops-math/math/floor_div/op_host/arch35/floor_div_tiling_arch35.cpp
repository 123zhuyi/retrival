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
 * \file floor_div_tiling.cpp
 * \brief floor_div_tiling source file
 */
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/floor_div/op_kernel/arch35/floor_div_dag.h"
#include "math/floor_div/op_kernel/arch35/floor_div_struct.h"
#include "floor_div_tiling_arch35.h"

namespace optiling {

constexpr static uint64_t FLOOR_DIV_COMMON_TILING_PRIORITY = 0;
constexpr static int64_t DCACHE_SIZE = 32 * 1024;

ge::graphStatus FloorDivTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool FloorDivTiling::IsCapable()
{
    return true;
}

ge::graphStatus FloorDivTiling::DoOpTiling()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();
    if (input0DType != input1DType) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(input0DType) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(input1DType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1 and x2", dtypeMsg.c_str(), "The dtypes of x1, x2 must be the same");
        return ge::GRAPH_FAILED;
    }

    int64_t maxLiveNodeCnt = 0;
    int64_t extraBuf = DCACHE_SIZE;

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0DType == ge::DT_INT64) {
        BroadcastBaseTiling<FloorDivOp::FloorDivIntegerS64<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling(extraBuf, maxLiveNodeCnt);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT32) {
        BroadcastBaseTiling<FloorDivOp::FloorDivIntegerS32<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling(extraBuf, maxLiveNodeCnt);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT16 || input0DType == ge::DT_BF16) {
        BroadcastBaseTiling<FloorDivOp::FloorDivFloatWithCast<half, float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<FloorDivOp::FloorDivFloatWithoutCast<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT8) {
        BroadcastBaseTiling<FloorDivOp::FloorDivIntegerU8<uint8_t, uint16_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT8) {
        BroadcastBaseTiling<FloorDivOp::FloorDivIntegerS8<int8_t, int16_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0DType).c_str(),
            "fp16, bf16, fp32, int64, int32, uint8 and int8");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus FloorDivTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t FloorDivTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus FloorDivTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FloorDivTiling::PostTiling()
{
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FloorDivTiling::GetPlatformInfo()
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

ge::graphStatus TilingForFloorDiv(gert::TilingContext* context)
{
    OP_LOGD("FloorDivTiling", "Enter TilingForFloorDiv");
    if (context == nullptr) {
        OP_LOGE("FloorDivTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const FloorDivCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc FloorDivTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForFloorDiv(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<FloorDivCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FloorDiv).Tiling(TilingForFloorDiv).TilingParse<FloorDivCompileInfo>(TilingPrepareForFloorDiv);

REGISTER_OPS_TILING_TEMPLATE(FloorDiv, FloorDivTiling, FLOOR_DIV_COMMON_TILING_PRIORITY);
} // namespace optiling
