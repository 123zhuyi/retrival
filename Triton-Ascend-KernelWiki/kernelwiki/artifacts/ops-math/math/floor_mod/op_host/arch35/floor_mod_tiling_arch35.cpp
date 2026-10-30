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
 * \file floor_mod_tiling_arch35.cpp
 * \brief floor_mod_tiling_arch35 source file
 */

#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/floor_mod/op_kernel/arch35/floor_mod_dag.h"
#include "math/floor_mod/op_kernel/arch35/floor_mod_struct.h"
#include "floor_mod_tiling_arch35.h"

using namespace Ops::Base;

namespace optiling {

constexpr static uint64_t FLOOR_MOD_COMMON_TILING_PRIORITY = 0;
constexpr static int64_t DCACHE_SIZE = 32 * 1024;

ge::graphStatus FloorModTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool FloorModTiling::IsCapable()
{
    return true;
}

bool FloorModTiling::CheckDtype(
    const ge::DataType& inputX1Dtype, const ge::DataType& inputX2Dtype, const ge::DataType& outputDtype) const
{
    if (inputX1Dtype != inputX2Dtype || inputX1Dtype != outputDtype) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(inputX1Dtype) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(inputX2Dtype) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(outputDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1, x2 and y", dtypeMsg.c_str(), "The dtypes of x1, x2 and y must be the same");
        return false;
    }
    return true;
}

ge::graphStatus FloorModTiling::DoOpTiling()
{
    // 获取输入
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0Dtype = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1Dtype = input1Desc->GetDataType();
    // 获取输出
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    // 数据类型校验
    if (!CheckDtype(input0Dtype, input1Dtype, outputDtype)) {
        return ge::GRAPH_FAILED;
    }
    // 获取fmod额外空间和存活节点
    uint32_t maxLiveNodeCnt = 0;
    uint32_t extraBuf = 0;
    // kernel中所有输入类型均cast到float32后进入高阶API计算，因此临时buffer按float大小计算
    AscendC::GetFmodTmpBufferFactorSize(sizeof(float), maxLiveNodeCnt, extraBuf);

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0Dtype == ge::DT_FLOAT16 || input0Dtype == ge::DT_BF16) {
        BroadcastBaseTiling<FloorModOp::FloorModFloatWithCastOp<half>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        ret = brcBaseTiling.DoTiling(extraBuf, maxLiveNodeCnt, true);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<FloorModOp::FloorModFloatOp<float>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        ret = brcBaseTiling.DoTiling(extraBuf, maxLiveNodeCnt, true);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT32) {
        BroadcastBaseTiling<FloorModOp::FloorModInt32Op<int32_t>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        extraBuf += DCACHE_SIZE;
        ret = brcBaseTiling.DoTiling(extraBuf, maxLiveNodeCnt, true);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT64) {
        BroadcastBaseTiling<FloorModOp::FloorModInt64Op<int64_t>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        extraBuf += DCACHE_SIZE;
        ret = brcBaseTiling.DoTiling(extraBuf, maxLiveNodeCnt, true);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str(),
            "fp16, bf16, fp32, int32 and int64");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus FloorModTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t FloorModTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus FloorModTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FloorModTiling::PostTiling()
{
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FloorModTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FloorModCompileInfo*>(context_->GetCompileInfo());
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

ge::graphStatus TilingForFloorMod(gert::TilingContext* context)
{
    OP_LOGD("FloorModTiling", "Enter TilingForFloorMod");
    if (context == nullptr) {
        OP_LOGE("FloorModTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const FloorModCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc FloorModTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForFloorMod(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<FloorModCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FloorMod).Tiling(TilingForFloorMod).TilingParse<FloorModCompileInfo>(TilingPrepareForFloorMod);

REGISTER_OPS_TILING_TEMPLATE(FloorMod, FloorModTiling, FLOOR_MOD_COMMON_TILING_PRIORITY);
} // namespace optiling
