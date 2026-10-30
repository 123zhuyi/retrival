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
 * \file mod_tiling.cc
 * \brief mod_tiling source file
 */

#include "mod_tiling_arch35.h"

#include <graph/utils/type_utils.h>

#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "math/mod/op_kernel/arch35/mod_dag.h"
#include "math/mod/op_kernel/arch35/mod_struct.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"
#include "op_host/math_tiling_templates_registry.h"
using namespace ge;

namespace optiling {

using namespace Ops::Base;
constexpr static uint64_t MOD_COMMON_TILING_PRIORITY = 0;
constexpr static uint64_t DCACHE_SIZE = 32 * 1024;
ge::graphStatus ModTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool ModTiling::IsCapable()
{
    return true;
}

bool ModTiling::CheckDtype(
    const ge::DataType& inputX1Dtype, const ge::DataType& inputX2Dtype, const ge::DataType& outputDtype) const
{
    if (inputX1Dtype != inputX2Dtype || inputX1Dtype != outputDtype) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(inputX1Dtype) + ", " +
                                ge::TypeUtils::DataTypeToSerialString(inputX2Dtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1, x2 and y",
            dtypesStr.c_str(), "The dtypes of x1, x2 and y must be the same");
        return false;
    }
    return true;
}

ge::graphStatus ModTiling::DoOpTiling()
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
        BroadcastBaseTiling<ModOp::ModFloatWithCastOp<half>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        ret = brcBaseTiling.DoTiling(extraBuf, maxLiveNodeCnt);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<ModOp::ModFloatOp<float>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        ret = brcBaseTiling.DoTiling(extraBuf, maxLiveNodeCnt);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT8) {
        BroadcastBaseTiling<ModOp::ModIntWithCastOp<int8_t, int16_t>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        ret = brcBaseTiling.DoTiling(extraBuf, maxLiveNodeCnt);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_UINT8) {
        BroadcastBaseTiling<ModOp::ModIntWithCastOp<uint8_t, uint16_t>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        ret = brcBaseTiling.DoTiling(extraBuf, maxLiveNodeCnt);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT32) {
        BroadcastBaseTiling<ModOp::ModIntOp<int32_t>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        ret = brcBaseTiling.DoTiling(extraBuf, maxLiveNodeCnt);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0Dtype == ge::DT_INT64) {
        BroadcastBaseTiling<ModOp::ModIntOp<int64_t>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        extraBuf = DCACHE_SIZE; 
        ret = brcBaseTiling.DoTiling(extraBuf, 0, true);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(input0Dtype);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", dtypeStr.c_str(),
            "fp16, bf16, fp32, int8, uint8, int32 or int64");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus ModTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ModTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus ModTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ModTiling::PostTiling()
{
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ModTiling::GetPlatformInfo() {
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

ge::graphStatus TilingForMod(gert::TilingContext* context)
{
    OP_LOGD("ModTiling", "Enter TilingForMod");
    if (context == nullptr) {
        OP_LOGE("ModTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = context->GetCompileInfo<BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForMod([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(Mod, ModTiling, MOD_COMMON_TILING_PRIORITY);
IMPL_OP_OPTILING(Mod).Tiling(TilingForMod).TilingParse<BroadcastCompileInfo>(TilingPrepareForMod);
} // namespace optiling
