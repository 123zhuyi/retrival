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
 * \file    bitwise_or_tiling.cpp
 * \brief   bitwise_or_tiling source file
 */

#include "bitwise_or_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "op_host/math_tiling_templates_registry.h"
#include "math/bitwise_or/op_kernel/arch35/bitwise_or_dag.h"
#include "math/bitwise_or/op_kernel/arch35/bitwise_or_struct.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

static constexpr uint64_t BITWISE_OR_COMMON_TILING_PRIORITY = 0;

ge::graphStatus BitwiseOrTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool BitwiseOrTiling::IsCapable()
{
    return true;
}

ge::graphStatus BitwiseOrTiling::DoOpTiling()
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
    if (!((input0DType == input1DType) && (input0DType == outputDtype))) {
        std::string reasonMsg = "The dtype of x1 must be the same as the dtypes " +
                                ge::TypeUtils::DataTypeToSerialString(input1DType) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype) + " of x2 and y";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0DType).c_str(),
            reasonMsg.c_str());
         return ge::GRAPH_FAILED;
    }
    if (input0DType == ge::DT_INT8) {
        BroadcastBaseTiling<BitwiseOrOp::BitwiseOrCompute<int8_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT16) {
        BroadcastBaseTiling<BitwiseOrOp::BitwiseOrCompute<int16_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT8) {
        BroadcastBaseTiling<BitwiseOrOp::BitwiseOrCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT16) {
        BroadcastBaseTiling<BitwiseOrOp::BitwiseOrCompute<uint16_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT32) {
        BroadcastBaseTiling<BitwiseOrOp::BitwiseOrCompute<int32_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT32) {
        BroadcastBaseTiling<BitwiseOrOp::BitwiseOrCompute<uint32_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT64) {
        BroadcastBaseTiling<BitwiseOrOp::BitwiseOrCompute<int64_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT64) {
        BroadcastBaseTiling<BitwiseOrOp::BitwiseOrCompute<uint64_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0DType).c_str(),
            "int8, uint8, int16, uint16, int32, uint32, int64 or uint64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BitwiseOrTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t BitwiseOrTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus BitwiseOrTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BitwiseOrTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BitwiseOrTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForBitwiseOr(gert::TilingContext* context)
{
    OP_LOGD("BitwiseOrTiling", "Enter TilingForBitwiseOr");
    if (context == nullptr) {
        OP_LOGE("TilingForBitwiseOr", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const BitWiseOrCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc TilingForBitwiseOr");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForBitwiseOr(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<BitWiseOrCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BitwiseOr)
    .Tiling(TilingForBitwiseOr)
    .TilingParse<BitWiseOrCompileInfo>(TilingPrepareForBitwiseOr);

REGISTER_OPS_TILING_TEMPLATE(BitwiseOr, BitwiseOrTiling, BITWISE_OR_COMMON_TILING_PRIORITY);
} // namespace optiling