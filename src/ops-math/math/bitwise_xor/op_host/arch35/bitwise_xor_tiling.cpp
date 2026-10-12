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
 * \file    bitwise_xor_tiling.cpp
 * \brief   bitwise_xor_tiling source file
 */

#include "bitwise_xor_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "op_host/math_tiling_templates_registry.h"
#include "math/bitwise_xor/op_kernel/arch35/bitwise_xor_dag.h"
#include "math/bitwise_xor/op_kernel/arch35/bitwise_xor_struct.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

static constexpr uint64_t BITWISE_XOR_COMMON_TILING_PRIORITY = 0;

ge::graphStatus BitwiseXorTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool BitwiseXorTiling::IsCapable()
{
    return true;
}

ge::graphStatus BitwiseXorTiling::DoOpTiling()
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
        BroadcastBaseTiling<BitwiseXorOp::BitwiseXorCompute<int8_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT16) {
        BroadcastBaseTiling<BitwiseXorOp::BitwiseXorCompute<int16_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT8) {
        BroadcastBaseTiling<BitwiseXorOp::BitwiseXorCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT16) {
        BroadcastBaseTiling<BitwiseXorOp::BitwiseXorCompute<uint16_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT32) {
        BroadcastBaseTiling<BitwiseXorOp::BitwiseXorCompute<int32_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT32) {
        BroadcastBaseTiling<BitwiseXorOp::BitwiseXorCompute<uint32_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_INT64) {
        BroadcastBaseTiling<BitwiseXorOp::BitwiseXorCompute<int64_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_UINT64) {
        BroadcastBaseTiling<BitwiseXorOp::BitwiseXorCompute<uint64_t>::OpDag> brcBaseTiling(context_);
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

ge::graphStatus BitwiseXorTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t BitwiseXorTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus BitwiseXorTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BitwiseXorTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BitwiseXorTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForBitwiseXor(gert::TilingContext* context)
{
    OP_LOGD("BitwiseXorTiling", "Enter TilingForBitwiseXor");
    if (context == nullptr) {
        OP_LOGE("TilingForBitwiseXor", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const BitWiseXorCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc TilingForBitwiseXor");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForBitwiseXor(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<BitWiseXorCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BitwiseXor)
    .Tiling(TilingForBitwiseXor)
    .TilingParse<BitWiseXorCompileInfo>(TilingPrepareForBitwiseXor);

REGISTER_OPS_TILING_TEMPLATE(BitwiseXor, BitwiseXorTiling, BITWISE_XOR_COMMON_TILING_PRIORITY);
} // namespace optiling