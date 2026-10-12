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
 * \file clip_by_value_tiling.cpp
 * \brief clip_by_value tiling source file
 */

#include "clip_by_value_tiling.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/math_tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "conversion/clip_by_value/op_kernel/arch35/clip_by_value_dag.h"
#include "conversion/clip_by_value/op_kernel/arch35/clip_by_value_struct.h"

namespace optiling {
using namespace ge;
using namespace Ops::Base;

constexpr static uint64_t CLIP_BY_VALUE_COMMON_TILING_PRIORITY = 0;

ge::graphStatus ClipByValueTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool ClipByValueTiling::IsCapable()
{
    return true;
}

bool ClipByValueTiling::CheckDtype(
    const ge::DataType& xDtype, const ge::DataType& minDtype, const ge::DataType& maxDtype,
    const ge::DataType& yDtype) const
{
    if (xDtype != minDtype || xDtype != maxDtype || xDtype != yDtype) {
        std::string dtypeMsg = ge::TypeUtils::DataTypeToSerialString(xDtype) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(minDtype) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(maxDtype) + " and " +
                               ge::TypeUtils::DataTypeToSerialString(yDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x, clip_value_min, clip_value_max and y", dtypeMsg.c_str(),
            "The dtype of x, clip_value_min, clip_value_max and y must be the same");
        return false;
    }
    return true;
}

ge::graphStatus ClipByValueTiling::DoOpTiling()
{
    auto xDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDtype = xDesc->GetDataType();

    auto minDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, minDesc);
    ge::DataType minDtype = minDesc->GetDataType();

    auto maxDesc = context_->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxDesc);
    ge::DataType maxDtype = maxDesc->GetDataType();

    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    ge::DataType yDtype = yDesc->GetDataType();
    if (!CheckDtype(xDtype, minDtype, maxDtype, yDtype)) {
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (xDtype == ge::DT_FLOAT16 || xDtype == ge::DT_BF16) {
        BroadcastBaseTiling<ClipByValueOp::ClipByValueCompute<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (xDtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<ClipByValueOp::ClipByValueCompute<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (xDtype == ge::DT_INT32) {
        BroadcastBaseTiling<ClipByValueOp::ClipByValueCompute<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (xDtype == ge::DT_INT64) {
        BroadcastBaseTiling<ClipByValueOp::ClipByValueCompute<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
            "fp16, bf16, fp32, int32 or int64");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus ClipByValueTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ClipByValueTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus ClipByValueTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClipByValueTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForClipByValue(gert::TilingContext* context)
{
    OP_LOGD("ClipByValueTiling", "Enter TilingForClipByValue");
    if (context == nullptr) {
        OP_LOGE("TilingForClipByValue", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = context->GetCompileInfo<ClipByValueCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForClipByValue(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForClipByValue", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto compileInfoPtr = context->GetCompiledInfo<ClipByValueCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ClipByValue)
    .Tiling(TilingForClipByValue)
    .TilingParse<ClipByValueCompileInfo>(TilingPrepareForClipByValue);

REGISTER_OPS_TILING_TEMPLATE(ClipByValue, ClipByValueTiling, CLIP_BY_VALUE_COMMON_TILING_PRIORITY);
} // namespace optiling
