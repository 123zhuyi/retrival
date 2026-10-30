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
 * \file right_shift_tiling_arch35.cpp
 * \brief right_shift_tiling source file
 */

#include "right_shift_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "op_host/math_tiling_templates_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_info.h"
#include "math/right_shift/op_kernel/arch35/right_shift_dag.h"
#include "math/right_shift/op_kernel/arch35/right_shift_struct.h"

using namespace ge;
using namespace RightShiftOp;
using namespace Ops::Base;

namespace optiling {

static constexpr uint64_t RIGHT_SHIFT_COMMON_TILING_PRIORITY = 0;

bool RightShiftTiling::CheckDtype(
    const ge::DataType& xDtype, const ge::DataType& yDtype, const ge::DataType& zDtype) const
{
    if (xDtype != yDtype || xDtype != zDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x, y, z",
            std::string(ge::TypeUtils::DataTypeToSerialString(xDtype)) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(yDtype)) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(zDtype)),
            "The dtypes of x, y and z must be the same");
        return false;
    }
    return true;
}

bool RightShiftTiling::IsCapable()
{
    return true;
}

ge::graphStatus RightShiftTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RightShiftTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RightShiftTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "RightShift DoOpTiling start.");
    auto xDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDType = xDesc->GetDataType();

    auto yDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    ge::DataType yDType = yDesc->GetDataType();

    auto zDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, zDesc);
    ge::DataType zDType = zDesc->GetDataType();
    if (!CheckDtype(xDType, yDType, zDType)) {
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;

    OP_LOGD(context_->GetNodeName(), "Current data type is %s.", ge::TypeUtils::DataTypeToSerialString(xDType).c_str());
    if (xDType == ge::DT_INT8) {
        BroadcastBaseTiling<RightShiftDag8<int8_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), TPL_INT8);
    } else if (xDType == ge::DT_UINT8) {
        BroadcastBaseTiling<RightShiftDag8<uint8_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), TPL_UINT8);
    } else if (xDType == ge::DT_INT16) {
        BroadcastBaseTiling<RightShiftDag16<int16_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), TPL_INT16);
    } else if (xDType == ge::DT_UINT16) {
        BroadcastBaseTiling<RightShiftDag16<uint16_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), TPL_UINT16);
    } else if (xDType == ge::DT_INT32) {
        BroadcastBaseTiling<RightShiftDag32<int32_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), TPL_INT32);
    } else if (xDType == ge::DT_UINT32) {
        BroadcastBaseTiling<RightShiftDag32<uint32_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), TPL_UINT32);
    } else if (xDType == ge::DT_INT64) {
        BroadcastBaseTiling<RightShiftDag64<int64_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), TPL_INT64);
    } else if (xDType == ge::DT_UINT64) {
        BroadcastBaseTiling<RightShiftDag64<uint64_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), TPL_UINT64);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x",
            ge::TypeUtils::DataTypeToSerialString(xDType), "INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(context_->GetNodeName(), "broadcastBaseTiling doTiling failed."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "[RightShiftTilingData] : tilingKey=%lu", tilingKey);
    return baseTilingResult;
}

ge::graphStatus RightShiftTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t RightShiftTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus RightShiftTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RightShiftTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4RightShift(gert::TilingContext* context)
{
    OP_LOGD("RightShiftTiling", "Enter Tiling4RightShift");
    if (context == nullptr) {
        OP_LOGE("Tiling4RightShift", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc Tiling4RightShift");
    RightShiftTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForRightShift([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RightShift).Tiling(Tiling4RightShift).TilingParse<BroadcastCompileInfo>(TilingPrepareForRightShift);

} // namespace optiling
