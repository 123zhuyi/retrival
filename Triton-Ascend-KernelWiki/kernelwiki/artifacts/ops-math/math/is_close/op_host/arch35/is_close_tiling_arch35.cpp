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
 * \file is_close_tiling_arch35.cpp
 * \brief is_close_tiling
 */

#include "is_close_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/is_close_dag.h"
#include "../../op_kernel/arch35/is_close_struct.h"
#include "op_host/math_tiling_templates_registry.h"

#define IS_CLOSE_TPL_NOT_EQUAL_NAN 0
#define IS_CLOSE_TPL_EQUAL_NAN 1

using namespace AscendC;
using namespace ge;
using namespace IsCloseOp;

namespace optiling {
static constexpr uint64_t IS_CLOSE_COMMON_TILING_PRIORITY = 0;

bool IsCloseTiling::IsCapable()
{
    return true;
}

ge::graphStatus IsCloseTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsCloseTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsCloseTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "IsClose DoOpTiling start.");
    auto x1Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    ge::DataType x1DType = x1Desc->GetDataType();

    auto x2Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    ge::DataType x2DType = x2Desc->GetDataType();

    auto yDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* rtolValue = attrs->GetAttrPointer<float>(0);
    const float* atolValue = attrs->GetAttrPointer<float>(1);
    const bool* equalNanValue = attrs->GetAttrPointer<bool>(2);
    float rtol_ = rtolValue != nullptr ? *rtolValue : 1e-05;
    float atol_ = atolValue != nullptr ? *atolValue : 1e-08;
    bool equalNan_ = equalNanValue != nullptr ? *equalNanValue : false;
    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;

    OP_CHECK_IF(
        x1DType != x2DType,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1, x2",
            std::string(ge::TypeUtils::DataTypeToSerialString(x1DType)) + ", " + std::string(ge::TypeUtils::DataTypeToSerialString(x2DType)),
            "The dtypes of x1 and x2 must be the same"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        x1DType != ge::DT_FLOAT16 && x1DType != ge::DT_FLOAT && x1DType != ge::DT_INT32 && x1DType != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1",
            ge::TypeUtils::DataTypeToSerialString(x1DType),
            "FLOAT16, FLOAT, INT32, BF16"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        rtol_ < 0 || atol_ < 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context_->GetNodeName(), "rtol, atol",
            std::to_string(rtol_) + ", " + std::to_string(atol_),
            "The values of rtol and atol must be greater than or equal to 0"),
        return ge::GRAPH_FAILED);

    if (x1DType == ge::DT_FLOAT16 && equalNan_ == false) {
        OP_LOGD(context_->GetNodeName(), "Data type is float16, equal_nan is false.");
        Ops::Base::BroadcastBaseTiling<IsCloseDag<Ops::Base::half>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "broadcastBaseTiling doTiling failed"), return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), IS_CLOSE_TPL_NOT_EQUAL_NAN);
        brcBaseTiling.SetScalar<float>(rtol_);
        brcBaseTiling.SetScalar<float>(atol_);
    } else if (x1DType == ge::DT_FLOAT && equalNan_ == false) {
        OP_LOGD(context_->GetNodeName(), "Data type is float, equal_nan is false.");
        Ops::Base::BroadcastBaseTiling<IsCloseDag<float>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "broadcastBaseTiling doTiling failed"), return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), IS_CLOSE_TPL_NOT_EQUAL_NAN);
        brcBaseTiling.SetScalar<float>(rtol_);
        brcBaseTiling.SetScalar<float>(atol_);
    } else if (x1DType == ge::DT_INT32 && equalNan_ == false) {
        OP_LOGD(context_->GetNodeName(), "Data type is int32, equal_nan is false.");
        Ops::Base::BroadcastBaseTiling<IsCloseDag<int32_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "broadcastBaseTiling doTiling failed"), return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), IS_CLOSE_TPL_NOT_EQUAL_NAN);
        brcBaseTiling.SetScalar<float>(rtol_);
        brcBaseTiling.SetScalar<float>(atol_);
    } else if (x1DType == ge::DT_BF16 && equalNan_ == false) {
        OP_LOGD(context_->GetNodeName(), "Data type is bfloat16, equal_nan is false.");
        Ops::Base::BroadcastBaseTiling<IsCloseDag<Ops::Base::bfloat16_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "broadcastBaseTiling doTiling failed"), return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), IS_CLOSE_TPL_NOT_EQUAL_NAN);
        brcBaseTiling.SetScalar<float>(rtol_);
        brcBaseTiling.SetScalar<float>(atol_);
    } else if (x1DType == ge::DT_FLOAT16 && equalNan_ == true) {
        OP_LOGD(context_->GetNodeName(), "Data type is float16, equal_nan is true.");
        Ops::Base::BroadcastBaseTiling<IsCloseEqualNanDag<Ops::Base::half>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "broadcastBaseTiling doTiling failed"), return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), IS_CLOSE_TPL_EQUAL_NAN);
        brcBaseTiling.SetScalar<float>(rtol_);
        brcBaseTiling.SetScalar<float>(atol_);
    } else if (x1DType == ge::DT_FLOAT && equalNan_ == true) {
        OP_LOGD(context_->GetNodeName(), "Data type is float, equal_nan is true.");
        Ops::Base::BroadcastBaseTiling<IsCloseEqualNanDag<float>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "broadcastBaseTiling doTiling failed"), return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), IS_CLOSE_TPL_EQUAL_NAN);
        brcBaseTiling.SetScalar<float>(rtol_);
        brcBaseTiling.SetScalar<float>(atol_);
    } else if (x1DType == ge::DT_INT32 && equalNan_ == true) {
        OP_LOGD(context_->GetNodeName(), "Data type is int32, equal_nan is true.");
        Ops::Base::BroadcastBaseTiling<IsCloseEqualNanDag<int32_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "broadcastBaseTiling doTiling failed"), return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), IS_CLOSE_TPL_EQUAL_NAN);
        brcBaseTiling.SetScalar<float>(rtol_);
        brcBaseTiling.SetScalar<float>(atol_);
    } else if (x1DType == ge::DT_BF16 && equalNan_ == true) {
        OP_LOGD(context_->GetNodeName(), "Data type is bfloat16, equal_nan is true.");
        Ops::Base::BroadcastBaseTiling<IsCloseEqualNanDag<Ops::Base::bfloat16_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "broadcastBaseTiling doTiling failed"), return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), IS_CLOSE_TPL_EQUAL_NAN);
        brcBaseTiling.SetScalar<float>(rtol_);
        brcBaseTiling.SetScalar<float>(atol_);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(x1DType), "FLOAT16, FLOAT, INT32, BF16");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "[IsCloseTilingData] : tilingKey=%lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsCloseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t IsCloseTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus IsCloseTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsCloseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4IsClose(gert::TilingContext* context)
{
    OP_LOGD("IsCloseTiling", "Enter Tiling4IsClose");
    if (context == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE("Tiling4IsClose", "context", "nullptr", "not nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const Ops::Base::BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc Tiling4IsClose");
    IsCloseTiling tiling(context);
    return tiling.DoTiling();
}

// 临时copy，应该是公共函数
ge::graphStatus TilingPrepareForIsClose(gert::TilingParseContext* context)
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

IMPL_OP_OPTILING(IsClose)
    .Tiling(Tiling4IsClose)
    .TilingParse<Ops::Base::BroadcastCompileInfo>(TilingPrepareForIsClose);

REGISTER_OPS_TILING_TEMPLATE(IsClose, IsCloseTiling, IS_CLOSE_COMMON_TILING_PRIORITY);
} // namespace optiling