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
 * \file  not_equal_tiling_arch35.cpp
 * \brief not_equal_tiling_arch35
 */

#include "not_equal_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "math/not_equal/op_kernel/arch35/not_equal_dag.h"
#include "math/not_equal/op_kernel/arch35/not_equal_struct.h"
#include "op_host/math_tiling_templates_registry.h"

namespace optiling {
using namespace ge;
static constexpr uint64_t NOT_EQUAL_COMMON_TILING_PRIORITY = 0;

ge::graphStatus NotEqualTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool NotEqualTiling::IsCapable()
{
    return true;
}

ge::graphStatus NotEqualTiling::DoOpTiling()
{
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType inputDType = inputDesc->GetDataType();

    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();
    if (inputDType != input1DType) {
        std::string dtypeMsg = Ops::Base::ToString(inputDType) + " and " + Ops::Base::ToString(input1DType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x1 and x2", dtypeMsg.c_str(), "The dtypes of x1 and x2 must be the same");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (inputDType == ge::DT_INT64) {
        BroadcastBaseTiling<NotEqualOp::NotEqualCompute<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_INT32) {
        BroadcastBaseTiling<NotEqualOp::NotEqualCompute<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_FLOAT16 || inputDType == ge::DT_BF16) {
        BroadcastBaseTiling<NotEqualOp::NotEqualCompute<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_FLOAT) {
        BroadcastBaseTiling<NotEqualOp::NotEqualCompute<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_UINT8) {
        BroadcastBaseTiling<NotEqualOp::NotEqualCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_INT8 || inputDType == ge::DT_BOOL) {
        BroadcastBaseTiling<NotEqualOp::NotEqualCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_UINT64) {
        BroadcastBaseTiling<NotEqualOp::NotEqualCompute<uint64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", Ops::Base::ToString(inputDType).c_str(),
            "uint64, int64, int32, float16, bf16, fp32, uint8, int8 or bool");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus NotEqualTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t NotEqualTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus NotEqualTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NotEqualTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NotEqualTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForNotEqual(gert::TilingContext* context)
{
    OP_LOGD("NotEqualTiling", "Enter TilingForNotEqual");
    if (context == nullptr) {
        OP_LOGE("NotEqualTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc NotEqualTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForNotEqual(gert::TilingParseContext* context)
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

IMPL_OP_OPTILING(NotEqual).Tiling(TilingForNotEqual).TilingParse<BroadcastCompileInfo>(TilingPrepareForNotEqual);
REGISTER_OPS_TILING_TEMPLATE(NotEqual, NotEqualTiling, NOT_EQUAL_COMMON_TILING_PRIORITY);
} // namespace optiling