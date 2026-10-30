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
 * \file    less_equal_tiling_arch35.cpp
 * \brief   less_equal_tiling_arch35 source file
 */

#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/less_equal/op_kernel/arch35/less_equal_dag.h"
#include "math/less_equal/op_kernel/arch35/less_equal_struct.h"
#include "less_equal_tiling_arch35.h"

namespace optiling {

static constexpr uint64_t LESS_EQUAL_COMMON_TILING_PRIORITY = 0;

ge::graphStatus LessEqualTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool LessEqualTiling::IsCapable()
{
    return true;
}

ge::graphStatus LessEqualTiling::DoOpTiling()
{
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType inputDType = inputDesc->GetDataType();

    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1DType = input1Desc->GetDataType();
    if (inputDType != input1DType) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(inputDType) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(input1DType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1 and x2",
            dtypesStr.c_str(), "The dtypes of x1 and x2 must be the same");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (inputDType == ge::DT_INT64) {
        BroadcastBaseTiling<LessEqualOp::LessEqualCompute<int64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_INT32) {
        BroadcastBaseTiling<LessEqualOp::LessEqualCompute<int32_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_FLOAT16 || inputDType == ge::DT_BF16) {
        BroadcastBaseTiling<LessEqualOp::LessEqualCompute<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_FLOAT) {
        BroadcastBaseTiling<LessEqualOp::LessEqualCompute<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_UINT8) {
        BroadcastBaseTiling<LessEqualOp::LessEqualCompute<uint8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_INT8 || inputDType == ge::DT_BOOL) {
        BroadcastBaseTiling<LessEqualOp::LessEqualCompute<int8_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_UINT64) {
        BroadcastBaseTiling<LessEqualOp::LessEqualCompute<uint64_t>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(inputDType);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x1", dtypeStr.c_str(),
            "uint64, int64, int32, float16, bf16, fp32, uint8, int8 or bool");
        return ge::GRAPH_FAILED;
    }

    return ret;
}

ge::graphStatus LessEqualTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LessEqualTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus LessEqualTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LessEqualTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LessEqualTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4LessEqual(gert::TilingContext* context)
{
    OP_LOGD("LessEqualTiling", "Enter Tiling4LessEqual");
    if (context == nullptr) {
        OP_LOGE("Tiling4LessEqual", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const LessEqualCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc Tiling4LessEqual");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForLessEqual(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<LessEqualCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LessEqual).Tiling(Tiling4LessEqual).TilingParse<LessEqualCompileInfo>(TilingPrepareForLessEqual);

REGISTER_OPS_TILING_TEMPLATE(LessEqual, LessEqualTiling, LESS_EQUAL_COMMON_TILING_PRIORITY);
} // namespace optiling