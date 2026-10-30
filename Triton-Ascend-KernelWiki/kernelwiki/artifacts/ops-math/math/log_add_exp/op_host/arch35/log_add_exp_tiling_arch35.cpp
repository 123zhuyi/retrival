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
 * \file log_add_exp_tiling_arch35.cpp
 * \brief log_add_exp_tiling_arch35 source file
 */
#include <cmath>
#include <limits>
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/log_add_exp/op_kernel/arch35/log_add_exp_dag.h"
#include "math/log_add_exp/op_kernel/arch35/log_add_exp_struct.h"
#include "log_add_exp_tiling_arch35.h"

namespace optiling {

constexpr static uint64_t LOG_ADD_EXP_COMMON_TILING_PRIORITY = 0;
constexpr static uint64_t FORMULA_TYPE_SIMPLIFIED = 0;
constexpr static uint64_t FORMULA_TYPE_FULL = 1;
constexpr static float DEFAULT_BASE = -1.0f;
constexpr static float DEFAULT_SCALE = 1.0f;
constexpr static float DEFAULT_SHIFT = 0.0f;

ge::graphStatus LogAddExpTiling::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        const float* basePtr = attrs->GetAttrPointer<float>(0);
        if (basePtr != nullptr) {
            base_ = *basePtr;
        }
        const float* scalePtr = attrs->GetAttrPointer<float>(1);
        if (scalePtr != nullptr) {
            scale_ = *scalePtr;
        }
        const float* shiftPtr = attrs->GetAttrPointer<float>(2);
        if (shiftPtr != nullptr) {
            shift_ = *shiftPtr;
        }
    }
    // base must be -1 (natural log) or strictly positive
    if (base_ != DEFAULT_BASE && base_ <= 0.0f) {
        OP_LOGE(context_->GetNodeName(),
                "base must be -1 (natural log) or > 0, got %f", base_);
        return ge::GRAPH_FAILED;
    }
    useFullFormula_ = !(base_ == DEFAULT_BASE && scale_ == DEFAULT_SCALE && shift_ == DEFAULT_SHIFT);
    return ge::GRAPH_SUCCESS;
}

bool LogAddExpTiling::IsCapable()
{
    return true;
}

ge::graphStatus LogAddExpTiling::CheckDtype(ge::DataType& input0Dtype)
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    input0Dtype = input0Desc->GetDataType();
    auto input1Desc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input1Desc);
    ge::DataType input1Dtype = input1Desc->GetDataType();
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    if (input0Dtype != input1Dtype || input0Dtype != outputDtype) {
        std::string reasonMsg = "The dtypes of x1(" +
                                ge::TypeUtils::DataTypeToSerialString(input0Dtype) + "), x2(" +
                                ge::TypeUtils::DataTypeToSerialString(input1Dtype) + ") and y(" +
                                ge::TypeUtils::DataTypeToSerialString(outputDtype) +
                                ") must be the same";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str(),
            reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogAddExpTiling::DoFullFormulaTiling(ge::DataType input0Dtype)
{
    float negScale = -scale_;
    float lnBase = 1.0f;
    float invLnBase = 1.0f;
    if (base_ > 0.0f) {
        lnBase = std::log(base_);
        if (lnBase == 0) {
            lnBase = 1;
        }
        invLnBase = 1.0f / lnBase;
    }

    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0Dtype == ge::DT_BF16) {
        BroadcastBaseTiling<LogAddExpOp::LogAddExpFullWithCastCompute<bfloat16_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar<float>(negScale);
        brcBaseTiling.SetScalar<float>(shift_);
        brcBaseTiling.SetScalar<float>(lnBase);
        brcBaseTiling.SetScalar<float>(invLnBase);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), FORMULA_TYPE_FULL);
    } else if (input0Dtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<LogAddExpOp::LogAddExpFullWithCastCompute<half>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar<float>(negScale);
        brcBaseTiling.SetScalar<float>(shift_);
        brcBaseTiling.SetScalar<float>(lnBase);
        brcBaseTiling.SetScalar<float>(invLnBase);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), FORMULA_TYPE_FULL);
    } else if (input0Dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<LogAddExpOp::LogAddExpFullCompute<float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.SetScalar<float>(negScale);
        brcBaseTiling.SetScalar<float>(shift_);
        brcBaseTiling.SetScalar<float>(lnBase);
        brcBaseTiling.SetScalar<float>(invLnBase);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), FORMULA_TYPE_FULL);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str(),
            "fp16, bf16, fp32");
        return ge::GRAPH_FAILED;
    }
    return ret;
}

ge::graphStatus LogAddExpTiling::DoSimplifiedTiling(ge::DataType input0Dtype)
{
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    if (input0Dtype == ge::DT_BF16) {
        BroadcastBaseTiling<LogAddExpOp::LogAddExpSimplifiedWithCastCompute<bfloat16_t>::OpDag> brcBaseTiling(
            context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), FORMULA_TYPE_SIMPLIFIED);
    } else if (input0Dtype == ge::DT_FLOAT16) {
        BroadcastBaseTiling<LogAddExpOp::LogAddExpSimplifiedCompute<half>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), FORMULA_TYPE_SIMPLIFIED);
    } else if (input0Dtype == ge::DT_FLOAT) {
        BroadcastBaseTiling<LogAddExpOp::LogAddExpSimplifiedCompute<float>::OpDag> brcBaseTiling(context_);
        ret = brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), FORMULA_TYPE_SIMPLIFIED);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x1", ge::TypeUtils::DataTypeToSerialString(input0Dtype).c_str(),
            "fp16, bf16, fp32");
        return ge::GRAPH_FAILED;
    }
    return ret;
}

ge::graphStatus LogAddExpTiling::DoOpTiling()
{
    ge::DataType input0Dtype = ge::DT_UNDEFINED;
    if (CheckDtype(input0Dtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (useFullFormula_) {
        return DoFullFormulaTiling(input0Dtype);
    }
    return DoSimplifiedTiling(input0Dtype);
}

ge::graphStatus LogAddExpTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t LogAddExpTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus LogAddExpTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogAddExpTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LogAddExpTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForLogAddExp(gert::TilingContext* context)
{
    OP_LOGD("LogAddExpTiling", "Enter TilingForLogAddExp");
    if (context == nullptr) {
        OP_LOGE("LogAddExpTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const LogAddExpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc LogAddExpTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForLogAddExp(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<LogAddExpCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LogAddExp).Tiling(TilingForLogAddExp).TilingParse<LogAddExpCompileInfo>(TilingPrepareForLogAddExp);

REGISTER_OPS_TILING_TEMPLATE(LogAddExp, LogAddExpTiling, LOG_ADD_EXP_COMMON_TILING_PRIORITY);
} // namespace optiling
