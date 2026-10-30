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
 * \file truncate_div_tiling_arch35.cpp
 * \brief truncate_div_tiling source file
 */

#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/truncate_div/op_kernel/arch35/truncate_div_dag.h"
#include "math/truncate_div/op_kernel/arch35/truncate_div_struct.h"
#include "truncate_div_tiling_arch35.h"
#include "op_host/util/fp16.h"
#include "util/bfloat16.h"

using namespace Ops::Base;
using namespace ge;

namespace optiling {

constexpr static uint64_t TRUNCATE_DIV_COMMON_TILING_PRIORITY = 0;
constexpr static uint32_t INPUT_IDX_X1 = 0;
constexpr static uint32_t INPUT_IDX_X2 = 1;
constexpr static int64_t DCACHE_SIZE = 32 * 1024;

ge::graphStatus TruncateDivTiling::GetShapeAttrsInfo() { return ge::GRAPH_SUCCESS; }

bool TruncateDivTiling::IsCapable() { return true; }

float TruncateDivTiling::GetReciprocal(float data)
{
    float invScalarVal = 0.0f;
    float scalarValue = data;
    if (scalarValue == 0.0f) {
        float reciprocalZero = 1.0f / scalarValue;
        if (reciprocalZero < 0) {
            invScalarVal = -INFINITY;
        } else {
            invScalarVal = INFINITY;
        }
    } else if (scalarValue == INFINITY) {
        invScalarVal = 0.0f;
    } else if (scalarValue == -INFINITY) {
        invScalarVal = -0.0f;
    } else {
        invScalarVal = 1.0f / scalarValue;
        invScalarVal = invScalarVal * (2.0f - scalarValue * invScalarVal);
    }
    return invScalarVal;
}

template <typename T>
ge::graphStatus TruncateDivTiling::GetConstData(uint32_t inputIdx, T& data)
{
    auto tensor = context_->GetInputTensor(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensor);
    const T* value = tensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, value);
    data = value[0];
    OP_LOGI(context_->GetNodeName(), "scalarData %f", data);
    return ge::GRAPH_SUCCESS;
}

template <typename OpDag>
ge::graphStatus TruncateDivTiling::ExecTiling(bool isScalarBranch)
{
    BroadcastBaseTiling<OpDag> brcTiling(context_);
    if (isScalarBranch) {
        brcTiling.SetScalar(reciprocal_);
    }

    auto ret = brcTiling.DoTiling();
    schMode_ = brcTiling.GetSchMode();
    tilingKey_ = GET_TPL_TILING_KEY(schMode_, canUseMul_);

    return ret;
}

ge::graphStatus TruncateDivTiling::GetScalarReciprocal(ge::DataType x2DType)
{
    bool success = false;
    float scalarValue = 0.0f;

    switch (x2DType) {
        case ge::DT_FLOAT: {
            success = (GetConstData<float>(INPUT_IDX_X2, scalarValue) == ge::GRAPH_SUCCESS);
            break;
        }
        case ge::DT_FLOAT16: {
            uint16_t tmpValue = 0;
            success = (GetConstData<uint16_t>(INPUT_IDX_X2, tmpValue) == ge::GRAPH_SUCCESS);
            if (success) {
                scalarValue = float(fp16_t(tmpValue));
            }
            break;
        }
        case ge::DT_BF16: {
            uint16_t tmpValue = 0;
            success = (GetConstData<uint16_t>(INPUT_IDX_X2, tmpValue) == ge::GRAPH_SUCCESS);
            if (success) {
                scalarValue = float(bfloat16(tmpValue, bfloat16::from_bits()));
            }
            break;
        }
        default: {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "x2", ToString(x2DType).c_str(),
                "The dtype of x2 must be within the range DT_FLOAT, DT_FLOAT16 and DT_BF16");
            return ge::GRAPH_FAILED;
        }
    }

    if (!success) {
        return ge::GRAPH_FAILED;
    }

    reciprocal_ = GetReciprocal(scalarValue);
    OP_LOGI(context_->GetNodeName(), "scalar value = %f, reciprocal value = %f", scalarValue, reciprocal_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncateDivTiling::HandleFloat16WithFloat()
{
    return canUseMul_ ? ExecTiling<TruncateDivOp::TruncateDivFloatWithCastScalar<half, float, float>::OpDag>(true) :
                        ExecTiling<TruncateDivOp::TruncateDivFloatWithCast<half, float, float>::OpDag>(false);
}

ge::graphStatus TruncateDivTiling::HandleFloat16OrBf16()
{
    return canUseMul_ ? ExecTiling<TruncateDivOp::TruncateDivFloat16Scalar<half, float>::OpDag>(true) :
                        ExecTiling<TruncateDivOp::TruncateDivFloat16<half, float>::OpDag>(false);
}

ge::graphStatus TruncateDivTiling::HandleFloat(ge::DataType x2DType)
{
    if (x2DType == ge::DT_FLOAT) {
        return canUseMul_ ? ExecTiling<TruncateDivOp::TruncateDivFloatScalar<float>::OpDag>(true) :
                            ExecTiling<TruncateDivOp::TruncateDivFloat<float>::OpDag>(false);
    }
    if (x2DType == ge::DT_INT32) {
        return ExecTiling<TruncateDivOp::TruncateDivFloatToLowBit<float, int32_t, float>::OpDag>(false);
    }
    if (x2DType == ge::DT_FLOAT16) {
        return canUseMul_ ? ExecTiling<TruncateDivOp::TruncateDivFloatScalar<float>::OpDag>(true) :
                            ExecTiling<TruncateDivOp::TruncateDivFloatToLowBit<float, half, float>::OpDag>(false);
    }
    return ge::GRAPH_PARAM_INVALID;
}

ge::graphStatus TruncateDivTiling::HandleIntTypes(ge::DataType x1DType, ge::DataType x2DType)
{
    if (x1DType != x2DType) {
        return ge::GRAPH_PARAM_INVALID;
    }

    if (x1DType == ge::DT_INT8) {
        return ExecTiling<TruncateDivOp::TruncateDivIntS8<int8_t, half>::OpDag>(false);
    }
    if (x1DType == ge::DT_UINT8) {
        return ExecTiling<TruncateDivOp::TruncateDivIntU8<uint8_t, uint16_t>::OpDag>(false);
    }
    if (x1DType == ge::DT_INT16) {
        return ExecTiling<TruncateDivOp::TruncateDivInt<int16_t>::OpDag>(false);
    }
    if (x1DType == ge::DT_INT32) {
        return ExecTiling<TruncateDivOp::TruncateDivInt<int32_t>::OpDag>(false);
    }

    return ge::GRAPH_PARAM_INVALID;
}

ge::graphStatus TruncateDivTiling::HandleInt64()
{
    int64_t maxLiveNodeCnt = 0;
    int64_t extraBuf = DCACHE_SIZE;
    BroadcastBaseTiling<TruncateDivOp::TruncateDivInt64<int64_t>::OpDag> brcTiling(context_);
    auto ret = brcTiling.DoTiling(extraBuf, maxLiveNodeCnt);
    schMode_ = brcTiling.GetSchMode();
    tilingKey_ = GET_TPL_TILING_KEY(schMode_, canUseMul_);
    return ret;
}

ge::graphStatus TruncateDivTiling::SelectAndExecTiling(ge::DataType x1DType, ge::DataType x2DType)
{
    ge::graphStatus ret = ge::GRAPH_PARAM_INVALID;

    if (x1DType == ge::DT_FLOAT16 && x2DType == ge::DT_FLOAT) {
        return HandleFloat16WithFloat();
    }

    if ((x1DType == ge::DT_FLOAT16 || x1DType == ge::DT_BF16) && x2DType == x1DType) {
        return HandleFloat16OrBf16();
    }

    if (x1DType == ge::DT_FLOAT) {
        ret = HandleFloat(x2DType);
        if (ret != ge::GRAPH_PARAM_INVALID) {
            return ret;
        }
    }

    if (x1DType == ge::DT_INT32 && x2DType == ge::DT_FLOAT) {
        return ExecTiling<TruncateDivOp::TruncateDivIntToFloat<int32_t, float, float>::OpDag>(false);
    }

    ret = HandleIntTypes(x1DType, x2DType);
    if (ret != ge::GRAPH_PARAM_INVALID) {
        return ret;
    }

    if (x1DType == ge::DT_INT64) {
        return HandleInt64();
    }

    std::string errorDtype = ToString(x1DType) + ", " + ToString(x2DType);
    std::string errorMsg =
        std::string("The dtypes of these parameters support only the following combinations: ") +
        "((DT_FLOAT16, DT_FLOAT), (DT_FLOAT16, DT_BF16), all DT_FLOAT, (DT_FLOAT, DT_INT32), (DT_FLOAT, DT_FLOAT16), " +
        "all DT_INT8, all DT_UINT8, all DT_INT16, all DT_INT32, all DT_INT64 and (DT_INT32, DT_FLOAT))";
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1, x2", errorDtype.c_str(), errorMsg.c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus TruncateDivTiling::DoOpTiling()
{
    auto x1Desc = context_->GetInputDesc(INPUT_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    auto x2Desc = context_->GetInputDesc(INPUT_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);

    ge::DataType x1DType = x1Desc->GetDataType();
    ge::DataType x2DType = x2Desc->GetDataType();

    auto x1StorageShape = context_->GetInputShape(INPUT_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x1StorageShape);
    auto x2StorageShape = context_->GetInputShape(INPUT_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x2StorageShape);

    auto x2Shape = x2StorageShape->GetStorageShape();
    bool isScalar = x2Shape.IsScalar();
    canUseMul_ = isScalar && (x2DType == ge::DT_FLOAT || x2DType == ge::DT_FLOAT16 || x2DType == ge::DT_BF16);

    OP_LOGI(context_->GetNodeName(), "canUseMul_ %d", canUseMul_);
    if (canUseMul_) {
        auto ret = GetScalarReciprocal(x2DType);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }

    auto ret = SelectAndExecTiling(x1DType, x2DType);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    OP_LOGI(context_, "TruncateDiv tiling completed: tilingKey_=%lu, schMode_=%u, canUseMul_=%d, reciprocal %f",
            tilingKey_, schMode_, static_cast<int>(canUseMul_), reciprocal_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncateDivTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t TruncateDivTiling::GetTilingKey() const { return tilingKey_; }

ge::graphStatus TruncateDivTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus TruncateDivTiling::PostTiling()
{
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TruncateDivTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<BroadcastCompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfoPtr);
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

static ge::graphStatus TilingForTruncateDiv(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    auto compileInfo = context->GetCompileInfo<BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc TruncateDivTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForTruncateDiv([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TruncateDiv)
    .Tiling(TilingForTruncateDiv)
    .TilingInputsDataDependency({INPUT_IDX_X2})
    .TilingParse<BroadcastCompileInfo>(TilingPrepareForTruncateDiv);

REGISTER_OPS_TILING_TEMPLATE(TruncateDiv, TruncateDivTiling, TRUNCATE_DIV_COMMON_TILING_PRIORITY);
} // namespace optiling
