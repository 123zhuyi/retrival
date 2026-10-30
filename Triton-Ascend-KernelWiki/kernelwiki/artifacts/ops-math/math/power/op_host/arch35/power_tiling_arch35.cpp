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
 * \file power_tiling_arch35.cpp
 * \brief Power 算子 Tiling 实现：在 host 侧完成 power/scale/shift 三个标量属性的
 *        全部分支决策，将运行时分支前移到 tilingKey，kernel 端无需再做条件判断。
 */
#include "power_tiling_arch35.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "op_host/tiling_base_util.h"
#include "platform/platform_info.h"
#include "math/power/op_kernel/arch35/power_dag.h"
#include "math/power/op_kernel/arch35/power_struct.h"
#include "math/power/op_kernel/arch35/power_tiling_struct.h"

namespace optiling {
using namespace PowerOp;

// 算子默认 workspace 大小：16 MB，与同仓 elementwise 类算子保持一致。
static const size_t POWER_ASCEND_WORKSPACE = 16 * 1024 * 1024;
// isclose 判等使用的绝对/相对容差，与 math/is_close 默认参数对齐。
static constexpr float POWER_ISCLOSE_ATOL = 1e-8f;
static constexpr float POWER_ISCLOSE_RTOL = 1e-5f;

// 浮点近似相等：|a - b| <= atol + rtol * |b|。
// 用于在 host 端容忍浮点误差地判断 power/scale/shift 是否落到 0/1/2/3 等关键点。
bool PowerTiling::IsCloseScalar(float a, float b)
{
    return std::fabs(a - b) <= POWER_ISCLOSE_ATOL + POWER_ISCLOSE_RTOL * std::fabs(b);
}

// 判断 v 是否为整数（有限且 v == floor(v)）。
// 用于决定负底数 + power 的整数性，进而选择 (-1)^power 修正还是异常值。
bool PowerTiling::IsInteger(float v)
{
    if (!std::isfinite(v)) {
        return false;
    }
    return std::fabs(v - std::floor(v)) <= POWER_ISCLOSE_ATOL;
}

// 校验输入 x 的 dtype，仅支持 fp16 / bf16 / fp32。
ge::graphStatus PowerTiling::CalcInputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    this->inputDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        this->inputDtype != ge::DT_FLOAT16 && this->inputDtype != ge::DT_BF16 && this->inputDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            tilingContext->GetNodeName(),
            "x",
            Ops::Base::ToString(this->inputDtype).c_str(),
            "The dtype of x must be within the range [DT_FLOAT16, DT_BF16, DT_FLOAT]."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 校验输出 y 的 dtype 必须与输入 x 完全一致（无类型提升）。
ge::graphStatus PowerTiling::CalcOutputDtype()
{
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        this->outputDtype != this->inputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            tilingContext->GetNodeName(),
            "x, y",
            (Ops::Base::ToString(this->inputDtype) + ", " + Ops::Base::ToString(this->outputDtype)).c_str(),
            "The dtypes of x and y must be the same."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 校验输入与输出 shape 完全相等（elementwise 算子不支持广播）。
// EnsureNotScalar 把 0 维标量 shape 视为 [1]，避免后续 tiling 误判。
ge::graphStatus PowerTiling::CheckShape()
{
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& xShape = Ops::Base::EnsureNotScalar(inputStorageShape->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& yShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            tilingContext->GetNodeName(),
            "x, y",
            (Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(yShape)).c_str(),
            "The shapes of x and y must be the same."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 读取算子属性：power / scale / shift（与 op_def 注册顺序一致）。
// 三者皆为 OPTIONAL 浮点属性，缺省时使用算子定义中的默认值（1.0 / 1.0 / 0.0）。
ge::graphStatus PowerTiling::SetAttr()
{
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);

    const float* powerPtr = attrs->GetAttrPointer<float>(0);
    this->attrPower = powerPtr != nullptr ? *powerPtr : 1.0f;

    const float* scalePtr = attrs->GetAttrPointer<float>(1);
    this->attrScale = scalePtr != nullptr ? *scalePtr : 1.0f;

    const float* shiftPtr = attrs->GetAttrPointer<float>(2);
    this->attrShift = shiftPtr != nullptr ? *shiftPtr : 0.0f;

    OP_LOGD(
        tilingContext->GetNodeName(), "Power attrs: power=%f, scale=%f, shift=%f",
        this->attrPower, this->attrScale, this->attrShift);
    return ge::GRAPH_SUCCESS;
}

// -----------------------------------------------------------------------------
// 模板辅助函数：根据 dtype 分发 DoTiling 调用（单模板参数 DAG）。
// -----------------------------------------------------------------------------
template<template<typename> class DagT>
ge::graphStatus DispatchTilingByDtype(
    ElewiseBaseTiling& tiling, ge::DataType dtype, Ops::Base::EleBaseTilingData& baseTiling)
{
    if (dtype == ge::DT_FLOAT16) {
        return tiling.DoTiling<typename DagT<half>::OpDag>(baseTiling);
    } else if (dtype == ge::DT_BF16) {
        return tiling.DoTiling<typename DagT<bfloat16_t>::OpDag>(baseTiling);
    } else {
        return tiling.DoTiling<typename DagT<float>::OpDag>(baseTiling);
    }
}

// -----------------------------------------------------------------------------
// 模板辅助函数：根据 dtype 分发 DoTiling 调用（双模板参数 DAG，用于 GENERIC_POW）。
// -----------------------------------------------------------------------------
template<template<typename, int> class DagT, int PowSign>
ge::graphStatus DispatchTilingByDtypeGeneric(
    ElewiseBaseTiling& tiling, ge::DataType dtype, Ops::Base::EleBaseTilingData& baseTiling)
{
    if (dtype == ge::DT_FLOAT16) {
        return tiling.DoTiling<typename DagT<half, PowSign>::OpDag>(baseTiling);
    } else if (dtype == ge::DT_BF16) {
        return tiling.DoTiling<typename DagT<bfloat16_t, PowSign>::OpDag>(baseTiling);
    } else {
        return tiling.DoTiling<typename DagT<float, PowSign>::OpDag>(baseTiling);
    }
}

// -----------------------------------------------------------------------------
// 根据 culType 分发到对应的 DAG 进行 tiling，内部处理 dtype 映射。
// -----------------------------------------------------------------------------
ge::graphStatus PowerTiling::DispatchTilingByCulType(ElewiseBaseTiling& tiling, PowerOp::PowerTilingData* data)
{
    switch (culType) {
        case CulTypeEnum::ALL_ZEROS:
            return DispatchTilingByDtype<PowerAllZerosDag>(tiling, outputDtype, data->baseTiling);
        case CulTypeEnum::BROADCAST_SCALAR:
            return DispatchTilingByDtype<PowerBcastScalarDag>(tiling, outputDtype, data->baseTiling);
        case CulTypeEnum::LINEAR:
            return DispatchTilingByDtype<PowerLinearDag>(tiling, outputDtype, data->baseTiling);
        case CulTypeEnum::SQUARE:
            return DispatchTilingByDtype<PowerSquareDag>(tiling, outputDtype, data->baseTiling);
        case CulTypeEnum::CUBE:
            return DispatchTilingByDtype<PowerCubeDag>(tiling, outputDtype, data->baseTiling);
        case CulTypeEnum::GENERIC_POW_POS:
            return DispatchTilingByDtypeGeneric<PowerGenericDag, 1>(tiling, outputDtype, data->baseTiling);
        case CulTypeEnum::GENERIC_POW_NEG:
            return DispatchTilingByDtypeGeneric<PowerGenericDag, 0>(tiling, outputDtype, data->baseTiling);
        default:
            OP_LOGE(tilingContext->GetNodeName(), "unknown culType=%lu", culTypeKey);
            return ge::GRAPH_FAILED;
    }
}

// -----------------------------------------------------------------------------
// 根据 power/scale/shift 决策 culType 与 kernel 标量，所有分支前移至 tilingKey。
// -----------------------------------------------------------------------------
bool PowerTiling::DecideCulType()
{
    const bool powerIsInt = IsInteger(this->attrPower);
    const float scaleTimesPow = this->attrScale * this->attrPower;
    const float kNaN = std::numeric_limits<float>::quiet_NaN();
    const float kInfPos = std::numeric_limits<float>::infinity();

    // 分支1：scale*power≈0，输出退化为标量
    if (IsCloseScalar(scaleTimesPow, 0.0f)) {
        if (IsCloseScalar(this->attrPower, 0.0f)) {
            scalar0 = 1.0f;
            culType = CulTypeEnum::BROADCAST_SCALAR;
            return true;
        }
        // shift 符号决定 pow(shift, power) 取值
        if (this->attrShift < 0.0f) {
            scalar0 = powerIsInt ? std::pow(this->attrShift, this->attrPower) : kNaN;
            culType = CulTypeEnum::BROADCAST_SCALAR;
        } else if (IsCloseScalar(this->attrShift, 0.0f)) {
            scalar0 = (this->attrPower < 0.0f) ? kInfPos : 0.0f;
            culType = (this->attrPower < 0.0f) ? CulTypeEnum::BROADCAST_SCALAR : CulTypeEnum::ALL_ZEROS;
        } else {
            scalar0 = std::pow(this->attrShift, this->attrPower);
            culType = CulTypeEnum::BROADCAST_SCALAR;
        }
        return true;
    }

    // 分支2：优化整数幂 1/2/3，避免 exp/log 开销
    if (IsCloseScalar(this->attrPower, 1.0f) || IsCloseScalar(this->attrPower, 2.0f) || IsCloseScalar(this->attrPower, 3.0f)) {
        culType = IsCloseScalar(this->attrPower, 1.0f) ? CulTypeEnum::LINEAR :
                  IsCloseScalar(this->attrPower, 2.0f) ? CulTypeEnum::SQUARE : CulTypeEnum::CUBE;
        scalar0 = this->attrScale;
        scalar1 = this->attrShift;
        return true;
    }

    // 分支3：通用幂运算
    culType = (this->attrPower > 0.0f) ? CulTypeEnum::GENERIC_POW_POS : CulTypeEnum::GENERIC_POW_NEG;
    scalar0 = this->attrScale;
    scalar1 = this->attrShift;
    scalar2 = this->attrPower;
    long long intPow = static_cast<long long>(std::llround(this->attrPower));
    scalar3 = powerIsInt ? ((std::llabs(intPow) % 2 == 0) ? 1.0f : -1.0f) : kNaN;
    return true;
}

// -----------------------------------------------------------------------------
// Tiling 主入口。
// 流程：dtype 校验 → shape 校验 → 读属性 → 决策 culType
//      → 编码 dType → 调 elementwise 通用 tiling 完成多核 / UB 切分
//      → 写 scalarData / workspace / tilingKey / blockDim。
// -----------------------------------------------------------------------------
ge::graphStatus PowerTiling::PerformValidationChecks()
{
    OP_CHECK_IF(
        CalcInputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get input dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "check shape failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        SetAttr() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "set attrs failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !DecideCulType(), OP_LOGE(tilingContext->GetNodeName(), "DecideCulType failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PowerTiling::MapOutputDtypeToTplKey()
{
    if (this->outputDtype == ge::DT_FLOAT16) {
        dType = POWER_TPL_DTYPE_FP16;
    } else if (this->outputDtype == ge::DT_BF16) {
        dType = POWER_TPL_DTYPE_BF16;
    } else if (this->outputDtype == ge::DT_FLOAT) {
        dType = POWER_TPL_DTYPE_FP32;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            tilingContext->GetNodeName(),
            "y",
            Ops::Base::ToString(this->outputDtype).c_str(),
            "The dtype of y must be within the range [DT_FLOAT16, DT_BF16, DT_FLOAT].");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PowerTiling::SetTilingResults(PowerOp::PowerTilingData* powerTilingData)
{
    powerTilingData->scale     = scalar0;
    powerTilingData->shift     = scalar1;
    powerTilingData->power     = scalar2;
    powerTilingData->negScalar = scalar3;

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = POWER_ASCEND_WORKSPACE;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, culTypeKey, dType);
    OP_LOGD(tilingContext->GetNodeName(), "[TilingData] : tilingKey=%lu", tilingKey);
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(powerTilingData->baseTiling.blockNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PowerTiling::RunTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "PowerTiling::RunTiling enter.");
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);

    auto* powerTilingData = tilingContext->GetTilingData<PowerOp::PowerTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, powerTilingData);

    if (PerformValidationChecks() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    culTypeKey = static_cast<uint64_t>(culType);
    OP_LOGD(
        tilingContext->GetNodeName(),
        "Power culType=%lu (scalar0=%f, scalar1=%f, scalar2=%f, scalar3=%f)",
        culTypeKey, scalar0, scalar1, scalar2, scalar3);

    if (MapOutputDtypeToTplKey() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus baseTilingResult = DispatchTilingByCulType(elewiseBaseTiling, powerTilingData);
    OP_CHECK_IF(
        baseTilingResult == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling failed"),
        return ge::GRAPH_FAILED);

    return SetTilingResults(powerTilingData);
}

// runtime 2.0 注册的 Tiling 回调入口。
static ge::graphStatus Tiling4Power(gert::TilingContext* tilingContextGen)
{
    OP_LOGD(tilingContextGen->GetNodeName(), "Tiling4Power rt2.0 is running.");
    auto compileInfo = tilingContextGen->GetCompileInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContextGen, compileInfo);
    PowerTiling baseOpTiling(tilingContextGen);
    return baseOpTiling.RunTiling();
}

// 编译期回调：从平台信息中提取 AIV 核数与 UB 大小，供后续 tiling 使用。
static ge::graphStatus TilingPrepareForPower(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Power).Tiling(Tiling4Power).TilingParse<ElewiseCompileInfo>(TilingPrepareForPower);
} // namespace optiling
