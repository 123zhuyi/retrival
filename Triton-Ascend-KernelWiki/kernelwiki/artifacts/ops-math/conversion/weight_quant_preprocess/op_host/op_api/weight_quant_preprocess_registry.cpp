/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "weight_quant_preprocess_registry.h"

#include <inttypes.h>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "log/log.h"

#define LOGE_WITH_SCENARIO(err, fmt, ...)                                              \
    OP_LOGE(err, "[NpuArch=%u, DataFlow=%s] " fmt, static_cast<uint32_t>(ctx.npuArch), \
            QuantDataFlowToString(ctx.dataFlow), ##__VA_ARGS__)

namespace {

static constexpr size_t DIMS_1 = 1;
static constexpr size_t DIMS_2 = 2;
static constexpr size_t DIMS_3 = 3;
static constexpr size_t DIMS_4 = 4;
static constexpr size_t IDX_0 = 0;
static constexpr size_t IDX_1 = 1;
static constexpr size_t IDX_2 = 2;
static constexpr size_t IDX_3 = 3;
static constexpr size_t IDX_4 = 4;
static constexpr size_t DOUBLE = 2;
static constexpr int64_t KGROUP_SIZE_MX = 32;
static constexpr int64_t NZ_16 = 16;
static constexpr int64_t NZ_C0_32 = 32;

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CeilDiv divisor b must not be zero.");
        return INT64_MIN;
    }
    return (a + b - 1) / b;
}

static bool IsMMMxA8W4DataFlow(QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto scaleDtype = ctx.weightScale->GetDataType();
    auto weightViewShape = ctx.weight->GetViewShape();

    if (weightDtype == op::DataType::DT_FLOAT4_E2M1 && scaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        ctx.xDtype == op::DataType::DT_FLOAT8_E4M3FN && ctx.xScaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        weightViewShape.GetDimNum() == DIMS_2) {
        ctx.dataFlow = QuantDataFlow::MM_MX_A8W4;
        return true;
    }
    return false;
}

static bool IsGMMMxA8W4DataFlow(QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto scaleDtype = ctx.weightScale->GetDataType();
    auto weightViewShape = ctx.weight->GetViewShape();

    if (weightDtype == op::DataType::DT_FLOAT4_E2M1 && scaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        ctx.xDtype == op::DataType::DT_FLOAT8_E4M3FN && ctx.xScaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        weightViewShape.GetDimNum() == DIMS_3) {
        ctx.dataFlow = QuantDataFlow::GMM_MX_A8W4;
        return true;
    }
    return false;
}

static aclnnStatus CheckWeightNotEmpty(const QuantContext& ctx)
{
    OP_CHECK(!ctx.weight->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "weight_quant_preprocess", "weight", std::to_string(ctx.weight->GetViewShape().GetShapeSize()).c_str(),
                 (std::string("weight must not be empty tensor when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightFormatND(const QuantContext& ctx)
{
    auto weightFormat = ctx.weight->GetStorageFormat();
    OP_CHECK(weightFormat == op::Format::FORMAT_ND || weightFormat == op::Format::FORMAT_NCL,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "weight", op::ToString(weightFormat).GetString(),
                                        "ND or NCL"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightTrans(const QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();

    size_t dimNum = viewShape.GetDimNum();
    size_t lastIdx = dimNum - IDX_1;
    size_t secondLastIdx = dimNum - IDX_2;

    OP_CHECK(
        viewStrides[secondLastIdx] == 1 && viewStrides[lastIdx] == viewShape.GetDim(secondLastIdx),
        OP_LOGE_FOR_INVALID_STRIDE(
            "weight_quant_preprocess", "weight", op::ToString(ctx.weight->GetViewStrides()).GetString(),
            (std::string("last two dims stride [1, ") + std::to_string(viewShape.GetDim(secondLastIdx)) + "]").c_str()),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightScaleNotEmpty(const QuantContext& ctx)
{
    OP_CHECK(!ctx.weightScale->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "weight_quant_preprocess", "weightScale",
                 std::to_string(ctx.weightScale->GetViewShape().GetShapeSize()).c_str(),
                 (std::string("weightScale must not be empty tensor when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightScaleFormatND(const QuantContext& ctx)
{
    auto scaleFormat = ctx.weightScale->GetStorageFormat();
    OP_CHECK(scaleFormat == op::Format::FORMAT_ND || scaleFormat == op::Format::FORMAT_NCL ||
                 scaleFormat == op::Format::FORMAT_NCHW,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "weightScale", op::ToString(scaleFormat).GetString(),
                                        "ND or NCL or NCHW"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <size_t targetDim>
static aclnnStatus CheckWeightScaleMx(const QuantContext& ctx)
{
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    size_t scaleViewDim = scaleViewShape.GetDimNum();

    OP_CHECK(scaleViewDim == targetDim,
             OP_LOGE_FOR_INVALID_SHAPEDIM("weight_quant_preprocess", "weightScale",
                                          std::to_string(scaleViewDim).c_str(), std::to_string(targetDim).c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    auto weightViewShape = ctx.weight->GetViewShape();
    size_t weightViewDim = weightViewShape.GetDimNum();
    OP_CHECK(
        scaleViewShape.GetDim(scaleViewDim - IDX_3) ==
                CeilDiv(weightViewShape.GetDim(weightViewDim - IDX_2), KGROUP_SIZE_MX * DOUBLE) &&
            scaleViewShape.GetDim(scaleViewDim - IDX_2) == weightViewShape.GetDim(weightViewDim - IDX_1) &&
            scaleViewShape.GetDim(scaleViewDim - IDX_1) == DOUBLE,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", "weightScale, weight",
            (op::ToString(scaleViewShape).GetString() + std::string(", ") + op::ToString(weightViewShape).GetString())
                .c_str(),
            (std::string("weightScale last three dims must be {ceildiv(K,64), N, 2} when dataFlow is ") +
             QuantDataFlowToString(ctx.dataFlow) + ".")
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// 校验 weightScale 在相邻两维 (targetIdx, targetIdx+1) 上是否处于转置排列。
//   - MM 场景 (kScale, n, ...)：targetIdx=0，校验第 0、1 维转置
//   - GMM 场景 (g, kScale, n, ...)：targetIdx=1，校验第 1、2 维转置
template <size_t targetIdx>
static aclnnStatus CheckWeightScaleTrans(const QuantContext& ctx)
{
    auto viewShape = ctx.weightScale->GetViewShape();
    auto viewStrides = ctx.weightScale->GetViewStrides();

    OP_CHECK(viewStrides[targetIdx + IDX_1] == viewStrides[targetIdx] * viewShape.GetDim(targetIdx),
             OP_LOGE_FOR_INVALID_STRIDE("weight_quant_preprocess", "weightScale", op::ToString(viewStrides).GetString(),
                                        (std::string("transposed at dims ") + std::to_string(targetIdx) + " and " +
                                         std::to_string(targetIdx + IDX_1))
                                            .c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightOffsetOptionalNull(const QuantContext& ctx)
{
    OP_CHECK(ctx.weightOffsetOptional == nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "weightOffsetOptional must be nullptr."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(ctx.outWeightOffsetOptional == nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "outWeightOffsetOptional must be nullptr."),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckBiasOptionalNotEmpty(const QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(!ctx.biasOptional->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "weight_quant_preprocess", "biasOptional",
                 std::to_string(ctx.biasOptional->GetViewShape().GetShapeSize()).c_str(),
                 (std::string("biasOptional must not be empty tensor when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckBiasOptionalFormatND(const QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto biasFormat = ctx.biasOptional->GetStorageFormat();
    OP_CHECK(biasFormat == op::Format::FORMAT_ND,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "biasOptional", op::ToString(biasFormat).GetString(),
                                        "ND"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <op::DataType... allowedDtypes>
static aclnnStatus CheckBiasOptionalDtype(const QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto actualDtype = ctx.biasOptional->GetDataType();
    bool match = ((actualDtype == allowedDtypes) || ...);

    auto allowedDtypesList = {allowedDtypes...};

    OP_CHECK(match,
             OP_LOGE_FOR_INVALID_DTYPE("weight_quant_preprocess", "biasOptional", op::ToString(actualDtype).GetString(),
                                       op::ToString(allowedDtypesList).GetString()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <bool isGmm>
static aclnnStatus CheckBiasOptionalViewShape(const QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }

    auto biasViewShape = ctx.biasOptional->GetViewShape();
    auto weightViewShape = ctx.weight->GetViewShape();

    size_t weightViewDim = weightViewShape.GetDimNum();
    int64_t n = weightViewShape.GetDim(weightViewDim - IDX_1);

    if constexpr (isGmm) {
        OP_CHECK(biasViewShape.GetDimNum() == DIMS_2,
                 OP_LOGE_FOR_INVALID_SHAPEDIM("weight_quant_preprocess", "biasOptional",
                                              std::to_string(biasViewShape.GetDimNum()).c_str(), "2"),
                 return ACLNN_ERR_PARAM_INVALID);

        int64_t g = weightViewShape.GetDim(IDX_0);
        OP_CHECK(biasViewShape.GetDim(IDX_0) == g && biasViewShape.GetDim(IDX_1) == n,
                 OP_LOGE_FOR_INVALID_SHAPE(
                     "weight_quant_preprocess", "biasOptional", op::ToString(biasViewShape).GetString(),
                     (std::string("(") + std::to_string(g) + ", " + std::to_string(n) + ")").c_str()),
                 return ACLNN_ERR_PARAM_INVALID);
    } else {
        size_t biasViewDim = biasViewShape.GetDimNum();
        OP_CHECK(biasViewDim == DIMS_1 || biasViewDim == DIMS_2,
                 OP_LOGE_FOR_INVALID_SHAPEDIM("weight_quant_preprocess", "biasOptional",
                                              std::to_string(biasViewDim).c_str(), "1 or 2"),
                 return ACLNN_ERR_PARAM_INVALID);

        bool isValidShape = biasViewDim == DIMS_1 ?
                                biasViewShape.GetDim(IDX_0) == n :
                                biasViewShape.GetDim(IDX_0) == 1 && biasViewShape.GetDim(IDX_1) == n;
        OP_CHECK(isValidShape,
                 OP_LOGE_FOR_INVALID_SHAPE(
                     "weight_quant_preprocess", "biasOptional", op::ToString(biasViewShape).GetString(),
                     (std::string("(") + std::to_string(n) + ") or (1, " + std::to_string(n) + ")").c_str()),
                 return ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckBiasOptionalContiguous(const QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(IsContiguous(ctx.biasOptional),
             OP_LOGE_FOR_INVALID_STRIDE("weight_quant_preprocess", "biasOptional",
                                        op::ToString(ctx.biasOptional->GetViewStrides()).GetString(), "contiguous"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckKGroupSizeMx(const QuantContext& ctx)
{
    OP_CHECK(ctx.kGroupSize == KGROUP_SIZE_MX,
             OP_LOGE_FOR_INVALID_VALUE("weight_quant_preprocess", "kGroupSize", std::to_string(ctx.kGroupSize).c_str(),
                                       "32"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightNotNullEmpty(const QuantContext& ctx)
{
    OP_CHECK(ctx.outWeight != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_NULLPTR, "outWeight must not be nullptr."),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(
        !ctx.outWeight->IsEmpty(),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("weight_quant_preprocess", "outWeight",
                                                  std::to_string(ctx.outWeight->GetViewShape().GetShapeSize()).c_str(),
                                                  (std::string("outWeight must not be empty tensor when dataFlow is ") +
                                                   QuantDataFlowToString(ctx.dataFlow) + ".")
                                                      .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightDtypeSame(const QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto outWeightDtype = ctx.outWeight->GetDataType();
    OP_CHECK(outWeightDtype == weightDtype,
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 "weight_quant_preprocess", "outWeight, weight",
                 (op::ToString(outWeightDtype).GetString() + std::string(", ") + op::ToString(weightDtype).GetString())
                     .c_str(),
                 (std::string("outWeight and weight must have the same dtype when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightViewShapeSame(const QuantContext& ctx)
{
    auto weightViewShape = ctx.weight->GetViewShape();
    auto outViewShape = ctx.outWeight->GetViewShape();
    OP_CHECK(
        weightViewShape == outViewShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", "outWeight, weight",
            (op::ToString(outViewShape).GetString() + std::string(", ") + op::ToString(weightViewShape).GetString())
                .c_str(),
            (std::string("outWeight and weight must have the same viewShape when dataFlow is ") +
             QuantDataFlowToString(ctx.dataFlow) + ".")
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightNzStorageDim(const QuantContext& ctx)
{
    auto weightViewShape = ctx.weight->GetViewShape();
    auto outStorageShape = ctx.outWeight->GetStorageShape();
    OP_CHECK(outStorageShape.GetDimNum() == weightViewShape.GetDimNum() + 2,
             OP_LOGE_FOR_INVALID_SHAPEDIM("weight_quant_preprocess", "outWeight",
                                          std::to_string(outStorageShape.GetDimNum()).c_str(),
                                          std::to_string(weightViewShape.GetDimNum() + 2).c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <int64_t nzC0, op::Format outWeightFormat>
static aclnnStatus CheckOutWeightTransNz(const QuantContext& ctx)
{
    auto outFormat = ctx.outWeight->GetStorageFormat();
    OP_CHECK(outFormat == outWeightFormat,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "outWeight", op::ToString(outFormat).GetString(),
                                        op::ToString(outWeightFormat).GetString()),
             return ACLNN_ERR_PARAM_INVALID);

    auto outStorageShape = ctx.outWeight->GetStorageShape();
    auto weightViewShape = ctx.weight->GetViewShape();
    size_t outStorageDim = outStorageShape.GetDimNum();
    size_t viewDim = weightViewShape.GetDimNum();

    int64_t k = weightViewShape.GetDim(viewDim - IDX_2);
    int64_t n = weightViewShape.GetDim(viewDim - IDX_1);
    int64_t expectedNBlocks = CeilDiv(n, NZ_16);
    int64_t expectedKBlocks = CeilDiv(k, nzC0);

    OP_CHECK(
        outStorageShape.GetDim(outStorageDim - IDX_4) == expectedKBlocks &&
            outStorageShape.GetDim(outStorageDim - IDX_3) == expectedNBlocks &&
            outStorageShape.GetDim(outStorageDim - IDX_2) == NZ_16 &&
            outStorageShape.GetDim(outStorageDim - IDX_1) == nzC0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", "outWeight, weight",
            (op::ToString(outStorageShape).GetString() + std::string(", ") + op::ToString(weightViewShape).GetString())
                .c_str(),
            (std::string("outWeight storage shape last four dims must be {ceildiv(K, nzC0), ceildiv(N, 16), 16, nzC0} "
                         "when dataFlow is ") +
             QuantDataFlowToString(ctx.dataFlow) + ".")
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightScaleNotNullEmpty(const QuantContext& ctx)
{
    OP_CHECK(ctx.outWeightScale != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_NULLPTR, "outWeightScale must not be nullptr."),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(!ctx.outWeightScale->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "weight_quant_preprocess", "outWeightScale",
                 std::to_string(ctx.outWeightScale->GetViewShape().GetShapeSize()).c_str(),
                 (std::string("outWeightScale must not be empty tensor when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightScaleFormatND(const QuantContext& ctx)
{
    auto outScaleFormat = ctx.outWeightScale->GetStorageFormat();
    OP_CHECK(outScaleFormat == op::Format::FORMAT_ND || outScaleFormat == op::Format::FORMAT_NCL ||
                 outScaleFormat == op::Format::FORMAT_NCHW,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "outWeightScale",
                                        op::ToString(outScaleFormat).GetString(), "ND or NCL or NCHW"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightScaleDtypeSame(const QuantContext& ctx)
{
    auto scaleDtype = ctx.weightScale->GetDataType();
    auto outScaleDtype = ctx.outWeightScale->GetDataType();
    OP_CHECK(outScaleDtype == scaleDtype,
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 "weight_quant_preprocess", "outWeightScale, weightScale",
                 (op::ToString(outScaleDtype).GetString() + std::string(", ") + op::ToString(scaleDtype).GetString())
                     .c_str(),
                 (std::string("outWeightScale and weightScale must have the same dtype when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightScaleViewShapeSame(const QuantContext& ctx)
{
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    auto outScaleViewShape = ctx.outWeightScale->GetViewShape();
    OP_CHECK(
        outScaleViewShape == scaleViewShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", "outWeightScale, weightScale",
            (op::ToString(outScaleViewShape).GetString() + std::string(", ") + op::ToString(scaleViewShape).GetString())
                .c_str(),
            (std::string("outWeightScale and weightScale must have the same viewShape when dataFlow is ") +
             QuantDataFlowToString(ctx.dataFlow) + ".")
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightScaleStorageShapeSame(const QuantContext& ctx)
{
    auto scaleStorageShape = ctx.weightScale->GetStorageShape();
    auto outScaleStorageShape = ctx.outWeightScale->GetStorageShape();
    OP_CHECK(outScaleStorageShape == scaleStorageShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "weight_quant_preprocess", "outWeightScale, weightScale",
                 (op::ToString(outScaleStorageShape).GetString() + std::string(", ") +
                  op::ToString(scaleStorageShape).GetString())
                     .c_str(),
                 (std::string("outWeightScale and weightScale must have the same storageShape when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutBiasOptionalNotNullEmpty(const QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(ctx.outBiasOptional != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_NULLPTR,
                                "outBiasOptional must not be nullptr when biasOptional is not nullptr."),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(!ctx.outBiasOptional->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "weight_quant_preprocess", "outBiasOptional",
                 std::to_string(ctx.outBiasOptional->GetViewShape().GetShapeSize()).c_str(),
                 (std::string("outBiasOptional must not be empty tensor when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutBiasOptionalFormatND(const QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto outBiasFormat = ctx.outBiasOptional->GetStorageFormat();
    OP_CHECK(outBiasFormat == op::Format::FORMAT_ND,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "outBiasOptional",
                                        op::ToString(outBiasFormat).GetString(), "ND"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutBiasOptionalContiguous(const QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(IsContiguous(ctx.outBiasOptional),
             OP_LOGE_FOR_INVALID_STRIDE("weight_quant_preprocess", "outBiasOptional",
                                        op::ToString(ctx.outBiasOptional->GetViewStrides()).GetString(), "contiguous"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutBiasOptionalDtypeSame(const QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto biasDtype = ctx.biasOptional->GetDataType();
    auto outBiasDtype = ctx.outBiasOptional->GetDataType();
    OP_CHECK(
        outBiasDtype == biasDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "weight_quant_preprocess", "outBiasOptional, biasOptional",
            (op::ToString(outBiasDtype).GetString() + std::string(", ") + op::ToString(biasDtype).GetString()).c_str(),
            (std::string("outBiasOptional and biasOptional must have the same dtype when dataFlow is ") +
             QuantDataFlowToString(ctx.dataFlow) + ".")
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutBiasOptionalViewShapeSame(const QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto biasViewShape = ctx.biasOptional->GetViewShape();
    auto outBiasViewShape = ctx.outBiasOptional->GetViewShape();
    OP_CHECK(
        outBiasViewShape == biasViewShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", "outBiasOptional, biasOptional",
            (op::ToString(outBiasViewShape).GetString() + std::string(", ") + op::ToString(biasViewShape).GetString())
                .c_str(),
            (std::string("outBiasOptional and biasOptional must have the same viewShape when dataFlow is ") +
             QuantDataFlowToString(ctx.dataFlow) + ".")
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutBiasOptionalStorageShapeSame(const QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto biasStorageShape = ctx.biasOptional->GetStorageShape();
    auto outBiasStorageShape = ctx.outBiasOptional->GetStorageShape();
    OP_CHECK(outBiasStorageShape == biasStorageShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "weight_quant_preprocess", "outBiasOptional, biasOptional",
                 (op::ToString(outBiasStorageShape).GetString() + std::string(", ") +
                  op::ToString(biasStorageShape).GetString())
                     .c_str(),
                 (std::string("outBiasOptional and biasOptional must have the same storageShape when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <size_t viewKIdx>
static aclnnStatus ProcessWeightTransNd2Nz(QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto dstFormat = ctx.outWeight->GetStorageFormat();

    op::Shape storageShape(viewShape);
    // torch 侧传入的 storageShape 为 1 维，需要手动构造转置后的 storageShape
    std::swap(storageShape[viewKIdx], storageShape[viewKIdx + IDX_1]);

    auto weightTensor = const_cast<aclTensor*>(ctx.weight);
    // TransData 要求输入连续 Tensor，设置所有 Shape 为转置
    weightTensor->SetViewShape(storageShape);
    weightTensor->SetOriginalShape(storageShape);
    weightTensor->SetStorageShape(storageShape);
    auto outTensor = const_cast<aclTensor*>(l0op::TransData(weightTensor, dstFormat, 0, ctx.executor));
    OP_CHECK(outTensor != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "TransData failed, storageShape=%s, dstFormat=%d.",
                                op::ToString(storageShape).GetString(), static_cast<int>(dstFormat)),
             return ACLNN_ERR_INNER_NULLPTR);
    // TransData 输出 storageFormat 为 (dstFormat, 0) 组合状态，因 ViewCopy 要求完全一致，需要重新设置
    outTensor->SetStorageFormat(dstFormat);
    outTensor->SetViewShape(viewShape);
    // SetViewShape 内部会隐式设置 viewStride 为非转置连续，重新设置 viewStride 为转置非连续
    op::Strides viewStrides(outTensor->GetViewStrides());
    viewStrides[viewKIdx] = 1;
    viewStrides[viewKIdx + IDX_1] = viewShape.GetDim(viewKIdx);
    outTensor->SetViewStrides(viewStrides);

    auto viewCopyResult = l0op::ViewCopy(outTensor, ctx.outWeight, ctx.executor);
    OP_CHECK(viewCopyResult != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR,
                                "ViewCopy failed, outTensor viewShape=%s format=%d, outWeight viewShape=%s format=%d.",
                                op::ToString(outTensor->GetViewShape()).GetString(),
                                static_cast<int>(outTensor->GetStorageFormat()),
                                op::ToString(ctx.outWeight->GetViewShape()).GetString(),
                                static_cast<int>(ctx.outWeight->GetStorageFormat())),
             return ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessWeightScaleDirectCopy(QuantContext& ctx)
{
    auto srcScale = const_cast<aclTensor*>(ctx.weightScale);
    auto dstScale = ctx.outWeightScale;

    if (srcScale->GetDataType() == op::DataType::DT_FLOAT8_E8M0) {
        auto srcView = ctx.executor->CreateView(srcScale, srcScale->GetViewShape(), srcScale->GetViewOffset());
        OP_CHECK(srcView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView weightScale failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        srcView->SetDataType(op::DataType::DT_INT8);

        auto dstView = ctx.executor->CreateView(dstScale, dstScale->GetViewShape(), dstScale->GetViewOffset());
        OP_CHECK(dstView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView outWeightScale failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        dstView->SetDataType(op::DataType::DT_INT8);

        auto result = l0op::ViewCopy(srcView, dstView, ctx.executor);
        OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weightScale as INT8 failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto result = l0op::ViewCopy(srcScale, dstScale, ctx.executor);
        OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weightScale failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessBiasDirectCopy(QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto result = l0op::ViewCopy(const_cast<aclTensor*>(ctx.biasOptional), ctx.outBiasOptional, ctx.executor);
    OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy bias failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

const std::unordered_map<NpuArch, std::vector<DataFlowEntry>> NPU_DATA_FLOW_REGISTRY_MAP = {
    {NpuArch::DAV_3510,
     {{.judge = IsMMMxA8W4DataFlow,
       .checks =
           {CheckWeightNotEmpty, CheckWeightFormatND,
            CheckWeightTrans, // 校验 weight 最后 2 维是否转置
            CheckWeightScaleNotEmpty, CheckWeightScaleFormatND,
            CheckWeightScaleMx<DIMS_3>, // 校验 weightScale 的 viewShape 是否符合 Mx 场景的 3 维形式 (k/64, n, 2)
            CheckWeightScaleTrans<IDX_0>, // 校验 weightScale 的第 0 维和第 1 维是否转置
            CheckWeightOffsetOptionalNull, CheckBiasOptionalNotEmpty, CheckBiasOptionalFormatND,
            CheckBiasOptionalDtype<op::DataType::DT_FLOAT16, op::DataType::DT_BF16>, // 支持 bias 数据类型 FP16/BF16
            CheckBiasOptionalViewShape<false>, // false 表示非 GMM 场景，bias shape 要求 (n) 或 (1, n)
            CheckBiasOptionalContiguous,
            CheckKGroupSizeMx, // Mx 场景要求 kGroupSize 为 32
            CheckOutWeightNotNullEmpty, CheckOutWeightDtypeSame, CheckOutWeightViewShapeSame,
            CheckOutWeightNzStorageDim, // 输出 weight 为 NZ 格式，校验 storageShape 维度是否正确
            CheckOutWeightTransNz<NZ_C0_32, op::Format::FORMAT_FRACTAL_NZ_C0_32>, // 校验输出 weight storageShape 符合
                                                                                  // C0_32
            CheckOutWeightScaleNotNullEmpty, CheckOutWeightScaleFormatND, CheckOutWeightScaleDtypeSame,
            CheckOutWeightScaleViewShapeSame, CheckOutWeightScaleStorageShapeSame, CheckOutBiasOptionalNotNullEmpty,
            CheckOutBiasOptionalFormatND, CheckOutBiasOptionalContiguous, CheckOutBiasOptionalDtypeSame,
            CheckOutBiasOptionalViewShapeSame, CheckOutBiasOptionalStorageShapeSame},
       .processes = {ProcessWeightTransNd2Nz<IDX_0>, // 对 weight 进行 Nd2Nz 转换，参数表示 k 在 viewShape 中的下标
                     ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsGMMMxA8W4DataFlow,
       .checks =
           {CheckWeightNotEmpty, CheckWeightFormatND,
            CheckWeightTrans, // 校验 weight 最后 2 维是否转置
            CheckWeightScaleNotEmpty, CheckWeightScaleFormatND,
            CheckWeightScaleMx<DIMS_4>, // 校验 weightScale 的 viewShape 是否符合 Mx 场景的 4 维形式 (g, k/64, n, 2)
            CheckWeightScaleTrans<IDX_1>, // 校验 weightScale 的第 1 维和第 2 维是否转置
            CheckWeightOffsetOptionalNull, CheckBiasOptionalNotEmpty, CheckBiasOptionalFormatND,
            CheckBiasOptionalDtype<op::DataType::DT_FLOAT16, op::DataType::DT_BF16>, // 支持 bias 数据类型 FP16/BF16
            CheckBiasOptionalViewShape<true>, // true 表示 GMM 场景，bias shape 要求 (g, n)
            CheckBiasOptionalContiguous,
            CheckKGroupSizeMx, // Mx 场景要求 kGroupSize 为 32
            CheckOutWeightNotNullEmpty, CheckOutWeightDtypeSame, CheckOutWeightViewShapeSame,
            CheckOutWeightNzStorageDim, // 输出 weight 为 NZ 格式，校验 storageShape 维度是否正确
            CheckOutWeightTransNz<NZ_C0_32, op::Format::FORMAT_FRACTAL_NZ_C0_32>, // 校验输出 weight storageShape 符合
                                                                                  // C0_32
            CheckOutWeightScaleNotNullEmpty, CheckOutWeightScaleFormatND, CheckOutWeightScaleDtypeSame,
            CheckOutWeightScaleViewShapeSame, CheckOutWeightScaleStorageShapeSame, CheckOutBiasOptionalNotNullEmpty,
            CheckOutBiasOptionalFormatND, CheckOutBiasOptionalContiguous, CheckOutBiasOptionalDtypeSame,
            CheckOutBiasOptionalViewShapeSame, CheckOutBiasOptionalStorageShapeSame},
       .processes = {ProcessWeightTransNd2Nz<IDX_1>, // 对 weight 进行 Nd2Nz 转换，参数表示 k 在 viewShape 中的下标
                     ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}}}}};
