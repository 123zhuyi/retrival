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
static constexpr int64_t NZ_C0_16 = 16;
static constexpr int64_t NZ_C0_32 = 32;
static constexpr int64_t B4_NUMS_PER_BYTE = 2; // 1 字节打包 2 个 4-bit 值（INT4/FP4 通用）

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

// ===== A16S4/A16F4 judge 判定件 =====
// judge 按 scale 模式 × 转置 × 出 format 拆分，以下判定件负责组装；dtype 通过模板参数传入
// x dtype fp16/bf16 + weight 2D；A16W4 无 xScale 语义，xScale 由各 judge 以 IsXScaleUndefined 统一排除
template <op::DataType WType>
static bool IsA16W4Base(const QuantContext& ctx)
{
    bool xDtypeMatch = (ctx.xDtype == op::DataType::DT_FLOAT16 || ctx.xDtype == op::DataType::DT_BF16);
    return ctx.weight->GetDataType() == WType && xDtypeMatch && ctx.weight->GetViewShape().GetDimNum() == DIMS_2;
}

static bool IsXScaleUndefined(const QuantContext& ctx) { return ctx.xScaleDtype == op::DataType::DT_UNDEFINED; }

// 转置探测：最后两维 strides [1, size(-2)] 即转置（与 CheckWeightTrans 同一判定）
static bool IsWeightLastTwoDimsTrans(const QuantContext& ctx)
{
    const auto& viewShape = ctx.weight->GetViewShape();
    const auto& viewStrides = ctx.weight->GetViewStrides();
    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    return viewStrides[dimNum - IDX_2] == 1 && viewStrides[dimNum - IDX_1] == viewShape.GetDim(dimNum - IDX_2);
}

// scale 模式判定：per-tensor 单元素 / per-channel 1D 或 [1,N] 且 numel>1 / per-group 2D 且 G>1
static bool IsScalePerTensor(const QuantContext& ctx) { return ctx.weightScale->GetViewShape().GetShapeSize() == 1; }

static bool IsScalePerChannel(const QuantContext& ctx)
{
    const auto& scaleViewShape = ctx.weightScale->GetViewShape();
    int64_t scaleDim = static_cast<int64_t>(scaleViewShape.GetDimNum());
    return (scaleDim == DIMS_1 || (scaleDim == DIMS_2 && scaleViewShape.GetDim(IDX_0) == 1)) &&
           scaleViewShape.GetShapeSize() > 1;
}

static bool IsScalePerGroup(const QuantContext& ctx)
{
    const auto& scaleViewShape = ctx.weightScale->GetViewShape();
    return scaleViewShape.GetDimNum() == DIMS_2 && scaleViewShape.GetDim(IDX_0) > 1;
}

// out format 判定：NZ_C0_16 分形转换；ND/NCL 出（直拷）由转置 judge 命中，无需单独判定
static bool IsOutWeightNzC016(const QuantContext& ctx)
{
    return ctx.outWeight != nullptr && ctx.outWeight->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ_C0_16;
}

// A16S4 per-tensor：转置不验证（ND 直拷物理透传与转置无关），仅 ND 出
static bool IsMMA16S4PerTensorDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerTensor(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERTENSOR;
        return true;
    }
    return false;
}

// A16S4 per-channel：转置 → ND 直拷；非转置仅支持 NZ_C0_16 出（ND 出不支持，与 A16F4 对齐）
static bool IsMMA16S4PerChannelTransDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerChannel(ctx) &&
        IsWeightLastTwoDimsTrans(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERCHANNEL;
        return true;
    }
    return false;
}

static bool IsMMA16S4PerChannelNonTransNzDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerChannel(ctx) &&
        !IsWeightLastTwoDimsTrans(ctx) && IsOutWeightNzC016(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERCHANNEL;
        return true;
    }
    return false;
}

// A16S4 per-group：同 per-channel 的两路拆分（转置 ND 直拷 / 非转置 NZ_C0_16 转换）
static bool IsMMA16S4PerGroupTransDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerGroup(ctx) &&
        IsWeightLastTwoDimsTrans(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERGROUP;
        return true;
    }
    return false;
}

static bool IsMMA16S4PerGroupNonTransNzDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerGroup(ctx) &&
        !IsWeightLastTwoDimsTrans(ctx) && IsOutWeightNzC016(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERGROUP;
        return true;
    }
    return false;
}

// A16F4 per-group：FP4 weight + per-group scale [G, N]（G > 1）；NZ only，转置由 checks 的
// CheckWeightNotTrans 拒绝（保留明确报错）
static bool IsMMA16F4PerGroupDataFlow(QuantContext& ctx)
{
    auto scaleDtype = ctx.weightScale->GetDataType();
    bool scaleDtypeMatch = (scaleDtype == op::DataType::DT_FLOAT16 || scaleDtype == op::DataType::DT_BF16);

    if (IsA16W4Base<op::DataType::DT_FLOAT4_E2M1>(ctx) && scaleDtypeMatch && IsXScaleUndefined(ctx) &&
        IsScalePerGroup(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16F4_PERGROUP;
        return true;
    }
    return false;
}

// A16MXFP4：FP4 weight + MX scale（E8M0，2D [K/32, N] 连续，与 wqbmmv2 MX kernel 约定一致），
// NZ only，转置由 checks 的 CheckWeightNotTrans 拒绝
static bool IsMMA16MXF4DataFlow(QuantContext& ctx)
{
    auto scaleDtype = ctx.weightScale->GetDataType();

    if (IsA16W4Base<op::DataType::DT_FLOAT4_E2M1>(ctx) && scaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        IsXScaleUndefined(ctx) && ctx.weightScale->GetViewShape().GetDimNum() == DIMS_2) {
        ctx.dataFlow = QuantDataFlow::MM_A16MXF4;
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

    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;

    OP_CHECK(
        viewStrides[secondLastIdx] == 1 && viewStrides[lastIdx] == viewShape.GetDim(secondLastIdx),
        OP_LOGE_FOR_INVALID_STRIDE(
            "weight_quant_preprocess", "weight", op::ToString(ctx.weight->GetViewStrides()).GetString(),
            (std::string("last two dims stride [1, ") + std::to_string(viewShape.GetDim(secondLastIdx)) + "]").c_str()),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// A16S4 NZ 路径（per-channel / per-group）仅支持非转置 weight，转置返回错误
static aclnnStatus CheckWeightNotTrans(const QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();

    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;

    OP_CHECK(viewStrides[lastIdx] == 1 && viewStrides[secondLastIdx] == viewShape.GetDim(lastIdx),
             OP_LOGE_FOR_INVALID_STRIDE(
                 "weight_quant_preprocess", "weight", op::ToString(ctx.weight->GetViewStrides()).GetString(),
                 (std::string("transposed weight is not supported for ") + QuantDataFlowToString(ctx.dataFlow) +
                  ", last two dims stride should be [" + std::to_string(viewShape.GetDim(lastIdx)) + ", 1]")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 紧凑 4-bit（INT4/FP4）每字节打包 2 个值：打包维（连续维）必须为偶数
// 非转置 [K,N] 沿 N 打包 -> N 为偶数；转置 [K,N] strides [1,K] 沿 K 打包 -> K 为偶数
static aclnnStatus CheckWeightPackingDimEven(const QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();

    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;
    bool isTransposed = (viewStrides[secondLastIdx] == 1 && viewShape.GetDim(secondLastIdx) > 1);
    int64_t packingDim = isTransposed ? viewShape.GetDim(secondLastIdx) : viewShape.GetDim(lastIdx);

    OP_CHECK(packingDim % B4_NUMS_PER_BYTE == 0,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "weight_quant_preprocess", "weight", op::ToString(viewShape).GetString(),
                 (std::string("the packing dim of 4-bit weight must be even when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// INT4 直拷（ProcessWeightDirectCopy）以 GetViewOffset()/2 建 UINT8 打包视图按字节物理透传：
// weight 视图必须为连续或末两维严格转置（其余 strides 模式会按错误的打包维寻址），
// 且 weight/outWeight 的 viewOffset 须为偶数（奇数偏移除以 2 截断后错位半字节）
static aclnnStatus CheckWeightInt4DirectCopyView(const QuantContext& ctx)
{
    if (ctx.weight->GetDataType() != op::DataType::DT_INT4) {
        return ACLNN_SUCCESS;
    }
    // outWeight 为空属参数错误，由后续 CheckOutWeightNotNullEmpty 报告；此处提前返回避免空指针
    if (ctx.outWeight == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();
    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;
    bool isStrictTransposed = (viewStrides[secondLastIdx] == 1 && viewShape.GetDim(secondLastIdx) > 1 &&
                               viewStrides[lastIdx] == viewShape.GetDim(secondLastIdx));
    OP_CHECK(IsContiguous(ctx.weight) || isStrictTransposed,
             OP_LOGE_FOR_INVALID_STRIDE(
                 "weight_quant_preprocess", "weight", op::ToString(viewStrides).GetString(),
                 (std::string("contiguous or strictly transposed at last two dims when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        ctx.weight->GetViewOffset() % B4_NUMS_PER_BYTE == 0 && ctx.outWeight->GetViewOffset() % B4_NUMS_PER_BYTE == 0,
        OP_LOGE_FOR_INVALID_VALUE("weight_quant_preprocess", "weight/outWeight viewOffset",
                                  (std::to_string(ctx.weight->GetViewOffset()) + std::string("/") +
                                   std::to_string(ctx.outWeight->GetViewOffset()))
                                      .c_str(),
                                  "even (INT4 packs 2 values per byte, odd offset misaligns the nibble)"),
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

static aclnnStatus CheckWeightScalePerChannelViewShape(const QuantContext& ctx)
{
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    auto weightViewShape = ctx.weight->GetViewShape();
    int64_t scaleViewDim = static_cast<int64_t>(scaleViewShape.GetDimNum());
    int64_t n = weightViewShape.GetDim(weightViewShape.GetDimNum() - IDX_1);

    OP_CHECK(scaleViewDim == DIMS_1 || scaleViewDim == DIMS_2,
             OP_LOGE_FOR_INVALID_SHAPEDIM("weight_quant_preprocess", "weightScale",
                                          std::to_string(scaleViewDim).c_str(), "1 or 2"),
             return ACLNN_ERR_PARAM_INVALID);
    bool isValidShape = scaleViewDim == DIMS_1 ? scaleViewShape.GetDim(IDX_0) == n :
                                                 scaleViewShape.GetDim(IDX_0) == 1 && scaleViewShape.GetDim(IDX_1) == n;
    OP_CHECK(isValidShape,
             OP_LOGE_FOR_INVALID_SHAPE(
                 "weight_quant_preprocess", "weightScale", op::ToString(scaleViewShape).GetString(),
                 (std::string("(") + std::to_string(n) + ") or (1, " + std::to_string(n) + ")").c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// per-tensor 场景 weightScale 仅含单个元素（{1}/{1,1}）
static aclnnStatus CheckWeightScalePerTensorViewShape(const QuantContext& ctx)
{
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    OP_CHECK(scaleViewShape.GetShapeSize() == 1,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "weight_quant_preprocess", "weightScale", op::ToString(scaleViewShape).GetString(),
                 (std::string("weightScale must contain exactly one element when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightScalePerGroupViewShape(const QuantContext& ctx)
{
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    auto weightViewShape = ctx.weight->GetViewShape();
    size_t scaleViewDim = scaleViewShape.GetDimNum();
    size_t weightViewDim = weightViewShape.GetDimNum();
    int64_t k = weightViewShape.GetDim(weightViewDim - IDX_2);
    int64_t n = weightViewShape.GetDim(weightViewDim - IDX_1);
    // MX 分组数按 ceildiv 语义（与 CheckWeightScaleMx 一致），K 非 kGroupSize 整数倍时末组为部分组
    int64_t expectedG = CeilDiv(k, ctx.kGroupSize);

    OP_CHECK(scaleViewDim == DIMS_2,
             OP_LOGE_FOR_INVALID_SHAPEDIM("weight_quant_preprocess", "weightScale",
                                          std::to_string(scaleViewDim).c_str(), "2"),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(scaleViewShape.GetDim(IDX_0) == expectedG && scaleViewShape.GetDim(IDX_1) == n,
             OP_LOGE_FOR_INVALID_SHAPE(
                 "weight_quant_preprocess", "weightScale", op::ToString(scaleViewShape).GetString(),
                 (std::string("(") + std::to_string(expectedG) + ", " + std::to_string(n) + ")").c_str()),
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

// offset 无需布局转换，与 scale 同形同 dtype 直拷透传（下游 wqbmmv2 要求 antiquantOffset 与 antiquantScale 同形状）
static aclnnStatus CheckWeightOffsetOptionalNotEmpty(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(!ctx.weightOffsetOptional->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "weight_quant_preprocess", "weightOffsetOptional",
                 std::to_string(ctx.weightOffsetOptional->GetViewShape().GetShapeSize()).c_str(),
                 (std::string("weightOffsetOptional must not be empty tensor when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightOffsetOptionalFormatND(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto offsetFormat = ctx.weightOffsetOptional->GetStorageFormat();
    OP_CHECK(offsetFormat == op::Format::FORMAT_ND,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "weightOffsetOptional",
                                        op::ToString(offsetFormat).GetString(), "ND"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightOffsetOptionalDtypeSameAsScale(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto offsetDtype = ctx.weightOffsetOptional->GetDataType();
    auto scaleDtype = ctx.weightScale->GetDataType();
    OP_CHECK(
        offsetDtype == scaleDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "weight_quant_preprocess", "weightOffsetOptional, weightScale",
            (op::ToString(offsetDtype).GetString() + std::string(", ") + op::ToString(scaleDtype).GetString()).c_str(),
            (std::string("weightOffsetOptional and weightScale must have the same dtype when dataFlow is ") +
             QuantDataFlowToString(ctx.dataFlow) + ".")
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightOffsetOptionalViewShapeSameAsScale(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto offsetViewShape = ctx.weightOffsetOptional->GetViewShape();
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    OP_CHECK(
        offsetViewShape == scaleViewShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", "weightOffsetOptional, weightScale",
            (op::ToString(offsetViewShape).GetString() + std::string(", ") + op::ToString(scaleViewShape).GetString())
                .c_str(),
            (std::string("weightOffsetOptional and weightScale must have the same viewShape when dataFlow is ") +
             QuantDataFlowToString(ctx.dataFlow) + ".")
                .c_str()),
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
static aclnnStatus CheckWeightScaleDtype(const QuantContext& ctx)
{
    auto actualDtype = ctx.weightScale->GetDataType();
    bool match = ((actualDtype == allowedDtypes) || ...);
    auto allowedDtypesList = {allowedDtypes...};
    OP_CHECK(match,
             OP_LOGE_FOR_INVALID_DTYPE("weight_quant_preprocess", "weightScale", op::ToString(actualDtype).GetString(),
                                       op::ToString(allowedDtypesList).GetString()),
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

// per-channel 场景要求 kGroupSize 为 0：per-group 语义（kGroupSize>0）配 per-channel 形状
// scale {N}/{1,N} 属矛盾输入
static aclnnStatus CheckKGroupSizeZero(const QuantContext& ctx)
{
    OP_CHECK(ctx.kGroupSize == 0,
             OP_LOGE_FOR_INVALID_VALUE("weight_quant_preprocess", "kGroupSize", std::to_string(ctx.kGroupSize).c_str(),
                                       "kGroupSize must be 0 when weightScale is per-channel ({N} or {1, N})"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// per-group 场景要求 kGroupSize > 0：per-group 形状 scale {G, N} 配 kGroupSize<=0 属矛盾输入；
// 且须先于 CheckWeightScalePerGroupViewShape 执行——后者要用 kGroupSize 做除法（同 MX 条目 CheckKGroupSizeMx 前置）
static aclnnStatus CheckKGroupSizePositive(const QuantContext& ctx)
{
    OP_CHECK(ctx.kGroupSize > 0,
             OP_LOGE_FOR_INVALID_VALUE("weight_quant_preprocess", "kGroupSize", std::to_string(ctx.kGroupSize).c_str(),
                                       "kGroupSize must be > 0 when weightScale is per-group ({G, N})"),
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

static aclnnStatus CheckOutWeightFormatND(const QuantContext& ctx)
{
    auto outWeightFormat = ctx.outWeight->GetStorageFormat();
    OP_CHECK(outWeightFormat == op::Format::FORMAT_ND || outWeightFormat == op::Format::FORMAT_NCL,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "outWeight",
                                        op::ToString(outWeightFormat).GetString(), "FORMAT_ND or FORMAT_NCL"),
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

    // NZ_C0_16（A16S4/A16F4 紧凑 4-bit）物理布局为 [N/16, K/nzC0, 16, nzC0]（N 块在前）；
    // NZ_C0_32（A8W4）保持 master 的 [K/nzC0, N/16, 16, nzC0]（K 块在前）布局约定
    bool nFirst = (outWeightFormat == op::Format::FORMAT_FRACTAL_NZ_C0_16);
    int64_t expectedBlocks4 = nFirst ? expectedNBlocks : expectedKBlocks;
    int64_t expectedBlocks3 = nFirst ? expectedKBlocks : expectedNBlocks;
    const char* layoutDesc = nFirst ? "{ceildiv(N, 16), ceildiv(K, nzC0), 16, nzC0}" :
                                      "{ceildiv(K, nzC0), ceildiv(N, 16), 16, nzC0}";
    OP_CHECK(
        outStorageShape.GetDim(outStorageDim - IDX_4) == expectedBlocks4 &&
            outStorageShape.GetDim(outStorageDim - IDX_3) == expectedBlocks3 &&
            outStorageShape.GetDim(outStorageDim - IDX_2) == NZ_16 &&
            outStorageShape.GetDim(outStorageDim - IDX_1) == nzC0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", "outWeight, weight",
            (op::ToString(outStorageShape).GetString() + std::string(", ") + op::ToString(weightViewShape).GetString())
                .c_str(),
            (std::string("outWeight storage shape last four dims must be ") + layoutDesc +
             std::string(" when dataFlow is ") + QuantDataFlowToString(ctx.dataFlow) + ".")
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

static aclnnStatus CheckOutWeightOffsetOptionalNotNullEmpty(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(
        ctx.outWeightOffsetOptional != nullptr,
        LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_NULLPTR,
                           "outWeightOffsetOptional must not be nullptr when weightOffsetOptional is not nullptr."),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(!ctx.outWeightOffsetOptional->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "weight_quant_preprocess", "outWeightOffsetOptional",
                 std::to_string(ctx.outWeightOffsetOptional->GetViewShape().GetShapeSize()).c_str(),
                 (std::string("outWeightOffsetOptional must not be empty tensor when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightOffsetOptionalFormatND(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto outOffsetFormat = ctx.outWeightOffsetOptional->GetStorageFormat();
    OP_CHECK(outOffsetFormat == op::Format::FORMAT_ND,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "outWeightOffsetOptional",
                                        op::ToString(outOffsetFormat).GetString(), "ND"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightOffsetOptionalDtypeSame(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto offsetDtype = ctx.weightOffsetOptional->GetDataType();
    auto outOffsetDtype = ctx.outWeightOffsetOptional->GetDataType();
    OP_CHECK(outOffsetDtype == offsetDtype,
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 "weight_quant_preprocess", "outWeightOffsetOptional, weightOffsetOptional",
                 (op::ToString(outOffsetDtype).GetString() + std::string(", ") + op::ToString(offsetDtype).GetString())
                     .c_str(),
                 (std::string("outWeightOffsetOptional and weightOffsetOptional must have the same dtype when "
                              "dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightOffsetOptionalViewShapeSame(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto offsetViewShape = ctx.weightOffsetOptional->GetViewShape();
    auto outOffsetViewShape = ctx.outWeightOffsetOptional->GetViewShape();
    OP_CHECK(outOffsetViewShape == offsetViewShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "weight_quant_preprocess", "outWeightOffsetOptional, weightOffsetOptional",
                 (op::ToString(outOffsetViewShape).GetString() + std::string(", ") +
                  op::ToString(offsetViewShape).GetString())
                     .c_str(),
                 (std::string("outWeightOffsetOptional and weightOffsetOptional must have the same viewShape when "
                              "dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow) + ".")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightOffsetOptionalStorageShapeSame(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto offsetStorageShape = ctx.weightOffsetOptional->GetStorageShape();
    auto outOffsetStorageShape = ctx.outWeightOffsetOptional->GetStorageShape();
    OP_CHECK(outOffsetStorageShape == offsetStorageShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "weight_quant_preprocess", "outWeightOffsetOptional, weightOffsetOptional",
                 (op::ToString(outOffsetStorageShape).GetString() + std::string(", ") +
                  op::ToString(offsetStorageShape).GetString())
                     .c_str(),
                 (std::string("outWeightOffsetOptional and weightOffsetOptional must have the same storageShape when "
                              "dataFlow is ") +
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
    OP_CHECK(static_cast<int64_t>(viewStrides.size()) > static_cast<int64_t>(viewKIdx + IDX_1),
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "viewStrides size too small for viewKIdx."),
             return ACLNN_ERR_PARAM_INVALID);
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

// 非转置 weight 的 ND 到 NZ 转换（用于 A16INT4 等场景）
template <size_t viewKIdx>
static aclnnStatus ProcessWeightNonTransNd2Nz(QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto dstFormat = ctx.outWeight->GetStorageFormat();

    // INT4 和 FP4 一样，传入逻辑 shape [K, N]，不除以 2
    // runtime 会根据 dtype 自动计算物理大小
    op::Shape storageShape(viewShape);

    auto weightTensor = const_cast<aclTensor*>(ctx.weight);
    weightTensor->SetViewShape(storageShape);
    weightTensor->SetOriginalShape(storageShape);
    weightTensor->SetStorageShape(storageShape);

    auto outTensor = const_cast<aclTensor*>(l0op::TransData(weightTensor, dstFormat, 0, ctx.executor));
    OP_CHECK(outTensor != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "TransData failed, storageShape=%s, dstFormat=%d.",
                                op::ToString(storageShape).GetString(), static_cast<int>(dstFormat)),
             return ACLNN_ERR_INNER_NULLPTR);
    outTensor->SetStorageFormat(dstFormat);
    outTensor->SetViewShape(viewShape);
    op::Strides viewStrides(outTensor->GetViewStrides());
    OP_CHECK(static_cast<int64_t>(viewStrides.size()) > static_cast<int64_t>(viewKIdx + IDX_1),
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "viewStrides size too small for viewKIdx."),
             return ACLNN_ERR_PARAM_INVALID);
    viewStrides[viewKIdx] = viewShape.GetDim(viewKIdx + IDX_1);
    viewStrides[viewKIdx + IDX_1] = 1;
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

static aclnnStatus ProcessWeightOffsetDirectCopy(QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto result = l0op::ViewCopy(const_cast<aclTensor*>(ctx.weightOffsetOptional), ctx.outWeightOffsetOptional,
                                 ctx.executor);
    OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weightOffset failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessWeightDirectCopy(QuantContext& ctx)
{
    auto srcWeight = const_cast<aclTensor*>(ctx.weight);
    auto dstWeight = ctx.outWeight;

    if (srcWeight->GetDataType() == op::DataType::DT_INT4) {
        auto weightViewShape = srcWeight->GetViewShape();
        auto weightViewStrides = srcWeight->GetViewStrides();
        op::Shape packedShape(weightViewShape);
        size_t lastDimIdx = packedShape.GetDimNum() - 1;
        size_t secondLastDimIdx = packedShape.GetDimNum() - IDX_2;
        bool isTransposed = (weightViewStrides[secondLastDimIdx] == 1 && weightViewShape.GetDim(secondLastDimIdx) > 1);
        size_t packDimIdx = isTransposed ? secondLastDimIdx : lastDimIdx;
        packedShape.SetDim(packDimIdx, packedShape.GetDim(packDimIdx) / B4_NUMS_PER_BYTE);

        auto srcView = ctx.executor->CreateView(srcWeight, packedShape, srcWeight->GetViewOffset() / B4_NUMS_PER_BYTE);
        OP_CHECK(srcView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView weight failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        srcView->SetDataType(op::DataType::DT_UINT8);

        auto dstView = ctx.executor->CreateView(dstWeight, packedShape, dstWeight->GetViewOffset() / B4_NUMS_PER_BYTE);
        OP_CHECK(dstView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView outWeight failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        dstView->SetDataType(op::DataType::DT_UINT8);

        auto result = l0op::ViewCopy(srcView, dstView, ctx.executor);
        OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weight as UINT8 failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    }

    auto result = l0op::ViewCopy(srcWeight, dstWeight, ctx.executor);
    OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weight failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

// ===== A16S4/A16F4 检查组合：按粒度拼装，条目间无内部分流 =====
// 输入公共检查（紧凑 4-bit 打包维须为偶数；INT4 直拷视图须连续/严格转置且偏移字节对齐）
const std::vector<CheckFunc> INPUT_BASE_CHECKS = {CheckWeightNotEmpty,       CheckWeightFormatND,
                                                  CheckWeightPackingDimEven, CheckWeightInt4DirectCopyView,
                                                  CheckWeightScaleNotEmpty,  CheckWeightScaleFormatND};

const std::vector<CheckFunc> SCALE_DTYPE_F16_BF16_CHECK = {
    CheckWeightScaleDtype<op::DataType::DT_FLOAT16, op::DataType::DT_BF16>};

// per-channel 形状 scale 配 kGroupSize>0 属矛盾输入（CheckKGroupSizeZero）
const std::vector<CheckFunc> PER_CHANNEL_SCALE_CHECKS = {CheckWeightScalePerChannelViewShape, CheckKGroupSizeZero};

// per-group scale 须为 [ceildiv(K, kGroupSize), N]；kGroupSize<=0 由 CheckKGroupSizePositive 先行拒绝，
// 避免 CheckWeightScalePerGroupViewShape 中 CeilDiv 除 0
const std::vector<CheckFunc> PER_GROUP_SCALE_CHECKS = {CheckKGroupSizePositive, CheckWeightScalePerGroupViewShape};

// offset 与 scale 同形同 dtype 直拷透传；bias 仅直拷透传，不校验 dtype（支持 fp16/bf16/fp32 等，由下游 matmul
// 信息库约束）
const std::vector<CheckFunc> BIAS_OFFSET_CHECKS = {CheckWeightOffsetOptionalNotEmpty,
                                                   CheckWeightOffsetOptionalFormatND,
                                                   CheckWeightOffsetOptionalDtypeSameAsScale,
                                                   CheckWeightOffsetOptionalViewShapeSameAsScale,
                                                   CheckBiasOptionalNotEmpty,
                                                   CheckBiasOptionalFormatND,
                                                   CheckBiasOptionalViewShape<false>,
                                                   CheckBiasOptionalContiguous};

// out weight ND 直拷检查
const std::vector<CheckFunc> OUT_WEIGHT_ND_CHECKS = {CheckOutWeightNotNullEmpty, CheckOutWeightDtypeSame,
                                                     CheckOutWeightFormatND, CheckOutWeightViewShapeSame};

// out weight NZ_C0_16 分形检查（CheckWeightNotTrans 防御：judge 已保证非转置）
const std::vector<CheckFunc> OUT_WEIGHT_NZ_C016_CHECKS = {
    CheckWeightNotTrans,        CheckOutWeightNotNullEmpty,
    CheckOutWeightDtypeSame,    CheckOutWeightViewShapeSame,
    CheckOutWeightNzStorageDim, CheckOutWeightTransNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ_C0_16>};

// 输出 scale/offset/bias 与入参一致性检查（公共尾部）
const std::vector<CheckFunc> OUT_TAIL_CHECKS = {CheckOutWeightScaleNotNullEmpty,
                                                CheckOutWeightScaleFormatND,
                                                CheckOutWeightScaleDtypeSame,
                                                CheckOutWeightScaleViewShapeSame,
                                                CheckOutWeightScaleStorageShapeSame,
                                                CheckOutWeightOffsetOptionalNotNullEmpty,
                                                CheckOutWeightOffsetOptionalFormatND,
                                                CheckOutWeightOffsetOptionalDtypeSame,
                                                CheckOutWeightOffsetOptionalViewShapeSame,
                                                CheckOutWeightOffsetOptionalStorageShapeSame,
                                                CheckOutBiasOptionalNotNullEmpty,
                                                CheckOutBiasOptionalFormatND,
                                                CheckOutBiasOptionalContiguous,
                                                CheckOutBiasOptionalDtypeSame,
                                                CheckOutBiasOptionalViewShapeSame,
                                                CheckOutBiasOptionalStorageShapeSame};

template <typename... Groups>
static std::vector<CheckFunc> CombineChecks(const Groups&... groups)
{
    std::vector<CheckFunc> combined;
    (combined.insert(combined.end(), groups.begin(), groups.end()), ...);
    return combined;
}

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
                     ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsMMA16S4PerTensorDataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND,
                  CheckWeightPackingDimEven,     // 紧凑 4-bit 打包维须为偶数
                  CheckWeightInt4DirectCopyView, // INT4 直拷视图须连续/严格转置且偏移字节对齐
                  CheckWeightScaleNotEmpty, CheckWeightScaleFormatND,
                  CheckWeightScaleDtype<op::DataType::DT_FLOAT16, op::DataType::DT_BF16>,
                  CheckWeightScalePerTensorViewShape,
                  CheckKGroupSizeZero, // per-tensor scale 配 kGroupSize>0 属矛盾输入
                  CheckWeightOffsetOptionalNotEmpty, CheckWeightOffsetOptionalFormatND,
                  CheckWeightOffsetOptionalDtypeSameAsScale, CheckWeightOffsetOptionalViewShapeSameAsScale,
                  CheckBiasOptionalNotEmpty, CheckBiasOptionalFormatND, CheckBiasOptionalViewShape<false>,
                  CheckBiasOptionalContiguous, CheckOutWeightNotNullEmpty, CheckOutWeightDtypeSame,
                  // per-tensor 不支持 NZ：outWeight 必须 ND/NCL，转置/非转置 weight 均直拷
                  CheckOutWeightFormatND, CheckOutWeightViewShapeSame, CheckOutWeightScaleNotNullEmpty,
                  CheckOutWeightScaleFormatND, CheckOutWeightScaleDtypeSame, CheckOutWeightScaleViewShapeSame,
                  CheckOutWeightScaleStorageShapeSame, CheckOutWeightOffsetOptionalNotNullEmpty,
                  CheckOutWeightOffsetOptionalFormatND, CheckOutWeightOffsetOptionalDtypeSame,
                  CheckOutWeightOffsetOptionalViewShapeSame, CheckOutWeightOffsetOptionalStorageShapeSame,
                  CheckOutBiasOptionalNotNullEmpty, CheckOutBiasOptionalFormatND, CheckOutBiasOptionalContiguous,
                  CheckOutBiasOptionalDtypeSame, CheckOutBiasOptionalViewShapeSame,
                  CheckOutBiasOptionalStorageShapeSame},
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-channel 转置：ND 直拷（物理透传）
      {.judge = IsMMA16S4PerChannelTransDataFlow,
       .checks = CombineChecks(INPUT_BASE_CHECKS, SCALE_DTYPE_F16_BF16_CHECK, PER_CHANNEL_SCALE_CHECKS,
                               BIAS_OFFSET_CHECKS, OUT_WEIGHT_ND_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-channel 非转置 + NZ_C0_16 出：ND→NZ 转换（非转置仅支持 NZ 出）
      {.judge = IsMMA16S4PerChannelNonTransNzDataFlow,
       .checks = CombineChecks(INPUT_BASE_CHECKS, SCALE_DTYPE_F16_BF16_CHECK, PER_CHANNEL_SCALE_CHECKS,
                               BIAS_OFFSET_CHECKS, OUT_WEIGHT_NZ_C016_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightNonTransNd2Nz<IDX_0>, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-group 转置：ND 直拷（scale 形状须匹配 per-group 分组语义）
      {.judge = IsMMA16S4PerGroupTransDataFlow,
       .checks = CombineChecks(INPUT_BASE_CHECKS, SCALE_DTYPE_F16_BF16_CHECK, PER_GROUP_SCALE_CHECKS,
                               BIAS_OFFSET_CHECKS, OUT_WEIGHT_ND_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-group 非转置 + NZ_C0_16 出：ND→NZ 转换（非转置仅支持 NZ 出）
      {.judge = IsMMA16S4PerGroupNonTransNzDataFlow,
       .checks = CombineChecks(INPUT_BASE_CHECKS, SCALE_DTYPE_F16_BF16_CHECK, PER_GROUP_SCALE_CHECKS,
                               BIAS_OFFSET_CHECKS, OUT_WEIGHT_NZ_C016_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightNonTransNd2Nz<IDX_0>, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      {.judge = IsMMA16F4PerGroupDataFlow,
       .checks = {CheckWeightNotEmpty,
                  CheckWeightFormatND,
                  CheckWeightNotTrans,       // A16F4 per-group NZ 路径仅支持非转置 weight，转置返回错误
                  CheckWeightPackingDimEven, // 紧凑 FP4 打包维须为偶数（与 INT4 同约束）
                  CheckWeightScaleNotEmpty,
                  CheckWeightScaleFormatND,
                  CheckWeightScaleDtype<op::DataType::DT_FLOAT16, op::DataType::DT_BF16>,
                  CheckKGroupSizePositive,           // 先于 PerGroupViewShape：后者要用 kGroupSize 做除法
                  CheckWeightScalePerGroupViewShape, // scale 须为 [ceildiv(K, kGroupSize), N]
                  CheckWeightOffsetOptionalNull,
                  CheckBiasOptionalNotEmpty,
                  CheckBiasOptionalFormatND,
                  CheckBiasOptionalViewShape<false>,
                  CheckBiasOptionalContiguous,
                  CheckOutWeightNotNullEmpty,
                  CheckOutWeightDtypeSame,
                  CheckOutWeightViewShapeSame,
                  CheckOutWeightNzStorageDim,
                  CheckOutWeightTransNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ_C0_16>,
                  CheckOutWeightScaleNotNullEmpty,
                  CheckOutWeightScaleFormatND,
                  CheckOutWeightScaleDtypeSame,
                  CheckOutWeightScaleViewShapeSame,
                  CheckOutWeightScaleStorageShapeSame,
                  CheckOutBiasOptionalNotNullEmpty,
                  CheckOutBiasOptionalFormatND,
                  CheckOutBiasOptionalContiguous,
                  CheckOutBiasOptionalDtypeSame,
                  CheckOutBiasOptionalViewShapeSame,
                  CheckOutBiasOptionalStorageShapeSame},
       .processes = {ProcessWeightNonTransNd2Nz<IDX_0>, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsMMA16MXF4DataFlow,
       .checks = {CheckWeightNotEmpty,
                  CheckWeightFormatND,
                  CheckWeightNotTrans, // A16MXFP4 路径仅支持非转置 weight，转置返回错误
                  CheckWeightPackingDimEven,
                  CheckWeightScaleNotEmpty,
                  CheckWeightScaleFormatND,
                  CheckWeightScaleDtype<op::DataType::DT_FLOAT8_E8M0>, // MX scale 固定 E8M0
                  CheckKGroupSizeMx, // 先于 PerGroupViewShape：后者要用 kGroupSize 做除法；MX 场景 kGroupSize 固定为 32
                  CheckWeightScalePerGroupViewShape, // MX scale 为 2D [ceildiv(K,32), N] 连续
                  CheckWeightOffsetOptionalNull,
                  CheckBiasOptionalNotEmpty,
                  CheckBiasOptionalFormatND,
                  CheckBiasOptionalViewShape<false>,
                  CheckBiasOptionalContiguous,
                  CheckOutWeightNotNullEmpty,
                  CheckOutWeightDtypeSame,
                  CheckOutWeightViewShapeSame,
                  CheckOutWeightNzStorageDim,
                  CheckOutWeightTransNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ_C0_16>,
                  CheckOutWeightScaleNotNullEmpty,
                  CheckOutWeightScaleFormatND,
                  CheckOutWeightScaleDtypeSame,
                  CheckOutWeightScaleViewShapeSame,
                  CheckOutWeightScaleStorageShapeSame,
                  CheckOutBiasOptionalNotNullEmpty,
                  CheckOutBiasOptionalFormatND,
                  CheckOutBiasOptionalContiguous,
                  CheckOutBiasOptionalDtypeSame,
                  CheckOutBiasOptionalViewShapeSame,
                  CheckOutBiasOptionalStorageShapeSame},
       .processes = {ProcessWeightNonTransNd2Nz<IDX_0>, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}}}}};
