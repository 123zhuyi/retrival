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
 * \file gcd_tiling.cpp
 * \brief Gcd tiling
 */

#include "experimental/math/gcd/op_kernel/gcd_tiling_data.h"
#include "experimental/math/gcd/op_kernel/gcd_tiling_key.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace {

constexpr int64_t GCD_MIN_OUTPUT_WORDS_PER_BLOCK = 256;

struct GcdCompileInfo {};

static int64_t CeilDiv(int64_t value, int64_t divisor)
{
    if (divisor <= 0) {
        return 0;
    }
    const int64_t safeDivisor = divisor;
    return (value + safeDivisor - 1) / safeDivisor;
}

static ge::graphStatus GetCoreNum(gert::TilingContext* context, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(gert::TilingContext* context)
{
    auto input0Desc = context->GetInputDesc(0);
    auto input1Desc = context->GetInputDesc(1);
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, input1Desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);

    ge::DataType x1Type = input0Desc->GetDataType();
    ge::DataType x2Type = input1Desc->GetDataType();
    ge::DataType yType = outputDesc->GetDataType();
    bool sameSupported = x1Type == x2Type && x1Type == yType &&
                         (x1Type == ge::DT_FLOAT || x1Type == ge::DT_FLOAT16 || x1Type == ge::DT_BF16 ||
                          x1Type == ge::DT_UINT8 || x1Type == ge::DT_INT8 || x1Type == ge::DT_INT16 ||
                          x1Type == ge::DT_INT32 || x1Type == ge::DT_INT64);
    bool mixedFusedSupported = yType == ge::DT_UINT8 && ((x1Type == ge::DT_UINT8 && x2Type == ge::DT_BF16) ||
                                                         (x1Type == ge::DT_BF16 && x2Type == ge::DT_UINT8));
    mixedFusedSupported = mixedFusedSupported ||
                          (yType == ge::DT_INT8 && ((x1Type == ge::DT_INT8 && x2Type == ge::DT_FLOAT) ||
                                                    (x1Type == ge::DT_FLOAT && x2Type == ge::DT_INT8)));
    mixedFusedSupported = mixedFusedSupported ||
                          (yType == ge::DT_INT16 && ((x1Type == ge::DT_INT16 && x2Type == ge::DT_FLOAT16) ||
                                                     (x1Type == ge::DT_FLOAT16 && x2Type == ge::DT_INT16)));
    OP_CHECK_IF(!sameSupported && !mixedFusedSupported,
                OP_LOGE(context, "Gcd supports same dtype triples plus selected small mixed-output triples"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetOutputWordCount(gert::TilingContext* context, int64_t totalNum, int64_t& wordCount)
{
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);

    ge::DataType dtype = outputDesc->GetDataType();
    if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
        wordCount = CeilDiv(totalNum, 4);
    } else if (dtype == ge::DT_INT16 || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        wordCount = CeilDiv(totalNum, 2);
    } else if (dtype == ge::DT_INT64) {
        wordCount = totalNum * 2;
    } else {
        wordCount = totalNum;
    }

    OP_CHECK_IF(wordCount <= 0, OP_LOGE(context, "Gcd output word count must be positive"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static int64_t SelectBlockDim(int64_t coreNum, int64_t outputWordCount)
{
    int64_t blockDim = CeilDiv(outputWordCount, GCD_MIN_OUTPUT_WORDS_PER_BLOCK);
    if (blockDim < 1) {
        blockDim = 1;
    }
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }
    return blockDim;
}

template <typename ShapeT>
static int64_t GetRank(const ShapeT& shape)
{
    int64_t rank = static_cast<int64_t>(shape.GetDimNum());
    return rank == 0 ? 1 : rank;
}

template <typename ShapeT>
static int64_t GetDimOrOne(const ShapeT& shape, int64_t dim)
{
    if (shape.GetDimNum() == 0) {
        return 1;
    }
    return shape.GetDim(dim);
}

template <typename ShapeT>
static ge::graphStatus FillOutputDims(gert::TilingContext* context, const ShapeT& outputShape, GcdTilingData* tiling)
{
    tiling->rank = GetRank(outputShape);
    OP_CHECK_IF(tiling->rank <= 0 || tiling->rank > GCD_MAX_DIMS, OP_LOGE(context, "Gcd output rank must be in [1, 8]"),
                return ge::GRAPH_FAILED);

    tiling->totalNum = 1;
    for (int64_t i = 0; i < tiling->rank; ++i) {
        int64_t dim = GetDimOrOne(outputShape, i);
        OP_CHECK_IF(dim <= 0, OP_LOGE(context, "Gcd output dim must be positive"), return ge::GRAPH_FAILED);
        tiling->outputDims[i] = dim;
        tiling->totalNum *= dim;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename ShapeT>
static ge::graphStatus FillInputStrides(gert::TilingContext* context, const ShapeT& inputShape,
                                        const GcdTilingData* tiling, int64_t* strides)
{
    int64_t inputRankRaw = static_cast<int64_t>(inputShape.GetDimNum());
    OP_CHECK_IF(inputRankRaw < 0 || inputRankRaw > GCD_MAX_DIMS, OP_LOGE(context, "Gcd input rank must be in [0, 8]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputRankRaw > tiling->rank, OP_LOGE(context, "Gcd input rank cannot exceed output rank"),
                return ge::GRAPH_FAILED);

    int64_t contiguousStrides[GCD_MAX_DIMS] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t runningStride = 1;
    for (int64_t i = inputRankRaw - 1; i >= 0; --i) {
        int64_t dim = inputShape.GetDim(i);
        OP_CHECK_IF(dim <= 0, OP_LOGE(context, "Gcd input dim must be positive"), return ge::GRAPH_FAILED);
        contiguousStrides[i] = runningStride;
        runningStride *= dim;
    }

    int64_t leading = tiling->rank - inputRankRaw;
    for (int64_t outDim = 0; outDim < tiling->rank; ++outDim) {
        int64_t inputDim = outDim - leading;
        if (inputDim < 0) {
            strides[outDim] = 0;
            continue;
        }
        int64_t dim = inputShape.GetDim(inputDim);
        int64_t outputDim = tiling->outputDims[outDim];
        if (dim == outputDim) {
            strides[outDim] = contiguousStrides[inputDim];
        } else if (dim == 1) {
            strides[outDim] = 0;
        } else {
            OP_LOGE(context, "Gcd input shape is not broadcast-compatible with output shape");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillShapeInfo(gert::TilingContext* context, GcdTilingData* tiling)
{
    auto x1ShapeInfo = context->GetInputShape(0);
    auto x2ShapeInfo = context->GetInputShape(1);
    auto yShapeInfo = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1ShapeInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2ShapeInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapeInfo);

    auto x1Shape = x1ShapeInfo->GetStorageShape();
    auto x2Shape = x2ShapeInfo->GetStorageShape();
    auto yShape = yShapeInfo->GetStorageShape();
    OP_CHECK_IF(FillOutputDims(context, yShape, tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "FillOutputDims failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(FillInputStrides(context, x1Shape, tiling, tiling->x1Strides) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Fill x1 strides failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(FillInputStrides(context, x2Shape, tiling, tiling->x2Strides) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Fill x2 strides failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForGcd(gert::TilingContext* context)
{
    OP_LOGD("GcdTiling", "Enter TilingForGcd");
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    int64_t coreNum = 0;
    OP_CHECK_IF(GetCoreNum(context, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetCoreNum failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckDtype failed"),
                return ge::GRAPH_FAILED);

    GcdTilingData* tiling = context->GetTilingData<GcdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    *tiling = GcdTilingData{};
    OP_CHECK_IF(FillShapeInfo(context, tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "FillShapeInfo failed"),
                return ge::GRAPH_FAILED);

    int64_t outputWordCount = 0;
    OP_CHECK_IF(GetOutputWordCount(context, tiling->totalNum, outputWordCount) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetOutputWordCount failed"), return ge::GRAPH_FAILED);
    context->SetBlockDim(SelectBlockDim(coreNum, outputWordCount));
    context->SetTilingKey(GCD_TILING_KEY);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGcd([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_OPTILING(Gcd).Tiling(TilingForGcd).TilingParse<GcdCompileInfo>(TilingParseForGcd);
} // namespace optiling
