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
 * \file im2col_infershape.cpp
 * \brief
 */
#include <array>
#include <limits>
#include <string_view>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/input_util.h"
#include "op_host/util/shape_util.h"

using namespace ge;
namespace ops {
// proto input
static constexpr size_t X_IDX = 0;
// proto output
static constexpr size_t Y_IDX = 0;
// proto attributes
static constexpr size_t ATTR_IDX_KSIZE = 0;
static constexpr size_t ATTR_IDX_STRIDES = 1;
static constexpr size_t ATTR_IDX_DILATIONS = 2;
static constexpr size_t ATTR_IDX_PADDING_MODE = 3;
static constexpr size_t ATTR_IDX_PADS = 4;
static constexpr size_t OUTPUT_DIM_NUM = 3;
static constexpr size_t PAIR_VALUE_COUNT = 2;
static constexpr size_t PADS_VALUE_COUNT = 4;
static constexpr size_t PAD_TOP_INDEX = 0;
static constexpr size_t PAD_BOTTOM_INDEX = 1;
static constexpr size_t PAD_LEFT_INDEX = 2;
static constexpr size_t PAD_RIGHT_INDEX = 3;
static constexpr size_t OUTPUT_N_INDEX = 0;
static constexpr size_t OUTPUT_C_INDEX = 1;
static constexpr size_t OUTPUT_SPATIAL_INDEX = 2;
static constexpr int64_t UNKNOWN_DIM = -1;

static bool SafeMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool CalculateOutputDim(int64_t input, int64_t kernel, int64_t stride, int64_t dilation, int64_t padBefore,
                               int64_t padAfter, int64_t& output)
{
    if (input == UNKNOWN_DIM) {
        output = UNKNOWN_DIM;
        return true;
    }
    if (input <= 0) {
        return false;
    }
    const __int128 effective = static_cast<__int128>(kernel - 1) * dilation + 1;
    const __int128 padding = static_cast<__int128>(padBefore) + padAfter;
    const __int128 numerator = static_cast<__int128>(input) + padding - effective;
    if (numerator < 0) {
        return false;
    }
    const __int128 result = numerator / stride + 1;
    if (result <= 0 || result > std::numeric_limits<int64_t>::max()) {
        return false;
    }
    output = static_cast<int64_t>(result);
    return true;
}

static ge::graphStatus InferShape4Im2colCalcOut(gert::InferShapeContext* context, const gert::Shape* shapeIn,
                                                gert::Shape* shapeOut, const Format dataFormat,
                                                const std::array<int64_t, PAIR_VALUE_COUNT>& ksizes,
                                                const std::array<int64_t, PAIR_VALUE_COUNT>& strides,
                                                const std::array<int64_t, PAIR_VALUE_COUNT>& dilations)
{
    auto [ret, shapeNCHW] = Ops::Math::GetImgDataDimsByNCHWOrder(context, "x", *shapeIn, dataFormat);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "Get input shape failed"), return ret);

    auto [inN, inC, inH, inW] = shapeNCHW;
    auto [kernelH, kernelW] = ksizes;
    auto [strideH, strideW] = strides;
    auto [dilationH, dilationW] = dilations;

    auto attrPads = context->GetAttrs()->GetListInt(ATTR_IDX_PADS);
    auto [padsStatus, pads] = Ops::Math::UnpackAdaptDimListIntAttr<PADS_VALUE_COUNT>(
        context, "pads", attrPads, [](int64_t val) { return val >= 0; }, "The value of pads cannot be negative");
    OP_CHECK_IF(padsStatus != ge::GRAPH_SUCCESS, OP_LOGE(context, "pads check failed"), return padsStatus);

    int64_t outH = 0;
    int64_t outW = 0;
    OP_CHECK_IF(
        !CalculateOutputDim(inH, kernelH, strideH, dilationH, pads[PAD_TOP_INDEX], pads[PAD_BOTTOM_INDEX], outH),
        OP_LOGE(context, "calculated output height is invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CalculateOutputDim(inW, kernelW, strideW, dilationW, pads[PAD_LEFT_INDEX], pads[PAD_RIGHT_INDEX], outW),
        OP_LOGE(context, "calculated output width is invalid"), return ge::GRAPH_FAILED);

    int64_t outC = UNKNOWN_DIM;
    if (inC != UNKNOWN_DIM) {
        int64_t channelProduct = 0;
        OP_CHECK_IF(!SafeMul(inC, kernelH, channelProduct) || !SafeMul(channelProduct, kernelW, channelProduct) ||
                        channelProduct <= 0,
                    OP_LOGE(context, "calculated output channel size is invalid"), return ge::GRAPH_FAILED);
        outC = channelProduct;
    }
    int64_t outSpatial = UNKNOWN_DIM;
    if (outH != UNKNOWN_DIM && outW != UNKNOWN_DIM) {
        int64_t spatialProduct = 0;
        OP_CHECK_IF(!SafeMul(outH, outW, spatialProduct), OP_LOGE(context, "calculated output spatial size overflows"),
                    return ge::GRAPH_FAILED);
        outSpatial = spatialProduct;
    }

    shapeOut->SetDimNum(OUTPUT_DIM_NUM);
    shapeOut->SetDim(OUTPUT_N_INDEX, inN);
    shapeOut->SetDim(OUTPUT_C_INDEX, outC);
    shapeOut->SetDim(OUTPUT_SPATIAL_INDEX, outSpatial);
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferShape4Im2col(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Im2col infershape function start!");
    // Get input desc
    const gert::CompileTimeTensorDesc* tensorDescIn = context->GetInputDesc(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorDescIn);
    // Get runtime attrs
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context, "Get attrs failed."), return ge::GRAPH_FAILED);

    // Get attr ksizes
    auto attrKsizes = attrs->GetListInt(ATTR_IDX_KSIZE);
    auto [ksizesStatus, ksizes] = Ops::Math::UnpackFixedDimListIntAttr<PAIR_VALUE_COUNT>(
        context, "ksizes", attrKsizes, [](int64_t val) { return val > 0; }, "The value of ksizes must be positive");
    OP_CHECK_IF(ksizesStatus != ge::GRAPH_SUCCESS, OP_LOGE(context, "ksizes check failed"), return ksizesStatus);

    // Get attr strides
    auto attrStrides = attrs->GetListInt(ATTR_IDX_STRIDES);
    auto [stridesStatus, strides] = Ops::Math::UnpackAdaptDimListIntAttr<PAIR_VALUE_COUNT>(
        context, "strides", attrStrides, [](int64_t val) { return val > 0; }, "The value of strides must be positive");
    OP_CHECK_IF(stridesStatus != ge::GRAPH_SUCCESS, OP_LOGE(context, "strides check failed"), return stridesStatus);

    // Get attr dilations
    auto attrDilations = attrs->GetListInt(ATTR_IDX_DILATIONS);
    auto [dilationsStatus, dilations] = Ops::Math::UnpackAdaptDimListIntAttr<PAIR_VALUE_COUNT>(
        context, "dilations", attrDilations, [](int64_t val) { return val > 0; },
        "The value of dilations must be positive");
    OP_CHECK_IF(dilationsStatus != ge::GRAPH_SUCCESS, OP_LOGE(context, "dilations check failed"),
                return dilationsStatus);

    // Get attr padding_mode
    const char* attrPaddingMode = attrs->GetStr(ATTR_IDX_PADDING_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPaddingMode);
    const std::string_view paddingMode = std::string_view(attrPaddingMode);
    OP_CHECK_IF(paddingMode != "CALCULATED", OP_LOGE(context, "padding_mode only supports CALCULATED"),
                return ge::GRAPH_FAILED);

    // Get input shape
    const gert::Shape* shapeIn = context->GetInputShape(X_IDX);
    auto shapeOut = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeIn);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeOut);
    if (Ops::Base::IsUnknownRank(*shapeIn)) {
        Ops::Base::SetUnknownRank(*shapeOut);
        return ge::GRAPH_SUCCESS;
    }
    // Get input format
    const Format dataFormat = tensorDescIn->GetOriginFormat();

    return InferShape4Im2colCalcOut(context, shapeIn, shapeOut, dataFormat, ksizes, strides, dilations);
}

IMPL_OP_INFERSHAPE(Im2col).InferShape(InferShape4Im2col);
} // namespace ops
