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
 * \file im2col_infershape.cpp
 * \brief
 */
#include <cmath>
#include <climits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/input_util.h"
#include "op_host/util/shape_util.h"

using namespace ge;
using namespace std;
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
// support information
static constexpr size_t SUPPORTED_DIM_NUM = 4;

static const std::map<char, size_t> NHWC_INPUT_IDX_MAP{{'N', 0}, {'H', 1}, {'W', 2}, {'C', 3}};
static const std::map<char, size_t> NCHW_INPUT_IDX_MAP{{'N', 0}, {'C', 1}, {'H', 2}, {'W', 3}};

static inline bool IsOutShapeInvalid(int64_t in, int64_t out)
{
    return (in > 0) && (out <= 0);
}

static ge::graphStatus InferShape4Im2colCalcOut(
    gert::InferShapeContext* context, const gert::Shape* shapeIn, gert::Shape* shapeOut, const Format dataFormat,
    const std::array<int64_t, 2>& ksizes, const std::array<int64_t, 2>& strides,
    const std::array<int64_t, 2>& dilations, const std::string_view paddingMode)
{
    auto [ret, shapeNCHW] = Ops::Math::GetImgDataDimsByNCHWOrder(context, "x", *shapeIn, dataFormat);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "Get input shape failed"), return ret);

    auto [inN, inC, inH, inW] = shapeNCHW;
    auto [kernelH, kernelW] = ksizes;
    auto [strideH, strideW] = strides;
    auto [dilationH, dilationW] = dilations;

    int64_t effectiveH = (kernelH - 1) * dilationH + 1;
    int64_t effectiveW = (kernelW - 1) * dilationW + 1;
    int64_t outH{0};
    int64_t outW{0};
    int64_t outC{0};

    if (paddingMode == "CALCULATED") {
        // Get attr pads
        auto attrPads = context->GetAttrs()->GetListInt(ATTR_IDX_PADS);
        auto [ret4, pads] = Ops::Math::UnpackAdaptDimListIntAttr<4>(
            context, "pads", attrPads, [](int64_t val) { return val >= 0; }, "The value of pads cannot be negative");
        OP_CHECK_IF(ret4 != ge::GRAPH_SUCCESS, OP_LOGE(context, "pads check failed"), return ret4);
        outH = (inH == -1) ? -1 : (inH + pads[0] + pads[1] - effectiveH) / strideH + 1;
        outW = (inW == -1) ? -1 : (inW + pads[2] + pads[3] - effectiveW) / strideW + 1;
    } else if (paddingMode == "SAME") {
        outH = (inH == -1) ? -1 : (inH + strideH - 1) / strideH;
        outW = (inW == -1) ? -1 : (inW + strideW - 1) / strideW;
    } else {
        outH = (inH == -1) ? -1 : (inH - effectiveH + strideH) / strideH;
        outW = (inW == -1) ? -1 : (inW - effectiveW + strideW) / strideW;
    }

    OP_CHECK_IF(
        (IsOutShapeInvalid(inH, outH) || IsOutShapeInvalid(inW, outW)),
        OP_LOGE(
            context, "The calculated shape of the array of sliding blocks is (%ld, %ld), which must be positive", outH,
            outW),
        return ge::GRAPH_FAILED);

    outC = (inC == -1) ? -1 : inC * kernelH * kernelW;

    const std::map<char, size_t>& idxMap = dataFormat == FORMAT_NCHW ? NCHW_INPUT_IDX_MAP : NHWC_INPUT_IDX_MAP;
    shapeOut->SetDimNum(SUPPORTED_DIM_NUM);
    shapeOut->SetDim(idxMap.at('N'), inN);
    shapeOut->SetDim(idxMap.at('C'), outC);
    shapeOut->SetDim(idxMap.at('H'), outH);
    shapeOut->SetDim(idxMap.at('W'), outW);
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferShape4Im2col(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Im2col infershape funtion start!");
    // Get input desc
    const gert::CompileTimeTensorDesc* tensorDescIn = context->GetInputDesc(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorDescIn);
    // Get output desc
    const gert::CompileTimeTensorDesc* tensorDescOutput = context->GetOutputDesc(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorDescOutput);

    // Get runtime attrs
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context, "Get attrs failed."), return ge::GRAPH_FAILED);

    // Get attr ksizes
    auto attrKsizes = attrs->GetListInt(ATTR_IDX_KSIZE);
    auto [ret1, ksizes] = Ops::Math::UnpackFixedDimListIntAttr<2>(
        context, "ksizes", attrKsizes, [](int64_t val) { return val > 0; }, "The value of ksizes must be positive");
    OP_CHECK_IF(ret1 != ge::GRAPH_SUCCESS, OP_LOGE(context, "ksizes check failed"), return ret1);

    // Get attr strides
    auto attrStrides = attrs->GetListInt(ATTR_IDX_STRIDES);
    auto [ret2, strides] = Ops::Math::UnpackAdaptDimListIntAttr<2>(
        context, "strides", attrStrides, [](int64_t val) { return val > 0; }, "The value of strides must be positive");
    OP_CHECK_IF(ret2 != ge::GRAPH_SUCCESS, OP_LOGE(context, "strides check failed"), return ret2);

    // Get attr dilations
    auto attrDilations = attrs->GetListInt(ATTR_IDX_DILATIONS);
    auto [ret3, dilations] = Ops::Math::UnpackAdaptDimListIntAttr<2>(
        context, "dilations", attrDilations, [](int64_t val) { return val > 0; },
        "The value of dilations must be positive");
    OP_CHECK_IF(ret3 != ge::GRAPH_SUCCESS, OP_LOGE(context, "dilations check failed"), return ret3);

    // Get attr padding_mode
    const char* attrPaddingMode = attrs->GetStr(ATTR_IDX_PADDING_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPaddingMode);
    const std::string_view paddingMode = std::string_view(attrPaddingMode);
    OP_CHECK_IF(
        paddingMode != "VALID" && paddingMode != "SAME" && paddingMode != "CALCULATED",
        OP_LOGE(context, "The padding_mode only support VALID, SAME and CALCULATED."), return ge::GRAPH_FAILED);

    // Get input shape
    const gert::Shape* shapeIn = context->GetInputShape(X_IDX);
    auto shapeOut = context->GetOutputShape(Y_IDX);
    if (Ops::Base::IsUnknownRank(*shapeIn)) {
        Ops::Base::SetUnknownRank(*shapeOut);
        return ge::GRAPH_SUCCESS;
    }
    // Get input format
    const Format dataFormat = tensorDescIn->GetOriginFormat();

    return InferShape4Im2colCalcOut(context, shapeIn, shapeOut, dataFormat, ksizes, strides, dilations, paddingMode);
}

IMPL_OP_INFERSHAPE(Im2col).InferShape(InferShape4Im2col);
} // namespace ops
