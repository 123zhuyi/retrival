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
 * \file random_infershape_base.cpp
 * \brief
 */
#include "random_infershape_base.h"

namespace ops {
namespace randomCommon {
template <typename T>
ge::graphStatus HandleShapeTensor(gert::Shape& outputShape, size_t xShapeSize, const T* xShapeData)
{
    std::cerr << "[DEBUG] HandleShapeTensor with type: " << typeid(T).name() << ", dims " << xShapeSize << std::endl;
    outputShape.SetDimNum(xShapeSize);
    for (size_t i = 0U; i < xShapeSize; i++) {
        outputShape.SetDim(i, xShapeData[i]);
    }
    return ge::GRAPH_SUCCESS;
}

bool InferShapeForUnknow(
    gert::InferShapeContext* context, const gert::Shape& inShape, gert::Shape& outShape, int64_t& maskIndex,
    int64_t& offsetIndex)
{
    if (Ops::Base::IsUnknownRank(inShape)) {
        Ops::Base::SetUnknownRank(outShape);
        if (maskIndex >= 0) {
            gert::Shape* maskOutputShape = context->GetOutputShape(maskIndex);
            Ops::Base::SetUnknownRank(*maskOutputShape);
        }
        if (offsetIndex >= 0) {
            gert::Shape* offsetOutputShape = context->GetOutputShape(offsetIndex);
            Ops::Base::SetUnknownRank(*offsetOutputShape);
        }
        return true;
    }
    if (Ops::Base::IsUnknownShape(inShape)) {
        Ops::Base::SetUnknownShape(inShape.GetDimNum(), outShape);
        if (maskIndex >= 0) {
            gert::Shape* maskOutputShape = context->GetOutputShape(maskIndex);
            Ops::Base::SetUnknownShape(1, *maskOutputShape);
        }
        if (offsetIndex >= 0) {
            gert::Shape* offsetOutputShape = context->GetOutputShape(offsetIndex);
            Ops::Base::SetUnknownShape(1, *offsetOutputShape);
            ;
        }
        return true;
    }
    return false;
}

bool DependencyMode(const gert::Tensor* inTensor, gert::Shape& outShape, size_t xShapeSize)
{
    ge::DataType shapeDtype = inTensor->GetDataType();
    if (shapeDtype == ge::DT_INT32) {
        auto xShapeData = inTensor->GetData<int32_t>();
        if (xShapeData == nullptr) {
            std::cerr << "[WARN] Empty DT_INT32 shape tensor, set 0-dim output" << std::endl;
            Ops::Base::SetUnknownShape(xShapeSize, outShape);
            return true;
        }
        if (HandleShapeTensor<int32_t>(outShape, xShapeSize, xShapeData) == ge::GRAPH_SUCCESS) {
            return true;
        }
    } else if (shapeDtype == ge::DT_INT64) {
        auto xShapeData = inTensor->GetData<int64_t>();
        if (xShapeData == nullptr) {
            std::cerr << "[WARN] Empty DT_INT64 shape tensor, set 0-dim output" << std::endl;
            Ops::Base::SetUnknownShape(xShapeSize, outShape);
            return true;
        }
        if (HandleShapeTensor<int64_t>(outShape, xShapeSize, xShapeData) == ge::GRAPH_SUCCESS) {
            return true;
        }
    }
    std::cerr << "[ERROR] Unsupported dtype: " << static_cast<int>(shapeDtype) << std::endl;
    return false;
}

bool InputAndOutputCheck(
    gert::InferShapeContext* context, const std::unordered_map<std::string, size_t>& requiredInputMap,
    const std::unordered_map<std::string, size_t>& outputMap, int64_t& maskIndex, int64_t& offsetIndex,
    const std::unordered_map<std::string, size_t>& optionalInputMap)
{
    OP_LOGD(context->GetNodeName(), "InputAndOutputCheck start");
    for (const auto& item : requiredInputMap) {
        size_t inputIndex = item.second;
        auto inputShape = context->GetRequiredInputShape(inputIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    }

    for (const auto& item : optionalInputMap) {
        size_t inputIndex = item.second;
        auto inputShape = context->GetOptionalInputShape(inputIndex);
        if (inputShape != nullptr) {
            OP_LOGD(context->GetNodeName(), "Optional input %zu is provided", inputIndex);
        }
    }

    for (const auto& item : outputMap) {
        const std::string& outputName = item.first;
        size_t outputIndex = item.second;
        auto output = context->GetOutputShape(outputIndex);
        OP_CHECK_NULL_WITH_CONTEXT(context, output);
        if (outputName == "mask") {
            maskIndex = outputIndex;
        }
        if (outputName == "offset") {
            offsetIndex = outputIndex;
        }
    }
    OP_LOGD(
        context->GetNodeName(), "InputAndOutputCheck end, maskIndex = %ld, offsetIndex = %ld", maskIndex, offsetIndex);
    return true;
}

ge::graphStatus CommonInferShape(
    gert::InferShapeContext* context, const std::unordered_map<std::string, size_t>& requiredInputMap,
    const std::unordered_map<std::string, size_t>& outputMap, int32_t mode,
    const std::unordered_map<std::string, size_t>& optionalInputMap)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t maskIndex = -1;
    int64_t offsetIndex = -1;
    if (!InputAndOutputCheck(context, requiredInputMap, outputMap, maskIndex, offsetIndex, optionalInputMap)) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* inShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto* outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    int64_t xShapeSize = inShape->GetShapeSize();
    if (InferShapeForUnknow(context, *inShape, *outShape, maskIndex, offsetIndex)) {
        OP_LOGI(context->GetNodeName(), "Success unknown shape");
        return ge::GRAPH_SUCCESS;
    }
    if (maskIndex >= 0) {
        static constexpr int64_t MASK_ALIGN_SIZE = 128;
        static constexpr int64_t MASK_BIT_TO_UINT8 = 8;
        gert::Shape* maskOutputShape = context->GetOutputShape(maskIndex);
        int64_t maskSize = (xShapeSize + MASK_ALIGN_SIZE - 1) / MASK_ALIGN_SIZE * MASK_ALIGN_SIZE / MASK_BIT_TO_UINT8;
        maskOutputShape->SetDimNum(1);
        maskOutputShape->SetDim(0, maskSize);
    }
    if (offsetIndex >= 0) {
        gert::Shape* offsetOutputShape = context->GetOutputShape(offsetIndex);
        offsetOutputShape->SetDimNum(1);
        offsetOutputShape->SetDim(0, 1);
    }
    if (mode == MODE_NO_DEPENDENCY) {
        *outShape = *inShape;
        OP_LOGI(context->GetNodeName(), "Success no dependency Mode");
        return ge::GRAPH_SUCCESS;
    }
    if (mode == MODE_DEPENDENCY) {
        const gert::Tensor* inTensor = context->GetInputTensor(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, inTensor);
        if (DependencyMode(inTensor, *outShape, static_cast<size_t>(xShapeSize))) {
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGE(context->GetNodeName(), "Failed to infer shape! mode = %d, xShapeSize=%ld", mode, xShapeSize);
    return ge::GRAPH_FAILED;
}
} // namespace randomCommon
} // namespace ops
