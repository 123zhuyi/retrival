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
 * \file histogram_fusion_pass.cpp
 * \brief histogram fusion pass (Histogram --> HistogramV2)
 *
 * Pattern:
 *              x                     x, min, max
 *              |                     |
 *        Histogram    ==>    HistogramV2
 *              |                     |
 *              y                     y
 *
 * Key differences:
 * - Histogram: min/max are attributes
 * - HistogramV2: min/max are input tensors
 * - min/max tensors dtype should match x's dtype
 * - is_out_dtype_int32 attr depends on x dtype (false for float/f16)
 */

#include <vector>
#include <string>
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "histogram_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {

static const std::string kPassName = "HistogramFusionPass";
static const int64_t kCaptureHistogramNode = 0l;

std::vector<PatternUniqPtr> HistogramFusionPass::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns for HistogramFusionPass");
    std::vector<PatternUniqPtr> patternGraphs;

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());

    auto x = graphBuilder.CreateInput(0);

    auto output = es::Histogram(x, 0.0, 0.0, 100);

    auto graph = graphBuilder.BuildAndReset({output});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});

    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

bool HistogramFusionPass::MeetRequirements(const std::unique_ptr<MatchResult> &match_result)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements for HistogramFusionPass");

    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get platformInfo failed.");
        return false;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    if (soc != "Ascend950") {
        OP_LOGD(kPassName.c_str(), "Platform %s is not supported, only Ascend950.", soc.c_str());
        return false;
    }

    NodeIo histogramNodeIo;
    if (match_result->GetCapturedTensor(kCaptureHistogramNode, histogramNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to GetCaptured tensor");
        return false;
    }
    AscendString nodeType;
    histogramNodeIo.node.GetType(nodeType);
    std::string typeStr = nodeType.GetString();
    if (typeStr != "Histogram") {
        OP_LOGD(kPassName.c_str(), "Node type %s is not Histogram, skip.", typeStr.c_str());
        return false;
    }

    TensorDesc inputDesc;
    histogramNodeIo.node.GetInputDesc(0, inputDesc);
    DataType inputDtype = inputDesc.GetDataType();
    if (inputDtype != DT_FLOAT16 && inputDtype != DT_FLOAT && inputDtype != DT_INT64 && inputDtype != DT_INT32 &&
        inputDtype != DT_INT16 && inputDtype != DT_INT8 && inputDtype != DT_UINT8) {
        OP_LOGD(kPassName.c_str(), "Input dtype %d not supported, skip.", inputDtype);
        return false;
    }

    TensorDesc outputDesc;
    histogramNodeIo.node.GetOutputDesc(0, outputDesc);
    DataType outputDtype = outputDesc.GetDataType();
    if (outputDtype != DT_FLOAT && outputDtype != DT_INT32) {
        OP_LOGD(kPassName.c_str(), "Output dtype %d is not DT_FLOAT or DT_INT32, skip fusion.", outputDtype);
        return false;
    }

    return true;
}

std::unique_ptr<Graph> HistogramFusionPass::Replacement(const std::unique_ptr<MatchResult> &match_result)
{
    OP_LOGD(kPassName.c_str(), "Enter Replacement for HistogramFusionPass");

    std::vector<SubgraphInput> subgraphInputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    NodeIo histogramNodeIo;
    if (match_result->GetCapturedTensor(kCaptureHistogramNode, histogramNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to GetCaptured tensor in Replacement");
        return nullptr;
    }

    int64_t bins = 100;
    histogramNodeIo.node.GetAttr("bins", bins);
    OP_LOGD(kPassName.c_str(), "bins: %ld", bins);

    float minVal = 0.0f;
    histogramNodeIo.node.GetAttr("min", minVal);
    OP_LOGD(kPassName.c_str(), "min: %f", minVal);

    float maxVal = 0.0f;
    histogramNodeIo.node.GetAttr("max", maxVal);
    OP_LOGD(kPassName.c_str(), "max: %f", maxVal);

    DataType inputDtype = inputDtypes[0];
    DataType yDtype = inputDtype;
    if (inputDtype == DT_FLOAT || inputDtype == DT_FLOAT16) {
        yDtype = DT_FLOAT;
    }
    OP_LOGD(kPassName.c_str(), "input dtype: %d, y_dtype: %d", inputDtype, yDtype);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");

    std::vector<int64_t> xDims;
    for (size_t i = 0; i < inputShapes[0].GetDimNum(); i++) {
        xDims.push_back(inputShapes[0].GetDim(i));
    }

    auto rX = replaceGraphBuilder.CreateInput(0, "x", inputDtypes[0], inputFormats[0], xDims);

    auto rMin = replaceGraphBuilder.CreateScalar(minVal);
    auto rMax = replaceGraphBuilder.CreateScalar(maxVal);

    auto output = es::HistogramV2(rX, rMin, rMax, bins, yDtype);

    std::vector<es::EsTensorHolder> outputs = {output};
    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset(outputs);
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Infershape for replacement failed.");
        return nullptr;
    }

    return replaceGraph;
}

static void GetInputsInfo(const std::vector<SubgraphInput> &subgraphInputs, std::vector<Shape> &inputShapes,
    std::vector<DataType> &inputDtypes, std::vector<Format> &inputFormats)
{
    for (const auto &subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

static Status InferShape(const GraphUniqPtr &replaceGraph, const std::vector<SubgraphInput> &subgraphInputs)
{
    OP_LOGD(kPassName.c_str(), "Begin infershape for replacements.");
    std::vector<Shape> inputShapes;
    for (const auto &subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

REG_FUSION_PASS(HistogramFusionPass).Stage(CustomPassStage::kAfterInferShape);

}  // namespace ops