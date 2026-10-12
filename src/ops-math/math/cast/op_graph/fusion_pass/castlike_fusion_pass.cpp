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
 * \file castlike_fusion_pass.cpp
 * \brief CastLike fusion pass (CastLike --> Cast)
 *
 *        x, y(input)               x(input), dst_type(attr)
 *           |                         |
 *        CastLike       ==>         Cast
 *           |                         |
 *           y                         y
 *
 * The key transformation:
 * - CastLike has 2 inputs: x (data tensor) and y (type reference tensor)
 * - Cast has 1 input: x, and attribute dst_type
 * - dst_type is extracted from y's data type
 */

#include <vector>
#include <string>
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "castlike_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {
namespace es = ge::es;

static const std::string kPassName = "CastlikeFusionPass";
static const int64_t kCaptureIdxCastLikeNode = 0L;

// Helper function to get inputs info
static void GetInputsInfo(const std::vector<SubgraphInput>& subgraphInputs,
                          std::vector<Shape>& inputShapes,
                          std::vector<DataType>& inputDtypes,
                          std::vector<Format>& inputFormats)
{
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

// Helper function to infer shape
static Status InferShape(const std::unique_ptr<Graph>& replaceGraph,
                         const std::vector<SubgraphInput>& subgraphInputs)
{
    OP_LOGD(kPassName.c_str(), "Begin infershape for replacement.");
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

PatternUniqPtr MakePatternCastLike()
{
    auto graph_builder = es::EsGraphBuilder(kPassName.c_str());
    auto x = graph_builder.CreateInput(0, "x");
    auto y = graph_builder.CreateInput(1, "y");
    
    ge::Graph* graph = graph_builder.GetCGraphBuilder()->GetGraph();
    GNode castlike = es::CompliantNodeBuilder(graph)
        .OpType("CastLike")
        .Name("CastLike")
        .IrDefInputs({
            {"input", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"target_type", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({
            {"output", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .Build();

    // Connect input to two input one output CastLike using AddEdgeAndUpdatePeerDesc
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), castlike, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *y.GetProducer(), y.GetProducerOutIndex(), castlike, 1);
    
    auto output = es::EsTensorHolder(graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(castlike, 0));
    auto build_graph = graph_builder.BuildAndReset({output});
    auto pattern = std::make_unique<Pattern>(std::move(*build_graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});  // Capture the Cast CastLike

    return pattern;
}

std::vector<PatternUniqPtr> CastlikeFusionPass::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns for CastlikeFusionPass");
    std::vector<PatternUniqPtr> patternGraphs;
    patternGraphs.emplace_back(MakePatternCastLike());
    return patternGraphs;
}

static bool IsTargetPlatform()
{
    PlatformInfo platform_info;
    OptionalInfo optional_info;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get platform_info failed.");
        return false;
    }
    const std::string soc = platform_info.str_info.short_soc_version;
    bool is_platform950 = (soc == "Ascend950");
    OP_LOGD(kPassName.c_str(), "Platform short soc: %s", soc.c_str());
    if (!is_platform950) {
        OP_LOGD(kPassName.c_str(), "Platform is not support, only work on Ascend950.");
        return false;
    }
    return true;
}

bool CastlikeFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements for CastlikeFusionPass");
    if (!IsTargetPlatform()) {
        OP_LOGD(kPassName.c_str(), "Check platform fail");
        return false;
    }

    (void)match_result;
    return true;
}

std::unique_ptr<Graph> CastlikeFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OP_LOGD(kPassName.c_str(), "Enter Replacement for CastlikeFusionPass");

    // 1. Get inputs info from matched subgraph
    std::vector<SubgraphInput> subgraphInputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    // 2. Get matched node to extract dst_type
    NodeIo nodeIo;
    if (match_result->GetCapturedTensor(kCaptureIdxCastLikeNode, nodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured tensor in Replacement.");
        return nullptr;
    }

    TensorDesc input1Desc;
    nodeIo.node.GetInputDesc(1, input1Desc);
    DataType dstDtype = input1Desc.GetDataType();

    OP_LOGD(kPassName.c_str(), "Building Cast node with dst_type: %d", dstDtype);

    // 3. Build replacement graph with Cast operator
    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");

    std::vector<int64_t> xDims;
    for (size_t i = 0; i < inputShapes[0].GetDimNum(); i++) {
        xDims.push_back(inputShapes[0].GetDim(i));
    }

    auto rX = replaceGraphBuilder.CreateInput(0, "x", inputDtypes[0], inputFormats[0], xDims);
    auto rY = replaceGraphBuilder.CreateInput(1, "y", inputDtypes[1], inputFormats[1], xDims);
    auto rOutput = es::Cast(rX, static_cast<int64_t>(dstDtype));
    (void)rY;

    auto replaceGraph = replaceGraphBuilder.BuildAndReset(std::vector<ge::es::EsTensorHolder>{rOutput});
    // Infer shape
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }

    OP_LOGI(kPassName.c_str(), "CastlikeFusionPass fusion success!");
    return replaceGraph;
}

REG_FUSION_PASS(CastlikeFusionPass).Stage(CustomPassStage::kAfterInferShape);

} // namespace ops