/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"
#include "reduce_mean_with_cast_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {

// The ReduceMeanWithCastFusionPass uses the new ES framework (es::EsGraphBuilder,
// es::CompliantNodeBuilder, PatternFusionPass) introduced in GE 9.0.0.
// D1 scenario: uses kCompatibleInherited stage (9.0.0+).
// Strategy: compile-time macro guard + runtime version check + overall silence.
#define GE_COMPILER_VERSION_900 90000000
#define GE_COMPILER_VERSION_910 90100000
#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900

// Weak declare aclsysGetVersionNum to avoid hard link dependency on libascendcl.
// At runtime: if GE >= 9.0.0, symbol resolves normally; if GE 8.5.0, pointer is NULL.
extern "C" {
__attribute__((weak)) int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum);
}

const std::string kPassName = "ReduceMeanWithCastFusionPass";
const int64_t kCaptureTensorIdx = 0;

namespace {
CustomPassStage GetReduceMeanWithCastPassStage()
{
    int32_t version = 0;
    char pkgName[] = "ge_compiler";
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(pkgName, &version);
    }
    if (version >= GE_COMPILER_VERSION_900) {
        return CustomPassStage::kCompatibleInherited;
    }
    return CustomPassStage::kBeforeInferShape; // fallback to old stage for 8.5.0
}
} // anonymous namespace

static void GetInputsInfo(const std::vector<SubgraphInput>& subgraphInputs, std::vector<Shape>& inputShapes,
                          std::vector<DataType>& inputDtypes, std::vector<Format>& inputFormats)
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

static Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    OP_LOGD(kPassName.c_str(), "Begin InferShape for replacement.");
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

static bool IsTargetVersion()
{
    int32_t version = 0;
    char pkgName[] = "ge_compiler";
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(pkgName, &version);
    }
    if (version >= GE_COMPILER_VERSION_910) {
        return true;
    }
    return false;
}

std::vector<PatternUniqPtr> ReduceMeanWithCastFusionPass::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns for ReduceMeanWithCastFusionPass");
    std::vector<PatternUniqPtr> patternGraphs;

    if (!IsTargetVersion()) {
        return patternGraphs;
    }

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());

    // Create input x (index 0)
    auto x = graphBuilder.CreateInput(0);
    // Create input axes (index 1)
    auto axes = graphBuilder.CreateInput(1);

    // Build ReduceMeanWithCast node using CompliantNodeBuilder (no ES API available)
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto reduceMeanWithCast = es::CompliantNodeBuilder(graph)
                                  .OpType("ReduceMeanWithCast")
                                  .Name("reduce_mean_with_cast")
                                  .IrDefInputs({
                                      {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  })
                                  .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                  .Build();
    // Connect x to ReduceMeanWithCast input 0
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), reduceMeanWithCast, 0) !=
        GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(kPassName.c_str(), "Failed to add edge for x input in pattern");
        return patternGraphs;
    }

    // Connect axes to ReduceMeanWithCast input 1
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *axes.GetProducer(), axes.GetProducerOutIndex(), reduceMeanWithCast, 1) !=
        GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(kPassName.c_str(), "Failed to add edge for axes input in pattern");
        return patternGraphs;
    }

    // Get output tensor holder
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reduceMeanWithCast, 0);

    // Build graph and create pattern
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    auto builtGraph = graphBuilder.BuildAndReset(outputs);
    auto pattern = std::make_unique<Pattern>(std::move(*builtGraph));

    // Capture the output tensor of ReduceMeanWithCast
    NodeIo nodeIo = {y->GetProducer(), 0};
    pattern->CaptureTensor(nodeIo);

    patternGraphs.emplace_back(std::move(pattern));
    OP_LOGD(kPassName.c_str(), "Pattern created successfully for ReduceMeanWithCastFusionPass");
    return patternGraphs;
}

bool ReduceMeanWithCastFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements for ReduceMeanWithCastFusionPass");

    // Runtime version check: on GE 8.5.0, return false to no-op.
    int32_t version = 0;
    char pkgName[] = "ge_compiler";
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(pkgName, &version);
    }
    if (version < GE_COMPILER_VERSION_900) {
        OP_LOGD(kPassName.c_str(), "GE runtime version %d < 90000000, skip pass.", version);
        return false;
    }

    // Get captured tensor to verify the match is valid
    NodeIo matchedNodeIo;
    auto status = matchResult->GetCapturedTensor(kCaptureTensorIdx, matchedNodeIo);
    if (status != SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(kPassName.c_str(), "Failed to get captured tensor");
        return false;
    }

    // No platform differentiation (不区分平台): always return true
    OP_LOGD(kPassName.c_str(), "MeetRequirements returns TRUE");
    return true;
}

GraphUniqPtr ReduceMeanWithCastFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter Replacement for ReduceMeanWithCastFusionPass");

    // Get subgraph boundary inputs (x and axes)
    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    // Get matched ReduceMeanWithCast node to read attributes
    NodeIo matchedNodeIo;
    if (matchResult->GetCapturedTensor(kCaptureTensorIdx, matchedNodeIo) != SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(kPassName.c_str(), "Failed to get captured tensor in Replacement");
        return nullptr;
    }
    auto matchedNode = matchedNodeIo.node;

    // Read attributes from the matched ReduceMeanWithCast node
    bool keepDims = false;
    bool noopWithEmptyAxes = true;
    matchedNode.GetAttr("keep_dims", keepDims);
    matchedNode.GetAttr("noop_with_empty_axes", noopWithEmptyAxes);

    // Read dtype attribute (Type type in proto, stored as ge::DataType enum)
    ge::DataType dataType = ge::DT_UNDEFINED;
    bool hasDtype = (matchedNode.GetAttr("dtype", dataType) == GRAPH_SUCCESS);
    // DT_UNDEFINED means no cast needed
    if (dataType == ge::DT_UNDEFINED) {
        hasDtype = false;
    }

    OP_LOGD(kPassName.c_str(), "hasDtype=%d, dataType=%d, keep_dims=%d, noop_with_empty_axes=%d",
            static_cast<int>(hasDtype), static_cast<int>(dataType), static_cast<int>(keepDims),
            static_cast<int>(noopWithEmptyAxes));

    // Build replacement graph
    auto replaceBuilder = es::EsGraphBuilder("replacement");

    // Create input x
    auto rX = replaceBuilder.CreateInput(0, "x", inputDtypes[0], inputFormats[0], inputShapes[0].GetDims());
    // Create input axes
    auto rAxes = replaceBuilder.CreateInput(1, "axes", inputDtypes[1], inputFormats[1], inputShapes[1].GetDims());

    auto* graph = replaceBuilder.GetCGraphBuilder()->GetGraph();

    // Determine the input to ReduceMean: either original x or cast output
    GNode reduceMeanInputNode;
    int64_t reduceMeanInputIdx = 0;
    bool useCast = false;

    if (hasDtype) {
        // Build Cast node
        auto castNode = es::CompliantNodeBuilder(graph)
                            .OpType("Cast")
                            .Name("cast_node")
                            .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                            .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                            .IrDefAttrs({
                                {"dst_type", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                 es::CreateFrom(static_cast<int64_t>(dataType))},
                            })
                            .Build();
        // Connect x to Cast input 0
        if (es::AddEdgeAndUpdatePeerDesc(*graph, *rX.GetProducer(), rX.GetProducerOutIndex(), castNode, 0) !=
            GRAPH_SUCCESS) {
            OP_LOGE_WITHOUT_REPORT(kPassName.c_str(), "Failed to add edge for Cast input");
            return nullptr;
        }

        reduceMeanInputNode = castNode;
        reduceMeanInputIdx = 0;
        useCast = true;
        OP_LOGD(kPassName.c_str(), "Cast node built and connected, dst_type=%d", static_cast<int>(dataType));
    }

    // Build ReduceMean node
    auto reduceMeanNode = es::CompliantNodeBuilder(graph)
                              .OpType("ReduceMean")
                              .Name("reduce_mean")
                              .IrDefInputs({
                                  {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                              })
                              .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .IrDefAttrs({
                                  {"keep_dims", es::CompliantNodeBuilder::kEsAttrRequired, "Bool",
                                   es::CreateFrom(keepDims)},
                                  {"noop_with_empty_axes", es::CompliantNodeBuilder::kEsAttrRequired, "Bool",
                                   es::CreateFrom(noopWithEmptyAxes)},
                              })
                              .Build();

    // Connect data input to ReduceMean
    if (useCast) {
        // Connect Cast output to ReduceMean input 0
        if (es::AddEdgeAndUpdatePeerDesc(*graph, reduceMeanInputNode, reduceMeanInputIdx, reduceMeanNode, 0) !=
            GRAPH_SUCCESS) {
            OP_LOGE_WITHOUT_REPORT(kPassName.c_str(), "Failed to add edge for Cast->ReduceMean");
            return nullptr;
        }
    } else {
        // Connect x directly to ReduceMean input 0
        if (es::AddEdgeAndUpdatePeerDesc(*graph, *rX.GetProducer(), rX.GetProducerOutIndex(), reduceMeanNode, 0) !=
            GRAPH_SUCCESS) {
            OP_LOGE_WITHOUT_REPORT(kPassName.c_str(), "Failed to add edge for x->ReduceMean");
            return nullptr;
        }
    }

    // Connect axes to ReduceMean input 1
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *rAxes.GetProducer(), rAxes.GetProducerOutIndex(), reduceMeanNode, 1) !=
        GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(kPassName.c_str(), "Failed to add edge for axes->ReduceMean");
        return nullptr;
    }

    // Get output tensor
    auto reduceMeanOutput = replaceBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reduceMeanNode, 0);

    // Build and return graph
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(reduceMeanOutput));
    GraphUniqPtr replaceGraph = replaceBuilder.BuildAndReset(outputs);

    // Call InferShape
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(kPassName.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }

    OP_LOGD(kPassName.c_str(), "Replacement completed successfully");
    return replaceGraph;
}

REG_FUSION_PASS(ReduceMeanWithCastFusionPass).Stage(GetReduceMeanWithCastPassStage());

#endif // GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900

} // namespace ops
