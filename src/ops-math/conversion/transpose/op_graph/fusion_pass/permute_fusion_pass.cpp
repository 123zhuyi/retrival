/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief Permute fusion pass (Permute --> TransposeD / Transpose)
 * @details
 *   Pattern: matches a Permute node.
 *   Replacement:
 *     - 310B/310P/910/910B/910_93 platforms: Permute --> TransposeD (perm as attr)
 *     - All other platforms (950 and newer): Permute --> Transpose (perm as input)
 */

#include <vector>
#include <string>
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"
#include "permute_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace ge::fusion;

namespace ops {

// The PermuteFusionPass uses the new ES framework (es::EsGraphBuilder,
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

const std::string kFusionPassName = "PermuteFusionPass";
const int64_t kCapturePermuteIdx = 0L;

namespace {
CustomPassStage GetPermutePassStage()
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

// Platforms that use TransposeD (perm as attr); all others use Transpose (perm as input)
const std::set<std::string> kTransposeDPlatformList = {"Ascend310B", "Ascend310P", "Ascend910", "Ascend910B",
                                                       "Ascend910_93"};

static bool IsTransposeDPlatform()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (unlikely(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
                 SUCCESS)) {
        OP_LOGE(kFusionPassName.c_str(), "Get platform_info failed.");
        return false;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    return kTransposeDPlatformList.count(soc) > 0;
}

static bool GetPermutePermAttr(const GNode& permuteNode, std::vector<int64_t>& permList)
{
    if (permuteNode.GetAttr("perm", permList) == GRAPH_SUCCESS) {
        return true;
    }
    // Try the "order" attribute as fallback (Permute IR uses "order")
    if (permuteNode.GetAttr("order", permList) == GRAPH_SUCCESS) {
        OP_LOGD(kFusionPassName.c_str(), "Using 'order' attr as perm.");
        return true;
    }
    OP_LOGE(kFusionPassName.c_str(), "Failed to get perm/order attr from Permute node.");
    return false;
}

// Resolve the TensorDesc of a subgraph input's first matched input.
// Fallback: if the input desc's shape is empty, try to get it from the source data node's output desc instead.
static TensorDesc ResolveSubgraphInputTensorDesc(const SubgraphInput& subgraphInput)
{
    auto matchNode = subgraphInput.GetAllInputs().at(0);
    TensorDesc tensorDesc;
    matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
    if (tensorDesc.GetShape().GetDims().empty()) {
        auto srcInfo = matchNode.node.GetInDataNodesAndPortIndexs(0);
        if (srcInfo.first != nullptr) {
            GNode srcNode = *srcInfo.first;
            srcNode.GetOutputDesc(srcInfo.second, tensorDesc);
        }
    }
    return tensorDesc;
}

static void GetInputsInfo(const std::vector<SubgraphInput>& subgraphInputs, std::vector<Shape>& inputShapes,
                          std::vector<DataType>& inputDtypes, std::vector<Format>& inputFormats)
{
    for (const auto& subgraphInput : subgraphInputs) {
        TensorDesc tensorDesc = ResolveSubgraphInputTensorDesc(subgraphInput);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

static Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    OP_LOGD(kFusionPassName.c_str(), "Begin infershape for replacement.");
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        TensorDesc tensorDesc = ResolveSubgraphInputTensorDesc(subgraphInput);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

// Helper to build TransposeD using CompliantNodeBuilder (for 310B/310P/910/910B/910_93 platforms)
// Returns the output EsTensorHolder, or empty if failed.
static es::EsTensorHolder BuildTransposeDNode(es::EsGraphBuilder& replaceGraphBuilder, GNode inputNode,
                                              int32_t inputIndex, const std::vector<int64_t>& permAttr)
{
    auto* graph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();

    auto transposeDNode = es::CompliantNodeBuilder(graph)
                              .OpType("TransposeD")
                              .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                              .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .IrDefAttrs({
                                  {"perm", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                   es::CreateFrom(permAttr)},
                              })
                              .Build();
    // Connect input
    if (es::AddEdgeAndUpdatePeerDesc(*graph, inputNode, inputIndex, transposeDNode, 0) != GRAPH_SUCCESS) {
        OP_LOGE(kFusionPassName.c_str(), "Failed to add edge for TransposeD input.");
        return es::EsTensorHolder();
    }

    auto output = replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transposeDNode, 0);
    return es::EsTensorHolder(output);
}

// Helper to build Transpose using CompliantNodeBuilder (for Ascend950 and newer platforms)
// Returns the output EsTensorHolder, or empty if failed.
static es::EsTensorHolder BuildTransposeNode(es::EsGraphBuilder& replaceGraphBuilder, GNode inputNode,
                                             int32_t inputIndex, const std::vector<int64_t>& permAttr)
{
    auto* graph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();

    // Create perm Const node
    auto permConst = replaceGraphBuilder.CreateConst(permAttr,
                                                     std::vector<int64_t>{static_cast<int64_t>(permAttr.size())});

    auto transposeNode = es::CompliantNodeBuilder(graph)
                             .OpType("Transpose")
                             .IrDefInputs({
                                 {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                 {"perm", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             })
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();
    // Connect x input
    if (es::AddEdgeAndUpdatePeerDesc(*graph, inputNode, inputIndex, transposeNode, 0) != GRAPH_SUCCESS) {
        OP_LOGE(kFusionPassName.c_str(), "Failed to add edge for Transpose x input.");
        return es::EsTensorHolder();
    }

    // Connect perm input
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *permConst.GetProducer(), permConst.GetProducerOutIndex(), transposeNode,
                                     1) != GRAPH_SUCCESS) {
        OP_LOGE(kFusionPassName.c_str(), "Failed to add edge for Transpose perm input.");
        return es::EsTensorHolder();
    }

    auto output = replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transposeNode, 0);
    return es::EsTensorHolder(output);
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

std::vector<PatternUniqPtr> PermuteFusionPass::Patterns()
{
    OP_LOGD(kFusionPassName.c_str(), "Enter Patterns for PermuteFusionPass.");
    std::vector<PatternUniqPtr> patternGraphs;

    if (!IsTargetVersion()) {
        return patternGraphs;
    }

    auto graphBuilder = es::EsGraphBuilder("PermuteFusionPass");

    // Create input node
    auto x = graphBuilder.CreateInput(0);

    // Build Permute node using CompliantNodeBuilder (no ES API for Permute)
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto permuteNode = es::CompliantNodeBuilder(graph)
                           .OpType("Permute")
                           .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .Build();
    // Connect input x to Permute node
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), permuteNode, 0) !=
        GRAPH_SUCCESS) {
        OP_LOGE(kFusionPassName.c_str(), "Failed to add edge in pattern for Permute.");
        return patternGraphs;
    }

    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(permuteNode, 0);
    auto builtGraph = graphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{es::EsTensorHolder(y)});

    auto pattern = std::make_unique<Pattern>(std::move(*builtGraph));
    NodeIo nodeIo = {y->GetProducer(), 0};
    pattern->CaptureTensor(nodeIo); // Capture the Permute node
    patternGraphs.emplace_back(std::move(pattern));

    return patternGraphs;
}

bool PermuteFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kFusionPassName.c_str(), "Enter MeetRequirements for PermuteFusionPass.");

    // Runtime version check: on GE 8.5.0, return false to no-op.
    int32_t version = 0;
    char pkgName[] = "ge_compiler";
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(pkgName, &version);
    }
    if (version < GE_COMPILER_VERSION_900) {
        OP_LOGD(kFusionPassName.c_str(), "GE runtime version %d < 90000000, skip pass.", version);
        return false;
    }

    NodeIo permuteNodeIo;
    if (unlikely(matchResult->GetCapturedTensor(kCapturePermuteIdx, permuteNodeIo) != SUCCESS)) {
        OP_LOGE(kFusionPassName.c_str(), "Failed to get captured tensor.");
        return false;
    }

    AscendString nodeType;
    permuteNodeIo.node.GetType(nodeType);
    if (nodeType != "Permute") {
        OP_LOGE(kFusionPassName.c_str(), "Node type %s is not Permute, skip.", nodeType.GetString());
        return false;
    }

    return true;
}

GraphUniqPtr PermuteFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kFusionPassName.c_str(), "Enter Replacement for PermuteFusionPass.");

    // Get captured Permute node and extract perm attribute
    NodeIo permuteNodeIo;
    if (matchResult->GetCapturedTensor(kCapturePermuteIdx, permuteNodeIo) != SUCCESS) {
        OP_LOGE(kFusionPassName.c_str(), "Failed to get captured tensor in Replacement.");
        return nullptr;
    }

    std::vector<int64_t> permList;
    if (!GetPermutePermAttr(permuteNodeIo.node, permList)) {
        OP_LOGE(kFusionPassName.c_str(), "Cannot extract perm attribute, skip.");
        return nullptr;
    }

    // Get platform info
    bool isTransposeDPlatform = IsTransposeDPlatform();

    // Get subgraph input info
    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);
    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    auto inDims = inputShapes[0].GetDims();

    // Note: In the old framework, a special case (4D input with perm=[0,3,2,1])
    // was split into two consecutive TransposeD/Transpose nodes as an optimization.
    // The new PatternFusionPass framework limits replacement graphs to structurally
    // consistent subgraphs, so multi-node replacement is not feasible here.
    // Instead, we perform a simple single-node replacement, preserving the original
    // perm attribute, which is functionally equivalent.

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto xTensor = replaceGraphBuilder.CreateInput(0, "x", inputDtypes[0], inputFormats[0], inDims);

    if (isTransposeDPlatform) {
        // ========== 310B/310P/910/910B/910_93: use CompliantNodeBuilder to build TransposeD ==========
        GNode xNode = *xTensor.GetProducer();
        int32_t xIndex = xTensor.GetProducerOutIndex();

        auto output = BuildTransposeDNode(replaceGraphBuilder, xNode, xIndex, permList);
        if (output.GetCTensorHolder() == nullptr) {
            OP_LOGE(kFusionPassName.c_str(), "Failed to build TransposeD node.");
            return nullptr;
        }

        auto replaceGraph = replaceGraphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});
        if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
            OP_LOGE(kFusionPassName.c_str(), "Infershape for replacement (TransposeD) failed.");
            return nullptr;
        }
        return replaceGraph;
    } else {
        // ========== All other platforms (950 and newer): use CompliantNodeBuilder to build Transpose ==========
        GNode xNode = *xTensor.GetProducer();
        int32_t xIndex = xTensor.GetProducerOutIndex();

        auto output = BuildTransposeNode(replaceGraphBuilder, xNode, xIndex, permList);
        if (output.GetCTensorHolder() == nullptr) {
            OP_LOGE(kFusionPassName.c_str(), "Failed to build Transpose node.");
            return nullptr;
        }

        auto replaceGraph = replaceGraphBuilder.BuildAndReset(std::vector<es::EsTensorHolder>{output});
        if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
            OP_LOGE(kFusionPassName.c_str(), "Infershape for replacement (Transpose) failed.");
            return nullptr;
        }
        return replaceGraph;
    }
}

REG_FUSION_PASS(PermuteFusionPass).Stage(GetPermutePassStage());

#endif // GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900

} // namespace ops
