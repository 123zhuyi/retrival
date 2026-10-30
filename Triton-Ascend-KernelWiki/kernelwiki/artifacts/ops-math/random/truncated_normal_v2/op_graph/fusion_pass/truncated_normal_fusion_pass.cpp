/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file truncated_normal_fusion_pass.cpp
 * \brief TruncatedNormal fusion pass (TruncatedNormal --> TruncatedNormalV2)
 *
 *                                                     offset(variable)
 *       shape                       shape            /   |
 *         |                            \            /    |
 *   TruncatedNormal          ==>    TruncatedNormalV2    |
 *         |                            /           \     |
 *         y                           y             \    |
 *                                                     offset
 */

#include <vector>
#include <string>
#include <set>
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "ge/compliant_node_builder.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"
#include "truncated_normal_fusion_pass.h"
#include "common/inc/op_graph/fusion_pass/fusion_pass_common.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {

// D1 scenario: uses kCompatibleInherited stage (9.0.0+).
// Strategy: compile-time macro guard + runtime version check + overall silence.
#define GE_COMPILER_VERSION_900 90000000

namespace {
#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
CustomPassStage GetTruncatedNormalFusionPassStage()
{
    int32_t version = 0;
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    if (version >= GE_COMPILER_VERSION_900) {
        return CustomPassStage::kCompatibleInherited;
    }
    return CustomPassStage::kBeforeInferShape;
}
#endif
} // namespace

static const std::string kPassName = "TruncatedNormalFusionPass";
static const int64_t kCaptureIdxNode = 0L;
static const std::set<DataType> kAicoreDtypeSupportList = {DT_FLOAT, DT_FLOAT16, DT_BF16};

std::vector<PatternUniqPtr> TruncatedNormalFusionPass::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns");
    std::vector<PatternUniqPtr> patternGraphs;

    if (!IsTargetVersion()) {
        OP_LOGD(kPassName.c_str(), "GE runtime version < %d, skip pass.", GE_COMPILER_VERSION_910);
        return patternGraphs;
    }

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    auto shape = graphBuilder.CreateInput(0);

    auto graphPtr = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto srcBuilder = es::CompliantNodeBuilder(graphPtr);
    srcBuilder.OpType("TruncatedNormal")
        .Name("pattern_truncated_normal")
        .IrDefInputs({{"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs(
            {{"seed", es::CompliantNodeBuilder::kEsAttrOptional, "Int", es::CreateFrom(static_cast<int64_t>(0))},
             {"seed2", es::CompliantNodeBuilder::kEsAttrOptional, "Int", es::CreateFrom(static_cast<int64_t>(0))},
             {"dtype", es::CompliantNodeBuilder::kEsAttrOptional, "Type", es::CreateFrom(DT_FLOAT)}});
    GNode srcNode = srcBuilder.Build();

    auto dataNode = shape.GetProducer();
    if (dataNode != nullptr) {
        es::AddEdgeAndUpdatePeerDesc(*graphPtr, *dataNode, 0, srcNode, 0);
    }

    es::EsGraphBuilder::SetOutput(shape, 0);
    auto graph = graphBuilder.BuildAndReset();

    std::vector<std::pair<GNode, int32_t>> outputs = {{srcNode, 0}};
    graph->SetOutputs(outputs);

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({srcNode, 0});

    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

bool TruncatedNormalFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements");

    int32_t version = 0;
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    if (version < GE_COMPILER_VERSION_900) {
        OP_LOGD(kPassName.c_str(), "GE runtime version %d < 90000000, skip pass.", version);
        return false;
    }

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

    NodeIo nodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxNode, nodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to GetCaptured tensor.");
        return false;
    }
    AscendString nodeType;
    nodeIo.node.GetType(nodeType);
    if (std::string(nodeType.GetString()) != "TruncatedNormal") {
        OP_LOGD(kPassName.c_str(), "Node type is not TruncatedNormal, skip.");
        return false;
    }

    DataType dtype = DT_FLOAT;
    nodeIo.node.GetAttr("dtype", dtype);
    if (kAicoreDtypeSupportList.count(dtype) == 0) {
        OP_LOGD(kPassName.c_str(), "Dtype %d not supported.", static_cast<int>(dtype));
        return false;
    }

    return true;
}

std::unique_ptr<Graph> TruncatedNormalFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter Replacement");

    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    NodeIo nodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxNode, nodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to GetCaptured tensor in Replacement.");
        return nullptr;
    }

    DataType dtype = DT_FLOAT;
    nodeIo.node.GetAttr("dtype", dtype);
    int64_t dtypeInt = static_cast<int64_t>(dtype);

    int64_t seed = 0;
    nodeIo.node.GetAttr("seed", seed);

    int64_t seed2 = 0;
    nodeIo.node.GetAttr("seed2", seed2);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");

    auto rShape = replaceGraphBuilder.CreateInput(0, "shape", inputDtypes[0], inputFormats[0],
                                                  inputShapes[0].GetDims());

    AscendString nodeName;
    nodeIo.node.GetName(nodeName);
    std::string varName = std::string(nodeName.GetString()) + "/offsetVariable";

    TensorDesc offsetDesc(Shape({1}), FORMAT_ND, DT_INT64);

    auto rOffset = replaceGraphBuilder.CreateVariable(1, varName.c_str());
    auto varNodePtr = rOffset.GetProducer();
    if (varNodePtr != nullptr) {
        varNodePtr->UpdateOutputDesc(0, offsetDesc);
        int64_t initValue = 0;
        Tensor initTensor(offsetDesc, reinterpret_cast<uint8_t*>(&initValue), sizeof(int64_t));
        varNodePtr->SetAttr("init_value", initTensor);
    }

    auto v2Output = es::TruncatedNormalV2(rShape, rOffset, seed, seed2, dtypeInt);
    GNode v2NodePtr = *v2Output.y.GetProducer();

    TensorDesc shapeInputDesc(inputShapes[0], inputFormats[0], inputDtypes[0]);
    v2NodePtr.UpdateInputDesc(0, shapeInputDesc);
    v2NodePtr.UpdateInputDesc(1, offsetDesc);

    TensorDesc outputYDesc;
    nodeIo.node.GetOutputDesc(0, outputYDesc);
    v2NodePtr.UpdateOutputDesc(0, outputYDesc);
    v2NodePtr.UpdateOutputDesc(1, offsetDesc);

    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({v2Output.y});
    return replaceGraph;
}

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

#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
REG_FUSION_PASS(TruncatedNormalFusionPass).Stage(GetTruncatedNormalFusionPassStage());
#else
REG_FUSION_PASS(TruncatedNormalFusionPass).Stage(CustomPassStage::kBeforeInferShape);
#endif

} // namespace ops
