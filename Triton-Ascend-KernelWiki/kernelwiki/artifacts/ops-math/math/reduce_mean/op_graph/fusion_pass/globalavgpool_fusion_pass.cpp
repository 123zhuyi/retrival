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
#include "globalavgpool_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {

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

const std::string FUSION_PASS_NAME = "GlobalavgpoolPass";
const int64_t CAPTURE_TENSOR_IDX_INPUT = 0;

// 输入维度常量
const int64_t INPUT_X_DIM_THREE = 3;
const int64_t INPUT_X_NUM_FOUR = 4;
const int64_t INPUT_X_NUM_FIVE = 5;

namespace {
CustomPassStage GetGlobalavgpoolPassStage()
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

static void GetInputsInfo(const std::vector<SubgraphInput>& subGraphInputs, std::vector<Shape>& inputShapes,
                          std::vector<DataType>& inputDtypes, std::vector<Format>& inputFormats)
{
    for (const auto& subGraphInput : subGraphInputs) {
        auto matchNode = subGraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        // matchNode.node是GlobalAveragePool节点，matchNode.index是输出索引（0）
        // 我们需要获取GlobalAveragePool的输入描述（索引0）
        auto status = matchNode.node.GetInputDesc(0, tensorDesc);
        if (status != GRAPH_SUCCESS) {
            OP_LOGE("GlobalavgpoolPass", "Failed to get input desc from GlobalAveragePool node, status: %u", status);
            // 如果失败，尝试获取输出描述
            status = matchNode.node.GetOutputDesc(matchNode.index, tensorDesc);
            if (status != GRAPH_SUCCESS) {
                OP_LOGE("GlobalavgpoolPass", "Failed to get output desc from GlobalAveragePool node, status: %u",
                        status);
                // 使用默认值
                tensorDesc.SetDataType(DT_FLOAT);
                tensorDesc.SetFormat(FORMAT_ND);
                // 使用合理的默认形状
                Shape defaultShape({1, 1, 1}); // 3D默认形状
                tensorDesc.SetShape(defaultShape);
            }
        }
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

static Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subGraphInputs)
{
    std::vector<Shape> inputShapes;
    for (const auto& subGraphInput : subGraphInputs) {
        auto matchNode = subGraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        // matchNode.node是GlobalAveragePool节点，matchNode.index是输出索引（0）
        // 我们需要获取GlobalAveragePool的输入描述（索引0）
        if (matchNode.node.GetInputDesc(0, tensorDesc) != GRAPH_SUCCESS) {
            OP_LOGE_WITHOUT_REPORT("GlobalavgpoolPass",
                                   "Failed to get input desc from GlobalAveragePool node in InferShape");
            // 如果失败，尝试获取输出描述
            matchNode.node.GetOutputDesc(matchNode.index, tensorDesc);
        }
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

std::vector<PatternUniqPtr> GlobalavgpoolPass::Patterns()
{
    OP_LOGD(FUSION_PASS_NAME.c_str(), "Enter Patterns for GlobalavgpoolPass");
    std::vector<PatternUniqPtr> patternGraphs;
    if (!IsTargetVersion()) {
        return patternGraphs;
    }
    auto graphBuilder = es::EsGraphBuilder(FUSION_PASS_NAME.c_str());

    // 创建输入
    auto x = graphBuilder.CreateInput(0);
    OP_LOGD(FUSION_PASS_NAME.c_str(), "Created input node");

    // 构建 GlobalAveragePool 节点
    // 注意：我们需要获取Graph对象来构建CompliantNode
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    OP_LOGD(FUSION_PASS_NAME.c_str(), "Got graph pointer: %p", graph);
    auto globalAvgPool = es::CompliantNodeBuilder(graph)
                             .OpType("GlobalAveragePool")
                             .Name("global_avg_pool")
                             .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();
    OP_LOGD(FUSION_PASS_NAME.c_str(), "Built GlobalAveragePool node");

    // 连接输入 - 使用正确的Graph参数
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), globalAvgPool, 0) !=
        GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(FUSION_PASS_NAME.c_str(), "Failed to add edge in pattern");
        return patternGraphs;
    }
    OP_LOGD(FUSION_PASS_NAME.c_str(), "Added edge successfully");

    // 获取输出并构建图
    // 注意：GetTensorHolderFromNode需要EsCGraphBuilder和节点
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(globalAvgPool, 0);
    OP_LOGD(FUSION_PASS_NAME.c_str(), "Got tensor holder from node");
    // 创建输出向量
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    auto builtGraph = graphBuilder.BuildAndReset(outputs);
    OP_LOGD(FUSION_PASS_NAME.c_str(), "Built graph successfully");
    auto pattern = std::make_unique<Pattern>(std::move(*builtGraph));
    // 捕获输出张量（GlobalAveragePool的输出）
    NodeIo nodeIo = {y->GetProducer(), 0};
    pattern->CaptureTensor(nodeIo);
    OP_LOGD(FUSION_PASS_NAME.c_str(), "Pattern created and tensor captured");

    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

bool GlobalavgpoolPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OP_LOGD(FUSION_PASS_NAME.c_str(), "=== Enter MeetRequirements for GlobalavgpoolPass ===");

    // Runtime version check: on GE 8.5.0, return false to no-op.
    int32_t version = 0;
    char pkgName[] = "ge_compiler";
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(pkgName, &version);
    }
    if (version < GE_COMPILER_VERSION_900) {
        OP_LOGD(FUSION_PASS_NAME.c_str(), "GE runtime version %d < 90000000, skip pass.", version);
        return false;
    }

    // 获取匹配的节点
    NodeIo matchedNode;
    auto status = match_result->GetCapturedTensor(CAPTURE_TENSOR_IDX_INPUT, matchedNode);
    OP_LOGD(FUSION_PASS_NAME.c_str(), "GetCapturedTensor returned: %d", status);
    if (status != SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(FUSION_PASS_NAME.c_str(), "Failed to GetCaptrue tensor");
        return false;
    }

    auto node = matchedNode.node;
    AscendString nodeType;
    node.GetType(nodeType);
    AscendString nodeName;
    node.GetName(nodeName);
    OP_LOGD(FUSION_PASS_NAME.c_str(), "Matched node type: %s, name: %s", nodeType.GetString(), nodeName.GetString());

    // 检查输入维度是否支持
    TensorDesc inputDesc;
    auto descStatus = node.GetInputDesc(0, inputDesc);
    OP_LOGD(FUSION_PASS_NAME.c_str(), "GetInputDesc returned: %d", descStatus);
    if (descStatus != GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(FUSION_PASS_NAME.c_str(), "Failed to get input desc");
        return false;
    }

    auto dims = inputDesc.GetShape().GetDims();
    int64_t inputDim = dims.size();
    OP_LOGD(FUSION_PASS_NAME.c_str(), "Input shape dims: %ld", inputDim);
    for (size_t i = 0; i < dims.size(); ++i) {
        OP_LOGD(FUSION_PASS_NAME.c_str(), "  dim[%zu] = %ld", i, dims[i]);
    }

    if (!(inputDim == INPUT_X_DIM_THREE || inputDim == INPUT_X_NUM_FOUR || inputDim == INPUT_X_NUM_FIVE)) {
        OP_LOGD(FUSION_PASS_NAME.c_str(), "Input dim %ld not supported, only support 3, 4, 5 dimensions.", inputDim);
        return false;
    }

    OP_LOGD(FUSION_PASS_NAME.c_str(), "MeetRequirements returns TRUE");
    return true;
}

GraphUniqPtr GlobalavgpoolPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    std::vector<SubgraphInput> subGraphInputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subGraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subGraphInputs, inputShapes, inputDtypes, inputFormats);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");

    // 创建输入节点 - 使用带有数据类型和形状的重载版本
    auto reduceMeanInput = replaceGraphBuilder.CreateInput(0, "x", inputDtypes[0], inputFormats[0],
                                                           inputShapes[0].GetDims());

    // 根据输入维度计算axes
    int64_t inputDim = inputShapes[0].GetDims().size();
    std::vector<int64_t> axes;
    if (inputDim == INPUT_X_DIM_THREE) {
        axes = {2};
    } else if (inputDim == INPUT_X_NUM_FOUR) {
        axes = {2, 3};
    } else if (inputDim == INPUT_X_NUM_FIVE) {
        axes = {2, 3, 4};
    } else {
        // 正常情况下 MeetRequirements 已经过滤，这里不会执行
        OP_LOGE_WITHOUT_REPORT(FUSION_PASS_NAME.c_str(), "Input dim %ld not supported.", inputDim);
        return nullptr;
    }

    // 创建axes常量节点 - 使用CreateConst方法
    auto axesConst = replaceGraphBuilder.CreateConst(axes, {static_cast<int64_t>(axes.size())});

    // 使用CompliantNodeBuilder创建ReduceMean节点
    auto* graph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();
    auto reduceMeanNode = es::CompliantNodeBuilder(graph)
                              .OpType("ReduceMean")
                              .Name("reduce_mean")
                              .IrDefInputs({
                                  {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                              })
                              .IrDefOutputs({
                                  {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                              })
                              .IrDefAttrs({
                                  {"keep_dims", es::CompliantNodeBuilder::kEsAttrRequired, "Bool",
                                   es::CreateFrom(true)},
                                  {"noop_with_empty_axes", es::CompliantNodeBuilder::kEsAttrRequired, "Bool",
                                   es::CreateFrom(true)},
                              })
                              .Build();
    // 连接输入
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *reduceMeanInput.GetProducer(), reduceMeanInput.GetProducerOutIndex(),
                                     reduceMeanNode, 0) != GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(FUSION_PASS_NAME.c_str(), "Failed to add edge for reduceMean input");
        return nullptr;
    }

    // 连接axes常量节点
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *axesConst.GetProducer(), axesConst.GetProducerOutIndex(), reduceMeanNode,
                                     1) != GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(FUSION_PASS_NAME.c_str(), "Failed to add edge for axes constant");
        return nullptr;
    }

    // 获取输出张量
    auto reduceMeanOutput = replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reduceMeanNode, 0);

    // 构建图
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(reduceMeanOutput));
    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset(outputs);

    // 调用 InferShape
    if (InferShape(replaceGraph, subGraphInputs) != SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(FUSION_PASS_NAME.c_str(), "Infershape for replacement failed.");
        return nullptr;
    }
    return replaceGraph;
}

REG_FUSION_PASS(GlobalavgpoolPass).Stage(GetGlobalavgpoolPassStage());

#endif // GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900

} // namespace ops
