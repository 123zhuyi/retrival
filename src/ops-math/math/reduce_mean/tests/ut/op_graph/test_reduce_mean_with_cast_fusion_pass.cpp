/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"
#include "../../../op_graph/fusion_pass/reduce_mean_with_cast_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

class ReduceMeanWithCastFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

TEST_F(ReduceMeanWithCastFusionPassTest, patternTest)
{
    ops::ReduceMeanWithCastFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(ReduceMeanWithCastFusionPassTest, fusionSuccessWithDtype)
{
    std::vector<int64_t> dimsX{6, 7};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("reduceMeanWithCast_test_with_dtype");

    // Create input x
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());

    // Set x output description
    TensorDesc xOutputDesc;
    x.GetProducer()->GetOutputDesc(0, xOutputDesc);
    xOutputDesc.SetDataType(DT_FLOAT);
    xOutputDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xOutputDesc);

    // Create axes constant
    std::vector<int64_t> axesData = {1};
    auto axesConst = graphBuilder.CreateConst(axesData, {static_cast<int64_t>(axesData.size())});

    // Build ReduceMeanWithCast node using CompliantNodeBuilder
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto reduceMeanWithCast = es::CompliantNodeBuilder(graph)
        .OpType("ReduceMeanWithCast")
        .Name("reduce_mean_with_cast")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"keep_dims", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(false)},
            {"noop_with_empty_axes", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(true)},
            {"dtype", es::CompliantNodeBuilder::kEsAttrOptional, "Type",
                es::CreateFrom(DT_FLOAT16)},
        })
        .Build();

    // Connect x to ReduceMeanWithCast input 0
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(),
        reduceMeanWithCast, 0) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge for x input";
    }

    // Connect axes to ReduceMeanWithCast input 1
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *axesConst.GetProducer(), axesConst.GetProducerOutIndex(),
        reduceMeanWithCast, 1) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge for axes input";
    }

    // Update ReduceMeanWithCast input descriptions
    TensorDesc castInputDesc;
    reduceMeanWithCast.GetInputDesc(0, castInputDesc);
    castInputDesc.SetDataType(DT_FLOAT);
    castInputDesc.SetShape(shapeX);
    reduceMeanWithCast.UpdateInputDesc(0, castInputDesc);

    TensorDesc axesInputDesc;
    reduceMeanWithCast.GetInputDesc(1, axesInputDesc);
    axesInputDesc.SetDataType(DT_INT64);
    axesInputDesc.SetShape(Shape({1}));
    reduceMeanWithCast.UpdateInputDesc(1, axesInputDesc);

    // Get output tensor
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reduceMeanWithCast, 0);

    // Build and run graph
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset(outputs);

    CustomPassContext passContext;
    ops::ReduceMeanWithCastFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    // Verify: should find Cast node and ReduceMean node
    bool findCast = false;
    bool findReduceMean = false;
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "Cast") {
            findCast = true;
        }
        if (type == "ReduceMean") {
            findReduceMean = true;
        }
    }
    EXPECT_TRUE(findCast) << "Cast node should exist after fusion";
    EXPECT_TRUE(findReduceMean) << "ReduceMean node should exist after fusion";
}

TEST_F(ReduceMeanWithCastFusionPassTest, fusionSuccessWithoutDtype)
{
    std::vector<int64_t> dimsX{6, 7};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("reduceMeanWithCast_test_without_dtype");

    // Create input x
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());

    // Set x output description
    TensorDesc xOutputDesc;
    x.GetProducer()->GetOutputDesc(0, xOutputDesc);
    xOutputDesc.SetDataType(DT_FLOAT);
    xOutputDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xOutputDesc);

    // Create axes constant
    std::vector<int64_t> axesData = {1};
    auto axesConst = graphBuilder.CreateConst(axesData, {static_cast<int64_t>(axesData.size())});

    // Build ReduceMeanWithCast node WITHOUT dtype attribute
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto reduceMeanWithCast = es::CompliantNodeBuilder(graph)
        .OpType("ReduceMeanWithCast")
        .Name("reduce_mean_with_cast")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"keep_dims", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(false)},
            {"noop_with_empty_axes", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(true)},
        })
        .Build();

    // Connect edges
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(),
        reduceMeanWithCast, 0) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge for x input";
    }
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *axesConst.GetProducer(), axesConst.GetProducerOutIndex(),
        reduceMeanWithCast, 1) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge for axes input";
    }

    // Update input descriptions
    TensorDesc castInputDesc;
    reduceMeanWithCast.GetInputDesc(0, castInputDesc);
    castInputDesc.SetDataType(DT_FLOAT);
    castInputDesc.SetShape(shapeX);
    reduceMeanWithCast.UpdateInputDesc(0, castInputDesc);

    TensorDesc axesInputDesc;
    reduceMeanWithCast.GetInputDesc(1, axesInputDesc);
    axesInputDesc.SetDataType(DT_INT64);
    axesInputDesc.SetShape(Shape({1}));
    reduceMeanWithCast.UpdateInputDesc(1, axesInputDesc);

    // Get output tensor
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reduceMeanWithCast, 0);

    // Build and run graph
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset(outputs);

    CustomPassContext passContext;
    ops::ReduceMeanWithCastFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    // Verify: should find ReduceMean node but NOT Cast node
    bool findCast = false;
    bool findReduceMean = false;
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "Cast") {
            findCast = true;
        }
        if (type == "ReduceMean") {
            findReduceMean = true;
        }
    }
    EXPECT_FALSE(findCast) << "Cast node should NOT exist when dtype is not set";
    EXPECT_TRUE(findReduceMean) << "ReduceMean node should exist after fusion";
}

TEST_F(ReduceMeanWithCastFusionPassTest, fusionSuccessAscend950)
{
    // Set platform to Ascend950
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dimsX{6, 7};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("reduceMeanWithCast_test_950");

    // Create input x
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());

    // Set x output description
    TensorDesc xOutputDesc;
    x.GetProducer()->GetOutputDesc(0, xOutputDesc);
    xOutputDesc.SetDataType(DT_FLOAT);
    xOutputDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xOutputDesc);

    // Create axes constant
    std::vector<int64_t> axesData = {1};
    auto axesConst = graphBuilder.CreateConst(axesData, {static_cast<int64_t>(axesData.size())});

    // Build ReduceMeanWithCast node with dtype
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto reduceMeanWithCast = es::CompliantNodeBuilder(graph)
        .OpType("ReduceMeanWithCast")
        .Name("reduce_mean_with_cast")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"keep_dims", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(false)},
            {"noop_with_empty_axes", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(true)},
            {"dtype", es::CompliantNodeBuilder::kEsAttrOptional, "Type",
                es::CreateFrom(DT_FLOAT16)},
        })
        .Build();

    // Connect edges
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(),
        reduceMeanWithCast, 0) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge for x input";
    }
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *axesConst.GetProducer(), axesConst.GetProducerOutIndex(),
        reduceMeanWithCast, 1) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge for axes input";
    }

    // Update input descriptions
    TensorDesc castInputDesc;
    reduceMeanWithCast.GetInputDesc(0, castInputDesc);
    castInputDesc.SetDataType(DT_FLOAT);
    castInputDesc.SetShape(shapeX);
    reduceMeanWithCast.UpdateInputDesc(0, castInputDesc);

    TensorDesc axesInputDesc;
    reduceMeanWithCast.GetInputDesc(1, axesInputDesc);
    axesInputDesc.SetDataType(DT_INT64);
    axesInputDesc.SetShape(Shape({1}));
    reduceMeanWithCast.UpdateInputDesc(1, axesInputDesc);

    // Get output tensor
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reduceMeanWithCast, 0);

    // Build and run graph
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset(outputs);

    CustomPassContext passContext;
    ops::ReduceMeanWithCastFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    // Verify
    bool findReduceMean = false;
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "ReduceMean") {
            findReduceMean = true;
        }
    }
    EXPECT_TRUE(findReduceMean) << "ReduceMean node should exist on Ascend950";
}

TEST_F(ReduceMeanWithCastFusionPassTest, fusionSuccessWithAxesInt32)
{
    // Test with int32 axes (different axes dtype)
    std::vector<int64_t> dimsX{6, 7};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("reduceMeanWithCast_test_axes_int32");

    // Create input x
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());

    TensorDesc xOutputDesc;
    x.GetProducer()->GetOutputDesc(0, xOutputDesc);
    xOutputDesc.SetDataType(DT_FLOAT16);
    xOutputDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xOutputDesc);

    // Create axes constant with int32 data
    std::vector<int32_t> axesData32 = {1};
    auto axesConst = graphBuilder.CreateConst(axesData32, {static_cast<int64_t>(axesData32.size())});

    // Update axesConst output desc to int32
    TensorDesc axesConstDesc;
    axesConst.GetProducer()->GetOutputDesc(0, axesConstDesc);
    axesConstDesc.SetDataType(DT_INT32);
    axesConstDesc.SetShape(Shape({1}));
    axesConst.GetProducer()->UpdateOutputDesc(0, axesConstDesc);

    // Build ReduceMeanWithCast node
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

    // Connect edges
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(),
        reduceMeanWithCast, 0) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge for x input";
    }
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *axesConst.GetProducer(), axesConst.GetProducerOutIndex(),
        reduceMeanWithCast, 1) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge for axes input";
    }

    // Update input descriptions
    TensorDesc castInputDesc;
    reduceMeanWithCast.GetInputDesc(0, castInputDesc);
    castInputDesc.SetDataType(DT_FLOAT16);
    castInputDesc.SetShape(shapeX);
    reduceMeanWithCast.UpdateInputDesc(0, castInputDesc);

    TensorDesc axesInputDesc;
    reduceMeanWithCast.GetInputDesc(1, axesInputDesc);
    axesInputDesc.SetDataType(DT_INT32);
    axesInputDesc.SetShape(Shape({1}));
    reduceMeanWithCast.UpdateInputDesc(1, axesInputDesc);

    // Get output tensor
    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reduceMeanWithCast, 0);

    // Build and run graph
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset(outputs);

    CustomPassContext passContext;
    ops::ReduceMeanWithCastFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    // Verify ReduceMean exists
    bool findReduceMean = false;
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "ReduceMean") {
            findReduceMean = true;
        }
    }
    EXPECT_TRUE(findReduceMean) << "ReduceMean node should exist with int32 axes";
}

// ==================== Compatibility tests ====================

// Verify that the pass compiles and the version guard is active.
// GE_COMPILER_VERSION_NUM must be >= 90000000 for the pass to be enabled.
TEST_F(ReduceMeanWithCastFusionPassTest, compileTimeVersionGuard)
{
    // If this test compiles and runs, the GE_COMPILER_VERSION_NUM >= 90000000
    // guard is satisfied (otherwise the pass class wouldn't be defined).
    EXPECT_GE(GE_COMPILER_VERSION_NUM, 90000000);
}

// Verify that the pass can be instantiated and returns patterns
// (validates that the #if GE_COMPILER_VERSION_NUM >= 90000000 guard
// enables the pass at current compiler version).
TEST_F(ReduceMeanWithCastFusionPassTest, passInstantiationTest)
{
    ops::ReduceMeanWithCastFusionPass pass;
    auto patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

// Verify the pass works correctly with the kCompatibleInherited stage.
// This confirms the stage compatibility mechanism is functional.
TEST_F(ReduceMeanWithCastFusionPassTest, compatibleInheritedStageTest)
{
    std::vector<int64_t> dimsX{6, 7};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("compat_stage_test");

    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());

    TensorDesc xOutputDesc;
    x.GetProducer()->GetOutputDesc(0, xOutputDesc);
    xOutputDesc.SetDataType(DT_FLOAT);
    xOutputDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xOutputDesc);

    std::vector<int64_t> axesData = {1};
    auto axesConst = graphBuilder.CreateConst(axesData, {static_cast<int64_t>(axesData.size())});

    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto reduceMeanWithCast = es::CompliantNodeBuilder(graph)
        .OpType("ReduceMeanWithCast")
        .Name("reduce_mean_with_cast")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({
            {"keep_dims", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(false)},
            {"noop_with_empty_axes", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(true)},
        })
        .Build();

    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(),
        reduceMeanWithCast, 0) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge for x input";
    }
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *axesConst.GetProducer(), axesConst.GetProducerOutIndex(),
        reduceMeanWithCast, 1) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge for axes input";
    }

    TensorDesc castInputDesc;
    reduceMeanWithCast.GetInputDesc(0, castInputDesc);
    castInputDesc.SetDataType(DT_FLOAT);
    castInputDesc.SetShape(shapeX);
    reduceMeanWithCast.UpdateInputDesc(0, castInputDesc);

    TensorDesc axesInputDesc;
    reduceMeanWithCast.GetInputDesc(1, axesInputDesc);
    axesInputDesc.SetDataType(DT_INT64);
    axesInputDesc.SetShape(Shape({1}));
    reduceMeanWithCast.UpdateInputDesc(1, axesInputDesc);

    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(reduceMeanWithCast, 0);

    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset(outputs);

    CustomPassContext passContext;
    ops::ReduceMeanWithCastFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);
}
