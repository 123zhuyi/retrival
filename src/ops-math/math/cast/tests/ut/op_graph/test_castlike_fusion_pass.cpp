/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "../../../op_graph/fusion_pass/castlike_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

namespace {
const std::string kPassName = "CastlikeFusionPass";
}

class CastlikeFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetPlatform(const std::string& soc)
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = soc;
        optiCompilationInfo.soc_version = soc;
        PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    es::EsTensorHolder BuildCastLikeNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x,
                                         const es::EsTensorHolder& y)
    {
        ge::Graph* graph = builder.GetCGraphBuilder()->GetGraph();
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

        // Connect input to two input one output castlike using AddEdgeAndUpdatePeerDesc
        es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), castlike, 0);
        es::AddEdgeAndUpdatePeerDesc(*graph, *y.GetProducer(), y.GetProducerOutIndex(), castlike, 1);

        auto output = es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(castlike, 0));
        return output;
    }

    static void InferShapeForTest(
        DataType dtypeX, DataType dtypeY, Shape& shape,
        es::EsTensorHolder& x, es::EsTensorHolder& y, es::EsTensorHolder& castlike)
    {
        // set input
        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(dtypeX);
        xDesc.SetShape(shape);
        x.GetProducer()->UpdateOutputDesc(0, xDesc);

        TensorDesc yDesc;
        y.GetProducer()->GetOutputDesc(0, yDesc);
        yDesc.SetDataType(dtypeY);
        yDesc.SetShape(shape);
        y.GetProducer()->UpdateOutputDesc(0, yDesc);

        castlike.GetProducer()->UpdateInputDesc(0, xDesc);
        castlike.GetProducer()->UpdateInputDesc(1, yDesc);
        castlike.GetProducer()->UpdateOutputDesc(0, yDesc);
    }
};

// Test 1: Pattern creation test
TEST_F(CastlikeFusionPassTest, patternTest)
{
    CastlikeFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

// Test 2: Fusion success - FP16 to FP32
TEST_F(CastlikeFusionPassTest, fusionSuccessFp16ToFp32)
{
    std::vector<int64_t> dimsX{2, 32, 128};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto y = graphBuilder.CreateInput(1, "y", DT_FLOAT, FORMAT_ND, dimsX);

    auto output = BuildCastLikeNode(graphBuilder, x, y);
    InferShapeForTest(DT_FLOAT16, DT_FLOAT, shapeX, x, y, output);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({{output}});

    // Use dynamic allocation to avoid CustomPassContextImpl incomplete type destructor issue
    CustomPassContext* passContextPtr = new CustomPassContext();
    CastlikeFusionPass pass;
    Status status = pass.Run(graph, *passContextPtr);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool findCast = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "Cast") {
            findCast = true;
        }
    }
    EXPECT_EQ(findCast, true);
    EXPECT_EQ(node_count, 4);
}

// Test 3: Fusion success - FP32 to FP16
TEST_F(CastlikeFusionPassTest, fusionSuccessFp32ToFp16)
{
    std::vector<int64_t> dimsX{2, 32, 128};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto y = graphBuilder.CreateInput(1, "y", DT_FLOAT16, FORMAT_ND, dimsX);

    auto output = BuildCastLikeNode(graphBuilder, x, y);
    InferShapeForTest(DT_FLOAT, DT_FLOAT16, shapeX, x, y, output);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    // Use dynamic allocation to avoid CustomPassContextImpl incomplete type destructor issue
    CustomPassContext* passContextPtr = new CustomPassContext();
    CastlikeFusionPass pass;
    Status status = pass.Run(graph, *passContextPtr);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool findCast = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "Cast") {
            findCast = true;
        }
    }
    EXPECT_EQ(findCast, true);
    EXPECT_EQ(node_count, 4);
}

// Test 4: Fusion success - BF16 to FP32
TEST_F(CastlikeFusionPassTest, fusionSuccessBF16ToFp32)
{
    std::vector<int64_t> dimsX{2, 32, 128};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_BF16, FORMAT_ND, dimsX);
    auto y = graphBuilder.CreateInput(1, "y", DT_FLOAT, FORMAT_ND, dimsX);

    auto output = BuildCastLikeNode(graphBuilder, x, y);
    InferShapeForTest(DT_BF16, DT_FLOAT, shapeX, x, y, output);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    // Use dynamic allocation to avoid CustomPassContextImpl incomplete type destructor issue
    CustomPassContext* passContextPtr = new CustomPassContext();
    CastlikeFusionPass pass;
    Status status = pass.Run(graph, *passContextPtr);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool findCast = false;
    int node_count = 0;
    for (auto node : graph->GetAllNodes()) {
        node_count++;
        AscendString type;
        node.GetType(type);
        if (type == "Cast") {
            findCast = true;
        }
    }
    EXPECT_EQ(findCast, true);
    EXPECT_EQ(node_count, 4);
}
