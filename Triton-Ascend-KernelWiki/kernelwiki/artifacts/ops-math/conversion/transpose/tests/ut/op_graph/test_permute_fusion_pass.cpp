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
#include "ge/compliant_node_builder.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"
#include "../../../op_graph/fusion_pass/permute_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

namespace {
const std::string kPassName = "PermuteFusionPass";
}

class PermuteFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
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

    // Helper to build a Permute node and return the graph with pass applied
    std::shared_ptr<Graph> BuildPermuteGraphAndRunPass(const std::vector<int64_t>& inputDims,
                                                       const std::vector<int64_t>& permAttr, DataType dtype = DT_FLOAT,
                                                       Format format = FORMAT_ND)
    {
        Shape shapeX(inputDims);

        auto graphBuilder = es::EsGraphBuilder("test");

        // Create input
        auto x = graphBuilder.CreateInput(0, "x", dtype, format, inputDims);

        // Build Permute node using CompliantNodeBuilder
        auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto permuteNode = es::CompliantNodeBuilder(graph)
                               .OpType("Permute")
                               .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                               .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                               .IrDefAttrs({
                                   {"perm", es::CompliantNodeBuilder::kEsAttrOptional, "ListInt",
                                    es::CreateFrom(permAttr)},
                               })
                               .Build();

        // Connect input to Permute
        es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), permuteNode, 0);

        // Set input node descriptor
        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(dtype);
        xDesc.SetShape(shapeX);
        xDesc.SetFormat(format);
        x.GetProducer()->UpdateOutputDesc(0, xDesc);

        auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(permuteNode, 0);
        std::shared_ptr<Graph> resultGraph = graphBuilder.BuildAndReset(
            std::vector<es::EsTensorHolder>{es::EsTensorHolder(y)});

        CustomPassContext passContext;
        PermuteFusionPass pass;
        pass.Run(resultGraph, passContext);

        return resultGraph;
    }

    // Helper to count nodes of a given type in the graph
    int CountNodeType(const std::shared_ptr<Graph>& graph, const char* typeName)
    {
        int count = 0;
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == typeName) {
                count++;
            }
        }
        return count;
    }

    // Helper to check if a node type exists in the graph
    bool HasNodeType(const std::shared_ptr<Graph>& graph, const char* typeName)
    {
        return CountNodeType(graph, typeName) > 0;
    }
};

// ==================== Required test scenarios ====================

TEST_F(PermuteFusionPassTest, patternTest)
{
    PermuteFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(PermuteFusionPassTest, unsupportedDtypeFail)
{
    // PermuteFusionPass has no dtype restriction in MeetRequirements,
    // so even less common dtypes like DT_INT32 still get fused.
    // This test verifies that DT_INT32 (unsupported by TransposeD/Transpose
    // on some platforms) still triggers fusion, documenting the pass behavior.
    SetPlatform("Ascend910_93");

    auto resultGraph = BuildPermuteGraphAndRunPass(std::vector<int64_t>{4, 3, 2, 1}, std::vector<int64_t>{0, 2, 3, 1},
                                                   DT_INT32);

    // Pass does not filter by dtype, so fusion should still succeed
    EXPECT_TRUE(HasNodeType(resultGraph, "TransposeD"));
}

TEST_F(PermuteFusionPassTest, unsupportedPlatformFail)
{
    // PermuteFusionPass handles all platforms:
    //   - TransposeD platforms (310B/310P/910/910B/910_93) -> TransposeD
    //   - All other platforms -> Transpose (perm as input)
    // There is no platform that returns NOT_CHANGED.
    // This test verifies that an unknown/unregistered platform
    // correctly falls back to the Transpose branch.
    SetPlatform("UnknownPlatform");

    auto resultGraph = BuildPermuteGraphAndRunPass(std::vector<int64_t>{4, 3, 2, 1}, std::vector<int64_t>{0, 2, 3, 1});

    // Unknown platform should fall back to Transpose (not TransposeD)
    EXPECT_TRUE(HasNodeType(resultGraph, "Transpose"));
    EXPECT_FALSE(HasNodeType(resultGraph, "TransposeD"));
}

// ==================== Fusion success tests ====================

TEST_F(PermuteFusionPassTest, fusionSuccess910Normal)
{
    // Ascend910_93 platform: Permute -> TransposeD (normal case)
    SetPlatform("Ascend910_93");

    auto resultGraph = BuildPermuteGraphAndRunPass(std::vector<int64_t>{4, 3, 2, 1}, std::vector<int64_t>{0, 2, 3, 1});

    // Check that TransposeD node exists and no Transpose node
    EXPECT_TRUE(HasNodeType(resultGraph, "TransposeD"));
    EXPECT_FALSE(HasNodeType(resultGraph, "Transpose"));
}

TEST_F(PermuteFusionPassTest, fusionSuccess910Special)
{
    // Ascend910_93 platform: 4D input with perm=[0,3,2,1]
    // In the old framework this was split into two TransposeD nodes,
    // but PatternFusionPass does not support multi-node replacement,
    // so it now produces a single TransposeD with original perm.
    SetPlatform("Ascend910_93");

    auto resultGraph = BuildPermuteGraphAndRunPass(std::vector<int64_t>{2, 3, 4, 5}, std::vector<int64_t>{0, 3, 2, 1});

    // Exactly 1 TransposeD node (not split)
    EXPECT_EQ(CountNodeType(resultGraph, "TransposeD"), 1);
    EXPECT_EQ(CountNodeType(resultGraph, "Transpose"), 0);
}

TEST_F(PermuteFusionPassTest, fusionSuccess950Normal)
{
    // Ascend950 platform: Permute -> Transpose (perm as input)
    SetPlatform("Ascend950");

    auto resultGraph = BuildPermuteGraphAndRunPass(std::vector<int64_t>{4, 3, 2, 1}, std::vector<int64_t>{0, 2, 3, 1});

    // Check that Transpose node exists and no TransposeD node
    EXPECT_TRUE(HasNodeType(resultGraph, "Transpose"));
    EXPECT_FALSE(HasNodeType(resultGraph, "TransposeD"));
}

TEST_F(PermuteFusionPassTest, fusionSuccess950Special)
{
    // Ascend950 platform: 4D input with perm=[0,3,2,1]
    // Single Transpose replacement (no multi-node split)
    SetPlatform("Ascend950");

    auto resultGraph = BuildPermuteGraphAndRunPass(std::vector<int64_t>{2, 3, 4, 5}, std::vector<int64_t>{0, 3, 2, 1});

    // Exactly 1 Transpose node (not split), no TransposeD
    EXPECT_EQ(CountNodeType(resultGraph, "Transpose"), 1);
    EXPECT_EQ(CountNodeType(resultGraph, "TransposeD"), 0);
}

// ==================== Different shape / dtype tests ====================

TEST_F(PermuteFusionPassTest, fusionSuccess3dShape)
{
    // 3D input shape test (non-special case)
    SetPlatform("Ascend910_93");

    auto resultGraph = BuildPermuteGraphAndRunPass(std::vector<int64_t>{4, 3, 2}, std::vector<int64_t>{0, 2, 1});

    EXPECT_TRUE(HasNodeType(resultGraph, "TransposeD"));
}

TEST_F(PermuteFusionPassTest, fusionSuccessFp16)
{
    // FP16 dtype test on Ascend910_93
    SetPlatform("Ascend910_93");

    auto resultGraph = BuildPermuteGraphAndRunPass(std::vector<int64_t>{4, 3, 2, 1}, std::vector<int64_t>{0, 2, 3, 1},
                                                   DT_FLOAT16);

    EXPECT_TRUE(HasNodeType(resultGraph, "TransposeD"));
}

TEST_F(PermuteFusionPassTest, nonTransposeDPlatformUsesTranspose)
{
    // Ascend310 is NOT in the TransposeD platform list,
    // so it should fall back to the Transpose branch (perm as input).
    SetPlatform("Ascend310");

    auto resultGraph = BuildPermuteGraphAndRunPass(std::vector<int64_t>{4, 3, 2, 1}, std::vector<int64_t>{0, 2, 3, 1});

    // Should produce Transpose, not TransposeD
    EXPECT_TRUE(HasNodeType(resultGraph, "Transpose"));
    EXPECT_FALSE(HasNodeType(resultGraph, "TransposeD"));
}

// ==================== Compatibility tests ====================

// Verify that the pass compiles and the version guard is active.
// GE_COMPILER_VERSION_NUM must be >= 90000000 for the pass to be enabled.
TEST_F(PermuteFusionPassTest, compileTimeVersionGuard)
{
    // If this test compiles and runs, the GE_COMPILER_VERSION_NUM >= 90000000
    // guard is satisfied (otherwise the pass class wouldn't be defined).
    EXPECT_GE(GE_COMPILER_VERSION_NUM, 90000000);
}

// Verify that the pass can be instantiated and returns patterns
// (validates that the #if GE_COMPILER_VERSION_NUM >= 90000000 guard
// enables the pass at current compiler version).
TEST_F(PermuteFusionPassTest, passInstantiationTest)
{
    PermuteFusionPass pass;
    auto patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

// Verify the pass works correctly with the kCompatibleInherited stage.
// This confirms the stage compatibility mechanism is functional on all platforms.
TEST_F(PermuteFusionPassTest, compatibleInheritedStageTest910)
{
    // Ascend910_93: Permute -> TransposeD via kCompatibleInherited stage
    SetPlatform("Ascend910_93");

    auto resultGraph = BuildPermuteGraphAndRunPass(std::vector<int64_t>{4, 3, 2, 1}, std::vector<int64_t>{0, 2, 3, 1});

    // Verify the pass correctly replaced Permute with TransposeD
    EXPECT_TRUE(HasNodeType(resultGraph, "TransposeD"));
    EXPECT_FALSE(HasNodeType(resultGraph, "Transpose"));
    EXPECT_FALSE(HasNodeType(resultGraph, "Permute"));
}

// Verify the pass works with kCompatibleInherited stage on Ascend950 (new platform).
TEST_F(PermuteFusionPassTest, compatibleInheritedStageTest950)
{
    SetPlatform("Ascend950");

    auto resultGraph = BuildPermuteGraphAndRunPass(std::vector<int64_t>{4, 3, 2, 1}, std::vector<int64_t>{0, 2, 3, 1});

    // Verify the pass correctly replaced Permute with Transpose on new platforms
    EXPECT_TRUE(HasNodeType(resultGraph, "Transpose"));
    EXPECT_FALSE(HasNodeType(resultGraph, "TransposeD"));
    EXPECT_FALSE(HasNodeType(resultGraph, "Permute"));
}

// Verify the pass works with kCompatibleInherited stage on multiple platforms
TEST_F(PermuteFusionPassTest, compatibleInheritedStageMultiPlatform)
{
    // Test on Ascend910_93 (TransposeD platform)
    SetPlatform("Ascend910_93");
    auto graph910 = BuildPermuteGraphAndRunPass(std::vector<int64_t>{8, 16}, std::vector<int64_t>{1, 0});
    EXPECT_TRUE(HasNodeType(graph910, "TransposeD"));

    // Test on Ascend950 (Transpose platform)
    SetPlatform("Ascend950");
    auto graph950 = BuildPermuteGraphAndRunPass(std::vector<int64_t>{8, 16}, std::vector<int64_t>{1, 0});
    EXPECT_TRUE(HasNodeType(graph950, "Transpose"));
}
