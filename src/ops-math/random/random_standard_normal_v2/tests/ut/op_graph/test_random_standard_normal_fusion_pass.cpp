/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "random/random_standard_normal_v2/op_graph/fusion_pass/random_standard_normal_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

class RandomStandardNormalFusionPassTest : public testing::Test {
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

// Test 1: Verify Pattern can be created correctly
TEST_F(RandomStandardNormalFusionPassTest, patternTest)
{
    ops::RandomStandardNormalFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

// Test 2: Verify unsupported dtype returns GRAPH_NOT_CHANGED
TEST_F(RandomStandardNormalFusionPassTest, unsupportedDtypeFail)
{
    std::vector<int64_t> shapeDims{2};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto shape = graphBuilder.CreateInput(0, "shape", DT_INT64, FORMAT_ND, shapeDims);
    // DT_INT32 is not in the supported dtype list
    auto output = es::RandomStandardNormal(shape, DT_INT32, 1024, 2048);

    TensorDesc shapeDesc;
    shape.GetProducer()->GetOutputDesc(0, shapeDesc);
    shapeDesc.SetDataType(DT_INT64);
    shapeDesc.SetShape(Shape(shapeDims));
    shapeDesc.SetFormat(FORMAT_ND);
    shape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});
    CustomPassContext passContext;
    ops::RandomStandardNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test 3: Verify unsupported platform returns GRAPH_NOT_CHANGED
TEST_F(RandomStandardNormalFusionPassTest, unsupportedPlatformFail)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.str_info.short_soc_version = "Ascend310";
    optiCompilationInfo.soc_version = "Ascend310";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend310"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> shapeDims{2};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto shape = graphBuilder.CreateInput(0, "shape", DT_INT64, FORMAT_ND, shapeDims);
    auto output = es::RandomStandardNormal(shape, DT_FLOAT, 1024, 2048);

    TensorDesc shapeDesc;
    shape.GetProducer()->GetOutputDesc(0, shapeDesc);
    shapeDesc.SetDataType(DT_INT64);
    shapeDesc.SetShape(Shape(shapeDims));
    shapeDesc.SetFormat(FORMAT_ND);
    shape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});
    CustomPassContext passContext;
    ops::RandomStandardNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

// Test 4: Verify successful fusion with float32 dtype
TEST_F(RandomStandardNormalFusionPassTest, fusionFloatSuccess)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> shapeDims{2};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto shape = graphBuilder.CreateInput(0, "shape", DT_INT64, FORMAT_ND, shapeDims);
    auto output = es::RandomStandardNormal(shape, DT_FLOAT, 1024, 2048);

    TensorDesc shapeDesc;
    shape.GetProducer()->GetOutputDesc(0, shapeDesc);
    shapeDesc.SetDataType(DT_INT64);
    shapeDesc.SetShape(Shape(shapeDims));
    shapeDesc.SetFormat(FORMAT_ND);
    shape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});
    CustomPassContext passContext;
    ops::RandomStandardNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);

    // Verify RandomStandardNormalV2 node exists
    bool foundV2 = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "RandomStandardNormalV2") {
            foundV2 = true;
        }
    }
    EXPECT_TRUE(foundV2);
}

// Test 5: Verify successful fusion with float16 dtype
TEST_F(RandomStandardNormalFusionPassTest, fusionFloat16Success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> shapeDims{3};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto shape = graphBuilder.CreateInput(0, "shape", DT_INT64, FORMAT_ND, shapeDims);
    auto output = es::RandomStandardNormal(shape, DT_FLOAT16, 0, 0);

    TensorDesc shapeDesc;
    shape.GetProducer()->GetOutputDesc(0, shapeDesc);
    shapeDesc.SetDataType(DT_INT64);
    shapeDesc.SetShape(Shape(shapeDims));
    shapeDesc.SetFormat(FORMAT_ND);
    shape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});
    CustomPassContext passContext;
    ops::RandomStandardNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);

    bool foundV2 = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "RandomStandardNormalV2") {
            foundV2 = true;
        }
    }
    EXPECT_TRUE(foundV2);
}

// Test 6: Verify successful fusion with bfloat16 dtype
TEST_F(RandomStandardNormalFusionPassTest, fusionBf16Success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> shapeDims{2};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto shape = graphBuilder.CreateInput(0, "shape", DT_INT64, FORMAT_ND, shapeDims);
    auto output = es::RandomStandardNormal(shape, DT_BF16, 512, 1024);

    TensorDesc shapeDesc;
    shape.GetProducer()->GetOutputDesc(0, shapeDesc);
    shapeDesc.SetDataType(DT_INT64);
    shapeDesc.SetShape(Shape(shapeDims));
    shapeDesc.SetFormat(FORMAT_ND);
    shape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});
    CustomPassContext passContext;
    ops::RandomStandardNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);

    bool foundV2 = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "RandomStandardNormalV2") {
            foundV2 = true;
        }
    }
    EXPECT_TRUE(foundV2);
}

// Test 7: Verify successful fusion on Ascend950 platform
TEST_F(RandomStandardNormalFusionPassTest, fusion950Success)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> shapeDims{4};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto shape = graphBuilder.CreateInput(0, "shape", DT_INT64, FORMAT_ND, shapeDims);
    auto output = es::RandomStandardNormal(shape, DT_FLOAT, 512, 1024);

    TensorDesc shapeDesc;
    shape.GetProducer()->GetOutputDesc(0, shapeDesc);
    shapeDesc.SetDataType(DT_INT64);
    shapeDesc.SetShape(Shape(shapeDims));
    shapeDesc.SetFormat(FORMAT_ND);
    shape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});
    CustomPassContext passContext;
    ops::RandomStandardNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);

    bool foundV2 = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "RandomStandardNormalV2") {
            foundV2 = true;
        }
    }
    EXPECT_TRUE(foundV2);
}

// Test 8: Verify successful fusion with different shape dimension
TEST_F(RandomStandardNormalFusionPassTest, fusionDifferentShapeSuccess)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> shapeDims{4};

    auto graphBuilder = es::EsGraphBuilder("test");
    auto shape = graphBuilder.CreateInput(0, "shape", DT_INT64, FORMAT_ND, shapeDims);
    auto output = es::RandomStandardNormal(shape, DT_FLOAT, 0, 0);

    TensorDesc shapeDesc;
    shape.GetProducer()->GetOutputDesc(0, shapeDesc);
    shapeDesc.SetDataType(DT_INT64);
    shapeDesc.SetShape(Shape(shapeDims));
    shapeDesc.SetFormat(FORMAT_ND);
    shape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});
    CustomPassContext passContext;
    ops::RandomStandardNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);

    bool foundV2 = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == "RandomStandardNormalV2") {
            foundV2 = true;
        }
    }
    EXPECT_TRUE(foundV2);
}