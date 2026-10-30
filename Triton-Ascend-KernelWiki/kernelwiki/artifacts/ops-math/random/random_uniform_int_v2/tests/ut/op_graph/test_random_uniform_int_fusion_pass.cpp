/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "gtest/gtest.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "../../../op_graph/fusion_pass/random_uniform_int_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

class RandomUniformIntFusionPassTest : public testing::Test {
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
        SetPlatform950();
    }

    void SetPlatform950()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUnsupportedPlatform()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

static std::shared_ptr<Graph> BuildTestGraph(const std::string& graphName, DataType dtype,
                                             const std::vector<int64_t>& shapeDims, int64_t seed = 1024,
                                             int64_t seed2 = 2048)
{
    Shape outShape(shapeDims);

    auto graphBuilder = es::EsGraphBuilder(graphName.c_str());
    auto shape = graphBuilder.CreateInput(0, "shape", DT_INT64, FORMAT_ND,
                                          std::vector<int64_t>{static_cast<int64_t>(shapeDims.size())});
    auto min = graphBuilder.CreateInput(1, "min", dtype, FORMAT_ND, std::vector<int64_t>{});
    auto max = graphBuilder.CreateInput(2, "max", dtype, FORMAT_ND, std::vector<int64_t>{});

    auto output = es::RandomUniformInt(shape, min, max, seed, seed2);

    TensorDesc shapeDesc;
    shape.GetProducer()->GetOutputDesc(0, shapeDesc);
    shapeDesc.SetDataType(DT_INT64);
    shapeDesc.SetShape(Shape(std::vector<int64_t>{static_cast<int64_t>(shapeDims.size())}));
    shapeDesc.SetFormat(FORMAT_ND);
    shape.GetProducer()->UpdateOutputDesc(0, shapeDesc);

    TensorDesc minDesc;
    min.GetProducer()->GetOutputDesc(0, minDesc);
    minDesc.SetDataType(dtype);
    minDesc.SetShape(Shape(std::vector<int64_t>{}));
    minDesc.SetFormat(FORMAT_ND);
    min.GetProducer()->UpdateOutputDesc(0, minDesc);

    TensorDesc maxDesc;
    max.GetProducer()->GetOutputDesc(0, maxDesc);
    maxDesc.SetDataType(dtype);
    maxDesc.SetShape(Shape(std::vector<int64_t>{}));
    maxDesc.SetFormat(FORMAT_ND);
    max.GetProducer()->UpdateOutputDesc(0, maxDesc);

    TensorDesc outputDesc;
    output.GetProducer()->GetOutputDesc(0, outputDesc);
    outputDesc.SetDataType(dtype);
    outputDesc.SetShape(outShape);
    outputDesc.SetFormat(FORMAT_ND);
    output.GetProducer()->UpdateOutputDesc(0, outputDesc);

    output.GetProducer()->UpdateInputDesc(0, shapeDesc);
    output.GetProducer()->UpdateInputDesc(1, minDesc);
    output.GetProducer()->UpdateInputDesc(2, maxDesc);

    auto graph = graphBuilder.BuildAndReset({output});

    return graph;
}

static bool HasNodeType(const std::shared_ptr<Graph>& graph, const std::string& targetType)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) == targetType) {
            return true;
        }
    }
    return false;
}

TEST_F(RandomUniformIntFusionPassTest, patternTest)
{
    RandomUniformIntFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(RandomUniformIntFusionPassTest, unsupportedPlatformFail)
{
    SetUnsupportedPlatform();

    auto graph = BuildTestGraph("test_unsupported_platform", DT_INT32, {10});

    CustomPassContext passContext;
    RandomUniformIntFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(RandomUniformIntFusionPassTest, unsupportedDtypeFail)
{
    auto graph = BuildTestGraph("test_unsupported_dtype", DT_FLOAT, {10});

    CustomPassContext passContext;
    RandomUniformIntFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(RandomUniformIntFusionPassTest, fusionSuccessInt32)
{
    auto graph = BuildTestGraph("test_int32", DT_INT32, {10});

    CustomPassContext passContext;
    RandomUniformIntFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(HasNodeType(graph, "RandomUniformIntV2"));
}

TEST_F(RandomUniformIntFusionPassTest, fusionSuccessInt64)
{
    auto graph = BuildTestGraph("test_int64", DT_INT64, {3});

    CustomPassContext passContext;
    RandomUniformIntFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(HasNodeType(graph, "RandomUniformIntV2"));
}

TEST_F(RandomUniformIntFusionPassTest, fusionSuccessMultiDimShape)
{
    auto graph = BuildTestGraph("test_multidim", DT_INT32, {4, 5});

    CustomPassContext passContext;
    RandomUniformIntFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(HasNodeType(graph, "RandomUniformIntV2"));
}

TEST_F(RandomUniformIntFusionPassTest, fusionSuccessLargeShape)
{
    auto graph = BuildTestGraph("test_large", DT_INT64, {128, 64, 32});

    CustomPassContext passContext;
    RandomUniformIntFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(HasNodeType(graph, "RandomUniformIntV2"));
}

TEST_F(RandomUniformIntFusionPassTest, fusionSuccessDefaultSeed)
{
    auto graph = BuildTestGraph("test_default_seed", DT_INT32, {10}, 0, 0);

    CustomPassContext passContext;
    RandomUniformIntFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, SUCCESS);
    EXPECT_TRUE(HasNodeType(graph, "RandomUniformIntV2"));
}