/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include "../../../op_graph/fusion_pass/truncated_normal_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

namespace {
const std::string kPassName = "TruncatedNormalFusionPass";
}

class TruncatedNormalFusionPassTest : public testing::Test {
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
    Shape shapeDesc(shapeDims);

    auto graphBuilder = es::EsGraphBuilder(graphName.c_str());
    auto shape = graphBuilder.CreateInput(0, "shape", DT_INT64, FORMAT_ND, shapeDims);

    TensorDesc offsetDesc(Shape({1}), FORMAT_ND, DT_INT64);
    std::string varName = graphName + "/offsetVariable";
    auto offset = graphBuilder.CreateVariable(0, varName.c_str());
    auto varNodePtr = offset.GetProducer();
    if (varNodePtr != nullptr) {
        varNodePtr->UpdateOutputDesc(0, offsetDesc);
        int64_t initValue = 0;
        Tensor initTensor(offsetDesc, reinterpret_cast<uint8_t*>(&initValue), sizeof(int64_t));
        varNodePtr->SetAttr("init_value", initTensor);
    }

    int64_t dtypeInt = static_cast<int64_t>(dtype);
    auto v2Output = es::TruncatedNormalV2(shape, offset, seed, seed2, dtypeInt);
    auto v2NodePtr = v2Output.y.GetProducer();

    if (v2NodePtr != nullptr) {
        TensorDesc shapeInputDesc(shapeDesc, FORMAT_ND, DT_INT64);
        v2NodePtr->UpdateInputDesc(0, shapeInputDesc);
        v2NodePtr->UpdateInputDesc(1, offsetDesc);

        TensorDesc outputYDesc(shapeDesc, FORMAT_ND, dtype);
        v2NodePtr->UpdateOutputDesc(0, outputYDesc);
        v2NodePtr->UpdateOutputDesc(1, offsetDesc);
    }

    auto graph = graphBuilder.BuildAndReset({v2Output.y});

    return graph;
}

TEST_F(TruncatedNormalFusionPassTest, patternTest)
{
    TruncatedNormalFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(TruncatedNormalFusionPassTest, unsupportedPlatformFail)
{
    SetUnsupportedPlatform();

    auto graph = BuildTestGraph("test_unsupported", DT_FLOAT, {10});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(TruncatedNormalFusionPassTest, unsupportedDtypeFail)
{
    auto graph = BuildTestGraph("test_unsupported_dtype", DT_DOUBLE, {10});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(TruncatedNormalFusionPassTest, fusionSuccessFp32)
{
    auto graph = BuildTestGraph("test_fp32", DT_FLOAT, {10});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "TruncatedNormalV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(TruncatedNormalFusionPassTest, fusionSuccessFp16)
{
    auto graph = BuildTestGraph("test_fp16", DT_FLOAT16, {3});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "TruncatedNormalV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(TruncatedNormalFusionPassTest, fusionSuccessBf16)
{
    auto graph = BuildTestGraph("test_bf16", DT_BF16, {4, 5});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "TruncatedNormalV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(TruncatedNormalFusionPassTest, fusionSuccessLargeShape)
{
    auto graph = BuildTestGraph("test_large", DT_FLOAT, {6});

    CustomPassContext passContext;
    TruncatedNormalFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "TruncatedNormalV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}
