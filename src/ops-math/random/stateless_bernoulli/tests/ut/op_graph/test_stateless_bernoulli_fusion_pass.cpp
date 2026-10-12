/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "../../../op_graph/fusion_pass/stateless_bernoulli_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;

namespace {
const std::string kPassName = "BernoulliFusionPass";
}

class BernoulliFusionPassTest : public testing::Test {
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

    void SetPlatform91093()
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

TEST_F(BernoulliFusionPassTest, patternTest)
{
    BernoulliFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(BernoulliFusionPassTest, unsupportedPlatformFail)
{
    PlatformInfo platformInfo;
    OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910";
    optiCompilationInfo.soc_version = "Ascend910";
    PlatformInfoManager::Instance().platform_info_map_["Ascend910"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dimsX{-1, 3, 5, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);


    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);


    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessFp16On950)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-1, 3, 5, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);
    TensorDesc outputDesc;
    output.GetProducer()->GetOutputDesc(0, outputDesc);
    outputDesc.SetDataType(DT_FLOAT16);
    outputDesc.SetShape(shapeX);
    outputDesc.SetFormat(FORMAT_NCHW);
    output.GetProducer()->UpdateOutputDesc(0, outputDesc);

    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessFp16On91093)
{
    SetPlatform91093();

    std::vector<int64_t> dimsX{-1, 3, 5, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);


    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);


    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessFp32)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-1, 3, 5, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);


    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);


    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessInt64Dtype)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-1, 3, 5, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset, DT_INT64);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);


    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);


    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(BernoulliFusionPassTest, fusionFailUnknownRank)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-2};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset, DT_INT64);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);


    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);


    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessLargeShape)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-1, 7, 8, 6};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset, DT_INT64);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);


    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);


    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessHighDim)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-1, 3, 5, 4, 4, 2, 2};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset, DT_INT64);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);


    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);


    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}