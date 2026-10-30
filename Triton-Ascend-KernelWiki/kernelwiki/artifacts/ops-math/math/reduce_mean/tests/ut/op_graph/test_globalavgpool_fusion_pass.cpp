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
#include "version/ge-compiler_version.h"
// #include "es_math_ops.h"  // 暂时注释掉，可能不需要
#include "../../../op_graph/fusion_pass/globalavgpool_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

class GlobalavgpoolPassTest : public testing::Test {
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

TEST_F(GlobalavgpoolPassTest, pattern_test)
{
    ops::GlobalavgpoolPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(GlobalavgpoolPassTest, fusion_success_3d)
{
    std::vector<int64_t> dims_x{2, 3, 4};  // 3D input
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("globalavgpool_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    
    // 设置输入节点的输出描述（形状和数据类型）在连接边之前
    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_FLOAT);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);
    
    // 创建 GlobalAveragePool 节点（使用 CompliantNodeBuilder）
    // 需要获取Graph对象
    auto* graph = graph_builder.GetCGraphBuilder()->GetGraph();
    auto globalAvgPool = es::CompliantNodeBuilder(graph)
        .OpType("GlobalAveragePool")
        .Name("global_avg_pool")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
        })
        .Build();
    
    // 使用AddEdgeAndUpdatePeerDesc连接边
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), globalAvgPool, 0) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge in test";
    }
    
    // 更新GlobalAveragePool节点的输入描述，确保形状信息传递
    TensorDesc pool_input_desc;
    globalAvgPool.GetInputDesc(0, pool_input_desc);
    pool_input_desc.SetDataType(DT_FLOAT);
    pool_input_desc.SetShape(shape_x);
    globalAvgPool.UpdateInputDesc(0, pool_input_desc);
    
    // 获取输出张量
    auto y = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(globalAvgPool, 0);

    // 构建图 - 需要将EsTensorHolder放入vector中
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    std::shared_ptr<Graph> graph_ptr = graph_builder.BuildAndReset(outputs);
    CustomPassContext pass_context;
    ops::GlobalavgpoolPass pass;
    Status status = pass.Run(graph_ptr, pass_context);
    EXPECT_EQ(status, SUCCESS);

    // 检查是否转换为 ReduceMean
    bool findReduceMean = false;
    for (auto node : graph_ptr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "ReduceMean") {
            findReduceMean = true;
            // 检查属性
            bool keep_dims = true;
            bool noop_with_empty_axes = true;
            node.GetAttr("keep_dims", keep_dims);
            node.GetAttr("noop_with_empty_axes", noop_with_empty_axes);
            EXPECT_TRUE(keep_dims);
            EXPECT_TRUE(noop_with_empty_axes);
            break;
        }
    }
    EXPECT_TRUE(findReduceMean);
}

TEST_F(GlobalavgpoolPassTest, fusion_success_4d)
{
    std::vector<int64_t> dims_x{2, 3, 4, 5};  // 4D input
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("globalavgpool_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, shape_x.GetDims());
    
    // 设置输入节点的输出描述（形状和数据类型）在连接边之前
    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_FLOAT16);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);
    
    // 创建 GlobalAveragePool 节点（使用 CompliantNodeBuilder）
    // 需要获取Graph对象
    auto* graph = graph_builder.GetCGraphBuilder()->GetGraph();
    auto globalAvgPool = es::CompliantNodeBuilder(graph)
        .OpType("GlobalAveragePool")
        .Name("global_avg_pool")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
        })
        .Build();
    
    // 使用AddEdgeAndUpdatePeerDesc连接边
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), globalAvgPool, 0) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge in test";
    }
    
    // 更新GlobalAveragePool节点的输入描述，确保形状信息传递
    TensorDesc pool_input_desc;
    globalAvgPool.GetInputDesc(0, pool_input_desc);
    pool_input_desc.SetDataType(DT_FLOAT16);
    pool_input_desc.SetShape(shape_x);
    globalAvgPool.UpdateInputDesc(0, pool_input_desc);
    
    // 获取输出张量
    auto y = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(globalAvgPool, 0);

    // 构建图 - 需要将EsTensorHolder放入vector中
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    std::shared_ptr<Graph> graph_ptr = graph_builder.BuildAndReset(outputs);
    CustomPassContext pass_context;
    ops::GlobalavgpoolPass pass;
    Status status = pass.Run(graph_ptr, pass_context);
    EXPECT_EQ(status, SUCCESS);

    // 检查是否转换为 ReduceMean
    bool findReduceMean = false;
    for (auto node : graph_ptr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "ReduceMean") {
            findReduceMean = true;
            // 检查属性
            bool keep_dims = true;
            bool noop_with_empty_axes = true;
            node.GetAttr("keep_dims", keep_dims);
            node.GetAttr("noop_with_empty_axes", noop_with_empty_axes);
            EXPECT_TRUE(keep_dims);
            EXPECT_TRUE(noop_with_empty_axes);
            break;
        }
    }
    EXPECT_TRUE(findReduceMean);
}

TEST_F(GlobalavgpoolPassTest, fusion_success_5d)
{
    std::vector<int64_t> dims_x{2, 3, 4, 5, 6};  // 5D input
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("globalavgpool_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_INT32, FORMAT_ND, shape_x.GetDims());
    
    // 设置输入节点的输出描述（形状和数据类型）在连接边之前
    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_INT32);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);
    
    // 创建 GlobalAveragePool 节点（使用 CompliantNodeBuilder）
    // 需要获取Graph对象
    auto* graph = graph_builder.GetCGraphBuilder()->GetGraph();
    auto globalAvgPool = es::CompliantNodeBuilder(graph)
        .OpType("GlobalAveragePool")
        .Name("global_avg_pool")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
        })
        .Build();
    
    // 使用AddEdgeAndUpdatePeerDesc连接边
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), globalAvgPool, 0) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge in test";
    }
    
    // 更新GlobalAveragePool节点的输入描述，确保形状信息传递
    TensorDesc pool_input_desc;
    globalAvgPool.GetInputDesc(0, pool_input_desc);
    pool_input_desc.SetDataType(DT_INT32);
    pool_input_desc.SetShape(shape_x);
    globalAvgPool.UpdateInputDesc(0, pool_input_desc);
    
    // 获取输出张量
    auto y = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(globalAvgPool, 0);

    // 构建图 - 需要将EsTensorHolder放入vector中
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    std::shared_ptr<Graph> graph_ptr = graph_builder.BuildAndReset(outputs);
    CustomPassContext pass_context;
    ops::GlobalavgpoolPass pass;
    Status status = pass.Run(graph_ptr, pass_context);
    EXPECT_EQ(status, SUCCESS);

    // 检查是否转换为 ReduceMean
    bool findReduceMean = false;
    for (auto node : graph_ptr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "ReduceMean") {
            findReduceMean = true;
            // 检查属性
            bool keep_dims = true;
            bool noop_with_empty_axes = true;
            node.GetAttr("keep_dims", keep_dims);
            node.GetAttr("noop_with_empty_axes", noop_with_empty_axes);
            EXPECT_TRUE(keep_dims);
            EXPECT_TRUE(noop_with_empty_axes);
            break;
        }
    }
    EXPECT_TRUE(findReduceMean);
}

TEST_F(GlobalavgpoolPassTest, unsupported_dims_fail)
{
    std::vector<int64_t> dims_x{2, 3};  // 2D input, not supported
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("globalavgpool_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    
    // 设置输入节点的输出描述（形状和数据类型）在连接边之前
    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_FLOAT);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);
    
    // 创建 GlobalAveragePool 节点（使用 CompliantNodeBuilder）
    // 需要获取Graph对象
    auto* graph = graph_builder.GetCGraphBuilder()->GetGraph();
    auto globalAvgPool = es::CompliantNodeBuilder(graph)
        .OpType("GlobalAveragePool")
        .Name("global_avg_pool")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
        })
        .Build();
    
    // 使用AddEdgeAndUpdatePeerDesc连接边
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), globalAvgPool, 0) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge in test";
    }
    
    // 更新GlobalAveragePool节点的输入描述，确保形状信息传递
    TensorDesc pool_input_desc;
    globalAvgPool.GetInputDesc(0, pool_input_desc);
    pool_input_desc.SetDataType(DT_FLOAT);
    pool_input_desc.SetShape(shape_x);
    globalAvgPool.UpdateInputDesc(0, pool_input_desc);
    
    // 获取输出张量
    auto y = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(globalAvgPool, 0);

    // 构建图
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    std::shared_ptr<Graph> graph_ptr = graph_builder.BuildAndReset(outputs);
    CustomPassContext pass_context;
    ops::GlobalavgpoolPass pass;
    Status status = pass.Run(graph_ptr, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(GlobalavgpoolPassTest, fusion_success_950)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    std::vector<int64_t> dims_x{2, 3, 4, 5};
    Shape shape_x(dims_x);

    auto graph_builder = es::EsGraphBuilder("globalavgpool_fusion_test");
    auto x = graph_builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shape_x.GetDims());
    
    // 设置输入节点的输出描述（形状和数据类型）在连接边之前
    TensorDesc x_output_desc;
    x.GetProducer()->GetOutputDesc(0, x_output_desc);
    x_output_desc.SetDataType(DT_FLOAT);
    x_output_desc.SetShape(shape_x);
    x.GetProducer()->UpdateOutputDesc(0, x_output_desc);
    
    // 创建 GlobalAveragePool 节点（使用 CompliantNodeBuilder）
    // 需要获取Graph对象
    auto* graph = graph_builder.GetCGraphBuilder()->GetGraph();
    auto globalAvgPool = es::CompliantNodeBuilder(graph)
        .OpType("GlobalAveragePool")
        .Name("global_avg_pool")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
        })
        .Build();
    
    // 使用AddEdgeAndUpdatePeerDesc连接边
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), globalAvgPool, 0) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge in test";
    }
    
    // 更新GlobalAveragePool节点的输入描述，确保形状信息传递
    TensorDesc pool_input_desc;
    globalAvgPool.GetInputDesc(0, pool_input_desc);
    pool_input_desc.SetDataType(DT_FLOAT);
    pool_input_desc.SetShape(shape_x);
    globalAvgPool.UpdateInputDesc(0, pool_input_desc);
    
    // 获取输出张量
    auto y = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(globalAvgPool, 0);

    // 构建图 - 需要将EsTensorHolder放入vector中
    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    std::shared_ptr<Graph> graph_ptr = graph_builder.BuildAndReset(outputs);
    CustomPassContext pass_context;
    ops::GlobalavgpoolPass pass;
    Status status = pass.Run(graph_ptr, pass_context);
    EXPECT_EQ(status, SUCCESS);

    // 检查是否转换为 ReduceMean
    bool findReduceMean = false;
    for (auto node : graph_ptr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "ReduceMean") {
            findReduceMean = true;
            // 检查属性
            bool keep_dims = true;
            bool noop_with_empty_axes = true;
            node.GetAttr("keep_dims", keep_dims);
            node.GetAttr("noop_with_empty_axes", noop_with_empty_axes);
            EXPECT_TRUE(keep_dims);
            EXPECT_TRUE(noop_with_empty_axes);
            break;
        }
    }
    EXPECT_TRUE(findReduceMean);
}

// ==================== Compatibility tests ====================

// Verify that the pass compiles and the version guard is active.
TEST_F(GlobalavgpoolPassTest, compileTimeVersionGuard)
{
    EXPECT_GE(GE_COMPILER_VERSION_NUM, 90000000);
}

// Verify that the pass can be instantiated and returns patterns.
TEST_F(GlobalavgpoolPassTest, passInstantiationTest)
{
    ops::GlobalavgpoolPass pass;
    auto patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

// Verify the pass works correctly with the kCompatibleInherited stage mechanism.
TEST_F(GlobalavgpoolPassTest, compatibleInheritedStageTest)
{
    std::vector<int64_t> dimsX{2, 3, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("compat_stage_test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());

    TensorDesc xOutputDesc;
    x.GetProducer()->GetOutputDesc(0, xOutputDesc);
    xOutputDesc.SetDataType(DT_FLOAT);
    xOutputDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xOutputDesc);

    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto globalAvgPool = es::CompliantNodeBuilder(graph)
        .OpType("GlobalAveragePool")
        .Name("global_avg_pool")
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();

    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(),
        globalAvgPool, 0) != GRAPH_SUCCESS) {
        FAIL() << "Failed to add edge in test";
    }

    TensorDesc poolInputDesc;
    globalAvgPool.GetInputDesc(0, poolInputDesc);
    poolInputDesc.SetDataType(DT_FLOAT);
    poolInputDesc.SetShape(shapeX);
    globalAvgPool.UpdateInputDesc(0, poolInputDesc);

    auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(globalAvgPool, 0);

    std::vector<ge::es::EsTensorHolder> outputs;
    outputs.push_back(ge::es::EsTensorHolder(y));
    std::shared_ptr<Graph> graphPtr = graphBuilder.BuildAndReset(outputs);

    CustomPassContext passContext;
    ops::GlobalavgpoolPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    bool findReduceMean = false;
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "ReduceMean") {
            findReduceMean = true;
            break;
        }
    }
    EXPECT_TRUE(findReduceMean);
}