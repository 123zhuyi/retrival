/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class HistogramV2Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HistogramV2Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "HistogramV2Test TearDown" << std::endl; }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(std::vector<std::vector<int64_t> > expectResults,
                        const std::vector<gert::StorageShape>& inputShapes, // 存储所有输入StorageShape参数
                        const std::vector<ge::DataType>& dtypes,            // 存储所有DataType参数
                        gert::StorageShape& outStorageShape, ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& x1StorageShape = inputShapes[0];
    const auto& x2StorageShape = inputShapes[1];
    const auto& x3StorageShape = inputShapes[2];

    ge::DataType input1Dtype = dtypes[0];
    ge::DataType outputDtype = dtypes[1];

    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {(gert::Tensor*)&x1StorageShape, (gert::Tensor*)&x2StorageShape,
                                               (gert::Tensor*)&x3StorageShape};
    const std::string& attrName = "bins";
    int64_t value = 100;
    std::vector<gert::StorageShape*> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("HistogramV2")
                             .NodeIoNum(3, 1)
                             .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr(attrName, value)
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("HistogramV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(HistogramV2Test, HistogramV2_infershape_case_0)
{
    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {
        {{16, 16}, {16, 16}},
        {{16, 16}, {16, 16}},
        {{16, 16}, {16, 16}},
    };
    std::vector<ge::DataType> dtypes = {
        ge::DT_FLOAT16, // input1Dtype
        ge::DT_FLOAT16  // outputDtype
    };

    std::vector<int64_t> expectResult = {100};
    gert::StorageShape outStorageShape = {};

    // 简化后的函数调用
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

// bins <= 0 should make InferShape fail (covers the OP_CHECK_IF error branch).
static void RunInferShapeWithBins(int64_t bins, ge::graphStatus expect)
{
    gert::StorageShape x0 = {{16, 16}, {16, 16}};
    gert::StorageShape x1 = {{16, 16}, {16, 16}};
    gert::StorageShape x2 = {{16, 16}, {16, 16}};
    gert::StorageShape outStorageShape = {};
    std::vector<gert::Tensor*> inputTensors = {(gert::Tensor*)&x0, (gert::Tensor*)&x1, (gert::Tensor*)&x2};
    std::vector<gert::StorageShape*> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("HistogramV2")
                             .NodeIoNum(3, 1)
                             .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("bins", bins)
                             .Build();
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("HistogramV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), expect);
}

TEST_F(HistogramV2Test, HistogramV2_infershape_bins_negative) { RunInferShapeWithBins(-1, ge::GRAPH_FAILED); }

TEST_F(HistogramV2Test, HistogramV2_infershape_bins_zero) { RunInferShapeWithBins(0, ge::GRAPH_FAILED); }

TEST_F(HistogramV2Test, HistogramV2_infershape_bins_custom) { RunInferShapeWithBins(64, ge::GRAPH_SUCCESS); }
