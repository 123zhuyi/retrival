/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

class AddLora : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddLora SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddLora TearDown" << std::endl;
    }
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

static void ExeTestCase(
    std::vector<std::vector<int64_t> > expectResults,
    const std::vector<gert::StorageShape>& inputShapes,  // 存储所有输入StorageShape参数
    const std::vector<ge::DataType>& dtypes,             // 存储所有DataType参数
    gert::StorageShape& outStorageShape,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& yStorageShape = inputShapes[0];
    const auto& xStorageShape = inputShapes[1];
    const auto& weightAStorageShape = inputShapes[2];
    const auto& weightBStorageShape = inputShapes[3];
    const auto& indiceStorageShape = inputShapes[4];
    
    ge::DataType input1Dtype = dtypes[0];
    ge::DataType input2Dtype = dtypes[1];
    ge::DataType outputDtype = dtypes[2];

    /* make infershape context */
    std::vector<gert::Tensor *> inputTensors = {
        (gert::Tensor *)&yStorageShape,
        (gert::Tensor *)&xStorageShape,
        (gert::Tensor *)&weightBStorageShape,
        (gert::Tensor *)&indiceStorageShape,
        (gert::Tensor *)&weightAStorageShape
    };
    std::vector<gert::StorageShape *> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("AddLora")
        .NodeIoNum(5, 1)
        .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("AddLora")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(AddLora, AddLora_infershape_case_0)
{
    size_t Batch = 1;
    size_t H2 = 4096;
    size_t H1 = 16;
    size_t weight = 2;
    size_t layer = 1;
    size_t R = 16;

    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {
        {{Batch, H2}, {Batch, H2}},                       // y_shape
        {{Batch, H1}, {Batch, H1}},                       // x_shape
        {{weight, layer, R, H1}, {weight, layer, R, H1}}, // weightA_shape
        {{weight, layer, H2, R}, {weight, layer, H2, R}}, // weightB_shape
        {{Batch}, {Batch}}                                // indice_shape
    };
    
    std::vector<ge::DataType> dtypes = {
        ge::DT_FLOAT16,  // input1Dtype
        ge::DT_INT32,    // input2Dtype
        ge::DT_FLOAT16   // outputDtype
    };

    std::vector<int64_t> expectResult = {Batch, H2};
    gert::StorageShape outStorageShape = {};

    // 简化后的函数调用
    ExeTestCase({expectResult},inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}
