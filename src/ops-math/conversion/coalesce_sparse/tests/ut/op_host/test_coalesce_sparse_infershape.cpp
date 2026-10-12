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

class CoalesceSparseTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "CoalesceSparse Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CoalesceSparse Proto Test TearDown" << std::endl;
  }
};

static std::vector<int64_t> ToVectorForCoalesceSparse(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCaseForCoalesceSparse(
    std::vector<std::vector<int64_t> > expectResults,
    const std::vector<gert::StorageShape>& inputShapes,
    const std::vector<ge::DataType>& dtypes,
    std::vector<gert::StorageShape>& outStorageShapes,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    const auto& input0StorageShape = inputShapes[0];
    const auto& input1StorageShape = inputShapes[1];
    const auto& input2StorageShape = inputShapes[2];
    const auto& input3StorageShape = inputShapes[3];
    
    ge::DataType input0Dtype = dtypes[0];
    ge::DataType input1Dtype = dtypes[1];
    ge::DataType input2Dtype = dtypes[2];
    ge::DataType input3Dtype = dtypes[3];
    ge::DataType output0Dtype = dtypes[4];
    ge::DataType output1Dtype = dtypes[5];

    std::vector<gert::Tensor *> inputTensors = {
        (gert::Tensor *)&input0StorageShape,
        (gert::Tensor *)&input1StorageShape,
        (gert::Tensor *)&input2StorageShape,
        (gert::Tensor *)&input3StorageShape
    };
    std::vector<gert::StorageShape *> outputShapes = {
        &outStorageShapes[0],
        &outStorageShapes[1]
    };
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("CoalesceSparse")
        .NodeIoNum(2, 1)
        .NodeInputTd(0, input0Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, input3Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, output0Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, output1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("CoalesceSparse")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVectorForCoalesceSparse(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(CoalesceSparseTest, coalesce_sparse_test_success)
{
    std::vector<gert::StorageShape> inputShapes = {
        {{4}, {4}},
        {{5}, {5}},
        {{5, 2}, {5, 2}},
        {{5, 3}, {5, 3}},
    };
    std::vector<ge::DataType> dtypes = {
        ge::DT_INT32,
        ge::DT_INT32,
        ge::DT_INT32,
        ge::DT_FLOAT16,
        ge::DT_INT32,
        ge::DT_FLOAT16
    };

    std::vector<int64_t> expectResult1 = {4, 2};
    std::vector<int64_t> expectResult2 = {4, 3};
    std::vector<gert::StorageShape> outStorageShapes = {{}, {}};

    ExeTestCaseForCoalesceSparse({expectResult1, expectResult2}, inputShapes, dtypes, outStorageShapes, ge::GRAPH_SUCCESS);
}
