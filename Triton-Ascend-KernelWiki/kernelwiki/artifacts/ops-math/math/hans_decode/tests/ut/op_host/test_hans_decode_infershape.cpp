/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>  // NOLINT
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class HansDecode : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "HansDecode SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "HansDecode TearDown" << std::endl;
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
    const auto& input1StorageShape = inputShapes[0];
    const auto& input2StorageShape = inputShapes[1];
    const auto& input3StorageShape = inputShapes[2];
    const auto& input4StorageShape = inputShapes[3];
   
    ge::DataType input1Dtype = dtypes[0];
    ge::DataType input2Dtype = dtypes[1];
    ge::DataType input3Dtype = dtypes[2];
    ge::DataType input4Dtype = dtypes[3];
    ge::DataType output1Dtype = dtypes[4];

    /* make infershape context */
    std::vector<gert::Tensor *> inputTensors = {
    	(gert::Tensor *)&input1StorageShape,
        (gert::Tensor *)&input2StorageShape,
        (gert::Tensor *)&input3StorageShape,
        (gert::Tensor *)&input4StorageShape
    };
    std::vector<gert::StorageShape *> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("HansDecode")
        .NodeIoNum(4, 1)
        .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, input3Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, input4Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, output1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("HansDecode")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}


TEST_F(HansDecode, HansDecode_infershape_case_0) {
    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {
        {{4096}, {4096} },
	    {{16384}, {16384}},
	    {{16384}, {16384}},
        {{256}, {256}},
    };
    std::vector<ge::DataType> dtypes = {
        ge::DT_FLOAT, 
        ge::DT_FLOAT,   
        ge::DT_FLOAT,
        ge::DT_INT32, 
        ge::DT_FLOAT
    };

    std::vector<int64_t> expectResult = {};
    gert::StorageShape outStorageShape = {};
    // 简化后的函数调用
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

