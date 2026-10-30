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

class FeedsRepeat : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "FeedsRepeat Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "FeedsRepeat Proto Test TearDown" << std::endl;
  }
};

static std::vector<int64_t> ToVectorForFeedsRepeat(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCaseForFeedsRepeat(
    std::vector<std::vector<int64_t> > expectResults,
    const std::vector<gert::StorageShape>& inputShapes,  // 存储所有输入StorageShape参数
    const std::vector<ge::DataType>& dtypes,             // 存储所有DataType参数
    gert::StorageShape& outStorageShape,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS,
    int64_t attr = 0)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& selfStorageShape = inputShapes[0];
    const auto& feedsStorageShape = inputShapes[1];
    
    ge::DataType input1Dtype = dtypes[0];
    ge::DataType input2Dtype = dtypes[1];
    ge::DataType outputDtype = dtypes[2];

    /* make infershape context */
    std::vector<gert::Tensor *> inputTensors = {
        (gert::Tensor *)&selfStorageShape,
        (gert::Tensor *)&feedsStorageShape
    };
    std::vector<gert::StorageShape *> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("FeedsRepeat")
        .NodeIoNum(2, 1)
        .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Attr("output_feeds_size", attr)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("FeedsRepeat")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVectorForFeedsRepeat(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(FeedsRepeat, FeedsRepeat_infershape_case_tiling_key_1)
{
    size_t size1 = 4;
    size_t size2 = 5;
    size_t size3 = 6;
    size_t size4 = 7;
    size_t feeds_size = 4;
    size_t out_size = 15;

    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {
        {{size1, size2, size3, size4}, {size1, size2, size3, size4}},    // self_shape
        {{feeds_size,}, {feeds_size,}}                  // feeds_shape
    };
    std::vector<ge::DataType> dtypes = {
        ge::DT_FLOAT16,  // input1Dtype
        ge::DT_INT32,    // input2Dtype
        ge::DT_FLOAT16   // outputDtype
    };

    std::vector<int64_t> expectResult = {out_size, size2, size3, size4};
    gert::StorageShape outStorageShape = {};
    int64_t attr = 15;
    // 简化后的函数调用
    ExeTestCaseForFeedsRepeat({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS, attr);
}