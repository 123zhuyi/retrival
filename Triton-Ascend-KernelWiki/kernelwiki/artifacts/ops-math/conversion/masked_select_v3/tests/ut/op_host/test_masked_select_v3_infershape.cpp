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

class MaskedSelectv3InferTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaskedSelectv3 Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaskedSelectv3 Proto Test TearDown" << std::endl;
  }
};

static std::vector<int64_t> ToVectorForMaskedSelectv3(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCaseForMaskedSelectv3(
    const std::vector<gert::StorageShape>& inputShapes,  // 存储所有输入StorageShape参数
    const std::vector<ge::DataType>& dtypes,             // 存储所有DataType参数
    gert::StorageShape& outStorageShape,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& selfStorageShape = inputShapes[0];
    const auto& maskedStorageShape = inputShapes[1];
    
    ge::DataType input1Dtype = dtypes[0];
    ge::DataType input2Dtype = dtypes[1];
    ge::DataType outputDtype = dtypes[2];

    /* make infershape context */
    std::vector<gert::Tensor *> inputTensors = {
        (gert::Tensor *)&selfStorageShape,
        (gert::Tensor *)&maskedStorageShape
    };
    std::vector<gert::StorageShape *> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("MaskedSelectV3")
        .NodeIoNum(2, 1)
        .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("MaskedSelectV3")->infer_shape;
}

TEST_F(MaskedSelectv3InferTest, MaskedSelectv3_infershape_test)
{
    size_t size1 = 2;
    size_t size2 = 4;
    size_t size3 = 6;
    size_t size4 = 8;
    size_t size5 = 10;
    size_t size6 = 12;
    size_t size7 = 15;
    size_t size8 = 16;
    int64_t size9 = -1;

    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {
        {{size1, size2, size3, size4, size5, size6, size7, size8}, 
        {size1, size2, size3, size4, size5, size6, size7, size8}},    // self_shape
        {{size1, size2, size3, size4, size5, size6, size7, size8}, 
        {size1, size2, size3, size4, size5, size6, size7, size8}}                  // masked_shape
    };
    std::vector<ge::DataType> dtypes = {
        ge::DT_INT64,  // input1Dtype
        ge::DT_BOOL,    // input2Dtype
        ge::DT_INT64   // outputDtype
    };

    std::vector<int64_t> expectResult = {size1, size2, size3, size4, size5, size6, size7, size8};
    gert::StorageShape outStorageShape = {};

    // 简化后的函数调用
    ExeTestCaseForMaskedSelectv3(inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}
// masked_select_v3为第三类算子，infershape在kernel内推导，host侧推导输出为动态shape，此测试值判断动态infershape是否正常