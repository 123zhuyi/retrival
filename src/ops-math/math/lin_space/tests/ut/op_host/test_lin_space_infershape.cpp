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

class linspace : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "linspace Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "linspace Proto Test TearDown" << std::endl;
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

template <typename T>
gert::Tensor * ConstructInputConstTensor(std::unique_ptr<uint8_t[]>& input_tensor_holder,
                                          const T &const_value, ge::DataType const_dtype) {
  auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
  gert::Tensor tensor({{1}, {1}},                                 // shape
                      {ge::FORMAT_ND, ge::FORMAT_ND, {}},        // format
                      gert::kFollowing,                          // placement
                      const_dtype,                               //dt
                      nullptr);
  std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
  auto tensor_data = reinterpret_cast<T *>(input_tensor + 1);
  *tensor_data = const_value;
  std::cout<<" const_value:" << *tensor_data<< std::endl;
  input_tensor->SetData(gert::TensorData(tensor_data, nullptr));
  return input_tensor;
}

static void ExeTestCase(
    std::vector<std::vector<int64_t> > expectResults,
    const std::vector<gert::StorageShape>& inputShapes,  // 存储所有输入StorageShape参数
    const std::vector<ge::DataType>& dtypes,             // 存储所有DataType参数
    const std::vector<int32_t> inputValues,
    gert::StorageShape& outStorageShape,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& startStorageShape = inputShapes[0];
    const auto& stopStorageShape = inputShapes[1];
    const auto& numStorageShape = inputShapes[2];
    
    ge::DataType input1Dtype = dtypes[0];
    ge::DataType outputDtype = dtypes[1];

    auto start_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t)]);
    auto start_tensor = ConstructInputConstTensor(start_input_tensor, inputValues[0], ge::DT_INT32);
    auto stop_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t)]);
    auto stop_tensor = ConstructInputConstTensor(stop_input_tensor, inputValues[1], ge::DT_INT32);
    auto num_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t)]);
    auto num_tensor = ConstructInputConstTensor(num_input_tensor, inputValues[2], ge::DT_INT32);

    /* make infershape context */
    std::vector<gert::Tensor *> inputTensors = {
        start_tensor,
        stop_tensor,
        num_tensor
    };
    std::vector<gert::StorageShape *> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
        .SetOpType("LinSpace")
        .NodeIoNum(3, 1)
        .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputTensors(inputTensors)
        .OutputShapes(outputShapes)
        .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("LinSpace")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(linspace, Linspace_infershape_case_0)
{
    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {
        {{1}, {1}},                  // start_shape
        {{1}, {1}},                  // stop_shape
        {{1}, {1}}                           // num_shape
    };
    std::vector<ge::DataType> dtypes = {
        ge::DT_INT32,  // input1Dtype
        ge::DT_INT32   // outputDtype
    };
    std::vector<int32_t> inputValues = {1, 10, 10};
    std::vector<int64_t> expectResult = {10};
    gert::StorageShape outStorageShape = {};

    // 简化后的函数调用
    ExeTestCase({expectResult}, inputShapes, dtypes, inputValues, outStorageShape, ge::GRAPH_SUCCESS);
}

