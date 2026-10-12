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
#include <array>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class GetShapeInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GetShapeInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GetShapeInferShapeTest TearDown" << std::endl; }
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

TEST_F(GetShapeInferShapeTest, SingleInput_3D_Tensor)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GetShape")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape input_shape_0 = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GetShape")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&input_shape_0})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ToVector(*holder.GetContext()->GetOutputShape(0)), std::vector<int64_t>({3}));
}

TEST_F(GetShapeInferShapeTest, TwoInputs_2D_And_3D)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GetShape")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape input_shape_0 = {{2, 3}, {2, 3}};
    gert::StorageShape input_shape_1 = {{4, 5, 6}, {4, 5, 6}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GetShape")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({2}, {1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&input_shape_0, (gert::Tensor*)&input_shape_1})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ToVector(*holder.GetContext()->GetOutputShape(0)), std::vector<int64_t>({5}));
}

TEST_F(GetShapeInferShapeTest, ScalarInput_Fail)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GetShape")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape input_shape_0 = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GetShape")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&input_shape_0})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(GetShapeInferShapeTest, VectorInput)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GetShape")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape input_shape_0 = {{128}, {128}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GetShape")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&input_shape_0})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ToVector(*holder.GetContext()->GetOutputShape(0)), std::vector<int64_t>({1}));
}

TEST_F(GetShapeInferShapeTest, MultipleDtypes)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("GetShape")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape input_shape_0 = {{2, 3}, {2, 3}};
    gert::StorageShape input_shape_1 = {{4, 5}, {4, 5}};
    gert::StorageShape input_shape_2 = {{6, 7}, {6, 7}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GetShape")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({3}, {1})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors(
                          {(gert::Tensor*)&input_shape_0, (gert::Tensor*)&input_shape_1, (gert::Tensor*)&input_shape_2})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ToVector(*holder.GetContext()->GetOutputShape(0)), std::vector<int64_t>({6}));
}
