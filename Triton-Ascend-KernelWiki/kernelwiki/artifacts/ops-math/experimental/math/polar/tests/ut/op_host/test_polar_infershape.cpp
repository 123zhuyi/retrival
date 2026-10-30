/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_polar_infershape.cpp
 * \brief Polar InferShape UT：验证 numpy 广播（右对齐各轴取 max）推导 out shape。
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class PolarInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PolarInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PolarInfershape TearDown" << std::endl; }
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

// input、angle 均为 FLOAT，out 恒为 COMPLEX64；out shape = input 与 angle 的 numpy 广播结果。
static void ExeTestCase(const std::vector<int64_t>& expectResult, gert::StorageShape& inputStorageShape,
                        gert::StorageShape& angleStorageShape, gert::StorageShape& outStorageShape,
                        ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&inputStorageShape,
        (gert::Tensor*)&angleStorageShape,
    };
    std::vector<gert::StorageShape*> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("Polar")
                             .NodeIoNum(2, 1)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_COMPLEX64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("Polar")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(0)), expectResult);
}

// 同形：input[4,2] 与 angle[4,2] → out[4,2]
TEST_F(PolarInfershape, polar_infershape_same_shape)
{
    gert::StorageShape inputShape = {{4, 2}, {4, 2}};
    gert::StorageShape angleShape = {{4, 2}, {4, 2}};
    gert::StorageShape outShape = {};
    ExeTestCase({4, 2}, inputShape, angleShape, outShape, ge::GRAPH_SUCCESS);
}

// 同秩广播：input[4,1] 与 angle[1,2] → out[4,2]
TEST_F(PolarInfershape, polar_infershape_broadcast)
{
    gert::StorageShape inputShape = {{4, 1}, {4, 1}};
    gert::StorageShape angleShape = {{1, 2}, {1, 2}};
    gert::StorageShape outShape = {};
    ExeTestCase({4, 2}, inputShape, angleShape, outShape, ge::GRAPH_SUCCESS);
}

// 不同秩：input[2] 与 angle[3,2] → out[3,2]（右对齐，高维取 angle）
TEST_F(PolarInfershape, polar_infershape_diff_rank)
{
    gert::StorageShape inputShape = {{2}, {2}};
    gert::StorageShape angleShape = {{3, 2}, {3, 2}};
    gert::StorageShape outShape = {};
    ExeTestCase({3, 2}, inputShape, angleShape, outShape, ge::GRAPH_SUCCESS);
}

// 标量广播：input[1] 与 angle[5,6] → out[5,6]
TEST_F(PolarInfershape, polar_infershape_scalar_broadcast)
{
    gert::StorageShape inputShape = {{1}, {1}};
    gert::StorageShape angleShape = {{5, 6}, {5, 6}};
    gert::StorageShape outShape = {};
    ExeTestCase({5, 6}, inputShape, angleShape, outShape, ge::GRAPH_SUCCESS);
}
