/**
 * This file is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
 * All Rights Reserved.
 *
 * Author (account):
 * - Yang Zhenze <@gcw_5x5Ew5Ms>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_bias_add_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class BiasAdd : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BiasAdd infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BiasAdd infershape TearDown" << std::endl; }
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

static void ExeTestCase(const std::vector<int64_t>& expectResult, const gert::StorageShape& xStorageShape,
                        const gert::StorageShape& biasStorageShape, ge::DataType dtype, ge::Format format,
                        ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&xStorageShape,
        (gert::Tensor*)&biasStorageShape,
    };
    gert::StorageShape outStorageShape = {};
    std::vector<gert::StorageShape*> outputShapes = {&outStorageShape};

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("BiasAdd")
                             .NodeIoNum(2, 1)
                             .NodeInputTd(0, dtype, format, format)
                             .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, dtype, format, format)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("BiasAdd")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    if (testCaseResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(0)), expectResult);
    }
}

TEST_F(BiasAdd, bias_add_infershape_nhwc)
{
    gert::StorageShape xShape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape biasShape = {{5}, {5}};
    ExeTestCase({2, 3, 4, 5}, xShape, biasShape, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::GRAPH_SUCCESS);
}

TEST_F(BiasAdd, bias_add_infershape_nchw)
{
    gert::StorageShape xShape = {{8, 16, 7, 7}, {8, 16, 7, 7}};
    gert::StorageShape biasShape = {{16}, {16}};
    ExeTestCase({8, 16, 7, 7}, xShape, biasShape, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::GRAPH_SUCCESS);
}

TEST_F(BiasAdd, bias_add_infershape_2d)
{
    gert::StorageShape xShape = {{32, 19}, {32, 19}};
    gert::StorageShape biasShape = {{19}, {19}};
    ExeTestCase({32, 19}, xShape, biasShape, ge::DT_INT32, ge::FORMAT_ND, ge::GRAPH_SUCCESS);
}
