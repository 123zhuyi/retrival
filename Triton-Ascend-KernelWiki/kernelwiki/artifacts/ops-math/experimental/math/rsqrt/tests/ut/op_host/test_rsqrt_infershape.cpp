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
 * \file test_rsqrt_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RsqrtInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RsqrtInfershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "RsqrtInfershape TearDown" << std::endl; }
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

static void ExeTestCase(std::vector<std::vector<int64_t>> expectResults,
                        const std::vector<gert::StorageShape>& inputShapes, const std::vector<ge::DataType>& dtypes,
                        gert::StorageShape& outStorageShape, ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    const auto& xStorageShape = inputShapes[0];
    ge::DataType inputDtype = dtypes[0];
    ge::DataType outputDtype = dtypes[1];

    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&xStorageShape,
    };
    std::vector<gert::StorageShape*> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("Rsqrt")
                             .NodeIoNum(1, 1)
                             .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto opImpl = spaceRegistry->GetOpImpl("Rsqrt");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(RsqrtInfershape, Rsqrt_infershape_case_0)
{
    std::vector<gert::StorageShape> inputShapes = {
        {{2, 100, 4}, {2, 100, 4}},
    };
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};

    std::vector<int64_t> expectResult = {2, 100, 4};
    gert::StorageShape outStorageShape = {};

    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(RsqrtInfershape, Rsqrt_infershape_case_1)
{
    std::vector<gert::StorageShape> inputShapes = {
        {{8, 2048}, {8, 2048}},
    };
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};

    std::vector<int64_t> expectResult = {8, 2048};
    gert::StorageShape outStorageShape = {};

    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(RsqrtInfershape, Rsqrt_infershape_case_2)
{
    std::vector<gert::StorageShape> inputShapes = {
        {{1023, 2047}, {1023, 2047}},
    };
    std::vector<ge::DataType> dtypes = {ge::DT_BF16, ge::DT_BF16};

    std::vector<int64_t> expectResult = {1023, 2047};
    gert::StorageShape outStorageShape = {};

    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}
