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

class ReduceStdV2Update : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceStdV2Update SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceStdV2Update TearDown" << std::endl; }
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

static void ExeTestCase(std::vector<int64_t> expectedResult, const std::vector<gert::StorageShape>& inputShapes,
                        const std::vector<ge::DataType>& dtypes, gert::StorageShape& outStorageShape,
                        std::vector<int64_t> dim, bool if_std, bool unbiased, bool keepdim, int64_t correction,
                        ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    const auto& xStorageShape = inputShapes[0];
    ge::DataType outputDtype = dtypes[0];

    std::vector<gert::Tensor*> inputTensors = {(gert::Tensor*)&xStorageShape};
    std::vector<gert::StorageShape*> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("ReduceStdV2Update")
                             .NodeIoNum(2, 1)
                             .NodeInputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("dim", dim)
                             .Attr("if_std", if_std)
                             .Attr("unbiased", unbiased)
                             .Attr("keepdim", keepdim)
                             .Attr("correction", correction)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("ReduceStdV2Update")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(0)), expectedResult);
}

TEST_F(ReduceStdV2Update, infershape_keepdim_false)
{
    std::vector<gert::StorageShape> inputShapes = {{{4, 8}, {4, 8}}};
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<int64_t> expectedResult = {4};
    gert::StorageShape outStorageShape = {};
    ExeTestCase(expectedResult, inputShapes, dtypes, outStorageShape, {1}, false, true, false, 1);
}

TEST_F(ReduceStdV2Update, infershape_keepdim_true)
{
    std::vector<gert::StorageShape> inputShapes = {{{4, 8}, {4, 8}}};
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<int64_t> expectedResult = {4, 1};
    gert::StorageShape outStorageShape = {};
    ExeTestCase(expectedResult, inputShapes, dtypes, outStorageShape, {1}, false, true, true, 1);
}

TEST_F(ReduceStdV2Update, infershape_all_reduce)
{
    std::vector<gert::StorageShape> inputShapes = {{{4, 8}, {4, 8}}};
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<int64_t> expectedResult = {};
    gert::StorageShape outStorageShape = {};
    ExeTestCase(expectedResult, inputShapes, dtypes, outStorageShape, {}, false, true, false, 1);
}

TEST_F(ReduceStdV2Update, infershape_negative_dim)
{
    std::vector<gert::StorageShape> inputShapes = {{{4, 8}, {4, 8}}};
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
    std::vector<int64_t> expectedResult = {4};
    gert::StorageShape outStorageShape = {};
    ExeTestCase(expectedResult, inputShapes, dtypes, outStorageShape, {-1}, true, false, false, 0);
}
