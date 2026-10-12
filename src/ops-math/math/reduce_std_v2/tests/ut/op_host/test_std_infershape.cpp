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

class ReduceStdV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceStdV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReduceStdV2 TearDown" << std::endl;
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
    std::vector<std::vector<int64_t>> expectResults,
    const std::vector<gert::StorageShape>& inputShapes, // 存储所有输入StorageShape参数
    const std::vector<ge::DataType>& dtypes,            // 存储所有DataType参数
    gert::StorageShape& outStorageShape, std::vector<int64_t> dim, int64_t correction, bool keepdim, bool is_mean_out,
    ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& x1StorageShape = inputShapes[0];

    ge::DataType input1Dtype = dtypes[0];
    ge::DataType outputDtype = dtypes[0];

    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {(gert::Tensor*)&x1StorageShape};

    std::vector<int64_t> batchshape = {};
    std::vector<gert::StorageShape*> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("ReduceStdV2")
                             .NodeIoNum(1, 2)
                             .NodeInputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("dim", dim)
                             .Attr("correction", correction)
                             .Attr("keepdim", keepdim)
                             .Attr("is_mean_out", is_mean_out)
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("ReduceStdV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(ReduceStdV2, ReduceStdV2_infershape_case_0)
{
    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {{{64, 10240}, {64, 10240}}};

    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};

    std::vector<std::vector<int64_t>> expectResult = {{64, 1}, {64, 1}};
    gert::StorageShape outStorageShape = {};

    // 简化后的函数调用
    ExeTestCase(expectResult, inputShapes, dtypes, outStorageShape, {1}, 0, true, true, ge::GRAPH_SUCCESS);
}