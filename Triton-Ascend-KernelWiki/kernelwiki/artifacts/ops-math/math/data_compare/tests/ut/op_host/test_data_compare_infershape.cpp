/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

class DataCompareInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DataCompareInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DataCompareInferShapeTest TearDown" << std::endl;
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
    const std::vector<gert::StorageShape>& inputShapes,
    const std::vector<ge::DataType>& dtypes,
    gert::StorageShape& outStorageShape, ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    const auto& x1StorageShape = inputShapes[0];
    const auto& x2StorageShape = inputShapes[1];

    ge::DataType input1Dtype = dtypes[0];
    ge::DataType input2Dtype = dtypes[1];
    ge::DataType outputDtype = dtypes[2];

    std::vector<gert::Tensor*> inputTensors = {(gert::Tensor*)&x1StorageShape, (gert::Tensor*)&x2StorageShape};
    std::vector<gert::StorageShape*> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("DataCompare")
                             .NodeIoNum(2, 1)
                             .NodeInputTd(0, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("DataCompare")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    if (testCaseResult == ge::GRAPH_SUCCESS) {
        for (size_t i = 0; i < expectResults.size(); i++) {
            EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
        }
    }
}

// ─── 正常场景：各 dtype ───

TEST_F(DataCompareInferShapeTest, infershape_fp32_standard)
{
    std::vector<gert::StorageShape> inputShapes = {{{2, 3}, {2, 3}}, {{2, 3}, {2, 3}}};
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<int64_t> expectResult = {};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(DataCompareInferShapeTest, infershape_fp16_standard)
{
    std::vector<gert::StorageShape> inputShapes = {{{4, 5}, {4, 5}}, {{4, 5}, {4, 5}}};
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT};
    std::vector<int64_t> expectResult = {};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(DataCompareInferShapeTest, infershape_bf16_standard)
{
    std::vector<gert::StorageShape> inputShapes = {{{10}, {10}}, {{10}, {10}}};
    std::vector<ge::DataType> dtypes = {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT};
    std::vector<int64_t> expectResult = {};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(DataCompareInferShapeTest, infershape_int8_standard)
{
    std::vector<gert::StorageShape> inputShapes = {{{3, 4, 5}, {3, 4, 5}}, {{3, 4, 5}, {3, 4, 5}}};
    std::vector<ge::DataType> dtypes = {ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT};
    std::vector<int64_t> expectResult = {};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(DataCompareInferShapeTest, infershape_uint8_standard)
{
    std::vector<gert::StorageShape> inputShapes = {{{8, 8}, {8, 8}}, {{8, 8}, {8, 8}}};
    std::vector<ge::DataType> dtypes = {ge::DT_UINT8, ge::DT_UINT8, ge::DT_FLOAT};
    std::vector<int64_t> expectResult = {};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(DataCompareInferShapeTest, infershape_int32_standard)
{
    std::vector<gert::StorageShape> inputShapes = {{{100}, {100}}, {{100}, {100}}};
    std::vector<ge::DataType> dtypes = {ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT};
    std::vector<int64_t> expectResult = {};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

// ─── 边界场景 ───

TEST_F(DataCompareInferShapeTest, infershape_scalar_rank0)
{
    std::vector<gert::StorageShape> inputShapes = {{{}, {}}, {{}, {}}};
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<int64_t> expectResult = {};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(DataCompareInferShapeTest, infershape_single_element)
{
    std::vector<gert::StorageShape> inputShapes = {{{1}, {1}}, {{1}, {1}}};
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<int64_t> expectResult = {};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(DataCompareInferShapeTest, infershape_empty_tensor)
{
    std::vector<gert::StorageShape> inputShapes = {{{0, 4}, {0, 4}}, {{0, 4}, {0, 4}}};
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<int64_t> expectResult = {};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(DataCompareInferShapeTest, infershape_high_rank_8d)
{
    std::vector<gert::StorageShape> inputShapes = {
        {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}},
        {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}}
    };
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<int64_t> expectResult = {};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS);
}

// ─── 异常场景 ───

TEST_F(DataCompareInferShapeTest, infershape_failed_shape_mismatch_rank)
{
    std::vector<gert::StorageShape> inputShapes = {{{2, 3}, {2, 3}}, {{2, 3, 4}, {2, 3, 4}}};
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({}, inputShapes, dtypes, outStorageShape, ge::GRAPH_FAILED);
}

TEST_F(DataCompareInferShapeTest, infershape_failed_shape_mismatch_dim)
{
    std::vector<gert::StorageShape> inputShapes = {{{2, 3}, {2, 3}}, {{2, 4}, {2, 4}}};
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({}, inputShapes, dtypes, outStorageShape, ge::GRAPH_FAILED);
}

TEST_F(DataCompareInferShapeTest, infershape_failed_rank_exceeds_max)
{
    std::vector<gert::StorageShape> inputShapes = {
        {{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}},
        {{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}
    };
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    gert::StorageShape outStorageShape = {};
    ExeTestCase({}, inputShapes, dtypes, outStorageShape, ge::GRAPH_FAILED);
}
