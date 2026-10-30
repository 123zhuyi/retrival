/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_broadcast_to_infershape.cpp
 * \brief UT for BroadcastTo InferShape
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class BroadcastToInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "broadcast_to Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "broadcast_to Proto Test TearDown" << std::endl; }
};

// Test scenario: INT64 dtype for shape tensor, broadcast [1,7,2] -> [6,7,2]
TEST_F(BroadcastToInferShapeTest, broadcast_to_infershape_test_case_1)
{
    gert::StorageShape shape = {{1, 7, 2}, {1, 7, 2}};
    gert::StorageShape shape1 = {{3}, {3}};
    int64_t shapes[4] = {6, 7, 2};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shape1, ge::DT_INT64, ge::FORMAT_ND, true, &shapes);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("BroadcastTo", {x, shapeParams}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{6, 7, 2}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: INT32 dtype for shape tensor to cover GetConstValueToShape<int32_t> branch
TEST_F(BroadcastToInferShapeTest, broadcast_to_infershape_int32_shape_001)
{
    gert::StorageShape xShape = {{1, 3, 4}, {1, 3, 4}};
    gert::StorageShape shapeParamShape = {{3}, {3}};
    int32_t shapes[3] = {2, 3, 4};
    gert::InfershapeContextPara::TensorDescription x(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shapeParamShape, ge::DT_INT32, ge::FORMAT_ND, true,
                                                               &shapes);
    gert::InfershapeContextPara::TensorDescription y(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("BroadcastTo", {x, shapeParams}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{2, 3, 4}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: shape tensor with more dims than x (diff > 0) to cover i < diff continue branch
TEST_F(BroadcastToInferShapeTest, broadcast_to_infershape_expand_dims_001)
{
    gert::StorageShape xShape = {{3, 4}, {3, 4}};
    gert::StorageShape shapeParamShape = {{4}, {4}};
    int64_t shapes[4] = {5, 1, 3, 4};
    gert::InfershapeContextPara::TensorDescription x(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shapeParamShape, ge::DT_INT64, ge::FORMAT_ND, true,
                                                               &shapes);
    gert::InfershapeContextPara::TensorDescription y(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("BroadcastTo", {x, shapeParams}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{5, 1, 3, 4}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: shape tensor with -1 dim at position within diff range to cover SetDim(i, 1) branch
TEST_F(BroadcastToInferShapeTest, broadcast_to_infershape_minus1_in_diff_001)
{
    gert::StorageShape xShape = {{3, 4}, {3, 4}};
    gert::StorageShape shapeParamShape = {{4}, {4}};
    int64_t shapes[4] = {-1, 1, 3, 4};
    gert::InfershapeContextPara::TensorDescription x(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shapeParamShape, ge::DT_INT64, ge::FORMAT_ND, true,
                                                               &shapes);
    gert::InfershapeContextPara::TensorDescription y(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("BroadcastTo", {x, shapeParams}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{1, 1, 3, 4}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: shape tensor with -1 dim at position outside diff range to cover SetDim(i, x_shape) branch
TEST_F(BroadcastToInferShapeTest, broadcast_to_infershape_minus1_in_x_range_001)
{
    gert::StorageShape xShape = {{3, 4}, {3, 4}};
    gert::StorageShape shapeParamShape = {{2}, {2}};
    int64_t shapes[2] = {-1, 4};
    gert::InfershapeContextPara::TensorDescription x(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shapeParamShape, ge::DT_INT64, ge::FORMAT_ND, true,
                                                               &shapes);
    gert::InfershapeContextPara::TensorDescription y(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("BroadcastTo", {x, shapeParams}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{3, 4}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: attrs-based path (no shape tensor) to cover BroadcastToInferShapeWithShapeValues
// The "_shape_values" attribute is a gert::ContinuousVector stored as vector<int64_t> via AnyValue
TEST_F(BroadcastToInferShapeTest, broadcast_to_infershape_attrs_path_001)
{
    gert::StorageShape xShape = {{1, 3, 4}, {1, 3, 4}};
    gert::StorageShape yShape = {{1, 3, 4}, {1, 3, 4}};
    std::vector<int64_t> shapeValues = {2, 3, 4};
    gert::InfershapeContextPara::TensorDescription x(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::OpAttr shapeAttr("_shape_values",
                                                  Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(shapeValues));
    gert::InfershapeContextPara infershapeContextPara("BroadcastTo", {x}, {y}, {shapeAttr});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{2, 3, 4}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: shape_size < x dimNum to cover the OP_CHECK_IF failure path (lines 69-74)
TEST_F(BroadcastToInferShapeTest, broadcast_to_infershape_shape_size_less_than_x_001)
{
    gert::StorageShape xShape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
    gert::StorageShape shapeParamShape = {{2}, {2}};
    int64_t shapes[2] = {2, 3};
    gert::InfershapeContextPara::TensorDescription x(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shapeParamShape, ge::DT_INT64, ge::FORMAT_ND, true,
                                                               &shapes);
    gert::InfershapeContextPara::TensorDescription y(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("BroadcastTo", {x, shapeParams}, {y});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// Test scenario: invalid dtype for shape tensor to cover the OP_CHECK_IF failure path (lines 78-83)
TEST_F(BroadcastToInferShapeTest, broadcast_to_infershape_invalid_dtype_001)
{
    gert::StorageShape xShape = {{1, 3, 4}, {1, 3, 4}};
    gert::StorageShape shapeParamShape = {{3}, {3}};
    float shapes[3] = {2.0f, 3.0f, 4.0f};
    gert::InfershapeContextPara::TensorDescription x(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shapeParamShape, ge::DT_FLOAT, ge::FORMAT_ND, true,
                                                               &shapes);
    gert::InfershapeContextPara::TensorDescription y(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("BroadcastTo", {x, shapeParams}, {y});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// Test scenario: broadcast mismatch (x dim != 1 and != output dim) to cover OP_CHECK_IF failure (lines 102-107)
TEST_F(BroadcastToInferShapeTest, broadcast_to_infershape_mismatch_001)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape shapeParamShape = {{3}, {3}};
    int64_t shapes[3] = {5, 3, 4};
    gert::InfershapeContextPara::TensorDescription x(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shapeParamShape, ge::DT_INT64, ge::FORMAT_ND, true,
                                                               &shapes);
    gert::InfershapeContextPara::TensorDescription y(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("BroadcastTo", {x, shapeParams}, {y});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
