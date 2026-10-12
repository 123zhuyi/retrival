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
 * \file test_expand_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ExpandInferShapeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "expand Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "expand Proto Test TearDown" << std::endl;
  }
};

TEST_F(ExpandInferShapeTest, expand_infershape_test_case_1){
    gert::StorageShape shape = {{1,7,2}, {1,7,2}};
    gert::StorageShape shape1 = {{3}, {3}};
    int64_t shapes[4] = {6, 7, 2};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shape1, ge::DT_INT64, ge::FORMAT_ND, true, &shapes);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Expand",
        { x, shapeParams },
        { y });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{6,7,2}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: expand with DT_INT32 dtype for shape tensor to cover the INT32 branch in GetConstValueToShape
TEST_F(ExpandInferShapeTest, expand_infershape_int32_dtype_001)
{
    gert::StorageShape shape = {{3, 7}, {3, 7}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 7};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Expand", {x, shapeParams}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{3, 7}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: expand with -1 in shape tensor to cover the out_dim == -1 branch
TEST_F(ExpandInferShapeTest, expand_infershape_negative_one_001)
{
    gert::StorageShape shape = {{3, 7}, {3, 7}};
    gert::StorageShape shape1 = {{2}, {2}};
    int64_t shapes[2] = {-1, 7};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shape1, ge::DT_INT64, ge::FORMAT_ND, true, &shapes);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Expand", {x, shapeParams}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{3, 7}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: expand broadcast failure - x dim incompatible with target dim (x_dim != 1 && x_dim != out_dim)
TEST_F(ExpandInferShapeTest, expand_infershape_broadcast_fail_001)
{
    gert::StorageShape shape = {{3, 7}, {3, 7}};
    gert::StorageShape shape1 = {{2}, {2}};
    int64_t shapes[2] = {3, 5};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shape1, ge::DT_INT64, ge::FORMAT_ND, true, &shapes);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Expand", {x, shapeParams}, {y});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Test scenario: expand with fewer input dims and -1 in new leading dimensions (i < diff branch)
TEST_F(ExpandInferShapeTest, expand_infershape_new_dim_minus_one_001)
{
    gert::StorageShape shape = {{7, 2}, {7, 2}};
    gert::StorageShape shape1 = {{3}, {3}};
    int64_t shapes[3] = {-1, 7, 2};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shape1, ge::DT_INT64, ge::FORMAT_ND, true, &shapes);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Expand", {x, shapeParams}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{1, 7, 2}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: expand where target dim is 1 but x dim is not 1 (out_dim == 1 && x_dim != 1)
TEST_F(ExpandInferShapeTest, expand_infershape_out_dim_1_x_not_1_001)
{
    gert::StorageShape shape = {{3, 7}, {3, 7}};
    gert::StorageShape shape1 = {{2}, {2}};
    int64_t shapes[2] = {3, 1};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shapeParams(shape1, ge::DT_INT64, ge::FORMAT_ND, true, &shapes);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Expand", {x, shapeParams}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {{3, 7}},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
