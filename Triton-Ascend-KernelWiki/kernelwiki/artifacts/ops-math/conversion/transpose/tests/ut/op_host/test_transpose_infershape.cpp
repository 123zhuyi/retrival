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
 * \file test_transpose_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;
class TransposeInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TransposeInferShapeTest Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TransposeInferShapeTest Proto Test TearDown" << std::endl; }
};

TEST_F(TransposeInferShapeTest, transpose_test_case_1)
{
    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT64,
                                                     ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Transpose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{31, 36, 203, 26}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TransposeInferShapeTest, transpose_test_case_2)
{
    int32_t perm_value[4] = {3, 0, 1, 2};
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT32,
                                                     ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Transpose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{31, 36, 203, 26}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TransposeInferShapeTest, transpose_test_case_3)
{
    int16_t perm_value[4] = {3, 0, 1, 2};
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT16,
                                                     ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT16, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Transpose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{31, 36, 203, 26}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(TransposeInferShapeTest, transpose_test_case_4)
{
    int16_t perm_value[4] = {3, 0, 4, 2};
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT16,
                                                     ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT16, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Transpose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{31, 36, 203, 26}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Negative perm values, expect to resolve and infer shape correctly
TEST_F(TransposeInferShapeTest, transpose_infershape_negative_perm_int32)
{
    int32_t perm_value[4] = {-1, 0, 1, 2};
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT32,
                                                     ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Transpose", {x, perm}, {out});
    // perm = {-1, 0, 1, 2} resolved to {3, 0, 1, 2}, output shape = {31, 36, 203, 26}
    std::vector<std::vector<int64_t>> expectOutputShape = {{31, 36, 203, 26}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Negative perm values with int64 dtype, expect success
TEST_F(TransposeInferShapeTest, transpose_infershape_negative_perm_int64)
{
    int64_t perm_value[3] = {-1, -2, 0};
    gert::InfershapeContextPara::TensorDescription x({{10, 20, 30}, {10, 20, 30}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Transpose", {x, perm}, {out});
    // perm = {-1, -2, 0} resolved to {2, 1, 0}, output shape = {30, 20, 10}
    std::vector<std::vector<int64_t>> expectOutputShape = {{30, 20, 10}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Perm size mismatch with input dims, expect failure
TEST_F(TransposeInferShapeTest, transpose_infershape_perm_size_mismatch)
{
    int64_t perm_value[3] = {0, 1, 2};
    gert::InfershapeContextPara::TensorDescription x({{10, 20, 30, 40}, {10, 20, 30, 40}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Transpose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Perm value out of valid range (perm[i] >= dimSize), expect failure
TEST_F(TransposeInferShapeTest, transpose_infershape_perm_value_out_of_range)
{
    int32_t perm_value[4] = {3, 0, 4, 2};
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT32,
                                                     ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Transpose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// inserted_by_fe flag set to 1, expect shape unchanged (identity permutation)
TEST_F(TransposeInferShapeTest, transpose_infershape_inserted_by_fe_flag)
{
    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT64,
                                                     ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Transpose", {x, perm}, {out},
        {gert::InfershapeContextPara::OpAttr("_inserted_by_fe", Ops::Math::AnyValue::CreateFrom<int64_t>(1))});
    // With inserted_by_fe=1, output shape stays same as input (identity)
    std::vector<std::vector<int64_t>> expectOutputShape = {{36, 203, 26, 31}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 1D transpose with perm [0], expect tensor move (identity)
TEST_F(TransposeInferShapeTest, transpose_infershape_1d_tensor)
{
    int64_t perm_value[1] = {0};
    gert::InfershapeContextPara::TensorDescription x({{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Transpose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{100}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 场景：perm为INT64类型，perm中包含超出输入维度有效范围的值（perm[2]=4，但输入为4维tensor，perm值最大应为3）
// 期望：推导失败
TEST_F(TransposeInferShapeTest, transpose_infershape_int64_invalid_perm_value)
{
    int64_t perm_value[4] = {3, 0, 4, 2}; // perm[2]=4 is out of range for 4-dim input
    gert::InfershapeContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT64,
                                                     ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Transpose", {x, perm}, {out});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
