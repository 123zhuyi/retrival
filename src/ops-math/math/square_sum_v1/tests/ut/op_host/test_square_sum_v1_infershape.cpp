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
#include "gtest/gtest.h"
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SquareSumV1Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SquareSumV1Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SquareSumV1Test TearDown" << std::endl; }
};

// ==================== keep_dims=true, 多轴归约 ====================
TEST_F(SquareSumV1Test, SquareSumV1_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 1, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== keep_dims=false, 含负轴 ====================
TEST_F(SquareSumV1Test, SquareSumV1_keepdims_false_neg_axes)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, -2})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 标量输入, keep_dims=false ====================
TEST_F(SquareSumV1Test, SquareSumV1_scalar_keepdims_false)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 标量输入, keep_dims=true ====================
TEST_F(SquareSumV1Test, SquareSumV1_scalar_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 单轴归约, keep_dims=true ====================
TEST_F(SquareSumV1Test, SquareSumV1_single_axis_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 16, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 单轴归约, keep_dims=false ====================
TEST_F(SquareSumV1Test, SquareSumV1_single_axis_keepdims_false)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 16, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 归约所有轴, keep_dims=true ====================
TEST_F(SquareSumV1Test, SquareSumV1_all_dims_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 2, 3})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 归约所有轴, keep_dims=false → 标量 ====================
TEST_F(SquareSumV1Test, SquareSumV1_all_dims_keepdims_false)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 2, 3})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 1D输入, keep_dims=true ====================
TEST_F(SquareSumV1Test, SquareSumV1_1d_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 1D输入, keep_dims=false → 标量 ====================
TEST_F(SquareSumV1Test, SquareSumV1_1d_keepdims_false)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 动态shape(-1), keep_dims=true ====================
TEST_F(SquareSumV1Test, SquareSumV1_dynamic_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{3, -1, 16, 16}, {3, -1, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, -1, 1, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 动态shape(-1), keep_dims=false ====================
TEST_F(SquareSumV1Test, SquareSumV1_dynamic_keepdims_false)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{3, -1, 16, 16}, {3, -1, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 16, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 全负轴, keep_dims=true ====================
TEST_F(SquareSumV1Test, SquareSumV1_neg_axes_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-1, -2})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5, 1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 5D输入, keep_dims=true ====================
TEST_F(SquareSumV1Test, SquareSumV1_5d_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 4})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 1, 4, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 5D输入, keep_dims=false ====================
TEST_F(SquareSumV1Test, SquareSumV1_5d_keepdims_false)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 4})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 无效axes范围 (axis超出维度范围) → GRAPH_FAILED ====================
TEST_F(SquareSumV1Test, SquareSumV1_invalid_axes_bounds)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({4})}, // dimNum=4, 有效范围[-4,3]
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// ==================== 无效axes范围 (负轴超出范围) → GRAPH_FAILED ====================
TEST_F(SquareSumV1Test, SquareSumV1_invalid_axes_bounds_neg)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-5})}, // dimNum=4, 有效范围[-4,3]
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// ==================== 无效axes范围 (标量输入, axis!=0) → GRAPH_FAILED ====================
TEST_F(SquareSumV1Test, SquareSumV1_invalid_axes_bounds_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})}, // 标量只允许axis=0
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// ==================== 未知秩输入 ({-2}) → 输出 {-2} ====================
TEST_F(SquareSumV1Test, SquareSumV1_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 保留原有测试 ====================
TEST_F(SquareSumV1Test, square_sum_v1_infershape_test_01)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SquareSumV1Test, square_sum_v1_infershape_test_02)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{-1, 8, -1, -1}, {-1, 8, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 2})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 8, 1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SquareSumV1Test, static_square_sum_v1_infershape_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{16, 48, 16, 32}, {16, 48, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 2})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 48, 1, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SquareSumV1Test, InfershapeSquareSumV1_001)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{4, 3, 1}, {4, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-1, 2, 1})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
