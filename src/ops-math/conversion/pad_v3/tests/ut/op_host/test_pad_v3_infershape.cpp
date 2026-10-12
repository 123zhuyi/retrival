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
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class PadV3InfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PadV3InfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PadV3InfershapeTest TearDown" << std::endl; }
};

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_0)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{2, 3}, {2, 3}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_1_different_paddings)
{
    gert::StorageShape xShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape padShape = {{2, 4}, {2, 4}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_2_int64_dtype)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{2, 3}, {2, 3}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT64, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_3_wrong_paddings_num)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    std::vector<int32_t> padValues = {1, 2, 3};
    gert::StorageShape padShape = {{3}, {3}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND, true, padValues.data()},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_4_negative_output)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    std::vector<int32_t> padValues = {-5, -6, -5, -6, -5, -6};
    gert::StorageShape padShape = {{6}, {6}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND, true, padValues.data()},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_5_unknown_rank)
{
    gert::StorageShape xShape = {{-2}, {-2}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {xShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_6_2d_input)
{
    gert::StorageShape xShape = {{3, 5}, {3, 5}};
    gert::StorageShape padShape = {{2}, {2}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_7_1d_input)
{
    gert::StorageShape xShape = {{10}, {10}};
    gert::StorageShape padShape = {{2}, {2}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_8_4d_input)
{
    gert::StorageShape xShape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape padShape = {{4}, {4}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_10_empty_input)
{
    gert::StorageShape xShape = {{}, {{}}};
    gert::StorageShape padShape = {{2}, {2}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_11_with_values_attr)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{2, 3}, {2, 3}};

    std::vector<int64_t> paddingsValues = {1, 2, 3, 4, 5, 6};
    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND, false, nullptr},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("paddings_values",
                                             Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(paddingsValues))});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_12_unknown_dim_in_x)
{
    gert::StorageShape xShape = {{-1, 3, 4}, {-1, 3, 4}};
    gert::StorageShape padShape = {{2, 3}, {2, 3}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_13_zero_padding)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{3}, {3}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3InfershapeTest, pad_v3_infershape_case_14_large_padding)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{10, 10}, {10, 10}};

    gert::InfershapeContextPara infershapeContextPara("PadV3",
                                                      {{xShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND},
                                                       {padShape, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: pad_v3 with constant paddings INT32 and paddings_contiguous=true, expect correct output shape
TEST_F(PadV3InfershapeTest, pad_v3_infershape_const_int32_contiguous_true)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    // Contiguous layout: [front0, back0, front1, back1, front2, back2]
    std::vector<int32_t> padValues = {1, 2, 3, 4, 5, 6};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND, true, padValues.data()},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))});

    // Contiguous: front0=1, back0=2 => 2+1+2=5; front1=3, back1=4 => 3+3+4=10; front2=5, back2=6 => 4+5+6=15
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 10, 15},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: pad_v3 with constant paddings INT32 and paddings_contiguous=false, expect correct output shape with
// non-contiguous indexing
TEST_F(PadV3InfershapeTest, pad_v3_infershape_const_int32_contiguous_false)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    // Non-contiguous: paddings layout is [pad_front_dim0, pad_front_dim1, pad_front_dim2, pad_back_dim0, pad_back_dim1,
    // pad_back_dim2]
    std::vector<int32_t> padValues = {1, 3, 5, 2, 4, 6};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND, true, padValues.data()},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false))});

    // With non-contiguous: pad_front[0]=1, pad_back[0]=2 => dim0: 2+1+2=5
    // pad_front[1]=3, pad_back[1]=4 => dim1: 3+3+4=10
    // pad_front[2]=5, pad_back[2]=6 => dim2: 4+5+6=15
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 10, 15},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: pad_v3 with INT64 const paddings and paddings_contiguous=true, expect correct output shape
TEST_F(PadV3InfershapeTest, pad_v3_infershape_const_int64_contiguous_true)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    // Contiguous layout: [front0, back0, front1, back1, front2, back2]
    std::vector<int64_t> padValues = {1, 2, 3, 4, 5, 6};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t constantValue = 0;

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, padValues.data()},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))});

    // Contiguous: front0=1, back0=2 => 2+1+2=5; front1=3, back1=4 => 3+3+4=10; front2=5, back2=6 => 4+5+6=15
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 10, 15},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: pad_v3 with unsupported paddings dtype (FLOAT), expect GRAPH_FAILED
TEST_F(PadV3InfershapeTest, pad_v3_infershape_const_unsupported_paddings_dtype)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    float padValuesArr[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    float constantValue = 0.0f;

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {padShape, ge::DT_FLOAT, ge::FORMAT_ND, true, padValuesArr},
         {constantShape, ge::DT_FLOAT, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// Test scenario: pad_v3 with INT64 const paddings and paddings_contiguous=false, expect correct output shape
TEST_F(PadV3InfershapeTest, pad_v3_infershape_const_int64_contiguous_false)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    // Non-contiguous: [pad_front_dim0, pad_front_dim1, pad_front_dim2, pad_back_dim0, pad_back_dim1, pad_back_dim2]
    std::vector<int64_t> padValues = {1, 3, 5, 2, 4, 6};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t constantValue = 0;

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, padValues.data()},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false))});

    // Non-contiguous: pad_front[0]=1, pad_back[0]=2 => dim0: 2+1+2=5
    // pad_front[1]=3, pad_back[1]=4 => dim1: 3+3+4=10
    // pad_front[2]=5, pad_back[2]=6 => dim2: 4+5+6=15
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 10, 15},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: pad_v3 with const INT32 paddings and zero padding values, expect output same as input shape
TEST_F(PadV3InfershapeTest, pad_v3_infershape_const_int32_zero_padding)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    std::vector<int32_t> padValues = {0, 0, 0, 0, 0, 0};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND, true, padValues.data()},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))});

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: pad_v3 with const INT32 paddings and unknown dim in input, expect -1 for that output dim
TEST_F(PadV3InfershapeTest, pad_v3_infershape_const_int32_unknown_dim)
{
    gert::StorageShape xShape = {{-1, 3, 4}, {-1, 3, 4}};
    std::vector<int32_t> padValues = {1, 2, 3, 4, 5, 6};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND, true, padValues.data()},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))});

    // dim0 unknown => -1; dim1: 3+3+4=10; dim2: 4+5+6=15
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 10, 15},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 场景：INT32常量模式padding，paddings_contiguous=true，paddings数量不足（只有3个值，需要6个才能匹配3维输入）
// 期望：推导失败
TEST_F(PadV3InfershapeTest, pad_v3_infershape_const_int32_contiguous_true_wrong_paddings_num)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    // Only 3 values, but need 6 (3 dims * PAIR=2)
    std::vector<int32_t> padValues = {1, 2, 3};
    gert::StorageShape padShape = {{3}, {3}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND, true, padValues.data()},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 场景：INT32常量模式padding，paddings_contiguous=true，padding值均为负数导致输出维度为负值
// 期望：推导失败
TEST_F(PadV3InfershapeTest, pad_v3_infershape_const_int32_contiguous_true_negative_dim)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    // Negative padding values causing output dim < 0
    std::vector<int32_t> padValues = {-5, -6, -5, -6, -5, -6};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND, true, padValues.data()},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 场景：INT64常量模式padding，paddings_contiguous=true，paddings数量不足（只有3个值，需要6个才能匹配3维输入）
// 期望：推导失败
TEST_F(PadV3InfershapeTest, pad_v3_infershape_const_int64_contiguous_true_wrong_paddings_num)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    // Only 3 values, but need 6 (3 dims * PAIR=2)
    std::vector<int64_t> padValues = {1LL, 2LL, 3LL};
    gert::StorageShape padShape = {{3}, {3}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t constantValue = 0LL;

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, padValues.data()},
         {constantShape, ge::DT_INT64, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 场景：INT64常量模式padding，paddings_contiguous=true，padding值均为负数导致输出维度为负值
// 期望：推导失败
TEST_F(PadV3InfershapeTest, pad_v3_infershape_const_int64_contiguous_true_negative_dim)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    // Negative padding values causing output dim < 0
    std::vector<int64_t> padValues = {-5LL, -6LL, -5LL, -6LL, -5LL, -6LL};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int64_t constantValue = 0LL;

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT64, ge::FORMAT_ND, true, padValues.data()},
         {constantShape, ge::DT_INT64, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true))});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// ========== paddings输入tensor为空、通过_paddings_values属性提供padding信息的场景 ==========

// 场景：paddings输入tensor为空（由_paddings_values属性提供padding值），paddings_contiguous=true，输入shape为[2,3,4]，paddings_values为[1,2,3,4,5,6]
// 期望：推导成功，输出shape为[5,10,15]
TEST_F(PadV3InfershapeTest, pad_v3_infershape_with_paddings_values_contiguous_true)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}}; // paddings shape (used for context metadata only, tensor will be null)
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;
    // Contiguous layout: [front0, back0, front1, back1, front2, back2]
    std::vector<int64_t> paddingsValues = {1, 2, 3, 4, 5, 6};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND}, // paddings tensor (metadata), will be set to null
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("_paddings_values",
                                             Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(paddingsValues))},
        {}, {}, {1});

    // Contiguous: front0=1, back0=2 => 2+1+2=5; front1=3, back1=4 => 3+3+4=10; front2=5, back2=6 => 4+5+6=15
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 10, 15}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 场景：paddings输入tensor为空（由_paddings_values属性提供padding值），paddings_contiguous=false（非连续排列），输入shape为[2,3,4]，paddings_values为[1,3,5,2,4,6]
// 期望：推导成功，输出shape为[5,10,15]
TEST_F(PadV3InfershapeTest, pad_v3_infershape_with_paddings_values_contiguous_false)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;
    // Non-contiguous layout: [front0, front1, front2, back0, back1, back2]
    std::vector<int64_t> paddingsValues = {1, 3, 5, 2, 4, 6};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("_paddings_values",
                                             Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(paddingsValues))},
        {}, {}, {1});

    // Non-contiguous: pad_front[0]=1, pad_back[0]=2 => 2+1+2=5
    // pad_front[1]=3, pad_back[1]=4 => 3+3+4=10
    // pad_front[2]=5, pad_back[2]=6 => 4+5+6=15
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 10, 15}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 场景：paddings输入tensor为空，输入shape为空（0维tensor），paddings_values数量与输入维度不匹配
// 期望：推导失败
TEST_F(PadV3InfershapeTest, pad_v3_infershape_with_paddings_values_empty_input)
{
    gert::StorageShape xShape = {{}, {{}}};
    gert::StorageShape padShape = {{2}, {2}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;
    std::vector<int64_t> paddingsValues = {1, 2};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("_paddings_values",
                                             Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(paddingsValues))},
        {}, {}, {1});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 场景：paddings输入tensor为空，paddings_contiguous=true，但paddings_values数量不足（只有3个值，需要6个才能匹配3维输入）
// 期望：推导失败
TEST_F(PadV3InfershapeTest, pad_v3_infershape_with_paddings_values_wrong_paddings_num)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{3}, {3}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;
    // Only 3 values, but need 6 (3 dims * PAIR=2)
    std::vector<int64_t> paddingsValues = {1, 2, 3};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("_paddings_values",
                                             Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(paddingsValues))},
        {}, {}, {1});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 场景：paddings输入tensor为空，paddings_contiguous=true，paddings_values包含负值导致输出维度为负值
// 期望：推导失败
TEST_F(PadV3InfershapeTest, pad_v3_infershape_with_paddings_values_negative_dim)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;
    // Contiguous layout with negative values causing dim_value < 0
    std::vector<int64_t> paddingsValues = {-5, -6, -5, -6, -5, -6};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("_paddings_values",
                                             Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(paddingsValues))},
        {}, {}, {1});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 场景：paddings输入tensor为空，paddings_contiguous=false，paddings_values包含负值导致输出维度为负值
// 期望：推导失败
TEST_F(PadV3InfershapeTest, pad_v3_infershape_with_paddings_values_negative_dim_non_contiguous)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{6}, {6}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;
    // Non-contiguous layout with negative values causing dim_value < 0
    std::vector<int64_t> paddingsValues = {-5, -5, -5, -6, -6, -6};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("_paddings_values",
                                             Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(paddingsValues))},
        {}, {}, {1});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 场景：paddings输入tensor为空，paddings_contiguous=false，paddings_values数量不足（只有4个值，需要6个才能匹配3维输入）
// 期望：推导失败
TEST_F(PadV3InfershapeTest, pad_v3_infershape_with_paddings_values_wrong_paddings_num_non_contiguous)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape padShape = {{4}, {4}};
    gert::StorageShape constantShape = {{1}, {1}};
    int32_t constantValue = 0;
    // Only 4 values, but need 6 (3 dims * PAIR=2)
    std::vector<int64_t> paddingsValues = {1, 3, 5, 2};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3",
        {{xShape, ge::DT_INT32, ge::FORMAT_ND},
         {padShape, ge::DT_INT32, ge::FORMAT_ND},
         {constantShape, ge::DT_INT32, ge::FORMAT_ND, true, &constantValue}},
        {{{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("constant")),
         gert::InfershapeContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("_paddings_values",
                                             Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(paddingsValues))},
        {}, {}, {1});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
