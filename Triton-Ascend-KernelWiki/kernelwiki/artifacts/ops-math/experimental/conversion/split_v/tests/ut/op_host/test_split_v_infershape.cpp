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
 * \file test_split_v_infershape.cpp
 * \brief SplitV infershape UT.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

using namespace std;

class SplitVInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SplitVInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SplitVInfershape TearDown" << std::endl; }
};

TEST_F(SplitVInfershape, split_v_infershape_single_output_int64)
{
    vector<int64_t> sizeSplitValue = {72};
    vector<int64_t> splitDimValue = {-1};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{1, 1, 160, 72}, {1, 1, 160, 72}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeSplitValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, splitDimValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 160, 72}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SplitVInfershape, split_v_infershape_multi_output_int32)
{
    vector<int32_t> sizeSplitValue = {4, 3, 3};
    vector<int64_t> splitDimValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{10, 8, 6}, {10, 8, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeSplitValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, splitDimValue.data()},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8, 6}, {3, 8, 6}, {3, 8, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SplitVInfershape, split_v_infershape_negative_dim)
{
    vector<int64_t> sizeSplitValue = {4, 4};
    vector<int64_t> splitDimValue = {-1};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeSplitValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, splitDimValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 4}, {8, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SplitVInfershape, split_v_infershape_dynamic_split_value)
{
    vector<int64_t> sizeSplitValue = {3, -1, 2};
    vector<int64_t> splitDimValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{10, 8}, {10, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeSplitValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, splitDimValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 8}, {5, 8}, {2, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SplitVInfershape, split_v_infershape_unknown_split_dim_const)
{
    vector<int64_t> sizeSplitValue = {4, -1};
    vector<int64_t> splitDimValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{-1, 8}, {-1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeSplitValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, splitDimValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8}, {-1, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SplitVInfershape, split_v_infershape_invalid_dim)
{
    vector<int64_t> sizeSplitValue = {4, 4};
    vector<int64_t> splitDimValue = {2};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeSplitValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, splitDimValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
