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
 * \file test_SplitV_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include <vector>

using namespace std;

class SplitVInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SplitV SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SplitV TearDown" << std::endl;
    }
};

TEST_F(SplitVInfershape, split_v_infershape_test1)
{
    vector<int64_t> sizeSplitValue = {1};
    vector<int64_t> splitDimValue = {-1};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{1, 1, 160, 72, 1}, {1, 1, 160, 72, 1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeSplitValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, splitDimValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 160, 72, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SplitVInfershape, split_v_infershape_int32)
{
    vector<int64_t> sizeSplitValue = {4, 3, 3};
    vector<int64_t> splitDimValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{10, 8, 6}, {10, 8, 6}}, ge::DT_INT64, ge::FORMAT_ND},
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
            {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
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

TEST_F(SplitVInfershape, split_v_infershape_dynamic_value)
{
    vector<int64_t> sizeSplitValue = {4, -1, 2};
    vector<int64_t> splitDimValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{10, 8}, {10, 8}}, ge::DT_INT64, ge::FORMAT_ND},
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
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8}, {4, 8}, {2, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SplitVInfershape, split_v_infershape_3d)
{
    vector<int64_t> sizeSplitValue = {2, 2, 2};
    vector<int64_t> splitDimValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{4, 6, 8}, {4, 6, 8}}, ge::DT_INT64, ge::FORMAT_ND},
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
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2, 8}, {4, 2, 8}, {4, 2, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SplitVInfershape, split_v_infershape_unknown_rank)
{
    vector<int64_t> sizeSplitValue = {4, 4};
    vector<int64_t> splitDimValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_INT64, ge::FORMAT_ND},
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
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, -1}, {4, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SplitVInfershape, split_v_infershape_split_dim_not_const)
{
    vector<int64_t> sizeSplitValue = {4, 4};
    vector<int64_t> splitDimValue = {-1};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeSplitValue.data()},
            {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND, true, splitDimValue.data()},
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

TEST_F(SplitVInfershape, split_v_infershape_dynamic_dim_with_sum)
{
    vector<int64_t> sizeSplitValue = {3, -1, 2};
    vector<int64_t> splitDimValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "SplitV",
        {
            {{{10, 8}, {10, 8}}, ge::DT_INT64, ge::FORMAT_ND},
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