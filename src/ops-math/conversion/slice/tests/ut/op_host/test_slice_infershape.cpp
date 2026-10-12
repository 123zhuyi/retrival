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
 * \file test_slice_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include <vector>

using namespace std;

class SliceInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Slice SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Slice TearDown" << std::endl;
    }
};

TEST_F(SliceInfershape, slice_infershape_test1)
{
    vector<int64_t> offsetsValue = {7};
    vector<int64_t> sizeValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "Slice",
        {
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetsValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceInfershape, slice_infershape_multi_dim)
{
    vector<int64_t> offsetsValue = {1, 2, 3};
    vector<int64_t> sizeValue = {2, 3, 4};
    gert::InfershapeContextPara infershapeContextPara(
        "Slice",
        {
            {{{8, 9, 10}, {8, 9, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetsValue.data()},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceInfershape, slice_infershape_size_minus_one)
{
    vector<int64_t> offsetsValue = {1, 2};
    vector<int64_t> sizeValue = {-1, 3};
    gert::InfershapeContextPara infershapeContextPara(
        "Slice",
        {
            {{{8, 9}, {8, 9}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetsValue.data()},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{7, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceInfershape, slice_infershape_dynamic_offset)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Slice",
        {
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, false, nullptr},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceInfershape, slice_infershape_dynamic_size)
{
    vector<int64_t> offsetsValue = {7};
    gert::InfershapeContextPara infershapeContextPara(
        "Slice",
        {
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetsValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, false, nullptr},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceInfershape, slice_infershape_2d)
{
    vector<int64_t> offsetsValue = {2, 3};
    vector<int64_t> sizeValue = {4, 5};
    gert::InfershapeContextPara infershapeContextPara(
        "Slice",
        {
            {{{10, 12}, {10, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetsValue.data()},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceInfershape, slice_infershape_4d)
{
    vector<int64_t> offsetsValue = {1, 2, 3, 4};
    vector<int64_t> sizeValue = {2, 3, 4, 5};
    gert::InfershapeContextPara infershapeContextPara(
        "Slice",
        {
            {{{8, 9, 10, 12}, {8, 9, 10, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetsValue.data()},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
