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
 * \file test_stateless_random_infershape.cpp
 * \brief InferShape unit test for StatelessRandom operator
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace std;

class StatelessRandomInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StatelessRandom SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StatelessRandom TearDown" << std::endl;
    }
};

TEST_F(StatelessRandomInfershape, stateless_random_infershape_test1)
{
    vector<int64_t> shapeValue = {4};
    vector<int64_t> seedValue = {42};
    vector<int64_t> offsetValue = {0};
    vector<int64_t> fromValue = {0};
    vector<int64_t> toValue = {100};

    gert::InfershapeContextPara infershapeContextPara(
        "StatelessRandom",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, fromValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, toValue.data()},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StatelessRandomInfershape, stateless_random_infershape_test2)
{
    vector<int64_t> shapeValue = {16};
    vector<int64_t> seedValue = {123};
    vector<int64_t> offsetValue = {8};
    vector<int64_t> fromValue = {10};
    vector<int64_t> toValue = {50};

    gert::InfershapeContextPara infershapeContextPara(
        "StatelessRandom",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, fromValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, toValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StatelessRandomInfershape, stateless_random_infershape_test3)
{
    vector<int32_t> shapeValue = {8};
    vector<int64_t> seedValue = {7};
    vector<int64_t> offsetValue = {0};
    vector<int64_t> fromValue = {0};
    vector<int64_t> toValue = {2};

    gert::InfershapeContextPara infershapeContextPara(
        "StatelessRandom",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, fromValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, toValue.data()},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}