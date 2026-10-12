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

using namespace ge;

class EqualTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EqualTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EqualTest TearDown" << std::endl;
    }
};

TEST_F(EqualTest, equal_infershape_diff_test_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Equal",
        {
            {{{3, 4, 5, 6, 7}, {3, 4, 5, 6, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(EqualTest, equal_infershape_diff_test_5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Equal",
        {
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{10, -1}, {10, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(EqualTest, equal_infershape_diff_test_6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Equal",
        {
            {{{3, 4, 5, 6, -1}, {3, 4, 5, 6, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 4, 5, 6, 1}, {3, 4, 5, 6, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 5, 6, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(EqualTest, equal_infershape_diff_test_7)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Equal",
        {
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(EqualTest, equal_infershape_diff_test_8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Equal",
        {
            {{{-1, 2}, {-1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(EqualTest, equal_infershape_diff_test_9)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Equal",
        {
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{17, 2, 5, 1}, {17, 2, 5, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {17, 2, 5, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
