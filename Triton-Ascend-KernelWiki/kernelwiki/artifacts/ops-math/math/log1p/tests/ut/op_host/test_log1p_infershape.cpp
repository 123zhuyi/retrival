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

class Log1pTest_UT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Log1pTest_UT SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Log1pTest_UT TearDown" << std::endl;
    }
};

TEST_F(Log1pTest_UT, log1p_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Log1p",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Log1pTest_UT, log1p_infershape_test2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Log1p",
        {
            {{{5, -1}, {5, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Log1pTest_UT, log1p_infershape_test3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Log1p",
        {
            {{{1}, {1}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Log1pTest_UT, log1p_infershape_test4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Log1p",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Log1pTest_UT, log1p_infershape_test5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Log1p",
        {
            {{{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 2, 3, 4, 5, 6, 7, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Log1pTest_UT, log1p_infershape_test6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Log1p",
        {
            {{{4, 5, 6, 7, 8, 1, 5, 9}, {4, 5, 6, 7, 8, 1, 5, 9}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 5, 6, 7, 8, 1, 5, 9},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Log1pTest_UT, log1p_infershape_test7)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Log1p",
        {
            {{{3, 3, 3, 3, 3, 3, 3, 3}, {3, 3, 3, 3, 3, 3, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 3, 3, 3, 3, 3, 3, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Log1pTest_UT, log1p_infershape_test8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Log1p",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
