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

class is_nan : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "is_nan SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "is_nan TearDown" << std::endl;
    }
};

TEST_F(is_nan, is_nan_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IsNan",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(is_nan, is_nan_infershape_test2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IsNan",
        {
            {{{5, -1}, {5, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(is_nan, is_nan_infershape_test3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IsNan",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(is_nan, is_nan_infershape_test4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "IsNan",
        {
            {{{4, 5, 6, 7, 8, 1, 5, 9}, {4, 5, 6, 7, 8, 1, 5, 9}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 5, 6, 7, 8, 1, 5, 9},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
