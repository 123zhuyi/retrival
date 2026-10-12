/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

class InvertPermutationInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InvertPermutationInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InvertPermutationInfershape TearDown" << std::endl;
    }
};

TEST_F(InvertPermutationInfershape, invert_permutation_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InvertPermutation",
        {
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InvertPermutationInfershape, invert_permutation_infershape_test2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InvertPermutation",
        {
            {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InvertPermutationInfershape, invert_permutation_infershape_test3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InvertPermutation",
        {
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InvertPermutationInfershape, invert_permutation_infershape_test4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InvertPermutation",
        {
            {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {100},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InvertPermutationInfershape, invert_permutation_infershape_test5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InvertPermutation",
        {
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
