/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_range_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class RangeInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RangeInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RangeInfershape TearDown" << std::endl;
    }
};

// Test: range infershape with int32, positive step
TEST_F(RangeInfershape, range_infershape_int32_positive_step)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Range",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, {}},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: range infershape with int32, negative step
TEST_F(RangeInfershape, range_infershape_int32_negative_step)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Range",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, {}},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: range infershape with float32
TEST_F(RangeInfershape, range_infershape_float32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Range",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, {}},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: range infershape with float16
TEST_F(RangeInfershape, range_infershape_float16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Range",
        {
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND, {}},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: range infershape with bfloat16
TEST_F(RangeInfershape, range_infershape_bfloat16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Range",
        {
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND, {}},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: range infershape with int64
TEST_F(RangeInfershape, range_infershape_int64)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Range",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, {}},
        },
        {
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: range infershape with double
TEST_F(RangeInfershape, range_infershape_double)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Range",
        {
            {{{1}, {1}}, ge::DT_DOUBLE, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_DOUBLE, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_DOUBLE, ge::FORMAT_ND, {}},
        },
        {
            {{{}, {}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RangeInfershape, range_infershape_test0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Range",
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND, {}},
        },
        {
            {{{7}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RangeInfershape, range_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Range",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, {}},
        },
        {
            {{{7,}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(RangeInfershape, range_infershape_test2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Range",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, {}},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, {}},
        },
        {
            {{{7,}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}