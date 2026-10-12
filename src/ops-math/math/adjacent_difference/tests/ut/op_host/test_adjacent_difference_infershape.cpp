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
 * \file test_adjacent_difference_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class AdjacentDifferenceInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdjacentDifferenceInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdjacentDifferenceInfershape TearDown" << std::endl;
    }
};

// Test: adjacent_difference infershape with float32
TEST_F(AdjacentDifferenceInfershape, adjacent_difference_infershape_float32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdjacentDifference",
        {
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: adjacent_difference infershape with float16
TEST_F(AdjacentDifferenceInfershape, adjacent_difference_infershape_float16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdjacentDifference",
        {
            {{{10, 20}, {10, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10, 20}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: adjacent_difference infershape with int32
TEST_F(AdjacentDifferenceInfershape, adjacent_difference_infershape_int32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdjacentDifference",
        {
            {{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{100}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: adjacent_difference infershape with int64
TEST_F(AdjacentDifferenceInfershape, adjacent_difference_infershape_int64)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdjacentDifference",
        {
            {{{50, 50}, {50, 50}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{50, 50}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: adjacent_difference infershape with bfloat16
TEST_F(AdjacentDifferenceInfershape, adjacent_difference_infershape_bfloat16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdjacentDifference",
        {
            {{{8, 16}, {8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: adjacent_difference infershape with int8
TEST_F(AdjacentDifferenceInfershape, adjacent_difference_infershape_int8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdjacentDifference",
        {
            {{{32, 64}, {32, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: adjacent_difference infershape with uint8
TEST_F(AdjacentDifferenceInfershape, adjacent_difference_infershape_uint8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdjacentDifference",
        {
            {{{32, 64}, {32, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: adjacent_difference infershape with 3D tensor
TEST_F(AdjacentDifferenceInfershape, adjacent_difference_infershape_3d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdjacentDifference",
        {
            {{{4, 5, 6}, {4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 5, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: adjacent_difference infershape with small 1D tensor
TEST_F(AdjacentDifferenceInfershape, adjacent_difference_infershape_small_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdjacentDifference",
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: adjacent_difference infershape with NHWC format
TEST_F(AdjacentDifferenceInfershape, adjacent_difference_infershape_nhwc)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdjacentDifference",
        {
            {{{1, 16, 16, 3}, {1, 16, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 16, 16, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
