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

using namespace ge;

class MulNoNanInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MulNoNanInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MulNoNanInfershape TearDown" << std::endl;
    }
};

// Case 1: two inputs share the same shape, fp16.
TEST_F(MulNoNanInfershape, same_shape_fp16)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 2: x2 is a scalar [1] -> output follows x1's shape.
TEST_F(MulNoNanInfershape, x2_scalar)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 3: x1 is a scalar [1] -> output follows x2's shape.
TEST_F(MulNoNanInfershape, x1_scalar)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 4: per-axis broadcast x1=[16,1], x2=[1,16] -> [16,16], fp16.
TEST_F(MulNoNanInfershape, axis_broadcast_fp16)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{16, 1}, {16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 5: cross-rank broadcast x1=[3,4,5], x2=[4,5] -> [3,4,5].
TEST_F(MulNoNanInfershape, cross_rank_broadcast_fp32)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{3, 4, 5}, {3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 6: dynamic shape -1, bf16.
TEST_F(MulNoNanInfershape, dynamic_shape_bf16)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 7: dynamic rank -2.
TEST_F(MulNoNanInfershape, dynamic_rank)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 8: int32 vector path.
TEST_F(MulNoNanInfershape, vector_1d_int32)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 9: row x column mutual broadcast x1=[1,5], x2=[5,1] -> [5,5], fp32.
TEST_F(MulNoNanInfershape, row_col_broadcast_fp32)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{1, 5}, {1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 1}, {5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 10: trailing column-vector broadcast x1=[4,1], x2=[4,6] -> [4,6], fp32.
TEST_F(MulNoNanInfershape, col_vector_broadcast_fp32)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{4, 1}, {4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 6}, {4, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 6}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 11: 3-D mutual broadcast x1=[2,1,4], x2=[1,3,4] -> [2,3,4], fp16.
TEST_F(MulNoNanInfershape, mutual_3d_broadcast_fp16)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{2, 1, 4}, {2, 1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 3, 4}, {1, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 12: cross-rank where x2 is the higher-rank input x1=[5], x2=[3,4,5] -> [3,4,5], int32.
TEST_F(MulNoNanInfershape, cross_rank_x2_bigger_int32)
{
    gert::InfershapeContextPara para(
        "MulNoNan",
        {
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 4, 5}, {3, 4, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
