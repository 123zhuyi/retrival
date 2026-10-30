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

class FusedMulAddInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedMulAddInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedMulAddInfershape TearDown" << std::endl;
    }
};

// Case 1: three inputs share the same shape, fp16.
TEST_F(FusedMulAddInfershape, same_shape_fp16)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 2: x3 is a scalar [1].
TEST_F(FusedMulAddInfershape, x3_scalar)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 3: per-axis broadcast x1=[16,1], x2=[1,16], x3=[16,16] -> [16,16].
TEST_F(FusedMulAddInfershape, axis_broadcast_fp32)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 4: cross-rank broadcast x1=[3,4,5], x2=[4,5], x3=[5] -> [3,4,5].
TEST_F(FusedMulAddInfershape, cross_rank_broadcast_fp32)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{3, 4, 5}, {3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 5: dynamic shape -1.
TEST_F(FusedMulAddInfershape, dynamic_shape_fp16)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 6: dynamic rank -2.
TEST_F(FusedMulAddInfershape, dynamic_rank)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 7: 1D scalar-like inputs, fp32.
TEST_F(FusedMulAddInfershape, vector_1d_fp32)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 8: int32 vector path.
TEST_F(FusedMulAddInfershape, vector_1d_int32)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 9: x1/x2 are scalars, x3 carries the full shape -> output takes x3's shape.
TEST_F(FusedMulAddInfershape, scalar_mul_full_add_fp32)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 10: row x column mutual broadcast x1=[1,5], x2=[5,1], x3=[1] -> [5,5], fp32.
TEST_F(FusedMulAddInfershape, row_col_broadcast_fp32)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{1, 5}, {1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 1}, {5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 11: 3-D mutual broadcast x1=[2,1,4], x2=[1,3,4], x3=[4] -> [2,3,4], fp16.
TEST_F(FusedMulAddInfershape, mutual_3d_broadcast_fp16)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{2, 1, 4}, {2, 1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 3, 4}, {1, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 12: every input broadcasts on a different axis
// x1=[4,1,1], x2=[1,3,1], x3=[1,1,5] -> [4,3,5], int32.
TEST_F(FusedMulAddInfershape, all_three_broadcast_int32)
{
    gert::InfershapeContextPara para(
        "FusedMulAdd",
        {
            {{{4, 1, 1}, {4, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 3, 1}, {1, 3, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1, 5}, {1, 1, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
