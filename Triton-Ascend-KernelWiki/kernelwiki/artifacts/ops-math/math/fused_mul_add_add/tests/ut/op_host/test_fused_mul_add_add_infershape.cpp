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

class FusedMulAddAddInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedMulAddAddInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedMulAddAddInfershape TearDown" << std::endl;
    }
};

// Case 1: four inputs share the same shape, fp16.
TEST_F(FusedMulAddAddInfershape, same_shape_fp16)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
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

// Case 2: x4 is a scalar [1] (residual scalar), fp32.
TEST_F(FusedMulAddAddInfershape, x4_scalar_fp32)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
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

// Case 3: output equals x1's shape (no broadcast): x1=[16,1] -> [16,1].
TEST_F(FusedMulAddAddInfershape, axis_broadcast_fp32)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 4: cross-rank broadcast: x1=[3,4,5], x2=[4,5], x3=[5], x4=[1] -> [3,4,5].
TEST_F(FusedMulAddAddInfershape, cross_rank_broadcast_int32)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{3, 4, 5}, {3, 4, 5}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 5: dynamic shape -1 is not supported -> GRAPH_FAILED.
TEST_F(FusedMulAddAddInfershape, dynamic_shape_fp16)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

// Case 6: dynamic rank -2 is not supported -> GRAPH_FAILED.
TEST_F(FusedMulAddAddInfershape, dynamic_rank)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(para, ge::GRAPH_FAILED, expectOutputShape);
}

// Case 7: 1D vectors, fp32.
TEST_F(FusedMulAddAddInfershape, vector_1d_fp32)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
TEST_F(FusedMulAddAddInfershape, vector_1d_int32)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
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

// Case 9: output equals x1's shape (no broadcast): x1=[1] -> [1].
TEST_F(FusedMulAddAddInfershape, scalar_mul_full_add_fp32)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 10: output equals x1's shape (no broadcast): x1=[1,5] -> [1,5].
TEST_F(FusedMulAddAddInfershape, row_col_broadcast_fp32)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{1, 5}, {1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 1}, {5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 5}, {5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 11: output equals x1's shape (no broadcast): x1=[2,1,4] -> [2,1,4], fp16.
TEST_F(FusedMulAddAddInfershape, mutual_3d_broadcast_fp16)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{2, 1, 4}, {2, 1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 3, 4}, {1, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 12: output equals x1's shape (no broadcast): x1=[4,1,1] -> [4,1,1], int32.
TEST_F(FusedMulAddAddInfershape, all_four_broadcast_int32)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{4, 1, 1}, {4, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 3, 1}, {1, 3, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 1, 5}, {1, 1, 5}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 1, 1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Case 13: output equals x1's shape (no broadcast): x1=[4,1] -> [4,1], fp32.
TEST_F(FusedMulAddAddInfershape, col_vector_broadcast_fp32)
{
    gert::InfershapeContextPara para(
        "FusedMulAddAdd",
        {
            {{{4, 1}, {4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 6}, {4, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
