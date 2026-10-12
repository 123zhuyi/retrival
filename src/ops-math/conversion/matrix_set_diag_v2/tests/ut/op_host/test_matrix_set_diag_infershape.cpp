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

class MatrixSetDiagV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MatrixSetDiagV2Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MatrixSetDiagV2Infershape TearDown" << std::endl; }
};

TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_test1)
{
    std::vector<int32_t> kValues = {0};
    gert::InfershapeContextPara infershapeContextPara("MatrixSetDiagV2",
                                                      {
                                                          {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_test2)
{
    std::vector<int32_t> kValues = {0, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "MatrixSetDiagV2",
        {
            {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2, 4}, {3, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_test3)
{
    std::vector<int32_t> kValues = {-1, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "MatrixSetDiagV2",
        {
            {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 3, 4}, {3, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_test4)
{
    std::vector<int32_t> kValues = {5};
    gert::InfershapeContextPara infershapeContextPara(
        "MatrixSetDiagV2",
        {
            {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 3, 4}, {3, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_test5)
{
    std::vector<int32_t> kValues = {-5, 4};
    gert::InfershapeContextPara infershapeContextPara(
        "MatrixSetDiagV2",
        {
            {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 3, 4}, {3, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_test6)
{
    std::vector<int32_t> kValues = {-1, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "MatrixSetDiagV2",
        {
            {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 3, 4}, {3, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_test7)
{
    std::vector<int32_t> kValues = {2, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "MatrixSetDiagV2",
        {
            {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 3, 4}, {3, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_test8)
{
    std::vector<int32_t> kValues = {1, 2};
    gert::InfershapeContextPara infershapeContextPara("MatrixSetDiagV2",
                                                      {
                                                          {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 3, 4}, {3, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_test9)
{
    std::vector<int32_t> kValues = {7, 7};
    gert::InfershapeContextPara infershapeContextPara(
        "MatrixSetDiagV2",
        {
            {{{83, 192, 3, 8}, {83, 192, 3, 8}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{83, 192, 3}, {83, 192, 3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_test10)
{
    std::vector<int32_t> kValues = {4, 4};
    gert::InfershapeContextPara infershapeContextPara(
        "MatrixSetDiagV2",
        {
            {{{1252, 3, 4}, {1252, 3, 4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1252, 3}, {1252, 3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_test11)
{
    std::vector<int32_t> kValues = {4};
    gert::InfershapeContextPara infershapeContextPara("MatrixSetDiagV2",
                                                      {
                                                          {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 场景：k 为非常量张量（不携带数据），diagonal 维度(1) 小于 input 维度(3) - 1，即 3-1=2 > 1。
// 期望：CheckShape 中非 const k 分支的 xDimNum_-1 > diagDimNum_ 校验失败，返回 GRAPH_FAILED。
TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_nonconst_k_diag_dim_too_small)
{
    gert::InfershapeContextPara infershapeContextPara("MatrixSetDiagV2",
                                                      {
                                                          {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 场景：k 为非常量张量（不携带数据），diagonal 维度(4) 大于 input 维度(3)。
// 期望：CheckShape 中非 const k 分支的 diagDimNum_ > xDimNum_ 校验失败，返回 GRAPH_FAILED。
TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_nonconst_k_diag_dim_too_large)
{
    gert::InfershapeContextPara infershapeContextPara("MatrixSetDiagV2",
                                                      {
                                                          {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 4, 4, 4}, {3, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 场景：k 为非常量张量（不携带数据），diagonal 维度(2) 介于 input 维度(3)-1 与 input 维度(3) 之间，
// 且 diagonal 前导维(3) 与 input 前导维(3) 相等。
// 期望：CheckShape 非 const k 分支校验通过并成功返回，输出 shape 为 {3,4,4}。
TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_nonconst_k_success)
{
    gert::InfershapeContextPara infershapeContextPara("MatrixSetDiagV2",
                                                      {
                                                          {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 场景：k 为常量 {0} 且 const 张量校验通过；diagonal 前导维为 UNKNOWN_DIM(-1)，其余维度合法。
// 期望：SetOutputShape 循环中遇到 diag 未知前导维时 continue，输出 shape 保持 {3,4,4}。
TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_diag_leading_dim_unknown)
{
    std::vector<int32_t> kValues = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "MatrixSetDiagV2",
        {
            {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, 4}, {-1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 场景：k 为常量 {0} 且 const 张量校验通过；input 前导维为 UNKNOWN_DIM(-1)，diagonal 前导维已知为 3。
// 期望：SetOutputShape 循环中当 x 前导维未知时用 diagonal 前导维填充，输出 shape 为 {3,4,4}。
TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_x_leading_dim_unknown_fill_diag)
{
    std::vector<int32_t> kValues = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "MatrixSetDiagV2",
        {
            {{{-1, 4, 4}, {-1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, kValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 场景：input 为未知 rank（{-2}），diagonal/k 正常。
// 期望：Inference 检测到 x 未知 rank 后直接 SetUnknownRank(y)，成功返回，输出 shape 为 {-2}。
TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_x_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("MatrixSetDiagV2",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 场景：diagonal 为未知 rank（{-2}），input/k 正常。
// 期望：Inference 检测到 diag 未知 rank 后直接 SetUnknownRank(y)，成功返回，输出 shape 为 {-2}。
TEST_F(MatrixSetDiagV2Infershape, matrix_set_diag_infershape_diag_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("MatrixSetDiagV2",
                                                      {
                                                          {{{3, 4, 4}, {3, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
