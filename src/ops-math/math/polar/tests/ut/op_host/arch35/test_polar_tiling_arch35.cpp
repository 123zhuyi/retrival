/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/polar_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class PolarTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PolarTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PolarTilingTest TearDown" << std::endl; }
};

TEST_F(PolarTilingTest, polar_test_fp32_same_shape)
{
    optiling::PolarCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara("Polar",
                                              {
                                                  {{{16, 7, 14}, {16, 7, 14}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 7, 14}, {16, 7, 14}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 7, 14}, {16, 7, 14}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PolarTilingTest, polar_test_fp32_broadcast)
{
    optiling::PolarCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara("Polar",
                                              {
                                                  {{{15, 5}, {15, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 5}, {1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{15, 5}, {15, 5}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PolarTilingTest, polar_test_fp32_broadcast_multidim)
{
    optiling::PolarCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara(
        "Polar",
        {
            {{{1, 3, 1, 7, 1, 5, 1, 1}, {1, 3, 1, 7, 1, 5, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 3, 3, 7, 8, 5, 10, 1}, {3, 3, 3, 7, 8, 5, 10, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 3, 3, 7, 8, 5, 10, 1}, {3, 3, 3, 7, 8, 5, 10, 1}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PolarTilingTest, polar_test_fp32_scalar_broadcast)
{
    optiling::PolarCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara(
        "Polar",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PolarTilingTest, polar_test_fp32_broadcast_diff_shape)
{
    optiling::PolarCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara("Polar",
                                              {
                                                  {{{2, 3, 1, 4}, {2, 3, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 1, 5, 1}, {2, 1, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 3, 5, 4}, {2, 3, 5, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PolarTilingTest, polar_test_failed_dtype_mismatch_input)
{
    optiling::PolarCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara("Polar",
                                              {
                                                  {{{16, 7, 14}, {16, 7, 14}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{16, 7, 14}, {16, 7, 14}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 7, 14}, {16, 7, 14}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(PolarTilingTest, polar_test_failed_dtype_mismatch_output)
{
    optiling::PolarCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara("Polar",
                                              {
                                                  {{{16, 7, 14}, {16, 7, 14}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16, 7, 14}, {16, 7, 14}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 7, 14}, {16, 7, 14}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(PolarTilingTest, polar_test_failed_not_broadcastable)
{
    optiling::PolarCompileInfo compileInfo = {64};
    gert::TilingContextPara tilingContextPara("Polar",
                                              {
                                                  {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 5}, {3, 5}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}
