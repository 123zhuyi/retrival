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
 * \file test_squared_difference_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/squared_difference_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class SquaredDifferenceTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SquaredDifferenceTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SquaredDifferenceTiling TearDown" << std::endl;
    }
};

TEST_F(SquaredDifferenceTiling, squared_difference_test_0)
{
    optiling::SquaredDifferenceCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "SquaredDifference",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquaredDifferenceTiling, squared_difference_test_1)
{
    optiling::SquaredDifferenceCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "SquaredDifference",
        {
            {{{1, 9, 6, 1, 7, 10}, {1, 9, 6, 1, 7, 10}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{13, 1, 6, 2, 7, 1}, {13, 1, 6, 2, 7, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{13, 9, 6, 2, 7, 10}, {13, 9, 6, 2, 7, 10}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquaredDifferenceTiling, squared_difference_test_2)
{
    optiling::SquaredDifferenceCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "SquaredDifference",
        {
            {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquaredDifferenceTiling, squared_difference_test_3)
{
    optiling::SquaredDifferenceCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "SquaredDifference",
        {
            {{{20, 5, 1, 9, 17, 10, 3}, {20, 5, 1, 9, 17, 10, 3}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 3, 9, 1, 10, 1}, {1, 1, 3, 9, 1, 10, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{20, 5, 3, 9, 17, 10, 3}, {20, 5, 3, 9, 17, 10, 3}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquaredDifferenceTiling, squared_difference_test_4)
{
    optiling::SquaredDifferenceCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "SquaredDifference",
        {
            {{{15, 5}, {15, 5}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 5}, {1, 5}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{15, 5}, {15, 5}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquaredDifferenceTiling, squared_difference_test_5)
{
    optiling::SquaredDifferenceCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "SquaredDifference",
        {
            {{{1, 3, 1, 7, 1, 5, 1, 1}, {1, 3, 1, 7, 1, 5, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3, 3, 3, 7, 8, 5, 10, 1}, {3, 3, 3, 7, 8, 5, 10, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{3, 3, 3, 7, 8, 5, 10, 1}, {3, 3, 3, 7, 8, 5, 10, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquaredDifferenceTiling, squared_difference_test_6)
{
    optiling::SquaredDifferenceCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "SquaredDifference",
        {
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{15, 2, 1}, {15, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{15, 2, 1}, {15, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquaredDifferenceTiling, squared_difference_test_7)
{
    optiling::SquaredDifferenceCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "SquaredDifference",
        {
            {{{16, 5, 4, 6, 2, 2, 17}, {16, 5, 4, 6, 2, 2, 17}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 5, 1, 6, 1, 2, 1}, {16, 5, 1, 6, 1, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 5, 4, 6, 2, 2, 17}, {16, 5, 4, 6, 2, 2, 17}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquaredDifferenceTiling, squared_difference_test_8)
{
    optiling::SquaredDifferenceCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "SquaredDifference",
        {
            {{{14, 16}, {14, 16}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{14, 16}, {14, 16}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{14, 16}, {14, 16}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(SquaredDifferenceTiling, squared_difference_test_failed_different_dtype)
{
    optiling::SquaredDifferenceCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "SquaredDifference",
        {
            {{{4, 17, 1, 12}, {4, 17, 1, 12}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1, 12}, {1, 1, 1, 12}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{4, 17, 1, 12}, {4, 17, 1, 12}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(SquaredDifferenceTiling, squared_difference_test_failed_invalid_dtype)
{
    optiling::SquaredDifferenceCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "SquaredDifference",
        {
            {{{4, 17, 1, 12}, {4, 17, 1, 12}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1, 1, 1, 12}, {1, 1, 1, 12}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            {{{4, 17, 1, 12}, {4, 17, 1, 12}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}