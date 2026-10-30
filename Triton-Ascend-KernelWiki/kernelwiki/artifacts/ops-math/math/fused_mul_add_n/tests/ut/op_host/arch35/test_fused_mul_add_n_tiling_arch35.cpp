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
 * /file test_fused_mul_add_n_tiling_arch35.cpp
 * /brief
 */

#include "../../../../op_host/arch35/fused_mul_add_n_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class FusedMulAddNTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedMulAddNTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedMulAddNTilingTest TearDown" << std::endl;
    }
};

TEST_F(FusedMulAddNTilingTest, fused_mul_add_n_test_0)
{
    optiling::FusedMulAddNCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FusedMulAddN",
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100000000000100;
    string expectTilingData = "2048 1024 10880 1 1 1024 1024 10880 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FusedMulAddNTilingTest, fused_mul_add_n_test_1)
{
    optiling::FusedMulAddNCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FusedMulAddN",
        {
            {{{12, 13, 9, 10, 12, 6}, {12, 13, 9, 10, 12, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12, 13, 9, 10, 12, 6}, {12, 13, 9, 10, 12, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{12, 13, 9, 10, 12, 6}, {12, 13, 9, 10, 12, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 200000000000100;
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(FusedMulAddNTilingTest, fused_mul_add_n_test_3)
{
    optiling::FusedMulAddNCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FusedMulAddN",
        {
            {{{4, 9, 1}, {4, 9, 1}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{4, 9, 1}, {4, 9, 1}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            {{{4, 9, 1}, {4, 9, 1}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 400000000000100;
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(FusedMulAddNTilingTest, fused_mul_add_n_test_4)
{
    optiling::FusedMulAddNCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FusedMulAddN",
        {
            {{{15, 6, 2}, {15, 6, 2}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{15, 6, 2}, {15, 6, 2}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{15, 6, 2}, {15, 6, 2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 500000000000100;
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(FusedMulAddNTilingTest, fused_mul_add_n_test_5)
{
    optiling::FusedMulAddNCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FusedMulAddN",
        {
            {{{0, 16, 7, 17}, {0, 16, 7, 17}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 16, 7, 17}, {0, 16, 7, 17}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{15, 16, 7, 17}, {15, 16, 7, 17}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedMulAddNTilingTest, fused_mul_add_n_test_6)
{
    optiling::FusedMulAddNCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FusedMulAddN",
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 1, 4, 4}, {16, 1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedMulAddNTilingTest, fused_mul_add_n_test_7)
{
    optiling::FusedMulAddNCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FusedMulAddN",
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2}, {1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedMulAddNTilingTest, fused_mul_add_n_test_8)
{
    optiling::FusedMulAddNCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FusedMulAddN",
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FusedMulAddNTilingTest, fused_mul_add_n_test_9)
{
    optiling::FusedMulAddNCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FusedMulAddN",
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 1, 5, 4, 8}, {16, 1, 5, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}