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
 * \file test_lerp_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/lerp_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class LerpTiling : public testing::Test {
   protected:
    static void SetUpTestCase()
    {
        std::cout << "LerpTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LerpTiling TearDown" << std::endl;
    }
};

TEST_F(LerpTiling, lerp_test_0)
{
    optiling::LerpCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Lerp",
                                              {
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 1, 1, 4, 1}, {1, 1, 1, 4, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 100000001000100;
    string expectTilingData = "1 512 1 64 1 3 0 1 16384 1 1 1 0 0 0 0 0 1 4 1 0 0 0 0 0 64 4 8 0 0 0 0 0 64 4 8 0 0 0 0 0 0 "
                        "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 32 8 1 0 0 0 0 0 32 8 1 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LerpTiling, lerp_test_1)
{
    optiling::LerpCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Lerp",
                                              {
                                                {{{9, 1, 1, 7, 1, 9, 1, 1}, {9, 1, 1, 7, 1, 9, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 8, 1, 1, 1, 9, 6, 1}, {1, 8, 1, 1, 1, 9, 6, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{9, 8, 7, 7, 2, 1, 6, 2}, {9, 8, 7, 7, 2, 1, 6, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{9, 8, 7, 7, 2, 9, 6, 2}, {9, 8, 7, 7, 2, 9, 6, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
                                              uint64_t expectTilingKey = 100000001001100;
    string expectTilingData = "2 1 8 1 2 8 1 72 16384 9 1 1 7 1 9 1 1 1 8 1 1 1 9 6 1 9 8 7 7 2 1 6 2 9 8 7 7 2 9 6 2 63 0 0 "
                        "9 0 1 0 0 0 54 0 0 0 6 1 0 9408 1176 168 24 12 0 2 1 84672 10584 1512 216 108 12 2 1 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LerpTiling, lerp_test_2)
{
    optiling::LerpCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Lerp",
                                              {
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{20, 14, 16, 20, 11}, {20, 14, 16, 20, 11}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 14, 1, 1, 11}, {1, 14, 1, 1, 11}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{20, 14, 16, 20, 11}, {20, 14, 16, 20, 11}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
                                              uint64_t expectTilingKey = 200000001000100;
    string expectTilingData = "3 2 7 2 2 4 1 140 8192 1 1 1 1 0 0 0 0 20 14 320 11 0 0 0 0 1 14 1 11 0 0 0 0 20 14 320 11 0 "
                        "0 0 0 0 0 0 0 0 0 0 0 49280 3520 11 1 0 0 0 0 0 11 0 1 0 0 0 0 49280 3520 11 1 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LerpTiling, lerp_test_3)
{
    optiling::LerpCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Lerp",
                                              {
                                                {{{1, 13, 2, 1, 1, 2}, {1, 13, 2, 1, 1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{17, 13, 2, 6, 16, 2}, {17, 13, 2, 6, 16, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 13, 1, 1, 16, 1}, {1, 13, 1, 1, 16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{17, 13, 2, 6, 16, 2}, {17, 13, 2, 6, 16, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
                                              uint64_t expectTilingKey = 200000001001100;
    string expectTilingData = "1 1 17 1 1 6 0 17 8192 1 13 2 1 1 2 0 0 17 13 2 6 16 2 0 0 1 13 1 1 16 1 0 0 17 13 2 6 16 2 0 "
                        "0 0 4 2 0 0 1 0 0 4992 384 192 32 2 1 0 0 0 16 0 0 1 0 0 0 4992 384 192 32 2 1 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LerpTiling, lerp_test_4)
{
    optiling::LerpCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Lerp",
                                              {
                                                {{{7}, {7}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{7}, {7}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{7}, {7}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
                                              uint64_t expectTilingKey = 300000001000100;
    string expectTilingData = "1 16384 1 7 1 1 0 1 16384 7 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 7 0 0 0 0 0 0 0 7 0 0 0 0 0 0 0 1 0 "
                        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LerpTiling, lerp_test_5)
{
    optiling::LerpCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Lerp",
                                              {
                                                {{{1, 9, 6, 1, 7, 10}, {1, 9, 6, 1, 7, 10}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1, 1, 6, 2, 7, 1}, {1, 1, 6, 2, 7, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{13, 9, 6, 2, 7, 10}, {13, 9, 6, 2, 7, 10}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{13, 9, 6, 2, 7, 10}, {13, 9, 6, 2, 7, 10}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
                                              uint64_t expectTilingKey = 300000001001100;
    string expectTilingData = "1 2 7 1 1 6 0 7 16384 1 9 6 1 7 10 0 0 1 1 6 2 7 1 0 0 13 9 6 2 7 10 0 0 13 9 6 2 7 10 0 0 "
                        "0 420 70 0 10 1 0 0 0 0 14 7 1 0 0 0 7560 840 140 70 10 1 0 0 7560 840 140 70 10 1 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LerpTiling, lerp_test_6)
{
    optiling::LerpCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Lerp",
                                              {
                                                {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
                                              uint64_t expectTilingKey = 400000001000100;
    string expectTilingData = "1 12288 8 2688 1 1 0 8 12288 88704 0 0 0 0 0 0 0 88704 0 0 0 0 0 0 0 88704 0 0 0 0 0 0 0 "
                        "88704 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LerpTiling, lerp_test_7)
{
    optiling::LerpCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Lerp",
                                              {
                                                {{{1, 3, 1, 7, 1, 5, 1, 1}, {1, 3, 1, 7, 1, 5, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{3, 3, 3, 7, 8, 5, 10, 1}, {3, 3, 3, 7, 8, 5, 10, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{3, 3, 3, 7, 8, 5, 10, 1}, {3, 3, 3, 7, 8, 5, 10, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{3, 3, 3, 7, 8, 5, 10, 1}, {3, 3, 3, 7, 8, 5, 10, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
                                              uint64_t expectTilingKey = 400000001001100;
    string expectTilingData = "1 1 3 1 1 7 1 9 12288 1 3 1 7 1 5 1 0 3 3 3 7 8 5 10 0 3 3 3 7 8 5 10 0 3 3 3 7 8 5 10 0 0 35 0 5 0 "
                  "1 0 0 25200 8400 2800 400 50 10 1 0 25200 8400 2800 400 50 10 1 0 25200 8400 2800 400 50 10 1 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LerpTiling, lerp_test_8)
{
    optiling::LerpCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Lerp",
                                              {
                                                {{{1, 1, 1}, {1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{15, 2, 1}, {15, 2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{15, 2, 1}, {15, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{15, 2, 1}, {15, 2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
                                              uint64_t expectTilingKey = 500000001000100;
    string expectTilingData = "1 12288 1 30 1 1 0 1 12288 1 0 0 0 0 0 0 0 30 0 0 0 0 0 0 0 30 0 0 0 0 0 0 0 30 0 0 0 0 0 0 0 "
                        "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LerpTiling, lerp_test_9)
{
    optiling::LerpCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Lerp",
                                              {
                                                {{{15, 1, 2, 1, 1, 15}, {15, 1, 2, 1, 1, 15}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{15, 3, 2, 2, 14, 15}, {15, 3, 2, 2, 14, 15}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 1, 1, 2, 1, 15}, {1, 1, 1, 2, 1, 15}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{15, 3, 2, 2, 14, 15}, {15, 3, 2, 2, 14, 15}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
                                              uint64_t expectTilingKey = 500000001001100;
    string expectTilingData = "1 4 4 3 1 6 0 4 12288 15 1 2 1 1 15 0 0 15 3 2 2 14 15 0 0 1 1 1 2 1 15 0 0 15 3 2 2 14 15 0 "
                        "0 30 0 15 0 0 1 0 0 2520 840 420 210 15 1 0 0 0 0 0 15 0 1 0 0 2520 840 420 210 15 1 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LerpTiling, lerp_test_10)
{
    optiling::LerpCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Lerp",
                                              {
                                                {{{0, 1, 14, 1, 1}, {0, 1, 14, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{0, 1, 1, 1, 1}, {0, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{0, 9, 14, 7, 1}, {0, 9, 14, 7, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{7, 9, 14, 7, 1}, {7, 9, 14, 7, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
                                              uint64_t expectTilingKey = 107133664305920;
    string expectTilingData = "1 4 4 3 1 6 0 4 12288 15 1 2 1 1 15 0 0 15 3 2 2 14 15 0 0 1 1 1 2 1 15 0 0 15 3 2 2 14 15 0 "
                        "0 30 0 15 0 0 1 0 0 2520 840 420 210 15 1 0 0 0 0 0 15 0 1 0 0 2520 840 420 210 15 1 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
