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
 * \file test_not_equal_tiling.cpp
 * \brief
 */

#include "../../../../op_host/arch35/not_equal_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class NotEqualTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NotEqualTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NotEqualTilingTest TearDown" << std::endl;
    }
};

TEST_F(NotEqualTilingTest, not_equal_test_0)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_1)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{1, 9, 6, 1, 7, 10}, {1, 9, 6, 1, 7, 10}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{13, 1, 6, 2, 7, 1}, {13, 1, 6, 2, 7, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{13, 9, 6, 2, 7, 10}, {13, 9, 6, 2, 7, 10}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_2)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_3)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{20, 5, 1, 9, 17, 10, 3}, {20, 5, 1, 9, 17, 10, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 3, 9, 1, 10, 1}, {1, 1, 3, 9, 1, 10, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{20, 5, 3, 9, 17, 10, 3}, {20, 5, 3, 9, 17, 10, 3}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_4)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{15, 5}, {15, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 5}, {1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{15, 5}, {15, 5}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_5)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{1, 3, 1, 7, 1, 5, 1, 1}, {1, 3, 1, 7, 1, 5, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 3, 3, 7, 8, 5, 10, 1}, {3, 3, 3, 7, 8, 5, 10, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 3, 3, 7, 8, 5, 10, 1}, {3, 3, 3, 7, 8, 5, 10, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_6)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{1, 1, 1}, {1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{15, 2, 1}, {15, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{15, 2, 1}, {15, 2, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_7)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{16, 5, 4, 6, 2, 2, 17}, {16, 5, 4, 6, 2, 2, 17}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 5, 1, 6, 1, 2, 1}, {16, 5, 1, 6, 1, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 5, 4, 6, 2, 2, 17}, {16, 5, 4, 6, 2, 2, 17}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_8)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{4, 14, 6, 1}, {4, 14, 6, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 1, 1, 4}, {1, 1, 1, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{4, 14, 6, 4}, {4, 14, 6, 4}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_9)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{13, 1, 3, 1, 1, 2, 1}, {13, 1, 3, 1, 1, 2, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{13, 10, 3, 14, 1, 2, 13}, {13, 10, 3, 14, 1, 2, 13}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{13, 10, 3, 14, 1, 2, 13}, {13, 10, 3, 14, 1, 2, 13}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_10)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{18}, {18}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{18}, {18}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_11)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{18, 7, 8, 6, 15, 3, 7}, {18, 7, 8, 6, 15, 3, 7}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1, 8, 1, 15, 1, 7}, {1, 1, 8, 1, 15, 1, 7}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{18, 7, 8, 6, 15, 3, 7}, {18, 7, 8, 6, 15, 3, 7}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_12)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{14, 16}, {14, 16}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{14, 16}, {14, 16}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{14, 16}, {14, 16}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_13)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{7, 1, 6, 1, 14, 1, 2}, {7, 1, 6, 1, 14, 1, 2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1, 20, 6, 4, 1, 4, 2}, {1, 20, 6, 4, 1, 4, 2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{7, 20, 6, 4, 14, 4, 2}, {7, 20, 6, 4, 14, 4, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_14)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{9, 3, 9, 16}, {9, 3, 9, 16}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            {{{9, 3, 9, 16}, {9, 3, 9, 16}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_15)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{1, 7, 1, 6, 12, 6, 2}, {1, 7, 1, 6, 12, 6, 2}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{12, 1, 2, 1, 12, 1, 2}, {12, 1, 2, 1, 12, 1, 2}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            {{{12, 7, 2, 6, 12, 6, 2}, {12, 7, 2, 6, 12, 6, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_16)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{10, 20, 4, 20, 13}, {10, 20, 4, 20, 13}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{10, 20, 4, 20, 13}, {10, 20, 4, 20, 13}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_17)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{1, 10, 1, 2, 6, 1, 16, 1}, {1, 10, 1, 2, 6, 1, 16, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{5, 1, 2, 1, 1, 4, 1, 18}, {5, 1, 2, 1, 1, 4, 1, 18}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{5, 10, 2, 2, 6, 4, 16, 18}, {5, 10, 2, 2, 6, 4, 16, 18}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(NotEqualTilingTest, not_equal_test_failed_different_dtype)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{1, 17, 1, 12}, {1, 17, 1, 12}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1, 12}, {1, 1, 1, 12}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4, 17, 1, 12}, {4, 17, 1, 12}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(NotEqualTilingTest, not_equal_test_failed_invalid_dtype)
{
    optiling::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "NotEqual",
        {
            {{{1, 17, 1, 12}, {1, 17, 1, 12}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1, 1, 1, 12}, {1, 1, 1, 12}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            {{{4, 17, 1, 12}, {4, 17, 1, 12}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}