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
 * \file test_mod_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/mod_tiling_arch35.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace Ops::Base;

class ModTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ModTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ModTilingTest TearDown" << std::endl;
    }
};

// Test: mod tiling with float32
TEST_F(ModTilingTest, test_tiling_float32)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "8192 137438953600 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test: mod tiling with float16
TEST_F(ModTilingTest, test_tiling_float16)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{1, 32, 4, 32}, {1, 32, 4, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 32, 4, 32}, {1, 32, 4, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 32, 4, 32}, {1, 32, 4, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "4096 34359738624 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test: mod tiling with int32
TEST_F(ModTilingTest, test_tiling_int32)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "8192 137438953600 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test: mod tiling with broadcast
TEST_F(ModTilingTest, test_tiling_broadcast)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Test: mod tiling with small shape
TEST_F(ModTilingTest, test_tiling_small_shape)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Test: mod tiling with large shape
TEST_F(ModTilingTest, test_tiling_large_shape)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{16, 128, 16, 128}, {16, 128, 16, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 128, 16, 128}, {16, 128, 16, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 128, 16, 128}, {16, 128, 16, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Test: mod tiling with 1D shape
TEST_F(ModTilingTest, test_tiling_1d)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{1000}, {1000}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1000}, {1000}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1000}, {1000}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Test: mod tiling with NHWC format
TEST_F(ModTilingTest, test_tiling_nhwc)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{1, 16, 16, 3}, {1, 16, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{1, 16, 16, 3}, {1, 16, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{1, 16, 16, 3}, {1, 16, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Test: mod tiling with int8
TEST_F(ModTilingTest, test_tiling_int8)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "8192 34359738880 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test: mod tiling with uint8
TEST_F(ModTilingTest, test_tiling_uint8)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "8192 34359738880 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test: mod tiling with int64
TEST_F(ModTilingTest, test_tiling_int64)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 8;
    string expectTilingData = "8192 137438953600 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ModTilingTest, test_tiling_bf16)
{
    BroadcastCompileInfo compileInfo{};
    gert::TilingContextPara tilingContextPara(
        "Mod",
        {
            {{{1, 32, 4, 32}, {1, 32, 4, 32}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 32, 4, 32}, {1, 32, 4, 32}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 32, 4, 32}, {1, 32, 4, 32}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}