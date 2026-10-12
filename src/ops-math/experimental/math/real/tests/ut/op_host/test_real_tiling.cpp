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
#include "../../../op_kernel/real_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class RealTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RealTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RealTiling TearDown" << std::endl; }
};

// Test case for complex32 -> float16
TEST_F(RealTiling, test_real_complex32)
{
    RealCompileInfo compileInfo = {8, 262144};
    gert::TilingContextPara tilingContextPara(
        "Real",
        {
            {{{16, 16}, {16, 16}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1; // COMPLEX32_MODE
    // totalUsedCoreNum=1, tailBlockNum=0, ubPartDataNum=16320,
    // smallCoreDataNum=256, smallCoreLoopNum=1, smallCoreTailDataNum=256,
    // bigCoreDataNum=0, bigCoreLoopNum=0, bigCoreTailDataNum=0, tilingKey=1
    string expectTilingData = "1 0 16320 256 1 256 0 0 0 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216}; // RESERVED_WORKSPACE = 16MB
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case for complex64 -> float
TEST_F(RealTiling, test_real_complex64)
{
    RealCompileInfo compileInfo = {8, 262144};
    gert::TilingContextPara tilingContextPara(
        "Real",
        {
            {{{16, 16}, {16, 16}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 2; // COMPLEX64_MODE
    string expectTilingData = "1 0 8160 256 1 256 0 0 0 2 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case for float16 -> float16 (identity)
TEST_F(RealTiling, test_real_float16)
{
    RealCompileInfo compileInfo = {8, 262144};
    gert::TilingContextPara tilingContextPara(
        "Real",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 4; // FLOAT16_MODE
    // totalUsedCoreNum=1, tailBlockNum=0, ubPartDataNum=32704,
    // smallCoreDataNum=256, smallCoreLoopNum=1, smallCoreTailDataNum=256,
    // bigCoreDataNum=0, bigCoreLoopNum=0, bigCoreTailDataNum=0, tilingKey=4
    string expectTilingData = "1 0 32704 256 1 256 0 0 0 4 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case for float -> float (identity)
TEST_F(RealTiling, test_real_float)
{
    RealCompileInfo compileInfo = {8, 262144};
    gert::TilingContextPara tilingContextPara(
        "Real",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 5; // FLOAT_MODE
    // totalUsedCoreNum=1, tailBlockNum=0, ubPartDataNum=16352,
    // smallCoreDataNum=256, smallCoreLoopNum=1, smallCoreTailDataNum=256,
    // bigCoreDataNum=0, bigCoreLoopNum=0, bigCoreTailDataNum=0, tilingKey=5
    string expectTilingData = "1 0 16352 256 1 256 0 0 0 5 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case with larger shape (multi-core scenario)
TEST_F(RealTiling, test_real_complex64_large)
{
    RealCompileInfo compileInfo = {8, 262144};
    gert::TilingContextPara tilingContextPara(
        "Real",
        {
            {{{128, 128}, {128, 128}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        {
            {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 2; // COMPLEX64_MODE
    // totalLength=16384, dataTypeLength=4, totalBlocks=2048, coreNum=64
    // everyCoreBlockNum=32, tailBlockNum=0
    // smallCoreDataNum=256, smallCoreLoopNum=1, smallCoreTailDataNum=256
    string expectTilingData = "64 0 8160 256 1 256 0 0 0 2 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case for small shape (< 32B aligned, single core)
TEST_F(RealTiling, test_real_complex32_small)
{
    RealCompileInfo compileInfo = {8, 262144};
    gert::TilingContextPara tilingContextPara(
        "Real",
        {
            {{{4, 4}, {4, 4}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
        },
        {
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1; // COMPLEX32_MODE
    // totalLength=16, dataTypeLength=2, totalBytes=32, totalBlocks=1, coreNum=1
    // smallCoreDataNum=16, smallCoreLoopNum=1, smallCoreTailDataNum=16
    string expectTilingData = "1 0 10880 16 1 16 0 0 0 1 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
