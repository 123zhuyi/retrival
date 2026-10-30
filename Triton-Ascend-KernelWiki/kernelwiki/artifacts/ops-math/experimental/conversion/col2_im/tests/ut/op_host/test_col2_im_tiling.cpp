/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Qiu Zhuang <@qiu-zhuang>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_kernel/col2_im_tiling_data.h"
#include "../../../op_kernel/col2_im_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class Col2ImTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Col2ImTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "Col2ImTiling TearDown" << endl; }
};

// ============================================================================
// 测试用例 1: FP32 基础场景 - 小尺寸输入
// ============================================================================
TEST_F(Col2ImTiling, ascend910b_test_tiling_fp32_001)
{
    optiling::Col2ImCompileInfo compileInfo = {};

    // 输入: [1, 18, 9] -> C*kH*kW=18 (C=2, k=3x3), L=9 (3x3输出位置)
    // 属性: H=4, W=4, kernel=3x3, stride=1, padding=0, dilation=1
    // 输出: [1, 2, 4, 4] = 32像素
    gert::TilingContextPara tilingContextPara(
        "Col2Im",
        {
            {{{1, 18, 4}, {1, 18, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 2, 4, 4}, {1, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo, {4, 4, 3, 3, 1, 0, 1} // H, W, kernel_h, kernel_w, stride, padding, dilation
    );

    uint64_t expectTilingKey = 0; // FP32 -> ELEMENTWISE_TPL_SCH_MODE_0
    string expectTilingData = "1 2 4 4 3 3 1 0 1 18 4 2 2 72 32 32 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================================================
// 测试用例 2: FP16 基础场景 - 中等尺寸
// ============================================================================
TEST_F(Col2ImTiling, ascend910b_test_tiling_fp16_002)
{
    optiling::Col2ImCompileInfo compileInfo = {};

    // 输入: [2, 32, 64] -> C*kH*kW=32 (C=8, k=2x2), L=64 (8x8输出位置)
    // 属性: H=10, W=10, kernel=2x2, stride=1, padding=1, dilation=1
    // 输出: [2, 8, 10, 10] = 1600像素
    gert::TilingContextPara tilingContextPara(
        "Col2Im",
        {
            {{{2, 32, 64}, {2, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 10, 10}, {2, 8, 10, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo, {10, 10, 2, 2, 1, 1, 1} // H, W, kernel_h, kernel_w, stride, padding, dilation
    );

    uint64_t expectTilingKey = 1; // FP16 -> ELEMENTWISE_TPL_SCH_MODE_1
    string expectTilingData = "2 8 10 10 2 2 1 1 1 32 64 8 8 4096 1600 1600 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================================================
// 测试用例 3: FP32 - 大stride场景
// ============================================================================
TEST_F(Col2ImTiling, ascend910b_test_tiling_fp32_003)
{
    optiling::Col2ImCompileInfo compileInfo = {};

    // 输入: [1, 12, 16] -> C*kH*kW=12 (C=3, k=2x2), L=16 (4x4输出位置)
    // 属性: H=8, W=8, kernel=2x2, stride=2, padding=0, dilation=1
    // 输出: [1, 3, 8, 8] = 192像素
    gert::TilingContextPara tilingContextPara(
        "Col2Im",
        {
            {{{1, 12, 16}, {1, 12, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 3, 8, 8}, {1, 3, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo, {8, 8, 2, 2, 2, 0, 1} // stride=2
    );

    uint64_t expectTilingKey = 0;
    string expectTilingData = "1 3 8 8 2 2 2 0 1 12 16 4 4 192 192 192 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================================================
// 测试用例 4: FP16 - 带padding场景
// ============================================================================
TEST_F(Col2ImTiling, ascend910b_test_tiling_fp16_004)
{
    optiling::Col2ImCompileInfo compileInfo = {};

    // 输入: [1, 27, 36] -> C*kH*kW=27 (C=3, k=3x3), L=36 (6x6输出位置)
    // 属性: H=8, W=8, kernel=3x3, stride=1, padding=1, dilation=1
    // 输出: [1, 3, 8, 8] = 192像素
    gert::TilingContextPara tilingContextPara(
        "Col2Im",
        {
            {{{1, 27, 36}, {1, 27, 36}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 3, 8, 8}, {1, 3, 8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo, {8, 8, 3, 3, 1, 1, 1} // padding=1
    );

    uint64_t expectTilingKey = 1;
    string expectTilingData = "1 3 8 8 3 3 1 1 1 27 36 6 6 972 192 192 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================================================
// 测试用例 5: FP32 - 大batch场景
// ============================================================================
TEST_F(Col2ImTiling, ascend910b_test_tiling_fp32_005)
{
    optiling::Col2ImCompileInfo compileInfo = {};

    // 输入: [4, 16, 25] -> C*kH*kW=16 (C=4, k=2x2), L=25 (5x5输出位置)
    // 属性: H=6, W=6, kernel=2x2, stride=1, padding=0, dilation=1
    // 输出: [4, 4, 6, 6] = 576像素
    gert::TilingContextPara tilingContextPara(
        "Col2Im",
        {
            {{{4, 16, 25}, {4, 16, 25}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4, 4, 6, 6}, {4, 4, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo, {6, 6, 2, 2, 1, 0, 1});

    uint64_t expectTilingKey = 0;
    string expectTilingData = "4 4 6 6 2 2 1 0 1 16 25 5 5 1600 576 576 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================================================
// 测试用例 6: FP16 - 非正方形kernel
// ============================================================================
TEST_F(Col2ImTiling, ascend910b_test_tiling_fp16_006)
{
    optiling::Col2ImCompileInfo compileInfo = {};

    // 输入: [1, 24, 20] -> C*kH*kW=24 (C=4, k=2x3), L=20 (4x5输出位置)
    // 属性: H=5, W=7, kernel=2x3, stride=1, padding=0, dilation=1
    // 输出: [1, 4, 5, 7] = 140像素
    gert::TilingContextPara tilingContextPara(
        "Col2Im",
        {
            {{{1, 24, 20}, {1, 24, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 4, 5, 7}, {1, 4, 5, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo, {5, 7, 2, 3, 1, 0, 1} // kernel=2x3
    );

    uint64_t expectTilingKey = 1;
    string expectTilingData = "1 4 5 7 2 3 1 0 1 24 20 4 5 480 140 140 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================================================
// 测试用例 7: FP32 - dilation场景
// ============================================================================
TEST_F(Col2ImTiling, ascend910b_test_tiling_fp32_007)
{
    optiling::Col2ImCompileInfo compileInfo = {};

    // 输入: [1, 18, 16] -> C*kH*kW=18 (C=2, k=3x3), L=16 (4x4输出位置)
    // 属性: H=10, W=10, kernel=3x3, stride=1, padding=0, dilation=2
    // 输出: [1, 2, 10, 10] = 200像素
    gert::TilingContextPara tilingContextPara(
        "Col2Im",
        {
            {{{1, 18, 16}, {1, 18, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 2, 10, 10}, {1, 2, 10, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo, {10, 10, 3, 3, 1, 0, 2} // dilation=2
    );

    uint64_t expectTilingKey = 0;
    string expectTilingData = "1 2 10 10 3 3 1 0 2 18 16 4 4 288 200 200 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================================================
// 测试用例 8: FP16 - 大尺寸场景
// ============================================================================
TEST_F(Col2ImTiling, ascend910b_test_tiling_fp16_008)
{
    optiling::Col2ImCompileInfo compileInfo = {};

    // 输入: [1, 64, 256] -> C*kH*kW=64 (C=16, k=2x2), L=256 (16x16输出位置)
    // 属性: H=18, W=18, kernel=2x2, stride=1, padding=1, dilation=1
    // 输出: [1, 16, 18, 18] = 5184像素
    gert::TilingContextPara tilingContextPara(
        "Col2Im",
        {
            {{{1, 64, 256}, {1, 64, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 16, 18, 18}, {1, 16, 18, 18}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo, {18, 18, 2, 2, 1, 1, 1});

    uint64_t expectTilingKey = 1;
    string expectTilingData = "1 16 18 18 2 2 1 1 1 64 256 16 16 16384 5184 5184 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================================================
// 测试用例 9: FP32 - 多channel场景
// ============================================================================
TEST_F(Col2ImTiling, ascend910b_test_tiling_fp32_009)
{
    optiling::Col2ImCompileInfo compileInfo = {};

    // 输入: [2, 72, 49] -> C*kH*kW=72 (C=8, k=3x3), L=49 (7x7输出位置)
    // 属性: H=8, W=8, kernel=3x3, stride=1, padding=0, dilation=1
    // 输出: [2, 8, 8, 8] = 1024像素
    gert::TilingContextPara tilingContextPara(
        "Col2Im",
        {
            {{{2, 72, 49}, {2, 72, 49}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 8, 8}, {2, 8, 8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo, {8, 8, 3, 3, 1, 0, 1});

    uint64_t expectTilingKey = 0;
    string expectTilingData = "2 8 8 8 3 3 1 0 1 72 49 7 7 7056 1024 1024 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================================================
// 测试用例 10: FP16 - 边界条件：最小尺寸
// ============================================================================
TEST_F(Col2ImTiling, ascend910b_test_tiling_fp16_010)
{
    optiling::Col2ImCompileInfo compileInfo = {};

    // 输入: [1, 4, 1] -> C*kH*kW=4 (C=1, k=2x2), L=1 (1x1输出位置)
    // 属性: H=2, W=2, kernel=2x2, stride=1, padding=0, dilation=1
    // 输出: [1, 1, 2, 2] = 4像素
    gert::TilingContextPara tilingContextPara(
        "Col2Im",
        {
            {{{1, 4, 1}, {1, 4, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 2, 2}, {1, 1, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo, {2, 2, 2, 2, 1, 0, 1});

    uint64_t expectTilingKey = 1;
    string expectTilingData = "1 1 2 2 2 2 1 0 1 4 1 1 1 4 4 4 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}