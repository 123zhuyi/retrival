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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class test_bincount_tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "test_bincount_tiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "test_bincount_tiling TearDown" << std::endl; }

    // TilingContextBuilder 要求 compileInfo 非空；Bincount tiling 不读取其内容。
    char compileInfoPlaceholder_ = 0;
};

TEST_F(test_bincount_tiling, tiling_basic)
{
    gert::TilingContextPara tilingContextPara("Bincount",
                                              {
                                                  {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND}, // array
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},       // size
                                                  {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND}, // weights
                                              },
                                              {
                                                  {{{32}, {32}}, ge::DT_FLOAT, ge::FORMAT_ND}, // bins
                                              },
                                              {}, &compileInfoPlaceholder_, 64, 262144, 4096);

    uint64_t expectTilingKey = 0;
    // 小输入(totalNum=2048<=阈值)走单核: coreNum=1。
    // minWs=1*32=32; histWs=1*(32+32)*8=512; sys=16777216 -> 16777760
    std::vector<size_t> expectWorkspaces = {16777760};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(test_bincount_tiling, tiling_small)
{
    gert::TilingContextPara tilingContextPara("Bincount",
                                              {
                                                  {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND}, // array
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}, // size
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // weights
                                              },
                                              {
                                                  {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // bins
                                              },
                                              {}, &compileInfoPlaceholder_, 64, 262144, 4096);

    uint64_t expectTilingKey = 0;
    // 小输入(totalNum=8)走单核: coreNum=1。
    // minWs=1*32=32; histWs=1*(4+32)*8=288; sys=16777216 -> 16777536
    std::vector<size_t> expectWorkspaces = {16777536};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 大 outLength 超 UB（非 double）：私有直方图放不下 UB，tiling 应转走 GM 散射回退路径
// （largeL=1）并返回 GRAPH_SUCCESS；此路径不申请跨核直方图 workspace，
// workspace = minWs(coreNum*blockSize) + 0 + sysWorkspace。
// bins=int64（acc=8B），L=50000 -> 2*L*8 ≈ 800KB 远超 faker UB(256KB)。
TEST_F(test_bincount_tiling, tiling_largeL_nondouble_gm_scatter)
{
    gert::TilingContextPara tilingContextPara(
        "Bincount",
        {
            {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND}, // array（N>=coreNum，避免 coreNum 被裁剪）
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},       // size
        },
        {
            {{{50000}, {50000}}, ge::DT_INT64, ge::FORMAT_ND}, // bins：L 远超 UB -> GM 散射路径
        },
        {}, &compileInfoPlaceholder_, 64, 262144, 4096);

    uint64_t expectTilingKey = 0;
    // minWs = 64*32 = 2048；histWs = 0（GM 路径）；sysWorkspace = 16777216
    std::vector<size_t> expectWorkspaces = {16779264};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 大 outLength 超 UB 且 out=double：GM 路径需对输出原子读改写，位拼接 double 无法支持，
// tiling 应拒绝并返回 GRAPH_FAILED。
TEST_F(test_bincount_tiling, tiling_largeL_double_reject)
{
    gert::TilingContextPara tilingContextPara(
        "Bincount",
        {
            {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND}, // array
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},   // size
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND}, // weights
        },
        {
            {{{50000}, {50000}}, ge::DT_DOUBLE, ge::FORMAT_ND}, // bins=double + L 超 UB -> 拒绝
        },
        {}, &compileInfoPlaceholder_, 64, 262144, 4096);

    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}
