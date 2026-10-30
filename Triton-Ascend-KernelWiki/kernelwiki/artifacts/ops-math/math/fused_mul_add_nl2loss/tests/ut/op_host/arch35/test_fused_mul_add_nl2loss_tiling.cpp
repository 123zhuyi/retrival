/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_fused_mul_add_nl2loss_tiling.cpp
 * \brief FusedMulAddNL2loss arch35 tiling UT
 *        tiling data 字段序：totalElements coreElements tailCoreElements ubTileSize（4 x int64）
 *        平台参数按 faker 平台生效：coreNum=64, ubSize=262144（对应 ubTile=9024）；
 *        TilingContextPara 会提供 platformInfo，compileInfo 传参被忽略
 */

#include "../../../../op_host/arch35/fused_mul_add_nl2loss_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class FusedMulAddNL2lossTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FusedMulAddNL2lossTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FusedMulAddNL2lossTilingTest TearDown" << std::endl; }
};

// fp32 一维中 shape：16 核均分
TEST_F(FusedMulAddNL2lossTilingTest, test_tiling_fp32_1024)
{
    optiling::FusedMulAddNL2lossCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("FusedMulAddNL2loss",
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1024 64 64 9024 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// fp32 尾块：尾核 75
TEST_F(FusedMulAddNL2lossTilingTest, test_tiling_fp32_1035)
{
    optiling::FusedMulAddNL2lossCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("FusedMulAddNL2loss",
                                              {
                                                  {{{1035}, {1035}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1035}, {1035}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1035}, {1035}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1035 64 75 9024 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// fp16 尾块：tiling 与 dtype 无关，结果同 fp32
TEST_F(FusedMulAddNL2lossTilingTest, test_tiling_fp16_1035)
{
    optiling::FusedMulAddNL2lossCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("FusedMulAddNL2loss",
                                              {
                                                  {{{1035}, {1035}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1035}, {1035}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1035}, {1035}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1035 64 75 9024 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// fp32 单核：N=64 恰好一个 VL
TEST_F(FusedMulAddNL2lossTilingTest, test_tiling_fp32_64)
{
    optiling::FusedMulAddNL2lossCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("FusedMulAddNL2loss",
                                              {
                                                  {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "64 64 64 9024 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// fp32 单元素：N=1
TEST_F(FusedMulAddNL2lossTilingTest, test_tiling_fp32_1)
{
    optiling::FusedMulAddNL2lossCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("FusedMulAddNL2loss",
                                              {
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1 1 1 9024 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// fp32 二维大 shape：64 核均分（1024000 / 64 = 16000，尾核与前核相同）
TEST_F(FusedMulAddNL2lossTilingTest, test_tiling_fp32_8x128000)
{
    optiling::FusedMulAddNL2lossCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("FusedMulAddNL2loss",
                                              {
                                                  {{{8, 128000}, {8, 128000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8, 128000}, {8, 128000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 128000}, {8, 128000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1024000 16000 16000 9024 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// fp32 三维小 shape：N=32 单核
TEST_F(FusedMulAddNL2lossTilingTest, test_tiling_fp32_2x4x4)
{
    optiling::FusedMulAddNL2lossCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("FusedMulAddNL2loss",
                                              {
                                                  {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "32 32 32 9024 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
