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
 * \file test_abs_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/abs_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class AbsTilingTest : public testing::Test {
   protected:
    static void SetUpTestCase()
    {
        std::cout << "AbsTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AbsTilingTest TearDown" << std::endl;
    }
};

TEST_F(AbsTilingTest, test_tiling_fp16_001)
{
    optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "8192 140737488355332 2048 4 1 1 2048 2048 32768 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsTilingTest, test_tiling_fp32_002)
{
    optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "8192 70368744177672 1024 8 1 1 1024 1024 16384 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsTilingTest, test_tiling_int8_003)
{
    optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "8192 281474976710658 4096 2 1 1 4096 4096 65536 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsTilingTest, test_tiling_int16_004)
{
    optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "8192 140737488355332 2048 4 1 1 2048 2048 32768 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsTilingTest, test_tiling_int32_005)
{
    optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "8192 70368744177672 1024 8 1 1 1024 1024 16384 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsTilingTest, test_tiling_int64_006)
{
    optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "8192 35184372088848 512 16 1 1 512 512 8192 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsTilingTest, test_tiling_bf16_007)
{
    optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "8192 46729244180484 2048 4 1 1 2048 2048 10880 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsTilingTest, test_tiling_failed_bf16_fp32_008)
{
    optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsTilingTest, test_tiling_failed_empty_tensor_009)
{
optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsTilingTest, test_tiling_failed_empty_tensor_010)
{
optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 1, 2, 64}, {1, 1, 2, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 1, 2, 64}, {1, 1, 2, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsTilingTest, test_tiling_complex64_011)
{
    optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 103;
    string expectTilingData = "8192 35184372088840 1024 8 1 1 1024 1024 8192 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AbsTilingTest, test_tiling_complex32_012)
{
    optiling::AbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Abs",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 103;
    string expectTilingData = "8192 70368744177668 2048 4 1 1 2048 2048 16384 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}