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
 * \file test_tril_tiling.cpp
 * \brief
 */

#include "conversion/triu/op_host/arch35/triu_tiling.h"
#include <gtest/gtest.h>
#include <iostream>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class TrilTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TrilTest Setup" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "TrilTest TearDown" << std::endl;
  }
};

TEST_F(TrilTest, Tril_tiling_ascendc_bfloat16_dim3)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{1, 64, 64}, {1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 64}, {1, 64, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 42;
    string expectTilingData = "59392 64 64 1 1 1 0 1 1 1 0 64 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_float16_dim2)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{757, 2048}, {757, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{757, 2048}, {757, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 22;
    string expectTilingData = "29696 757 2048 0 64 1 45 7 2048 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_float_dim2)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{467, 47}, {467, 47}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{467, 47}, {467, 47}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 24;
    string expectTilingData = "29696 467 47 0 4 1 0 154 48 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_double_dim4)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{139, 9, 5, 19}, {139, 9, 5, 19}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{139, 9, 5, 19}, {139, 9, 5, 19}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 44;
    string expectTilingData = "59392 5 19 1 63 1 0 20 63 11 0 5 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_uint8_dim5)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{1, 5, 1, 83, 11}, {1, 5, 1, 83, 11}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 5, 1, 83, 11}, {1, 5, 1, 83, 11}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 41;
    string expectTilingData = "59392 83 11 1 5 1 0 1 5 1 0 11 72 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_int8_dim6)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{1, 3, 1, 8, 192, 100}, {1, 3, 1, 8, 192, 100}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 3, 1, 8, 192, 100}, {1, 3, 1, 8, 192, 100}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 41;
    string expectTilingData = "59392 192 100 1 24 1 0 1 24 1 0 100 92 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_uint16_dim7)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{23, 1, 41, 13, 1, 35, 72}, {23, 1, 41, 13, 1, 35, 72}}, ge::DT_UINT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{23, 1, 41, 13, 1, 35, 72}, {23, 1, 41, 13, 1, 35, 72}}, ge::DT_UINT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 42;
    string expectTilingData = "59392 35 72 1 64 19 10 10 1226 9 0 35 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_int16_dim8)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{7, 1, 4, 1, 3, 1, 1, 11}, {7, 1, 4, 1, 3, 1, 1, 11}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{7, 1, 4, 1, 3, 1, 1, 11}, {7, 1, 4, 1, 3, 1, 1, 11}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 32;
    string expectTilingData = "59392 11 42 1 0 2 42 2 1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_uint32_dim4)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{1, 1, 2048, 128}, {1, 1, 2048, 128}}, ge::DT_UINT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 1, 2048, 128}, {1, 1, 2048, 128}}, ge::DT_UINT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 24;
    string expectTilingData = "29696 2048 128 0 36 1 0 58 128 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_int32_dim4)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{1, 12, 32, 128}, {1, 12, 32, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 12, 32, 128}, {1, 12, 32, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 44;
    string expectTilingData = "59392 32 128 1 12 1 0 1 12 1 0 32 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_uint64_dim4)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{40, 16, 257, 257}, {40, 16, 257, 257}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{40, 16, 257, 257}, {40, 16, 257, 257}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 24;
    string expectTilingData = "29696 257 257 0 64 120 0 23 320 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_int64_dim3)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{6, 77, 768}, {6, 77, 768}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{6, 77, 768}, {6, 77, 768}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 28;
    string expectTilingData = "29696 77 768 0 64 2 16 77 32 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_int64_small_col_dim3)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{6, 7777, 2}, {6, 7777, 2}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{6, 7777, 2}, {6, 7777, 2}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 28;
    string expectTilingData = "29696 7777 2 0 54 1 0 928 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_bool_dim3)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{40, 7, 1408}, {40, 7, 1408}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{40, 7, 1408}, {40, 7, 1408}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 41;
    string expectTilingData = "59392 7 1408 1 40 1 0 1 40 1 0 7 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_complex32_dim3)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{12, 640, 4096}, {12, 640, 4096}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{12, 640, 4096}, {12, 640, 4096}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 24;
    string expectTilingData = "29696 640 4096 0 64 120 0 1 4096 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_complex64_dim3)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{98, 17, 1}, {98, 17, 1}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{98, 17, 1}, {98, 17, 1}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 18;
    string expectTilingData = "118784 14 128 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TrilTest, Tril_tiling_ascendc_float16_dim_not_support)
{
    optiling::TriluCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Tril",
                                              {
                                                {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 42;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}