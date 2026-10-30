/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_host/arch35/masked_fill_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class MaskedFillTiling : public testing::Test {
  protected:
    static void SetUpTestCase() {
      std::cout << "MaskedFillTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "MaskedFillTiling TearDown" << std::endl;
    }
};

TEST_F(MaskedFillTiling, masked_fill_test_0)
{
    optiling::MaskedFillCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("MaskedFill",
                                              {
                                                {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "16 1125904201810432 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MaskedFillTiling, masked_fill_test_1)
{
    optiling::MaskedFillCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("MaskedFill",
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 1125908496777728 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MaskedFillTiling, masked_fill_test_2)
{
    optiling::MaskedFillCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("MaskedFill",
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 1, 1, 1, 1}, {16, 1, 1, 1, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MaskedFillTiling, masked_fill_test_3)
{
    optiling::MaskedFillCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("MaskedFill",
                                              {
                                                {{{16, 1, 1, 1, 1, 1, 1, 1}, {16, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 1, 1, 1, 1, 1, 1, 1}, {16, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 1, 1, 1, 1, 1, 1}, {16, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "16 1125904201810432 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MaskedFillTiling, masked_fill_test_4)
{
    optiling::MaskedFillCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("MaskedFill",
                                              {
                                                {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "16 1125904201810432 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MaskedFillTiling, masked_fill_test_5)
{
    optiling::MaskedFillCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("MaskedFill",
                                              {
                                                {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16}, {16}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "16 1125904201810432 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}