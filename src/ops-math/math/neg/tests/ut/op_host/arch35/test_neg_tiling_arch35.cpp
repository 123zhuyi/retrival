/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../../op_host/arch35/neg_tiling_arch35.h"
#include "atvoss/elewise/elewise_tiling.h"

using namespace std;
using namespace ge;

class NegTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NegTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NegTiling TearDown" << std::endl; }
};

TEST_F(NegTiling, neg_test_tiling_float16_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "64 140737488355329 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_float_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 7;
    string expectTilingData = "64 70368744177665 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_INT32_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 11;
    string expectTilingData = "64 70368744177665 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_int8_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 9;
    string expectTilingData = "64 281474976710657 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_int64_input)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 13;
    string expectTilingData = "64 35184372088833 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_invalid_shape)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NegTiling, neg_test_tiling_invalid_dtype)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("Neg",
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 8}, {8, 8}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
