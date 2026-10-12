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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "conversion/masked_select_v3/op_host/masked_select_v3_tiling.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace gert;
using namespace optiling;

class l2_masked_select_non_regbase_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "masked_select_test Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "masked_select_test Tiling TearDown" << std::endl; }
};

TEST_F(l2_masked_select_non_regbase_test, aclnnMaskedSelect_base_case_1)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara("MaskedSelectV3",
                                              {{{{
                                                     8,
                                                 },
                                                 {
                                                     8,
                                                 }},
                                                ge::DT_FLOAT,
                                                ge::FORMAT_ND},
                                               {{{
                                                     8,
                                                 },
                                                 {
                                                     8,
                                                 }},
                                                ge::DT_BOOL,
                                                ge::FORMAT_ND}},
                                              {
                                                  {{{
                                                        8,
                                                    },
                                                    {
                                                        8,
                                                    }},
                                                   ge::DT_FLOAT,
                                                   ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "1 8 1 8448 8 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777536};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(l2_masked_select_non_regbase_test, aclnnMaskedSelect_8_int32_nd_and_8_bool_nd_simple_random_case_1)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara("MaskedSelectV3",
                                              {{{{
                                                     8,
                                                 },
                                                 {
                                                     8,
                                                 }},
                                                ge::DT_INT32,
                                                ge::FORMAT_ND},
                                               {{{
                                                     8,
                                                 },
                                                 {
                                                     8,
                                                 }},
                                                ge::DT_BOOL,
                                                ge::FORMAT_ND}},
                                              {
                                                  {{{
                                                        8,
                                                    },
                                                    {
                                                        8,
                                                    }},
                                                   ge::DT_INT32,
                                                   ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "1 8 1 8448 8 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777536};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(l2_masked_select_non_regbase_test,
       aclnnMaskedSelect_2_4_6_8_10_12_14_16_int64_nd_and_2_4_6_8_10_12_14_16_bool_nd)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{2, 4, 6, 8, 10, 12, 14, 16}, {2, 4, 6, 8, 10, 12, 14, 16}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2, 4, 6, 8, 10, 12, 14, 16}, {2, 4, 6, 8, 10, 12, 14, 16}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {
            {{{2, 4, 6, 8, 10, 12, 14, 16}, {2, 4, 6, 8, 10, 12, 14, 16}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "48 215040 50 4352 1792 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {99355648};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_float16_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{256}, {256}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_bf16_2d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{64, 128}, {64, 128}}, ge::DT_BF16, ge::FORMAT_ND}, {{{64, 128}, {64, 128}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{64, 128}, {64, 128}}, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_int8_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{512}, {512}}, ge::DT_INT8, ge::FORMAT_ND}, {{{512}, {512}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{512}, {512}}, ge::DT_INT8, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_double_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{128}, {128}}, ge::DT_DOUBLE, ge::FORMAT_ND}, {{{128}, {128}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{128}, {128}}, ge::DT_DOUBLE, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_uint16_2d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{32, 64}, {32, 64}}, ge::DT_UINT16, ge::FORMAT_ND}, {{{32, 64}, {32, 64}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{32, 64}, {32, 64}}, ge::DT_UINT16, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_uint32_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{256}, {256}}, ge::DT_UINT32, ge::FORMAT_ND}, {{{256}, {256}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{256}, {256}}, ge::DT_UINT32, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_uint64_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{128}, {128}}, ge::DT_UINT64, ge::FORMAT_ND}, {{{128}, {128}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{128}, {128}}, ge::DT_UINT64, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_bool_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{512}, {512}}, ge::DT_BOOL, ge::FORMAT_ND}, {{{512}, {512}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{512}, {512}}, ge::DT_BOOL, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_fp32_tail_num)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{1000}, {1000}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1000}, {1000}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{1000}, {1000}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_fp32_large_tail_num_positive)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{100000}, {100000}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{100000}, {100000}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{100000}, {100000}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "4 8334 1 8448 8334 8 8333 1 8448 8333 ";
    std::vector<size_t> expectWorkspaces = {17178112};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_fp32_former_last_tile_zero)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{405504}, {405504}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{405504}, {405504}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{405504}, {405504}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "48 8448 1 8448 8448 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {18402304};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_int16_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{256}, {256}}, ge::DT_INT16, ge::FORMAT_ND}, {{{256}, {256}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{256}, {256}}, ge::DT_INT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "1 256 1 13824 256 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777792};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_uint8_1d)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3", {{{{512}, {512}}, ge::DT_UINT8, ge::FORMAT_ND}, {{{512}, {512}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{512}, {512}}, ge::DT_UINT8, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "1 512 1 12288 512 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777792};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(l2_masked_select_non_regbase_test, non_regbase_fp32_tail_last_tile_zero)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
        {{{{405528}, {405528}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{405528}, {405528}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{405528}, {405528}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
}
