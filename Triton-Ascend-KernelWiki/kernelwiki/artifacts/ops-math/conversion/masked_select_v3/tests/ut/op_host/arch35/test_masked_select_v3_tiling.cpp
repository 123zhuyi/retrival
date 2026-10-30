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

class l2_masked_select_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "masked_select_test Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "masked_select_test Tiling TearDown" << std::endl;
    }
};

TEST_F(l2_masked_select_test, aclnnMaskedSelect_base_case_1)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
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

TEST_F(l2_masked_select_test, aclnnMaskedSelect_8_int32_nd_and_8_bool_nd_simple_random_case_1)
{
    optiling::MaskedSelectV3CompileInfo compileInfo = {48, 196608, 16777216, false};

    gert::TilingContextPara tilingContextPara(
        "MaskedSelectV3",
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

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_6_8_10_12_14_16_int64_nd_and_2_4_6_8_10_12_14_16_bool_nd)
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