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
#include "../../../op_host/moe_token_unpermute_with_routing_map_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeTokenUnpermuteWithRoutingMapTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MoeTokenUnpermuteWithRoutingMapTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MoeTokenUnpermuteWithRoutingMapTiling TearDown" << std::endl;
  }
};

TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_fp32_droppad)
{
    optiling::MoeTokenUnpermuteWithRoutingMapCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithRoutingMap",
        {
            {{{40968*8, 7168}, {40968*8, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2568*8}, {2568*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4096, 265}, {4096, 256}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2568*8}, {2568*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2568*8}, {2568*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2568*8}, {2568*8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom(true)},
            {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({4096, 7168})},
        },
        &compileInfo);
    int64_t expectTilingKey = 1000;
    string expectTilingData =
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 4096 8 2568 64 1 321 0 321 0 1 321 0 321 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapTiling, test_tiling_bf16)
{
    optiling::MoeTokenUnpermuteWithRoutingMapCompileInfo compileInfo = {48, 65536};
    gert::TilingContextPara tilingContextPara(
        "MoeTokenUnpermuteWithRoutingMap",
        {
            {{{40968*8, 7168}, {40968*8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2568*8}, {2568*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4096, 265}, {4096, 256}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{4096, 8}, {4096, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2568*8}, {2568*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2568*8}, {2568*8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2568*8}, {2568*8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom(false)},
            {"restore_shape", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>({4096, 7168})},
        },
        &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData =
        "7168 80 327744 7168 1 0 64 0 64 1 0 64 4 8 4096 1 4096 4096 0 0 0 0 0 4096 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {2 * 16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
    // ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}