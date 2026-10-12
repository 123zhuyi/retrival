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
#include "../../../op_host/moe_finalize_routing_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeFinalizeRoutingTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingTiling TearDown" << std::endl;
    }
};

// ----------------------------------------------------------------------------------------------------------

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_float) {
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                            {
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                            &compileInfo);
    int64_t expectTilingKey = 20015;
    string expectTilingData = "64 16 0 16 16 16 16 16 0 0 0 0 1 1 1 1 1 1 1 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_bfloat16) {
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                            {
                                                {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},},
                                            &compileInfo);
    int64_t expectTilingKey = 20017;
    string expectTilingData = "64 16 0 16 16 16 16 16 0 0 0 0 1 1 1 1 1 1 1 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_float16) {
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                            {
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 1}, {16, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                            &compileInfo);
    int64_t expectTilingKey = 20016;
    string expectTilingData = "64 16 0 16 16 16 16 16 0 0 0 0 1 1 1 1 1 1 1 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_h_float) {
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                            {
                                                {{{4096 * 4, 8934}, {4096 * 4, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{8, 8934}, {8, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096, 4}, {4096, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{4096 * 4}, {4096 * 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{4096, 4}, {4096, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{4096, 8934}, {4096, 8934}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                            &compileInfo);
    int64_t expectTilingKey = 20009;
    string expectTilingData = "64 64 0 8 4096 8934 5448 3486 1 0 0 0 4 64 128 1 1 64 128 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_h_float16) {
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                            {
                                                {{{4096 * 2, 15080}, {4096 * 2, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 15080}, {4096, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 15080}, {4096, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 15080}, {16, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096, 2}, {4096, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{4096 * 2}, {4096 * 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{4096, 2}, {4096, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{4096, 15080}, {4096, 15080}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                            &compileInfo);
    int64_t expectTilingKey = 20016;
    string expectTilingData = "64 64 0 16 4096 15080 15080 15080 0 0 0 0 2 64 64 1 1 64 64 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_hk_float) {
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                            {
                                                {{{1024 * 258, 8936}, {1024 * 258, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{340, 8936}, {340, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024 * 258}, {1024 * 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{1024, 8936}, {1024, 8936}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                            &compileInfo);
    int64_t expectTilingKey = 20000;
    string expectTilingData = "64 64 0 340 1024 8936 8120 816 1 256 2 2 258 16 32 1 1 16 32 1 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_cut_k_float) {
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                            {
                                                {{{1024 * 258, 2560}, {1024 * 258, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{340, 2560}, {340, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024 * 258}, {1024 * 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                            &compileInfo);
    int64_t expectTilingKey = 20000;
    string expectTilingData = "64 64 0 340 1024 2560 2560 2560 0 256 2 2 258 16 4 5 1 16 4 5 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeFinalizeRoutingTiling, MoeFinalizeRouting_tiling_float_newwork) {
    optiling::MoeFinalizeRoutingCompileInfo compileInfo = {40, 65536};
    gert::TilingContextPara tilingContextPara("MoeFinalizeRouting",
                                            {
                                                {{{1024 * 258, 2560}, {1024 * 258, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{340, 2560}, {340, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1024 * 258}, {1024 * 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1024, 258}, {1024, 258}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {{{{1024, 2560}, {1024, 2560}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                            &compileInfo);
    int64_t expectTilingKey = 20000;
    string expectTilingData = "64 64 0 340 1024 2560 2560 2560 0 256 2 2 258 16 4 5 1 16 4 5 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}