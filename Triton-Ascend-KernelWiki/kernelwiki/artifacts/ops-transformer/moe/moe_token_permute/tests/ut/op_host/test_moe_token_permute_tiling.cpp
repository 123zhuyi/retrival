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
#include "../../../op_host/moe_token_permute_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeTokenPermuteTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenPermuteTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenPermuteTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_float) {
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermute",
                                            {
                                                {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 2 5 8 3 1 6 1 6 6 6 1 6 6 8160 0 0 1024 2 2 0 1 0 20 2 2976 5952 17856 1 1 1 1 1 0 5952 2 2976 1 1 6 95232 71424 ";
    std::vector<size_t> expectWorkspaces = {16777424};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_float16) {
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeTokenPermute",
                                            {
                                                {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{6, 5}, {6, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 2 5 16 3 1 6 1 6 6 6 1 6 6 8160 0 0 1024 2 2 0 1 0 10 2 2976 5952 17856 1 1 1 1 1 0 5952 2 2976 1 1 6 95232 71424 ";
    std::vector<size_t> expectWorkspaces = {16777424};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_float16_multi_core) {
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    int64_t tokenNum = 16384;
    int64_t topk = 3;

    gert::TilingContextPara tilingContextPara("MoeTokenPermute",
                                            {
                                                {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{tokenNum * topk, 5}, {tokenNum * topk, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 16384 5 16 3 16 3072 1 3072 3072 3072 1 3072 3072 8096 0 4 1024 64 64 0 256 256 10 2 2976 5952 17856 1 1 1 1 256 1 256 1 256 1 256 49152 95232 71424 ";
    std::vector<size_t> expectWorkspaces = {17960960};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_float16_long) {
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    int64_t tokenNum = 16384;
    int64_t topk = 3;
    int64_t h = 32768;

    gert::TilingContextPara tilingContextPara("MoeTokenPermute",
                                            {
                                                {{{tokenNum, h}, {tokenNum, h}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{tokenNum * topk, h}, {tokenNum * topk, h}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 16384 32768 32768 3 16 3072 1 3072 3072 3072 1 3072 3072 8096 0 4 1024 64 64 0 256 256 65536 10901 1 10901 32703 1 1 1 1 256 1 256 256 1 256 1 49152 65536 130816 ";
    std::vector<size_t> expectWorkspaces = {17960960};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}


TEST_F(MoeTokenPermuteTiling, MoeTokenPermute_tiling_float16_error) {
    optiling::MoeTokenPermuteCompileInfo compileInfo = {};
    int64_t tokenNum = 16384;
    int64_t topk = 3;
    int64_t h = 32768;

    gert::TilingContextPara tilingContextPara("MoeTokenPermute",
                                            {
                                                {{{tokenNum, h}, {tokenNum, h}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{tokenNum * topk, h}, {tokenNum * topk, h}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {"num_out_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
                                                {"padded_mode", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 16384 5 16 3 16 3072 1 3072 3072 3072 1 3072 3072 8096 0 4 1024 64 64 0 256 256 10 2 2976 5952 17856 1 1 1 1 256 1 256 1 256 1 256 49152 95232 17856 ";
    std::vector<size_t> expectWorkspaces = {17960960};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

