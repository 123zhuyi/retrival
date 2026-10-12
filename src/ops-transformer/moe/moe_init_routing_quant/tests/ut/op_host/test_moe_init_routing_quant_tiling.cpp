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
 * \file test_moe_init_routing_quant_tiling.h
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/moe_init_routing_quant_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class MoeInitRoutingQuantTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeInitRoutingQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeInitRoutingQuantTiling TearDown" << std::endl;
    }
};

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_float) {
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                            {
                                                {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{6, 5}, {6, 5}}, ge::DT_INT8, ge::FORMAT_ND},
                                                {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
                                                {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
                                                {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 1;
    string expectTilingData = "1 2 5 3 1065353216 1 6 1 6 6 6 1 6 6 8160 0 1024 1 0 6 0 0 0 6 6 6 0 0 0 6 6 0 0 1 6 2 3 3 3 2 2 2 3 3 3 2 2 5 0 ";
    std::vector<size_t> expectWorkspaces = {16777448};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_multi_core) {
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t tokenNum = 16384;
    int64_t topk = 3;
    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                            {
                                                {{{tokenNum, 5}, {tokenNum, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{tokenNum * topk, 5}, {tokenNum * topk, 5}}, ge::DT_INT8, ge::FORMAT_ND},
                                                {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
                                                {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
                                                {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 16384 5 3 1065353216 16 3072 1 3072 3072 3072 1 3072 3072 8096 4 1024 64 0 768 0 0 0 768 768 768 0 0 0 768 768 0 0 64 49152 256 3 3 3 256 256 256 3 3 3 256 256 5 0 ";
    std::vector<size_t> expectWorkspaces = {18157568};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_long_h) {
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t tokenNum = 32768;
    int64_t topk = 3;
    int64_t h = 16384;

    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                            {
                                                {{{tokenNum, h}, {tokenNum, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{tokenNum * topk, h}, {tokenNum * topk, h}}, ge::DT_INT8, ge::FORMAT_ND},
                                                {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
                                                {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
                                                {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 32768 16384 3 1065353216 16 6144 1 6144 6144 6144 1 6144 6144 8096 4 1024 64 0 1536 0 0 0 1536 1536 1536 0 0 0 1536 1536 0 0 64 98304 512 3 3 3 2 2 512 3 3 3 2 2 4094 0 ";
    std::vector<size_t> expectWorkspaces = {19533824};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeInitRoutingQuantTiling, MoeInitRoutingQuant_tiling_error) {
    optiling::MoeInitRoutingQuantCompileInfo compileInfo = {};
    int64_t tokenNum = 32768;
    int64_t topk = 3;
    int64_t h = 16384;

    gert::TilingContextPara tilingContextPara("MoeInitRoutingQuant",
                                            {
                                                {{{tokenNum, h}, {tokenNum, h}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{3, topk}, {3, topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{tokenNum, topk}, {tokenNum, topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{tokenNum * topk, h}, {tokenNum * topk, h}}, ge::DT_INT8, ge::FORMAT_ND},
                                                {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{tokenNum * topk}, {tokenNum * topk}}, ge::DT_INT32, ge::FORMAT_ND},
                                            },
                                            {
                                                {"active_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tokenNum * topk)},
                                                {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
                                                {"offset", Ops::Transformer::AnyValue::CreateFrom<float>(0.0)},
                                            },
                                            &compileInfo);
    int64_t expectTilingKey = 2;
    string expectTilingData = "64 16384 5 3 1065353216 16 3072 1 3072 3072 3072 1 3072 3072 8096 4 1024 64 0 768 0 0 0 768 768 768 0 0 0 768 768 0 0 64 49152 256 3 3 3 256 256 256 3 3 3 256 256 5 0 ";
    std::vector<size_t> expectWorkspaces = {18157568};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}