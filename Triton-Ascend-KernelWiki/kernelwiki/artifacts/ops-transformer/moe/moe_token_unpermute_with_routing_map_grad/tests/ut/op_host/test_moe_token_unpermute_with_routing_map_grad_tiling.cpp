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
 * \file moe_token_unpermute_with_routing_map_grad.cpp
 * \brief
 */
#include <iostream>


#include <gtest/gtest.h>
#include "../../../op_host/moe_token_unpermute_with_routing_map_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class MoeTokenUnpermuteWithRoutingMapGradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithRoutingMapGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeTokenUnpermuteWithRoutingMapGradTiling TearDown" << std::endl;
    }
};

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_prob_not_none_bf16_tilingkey_1) {
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                              &compileInfo,
                                              socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 1;
    string expectTilingData = "30 1 0 64 64 30 30 10 1 0 1 0 64 1 0 64 0 1 16 64 196352 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_prob_not_none_bf16_tilingkey_11) {
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
                                              &compileInfo,
                                              socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 11;
    string expectTilingData = "30 0 0 64 64 30 30 10 0 0 1 0 64 1 8 64 1 1 0 0 196352 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tiling_prob_none_bf16_tilingkey_0) {
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 64}, {30, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                              &compileInfo,
                                              socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 0;
    string expectTilingData = "30 1 0 0 64 30 30 10 0 0 1 0 64 1 0 64 1 1 0 0 196352 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tilingkey_101) {
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}},
                                              &compileInfo,
                                              socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 101;
    string expectTilingData = "30 1 0 8 8 30 30 10 1 0 1 0 16 1 0 8 0 1 16 32 196352 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeTokenUnpermuteWithRoutingMapGradTiling, test_tilingkey_111) {
    optiling::MoeTokenUnpermuteWithRoutingMapGradCompileInfo compileInfo = {};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeTokenUnpermuteWithRoutingMapGrad",
                                              {
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30}, {30}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{30, 8}, {30, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{30, 8}, {30, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {{"drop_and_pad", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}},
                                              &compileInfo,
                                              socVersion, coreNum, ubSize);
    int64_t expectTilingKey = 111;
    string expectTilingData = "30 0 3 8 8 30 30 10 0 0 1 0 16 1 8 8 1 1 0 0 196352 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}