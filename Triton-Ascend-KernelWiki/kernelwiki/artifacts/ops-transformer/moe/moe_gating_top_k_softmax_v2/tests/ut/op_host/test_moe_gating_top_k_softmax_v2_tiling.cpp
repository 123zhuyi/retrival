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
#include "../../../op_host/moe_gating_top_k_softmax_v2_tiling.h"

using namespace std;

class MoeGatingTopKSoftmaxV2Tiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeGatingTopKSoftmaxV2Tiling TearDown" << std::endl;
    }
};

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_001) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 101031;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_002) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                                {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 101111;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_003) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 102011;
    string expectTilingData =
        "32212254720024 137438953488 4294967320 17179869185 10720238373312 137438953484 0 51539607553 4294967308 "
        "34359738376 51539607552 0 8 0 0 0 0 0 0 0 0 0 0 10857677334400 339302421440 137438954104 137438953504 "
        "549755813952 17179869186 8589934598 4294972352 2714419331072 0 0 0 0 0 274877907200 8589934720 137438953488 "
        "137438953504 549755813952 17179869186 8589934598 4294967424 68719476736 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_004) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 103012;
    string expectTilingData = "68719476992 137438953488 34359738376 30064771109 4294967300 30064771073 17179869191 187647121184085 224 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_005) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 103032;
    string expectTilingData = "68719476992 137438953488 34359738376 30064771109 4294967300 30064771073 17179869191 187647121184085 224 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_006) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_007) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24}, {1, 24}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_008) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24}, {1, 24}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_009) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 5000}, {1, 24, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 1, 5000}, {1, 1, 5000}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(5000)},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0000;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces, 70);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_010) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                                {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 102111;
    string expectTilingData = "0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces, 47);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_011) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 24, 16}, {1, 24, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 24, 7500}, {1, 24, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                                {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 102011;
    string expectTilingData =
        "32212254720024 137438953488 4294967320 17179869185 10720238373312 137438953484 1 51539607553 4294967308 "
        "34359738376 51539607552 0 8 0 0 0 0 0 0 0 0 0 0 10857677334400 339302421440 137438954104 137438953504 "
        "549755813952 17179869186 8589934598 4294972352 2714419331072 0 0 0 0 0 274877907200 8589934720 137438953488 "
        "137438953504 549755813952 17179869186 8589934598 4294967424 68719476736 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MoeGatingTopKSoftmaxV2Tiling, moe_gating_top_k_softmax_v2_tiling_012) {
    optiling::MoeGatingTopKSoftmaxV2CompileInfo compileInfo = {40, 196608};
    std::string socVersion = "Ascend910B";
    uint64_t coreNum = 40;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("MoeGatingTopKSoftmaxV2",
                                              {
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 256, 8}, {1, 256,8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 256, 8}, {1, 256, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 256, 16}, {1, 256, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                                {"renorm", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                                {"output_softmax_result_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                                              },
                                              &compileInfo, socVersion, coreNum, ubSize);
    uint64_t expectTilingKey = 103012;
    string expectTilingData = "68719476992 137438953488 34359738376 30064771109 4294967300 30064771073 17179869191 187647121184085 4294967520 ";
    std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}