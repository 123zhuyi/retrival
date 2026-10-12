/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_real_div_tiling.cpp
 * \brief
 */

#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/real_div/op_host/arch35/real_div_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class RealDivTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RealDivTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RealDivTiling TearDown" << std::endl;
    }
};

TEST_F(RealDivTiling, real_div_test_bf16)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "RealDiv",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 281492156580096 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RealDivTiling, real_div_test_float16)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "RealDiv",
        {
            {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 11, 12, 14, 6}, {8, 11, 12, 14, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "88704 150323856640 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RealDivTiling, real_div_test_int32)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "RealDiv",
        {
            {{{7, 56, 2, 3, 11}, {7, 56, 2, 3, 11}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{7, 56, 2, 3, 11}, {7, 56, 2, 3, 11}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{7, 56, 2, 3, 11}, {7, 56, 2, 3, 11}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "25872 146028888448 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RealDivTiling, real_div_test_float32)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "RealDiv",
        {
            {{{1, 1, 789, 1}, {3, 5, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 789, 1}, {3, 5, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 5, 789, 5}, {3, 5, 789, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "12884901888 8589934592 1 8 1 1 8 10880 8 15 789 5 0 0 0 0 0 3945 5 1 0 0 0 0 0 0 0 0 0 15 1 5 0 0 0 0 0 15 1 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 1 0 0 0 0 0 5 0 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RealDivTiling, real_div_test_bool)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    gert::TilingContextPara tilingContextPara(
        "RealDiv",
        {
            {{{7, 56, 2, 3, 11}, {7, 56, 2, 3, 11}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{7, 56, 2, 3, 11}, {7, 56, 2, 3, 11}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{7, 56, 2, 3, 11}, {7, 56, 2, 3, 11}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "25872 111669150208 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
