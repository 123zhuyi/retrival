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
 * \file test_logical_and_tiling.cpp
 * \brief
 */

#include "../../../../op_host/arch35/logical_and_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
class LogicalAndTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LogicalAndTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LogicalAndTiling TearDown" << std::endl;
    }
};

TEST_F(LogicalAndTiling, logical_and_test_0)
{
    optiling::LogicalAndCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "LogicalAnd",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 281483566645760 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LogicalAndTiling, logical_and_test_1)
{
    optiling::LogicalAndCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "LogicalAnd",
        {
            {{{1, 9, 6, 1, 7, 10}, {1, 9, 6, 1, 7, 10}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{13, 1, 6, 2, 7, 1}, {13, 1, 6, 2, 7, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{13, 9, 6, 2, 7, 10}, {13, 9, 6, 2, 7, 10}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData =
        "25769803776 21474836480 3 3 1 1 3 43520 3 13 9 6 2 7 10 0 0 7560 840 140 70 10 1 0 0 0 0 0 0 1 9 6 1 7 10 0 0 "
        "13 1 6 2 7 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 420 70 0 10 1 0 0 84 0 14 7 1 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LogicalAndTiling, logical_and_test_failed_invalid_dtype)
{
    optiling::LogicalAndCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "LogicalAnd",
        {
            {{{4, 17, 1, 12}, {4, 17, 1, 12}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1, 1, 1, 12}, {1, 1, 1, 12}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            {{{4, 17, 1, 12}, {4, 17, 1, 12}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "11 549755814016 16 1 1 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}