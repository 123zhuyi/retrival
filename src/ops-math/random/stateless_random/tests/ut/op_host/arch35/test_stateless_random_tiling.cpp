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
 * \file test_stateless_random_tiling.cpp
 * \brief Tiling unit test for StatelessRandom operator
 */

#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_random_tiling_arch35.h"

class StatelessRandomTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StatelessRandomTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StatelessRandomTiling TearDown" << std::endl;
    }
};

TEST_F(StatelessRandomTiling, stateless_random_test_0)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 253952};
    vector<int64_t> shapeValue = {256};
    vector<int64_t> seedValue = {42};
    vector<int64_t> offsetValue = {0};
    vector<int64_t> fromValue = {0};
    vector<int64_t> toValue = {100};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandom",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, fromValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, toValue.data()},
        },
        {{{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_INT32))}}, &compileInfo);

    uint64_t expectTilingKey = 100;
    string expectTilingData = "1 256 0 0 229376 4 0 0 100 0 1 256 0 1 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StatelessRandomTiling, stateless_random_test_1)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 253952};
    vector<int64_t> shapeValue = {16, 16};
    vector<int64_t> seedValue = {123};
    vector<int64_t> offsetValue = {8};
    vector<int64_t> fromValue = {10};
    vector<int64_t> toValue = {50};

    gert::TilingContextPara tilingContextPara(
        "StatelessRandom",
        {
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, fromValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, toValue.data()},
        },
        {{{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT))}}, &compileInfo);

    uint64_t expectTilingKey = 100;
    string expectTilingData = "1 256 0 0 229376 4 0 10 40 0 1 256 0 1 256 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}