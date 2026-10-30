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
 * \file test_stateless_sample_multinomial_tiling.cpp
 * \brief StatelessSampleMultinomial tiling UT
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_sample_multinomial_tiling.h"

class StatelessSampleMultinomialTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StatelessSampleMultinomialTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StatelessSampleMultinomialTilingTest TearDown" << std::endl;
    }
};

TEST_F(StatelessSampleMultinomialTilingTest, one_dim_float)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    int64_t seedValue = 12345;
    int64_t offsetValue = 0;

    gert::TilingContextPara tilingContextPara(
        "StatelessSampleMultinomial",
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
        },
        {
            {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"num_samples", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(StatelessSampleMultinomialTilingTest, two_dim_float16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    int64_t seedValue = 7;
    int64_t offsetValue = 4;

    gert::TilingContextPara tilingContextPara(
        "StatelessSampleMultinomial",
        {
            {{{3, 10}, {3, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"num_samples", Ops::Math::AnyValue::CreateFrom<int64_t>(5)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(StatelessSampleMultinomialTilingTest, two_dim_bf16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    int64_t seedValue = 99;
    int64_t offsetValue = 8;

    gert::TilingContextPara tilingContextPara(
        "StatelessSampleMultinomial",
        {
            {{{2, 64}, {2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
        },
        {
            {{{2, 16}, {2, 16}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"num_samples", Ops::Math::AnyValue::CreateFrom<int64_t>(16)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(StatelessSampleMultinomialTilingTest, offset_multiple_of_four)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    int64_t seedValue = 1;
    int64_t offsetValue = 4;

    gert::TilingContextPara tilingContextPara(
        "StatelessSampleMultinomial",
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
        },
        {
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"num_samples", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
