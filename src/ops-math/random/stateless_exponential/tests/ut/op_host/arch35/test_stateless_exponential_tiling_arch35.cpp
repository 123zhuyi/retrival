/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_stateless_exponential_tiling_arch35.cpp
 * \brief StatelessExponential tiling UT (ascend950 / SIMT)
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_exponential_tiling_arch35.h"

class StatelessExponentialTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StatelessExponentialTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StatelessExponentialTilingTest TearDown" << std::endl; }
};

// FP32 -> TilingKey = 3
TEST_F(StatelessExponentialTilingTest, one_dim_float)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 196608};
    int64_t seedValue = 5;
    int64_t offsetValue = 4;

    gert::TilingContextPara tilingContextPara("StatelessExponential",
                                              {
                                                  {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
                                              },
                                              {
                                                  {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"lambd", Ops::Math::AnyValue::CreateFrom<float>(1.0f)},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// FP16 -> TilingKey = 1
TEST_F(StatelessExponentialTilingTest, one_dim_float16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 196608};
    int64_t seedValue = 7;
    int64_t offsetValue = 8;

    gert::TilingContextPara tilingContextPara("StatelessExponential",
                                              {
                                                  {{{300}, {300}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
                                              },
                                              {
                                                  {{{300}, {300}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"lambd", Ops::Math::AnyValue::CreateFrom<float>(1.0f)},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// BF16 -> TilingKey = 2
TEST_F(StatelessExponentialTilingTest, one_dim_bfloat16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 196608};
    int64_t seedValue = 99;
    int64_t offsetValue = 12;

    gert::TilingContextPara tilingContextPara("StatelessExponential",
                                              {
                                                  {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
                                              },
                                              {
                                                  {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"lambd", Ops::Math::AnyValue::CreateFrom<float>(2.0f)},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 2;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 2-D shape, FP32 -> TilingKey = 3, exercises multi-dim coalescing in splitBlocks
TEST_F(StatelessExponentialTilingTest, two_dim_float)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 196608};
    int64_t seedValue = 1;
    int64_t offsetValue = 16;

    gert::TilingContextPara tilingContextPara("StatelessExponential",
                                              {
                                                  {{{4, 64}, {4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
                                              },
                                              {
                                                  {{{4, 64}, {4, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"lambd", Ops::Math::AnyValue::CreateFrom<float>(0.5f)},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// lambd == 0 -> attrCheckRules fails -> GRAPH_FAILED
TEST_F(StatelessExponentialTilingTest, lambd_zero_failed)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 196608};
    int64_t seedValue = 5;
    int64_t offsetValue = 4;

    gert::TilingContextPara tilingContextPara("StatelessExponential",
                                              {
                                                  {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
                                              },
                                              {
                                                  {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"lambd", Ops::Math::AnyValue::CreateFrom<float>(0.0f)},
                                              },
                                              &compileInfo);

    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, expectWorkspaces);
}
