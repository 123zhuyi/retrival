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
 * \file test_stateless_random_normal_v3_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_random_normal_v3_tiling_arch35.h"

class StatelessRandomNormalV3Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessRandomNormalV3Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessRandomNormalV3Test TearDown" << std::endl;
  }
};

// Test 1: Valid case - scalar mean, scalar stdev, output FLOAT
TEST_F(StatelessRandomNormalV3Tiling, stateless_random_normal_v3_test_tiling_1)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {32, 512};
    vector<uint64_t> keyValue = {1};
    vector<int64_t> counterValue = {8, 9};
    float meanValue = 0.0;
    float stdevValue = 1.0;
    gert::TilingContextPara tilingContextPara(
        "StatelessRandomNormalV3",
        {
            {{{2,}, {2,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, keyValue.data()},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue.data()},
            {{{1,}, {1,}}, ge::DT_FLOAT, ge::FORMAT_ND, true, &meanValue},
            {{{1,}, {1,}}, ge::DT_FLOAT, ge::FORMAT_ND, true, &stdevValue},
        },
        {
            {{{32, 512}, {32, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 13088 0 0 0 16384 0 0 0 3 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 2: Valid case - tensor mean, tensor stdev (non-scalar), output BF16
TEST_F(StatelessRandomNormalV3Tiling, stateless_random_normal_v3_test_tiling_2)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {32, 512};
    vector<uint64_t> keyValue = {1};
    vector<int64_t> counterValue = {8, 9};
    vector<float> meanValues(16384, 0.0);
    vector<float> stdevValues(16384, 1.0);
    gert::TilingContextPara tilingContextPara(
        "StatelessRandomNormalV3",
        {
            {{{2,}, {2,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, keyValue.data()},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue.data()},
            {{{32, 512}, {32, 512}}, ge::DT_FLOAT, ge::FORMAT_ND, true, meanValues.data()},
            {{{32, 512}, {32, 512}}, ge::DT_FLOAT, ge::FORMAT_ND, true, stdevValues.data()},
        },
        {
            {{{32, 512}, {32, 512}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 9344 0 0 0 16384 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 3: Valid case - scalar mean, tensor stdev (mixed mode), output FLOAT16
TEST_F(StatelessRandomNormalV3Tiling, stateless_random_normal_v3_test_tiling_3)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {32, 512};
    vector<uint64_t> keyValue = {1};
    vector<int64_t> counterValue = {8, 9};
    float meanValue = 0.0;
    vector<float> stdevValues(16384, 1.0);
    gert::TilingContextPara tilingContextPara(
        "StatelessRandomNormalV3",
        {
            {{{2,}, {2,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, keyValue.data()},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue.data()},
            {{{1,}, {1,}}, ge::DT_FLOAT, ge::FORMAT_ND, true, &meanValue},
            {{{32, 512}, {32, 512}}, ge::DT_FLOAT, ge::FORMAT_ND, true, stdevValues.data()},
        },
        {
            {{{32, 512}, {32, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 10912 0 0 0 16384 0 0 0 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test 4: Valid case - tensor mean, scalar stdev (mixed mode), output FLOAT
TEST_F(StatelessRandomNormalV3Tiling, stateless_random_normal_v3_test_tiling_4)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {32, 512};
    vector<uint64_t> keyValue = {1};
    vector<int64_t> counterValue = {8, 9};
    vector<float> meanValues(16384, 0.0);
    float stdevValue = 1.0;
    gert::TilingContextPara tilingContextPara(
        "StatelessRandomNormalV3",
        {
            {{{2,}, {2,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, keyValue.data()},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue.data()},
            {{{32, 512}, {32, 512}}, ge::DT_FLOAT, ge::FORMAT_ND, true, meanValues.data()},
            {{{1,}, {1,}}, ge::DT_FLOAT, ge::FORMAT_ND, true, &stdevValue},
        },
        {
            {{{32, 512}, {32, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "32 512 512 10912 0 0 0 16384 0 0 0 2 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}