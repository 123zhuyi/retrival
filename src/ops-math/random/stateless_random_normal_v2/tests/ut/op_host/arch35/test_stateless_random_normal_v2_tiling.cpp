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
 * \file test_stateless_random_normal_v2_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_random_normal_v2_tiling_arch35.h"

class StatelessRandomNormalV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessRandomNormalV2Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessRandomNormalV2Test TearDown" << std::endl;
  }
};

TEST_F(StatelessRandomNormalV2Tiling, stateless_random_normal_v2_test_tiling_1)
{
    optiling::StatelessRandomNormalV2CompileInfo compileInfo = {40, 196608};
    vector<uint64_t> keyValue = {1};
    vector<int64_t> counterValue = {8, 9};
    vector<int64_t> algsetValue = {1};
    gert::TilingContextPara tilingContextPara(
        "StatelessRandomNormalV2",
        {
            {{{32, 512}, {32, 512}}, ge::DT_INT64, ge::FORMAT_ND,},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, keyValue.data()},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue.data()},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, algsetValue.data()},
        },
        {
            {{{32, 512}, {32, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "1099511627840 54563264528640 1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StatelessRandomNormalV2Tiling, stateless_random_normal_v2_test_tiling_2)
{
    optiling::StatelessRandomNormalV2CompileInfo compileInfo = {40, 196608};
    vector<int64_t> keyValue = {1};
    vector<int64_t> counterValue = {8, 9};
    vector<int64_t> algsetValue = {1};
    gert::TilingContextPara tilingContextPara(
        "StatelessRandomNormalV2",
        {
            {{{32, 512}, {32, 512}}, ge::DT_INT64, ge::FORMAT_ND,},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, keyValue.data()},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue.data()},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, algsetValue.data()},
        },
        {
            {{{32, 512}, {32, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "1099511627840 54563264528640 1 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}