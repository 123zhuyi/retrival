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
 * \file test_stateless_bernoulli_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_bernoulli_tiling_arch35.h"

class StatelessBernoulliTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessBernoulliTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessBernoulliTest TearDown" << std::endl;
  }
};

TEST_F(StatelessBernoulliTiling, stateless_bernoulli_test_0)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 253952};
    vector<int64_t> shapeValue = {1};
    vector<int64_t> seedValue = {2};
    vector<int64_t> offsetValue = {8};

    gert::TilingContextPara tilingContextPara(
        "StatelessBernoulli",
        {{{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},},
        {{{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo);

    uint64_t expectTilingKey = 100;
    string expectTilingData = "1 1 2 8 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(
        tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
