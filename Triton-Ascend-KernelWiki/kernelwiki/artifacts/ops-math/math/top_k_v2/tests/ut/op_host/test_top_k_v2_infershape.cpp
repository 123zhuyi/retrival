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
 * \file test_strided_slice_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include <vector>

using namespace std;

class TopKV2InferShape : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AscendTopKV2Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AscendTopKV2Test TearDown" << std::endl;
  }
};

TEST_F(TopKV2InferShape, top_k_v2_infershape_test1) {
    vector<int64_t> k = {8};

    gert::InfershapeContextPara infershapeContextPara(
        "TopKV2",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, k.data()},
        },
        {
            {{{10, 8}, {10, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 8}, {10, 8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"largest", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(9)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}