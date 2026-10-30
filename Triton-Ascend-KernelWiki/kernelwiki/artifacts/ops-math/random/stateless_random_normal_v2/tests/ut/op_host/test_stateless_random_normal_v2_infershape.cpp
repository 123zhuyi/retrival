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
 * \file test_stateless_random_normal_v2_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace std;

class StatelessRandomNormalV2Infershape : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessRandomNormalV2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessRandomNormalV2 TearDown" << std::endl;
  }
};

TEST_F(StatelessRandomNormalV2Infershape, stateless_random_normal_v2_infershape_test1)
{
    vector<int64_t> shapeValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessRandomNormalV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND,},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND,},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND,},
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StatelessRandomNormalV2Infershape, stateless_random_normal_v2_infershape_test2)
{
    vector<int32_t> shapeValue = {10};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessRandomNormalV2",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND,},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND,},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND,},
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}