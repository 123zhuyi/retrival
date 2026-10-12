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
 * \file test_stateless_random_uniform_v2_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace std;

class stateless_randperm : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessRandperm SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessRandperm TearDown" << std::endl;
  }
};

 	 

TEST_F(stateless_randperm, stateless_randperm_infershape_test1)
{
    vector<int64_t> shape_value = {7};
    vector<int64_t> seed_value = {0};
    vector<int64_t> offset_value = {0};
    gert::InfershapeContextPara infershapeContextPara("StatelessRandperm",
                                                      {
                                                        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, shape_value.data()},
                                                        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, seed_value.data()},
                                                        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, offset_value.data()},
                                                      },
                                                      {
                                                        {{{7}, {7}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                      {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{7}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
