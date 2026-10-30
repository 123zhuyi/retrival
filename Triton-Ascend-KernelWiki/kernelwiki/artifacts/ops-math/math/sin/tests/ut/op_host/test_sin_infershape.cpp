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
 * \file test_sin_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class sin : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "sin Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "sin Proto Test TearDown" << std::endl;
  }
};

TEST_F(sin, sin_infershape_same_test){
    gert::StorageShape shape = {{4, 3, 4}, {4, 3, 4}};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Sin",
        { x },
        { y });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

