/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SelectV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SelectV2Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SelectV2Test TearDown" << std::endl;
  }
};

TEST_F(SelectV2Test, select_v2_infer_shape_0) {
    gert::StorageShape shape = {{2, 100}, {2, 100}};
    gert::StorageShape yShape = {{2, 100}, {2, 100}};
    gert::InfershapeContextPara::TensorDescription condition(shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x1(shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x2(shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "SelectV2",
        { condition, x1, x2 },
        { y });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
