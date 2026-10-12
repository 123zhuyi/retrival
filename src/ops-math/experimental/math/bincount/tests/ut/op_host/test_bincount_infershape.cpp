/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

// 本算子 InferShape：bincount 输出长度依赖 array 取值（aclnn 流程下 bins 由调用方预分配），
// 故输出统一推导为 1 维动态 {-1}。算子原型输入为 array/size/weights（weights 可选）。
class test_bincount_infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "test_bincount_infershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "test_bincount_infershape TearDown" << std::endl; }
};

// 计数：array int32 + size（无 weights），输出动态 {-1}
TEST_F(test_bincount_infershape, infershape_count_int32)
{
    gert::InfershapeContextPara infershapeContextPara("Bincount",
                                                      {
                                                          {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND}, // array
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},   // size
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // bins
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 加权：array int32 + size + weights float，输出动态 {-1}
TEST_F(test_bincount_infershape, infershape_weighted)
{
    gert::InfershapeContextPara infershapeContextPara("Bincount",
                                                      {
                                                          {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND}, // array
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},   // size
                                                          {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // weights
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // bins
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// array int8 + size -> bins int64
TEST_F(test_bincount_infershape, infershape_int8_out64)
{
    gert::InfershapeContextPara infershapeContextPara("Bincount",
                                                      {
                                                          {{{2048}, {2048}}, ge::DT_INT8, ge::FORMAT_ND}, // array
                                                          {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},      // size
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}, // bins
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
