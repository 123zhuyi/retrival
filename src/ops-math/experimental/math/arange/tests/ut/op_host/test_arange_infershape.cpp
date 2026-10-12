/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_arange_infershape.cpp
 * \brief Arange op_host InferShape UT —— 验证契约：输出 1 维、dim0=-1（动态未知），不读 start/end/step 数值
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ArangeInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ArangeInfershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ArangeInfershape TearDown" << std::endl; }
};

// 核心契约：InferShape 给输出 1 维、dim0 = -1（动态未知），N 真值由 caller 经 out 张量传入
TEST_F(ArangeInfershape, output_is_1d_dynamic_unknown_fp32)
{
    gert::InfershapeContextPara infershapeContextPara("Arange",
                                                      {
                                                          {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // start
                                                          {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // end
                                                          {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}  // step
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} // out: 待推断
                                                      });
    // 预期输出 1 维、dim0 = -1（无论输入数值）
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dim0=-1 与 dtype 无关：fp16 同样给 -1
TEST_F(ArangeInfershape, output_is_1d_dynamic_unknown_fp16)
{
    gert::InfershapeContextPara infershapeContextPara("Arange",
                                                      {{{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                       {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                       {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// int32 同样给 -1
TEST_F(ArangeInfershape, output_is_1d_dynamic_unknown_int32)
{
    gert::InfershapeContextPara infershapeContextPara("Arange",
                                                      {{{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                       {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 全 dtype 覆盖：InferShape 的 dim0=-1 契约对【全 8 dtype】（含窄整型 int8/uint8/int16 + bf16/int64）
// 一律成立，与 dtype 无关。
TEST_F(ArangeInfershape, output_is_1d_dynamic_unknown_all_dtypes)
{
    const ge::DataType dtypes[] = {ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_INT64};
    for (ge::DataType dt : dtypes) {
        gert::InfershapeContextPara infershapeContextPara("Arange",
                                                          {
                                                              {{{1}, {1}}, dt, ge::FORMAT_ND}, // start
                                                              {{{1}, {1}}, dt, ge::FORMAT_ND}, // end
                                                              {{{1}, {1}}, dt, ge::FORMAT_ND}  // step
                                                          },
                                                          {
                                                              {{{}, {}}, dt, ge::FORMAT_ND} // out: 待推断
                                                          });
        std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
        ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
    }
}
