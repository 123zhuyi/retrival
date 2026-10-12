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

using namespace ge;

class SignBitsPackTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SignBitsPackTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SignBitsPackTest TearDown" << std::endl; }
};

// 正常1D输入 N=16, size=1 → 输出 [1, 2]
TEST_F(SignBitsPackTest, sign_bits_pack_infershape_1d_test)
{
    gert::InfershapeContextPara infershapeContextPara("SignBitsPack",
                                                      {
                                                          {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"size", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 正常1D输入 N=64, size=8 → 输出 [8, 1]
TEST_F(SignBitsPackTest, sign_bits_pack_infershape_size8_test)
{
    gert::InfershapeContextPara infershapeContextPara("SignBitsPack",
                                                      {
                                                          {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"size", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 空Tensor N=0, size=1 → 输出 [1, 0]
TEST_F(SignBitsPackTest, sign_bits_pack_infershape_empty_test)
{
    gert::InfershapeContextPara infershapeContextPara("SignBitsPack",
                                                      {
                                                          {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"size", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 0},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 非8倍数 N=17, size=1 → packed=3, 输出 [1, 3]
TEST_F(SignBitsPackTest, sign_bits_pack_infershape_non8_test)
{
    gert::InfershapeContextPara infershapeContextPara("SignBitsPack",
                                                      {
                                                          {{{17}, {17}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"size", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
