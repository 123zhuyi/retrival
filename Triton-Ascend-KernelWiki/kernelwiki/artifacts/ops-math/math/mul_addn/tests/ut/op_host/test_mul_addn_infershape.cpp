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
#include "base/registry/op_impl_space_registry_v2.h"

class MulAddn : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MulAddn Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MulAddn Proto Test TearDown" << std::endl;
    }
};

TEST_F(MulAddn, infershape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MulAddn",
        {// input info
         {{{1500, 512, 1}, {1500, 512, 1}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1500, 1, 128}, {1500, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND}},
        {
            // output info
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            // attr
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(6)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1500, 512, 128},};                                                                            // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}
