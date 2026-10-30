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

class UnpackInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "UnpackInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "UnpackInfershape TearDown" << std::endl;
    }
};

TEST_F(UnpackInfershape, unpack_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Unpack",
        {
            // input info
            {{{2, 3000, 1, 64}, {2, 3000, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // output info
            {{{3000, 1, 64}, {3000, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3000, 1, 64}, {3000, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // attr
            {"num", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3000, 1, 64}, {3000, 1, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}