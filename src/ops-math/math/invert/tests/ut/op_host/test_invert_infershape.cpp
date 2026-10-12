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

class InvertInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Invert SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Invert TearDown" << std::endl;
    }
};

TEST_F(InvertInfershape, invert_infershape_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Invert",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
