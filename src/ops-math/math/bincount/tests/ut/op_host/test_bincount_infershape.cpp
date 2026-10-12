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

class bincount : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "bincount SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "bincount TearDown" << std::endl;
    }
};

TEST_F(bincount, bincount_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Bincount",
        {
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(bincount, bincount_infershape_test2)
{
    std::vector<int64_t> inputSizeValues = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "Bincount",
        {
            {{{1, 3, 4}, {1, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
