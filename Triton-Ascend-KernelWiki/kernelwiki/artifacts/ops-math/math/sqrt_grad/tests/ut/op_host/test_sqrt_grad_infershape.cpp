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
 * \file test_addcdiv_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;

class SqrtGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SqrtGradTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SqrtGradTest TearDown" << std::endl;
    }
};

TEST_F(SqrtGradTest, InferShapeSqrtGrad_001)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SqrtGrad",
        {
            {{{2, 2, 1}, {2, 2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 1}, {2, 2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SqrtGradTest, InferShapeSqrtGrad_002)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SqrtGrad",
        {
            {{{3, 4, 5, 6, -1}, {3, 4, 5, 6, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 4, 5, 6, -1}, {3, 4, 5, 6, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SqrtGradTest, infer_axis_type_with_input_is_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SqrtGrad",
        {
            {{{3, 4, 5, 6, -1}, {3, 4, 5, 6, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 4, 5, 6, -1}, {3, 4, 5, 6, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 5, 6, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}