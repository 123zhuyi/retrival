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

class AssignAddInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AssignAddInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AssignAddInferShape TearDown" << std::endl;
    }
};

TEST_F(AssignAddInferShape, assign_sub_infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AssignAdd",
        {
            {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AssignAddInferShape, VerifyAssignAdd_001)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AssignAdd",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}