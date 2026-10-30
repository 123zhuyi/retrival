/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class DiagPartInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DiagPart InferShape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DiagPart InferShape TearDown" << std::endl; }
};

// Test 2D -> 1D
TEST_F(DiagPartInferShape, diag_part_infershape_2d_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DiagPart",
        {
            {{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test 4D -> 2D
TEST_F(DiagPartInferShape, diag_part_infershape_4d_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DiagPart",
        {
            {{{2, 3, 2, 3}, {2, 3, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test different dtypes
TEST_F(DiagPartInferShape, diag_part_infershape_int32_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DiagPart",
        {
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}