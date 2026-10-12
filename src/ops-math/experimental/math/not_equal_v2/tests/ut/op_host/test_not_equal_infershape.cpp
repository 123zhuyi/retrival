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

class NotEqualV2InferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NotEqualV2InferShape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NotEqualV2InferShape TearDown" << std::endl; }
};

TEST_F(NotEqualV2InferShape, not_equal_v2_infershape_same_shape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "NotEqualV2",
        {
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(NotEqualV2InferShape, not_equal_v2_infershape_broadcast)
{
    gert::InfershapeContextPara infershapeContextPara(
        "NotEqualV2",
        {
            {{{2, 3, 1}, {2, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 3, 5}, {1, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(NotEqualV2InferShape, not_equal_v2_infershape_dynamic_broadcast)
{
    gert::InfershapeContextPara infershapeContextPara(
        "NotEqualV2",
        {
            {{{3, -1, 1}, {3, -1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, -1, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(NotEqualV2InferShape, not_equal_v2_infershape_broadcast_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "NotEqualV2",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
