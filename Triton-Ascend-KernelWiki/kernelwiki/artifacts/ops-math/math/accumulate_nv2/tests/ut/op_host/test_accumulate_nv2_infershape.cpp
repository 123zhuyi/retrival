/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;

class AccumulateNV2InferShape : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AccumulateNV2InferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AccumulateNV2InferShape TearDown" << std::endl;
    }
};

// 2x float32 inputs, shape {8, 1024} => output shape {8, 1024}
TEST_F(AccumulateNV2InferShape, float32_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AccumulateNV2",
        {
            {{{8, 1024}, {8, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 1024}, {8, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 1024},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 1x float16 input, shape {16} => output shape {16}
TEST_F(AccumulateNV2InferShape, float16_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AccumulateNV2",
        {
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// scalar-like input, shape {1} => output shape {1}
TEST_F(AccumulateNV2InferShape, scalar_like)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AccumulateNV2",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
