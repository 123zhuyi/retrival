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

class Asin_UT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Asin_UT SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Asin_UT TearDown" << std::endl;
    }
};

// 测试 float16 输入输出 shape 一致
TEST_F(Asin_UT, InferShapeAsin_float16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Asin",
        {
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试 float 输入输出 shape 一致
TEST_F(Asin_UT, InferShapeAsin_float)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Asin",
        {
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试 bfloat16 输入输出 shape 一致
TEST_F(Asin_UT, InferShapeAsin_bfloat16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Asin",
        {
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试多维 shape
TEST_F(Asin_UT, InferShapeAsin_multidim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Asin",
        {
            {{{2, 16, 32, 16}, {2, 16, 32, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 16, 32, 16},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试一维 shape
TEST_F(Asin_UT, InferShapeAsin_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Asin",
        {
            {{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{100},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试空 tensor
TEST_F(Asin_UT, InferShapeAsin_empty)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Asin",
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{0},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
