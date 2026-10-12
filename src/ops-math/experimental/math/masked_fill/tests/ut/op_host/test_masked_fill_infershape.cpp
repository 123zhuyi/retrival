/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;

class MaskedFillInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 测试1：输入x为标量（形状{}），输出形状与x一致
TEST_F(MaskedFillInfershapeTest, scalar_input)
{
    gert::InfershapeContextPara para(
        "MaskedFill",
        {
            {{{}, {}}, DT_FLOAT, FORMAT_ND}, // x: 标量
            {{{}, {}}, DT_BOOL, FORMAT_ND},  // mask: 标量（与x同形）
            {{{}, {}}, DT_FLOAT, FORMAT_ND}  // value: 标量
        },
        {
            {{{}, {}}, DT_FLOAT, FORMAT_ND} // 输出：预期标量
        });
    std::vector<std::vector<int64_t>> expect_shape = {{}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expect_shape);
}

// 测试2：输入x为1D张量（形状(5)），输出形状与x一致
TEST_F(MaskedFillInfershapeTest, 1d_input)
{
    gert::InfershapeContextPara para(
        "MaskedFill",
        {
            {{{5}, {5}}, DT_INT32, FORMAT_ND}, // x: (5)
            {{{5}, {5}}, DT_BOOL, FORMAT_ND},  // mask: (5)（与x同形）
            {{{}, {}}, DT_INT32, FORMAT_ND}    // value: 标量
        },
        {
            {{{5}, {5}}, DT_INT32, FORMAT_ND} // 输出：预期(5)
        });
    std::vector<std::vector<int64_t>> expect_shape = {{5}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expect_shape);
}

// 测试3：输入x为2D张量（形状(2,3)），输出形状与x一致
TEST_F(MaskedFillInfershapeTest, 2d_input)
{
    gert::InfershapeContextPara para(
        "MaskedFill",
        {
            {{{2, 3}, {2, 3}}, DT_BF16, FORMAT_ND}, // x: (2,3)
            {{{2, 3}, {2, 3}}, DT_BOOL, FORMAT_ND}, // mask: (2,3)（与x同形）
            {{{}, {}}, DT_BF16, FORMAT_ND}          // value: 标量
        },
        {
            {{{2, 3}, {2, 3}}, DT_BF16, FORMAT_ND} // 输出：预期(2,3)
        });
    std::vector<std::vector<int64_t>> expect_shape = {{2, 3}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expect_shape);
}

// 测试4：输入x为3D张量（形状(4,2,3)），输出形状与x一致
TEST_F(MaskedFillInfershapeTest, 3d_input)
{
    gert::InfershapeContextPara para(
        "MaskedFill",
        {
            {{{4, 2, 3}, {4, 2, 3}}, DT_UINT8, FORMAT_ND}, // x: (4,2,3)
            {{{4, 2, 3}}, DT_BOOL, FORMAT_ND},             // mask: (4,2,3)（与x同形）
            {{{}, {}}, DT_UINT8, FORMAT_ND}                // value: 标量
        },
        {
            {{{4, 2, 3}, {4, 2, 3}}, DT_UINT8, FORMAT_ND} // 输出：预期(4,2,3)
        });
    std::vector<std::vector<int64_t>> expect_shape = {{4, 2, 3}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expect_shape);
}
s