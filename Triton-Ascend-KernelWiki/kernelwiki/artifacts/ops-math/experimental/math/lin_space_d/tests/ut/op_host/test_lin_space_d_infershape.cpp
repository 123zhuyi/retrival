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
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class LinSpaceDInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LinSpaceDInfershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "LinSpaceDInfershape TearDown" << std::endl; }
};

// 测试正常情况：num为int64类型，输出形状为[num]
TEST_F(LinSpaceDInfershape, lin_space_d_infershape_normal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LinSpaceD",
        {// 输入0: start（类型不影响形状推理）
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1: stop（类型不影响形状推理）
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入2: num（int64类型，值为5）
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {// 输出0: 待推断形状
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    // 预期输出形状为[5]
    std::vector<std::vector<int64_t>> expectOutputShape = {{5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试错误情况：num为非int64类型（如float），形状推理失败
TEST_F(LinSpaceDInfershape, lin_space_d_infershape_invalid_num_type)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LinSpaceD",
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // num为float类型（无效）
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    // 预期推理失败
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 测试num=1的边界情况
TEST_F(LinSpaceDInfershape, lin_space_d_infershape_num_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LinSpaceD",
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND} // num=1
        },
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}