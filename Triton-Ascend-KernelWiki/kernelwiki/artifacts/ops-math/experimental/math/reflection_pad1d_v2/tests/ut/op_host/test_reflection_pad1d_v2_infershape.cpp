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

class ReflectionPad1dV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReflectionPad1dV2Infershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ReflectionPad1dV2Infershape TearDown" << std::endl; }
};

// 测试用例1：正常场景 - 1D输入 + int32类型padding（左3，右4）
TEST_F(ReflectionPad1dV2Infershape, reflection_pad1d_v2_infershape_normal_1d_int32)
{
    // 构造形状推理上下文参数
    gert::InfershapeContextPara infershapeContextPara(
        "ReflectionPad1dV2", // 算子名，与注册名一致
        {                    // 输入0：x（1D形状，长度8，float类型，ND格式）
         {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1：padding（1D形状，长度2，int32类型，ND格式，对应[3,4]）
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {// 输出0：待推断形状（float类型，ND格式）
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    // 预期输出形状：1D，长度=8+3+4=15
    std::vector<std::vector<int64_t>> expectOutputShape = {{15}};
    // 执行测试用例：预期推理成功，输出形状匹配
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试用例2：正常场景 - 2D输入 + int64类型padding（左2，右3）
TEST_F(ReflectionPad1dV2Infershape, reflection_pad1d_v2_infershape_normal_2d_int64)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（2D形状[2,8]，int32类型，ND格式）
         {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         // 输入1：padding（1D形状[2]，int64类型，ND格式，对应[2,3]）
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}});
    // 预期输出形状：2D[2, 13]（仅最后一维变化：8+2+3=13）
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 13}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试用例3：正常场景 - 3D输入 + int32类型padding（左1，右2）
TEST_F(ReflectionPad1dV2Infershape, reflection_pad1d_v2_infershape_normal_3d_int32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（3D形状[2,3,8]，bfloat16类型，ND格式）
         {{{2, 3, 8}, {2, 3, 8}}, ge::DT_BF16, ge::FORMAT_ND},
         // 输入1：padding（1D形状[2]，int32类型，ND格式，对应[1,2]）
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}});
    // 预期输出形状：3D[2,3,11]（最后一维：8+1+2=11）
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 11}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 测试用例4：异常场景 - 输入维度为0D（无效，要求1D~3D）
TEST_F(ReflectionPad1dV2Infershape, reflection_pad1d_v2_infershape_invalid_0d_input)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（0D形状，无维度，float类型）
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1：padding（合法1D[2]，int32类型）
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    // 预期推理失败（GRAPH_FAILED），无预期输出形状
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 测试用例5：异常场景 - 输入维度为4D（无效，要求1D~3D）
TEST_F(ReflectionPad1dV2Infershape, reflection_pad1d_v2_infershape_invalid_4d_input)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（4D形状[2,3,4,8]，float类型，无效维度）
         {{{2, 3, 4, 8}, {2, 3, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1：padding（合法1D[2]，int32类型）
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    // 预期推理失败
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 测试用例6：异常场景 - padding形状为2D（无效，要求1D）
TEST_F(ReflectionPad1dV2Infershape, reflection_pad1d_v2_infershape_invalid_pad_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（合法1D[8]，float类型）
         {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1：padding（2D形状[1,2]，int32类型，无效形状）
         {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    // 预期推理失败
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 测试用例7：异常场景 - padding长度为1（无效，要求长度2）
TEST_F(ReflectionPad1dV2Infershape, reflection_pad1d_v2_infershape_invalid_pad_len_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（合法1D[8]，float类型）
         {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1：padding（1D形状[1]，int32类型，长度无效）
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    // 预期推理失败
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// 测试用例8：异常场景 - padding数据类型为float（无效，要求int32/int64）
TEST_F(ReflectionPad1dV2Infershape, reflection_pad1d_v2_infershape_invalid_pad_dtype_float)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（合法1D[8]，float类型）
         {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1：padding（合法1D[2]，float类型，数据类型无效）
         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    // 预期推理失败
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}