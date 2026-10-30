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

#include <iostream>
#include <gtest/gtest.h>
#include "reflection_pad1d_v2_tiling.h"
#include "../../../op_kernel/reflection_pad1d_v2_tiling_data.h"
#include "../../../op_kernel/reflection_pad1d_v2_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class ReflectionPad1dV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ReflectionPad1dV2Tiling SetUp" << endl; }
    static void TearDownTestCase() { cout << "ReflectionPad1dV2Tiling TearDown" << endl; }
};

// 测试用例1：正常场景 - float32类型 + 1D输入 + 合法padding（左3，右4）
TEST_F(ReflectionPad1dV2Tiling, test_tiling_float32_1d)
{
    // 空编译信息结构体
    optiling::ReflectionPad1dV2CompileInfo compileInfo;
    // 构造分块上下文参数
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad1dV2", // 算子名，与注册名一致
        {                    // 输入0：x（1D形状[8]，float32类型，ND格式）
         {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1：padding（1D形状[2]，int32类型，ND格式，对应[3,4]）
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {// 输出0：形状[15]（8+3+4），float32类型，ND格式
         {{{15}, {15}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        &compileInfo); // 编译信息指针
    // 预期工作空间：根据tiling逻辑，大填充场景有非0工作空间
    std::vector<size_t> expectWorkspaces = {15 * 1 * 4}; // outWSize * ncPerCore * dtypeBytes
    // 执行测试用例：预期分块成功（GRAPH_SUCCESS）
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", expectWorkspaces);
}

// 测试用例2：正常场景 - bf16类型 + 2D输入 + 合法padding（左2，右3）
TEST_F(ReflectionPad1dV2Tiling, test_tiling_bf16_2d)
{
    optiling::ReflectionPad1dV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（2D形状[2,8]，bf16类型，ND格式）
         {{{2, 8}, {2, 8}}, ge::DT_BF16, ge::FORMAT_ND},
         // 输入1：padding（1D形状[2]，int64类型，ND格式，对应[2,3]）
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND}},
        {// 输出0：形状[2,13]（8+2+3），bf16类型，ND格式
         {{{2, 13}, {2, 13}}, ge::DT_BF16, ge::FORMAT_ND}},
        &compileInfo);
    // 预期工作空间：2D场景下ncPerCore=1，工作空间=13*1*2
    std::vector<size_t> expectWorkspaces = {13 * 2};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", expectWorkspaces);
}

// 测试用例3：正常场景 - float16类型 + 3D输入 + 合法padding（左1，右2）
TEST_F(ReflectionPad1dV2Tiling, test_tiling_float16_3d)
{
    optiling::ReflectionPad1dV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（3D形状[2,3,8]，float16类型，ND格式）
         {{{2, 3, 8}, {2, 3, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         // 输入1：padding（1D形状[2]，int32类型，ND格式，对应[1,2]）
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {// 输出0：形状[2,3,11]（8+1+2），float16类型，ND格式
         {{{2, 3, 11}, {2, 3, 11}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        &compileInfo);
    // 预期工作空间：3D场景下ncPerCore=1（totalNc=6 ≤ coreNum=64），工作空间=11*1*2
    std::vector<size_t> expectWorkspaces = {11 * 2};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", expectWorkspaces);
}

// 测试用例4：异常场景 - 不支持的数据类型（DT_DOUBLE，不在DATATYPE_BYTES_MAP中）
TEST_F(ReflectionPad1dV2Tiling, test_tiling_invalid_dtype_double)
{
    optiling::ReflectionPad1dV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（1D形状[8]，double类型，不支持）
         {{{8}, {8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
         // 输入1：padding（合法1D[2]，int32类型）
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {// 输出0：形状[15]，double类型（无效）
         {{{15}, {15}}, ge::DT_DOUBLE, ge::FORMAT_ND}},
        &compileInfo);
    // 预期分块失败（GRAPH_FAILED），无预期工作空间
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// 测试用例5：异常场景 - 输入维度无效（0D，不满足1D~3D要求）
TEST_F(ReflectionPad1dV2Tiling, test_tiling_invalid_0d_input)
{
    optiling::ReflectionPad1dV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（0D形状，无效维度）
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1：padding（合法1D[2]，int32类型）
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {// 输出0：无效形状
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        &compileInfo);
    // 预期分块失败
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// 测试用例6：异常场景 - padding数据类型无效（DT_FLOAT，非int32/int64）
TEST_F(ReflectionPad1dV2Tiling, test_tiling_invalid_pad_dtype_float)
{
    optiling::ReflectionPad1dV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（合法1D[8]，float32类型）
         {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1：padding（1D[2]，float类型，无效）
         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {// 输出0：形状[15]，float32类型
         {{{15}, {15}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        &compileInfo);
    // 预期分块失败
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// 测试用例7：异常场景 - padding长度无效（1，非2）
TEST_F(ReflectionPad1dV2Tiling, test_tiling_invalid_pad_len_1)
{
    optiling::ReflectionPad1dV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（合法1D[8]，float32类型）
         {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1：padding（1D[1]，int32类型，长度无效）
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {// 输出0：形状[15]，float32类型
         {{{15}, {15}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        &compileInfo);
    // 预期分块失败
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}