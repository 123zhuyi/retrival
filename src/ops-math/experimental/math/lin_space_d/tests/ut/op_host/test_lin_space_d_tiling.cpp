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
#include "lin_space_d_tiling.h"
#include "../../../op_kernel/lin_space_d_tiling_data.h"
#include "../../../op_kernel/lin_space_d_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class LinSpaceDTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "LinSpaceDTiling SetUp" << endl; }
    static void TearDownTestCase() { cout << "LinSpaceDTiling TearDown" << endl; }
};

// 测试float32类型输入的分块策略
TEST_F(LinSpaceDTiling, test_tiling_float32)
{
    optiling::LinSpaceDCompileInfo compileInfo; // 空结构体，无特殊配置
    gert::TilingContextPara tilingContextPara(
        "LinSpaceD",
        {// 输入0: start（float32）
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1: stop（float32）
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入2: num（int64，值为128）
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {// 输出0: 形状[128]，float32
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        &compileInfo);
    // 预期工作空间为0（根据WS_SYS_SIZE=0）
    std::vector<size_t> expectWorkspaces = {0};
    // 验证分块执行成功（具体tiling数据需根据实际算子逻辑调整）
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", expectWorkspaces);
}

// 测试bfloat16类型输入的分块策略
TEST_F(LinSpaceDTiling, test_tiling_bf16)
{
    optiling::LinSpaceDCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LinSpaceD",
        {
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND}, // start: bf16
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND}, // stop: bf16
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND} // num=64
        },
        {{{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "", expectWorkspaces);
}

// 测试不支持的数据类型组合（如start为DT_DOUBLE）
TEST_F(LinSpaceDTiling, test_tiling_invalid_dtype)
{
    optiling::LinSpaceDCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LinSpaceD",
        {{{{1}, {1}}, ge::DT_DOUBLE, ge::FORMAT_ND}, // 不支持的类型
         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    // 预期分块失败
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}