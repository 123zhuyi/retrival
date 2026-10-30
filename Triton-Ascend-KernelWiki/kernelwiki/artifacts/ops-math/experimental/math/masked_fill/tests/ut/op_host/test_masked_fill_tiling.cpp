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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../op_kernel/masked_fill_tiling_data.h"
#include "../op_kernel/masked_fill_tiling_key.h"

using namespace optiling;

class MaskedFillTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 测试1：支持的数据类型（float32），形状兼容（x与mask同形），工作空间为0
TEST_F(MaskedFillTilingTest, valid_dtype_shape)
{
    MaskedFillCompileInfo compile_info;
    gert::TilingContextPara para(
        "MaskedFill",
        {
            {{{2, 3}, {2, 3}}, DT_FLOAT, FORMAT_ND}, // x: (2,3)
            {{{2, 3}, {2, 3}}, DT_BOOL, FORMAT_ND},  // mask: (2,3)（兼容）
            {{{}, {}}, DT_FLOAT, FORMAT_ND}          // value: 标量
        },
        {
            {{{2, 3}, {2, 3}}, DT_FLOAT, FORMAT_ND} // 输出：(2,3)
        },
        &compile_info);
    std::vector<size_t> expect_workspace = {0}; // 预期工作空间大小为0
    ExecuteTestCase(para, GRAPH_SUCCESS, 0, "", expect_workspace);
}

// 测试2：支持的数据类型（bfloat16），形状广播兼容（mask维度为1）
TEST_F(MaskedFillTilingTest, broadcast_shape)
{
    MaskedFillCompileInfo compile_info;
    gert::TilingContextPara para(
        "MaskedFill",
        {
            {{{2, 3}, {2, 3}}, DT_BF16, FORMAT_ND}, // x: (2,3)
            {{{1, 3}, {1, 3}}, DT_BOOL, FORMAT_ND}, // mask: (1,3)（广播兼容）
            {{{}, {}}, DT_BF16, FORMAT_ND}          // value: 标量
        },
        {
            {{{2, 3}, {2, 3}}, DT_BF16, FORMAT_ND} // 输出：(2,3)
        },
        &compile_info);
    std::vector<size_t> expect_workspace = {0};
    ExecuteTestCase(para, GRAPH_SUCCESS, 0, "", expect_workspace);
}

// 测试3：不支持的数据类型（DT_COMPLEX64），分块失败
TEST_F(MaskedFillTilingTest, invalid_dtype)
{
    MaskedFillCompileInfo compile_info;
    gert::TilingContextPara para(
        "MaskedFill",
        {{{{2, 3}, {2, 3}}, DT_COMPLEX64, FORMAT_ND}, // 不支持的类型
         {{{2, 3}, {2, 3}}, DT_BOOL, FORMAT_ND},
         {{{}, {}}, DT_COMPLEX64, FORMAT_ND}},
        {{{{2, 3}, {2, 3}}, DT_COMPLEX64, FORMAT_ND}}, &compile_info);
    ExecuteTestCase(para, GRAPH_FAILED, 0, "", {}); // 预期分块失败
}

// 测试4：形状不兼容（mask维度超过x），分块失败
TEST_F(MaskedFillTilingTest, invalid_shape_mask_dim_exceed)
{
    MaskedFillCompileInfo compile_info;
    gert::TilingContextPara para(
        "MaskedFill",
        {{{{2, 3}, {2, 3}}, DT_INT32, FORMAT_ND},      // x: 2维
         {{{2, 3, 4}, {2, 3, 4}}, DT_BOOL, FORMAT_ND}, // mask: 3维（超过x）
         {{{}, {}}, DT_INT32, FORMAT_ND}},
        {{{{2, 3}, {2, 3}}, DT_INT32, FORMAT_ND}}, &compile_info);
    ExecuteTestCase(para, GRAPH_FAILED, 0, "", {}); // 预期分块失败
}

// 测试5：形状不兼容（维度大小不匹配且不为1），分块失败
TEST_F(MaskedFillTilingTest, invalid_shape_dim_mismatch)
{
    MaskedFillCompileInfo compile_info;
    gert::TilingContextPara para(
        "MaskedFill",
        {{{{2, 3}, {2, 3}}, DT_UINT8, FORMAT_ND}, // x: (2,3)
         {{{2, 4}, {2, 4}}, DT_BOOL, FORMAT_ND},  // mask: (2,4)（维度3≠4且不为1）
         {{{}, {}}, DT_UINT8, FORMAT_ND}},
        {{{{2, 3}, {2, 3}}, DT_UINT8, FORMAT_ND}}, &compile_info);
    ExecuteTestCase(para, GRAPH_FAILED, 0, "", {}); // 预期分块失败
}