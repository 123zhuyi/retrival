/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_drop_out_v3_grad_tiling.cpp
 * \brief DropOutV3Grad tiling UT —— 成功路径 + 各类非法输入校验（dtype / scale 标量）。
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../../op_host/arch35/drop_out_v3_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class DropOutV3GradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DropOutV3GradTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "DropOutV3GradTilingTest TearDown" << std::endl; }
};

namespace {
// {coreNum, ubEle, ubSize}
optiling::DropOutV3GradCompileInfo MakeCompileInfo() { return optiling::DropOutV3GradCompileInfo{64, 0, 262144}; }
} // namespace

// ===================== 成功用例 =====================

// case 1: grad_y=float32 / mask=uint8 / scale=float32，常规 shape
TEST_F(DropOutV3GradTilingTest, drop_out_v3_grad_tiling_float32_uint8)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription grad_y({{4095, 1, 3}, {4095, 1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{1536}, {1536}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription scale({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription grad_x({{4095, 1, 3}, {4095, 1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutV3Grad", {grad_y, mask, scale}, {grad_x}, &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// case 2: grad_y=float16 / mask=uint8 / scale=float32
TEST_F(DropOutV3GradTilingTest, drop_out_v3_grad_tiling_float16_uint8)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription grad_y({{2, 1024}, {2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{256}, {256}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription scale({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription grad_x({{2, 1024}, {2, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutV3Grad", {grad_y, mask, scale}, {grad_x}, &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// case 3: grad_y=bfloat16 / mask=uint1 / scale=float32
TEST_F(DropOutV3GradTilingTest, drop_out_v3_grad_tiling_bf16_uint1)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription grad_y({{8192}, {8192}}, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{8192}, {8192}}, ge::DT_UINT1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription scale({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription grad_x({{8192}, {8192}}, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutV3Grad", {grad_y, mask, scale}, {grad_x}, &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// case 4: 小 shape（单核即可覆盖）
TEST_F(DropOutV3GradTilingTest, drop_out_v3_grad_tiling_small_shape)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription grad_y({{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{32}, {32}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription scale({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription grad_x({{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutV3Grad", {grad_y, mask, scale}, {grad_x}, &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ===================== 非法用例 =====================

// case 5: grad_y dtype 非法（DT_INT32，不在 {float,float16,bf16}）→ GRAPH_FAILED
TEST_F(DropOutV3GradTilingTest, drop_out_v3_grad_tiling_invalid_grad_y_dtype)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription grad_y({{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{32}, {32}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription scale({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription grad_x({{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutV3Grad", {grad_y, mask, scale}, {grad_x}, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// case 6: grad_y dtype 非法（DT_DOUBLE）→ GRAPH_FAILED
TEST_F(DropOutV3GradTilingTest, drop_out_v3_grad_tiling_invalid_grad_y_double)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription grad_y({{256}, {256}}, ge::DT_DOUBLE, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{32}, {32}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription scale({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription grad_x({{256}, {256}}, ge::DT_DOUBLE, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutV3Grad", {grad_y, mask, scale}, {grad_x}, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// case 7: mask dtype 非法（DT_INT8，只支持 uint8/uint1）→ GRAPH_FAILED
TEST_F(DropOutV3GradTilingTest, drop_out_v3_grad_tiling_invalid_mask_dtype)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription grad_y({{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{32}, {32}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription scale({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription grad_x({{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutV3Grad", {grad_y, mask, scale}, {grad_x}, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// case 8: mask dtype 非法（DT_FLOAT）→ GRAPH_FAILED
TEST_F(DropOutV3GradTilingTest, drop_out_v3_grad_tiling_invalid_mask_float)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription grad_y({{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{32}, {32}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription scale({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription grad_x({{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutV3Grad", {grad_y, mask, scale}, {grad_x}, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// case 9: scale dtype 非法（DT_FLOAT16，必须 float32）→ GRAPH_FAILED（对标 torch 的核心约束）
TEST_F(DropOutV3GradTilingTest, drop_out_v3_grad_tiling_invalid_scale_float16)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription grad_y({{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{32}, {32}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription scale({{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription grad_x({{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutV3Grad", {grad_y, mask, scale}, {grad_x}, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// case 10: scale dtype 非法（DT_BF16）→ GRAPH_FAILED
TEST_F(DropOutV3GradTilingTest, drop_out_v3_grad_tiling_invalid_scale_bf16)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription grad_y({{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{32}, {32}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription scale({{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription grad_x({{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutV3Grad", {grad_y, mask, scale}, {grad_x}, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// case 11: scale 非标量（shape {2}，size!=1）→ GRAPH_FAILED
TEST_F(DropOutV3GradTilingTest, drop_out_v3_grad_tiling_invalid_scale_not_scalar)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription grad_y({{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{32}, {32}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription scale({{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription grad_x({{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutV3Grad", {grad_y, mask, scale}, {grad_x}, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
