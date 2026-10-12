/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../../op_kernel/mul_no_nan_tiling_data.h"
#include "../../../../op_host/arch22/mul_no_nan_tiling.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class MulNoNanTilingArch22Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MulNoNanTilingArch22Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MulNoNanTilingArch22Test TearDown" << std::endl; }
};

// Case 1: fp16 same shape.
TEST_F(MulNoNanTilingArch22Test, same_shape_fp16)
{
    optiling::MulNoNanArch22CompileInfo compileInfo;
    gert::TilingContextPara para("MulNoNan",
                                 {
                                     {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {});
}

// Case 2: fp32 same shape.
TEST_F(MulNoNanTilingArch22Test, same_shape_fp32)
{
    optiling::MulNoNanArch22CompileInfo compileInfo;
    gert::TilingContextPara para("MulNoNan",
                                 {
                                     {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {});
}

// Case 3: int32 same shape.
TEST_F(MulNoNanTilingArch22Test, same_shape_int32_rejected)
{
    optiling::MulNoNanArch22CompileInfo compileInfo;
    gert::TilingContextPara para("MulNoNan",
                                 {
                                     {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_FAILED, false, 0, false, "", {});
}

// Case 4: bf16 same shape.
TEST_F(MulNoNanTilingArch22Test, same_shape_bf16)
{
    optiling::MulNoNanArch22CompileInfo compileInfo;
    gert::TilingContextPara para("MulNoNan",
                                 {
                                     {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {});
}

// Case 5: large tensor multi-core, fp32.
TEST_F(MulNoNanTilingArch22Test, large_tensor_fp32)
{
    optiling::MulNoNanArch22CompileInfo compileInfo;
    gert::TilingContextPara para("MulNoNan",
                                 {
                                     {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {});
}

// Case 6: dtype mismatch -> failure.
TEST_F(MulNoNanTilingArch22Test, dtype_mismatch_failed)
{
    optiling::MulNoNanArch22CompileInfo compileInfo;
    gert::TilingContextPara para("MulNoNan",
                                 {
                                     {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_FAILED, false, 0, false, "", {});
}
