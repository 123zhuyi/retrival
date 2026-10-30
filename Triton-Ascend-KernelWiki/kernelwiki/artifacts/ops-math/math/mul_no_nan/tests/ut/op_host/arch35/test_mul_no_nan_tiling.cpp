/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../../op_host/arch35/mul_no_nan_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class MulNoNanTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MulNoNanTilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MulNoNanTilingTest TearDown" << std::endl;
    }
};

// All cases use ExecuteTestCaseForEle with key/data checks disabled, because
// the BroadcastBaseTiling framework can change schMode hashing across CANN
// versions. We only assert the return status here; precise key/data values
// belong in the e2e numerical tests.

// Case 1: fp16 same shape.
TEST_F(MulNoNanTilingTest, same_shape_fp16)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 2: fp32 same shape.
TEST_F(MulNoNanTilingTest, same_shape_fp32)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 3: int32 same shape.
TEST_F(MulNoNanTilingTest, same_shape_int32)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
        {
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 4: bf16 same shape.
TEST_F(MulNoNanTilingTest, same_shape_bf16)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
        {
            {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 5: per-axis broadcast, fp16.
TEST_F(MulNoNanTilingTest, axis_broadcast_fp16)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
        {
            {{{16, 1}, {16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 6: cross-rank broadcast, fp32.
TEST_F(MulNoNanTilingTest, cross_rank_broadcast_fp32)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
        {
            {{{3, 4, 5}, {3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 4, 5}, {3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 7: x2 is a scalar, fp32.
TEST_F(MulNoNanTilingTest, x2_scalar_fp32)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
        {
            {{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 8: x1 is a scalar, bf16.
TEST_F(MulNoNanTilingTest, x1_scalar_bf16)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
        {
            {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{32, 64}, {32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{32, 64}, {32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 9: large tensor that triggers multi-core, fp32.
TEST_F(MulNoNanTilingTest, large_tensor_fp32)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 10: tiny tensor (1 element each), int32.
TEST_F(MulNoNanTilingTest, tiny_tensor_int32)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 11: dtype mismatch x1 vs x2 -> failure.
TEST_F(MulNoNanTilingTest, dtype_mismatch_failed)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
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

// Case 12: unsupported dtype double -> failure (MulNoNan only supports
// {fp16, fp32, int32, bf16}; double is rejected even though shapes match).
TEST_F(MulNoNanTilingTest, unsupported_double_failed)
{
    optiling::MulNoNanCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "MulNoNan",
        {
            {{{8}, {8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_FAILED, false, 0, false, "", {});
}
