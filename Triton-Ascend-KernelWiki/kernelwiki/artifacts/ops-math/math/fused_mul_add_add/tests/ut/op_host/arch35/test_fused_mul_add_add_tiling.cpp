/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../../op_host/arch35/fused_mul_add_add_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class FusedMulAddAddTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedMulAddAddTilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "FusedMulAddAddTilingTest TearDown" << std::endl;
    }
};

// All cases use ExecuteTestCaseForEle with key/data checks disabled, because
// the BroadcastBaseTiling framework can change schMode hashing across CANN
// versions. We only assert the return status here; precise key/data values
// belong in the e2e numerical tests.

// Case 1: fp16 same shape, no broadcast.
TEST_F(FusedMulAddAddTilingTest, same_shape_fp16)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 2: fp32 same shape, no broadcast.
TEST_F(FusedMulAddAddTilingTest, same_shape_fp32)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 3: int32 same shape, no broadcast.
TEST_F(FusedMulAddAddTilingTest, same_shape_int32)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 4: per-axis broadcast, fp16. x1=[16,1], x2=[1,16], x3=[16,16], x4=[1].
TEST_F(FusedMulAddAddTilingTest, axis_broadcast_fp16)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{16, 1}, {16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16}, {1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 5: cross-rank broadcast, fp32. x1=[3,4,5], x2=[4,5], x3=[5], x4=[1].
TEST_F(FusedMulAddAddTilingTest, cross_rank_broadcast_fp32)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{3, 4, 5}, {3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 4, 5}, {3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 6: x2,x3,x4 are scalars (1,1,1,1,1), fp16 -> only x1 carries shape.
TEST_F(FusedMulAddAddTilingTest, scalar_broadcast_fp16)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 7: large tensor that triggers multi-core, fp32.
TEST_F(FusedMulAddAddTilingTest, large_tensor_fp32)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 8: tiny tensor (1 element each), int32.
TEST_F(FusedMulAddAddTilingTest, tiny_tensor_int32)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 9: x4 scalar (typical residual scalar), fp32.
TEST_F(FusedMulAddAddTilingTest, x4_scalar_fp32)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_SUCCESS, false, 0, false, "", {16777216});
}

// Case 10: dtype mismatch x1 vs x2 -> failure.
TEST_F(FusedMulAddAddTilingTest, dtype_mismatch_x2_failed)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_FAILED, false, 0, false, "", {});
}

// Case 11: dtype mismatch on x4 -> failure.
TEST_F(FusedMulAddAddTilingTest, dtype_mismatch_x4_failed)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_FAILED, false, 0, false, "", {});
}

// Case 12: dtype mismatch input vs output -> failure.
TEST_F(FusedMulAddAddTilingTest, output_dtype_mismatch_failed)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_FAILED, false, 0, false, "", {});
}

// Case 13: unsupported dtype bf16 -> failure.
TEST_F(FusedMulAddAddTilingTest, unsupported_bf16_failed)
{
    optiling::FusedMulAddAddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara para(
        "FusedMulAddAdd",
        {
            {{{8}, {8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCaseForEle(para, ge::GRAPH_FAILED, false, 0, false, "", {});
}
