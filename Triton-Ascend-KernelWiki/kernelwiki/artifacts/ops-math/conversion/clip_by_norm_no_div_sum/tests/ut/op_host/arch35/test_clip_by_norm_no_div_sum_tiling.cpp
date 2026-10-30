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
 * \file test_clip_by_norm_no_div_sum_tiling.cpp
 * \brief ClipByNormNoDivSum Tiling UT — 4入1出 broadcast
 */
#include "conversion/clip_by_norm_no_div_sum/op_host/arch35/clip_by_norm_no_div_sum_tiling_arch35.h"
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class ClipByNormNoDivSumTilingTest : public testing::Test {};

// ============================================================
// PadAndSqueeze (2 cases)
// ============================================================

// case1: 同形 2D — 无 broadcast, 4 输入 shape=(4,8), 1 输出 shape=(4,8)
TEST_F(ClipByNormNoDivSumTilingTest, case1_same_shape_2d)
{
    optiling::ClipByNormNoDivSumCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "ClipByNormNoDivSum",
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND},
          {{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND},
          {{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND},
          {{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect = "2 4 1 4 1 1 1 0 2 52416 1 1 4 8 4 1 1 1 4 8 1 1 4 8 1 1 4 8 1 1 4 8 0 0 8 1 0 0 8 1 0 0 8 1 0 0 8 "
                    "1 1 1 4 8 0 0 8 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case2: broadcast — x0=(4,1), x1=(1,8), x2=(4,1), x3=(1,8) → max_bro=(4,8)
TEST_F(ClipByNormNoDivSumTilingTest, case2_broadcast_2d)
{
    optiling::ClipByNormNoDivSumCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "ClipByNormNoDivSum",
        {{{{{4, 1}, {4, 1}}, DT_FLOAT, FORMAT_ND},
          {{{1, 8}, {1, 8}}, DT_FLOAT, FORMAT_ND},
          {{{4, 1}, {4, 1}}, DT_FLOAT, FORMAT_ND},
          {{{1, 8}, {1, 8}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect = "2 4 1 4 1 1 1 0 2 52416 1 1 4 8 4 1 1 1 4 1 1 1 1 8 1 1 4 1 1 1 1 8 0 0 1 0 0 0 0 1 0 0 1 0 0 0 0 "
                    "1 1 1 4 8 0 0 8 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// FindSplitAxis (2 cases)
// ============================================================

// case3: 末轴溢出 — shape=(100000,), FP32 P=5, per_buf_bytes=52416, per_buf_elems=13104
TEST_F(ClipByNormNoDivSumTilingTest, case3_last_dim_split_fp32)
{
    optiling::ClipByNormNoDivSumCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "ClipByNormNoDivSum",
        {{{{{100000}, {100000}}, DT_FLOAT, FORMAT_ND},
          {{{100000}, {100000}}, DT_FLOAT, FORMAT_ND},
          {{{100000}, {100000}}, DT_FLOAT, FORMAT_ND},
          {{{100000}, {100000}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{100000}, {100000}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect = "3 13104 8 8272 8 8 1 0 1 52416 1 1 1 100000 4 1 1 1 1 100000 1 1 1 100000 1 1 1 100000 1 1 1 "
                    "100000 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 1 1 1 100000 0 0 0 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case4: 切分轴非末轴 — shape=(64,32,8), FP32 P=5, per_buf_elems=13104, 64*256>13104 → axis=0
TEST_F(ClipByNormNoDivSumTilingTest, case4_mid_dim_split_fp32)
{
    optiling::ClipByNormNoDivSumCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "ClipByNormNoDivSum",
        {{{{{64, 32, 8}, {64, 32, 8}}, DT_FLOAT, FORMAT_ND},
          {{{64, 32, 8}, {64, 32, 8}}, DT_FLOAT, FORMAT_ND},
          {{{64, 32, 8}, {64, 32, 8}}, DT_FLOAT, FORMAT_ND},
          {{{64, 32, 8}, {64, 32, 8}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{64, 32, 8}, {64, 32, 8}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect =
        "1 51 2 13 2 2 1 0 3 52416 1 64 32 8 4 1 1 64 32 8 1 64 32 8 1 64 32 8 1 64 32 8 0 256 8 1 0 256 8 1 "
        "0 256 8 1 0 256 8 1 1 64 32 8 0 256 8 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// MultiCoreSplit (1 case)
// ============================================================

// case5: 多核 — shape=(7,13,200,100), axis=2
TEST_F(ClipByNormNoDivSumTilingTest, case5_multi_core)
{
    optiling::ClipByNormNoDivSumCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "ClipByNormNoDivSum",
        {{{{{7, 13, 200, 100}, {7, 13, 200, 100}}, DT_FLOAT, FORMAT_ND},
          {{{7, 13, 200, 100}, {7, 13, 200, 100}}, DT_FLOAT, FORMAT_ND},
          {{{7, 13, 200, 100}, {7, 13, 200, 100}}, DT_FLOAT, FORMAT_ND},
          {{{7, 13, 200, 100}, {7, 13, 200, 100}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{7, 13, 200, 100}, {7, 13, 200, 100}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect =
        "2 131 2 69 56 182 3 14 4 52416 7 13 200 100 4 1 7 13 200 100 7 13 200 100 7 13 200 100 7 13 200 100 260000 "
        "20000 100 1 260000 20000 100 1 260000 20000 100 1 260000 20000 100 1 7 13 200 100 260000 20000 100 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// dtype (2 cases)
// ============================================================

// case6: FP16 — P=5, per_buf_bytes=52416, per_buf_elems=26208
TEST_F(ClipByNormNoDivSumTilingTest, case6_fp16_split)
{
    optiling::ClipByNormNoDivSumCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "ClipByNormNoDivSum",
        {{{{{50000}, {50000}}, DT_FLOAT16, FORMAT_ND},
          {{{50000}, {50000}}, DT_FLOAT16, FORMAT_ND},
          {{{50000}, {50000}}, DT_FLOAT16, FORMAT_ND},
          {{{50000}, {50000}}, DT_FLOAT16, FORMAT_ND}}},
        {{{{{50000}, {50000}}, DT_FLOAT16, FORMAT_ND}}}, &ci);
    string expect = "3 26208 2 23792 2 2 1 0 1 52416 1 1 1 50000 4 1 1 1 1 50000 1 1 1 50000 1 1 1 50000 1 1 1 50000 0 "
                    "0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 1 1 1 50000 0 0 0 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case7: 标量 broadcast — x0=(), x1=(100,), x2=(), x3=(100,) → max_bro=(100,)
TEST_F(ClipByNormNoDivSumTilingTest, case7_scalar_broadcast)
{
    optiling::ClipByNormNoDivSumCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "ClipByNormNoDivSum",
        {{{{{}, {}}, DT_FLOAT, FORMAT_ND},
          {{{100}, {100}}, DT_FLOAT, FORMAT_ND},
          {{{}, {}}, DT_FLOAT, FORMAT_ND},
          {{{100}, {100}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{100}, {100}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect = "3 100 1 100 1 1 1 0 1 52416 1 1 1 100 4 1 1 1 1 1 1 1 1 100 1 1 1 1 1 1 1 100 0 0 0 0 0 0 0 1 0 "
                    "0 0 0 0 0 0 1 1 1 1 100 0 0 0 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// 4D broadcast + RANK_8 (2 cases)
// ============================================================

// case8: 4D 多维 broadcast — x0=(4,1,1,1), x1=(1,8,1,1), x2=(1,1,16,1), x3=(1,1,1,32) → max_bro=(4,8,16,32)
TEST_F(ClipByNormNoDivSumTilingTest, case8_4d_broadcast)
{
    optiling::ClipByNormNoDivSumCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "ClipByNormNoDivSum",
        {{{{{4, 1, 1, 1}, {4, 1, 1, 1}}, DT_FLOAT, FORMAT_ND},
          {{{1, 8, 1, 1}, {1, 8, 1, 1}}, DT_FLOAT, FORMAT_ND},
          {{{1, 1, 16, 1}, {1, 1, 16, 1}}, DT_FLOAT, FORMAT_ND},
          {{{1, 1, 1, 32}, {1, 1, 1, 32}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8, 16, 32}, {4, 8, 16, 32}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect =
        "0 3 2 1 2 2 1 0 4 52416 4 8 16 32 4 1 4 1 1 1 1 8 1 1 1 1 16 1 1 1 1 32 1 0 0 0 0 1 0 0 0 0 1 0 0 0 "
        "0 1 4 8 16 32 4096 512 32 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case9: 5D rank>4 → RANK_8 — shape=(3,5,7,11,13)
TEST_F(ClipByNormNoDivSumTilingTest, case9_rank8_5d)
{
    optiling::ClipByNormNoDivSumCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "ClipByNormNoDivSum",
        {{{{{3, 5, 7, 11, 13}, {3, 5, 7, 11, 13}}, DT_FLOAT, FORMAT_ND},
          {{{3, 5, 7, 11, 13}, {3, 5, 7, 11, 13}}, DT_FLOAT, FORMAT_ND},
          {{{3, 5, 7, 11, 13}, {3, 5, 7, 11, 13}}, DT_FLOAT, FORMAT_ND},
          {{{3, 5, 7, 11, 13}, {3, 5, 7, 11, 13}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{3, 5, 7, 11, 13}, {3, 5, 7, 11, 13}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect =
        "3 2 2 1 2 2 1 0 5 52416 1 1 1 3 5 7 11 13 4 1 1 1 1 3 5 7 11 13 1 1 1 3 5 7 11 13 1 1 1 3 5 7 11 13 "
        "1 1 1 3 5 7 11 13 0 0 0 5005 1001 143 13 1 0 0 0 5005 1001 143 13 1 0 0 0 5005 1001 143 13 1 0 0 0 "
        "5005 1001 143 13 1 1 1 1 3 5 7 11 13 0 0 0 5005 1001 143 13 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 1, expect, {{16777216}});
}
