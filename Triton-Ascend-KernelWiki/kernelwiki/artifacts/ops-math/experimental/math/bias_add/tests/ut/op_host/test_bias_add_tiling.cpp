/**
 * This file is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
 * All Rights Reserved.
 *
 * Author (account):
 * - Yang Zhenze <@gcw_5x5Ew5Ms>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_bias_add_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_kernel/bias_add_tiling_data.h"
#include "../../../op_kernel/bias_add_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class BiasAddTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BiasAddTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BiasAddTiling TearDown" << std::endl; }
};

// Mirrors the layout of the (anonymous-namespace) optiling::BiasAddCompileInfo so the
// TilingParse stage has a correctly sized buffer to write coreNum/ubSize into. The actual
// platform values used by tiling come from the faker (coreNum=64, ubSize=262144 below).
struct BiasAddCompileInfo {
    std::vector<int64_t> broadcastBiasShape;
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
    bool isUnknownRank = false;
};

static gert::TilingContextPara::OpAttr DataFormatAttr(const std::string& fmt)
{
    return gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>(fmt));
}

// ---------------------------------------------------------------------------
// Success cases: cover the first-order schedule families (SEGMENT / ALIGNED /
// UNALIGNED-vector / TINY). The expected tilingKey is the schMode template key.
// Workspace is 0 for all paths (BiasAdd uses no GM workspace).
// ---------------------------------------------------------------------------

// innerSize > 1 (NCHW with H*W tail) -> SEGMENT family -> BASE schMode (key 0).
TEST_F(BiasAddTiling, bias_add_tiling_segment_nchw)
{
    BiasAddCompileInfo compileInfo;
    gert::TilingContextPara para("BiasAdd",
                                 {
                                     {{{3, 1024, 32, 32}, {3, 1024, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                     {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{3, 1024, 32, 32}, {3, 1024, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                 },
                                 {DataFormatAttr("NCHW")}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, BIAS_ADD_TPL_SCH_MODE_BASE, std::vector<size_t>{0});
}

// innerSize == 1, channel*sizeof aligned, not tiny -> ALIGNED family -> BASE schMode (key 0).
TEST_F(BiasAddTiling, bias_add_tiling_aligned_nhwc)
{
    BiasAddCompileInfo compileInfo;
    gert::TilingContextPara para("BiasAdd",
                                 {
                                     {{{100, 8}, {100, 8}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{100, 8}, {100, 8}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {DataFormatAttr("NHWC")}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, BIAS_ADD_TPL_SCH_MODE_BASE, std::vector<size_t>{0});
}

// innerSize == 1, unaligned channel, >=90 elems, rows<=255 -> UNALIGNED vector broadcast (key 2).
TEST_F(BiasAddTiling, bias_add_tiling_unaligned_vector_nhwc)
{
    BiasAddCompileInfo compileInfo;
    gert::TilingContextPara para("BiasAdd",
                                 {
                                     {{{2, 3, 3, 7}, {2, 3, 3, 7}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{7}, {7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 3, 3, 7}, {2, 3, 3, 7}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {DataFormatAttr("NHWC")}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, BIAS_ADD_TPL_SCH_MODE_THIN_TINY_VECTOR_BROADCAST, std::vector<size_t>{0});
}

// Sub-floor element count, not vector-eligible -> TINY noqueue (key 1).
TEST_F(BiasAddTiling, bias_add_tiling_tiny_noqueue)
{
    BiasAddCompileInfo compileInfo;
    gert::TilingContextPara para("BiasAdd",
                                 {
                                     {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {DataFormatAttr("NHWC")}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, BIAS_ADD_TPL_SCH_MODE_TINY_NOQUEUE, std::vector<size_t>{0});
}

// ---------------------------------------------------------------------------
// Failure cases: input validation in GetShapeAttrsInfo / ResolveLayoutInfo.
// ---------------------------------------------------------------------------

// bias length != resolved C (NCHW: C is dim 1).
TEST_F(BiasAddTiling, bias_add_tiling_fail_nchw_channel_mismatch)
{
    BiasAddCompileInfo compileInfo;
    gert::TilingContextPara para("BiasAdd",
                                 {
                                     {{{1, 8, 4, 4}, {1, 8, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                     {{{7}, {7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 8, 4, 4}, {1, 8, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                 },
                                 {DataFormatAttr("NCHW")}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{0});
}

// bias length != resolved C (NHWC: C is last dim).
TEST_F(BiasAddTiling, bias_add_tiling_fail_nhwc_channel_mismatch)
{
    BiasAddCompileInfo compileInfo;
    gert::TilingContextPara para("BiasAdd",
                                 {
                                     {{{1, 4, 4, 8}, {1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{7}, {7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 4, 4, 8}, {1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {DataFormatAttr("NHWC")}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{0});
}

// x dtype != bias dtype.
TEST_F(BiasAddTiling, bias_add_tiling_fail_dtype_mismatch)
{
    BiasAddCompileInfo compileInfo;
    gert::TilingContextPara para("BiasAdd",
                                 {
                                     {{{1, 8, 4, 4}, {1, 8, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                     {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 8, 4, 4}, {1, 8, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                 },
                                 {DataFormatAttr("NCHW")}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{0});
}

// bias rank != 1.
TEST_F(BiasAddTiling, bias_add_tiling_fail_bias_not_1d)
{
    BiasAddCompileInfo compileInfo;
    gert::TilingContextPara para("BiasAdd",
                                 {
                                     {{{1, 8, 4, 4}, {1, 8, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                     {{{1, 8}, {1, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 8, 4, 4}, {1, 8, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                 },
                                 {DataFormatAttr("NCHW")}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{0});
}

// bias length != resolved C (NCDHW: C is dim 1).
TEST_F(BiasAddTiling, bias_add_tiling_fail_ncdhw_channel_mismatch)
{
    BiasAddCompileInfo compileInfo;
    gert::TilingContextPara para("BiasAdd",
                                 {
                                     {{{1, 6, 2, 3, 4}, {1, 6, 2, 3, 4}}, ge::DT_INT32, ge::FORMAT_NCDHW},
                                     {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 6, 2, 3, 4}, {1, 6, 2, 3, 4}}, ge::DT_INT32, ge::FORMAT_NCDHW},
                                 },
                                 {DataFormatAttr("NCDHW")}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{0});
}
