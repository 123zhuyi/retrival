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
 * \file test_reduce_std_with_mean_tiling.cpp
 * \brief Unit tests for ReduceStdWithMean host-side tiling function
 *
 * Coverage targets:
 *   - Normal tiling: fp16/fp32/bf16, various shapes
 *   - Edge cases: small reduce dim, large reduce dim, multi-dim input
 *   - Error cases: unsupported dtype
 *   - Attribute handling: correction, eps, invert defaults
 *
 * TilingKey mapping (REDUCE_STD_SCH_* in reduce_std_with_mean_tiling_key.h):
 *   TilingKey 0: REDUCE_STD_SCH_FP16 (fp16)
 *   TilingKey 1: REDUCE_STD_SCH_FP32 (fp32)
 *   TilingKey 2: REDUCE_STD_SCH_BF16 (bf16)
 */

#include <gtest/gtest.h>

#include "reduce_std_with_mean_tiling.h"
#include "../../../op_kernel/reduce_std_with_mean_tiling_data.h"
#include "../../../op_kernel/reduce_std_with_mean_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

// ==========================================================================
// Test Fixture
// ==========================================================================

class ReduceStdWithMeanTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ReduceStdWithMeanTilingTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "ReduceStdWithMeanTilingTest TearDown" << endl; }
};

// ==========================================================================
// Normal Cases — fp32 (TilingKey=1, REDUCE_STD_SCH_FP32)
// ==========================================================================

/// Basic: fp32, 2D [64, 128]
TEST_F(ReduceStdWithMeanTilingTest, fp32_2d)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{64, 128}, {128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{64, 128}, {128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                {
                                    {{{64, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 1, std::vector<size_t>{0});
}

/// fp32, 3D [8, 16, 256]
TEST_F(ReduceStdWithMeanTilingTest, fp32_3d)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{8, 16, 256}, {4096, 256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{8, 16, 256}, {4096, 256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                {
                                    {{{8, 16, 1}, {16, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 1, std::vector<size_t>{0});
}

/// fp32, single-element reduce dim [32, 1], reduce=1
TEST_F(ReduceStdWithMeanTilingTest, fp32_reduce_dim_1)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{32, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{32, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                {
                                    {{{32, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 1, std::vector<size_t>{0});
}

/// fp32, non-reduce=1 [1, 256]
TEST_F(ReduceStdWithMeanTilingTest, fp32_nonreduce_1)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{1, 256}, {256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{1, 256}, {256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                {
                                    {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 1, std::vector<size_t>{0});
}

// ==========================================================================
// Normal Cases — fp16 (TilingKey=0, REDUCE_STD_SCH_FP16)
// ==========================================================================

/// Basic: fp16, 2D [64, 128], reduce last dim (128)
TEST_F(ReduceStdWithMeanTilingTest, fp16_2d_default)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{64, 128}, {128, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // self
                                    {{{64, 128}, {128, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // mean (expanded)
                                },
                                {
                                    {{{64, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // output
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}

/// fp16, large 2D [1024, 2048], reduce last dim
TEST_F(ReduceStdWithMeanTilingTest, fp16_large_2d)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{1024, 2048}, {2048, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{1024, 2048}, {2048, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{1024, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}

/// fp16, multi-dim 3D [4, 8, 64], reduce last dim (64), non-reduce=32
TEST_F(ReduceStdWithMeanTilingTest, fp16_3d_multi_dim)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{4, 8, 64}, {512, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{4, 8, 64}, {512, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{4, 8, 1}, {8, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}

/// fp16, 4D [2, 3, 4, 32], reduce last dim (32)
TEST_F(ReduceStdWithMeanTilingTest, fp16_4d)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{2, 3, 4, 32}, {384, 128, 32, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{2, 3, 4, 32}, {384, 128, 32, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{2, 3, 4, 1}, {12, 4, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}

/// fp16, 1D input [256] — rank-1, non-reduce=1, reduce=256
TEST_F(ReduceStdWithMeanTilingTest, fp16_1d)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{256}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{256}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}

// ==========================================================================
// Normal Cases — bf16 (TilingKey=2, now supported)
// ==========================================================================

/// Basic: bf16, 2D [64, 128]
TEST_F(ReduceStdWithMeanTilingTest, bf16_2d)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{64, 128}, {128, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                    {{{64, 128}, {128, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                },
                                {
                                    {{{64, 1}, {1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 2, std::vector<size_t>{0});
}

// ==========================================================================
// Error Cases
// ==========================================================================

/// Unsupported dtype (INT32) — should return GRAPH_FAILED
TEST_F(ReduceStdWithMeanTilingTest, unsupported_dtype_int32)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{64, 128}, {128, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                    {{{64, 128}, {128, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                },
                                {
                                    {{{64, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

/// Unsupported dtype (INT8)
TEST_F(ReduceStdWithMeanTilingTest, unsupported_dtype_int8)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{64, 128}, {128, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                    {{{64, 128}, {128, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                },
                                {
                                    {{{64, 1}, {1, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

// ==========================================================================
// Boundary Cases
// ==========================================================================

/// Empty tensor: totalNum=0 → should succeed with blockDim=1
TEST_F(ReduceStdWithMeanTilingTest, empty_tensor)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{0, 128}, {128, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{0, 128}, {128, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{0, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}

/// All-zero shape element (totalNum=0 case)
TEST_F(ReduceStdWithMeanTilingTest, zero_dim_element)
{
    ReduceStdWithMeanCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMean",
                                {
                                    {{{4, 0}, {0, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{4, 0}, {0, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{4, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}
