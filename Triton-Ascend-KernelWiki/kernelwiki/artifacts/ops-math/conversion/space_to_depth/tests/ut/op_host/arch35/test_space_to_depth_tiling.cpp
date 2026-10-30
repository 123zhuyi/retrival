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
 * \file test_space_to_depth_tiling.cpp
 * \brief Unit tests for SpaceToDepth tiling
 */
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "../../../../op_host/arch35/space_to_depth_tiling_arch35.h"

using namespace std;
using namespace ge;

class SpaceToDepthTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SpaceToDepthTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SpaceToDepthTiling TearDown" << std::endl; }
};

// NHWC format with valid block_size=2, FP16 dtype, expect tiling success
// Input: [1, 4, 4, 2], block_size=2 → Output: [1, 2, 2, 8]
TEST_F(SpaceToDepthTiling, test_nhwc_float16_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

// NCHW format with valid block_size=2, FP16 dtype, expect tiling success
// Input: [1, 2, 4, 4], block_size=2 → Output: [1, 8, 2, 2]
TEST_F(SpaceToDepthTiling, test_nchw_float16_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 2, 4, 4}, {1, 2, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{1, 8, 2, 2}, {1, 8, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

// NHWC format with valid block_size=2, FP32 dtype, expect tiling success
// Input: [1, 4, 4, 2], block_size=2 → Output: [1, 2, 2, 8]
TEST_F(SpaceToDepthTiling, test_nhwc_float32_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

// NCHW format with valid block_size=2, FP32 dtype, expect tiling success
// Input: [1, 2, 4, 4], block_size=2 → Output: [1, 8, 2, 2]
TEST_F(SpaceToDepthTiling, test_nchw_float32_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 2, 4, 4}, {1, 2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{1, 8, 2, 2}, {1, 8, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

// NHWC format with block_size=3, expect tiling success
// Input: [1, 6, 6, 2], block_size=3 → Output: [1, 2, 2, 18]
TEST_F(SpaceToDepthTiling, test_nhwc_block_size_3_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 6, 6, 2}, {1, 6, 6, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 2, 18}, {1, 2, 2, 18}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

// Invalid dims: 3D input shape, expect failure
TEST_F(SpaceToDepthTiling, test_invalid_dims_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 4, 2}, {1, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 8}, {1, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Invalid dims: 5D output shape, expect failure
TEST_F(SpaceToDepthTiling, test_invalid_output_dims_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 1, 2, 2, 8}, {1, 1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Format mismatch: x is NHWC, y is NCHW, expect failure
TEST_F(SpaceToDepthTiling, test_xy_format_mismatch_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 8, 2, 2}, {1, 8, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Unsupported format: FORMAT_ND, expect failure
TEST_F(SpaceToDepthTiling, test_unsupported_format_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("ND"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Invalid data_format attr (not NCHW or NHWC), expect failure
TEST_F(SpaceToDepthTiling, test_invalid_data_format_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW_V2_C"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Format mismatch between input format and data_format attr: NHWC input with NCHW attr, expect failure
TEST_F(SpaceToDepthTiling, test_format_attr_mismatch_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Format mismatch between input format and data_format attr: NCHW input with NHWC attr, expect failure
TEST_F(SpaceToDepthTiling, test_nchw_nhwc_attr_mismatch_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 2, 4, 4}, {1, 2, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{1, 8, 2, 2}, {1, 8, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Invalid block_size = 0, expect failure
TEST_F(SpaceToDepthTiling, test_block_size_zero_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Invalid block_size = -1, expect failure
TEST_F(SpaceToDepthTiling, test_block_size_negative_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// H not divisible by block_size (NHWC), expect failure
// Input: [1, 5, 4, 2] (H=5 not divisible by block_size=2)
TEST_F(SpaceToDepthTiling, test_h_not_divisible_nhwc_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 5, 4, 2}, {1, 5, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// W not divisible by block_size (NHWC), expect failure
// Input: [1, 4, 5, 2] (W=5 not divisible by block_size=2)
TEST_F(SpaceToDepthTiling, test_w_not_divisible_nhwc_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 4, 5, 2}, {1, 4, 5, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// H not divisible by block_size (NCHW), expect failure
// Input: [1, 2, 5, 4] (H=5 not divisible by block_size=2)
TEST_F(SpaceToDepthTiling, test_h_not_divisible_nchw_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 2, 5, 4}, {1, 2, 5, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{1, 8, 2, 2}, {1, 8, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// W not divisible by block_size (NCHW), expect failure
// Input: [1, 2, 4, 5] (W=5 not divisible by block_size=2)
TEST_F(SpaceToDepthTiling, test_w_not_divisible_nchw_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 2, 4, 5}, {1, 2, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{1, 8, 2, 2}, {1, 8, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// NCHW format with block_size=4, BF16 dtype, expect tiling success
// Input: [1, 1, 8, 8], block_size=4 → Output: [1, 16, 2, 2]
TEST_F(SpaceToDepthTiling, test_nchw_bf16_block_size_4_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "SpaceToDepth",
        {
            {{{1, 1, 8, 8}, {1, 1, 8, 8}}, ge::DT_BF16, ge::FORMAT_NCHW},
        },
        {
            {{{1, 16, 2, 2}, {1, 16, 2, 2}}, ge::DT_BF16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(4)),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}
