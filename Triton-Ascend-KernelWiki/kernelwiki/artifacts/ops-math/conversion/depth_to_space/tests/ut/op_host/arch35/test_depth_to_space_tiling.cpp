/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_depth_to_space_tiling.cpp
 * \brief Unit tests for DepthToSpace tiling
 */
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "../../../../op_host/arch35/depth_to_space_tiling_arch35.h"

using namespace std;
using namespace ge;

class DepthToSpaceTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DepthToSpaceTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "DepthToSpaceTiling TearDown" << std::endl; }
};

// Test case 1: NHWC format with DCR mode (default scenario)
// Input shape: [1, 2, 2, 8], block_size=2
// Output shape: [1, 4, 4, 2]
TEST_F(DepthToSpaceTiling, test_nhwc_dcr_float16_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    // TODO: Update these values after running the test to get actual tiling results
    uint64_t expectTilingKey = 10001;
    string expectTilingData = "4 0 0 0 0 0 0 32 1 0 253952 1 0 0 0 0 0 0 0 2 2 2 4 0 0 0 0 2 2 2 4 0 0 0 0 0 2 1 3 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 1 3 2 4 1 2 2 2 4 1 2 2 2 4 1 0 2 2 4 1 0 2 "
                              "2 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 2 2 4 1 0 2 2 4 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 2: NCHW format with DCR mode
// Input shape: [1, 8, 2, 2], block_size=2
// Output shape: [1, 2, 4, 4]
TEST_F(DepthToSpaceTiling, test_nchw_dcr_float16_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 8, 2, 2}, {1, 8, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{1, 2, 4, 4}, {1, 2, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    // TODO: Update these values after running the test to get actual tiling results
    uint64_t expectTilingKey = 10001;
    string expectTilingData = "4 0 0 0 0 0 0 32 1 0 253952 1 0 0 0 0 0 0 0 2 2 4 2 0 0 0 0 4 2 2 2 0 0 0 0 2 0 3 1 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 3 1 4 2 1 2 2 4 2 1 4 2 2 2 1 2 2 0 2 1 0 2 "
                              "2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 0 2 1 0 2 2 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 3: NHWC format with CRD mode
// Input shape: [1, 2, 2, 8], block_size=2
// Output shape: [1, 4, 4, 2]
TEST_F(DepthToSpaceTiling, test_nhwc_crd_float16_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("CRD")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    // TODO: Update these values after running the test to get actual tiling results
    uint64_t expectTilingKey = 10001;
    string expectTilingData = "5 0 0 0 0 0 0 32 1 0 253952 1 0 0 0 0 0 0 0 2 2 2 2 2 0 0 0 2 2 2 2 2 0 0 0 0 3 1 4 2 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 3 1 4 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 0 2 2 "
                              "2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0 2 2 2 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 4: NCHW format with CRD mode
// Input shape: [1, 8, 2, 2], block_size=2
// Output shape: [1, 2, 4, 4]
TEST_F(DepthToSpaceTiling, test_nchw_crd_float16_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 8, 2, 2}, {1, 8, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{1, 2, 4, 4}, {1, 2, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("CRD")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    // TODO: Update these values after running the test to get actual tiling results
    uint64_t expectTilingKey = 10001;
    string expectTilingData = "5 0 0 0 0 0 0 32 1 0 253952 1 0 0 0 0 0 0 0 2 2 2 2 2 0 0 0 2 2 2 2 2 0 0 0 0 3 1 4 2 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 3 1 4 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 0 2 2 "
                              "2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0 2 2 2 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Test case 5: Failure case - invalid block_size (block_size < 2)
TEST_F(DepthToSpaceTiling, test_invalid_block_size_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test case 6: Failure case - depth not divisible by block_size^2
// Input shape: [1, 2, 2, 7], block_size=2
// C=7, block_size^2=4, 7 % 4 != 0
TEST_F(DepthToSpaceTiling, test_invalid_depth_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 7}, {1, 2, 2, 7}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 1}, {1, 4, 4, 1}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Failure case: input format NHWC and data_format attr NCHW mismatch, expect failure
TEST_F(DepthToSpaceTiling, test_format_mismatch_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Failure case: x and y have different storage formats (NHWC vs NCHW), expect failure
TEST_F(DepthToSpaceTiling, test_xy_format_mismatch_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 4, 4}, {1, 2, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Failure case: input shape is 3D (not 4D), expect failure
TEST_F(DepthToSpaceTiling, test_invalid_dims_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 8}, {1, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 2}, {1, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Failure case: data_format attr is invalid string (not NCHW or NHWC), expect failure
TEST_F(DepthToSpaceTiling, test_invalid_data_format_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("ND"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Failure case: mode attr is invalid string (not DCR or CRD), expect failure
TEST_F(DepthToSpaceTiling, test_invalid_mode_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("INVALID")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
