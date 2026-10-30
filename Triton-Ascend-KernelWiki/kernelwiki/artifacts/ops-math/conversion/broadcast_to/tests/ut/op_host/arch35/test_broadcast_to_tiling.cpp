
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
 * \file test_broadcast_to_tiling.cpp
 * \brief UT for BroadcastTo Tiling
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/broadcast_to_tiling_arch35.h"

using namespace std;
using namespace ge;

class BroadcastToTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BroadcastToTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BroadcastToTiling TearDown" << std::endl; }
};

// Test scenario: negative coreNum to cover GetHardwareInfo failure path
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_test_failed_1)
{
    optiling::BroadcastToCompileInfo compileInfo = {-2, 245760, 256, 256, 32};
    gert::StorageShape shape = {{1, 1, 5}, {1, 1, 5}};
    gert::StorageShape shape1 = {{3}, {3}};
    int32_t shapes[3] = {1, 1, 5};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11001;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: LAST_DIM_LARGE_B tiling key with 4D broadcast of UINT8 dtype
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_test_success_2)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape outshape = {{1, 1, 313, 199}, {1, 1, 313, 199}};
    gert::StorageShape shape1 = {{4}, {4}};
    int32_t shapes[4] = {1, 1, 313, 199};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_UINT8, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_UINT8, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: broadcast rule violation (input dim != 1 and != output dim)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_test_failed_3)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 313, 198}, {1, 1, 313, 198}};
    gert::StorageShape outshape = {{1, 1, 313, 199}, {1, 1, 313, 199}};
    gert::StorageShape shape1 = {{4}, {4}};
    int32_t shapes[4] = {1, 1, 313, 199};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_UINT8, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_UINT8, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 1;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: 1D shape to cover dimNum==1 path in CalcTensorSize and GetUAxisInfo - 1D B axis broadcast
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_1d_ub_brc_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1}, {1}};
    gert::StorageShape outshape = {{100}, {100}};
    gert::StorageShape shape1 = {{1}, {1}};
    int32_t shapes[1] = {100};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: LAST_DIM_LARGE_A tiling key - last dim is A axis (same in input and output) and large enough
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_last_dim_large_a_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{3, 1000}, {3, 1000}};
    gert::StorageShape outshape = {{3, 1000}, {3, 1000}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 1000};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11002;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: UB_BRC tiling key with 2D broadcast where B axis is first dim (LAST_DIM_LARGE_A for last A axis)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_ub_brc_2d_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11002;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: NDDMA tiling key with 2D broadcast where both axes are B (LAST_DIM_LARGE_B)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_nddma_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1}, {1, 1}};
    gert::StorageShape outshape = {{2, 100}, {2, 100}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {2, 100};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: input dimNum > output dimNum to cover GetShapeInfo failure path (lines 193-198)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_input_more_dims_than_output_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape outshape = {{1, 1}, {1, 1}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: outDimNum > BRCTO_MAX_DIM_NUM (8) to cover GetShapeInfo failure (lines 200-205)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_output_dims_exceed_max_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape outshape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape shape1 = {{9}, {9}};
    int32_t shapes[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: empty shape (shapeSize == 0) to cover GetShapeInfo failure (lines 208-213)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_empty_shape_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 0}, {1, 0}};
    gert::StorageShape outshape = {{1, 0}, {1, 0}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {1, 0};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: broadcast rule violation in CheckBroadcastRule (lines 153-158)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_broadcast_rule_violation_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{3, 5}, {3, 5}};
    gert::StorageShape outshape = {{3, 7}, {3, 7}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 7};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: INT64 dtype for shape tensor (constDType = DT_INT64) - LAST_DIM_LARGE_B
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_int64_shape_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1}, {1, 1}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int64_t shapes[2] = {3, 64};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT64, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: FLOAT32 dtype (dtypeSize=4) to cover different dtype sizes - LAST_DIM_LARGE_B
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_float32_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape outshape = {{1, 1, 64}, {1, 1, 64}};
    gert::StorageShape shape1 = {{3}, {3}};
    int32_t shapes[3] = {1, 1, 64};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: larger 2D broadcast with B axis in first dim to cover aAxis branch in GetMCTilingInfo (lines 419-426)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_a_axis_split_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {4, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 200}, {1, 200}};
    gert::StorageShape outshape = {{1, 200}, {1, 200}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {1, 200};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11002;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: 3D shape with B axis in first dim to cover UB_BRC path (bAxis is 1st dim)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_b_axis_split_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {4, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 64}, {1, 1, 64}};
    gert::StorageShape outshape = {{8, 1, 64}, {8, 1, 64}};
    gert::StorageShape shape1 = {{3}, {3}};
    int32_t shapes[3] = {8, 1, 64};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11001;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: 1D no-broadcast (identity) to cover 1D LAST_DIM_LARGE_A path with !abInfo_[dimNum-1]
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_1d_no_broadcast_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{64}, {64}};
    gert::StorageShape outshape = {{64}, {64}};
    gert::StorageShape shape1 = {{1}, {1}};
    int32_t shapes[1] = {64};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11002;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: LAST_DIM_SMALL_A tiling key update - small last dim A axis with INT8 dtype
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_last_dim_small_a_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 4}, {1, 4}};
    gert::StorageShape outshape = {{2, 4}, {2, 4}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {2, 4};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11000;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: 4D shape with multiple B and A axes to cover full NDDMA and mixed axis paths
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_4d_mixed_axes_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 3, 1, 5}, {1, 3, 1, 5}};
    gert::StorageShape outshape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape shape1 = {{4}, {4}};
    int32_t shapes[4] = {2, 3, 4, 5};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11000;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: INT8 dtype (dtypeSize=1) to cover LAST_DIM_SMALL_A with smallest dtype
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_int8_dtype_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_INT8, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_INT8, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11005;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: BF16 dtype (dtypeSize=2) to cover LAST_DIM_LARGE_A with BF16
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_bf16_dtype_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_BF16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_BF16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11002;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: INT64 dtype (dtypeSize=8) to cover LAST_DIM_LARGE_A with large dtype
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_int64_dtype_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 16}, {1, 16}};
    gert::StorageShape outshape = {{3, 16}, {3, 16}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 16};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_INT64, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_INT64, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11002;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: 5D shape to cover DIM_NUM_THRESHOLD_FOR_R4_SIZE path in CalcTensorSize (line 573-574) - UB_BRC mode
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_5d_shape_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 1, 1, 32}, {1, 1, 1, 1, 32}};
    gert::StorageShape outshape = {{2, 3, 4, 1, 32}, {2, 3, 4, 1, 32}};
    gert::StorageShape shape1 = {{5}, {5}};
    int32_t shapes[5] = {2, 3, 4, 1, 32};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11001;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: UB zero ubSize to cover GetHardwareInfo failure for ubSize <= 0
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_zero_ubsize_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 0, 128, 256, 32};
    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: zero blockSize to cover GetHardwareInfo failure for blockSize <= 0
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_zero_blocksize_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 0, 32};
    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: zero clSize to cover GetHardwareInfo failure for cacheLine <= 0
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_zero_clsize_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 0, 256, 32};
    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: zero vRegSize to cover GetHardwareInfo failure for vlSize <= 0
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_zero_vregsize_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 0};
    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: 3D shape with all dims being B axis to cover pure B broadcast path
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_all_b_axes_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape outshape = {{4, 8, 16}, {4, 8, 16}};
    gert::StorageShape shape1 = {{3}, {3}};
    int32_t shapes[3] = {4, 8, 16};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: large data with small cores to cover UB_BRC path for 4D broadcast
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_brwd_path_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {4, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 64, 100}, {1, 1, 64, 100}};
    gert::StorageShape outshape = {{8, 1, 64, 100}, {8, 1, 64, 100}};
    gert::StorageShape shape1 = {{4}, {4}};
    int32_t shapes[4] = {8, 1, 64, 100};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11001;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: 3D shape where outLastDim < minTensorSize to cover CalcTensorSize path (line 607)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_small_last_dim_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 2}, {1, 1, 2}};
    gert::StorageShape outshape = {{3, 1, 2}, {3, 1, 2}};
    gert::StorageShape shape1 = {{3}, {3}};
    int32_t shapes[3] = {3, 1, 2};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11000;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: output shape with all dims being 1 to cover DeleteOneSizeAxis mIdx==0 path (lines 117-119)
// [1,1] -> [1,1] (all dims are size 1, which triggers the mIdx==0 fallback)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_all_one_dims_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1}, {1, 1}};
    gert::StorageShape outshape = {{1, 1}, {1, 1}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {1, 1};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11001;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: 2D shape where last dim is large A axis and outLastDim >= ubGate
// to cover CalcTensorSize path for data copy mode (lines 584-585)
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_large_last_dim_a_ub_gate_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1920}, {1, 1920}};
    gert::StorageShape outshape = {{3, 1920}, {3, 1920}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 1920};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11002;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: 4D shape with B axis last dim >= LAST_DIM_GATE and last dim A
// to cover UB broadcast path where abInfo[last-1] is true with large last dim
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_ub_brc_last_dim_b_large_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape outshape = {{1, 1, 1, 512}, {1, 1, 1, 512}};
    gert::StorageShape shape1 = {{4}, {4}};
    int32_t shapes[4] = {1, 1, 1, 512};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: 6D shape (>5 dims) to cover DIM_NUM_THRESHOLD_FOR_R4_SIZE path with actual r4DimSize calc -
// LAST_DIM_SMALL_A
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_6d_shape_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 1, 1, 2, 4}, {1, 1, 1, 1, 2, 4}};
    gert::StorageShape outshape = {{3, 1, 1, 1, 2, 4}, {3, 1, 1, 1, 2, 4}};
    gert::StorageShape shape1 = {{6}, {6}};
    int32_t shapes[6] = {3, 1, 1, 1, 2, 4};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11005;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: 3D [1,1,1] -> [1,8,16] to cover UB_BRC with B in mid and last dim
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_ub_brc_multi_b_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape outshape = {{1, 8, 16}, {1, 8, 16}};
    gert::StorageShape shape1 = {{3}, {3}};
    int32_t shapes[3] = {1, 8, 16};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: UINT32 dtype to cover UINT32 dtype branch
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_uint32_dtype_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 32}, {1, 32}};
    gert::StorageShape outshape = {{3, 32}, {3, 32}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 32};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_UINT32, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_UINT32, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11002;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: BOOL dtype to cover BOOL dtype branch
TEST_F(BroadcastToTilingTest, BroadcastTo_tiling_bool_dtype_001)
{
    optiling::BroadcastToCompileInfo compileInfo = {64, 245760, 128, 256, 32};
    gert::StorageShape inshape = {{1, 32}, {1, 32}};
    gert::StorageShape outshape = {{3, 32}, {3, 32}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 32};
    gert::TilingContextPara tilingContextPara(
        "BroadcastTo", {{inshape, ge::DT_BOOL, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_BOOL, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11005;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}
