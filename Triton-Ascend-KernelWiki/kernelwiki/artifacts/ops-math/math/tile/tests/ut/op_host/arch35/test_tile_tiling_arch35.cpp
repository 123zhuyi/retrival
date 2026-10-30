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
 * \file test_tile_tiling_arch35.cpp
 * \brief tile tiling ut test
 */

#include "../../../../op_host/arch35/tile_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace ge;

class TileTiling : public testing::Test
{
protected:
    static void SetUpTestCase() { std::cout << "TileTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TileTiling TearDown" << std::endl; }
};

TEST_F(TileTiling, TileTiling_001)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;  // 240 * 1024
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;
    gert::StorageShape xShape = {{2, 3, 1, 1000, 1, 1}, {2, 3, 1, 1000, 1, 1}};
    gert::StorageShape shape = {{2, 3, 1, 1, 2, 2}, {2, 3, 1, 1, 2, 2}};
    gert::StorageShape yShape = {{4, 9, 1, 1000, 2, 2}, {4, 9, 1, 1000, 2, 2}};
    gert::TilingContextPara tilingContextPara(
        "Tile",
        {{ xShape, ge::DT_FLOAT, ge::FORMAT_ND }, { shape, ge::DT_INT32, ge::FORMAT_ND }},
        {{ yShape, ge::DT_FLOAT, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 11000;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: tile where output shape is not divisible by input shape, expect tiling to fail due to broadcast rule
// violation
TEST_F(TileTiling, TileTiling_shape_mismatch_not_divisible)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;
    // in={2,3}, out={3,3}: 3%2 != 0, violates broadcast rule
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape shape = {{2}, {2}};
    gert::StorageShape yShape = {{3, 3}, {3, 3}};
    gert::TilingContextPara tilingContextPara(
        "Tile", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile where output dim is smaller than input dim, expect tiling to fail due to broadcast rule violation
TEST_F(TileTiling, TileTiling_shape_mismatch_out_smaller)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;
    // in={2,3}, out={1,3}: out[0]=1 < in[0]=2, violates broadcast rule
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape shape = {{2}, {2}};
    gert::StorageShape yShape = {{1, 3}, {1, 3}};
    gert::TilingContextPara tilingContextPara(
        "Tile", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile where input shape has more dimensions than output shape, expect tiling to fail
TEST_F(TileTiling, TileTiling_input_more_dims_than_output)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;
    // in is 4D, out is 2D → inDimNum > outDimNum
    gert::StorageShape xShape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape shape = {{4}, {4}};
    gert::StorageShape yShape = {{1, 1}, {1, 1}};
    gert::TilingContextPara tilingContextPara(
        "Tile", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile where output shape has more than 8 dimensions, expect tiling to fail
TEST_F(TileTiling, TileTiling_output_exceed_max_dims)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;
    // out has 9 dimensions, exceeds TILE_MAX_DIM_NUM=8
    gert::StorageShape xShape = {{1}, {1}};
    gert::StorageShape shape = {{9}, {9}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    gert::TilingContextPara tilingContextPara(
        "Tile", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile where input shape has zero elements, expect tiling to fail due to empty shape
TEST_F(TileTiling, TileTiling_empty_shape_input)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;
    // input shape has 0 elements
    gert::StorageShape xShape = {{1, 0}, {1, 0}};
    gert::StorageShape shape = {{2}, {2}};
    gert::StorageShape yShape = {{1, 0}, {1, 0}};
    gert::TilingContextPara tilingContextPara(
        "Tile", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile where output shape has zero elements but input is non-empty, expect tiling to fail due to empty
// output shape
TEST_F(TileTiling, TileTiling_empty_shape_output)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;
    // output shape has 0 elements but input is non-empty
    gert::StorageShape xShape = {{1, 3}, {1, 3}};
    gert::StorageShape shape = {{2}, {2}};
    gert::StorageShape yShape = {{1, 0}, {1, 0}};
    gert::TilingContextPara tilingContextPara(
        "Tile", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile with negative coreNum in compileInfo, expect tiling to fail because hardware info is invalid
TEST_F(TileTiling, TileTiling_negative_corenum)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = -2;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;
    gert::StorageShape xShape = {{2, 3, 1, 1000, 1, 1}, {2, 3, 1, 1000, 1, 1}};
    gert::StorageShape shape = {{2, 3, 1, 1, 2, 2}, {2, 3, 1, 1, 2, 2}};
    gert::StorageShape yShape = {{4, 9, 1, 1000, 2, 2}, {4, 9, 1, 1000, 2, 2}};
    gert::TilingContextPara tilingContextPara(
        "Tile", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile with zero ubSize in compileInfo, expect tiling to fail because hardware info is invalid
TEST_F(TileTiling, TileTiling_zero_ubsize)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 0;
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;
    gert::StorageShape xShape = {{1, 64}, {1, 64}};
    gert::StorageShape shape = {{2}, {2}};
    gert::StorageShape yShape = {{3, 64}, {3, 64}};
    gert::TilingContextPara tilingContextPara(
        "Tile", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile with zero clSize in compileInfo, expect tiling to fail because hardware info is invalid
TEST_F(TileTiling, TileTiling_zero_clsize)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 0;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;
    gert::StorageShape xShape = {{1, 64}, {1, 64}};
    gert::StorageShape shape = {{2}, {2}};
    gert::StorageShape yShape = {{3, 64}, {3, 64}};
    gert::TilingContextPara tilingContextPara(
        "Tile", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile with zero blockSize in compileInfo, expect tiling to fail because hardware info is invalid
TEST_F(TileTiling, TileTiling_zero_blocksize)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 0;
    gert::StorageShape xShape = {{1, 64}, {1, 64}};
    gert::StorageShape shape = {{2}, {2}};
    gert::StorageShape yShape = {{3, 64}, {3, 64}};
    gert::TilingContextPara tilingContextPara(
        "Tile", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test scenario: tile with zero vRegSize in compileInfo, expect tiling to fail because hardware info is invalid
TEST_F(TileTiling, TileTiling_zero_vregsize)
{
    optiling::TileCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 256;
    compileInfo.vRegSize = 0;
    compileInfo.blockSize = 32;
    gert::StorageShape xShape = {{1, 64}, {1, 64}};
    gert::StorageShape shape = {{2}, {2}};
    gert::StorageShape yShape = {{3, 64}, {3, 64}};
    gert::TilingContextPara tilingContextPara(
        "Tile", {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
