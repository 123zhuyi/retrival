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
 * \file test_tile_with_axis_tiling.cpp
 * \brief TileWithAxis Tiling 分支 UT 覆盖
 *
 * 验证 ubAxis=0 (切 outerDim), ubAxis=1 (切 tiles), ubAxis=2 (切 rowLength)
 * 三种 Tiling 分支的 tiling 函数正确性。
 *
 * 设计依据: DESIGN.md v1.6 3.5 节 Tiling 两步法
 * 3D 模型: inShape=[outerDim, 1, rowLength], outShape=[outerDim, tiles, rowLength]
 * 算法: 先 UB 切分 (基于 outShape 选 ubAxis + ubFactor), 再多核切分
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/tile_with_axis_tiling_data.h"

namespace TileWithAxisUT {
using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "TileWithAxis";

// ============================================================================
// Helper: build TilingContextPara with attributes
// ============================================================================
static TilingContextPara MakeTilingPara(
    const string& caseName,
    initializer_list<int64_t> xShape,
    DataType xDtype,
    int64_t axis,
    int64_t tiles,
    uint64_t ubSize = 253952,    // Ascend950 UB = 248KB
    uint64_t coreNum = 64,
    uint64_t tilingDataSize = 4096)
{
    StorageShape shape = {xShape, xShape};
    vector<TilingContextPara::TensorDescription> inputs = {
        {shape, xDtype, FORMAT_ND}
    };
    // Output shape: not used by tiling function directly; infershape computes it
    StorageShape outShape = {xShape, xShape};  // placeholder
    vector<TilingContextPara::TensorDescription> outputs = {
        {outShape, xDtype, FORMAT_ND}
    };

    vector<TilingContextPara::OpAttr> attrs = {
        {"axis",  Ops::Math::AnyValue::CreateFrom<int64_t>(axis)},
        {"tiles", Ops::Math::AnyValue::CreateFrom<int64_t>(tiles)}
    };

    struct TileWithAxisCompileInfo {} compileInfo;
    return TilingContextPara(OP_NAME, inputs, outputs, attrs, &compileInfo,
                             coreNum, ubSize, tilingDataSize);
}

// ============================================================================
// Helper: execute tiling and extract TilingData
// ============================================================================
static bool GetTilingData(const TilingContextPara& para, TilingInfo& info)
{
    return ExecuteTiling(para, info);
}

// ============================================================================
// P0: ubAxis=0 分支 — 切 outerDim
// 大 outerDim，小 rowLength，全 tiles 能放入 UB
// shape=[100, 10, 10], axis=1, tiles=3, fp16
//   outerDim=100, rowLength=100, tiles*rowLength=300
//   bufferSizeElements = (248KB/2)/2 = 63488 >= 300 → ubAxis=0
// ============================================================================
TEST(TileWithAxisTiling, ubAxis0_cut_outerDim_fp16)
{
    auto para = MakeTilingPara("ubAxis0_fp16",
        {100, 10, 10}, DT_FLOAT16, 1, 3);
    TilingInfo info;
    ASSERT_TRUE(GetTilingData(para, info));

    auto* td = reinterpret_cast<const TileWithAxisTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->ubAxis, static_cast<uint8_t>(0));
    EXPECT_EQ(td->tiles, 3);
    EXPECT_EQ(td->rowLength, 100);     // axisDim=10 * innerDim=10
    EXPECT_EQ(td->inShape[0], 100);    // outerDim
    EXPECT_EQ(td->inShape[1], 1);      // 输入中间维=1
    EXPECT_EQ(td->inShape[2], 100);    // rowLength
    EXPECT_EQ(td->outShape[0], 100);   // outerDim
    EXPECT_EQ(td->outShape[1], 3);     // tiles
    EXPECT_EQ(td->outShape[2], 100);   // rowLength
    // Step 2.5 核数感知: totalCount=1 < threshold=51, 降 ubFactor 增加 block
    // targetBlocks=min(51, 30000/4096=7, 100)=7, ubFactor=CeilDiv(100,7)=15
    EXPECT_EQ(td->totalCount, static_cast<uint64_t>(7));
    EXPECT_EQ(info.blockNum, static_cast<size_t>(7));
    // workspace: 0
    ASSERT_EQ(info.workspaceSizes.size(), static_cast<size_t>(1));
    EXPECT_EQ(info.workspaceSizes[0], static_cast<int64_t>(0));
}

// ============================================================================
// P0: ubAxis=0 分支 — fp32 dtype
// shape=[100, 6, 50], axis=1, tiles=3, fp32
//   outerDim=100, rowLength=300, tiles*rowLength=900
//   bufferSizeElements = (248KB/2)/4 = 31744 >= 900 → ubAxis=0
// ============================================================================
TEST(TileWithAxisTiling, ubAxis0_cut_outerDim_fp32)
{
    auto para = MakeTilingPara("ubAxis0_fp32",
        {100, 6, 50}, DT_FLOAT, 1, 3);
    TilingInfo info;
    ASSERT_TRUE(GetTilingData(para, info));

    auto* td = reinterpret_cast<const TileWithAxisTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->ubAxis, static_cast<uint8_t>(0));
    EXPECT_EQ(td->rowLength, 300);     // axisDim=6 * innerDim=50
    EXPECT_EQ(td->outShape[0], 100);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 300);
    // Step 2.5 核数感知: totalCount=3 < threshold=51, 降 ubFactor
    // targetBlocks=min(51, 90000/4096=21, 100)=21, ubFactor=CeilDiv(100,21)=5
    EXPECT_EQ(td->ubFactor, static_cast<uint32_t>(5));
    EXPECT_EQ(td->totalCount, static_cast<uint64_t>(20));
}

// ============================================================================
// P0: ubAxis=1 分支 — 切 tiles (中间维)
// outerDim 小, rowLength 中等, tiles 较多
// shape=[3, 1000, 60], axis=1, tiles=3, fp16
//   outerDim=3, rowLength=60000, tiles*rowLength=180000 > 63488 → ubFactor0=0
//   ubFactor_1 = min(3, 63488/60000) = min(3,1) = 1 → ubAxis=1
// ============================================================================
TEST(TileWithAxisTiling, ubAxis1_cut_tiles_fp16)
{
    auto para = MakeTilingPara("ubAxis1_fp16",
        {3, 1000, 60}, DT_FLOAT16, 1, 3);
    TilingInfo info;
    ASSERT_TRUE(GetTilingData(para, info));

    auto* td = reinterpret_cast<const TileWithAxisTilingData*>(info.tilingData.get());
    // Step 2.5 跨轴回退: ubAxis=1 totalCount=9 < threshold=51, 回退到 ubAxis=2
    // candTarget=51, bpot=CeilDiv(51,9)=6, ubFactor=CeilDiv(60000,6)=10000
    EXPECT_EQ(td->ubAxis, static_cast<uint8_t>(2));
    EXPECT_EQ(td->ubFactor, static_cast<uint32_t>(10000));
    EXPECT_EQ(td->rowLength, 60000);   // axisDim=1000 * innerDim=60
    EXPECT_EQ(td->outShape[0], 3);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 60000);
    // totalCount = 3 * 3 * CeilDiv(60000, 10000) = 54
    EXPECT_EQ(td->totalCount, static_cast<uint64_t>(54));
}

// ============================================================================
// P0: ubAxis=1 分支 — fp32 + 更多 tiles
// shape=[3, 500, 40], axis=1, tiles=5, fp32
//   outerDim=3, rowLength=20000, tiles*rowLength=100000 > 31744 → ubFactor0=0
//   ubFactor_1 = min(5, 31744/20000) = min(5,1) = 1 → ubAxis=1
// ============================================================================
TEST(TileWithAxisTiling, ubAxis1_cut_tiles_fp32)
{
    auto para = MakeTilingPara("ubAxis1_fp32",
        {3, 500, 40}, DT_FLOAT, 1, 5);
    TilingInfo info;
    ASSERT_TRUE(GetTilingData(para, info));

    auto* td = reinterpret_cast<const TileWithAxisTilingData*>(info.tilingData.get());
    // Step 2.5 跨轴回退: ubAxis=1 totalCount=15 < threshold=51, 回退到 ubAxis=2
    // candTarget=51, bpot=CeilDiv(51,15)=4, ubFactor=CeilDiv(20000,4)=5000
    EXPECT_EQ(td->ubAxis, static_cast<uint8_t>(2));
    EXPECT_EQ(td->ubFactor, static_cast<uint32_t>(5000));
    EXPECT_EQ(td->tiles, 5);
    EXPECT_EQ(td->rowLength, 20000);
    // totalCount = 3 * 5 * CeilDiv(20000, 5000) = 60
    EXPECT_EQ(td->totalCount, static_cast<uint64_t>(60));
}

// ============================================================================
// P0: ubAxis=2 分支 — 切 rowLength (内维)
// ubFactor0 = 0, ubFactor1 = 0 → 兜底 ubAxis=2
// shape=[2, 65536, 16], axis=1, tiles=3, fp16
//   outerDim=2, rowLength=1048576 → ubFactor0=0, ubFactor1=0
//   ubFactor_2 = min(1048576, 63488) = 63488 → ubAxis=2
// ============================================================================
TEST(TileWithAxisTiling, ubAxis2_cut_rowLength_fp16)
{
    auto para = MakeTilingPara("ubAxis2_fp16",
        {2, 65536, 16}, DT_FLOAT16, 1, 3);
    TilingInfo info;
    ASSERT_TRUE(GetTilingData(para, info));

    auto* td = reinterpret_cast<const TileWithAxisTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->ubAxis, static_cast<uint8_t>(2));
    EXPECT_EQ(td->ubFactor, static_cast<uint32_t>(63488));
    EXPECT_EQ(td->rowLength, 1048576); // axisDim=65536 * innerDim=16
    EXPECT_EQ(td->outShape[0], 2);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 1048576);
    // totalCount = outerDim * tiles * CeilDiv(rowLength, ubFactor)
    //            = 2 * 3 * 17 = 102
    EXPECT_EQ(td->totalCount, static_cast<uint64_t>(102));
}

// ============================================================================
// P0: ubAxis=2 分支 — fp32
// shape=[2, 32768, 16], axis=1, tiles=3, fp32
//   outerDim=2, rowLength=524288 → ubFactor0=0, ubFactor1=0
//   ubFactor_2 = min(524288, 31744) = 31744 → ubAxis=2
// ============================================================================
TEST(TileWithAxisTiling, ubAxis2_cut_rowLength_fp32)
{
    auto para = MakeTilingPara("ubAxis2_fp32",
        {2, 32768, 16}, DT_FLOAT, 1, 3);
    TilingInfo info;
    ASSERT_TRUE(GetTilingData(para, info));

    auto* td = reinterpret_cast<const TileWithAxisTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->ubAxis, static_cast<uint8_t>(2));
    EXPECT_EQ(td->ubFactor, static_cast<uint32_t>(31744));
    EXPECT_EQ(td->rowLength, 524288);
    // totalCount = 2 * 3 * CeilDiv(524288, 31744) = 6 * 17 = 102
    EXPECT_EQ(td->totalCount, static_cast<uint64_t>(102));
}

// ============================================================================
// Scalar (rank=0) 处理: shape=[], tiles=5, fp16
//   标量展开为 outerDim=1, rowLength=1, inShape=[1,1,1], outShape=[1,5,1]
//   ubAxis=0, ubFactor=1, totalCount=1
// ============================================================================
TEST(TileWithAxisTiling, scalar_input)
{
    auto para = MakeTilingPara("scalar",
        {}, DT_FLOAT16, 0, 5);
    TilingInfo info;
    ASSERT_TRUE(GetTilingData(para, info));

    auto* td = reinterpret_cast<const TileWithAxisTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->ubAxis, static_cast<uint8_t>(0));
    EXPECT_EQ(td->tiles, 5);
    EXPECT_EQ(td->inShape[0], 1);
    EXPECT_EQ(td->inShape[1], 1);
    EXPECT_EQ(td->inShape[2], 1);
    EXPECT_EQ(td->outShape[0], 1);
    EXPECT_EQ(td->outShape[1], 5);
    EXPECT_EQ(td->outShape[2], 1);
    EXPECT_EQ(td->totalCount, static_cast<uint64_t>(1));
    EXPECT_EQ(td->perCoreCount, static_cast<uint64_t>(1));
    EXPECT_EQ(td->ubFactor, static_cast<uint32_t>(1));
    EXPECT_EQ(info.blockNum, static_cast<size_t>(1));
}

// ============================================================================
// Negative axis: axis=-1 on 3D input → axis=2
// shape=[2, 3, 4], axis=-1, tiles=3, fp16
//   outerDim = 2*3=6, axisDim=4, innerDim=1, rowLength=4*1=4
//   ubFactor_0 = min(6, 63488/(3*4)) = min(6, 5290) = 6 → ubAxis=0
// ============================================================================
TEST(TileWithAxisTiling, negative_axis)
{
    auto para = MakeTilingPara("negative_axis",
        {2, 3, 4}, DT_FLOAT16, -1, 3);
    TilingInfo info;
    ASSERT_TRUE(GetTilingData(para, info));

    auto* td = reinterpret_cast<const TileWithAxisTilingData*>(info.tilingData.get());
    // Step 2.5 跨轴回退: ubAxis=0 totalCount=1 < threshold=51, 回退到 ubAxis=2
    // candTarget=12, bpot=CeilDiv(12,6)=2, ubFactor=CeilDiv(4,2)=2
    EXPECT_EQ(td->ubAxis, static_cast<uint8_t>(2));
    EXPECT_EQ(td->tiles, 3);
    EXPECT_EQ(td->rowLength, 4);       // axisDim=4, innerDim=1
    EXPECT_EQ(td->inShape[0], 6);      // outerDim = 2*3 = 6
    EXPECT_EQ(td->outShape[0], 6);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 4);
}

// ============================================================================
// 边界: tiles=1 (恒等变换)
// shape=[2, 3, 4], axis=1, tiles=1, fp16
//   任何 ubAxis 均可，算法按优先级选 ubAxis=0
// ============================================================================
TEST(TileWithAxisTiling, tiles_equals_one)
{
    auto para = MakeTilingPara("tiles_1",
        {2, 3, 4}, DT_FLOAT16, 1, 1);
    TilingInfo info;
    ASSERT_TRUE(GetTilingData(para, info));

    auto* td = reinterpret_cast<const TileWithAxisTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->tiles, 1);
    EXPECT_EQ(td->outShape[1], 1);
    EXPECT_NE(td->totalCount, static_cast<uint64_t>(0));
}

// ============================================================================
// 边界: tiles 为非法值 (tiles <= 0) — 应返回 GRAPH_FAILED
// ============================================================================
TEST(TileWithAxisTiling, tiles_invalid_zero)
{
    auto para = MakeTilingPara("tiles_0",
        {2, 3, 4}, DT_FLOAT16, 0, 0);
    TilingInfo info;
    EXPECT_FALSE(GetTilingData(para, info));
}

// ============================================================================
// 边界: axis 越界 — 应返回 GRAPH_FAILED
// ============================================================================
TEST(TileWithAxisTiling, axis_out_of_range)
{
    // rank=3, axis=3 → 越界
    auto para = MakeTilingPara("axis_oob",
        {2, 3, 4}, DT_FLOAT16, 3, 2);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ============================================================================
// 高维测试: 4D input with axis=2
// shape=[2, 4, 8, 16], axis=2, tiles=3, fp16
//   outerDim = 2*4=8, axisDim=8, innerDim=16, rowLength=128
//   ubFactor_0 = min(8, 63488/(3*128)) = min(8, 165) = 8 → ubAxis=0
// ============================================================================
TEST(TileWithAxisTiling, high_dim_4d_axis_2)
{
    auto para = MakeTilingPara("4d_axis2",
        {2, 4, 8, 16}, DT_FLOAT16, 2, 3);
    TilingInfo info;
    ASSERT_TRUE(GetTilingData(para, info));

    auto* td = reinterpret_cast<const TileWithAxisTilingData*>(info.tilingData.get());
    // Step 2.5 跨轴回退: ubAxis=0 totalCount=1 < threshold=51, 回退到 ubAxis=2
    // candTarget=51, bpot=CeilDiv(51,24)=3, ubFactor=CeilDiv(128,3)=43
    EXPECT_EQ(td->ubAxis, static_cast<uint8_t>(2));
    EXPECT_EQ(td->rowLength, 128);     // axisDim=8 * innerDim=16
    EXPECT_EQ(td->inShape[0], 8);      // outerDim = 2*4
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 128);
}

} // namespace TileWithAxisUT
