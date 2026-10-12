/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <cstring>
#include "tile_tiling.h"
#include "../../../op_kernel/tile_tiling_data.h"
#include "../../../op_kernel/tile_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class TileTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TileTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TileTiling TearDown" << std::endl; }
};

TEST_F(TileTiling, tile_tiling_2d_float32)
{
    optiling::TileCompileInfo compileInfo;
    int32_t multiplesData[] = {3, 2};
    gert::TilingContextPara tilingContextPara(
        "Tile",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, multiplesData},
        },
        {
            {{{6, 6}, {6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(ret);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(TileTilingData));
    const TileTilingData* tiling = reinterpret_cast<const TileTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tiling->numDims, 2);
    EXPECT_EQ(tiling->inputShape[0], 2);
    EXPECT_EQ(tiling->inputShape[1], 3);
    EXPECT_EQ(tiling->multiples[0], 3);
    EXPECT_EQ(tiling->multiples[1], 2);
    EXPECT_EQ(tiling->outputShape[0], 6);
    EXPECT_EQ(tiling->outputShape[1], 6);
    EXPECT_EQ(tiling->totalInputElems, 6);
    EXPECT_EQ(tiling->totalOutputElems, 36);
    EXPECT_EQ(tiling->elemBytes, 4);
}

TEST_F(TileTiling, tile_tiling_1d_int32)
{
    optiling::TileCompileInfo compileInfo;
    int32_t multiplesData[] = {8};
    gert::TilingContextPara tilingContextPara(
        "Tile",
        {
            {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, multiplesData},
        },
        {
            {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(ret);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(TileTilingData));
    const TileTilingData* tiling = reinterpret_cast<const TileTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tiling->numDims, 1);
    EXPECT_EQ(tiling->inputShape[0], 128);
    EXPECT_EQ(tiling->multiples[0], 8);
    EXPECT_EQ(tiling->outputShape[0], 1024);
    EXPECT_EQ(tiling->totalInputElems, 128);
    EXPECT_EQ(tiling->totalOutputElems, 1024);
    EXPECT_EQ(tiling->elemBytes, 4);
}

TEST_F(TileTiling, tile_tiling_int64_multiples)
{
    optiling::TileCompileInfo compileInfo;
    int64_t multiplesData[] = {2, 4};
    gert::TilingContextPara tilingContextPara(
        "Tile",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, multiplesData},
        },
        {
            {{{6, 20}, {6, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(ret);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(TileTilingData));
    const TileTilingData* tiling = reinterpret_cast<const TileTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tiling->numDims, 2);
    EXPECT_EQ(tiling->inputShape[0], 3);
    EXPECT_EQ(tiling->inputShape[1], 5);
    EXPECT_EQ(tiling->multiples[0], 2);
    EXPECT_EQ(tiling->multiples[1], 4);
    EXPECT_EQ(tiling->outputShape[0], 6);
    EXPECT_EQ(tiling->outputShape[1], 20);
    EXPECT_EQ(tiling->totalInputElems, 15);
    EXPECT_EQ(tiling->totalOutputElems, 120);
    EXPECT_EQ(tiling->elemBytes, 2);
}

TEST_F(TileTiling, tile_tiling_dim_merge)
{
    optiling::TileCompileInfo compileInfo;
    int32_t multiplesData[] = {3, 1, 2};
    gert::TilingContextPara tilingContextPara(
        "Tile",
        {
            {{{2, 4, 5}, {2, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, multiplesData},
        },
        {
            {{{6, 4, 10}, {6, 4, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(ret);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(TileTilingData));
    const TileTilingData* tiling = reinterpret_cast<const TileTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tiling->numDims, 2);
    EXPECT_EQ(tiling->inputShape[0], 8);
    EXPECT_EQ(tiling->inputShape[1], 5);
    EXPECT_EQ(tiling->multiples[0], 3);
    EXPECT_EQ(tiling->multiples[1], 2);
    EXPECT_EQ(tiling->outputShape[0], 24);
    EXPECT_EQ(tiling->outputShape[1], 10);
    EXPECT_EQ(tiling->totalInputElems, 40);
    EXPECT_EQ(tiling->totalOutputElems, 240);
}
