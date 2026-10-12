/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/space_to_batch_nd/op_kernel/arch35/space_to_batch_nd_tiling_data.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class SpaceToBatchNDTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SpaceToBatchNDTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SpaceToBatchNDTilingTest TearDown" << std::endl; }
};

namespace {
SpaceToBatchNDCompileInfo compileInfo;
}

// ---------- 1D spatial with padding ----------

TEST_F(SpaceToBatchNDTilingTest, Spatial1D_WithPadding)
{
    // in [279, 465, 385], bs=[3], pad=[[1,2]]
    // padded spatial = 468, out spatial = 156, batch = 279*3 = 837
    // out [837, 156, 385]
    std::vector<int32_t> blockShapeValues = {3};
    std::vector<int32_t> paddingsValues = {1, 2};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{279, 465, 385}, {279, 465, 385}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{837, 156, 385}, {837, 156, 385}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>{0});

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->numSpatialDims, 1);
    EXPECT_EQ(td->rank, 3);
    EXPECT_EQ(td->outShape[0], 837);
    EXPECT_EQ(td->outShape[1], 156);
    EXPECT_EQ(td->outShape[2], 385);
    EXPECT_EQ(td->blockShape[0], 3);
    EXPECT_EQ(td->padTop[0], 1);
    EXPECT_EQ(td->padBottom[0], 2);
    EXPECT_GE(td->totalCount, 1);
    EXPECT_GE(td->perCoreCount, 1);
    EXPECT_GE(td->ubFactor, 1);
    EXPECT_LE(td->ubAxis, 2);
    EXPECT_NE(info.tilingKey, static_cast<uint64_t>(-1));
}

// ---------- 2D spatial no padding ----------

TEST_F(SpaceToBatchNDTilingTest, Spatial2D_NoPadding)
{
    // in [2, 4, 4, 3], bs=[2,2], pad=0
    // batch = 2*4 = 8, out_spatial = [2, 2]
    // out [8, 2, 2, 3]
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 2, 2, 3}, {8, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->numSpatialDims, 2);
    EXPECT_EQ(td->rank, 4);
    EXPECT_EQ(td->outShape[0], 8);
    EXPECT_EQ(td->outShape[1], 2);
    EXPECT_EQ(td->outShape[2], 2);
    EXPECT_EQ(td->outShape[3], 3);
    EXPECT_EQ(td->blockShape[0], 2);
    EXPECT_EQ(td->blockShape[1], 2);
}

// ---------- 2D spatial with padding ----------

TEST_F(SpaceToBatchNDTilingTest, Spatial2D_WithPadding)
{
    // in [1, 4, 4, 3], bs=[2,2], pad=[[1,1],[1,1]]
    // padded = [6,6], out = [4, 3, 3, 3]
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {1, 1, 1, 1};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{1, 4, 4, 3}, {1, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{4, 3, 3, 3}, {4, 3, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->outShape[0], 4);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 3);
    EXPECT_EQ(td->outShape[3], 3);
    EXPECT_EQ(td->padTop[0], 1);
    EXPECT_EQ(td->padBottom[0], 1);
    EXPECT_EQ(td->padTop[1], 1);
    EXPECT_EQ(td->padBottom[1], 1);
}

// ---------- 4D spatial large batch ----------

TEST_F(SpaceToBatchNDTilingTest, Spatial4D_LargeBatch)
{
    // in [24, 28, 44, 42, 40], bs=[2,2,2,2], pad=0
    // batch = 24*16 = 384, out_spatial = [14, 22, 21, 20]
    // out [384, 14, 22, 21, 20, 1]
    std::vector<int32_t> blockShapeValues = {2, 2, 2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0, 0, 0, 0, 0};
    gert::TilingContextPara para(
        "SpaceToBatchND",
        {
            {{{24, 28, 44, 42, 40}, {24, 28, 44, 42, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{384, 14, 22, 21, 20, 1}, {384, 14, 22, 21, 20, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->numSpatialDims, 4);
    EXPECT_EQ(td->rank, 6);
    EXPECT_EQ(td->outShape[0], 384);
    EXPECT_EQ(td->outShape[1], 14);
    EXPECT_EQ(td->outShape[2], 22);
    EXPECT_EQ(td->outShape[3], 21);
    EXPECT_EQ(td->outShape[4], 20);
    EXPECT_EQ(td->outShape[5], 1);
    EXPECT_GE(td->totalCount, 1);
}

// ---------- Negative padding should fail ----------

TEST_F(SpaceToBatchNDTilingTest, NegativePaddingShouldFail)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> paddingsValues = {-1, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}
