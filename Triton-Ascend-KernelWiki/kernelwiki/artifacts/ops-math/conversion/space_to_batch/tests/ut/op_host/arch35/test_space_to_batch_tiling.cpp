/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/space_to_batch/op_kernel/arch35/space_to_batch_tiling_data.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class SpaceToBatchTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SpaceToBatchTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SpaceToBatchTilingTest TearDown" << std::endl; }
};

namespace {
SpaceToBatchCompileInfo compileInfo;
}

// ---------- No padding, typical case ----------

TEST_F(SpaceToBatchTilingTest, NoPadding_Typical)
{
    // in [1, 6, 6, 256], bs=2, pad=0 → out [4, 3, 3, 256]
    std::vector<int32_t> pads = {0, 0, 0, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 6, 6, 256}, {1, 6, 6, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{4, 3, 3, 256}, {4, 3, 3, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>{0});

    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->blockSize, 2);
    EXPECT_EQ(td->inShape[0], 1);
    EXPECT_EQ(td->inShape[1], 6);
    EXPECT_EQ(td->inShape[2], 6);
    EXPECT_EQ(td->inShape[3], 256);
    EXPECT_EQ(td->outShape[0], 4);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 3);
    EXPECT_EQ(td->outShape[3], 256);
    EXPECT_EQ(td->bufferSize, 65536);
    EXPECT_GE(td->totalCount, 1);
    EXPECT_GE(td->perCoreCount, 1);
    // ubAxis should be a valid axis (0-3)
    EXPECT_LE(td->ubAxis, 3);
}

// ---------- No padding, large C ----------

TEST_F(SpaceToBatchTilingTest, NoPadding_LargeC)
{
    // in [1, 4, 4, 9000], bs=2, pad=0 → out [4, 2, 2, 9000]
    std::vector<int32_t> pads = {0, 0, 0, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 4, 4, 9000}, {1, 4, 4, 9000}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{4, 2, 2, 9000}, {4, 2, 2, 9000}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->blockSize, 2);
    EXPECT_EQ(td->outShape[0], 4);
    EXPECT_EQ(td->outShape[1], 2);
    EXPECT_EQ(td->outShape[2], 2);
    EXPECT_EQ(td->outShape[3], 9000);
    EXPECT_GE(td->ubFactor, 1);
}

// ---------- No padding, large H ----------

TEST_F(SpaceToBatchTilingTest, NoPadding_LargeH)
{
    // in [1, 400, 4, 256], bs=2, pad=0 → out [4, 200, 2, 256]
    std::vector<int32_t> pads = {0, 0, 0, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 400, 4, 256}, {1, 400, 4, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{4, 200, 2, 256}, {4, 200, 2, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->blockSize, 2);
    EXPECT_EQ(td->outShape[0], 4);
    EXPECT_EQ(td->outShape[1], 200);
    EXPECT_GE(td->totalCount, 1);
}

// ---------- No padding, small tensor ----------

TEST_F(SpaceToBatchTilingTest, NoPadding_SmallTensor)
{
    // in [1, 4, 4, 32], bs=2, pad=0 → out [4, 2, 2, 32]
    std::vector<int32_t> pads = {0, 0, 0, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 4, 4, 32}, {1, 4, 4, 32}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{4, 2, 2, 32}, {4, 2, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->blockSize, 2);
    EXPECT_EQ(td->outShape[0], 4);
    EXPECT_EQ(td->outShape[1], 2);
    EXPECT_EQ(td->outShape[2], 2);
    EXPECT_EQ(td->outShape[3], 32);
    EXPECT_GE(td->totalCount, 1);
}

// ---------- With top-left padding ----------

TEST_F(SpaceToBatchTilingTest, WithTopLeftPadding)
{
    // in [1, 5, 5, 256], bs=2, pad=[1,0,1,0] → H_padded=6, W_padded=6 → out [4, 3, 3, 256]
    std::vector<int32_t> pads = {1, 0, 1, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 5, 5, 256}, {1, 5, 5, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{4, 3, 3, 256}, {4, 3, 3, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->blockSize, 2);
    EXPECT_EQ(td->paddings[0][0], 1); // pad_top
    EXPECT_EQ(td->paddings[0][1], 0); // pad_bottom
    EXPECT_EQ(td->paddings[1][0], 1); // pad_left
    EXPECT_EQ(td->paddings[1][1], 0); // pad_right
    EXPECT_EQ(td->outShape[0], 4);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 3);
    EXPECT_EQ(td->outShape[3], 256);
}

// ---------- All sides padding ----------

TEST_F(SpaceToBatchTilingTest, WithAllPadding)
{
    // in [1, 6, 6, 128], bs=2, pad=[1,1,1,1] → H_padded=8, W_padded=8 → out [4, 4, 4, 128]
    std::vector<int32_t> pads = {1, 1, 1, 1};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 6, 6, 128}, {1, 6, 6, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{4, 4, 4, 128}, {4, 4, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->blockSize, 2);
    EXPECT_EQ(td->paddings[0][0], 1);
    EXPECT_EQ(td->paddings[0][1], 1);
    EXPECT_EQ(td->paddings[1][0], 1);
    EXPECT_EQ(td->paddings[1][1], 1);
    EXPECT_EQ(td->outShape[0], 4);
    EXPECT_EQ(td->outShape[1], 4);
    EXPECT_EQ(td->outShape[2], 4);
}

// ---------- Larger block_size ----------

TEST_F(SpaceToBatchTilingTest, LargeBlockSize)
{
    // in [1, 12, 12, 256], bs=4, pad=0 → out [16, 3, 3, 256]
    std::vector<int32_t> pads = {0, 0, 0, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(4);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 12, 12, 256}, {1, 12, 12, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{16, 3, 3, 256}, {16, 3, 3, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->blockSize, 4);
    EXPECT_EQ(td->outShape[0], 16);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 3);
}

// ---------- Multi batch ----------

TEST_F(SpaceToBatchTilingTest, MultiBatch)
{
    // in [3, 6, 6, 128], bs=2, pad=0 → out [12, 3, 3, 128]
    std::vector<int32_t> pads = {0, 0, 0, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{3, 6, 6, 128}, {3, 6, 6, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{12, 3, 3, 128}, {12, 3, 3, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->inShape[0], 3);
    EXPECT_EQ(td->outShape[0], 12);
}

// ---------- TilingKey is set ----------

TEST_F(SpaceToBatchTilingTest, TilingKeyIsSet)
{
    std::vector<int32_t> pads = {0, 0, 0, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 6, 6, 256}, {1, 6, 6, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{4, 3, 3, 256}, {4, 3, 3, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_NE(info.tilingKey, static_cast<uint64_t>(-1));
}

// ---------- ubAxis=C: large C, small H/W ----------

TEST_F(SpaceToBatchTilingTest, UbAxisC_LargeChannel)
{
    // in [1, 2, 2, 16384], bs=2, pad=0 → out [4, 1, 1, 16384]
    // C is much larger than H*W → tiling should cut C axis
    std::vector<int32_t> pads = {0, 0, 0, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 2, 2, 16384}, {1, 2, 2, 16384}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{4, 1, 1, 16384}, {4, 1, 1, 16384}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->outShape[3], 16384);
    EXPECT_GE(td->ubFactor, 1);
}

// ---------- int64 paddings dtype ----------

TEST_F(SpaceToBatchTilingTest, Int64Paddings)
{
    std::vector<int64_t> pads = {0, 0, 0, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 6, 6, 128}, {1, 6, 6, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{4, 3, 3, 128}, {4, 3, 3, 128}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->blockSize, 2);
    EXPECT_EQ(td->outShape[0], 4);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 3);
}

// ---------- block_size=3 ----------

TEST_F(SpaceToBatchTilingTest, BlockSize3)
{
    // in [1, 9, 9, 64], bs=3, pad=0 → out [9, 3, 3, 64]
    std::vector<int32_t> pads = {0, 0, 0, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(3);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 9, 9, 64}, {1, 9, 9, 64}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{9, 3, 3, 64}, {9, 3, 3, 64}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->blockSize, 3);
    EXPECT_EQ(td->outShape[0], 9);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 3);
}

// ---------- block_size=3 with padding ----------

TEST_F(SpaceToBatchTilingTest, BlockSize3WithPadding)
{
    // in [1, 7, 7, 64], bs=3, pad=[2,0,2,0] → H_padded=9, W_padded=9 → out [9, 3, 3, 64]
    std::vector<int32_t> pads = {2, 0, 2, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(3);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 7, 7, 64}, {1, 7, 7, 64}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{9, 3, 3, 64}, {9, 3, 3, 64}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    auto* td = reinterpret_cast<SpaceToBatchTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->blockSize, 3);
    EXPECT_EQ(td->paddings[0][0], 2);
    EXPECT_EQ(td->paddings[1][0], 2);
    EXPECT_EQ(td->outShape[0], 9);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 3);
}

// ---------- Invalid padding (not divisible) should fail ----------

TEST_F(SpaceToBatchTilingTest, InvalidPaddingShouldFail)
{
    // H_padded = 4+1+0=5, not divisible by 2 → should fail
    std::vector<int32_t> pads = {1, 0, 0, 0};
    auto bsAny = Ops::Math::AnyValue::CreateFrom<int64_t>(2);
    gert::TilingContextPara para("SpaceToBatch",
                                 {
                                     {{{1, 4, 6, 256}, {1, 4, 6, 256}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, pads.data()},
                                 },
                                 {
                                     {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                 },
                                 {gert::TilingContextPara::OpAttr("block_size", bsAny)}, &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}
