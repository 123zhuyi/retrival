/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include "conversion/batch_to_space/op_kernel/arch35/batch_to_space_tiling_data.h"
#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class BatchToSpaceTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BatchToSpaceTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "BatchToSpaceTilingTest TearDown" << std::endl; }
};

namespace {
BatchToSpaceCompileInfo compileInfo;
}

// -----------------------------------------------------------------------
// 用例1: C对齐32B, ≥cacheLine → startAxis=C, 全轴遍历, 最后达标者胜
// inShape=[8,28,28,128], bs=2 → outShape=[2,56,56,128]
// axis=3: f=128,core=64 → best. axis=2: f=1..56全部达标 → best=(2,56,112)
// axis=1: f=1→core=56, f=2→core=56 → best=(1,2,56)
// → ubAxis=1, ubFactor=2, totalCount=56
// -----------------------------------------------------------------------
TEST_F(BatchToSpaceTilingTest, TailC_Aligned_Large)
{
    int64_t cropsData[4] = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpace",
        {{{{8, 28, 28, 128}, {8, 28, 28, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
         {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_NHWC, true, cropsData}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo, 64);
    TilingInfo ti;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, ti));
    auto* td = reinterpret_cast<const BatchToSpaceTilingData*>(ti.tilingData.get());
    EXPECT_EQ(td->outShape[0], 2);
    EXPECT_EQ(td->outShape[1], 56);
    EXPECT_EQ(td->outShape[2], 56);
    EXPECT_EQ(td->outShape[3], 128);
    EXPECT_EQ(td->ubFactor, 2u);
    EXPECT_EQ(td->totalCount, 56ull);
}

// -----------------------------------------------------------------------
// 用例2: 尾轴C对齐但 < cacheLineElements → startAxis=W
// inShape=[16, 14, 14, 64], bs=2 → outShape=[4, 28, 28, 64]
// C=64 < 128, W*C=28*64=1792≥128 → startAxis=2(W), minUbFactor=2
// factor=2: totalCount=4*14=56, core=56 ✓
// -----------------------------------------------------------------------
TEST_F(BatchToSpaceTilingTest, TailC_Aligned_Small_StartAtW)
{
    int64_t cropsData[4] = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpace",
        {{{{16, 14, 14, 64}, {16, 14, 14, 64}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
         {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_NHWC, true, cropsData}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo, 64);
    TilingInfo ti;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, ti));
    auto* td = reinterpret_cast<const BatchToSpaceTilingData*>(ti.tilingData.get());
    EXPECT_EQ(td->outShape[0], 4);
    EXPECT_EQ(td->outShape[3], 64);
    EXPECT_EQ(td->ubFactor, 2u);
    EXPECT_EQ(td->totalCount, 56ull);
}

// -----------------------------------------------------------------------
// 用例3: 尾轴C不对齐32B → aligned值参与CacheLine判定
// inShape=[4, 4, 4, 33], bs=2 → outShape=[1, 8, 8, 33]
// C=33, aligned=48 < 128, W*aligned=8*48=384≥128 → startAxis=2, minUbFactor=3
// factor=3: blockSize=144, totalCount=1*3=3, core=3 → best
// -----------------------------------------------------------------------
TEST_F(BatchToSpaceTilingTest, TailC_Unaligned)
{
    int64_t cropsData[4] = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpace",
        {{{{4, 4, 4, 33}, {4, 4, 4, 33}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
         {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_NHWC, true, cropsData}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo, 64);
    TilingInfo ti;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, ti));
    auto* td = reinterpret_cast<const BatchToSpaceTilingData*>(ti.tilingData.get());
    EXPECT_EQ(td->outShape[0], 1);
    EXPECT_EQ(td->outShape[1], 8);
    EXPECT_EQ(td->outShape[2], 8);
    EXPECT_EQ(td->outShape[3], 33);
    EXPECT_EQ(td->ubFactor, 3u);
    EXPECT_EQ(td->totalCount, 24ull);
}

// -----------------------------------------------------------------------
// 用例4: W*C 仍不足 CacheLine → startAxis=H
// inShape=[8, 7, 2, 6], bs=2 → outShape=[2, 14, 4, 6]
// C=6<128, W=4*16=64<128, H=14*64=896≥128 → startAxis=1(H)
// minUbFactor=CeilDiv(128,64)=2, factor=2: blockSize=128, totalCount=2*7=14
// -----------------------------------------------------------------------
TEST_F(BatchToSpaceTilingTest, WC_Insufficient_StartAtH)
{
    int64_t cropsData[4] = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpace",
        {{{{8, 7, 2, 6}, {8, 7, 2, 6}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
         {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_NHWC, true, cropsData}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo, 64);
    TilingInfo ti;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, ti));
    auto* td = reinterpret_cast<const BatchToSpaceTilingData*>(ti.tilingData.get());
    EXPECT_EQ(td->outShape[0], 2);
    EXPECT_EQ(td->outShape[1], 14);
    EXPECT_EQ(td->outShape[2], 4);
    EXPECT_EQ(td->outShape[3], 6);
    EXPECT_EQ(td->ubFactor, 2u);
    EXPECT_EQ(td->totalCount, 14ull);
}

// -----------------------------------------------------------------------
// 用例5: float32, 验证不同dSize的CacheLine判断
// dSize=4, ubBlockElements=8, cacheLineElements=64
// inShape=[4, 7, 7, 16], bs=2 → outShape=[1, 14, 14, 16]
// C=16≥64? No. W*16=14*16=224≥64 → startAxis=2, minUbFactor=CeilDiv(64,16)=4
// -----------------------------------------------------------------------
TEST_F(BatchToSpaceTilingTest, Float32_DifferentCacheLine)
{
    int64_t cropsData[4] = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpace",
        {{{{4, 7, 7, 16}, {4, 7, 7, 16}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
         {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_NHWC, true, cropsData}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo, 64);
    TilingInfo ti;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, ti));
    auto* td = reinterpret_cast<const BatchToSpaceTilingData*>(ti.tilingData.get());
    EXPECT_EQ(td->outShape[0], 1);
    EXPECT_EQ(td->outShape[3], 16);
}

// -----------------------------------------------------------------------
// 用例6: 有crop, 验证outShape推导
// inShape=[32, 8, 8, 32], bs=4, crops=[2,2,2,2]
// N_out = 32/16 = 2
// H_out = 8*4-2-2 = 28
// W_out = 8*4-2-2 = 28
// C_out = 32
// -----------------------------------------------------------------------
TEST_F(BatchToSpaceTilingTest, WithCrops_OutShapeCorrect)
{
    int64_t cropsData[4] = {2, 2, 2, 2};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpace",
        {{{{32, 8, 8, 32}, {32, 8, 8, 32}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
         {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_NHWC, true, cropsData}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(4))}, &compileInfo, 64);
    TilingInfo ti;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, ti));
    auto* td = reinterpret_cast<const BatchToSpaceTilingData*>(ti.tilingData.get());
    EXPECT_EQ(td->outShape[0], 2);
    EXPECT_EQ(td->outShape[1], 28);
    EXPECT_EQ(td->outShape[2], 28);
    EXPECT_EQ(td->outShape[3], 32);
    EXPECT_EQ(td->cropTop, 2);
    EXPECT_EQ(td->cropBottom, 2);
    EXPECT_EQ(td->cropLeft, 2);
    EXPECT_EQ(td->cropRight, 2);
}

// -----------------------------------------------------------------------
// 用例7: N_in 不被 bs^2 整除 → tiling 失败
// -----------------------------------------------------------------------
TEST_F(BatchToSpaceTilingTest, N_NotDivisible_Fail)
{
    int64_t cropsData[4] = {0, 0, 0, 0};
    gert::TilingContextPara tilingContextPara(
        "BatchToSpace",
        {{{{5, 7, 7, 16}, {5, 7, 7, 16}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
         {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_NHWC, true, cropsData}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo, 64);
    TilingInfo ti;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, ti));
}
