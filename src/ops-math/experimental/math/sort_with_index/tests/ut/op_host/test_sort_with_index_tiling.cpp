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
 * \file test_sort_with_index_tiling.cpp
 * \brief Core-path tiling UT for SortWithIndex (ascend910b, DAV_2201, iteration-1).
 *
 * Main-line branch: TilingKey (x = float16, index = int32, SIZE_MODE = 0 / SINGLE_TILE).
 * Verifies the host tiling computation (padding / sort-length / row distribution), the
 * SIZE_MODE selection, and multi-core row partition. Tiling data is asserted on the typed
 * SortWithIndexTilingData fields (read from the raw buffer) rather than a brittle uint64
 * decimal string, and the dispatched TilingKey is read from the run.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "sort_with_index_tiling_ut.h"
#include "../../../op_kernel/sort_with_index_tiling_data.h"
#include "../../../op_kernel/sort_with_index_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

namespace {
constexpr uint32_t kBlockSize = 32U; // Sort granularity (realSortLen = ceil(N, 32) * 32)
// This file only exercises the fp16 (2-byte) value path, where the iteration-3 dtype-aware host
// padLen formula elemsPer32B = 32/sizeof(value) reduces to a 16-element / 32B boundary (32/2 = 16).
// The 4-byte (fp32/int32) -> 8-element padLen variant is covered in test_sort_with_index_branch_tiling.cpp.
constexpr uint32_t kPadAlign = 16U; // DataCopyPad tail alignment for 2-byte half (16 elems = 32B)

// Mirror the host padLen formula for the 2-byte half path: pad N up to the 16-element / 32B boundary.
// Using rightPadding != 0 whenever N is not 16-aligned is REQUIRED so DataCopyPad dummy-fill uses the
// sentinel, not element[0] (validated in probe_fp16_int32_mrgsort; the old align8-based padLen could be
// 0 for N==8/16/24...).
constexpr uint32_t ExpectPadLen(uint32_t n)
{
    return (kPadAlign - (n % kPadAlign)) % kPadAlign;
}
constexpr uint32_t ExpectAlign8(uint32_t n)
{
    return n + ExpectPadLen(n);
}
constexpr uint32_t ExpectRealSortLen(uint32_t n)
{
    return ((n + kBlockSize - 1U) / kBlockSize) * kBlockSize;
}

// Reinterpret the raw tiling buffer as the typed struct after a successful run.
const SortWithIndexTilingData* AsTilingData(const TilingInfo& info)
{
    EXPECT_GE(info.tilingDataSize, sizeof(SortWithIndexTilingData));
    return reinterpret_cast<const SortWithIndexTilingData*>(info.tilingData.get());
}
} // namespace

class SortWithIndexTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SortWithIndexTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SortWithIndexTilingTest TearDown" << std::endl;
    }
};

// Main line: (float16, int32), SIZE_MODE = SINGLE. shape {3,8}: rowNum=3 < coreNum -> 3 cores,
// each core gets exactly one row (no remainder, no big core). Verifies padding + row split.
TEST_F(SortWithIndexTilingTest, test_tiling_fp16_int32_single_main_2d)
{
    SortWithIndexCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SortWithIndex",
        {
            {{{3, 8}, {3, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 8}, {3, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 8}, {3, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 8}, {3, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    // SIZE_MODE = SINGLE (0) is the iteration-1 main line; the TPL TilingKey encodes schMode=0.
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_SINGLE));
    // 3 rows distributed to 3 cores (one row each).
    EXPECT_EQ(tilingInfo.blockNum, 3U);

    const SortWithIndexTilingData* td = AsTilingData(tilingInfo);
    // shape-derived fields
    EXPECT_EQ(td->rowNum, 3U);
    EXPECT_EQ(td->sliceLen, 8U);
    EXPECT_EQ(td->realSortLen, ExpectRealSortLen(8U));     // 32
    EXPECT_EQ(td->align8, ExpectAlign8(8U));               // N(8) + padLen(8) = 16
    EXPECT_EQ(td->padLen, ExpectPadLen(8U));               // (16 - 8%16)%16 = 8
    EXPECT_EQ(td->dupCount, td->realSortLen - td->align8); // 32 - 16 = 16
    // tileLen = per-32-element sorted-run carrier (BLOCK_SIZE); tileCntPerRow = realSortLen/32 runs.
    EXPECT_EQ(td->tileLen, kBlockSize);                         // 32
    EXPECT_EQ(td->tileCntPerRow, td->realSortLen / kBlockSize); // 32/32 = 1 (single 32-block)
    // multi-core row distribution: 3 rows / 3 cores, no remainder
    EXPECT_EQ(td->validCoreNum, 3U);
    EXPECT_EQ(td->smallCoreNum, 3U);
    EXPECT_EQ(td->bigCoreNum, 0U);
    EXPECT_EQ(td->smallCoreRowNum, 1U);
    EXPECT_EQ(td->bigCoreRowNum, 1U);
    // attributes
    EXPECT_EQ(td->axis, 1U); // normalized last axis (rank-1)
    EXPECT_EQ(td->descending, false);
    EXPECT_EQ(td->stable, false);
}

// Big/small core split: rowNum (7) not divisible by core count. With coreNum=4, workCoreNum=4,
// smallCoreRowNum=1, remainder=3 -> 3 big cores (2 rows), 1 small core (1 row).
TEST_F(SortWithIndexTilingTest, test_tiling_fp16_int32_big_small_core_split)
{
    SortWithIndexCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SortWithIndex",
        {
            {{{7, 16}, {7, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7, 16}, {7, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{7, 16}, {7, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{7, 16}, {7, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo,
        /*coreNum=*/4U);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_SINGLE));
    EXPECT_EQ(tilingInfo.blockNum, 4U);

    const SortWithIndexTilingData* td = AsTilingData(tilingInfo);
    EXPECT_EQ(td->rowNum, 7U);
    EXPECT_EQ(td->sliceLen, 16U);
    EXPECT_EQ(td->realSortLen, ExpectRealSortLen(16U)); // 32
    EXPECT_EQ(td->align8, ExpectAlign8(16U));           // N(16) + padLen(0) = 16 (already 16-aligned)
    EXPECT_EQ(td->padLen, ExpectPadLen(16U));           // (16 - 16%16)%16 = 0
    EXPECT_EQ(td->dupCount, 16U);                       // 32 - 16
    // 7 rows over 4 cores: smallCoreRowNum = 1, remainder = 3
    EXPECT_EQ(td->validCoreNum, 4U);
    EXPECT_EQ(td->bigCoreNum, 3U);
    EXPECT_EQ(td->smallCoreNum, 1U);
    EXPECT_EQ(td->bigCoreRowNum, 2U);
    EXPECT_EQ(td->smallCoreRowNum, 1U);
    // attributes propagated
    EXPECT_EQ(td->axis, 1U);
    EXPECT_EQ(td->descending, true);
    EXPECT_EQ(td->stable, true);
}

// Non-16-aligned sort length: shape {2, 5} -> padLen = (16-5)%16 = 11, align8 = 16, realSortLen = 32.
TEST_F(SortWithIndexTilingTest, test_tiling_fp16_int32_unaligned_slice)
{
    SortWithIndexCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SortWithIndex",
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 5}, {2, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 5}, {2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 5}, {2, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_SINGLE));

    const SortWithIndexTilingData* td = AsTilingData(tilingInfo);
    EXPECT_EQ(td->sliceLen, 5U);
    EXPECT_EQ(td->align8, ExpectAlign8(5U));           // N(5) + padLen(11) = 16
    EXPECT_EQ(td->padLen, ExpectPadLen(5U));           // (16 - 5%16)%16 = 11
    EXPECT_EQ(td->realSortLen, ExpectRealSortLen(5U)); // 32
    EXPECT_EQ(td->dupCount, 16U);                      // 32 - 16
}

// axis out of last-dim range -> tiling fails (GRAPH_FAILED). axis=0 with rank=2 is not last dim.
TEST_F(SortWithIndexTilingTest, test_tiling_axis_not_last_dim_fail)
{
    SortWithIndexCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "SortWithIndex",
        {
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)}, // not last dim
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        },
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}
