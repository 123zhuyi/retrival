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
 * \file test_sort_with_index_branch_tiling.cpp
 * \brief Iteration-3 TilingKey branch coverage UT for SortWithIndex (ascend910b, DAV_2201).
 *
 * dtype scope (D1=1A, spec.yaml v1.2): the 910B first release declares ONLY the 4 int32-index
 * combinations value{fp16,fp32,bf16,int32} x index{int32}. The iteration-2 int64-index positive
 * cases have been REMOVED here: int64-index is no longer a declared (and therefore no longer a
 * tile-able / binary-selectable) combination, so a "successful int64 tiling" UT is misleading.
 * The retained heavy-dtype UB-budget coverage is now driven by the heaviest DECLARED combination,
 * int32-value + int32-index (4-byte value + Cast-to-float key path). int64 non-declaration is
 * verified at the op-info / aclnn layers (op_api UT + _def.cpp grep), not at the dtype-agnostic
 * host-tiling layer.
 *
 * The iteration-1 main-line tiling UT (test_sort_with_index_tiling.cpp) covers the
 * (float16, int32, SIZE_MODE=0) path + big/small core split + unaligned slice + axis-fail.
 * This file extends that to the FULL declared tiling dispatch space:
 *   - all 4 declared (value, index) dtype combinations (value{fp16,fp32,bf16,int32} x index{int32})
 *     -> TilingKey is SIZE_MODE-only (value/index dispatched via DTYPE_* macros), so all combos
 *        share schMode=0 for a single-tile slice; the test asserts the SINGLE TilingKey AND that
 *        the dtype-driven bytesPerElem / UB-budget path still yields a valid single-tile selection.
 *   - SIZE_MODE 0 / 1 / 2 (SINGLE_TILE / MULTI_TILE_MRGSORT / EMPTY).
 *   - dtype-aware padLen: 2-byte (fp16/bf16 -> 16-element/32B align) vs 4-byte (fp32/int32 ->
 *     8-element/32B align). The iteration-3 BUGFIX changed padLen from a fixed 16-element align to
 *     elemsPer32B = 32/sizeof(value): a fixed 16 over-pads 4-byte dtypes at small N (rightPadding
 *     15/12 elements * 4B = 60B/48B > the 32B DataCopyPad rightPadding limit -> 507035). This file
 *     asserts the dtype-aware padLen for BOTH width classes at small (unaligned) N.
 *   - dtype-driven bytesPerElem: bf16 and int32-value walk the float (4B) Cast path; their
 *     per-row UB budget is smaller than the half path but still keeps a small slice single-tile.
 *   - large axis (N > single-tile budget) -> in-core MrgSort UB ceiling exceeded -> graceful
 *     GRAPH_FAILED rejection (D2=2B), with the 32*(power-of-4) realSortLen.
 *   - empty / N==0 / rowNum==0 -> EMPTY (BlockDim==1, validCoreNum==1).
 *   - multi-row big/small core distribution for several dtype combos.
 *
 * Expected tiling values are derived from the host tiling formula (mirrored by the Expect* helpers
 * below) rather than hard-coded magic numbers, so they stay correct if BLOCK_SIZE / alignment change.
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
// Host-tiling constants (mirror op_host/sort_with_index_tiling.cpp).
constexpr uint32_t kBlockSize = 32U;         // Sort granularity (realSortLen = ceil(N, 32) * 32)
constexpr uint32_t kMrgWays = 4U;            // MrgSort fixed 4-way (run count must be a power of 4)
constexpr uint32_t kSortFullSortCap = 8160U; // Sort full-sort hardware cap = 32 * 255

// padLen pads N up to the next 32B boundary IN THE VALUE DTYPE: elemsPer32B = 32/sizeof(value)
// (16 for 2-byte fp16/bf16, 8 for 4-byte fp32/int32). Mirrors the iteration-3 dtype-aware host
// formula (BUGFIX: a fixed 16 over-pads 4-byte dtypes at small N -> rightPadding > 32B -> 507035).
constexpr uint32_t ElemsPer32B(uint32_t valueBytes)
{
    return kBlockSize / valueBytes; // 16 (2-byte) or 8 (4-byte)
}
constexpr uint32_t ExpectPadLen(uint32_t n, uint32_t valueBytes = 2U)
{
    const uint32_t e = ElemsPer32B(valueBytes);
    return (e - (n % e)) % e;
}
constexpr uint32_t ExpectAlign8(uint32_t n, uint32_t valueBytes = 2U)
{
    return n + ExpectPadLen(n, valueBytes);
}
// SINGLE-tile sort length: ceil(N, 32) * 32.
constexpr uint32_t ExpectSingleRealLen(uint32_t n)
{
    return ((n + kBlockSize - 1U) / kBlockSize) * kBlockSize;
}
// MRGSORT sort length: 32 * (power of 4) >= ceil(N, 32) * 32 (keeps run count a power of 4).
uint32_t ExpectMrgsortRealLen(uint32_t n)
{
    const uint32_t base = ExpectSingleRealLen(n);
    uint32_t blocks = base / kBlockSize;
    uint32_t pow4 = 1U;
    while (pow4 < blocks) {
        pow4 *= kMrgWays;
    }
    return pow4 * kBlockSize;
}

// Reinterpret the raw tiling buffer as the typed struct after a successful run.
const SortWithIndexTilingData* AsTilingData(const TilingInfo& info)
{
    EXPECT_GE(info.tilingDataSize, sizeof(SortWithIndexTilingData));
    return reinterpret_cast<const SortWithIndexTilingData*>(info.tilingData.get());
}

// Build a gert::StorageShape from a vector of dims (StorageShape's brace ctor only takes
// initializer_list, so a runtime vector must be filled via the Mutable*Shape() setters).
gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape ss;
    ss.MutableOriginShape().SetDimNum(dims.size());
    ss.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        ss.MutableOriginShape().SetDim(i, dims[i]);
        ss.MutableStorageShape().SetDim(i, dims[i]);
    }
    return ss;
}

// Build a SortWithIndex tiling context. rowNum < 0 builds a 1-D {sliceLen} tensor; rowNum >= 0 builds
// a 2-D {rowNum, sliceLen} tensor (rowNum == 0 -> empty leading dim -> EMPTY path).
gert::TilingContextPara MakeCtx(
    SortWithIndexCompileInfo& compileInfo, int64_t rowNum, int64_t sliceLen, ge::DataType valueDt, ge::DataType indexDt,
    int64_t axis, bool descending, bool stable, uint64_t coreNum, uint64_t ubSize)
{
    std::vector<int64_t> dims;
    if (rowNum >= 0) {
        dims.push_back(rowNum);
    }
    dims.push_back(sliceLen);
    gert::StorageShape shape = MakeStorageShape(dims);
    return gert::TilingContextPara(
        "SortWithIndex",
        {
            {shape, valueDt, ge::FORMAT_ND},
            {shape, indexDt, ge::FORMAT_ND},
        },
        {
            {shape, valueDt, ge::FORMAT_ND},
            {shape, indexDt, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(axis)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(descending)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(stable)},
        },
        &compileInfo, coreNum, ubSize);
}

// Common single-tile assertions for a (rowNum x N) tensor running SIZE_MODE=0 with one row per core
// (rowNum <= coreNum, evenly mapped). valueBytes = sizeof(value dtype) drives the dtype-aware padLen
// (2 for fp16/bf16, 4 for fp32/int32); the default 2 matches the half-path cases.
void CheckSingleTile(const TilingInfo& info, uint32_t rowNum, uint32_t n, uint32_t valueBytes = 2U)
{
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_SINGLE));
    EXPECT_EQ(info.blockNum, rowNum);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->rowNum, rowNum);
    EXPECT_EQ(td->sliceLen, n);
    EXPECT_EQ(td->realSortLen, ExpectSingleRealLen(n));
    EXPECT_EQ(td->padLen, ExpectPadLen(n, valueBytes));
    EXPECT_EQ(td->align8, ExpectAlign8(n, valueBytes));
    EXPECT_EQ(td->dupCount, td->realSortLen - td->align8);
    EXPECT_EQ(td->tileLen, kBlockSize);
    EXPECT_EQ(td->tileCntPerRow, td->realSortLen / kBlockSize);
    EXPECT_EQ(td->validCoreNum, rowNum);
}
} // namespace

class SortWithIndexBranchTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SortWithIndexBranchTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SortWithIndexBranchTilingTest TearDown" << std::endl;
    }
};

// ===========================================================================================
//  Part A: all 4 DECLARED (value, int32) dtype combinations, SIZE_MODE = SINGLE_TILE.
//  Each combo exercises the dtype-driven bytesPerElem / single-tile budget path; the dispatched
//  TilingKey is SIZE_MODE-only (value/index via DTYPE_* macros) so all share schMode=0 here.
//  index is int32 only (D1=1A): int64-index is not a declared combination in the 910B first release.
// ===========================================================================================

// 1/4: (fp16, int32). half direct Sort path, int32 single-Gather index path. 2-byte value padLen.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_dtype_fp16_int32_single)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/3, /*N=*/8, ge::DT_FLOAT16, ge::DT_INT32, -1, false, false, 64U, 262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    CheckSingleTile(info, 3U, 8U, /*valueBytes=*/2U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->padLen, ExpectPadLen(8U, /*valueBytes=*/2U)); // 2-byte: (16 - 8%16)%16 = 8
}

// 2/4: (fp32, int32). float direct Sort path (no Cast), int32 index. 4-byte value: N=16 already
// 8-aligned -> padLen=0; this is the heaviest non-Cast direct-float path.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_dtype_fp32_int32_single)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/2, /*N=*/16, ge::DT_FLOAT, ge::DT_INT32, -1, false, false, 64U, 262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    CheckSingleTile(info, 2U, 16U, /*valueBytes=*/4U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->padLen, 0U); // 4-byte: N=16 already 8-aligned -> (8 - 16%8)%8 = 0
}

// 3/4: (bf16, int32). bf16 -> Cast to float (4B) Sort path, int32 index. 2-byte value padLen.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_dtype_bf16_int32_single)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/4, /*N=*/32, ge::DT_BF16, ge::DT_INT32, -1, false, true, 64U, 262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    CheckSingleTile(info, 4U, 32U, /*valueBytes=*/2U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->dupCount, 0U); // N=32 == realSortLen, no sentinel fill
    EXPECT_EQ(td->stable, true); // attribute propagated
}

// 4/4: (int32-value, int32). int32 value -> Cast to float (|x|<=2^24) Sort path, int32 index.
// 4-byte value: this is now the heaviest DECLARED combination (Cast-to-float key buffers).
TEST_F(SortWithIndexBranchTilingTest, test_tiling_dtype_int32_int32_single)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/2, /*N=*/64, ge::DT_INT32, ge::DT_INT32, -1, false, false, 64U, 262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    CheckSingleTile(info, 2U, 64U, /*valueBytes=*/4U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->realSortLen, 64U);  // ceil(64,32)*32 = 64
    EXPECT_EQ(td->tileCntPerRow, 2U); // 64 / 32 = two 32-blocks (still single-tile full Sort)
    EXPECT_EQ(td->padLen, 0U);        // 4-byte: N=64 already 8-aligned
}

// ===========================================================================================
//  Part A2: dtype-aware padLen at SMALL / UNALIGNED N -- both width classes (iteration-3 BUGFIX).
//  2-byte dtypes pad to a 16-element boundary; 4-byte dtypes pad to an 8-element boundary so the
//  DataCopyPad rightPadding (padLen * sizeof(value)) never exceeds the 32B hardware limit (which
//  the old fixed 16-element pad violated for 4-byte dtypes at small N -> 507035).
// ===========================================================================================

// fp16 (2-byte) small N=5: padLen = (16 - 5%16)%16 = 11 -> rightPadding = 11*2 = 22B (<= 32B).
TEST_F(SortWithIndexBranchTilingTest, test_tiling_padlen_fp16_small_n)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/2, /*N=*/5, ge::DT_FLOAT16, ge::DT_INT32, -1, false, false, 64U, 262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    CheckSingleTile(info, 2U, 5U, /*valueBytes=*/2U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->padLen, 11U);      // (16 - 5%16)%16 = 11
    EXPECT_LE(td->padLen * 2U, 32U); // rightPadding <= 32B (no 507035)
}

// fp32 (4-byte) small N=1: padLen = (8 - 1%8)%8 = 7 -> rightPadding = 7*4 = 28B (<= 32B). A fixed
// 16-element pad would give 15*4 = 60B > 32B -> 507035 (the exact iteration-3 defect #1 case).
TEST_F(SortWithIndexBranchTilingTest, test_tiling_padlen_fp32_small_n1)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/4, /*N=*/1, ge::DT_FLOAT, ge::DT_INT32, -1, false, false, 64U, 262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    CheckSingleTile(info, 4U, 1U, /*valueBytes=*/4U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->padLen, ExpectPadLen(1U, /*valueBytes=*/4U)); // (8 - 1%8)%8 = 7
    EXPECT_EQ(td->padLen, 7U);
    EXPECT_LE(td->padLen * 4U, 32U); // 28B <= 32B (no 507035)
}

// fp32 (4-byte) N=4: padLen = (8 - 4%8)%8 = 4 -> rightPadding = 4*4 = 16B (<= 32B). The second
// iteration-3 defect #1 ST case (FULL-float32-N4). A fixed 16-element pad would give 12*4 = 48B.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_padlen_fp32_small_n4)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/2, /*N=*/4, ge::DT_FLOAT, ge::DT_INT32, -1, false, false, 64U, 262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    CheckSingleTile(info, 2U, 4U, /*valueBytes=*/4U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->padLen, 4U);       // (8 - 4%8)%8 = 4
    EXPECT_LE(td->padLen * 4U, 32U); // 16B <= 32B (no 507035)
}

// int32-value (4-byte) small N=5: padLen = (8 - 5%8)%8 = 3 -> rightPadding = 3*4 = 12B (<= 32B).
TEST_F(SortWithIndexBranchTilingTest, test_tiling_padlen_int32_small_n5)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/4, /*N=*/5, ge::DT_INT32, ge::DT_INT32, -1, false, false, 64U, 262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    CheckSingleTile(info, 4U, 5U, /*valueBytes=*/4U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->padLen, ExpectPadLen(5U, /*valueBytes=*/4U)); // (8 - 5%8)%8 = 3
    EXPECT_EQ(td->padLen, 3U);
    EXPECT_LE(td->padLen * 4U, 32U); // 12B <= 32B (no 507035)
}

// ===========================================================================================
//  Part B: SIZE_MODE = 1 (MULTI_TILE_MRGSORT) UB ceiling -> graceful rejection (D2=2B). The in-core
//  MrgSort holds the whole row's proposal buffers in UB; its per-element cost (incl. propB ping-pong)
//  is HIGHER than single-tile, while realSortLen can only take coarse 32*(power-of-4) steps. As a
//  result mrgRealLenCap (the largest UB-safe 32*4^k) is ALWAYS <= singleTileLimit, so any N large
//  enough to leave single-tile also exceeds the in-core MrgSort UB ceiling. The host therefore
//  REJECTS such N with a graceful tiling failure (GRAPH_FAILED) instead of overflowing UB on board
//  (which manifested as a 507035 "MPU address access invalid" aivec exception during ST iter2 NPU
//  runs). True large-axis support needs GM-workspace multi-tile merge (DESIGN 3.3.7, future
//  iteration). See issues/issue_20260531_iter2_int64_bin_and_mrgsort_ub_1.md (defect B).
// ===========================================================================================

// A small UB budget pushes N=64 past the single-tile limit; the required MrgSort realSortLen (128)
// exceeds the in-core MrgSort UB ceiling at ub=2048, so tiling fails gracefully (no MRGSORT emitted).
TEST_F(SortWithIndexBranchTilingTest, test_tiling_mrgsort_small_ub_rejected)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(
        ci, /*rows=*/2, /*N=*/64, ge::DT_FLOAT16, ge::DT_INT32, -1, false, false,
        /*coreNum=*/8U, /*ubSize=*/2048U);
    TilingInfo info;
    // Required MrgSort realSortLen (128) > mrgRealLenCap at ub=2048 -> host returns GRAPH_FAILED.
    EXPECT_FALSE(ExecuteTiling(ctx, info));
}

// A genuinely large axis at the default 256KB UB: N=8192 leaves single-tile and would need MrgSort
// realSortLen=8192, far above the in-core UB ceiling (mrgRealLenCap=2048 for fp16+int32). Rejected.
// (Mirrors ST negative case FULL-fp16-N8192 561103 graceful rejection.)
TEST_F(SortWithIndexBranchTilingTest, test_tiling_mrgsort_large_axis_rejected)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(
        ci, /*rows=*/2, /*N=*/8192, ge::DT_FLOAT16, ge::DT_INT32, -1, false, false,
        /*coreNum=*/8U, /*ubSize=*/262144U);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(ctx, info)); // exceeds in-core MrgSort UB ceiling -> GRAPH_FAILED
}

// Just past the single-tile budget for the heaviest DECLARED dtype (int32-value + int32-index, the
// Cast-to-float key path): even a modestly larger N exceeds the (smaller) in-core MrgSort ceiling
// for this dtype and is rejected. N=4096 is well above the int32+int32 single-tile limit and its
// MrgSort realSortLen (8192) blows the ceiling.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_mrgsort_heavy_dtype_rejected)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(
        ci, /*rows=*/2, /*N=*/4096, ge::DT_INT32, ge::DT_INT32, -1, false, false,
        /*coreNum=*/8U, /*ubSize=*/262144U);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(ctx, info)); // exceeds in-core MrgSort UB ceiling -> GRAPH_FAILED
}

// ===========================================================================================
//  Part B2: 85% single-tile UB budget (iteration-3 defect #2). The single tile does NOT allocate
//  propB, so its budget is widened to 85% (MrgSort stays at 75%). A mid-size axis that fits the
//  single-tile budget but whose MrgSort realSortLen would exceed the (narrower) MrgSort ceiling must
//  still be accepted as a SINGLE tile -- i.e. the wider single-tile budget keeps near-upper-bound N
//  on the single-tile path rather than falsely pushing it onto the rejected MrgSort path.
// ===========================================================================================

// N=128 (4 32-blocks) at full 256KB UB: singleRealLen=128 is comfortably within the 85% single-tile
// budget for fp16, so SIZE_MODE stays SINGLE (not MrgSort, not rejected). Asserts the budget keeps a
// multi-32-block-but-still-single-tile slice on the SINGLE path.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_single_budget_midsize_accepted)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(
        ci, /*rows=*/2, /*N=*/128, ge::DT_FLOAT16, ge::DT_INT32, -1, false, false,
        /*coreNum=*/8U, /*ubSize=*/262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_SINGLE));
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->sliceLen, 128U);
    EXPECT_EQ(td->realSortLen, ExpectSingleRealLen(128U)); // 128
    EXPECT_EQ(td->tileCntPerRow, 4U);                      // 128 / 32 = 4 blocks, one full Sort
}

// int32-value (heaviest declared, 4-byte Cast path) mid-size N=256 at full UB: still within the 85%
// single-tile budget, so SIZE_MODE stays SINGLE even for the heavier per-element footprint.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_single_budget_heavy_dtype_accepted)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(
        ci, /*rows=*/2, /*N=*/256, ge::DT_INT32, ge::DT_INT32, -1, false, false,
        /*coreNum=*/8U, /*ubSize=*/262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));
    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_SINGLE));
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->sliceLen, 256U);
    EXPECT_EQ(td->realSortLen, ExpectSingleRealLen(256U)); // 256
}

// ===========================================================================================
//  Part C: SIZE_MODE = 2 (EMPTY). empty tensor / N==0 / rowNum==0 -> BlockDim==1, validCoreNum==1.
//  All combos use declared int32-index dtypes.
// ===========================================================================================

// 1-D empty tensor {0}: sliceLen=0 -> EMPTY.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_empty_1d_zero_slice)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/-1 /*1-D*/, /*N=*/0, ge::DT_FLOAT16, ge::DT_INT32, -1, false, false, 64U, 262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_EMPTY));
    EXPECT_EQ(info.blockNum, 1U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->validCoreNum, 1U);
}

// 2-D with last axis = 0 ({4, 0}): sliceLen=0 -> EMPTY (rowNum=4 but no elements per row).
// fp32 + int32 (declared 4-byte value combo).
TEST_F(SortWithIndexBranchTilingTest, test_tiling_empty_2d_zero_slice)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/4, /*N=*/0, ge::DT_FLOAT, ge::DT_INT32, -1, false, false, 64U, 262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_EMPTY));
    EXPECT_EQ(info.blockNum, 1U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->validCoreNum, 1U);
}

// 2-D with leading dim = 0 ({0, 8}): rowNum=0 -> EMPTY. bf16 + int32.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_empty_zero_rows)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/0, /*N=*/8, ge::DT_BF16, ge::DT_INT32, -1, false, false, 64U, 262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_EMPTY));
    EXPECT_EQ(info.blockNum, 1U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->validCoreNum, 1U);
}

// ===========================================================================================
//  Part D: multi-row big/small core distribution for non-fp16 declared dtype combos (exercises the
//  big/small split with the cast-path and direct-float bytesPerElem still selecting single-tile).
// ===========================================================================================

// fp32 + int32, 7 rows over 4 cores: smallCoreRowNum=1, remainder=3 -> 3 big (2 rows), 1 small (1 row).
TEST_F(SortWithIndexBranchTilingTest, test_tiling_fp32_int32_big_small_core_split)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(
        ci, /*rows=*/7, /*N=*/16, ge::DT_FLOAT, ge::DT_INT32, -1, false, false,
        /*coreNum=*/4U, /*ubSize=*/262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_SINGLE));
    EXPECT_EQ(info.blockNum, 4U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->rowNum, 7U);
    EXPECT_EQ(td->validCoreNum, 4U);
    EXPECT_EQ(td->bigCoreNum, 3U);
    EXPECT_EQ(td->smallCoreNum, 1U);
    EXPECT_EQ(td->bigCoreRowNum, 2U);
    EXPECT_EQ(td->smallCoreRowNum, 1U);
}

// int32-value + int32, more rows than cores: 10 rows over 4 cores -> small=2, remainder=2 -> 2 big (3),
// 2 small (2). Validates the cast-path bytesPerElem still single-tiles and the row split is balanced.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_int32_int32_rows_exceed_cores)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(
        ci, /*rows=*/10, /*N=*/16, ge::DT_INT32, ge::DT_INT32, -1, false, false,
        /*coreNum=*/4U, /*ubSize=*/262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_SINGLE));
    EXPECT_EQ(info.blockNum, 4U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->rowNum, 10U);
    EXPECT_EQ(td->validCoreNum, 4U);
    EXPECT_EQ(td->bigCoreNum, 2U); // remainder = 10 % 4 = 2
    EXPECT_EQ(td->smallCoreNum, 2U);
    EXPECT_EQ(td->bigCoreRowNum, 3U);   // smallCoreRowNum(2) + 1
    EXPECT_EQ(td->smallCoreRowNum, 2U); // 10 / 4 = 2
}

// bf16 + int32, rows exactly divisible by cores (8 rows over 4 cores) -> no remainder, all small
// cores (2 rows each), bigCoreNum=0. Exercises the no-remainder balanced split on the cast path.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_bf16_int32_even_core_split)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(
        ci, /*rows=*/8, /*N=*/32, ge::DT_BF16, ge::DT_INT32, -1, true, true,
        /*coreNum=*/4U, /*ubSize=*/262144U);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    EXPECT_EQ(info.tilingKey, static_cast<int64_t>(SORT_WITH_INDEX_SIZE_MODE_SINGLE));
    EXPECT_EQ(info.blockNum, 4U);
    const SortWithIndexTilingData* td = AsTilingData(info);
    EXPECT_EQ(td->rowNum, 8U);
    EXPECT_EQ(td->validCoreNum, 4U);
    EXPECT_EQ(td->bigCoreNum, 0U); // 8 % 4 == 0, no big core
    EXPECT_EQ(td->smallCoreNum, 4U);
    EXPECT_EQ(td->smallCoreRowNum, 2U); // 8 / 4 = 2 rows per core
    EXPECT_EQ(td->bigCoreRowNum, 2U);   // == smallCoreRowNum when remainder == 0
    EXPECT_EQ(td->descending, true);    // attributes propagated
    EXPECT_EQ(td->stable, true);
}

// ===========================================================================================
//  Part E: exception branches (must return GRAPH_FAILED).
// ===========================================================================================

// axis = 0 with rank 2 -> not last dim -> GRAPH_FAILED (attribute_value_out_of_range).
TEST_F(SortWithIndexBranchTilingTest, test_tiling_axis_zero_rank2_fail)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/4, /*N=*/8, ge::DT_FLOAT16, ge::DT_INT32, /*axis=*/0, false, false, 64U, 262144U);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

// axis out of [-rank, rank) entirely (axis = 5, rank = 2) -> GRAPH_FAILED.
TEST_F(SortWithIndexBranchTilingTest, test_tiling_axis_out_of_range_fail)
{
    SortWithIndexCompileInfo ci;
    auto ctx = MakeCtx(ci, /*rows=*/4, /*N=*/8, ge::DT_FLOAT16, ge::DT_INT32, /*axis=*/5, false, false, 64U, 262144U);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}
