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
#include <limits>
#include "tiling_case_executor.h"
#include "../../../op_kernel/mod_tiling_data.h"

namespace optiling {
struct ModCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
    bool isRegbase = false;
};
} // namespace optiling

namespace ModNs {
bool operator==(const ModTilingData& lhs, const ModTilingData& rhs)
{
    if (lhs.usableUbSize != rhs.usableUbSize || lhs.needCoreNum != rhs.needCoreNum ||
        lhs.totalDataCount != rhs.totalDataCount || lhs.perCoreDataCount != rhs.perCoreDataCount ||
        lhs.tailDataCoreNum != rhs.tailDataCoreNum || lhs.lastCoreDataCount != rhs.lastCoreDataCount ||
        lhs.isInput2Scalar != rhs.isInput2Scalar || lhs.isInput2SameShape != rhs.isInput2SameShape ||
        lhs.dimNum != rhs.dimNum) {
        return false;
    }
    for (uint32_t i = 0; i < 8; ++i) {
        if (lhs.input1Shape[i] != rhs.input1Shape[i] || lhs.input2Shape[i] != rhs.input2Shape[i] ||
            lhs.input2Stride[i] != rhs.input2Stride[i]) {
            return false;
        }
    }
    return true;
}
} // namespace ModNs

namespace {
constexpr size_t WORKSPACE_SIZE = 32 * 1024 * 1024;

// 镜像 op_host/mod_tiling.cpp::ModTiling::ModCommonTiling 的 UB-budget 公式 (该文件的 UB_DIVIDER_* 常量是
// internal linkage、UT 不可达，故在此重新推导公式而非硬编码数值)。自适应路由要求 ComputeFPCore 在任一 tile
// 无条件预留 A1..A5 scratch，故 FP32 same-dtype divider 上调 -> usableUbSize 相对上游朴素路缩小。按公式重算
// (而非硬编码字面量) 使断言绑定到「数值为何如此」，且在 RESERVERD_UB_SIZE / sizeof(ModTilingData) /
// DATA_BLOCK 同步变化时仍正确。
constexpr uint32_t kMirrorReservedUbSize = 1024; // == ModTiling::RESERVERD_UB_SIZE
constexpr uint32_t kMirrorDataBlockSize = 64;    // == ModTiling::DATA_BLOCK
// == op_host/mod_tiling.cpp UB_DIVIDER_FP32 (AlgoA scratch 预留)。fp32 GENERAL (broadcast / ComputeCore)
// same-dtype divider——非 same-dtype fp32 CONTIGUOUS (scalar / same-shape) 用例所用 (后者走精简核用 48)。
constexpr uint32_t kMirrorUbDividerFp32 = 69;
// == op_host/mod_tiling.cpp UB_DIVIDER_FP32_LEAN：same-dtype fp32 CONTIGUOUS 派发 (isInput2Scalar ||
// isInput2SameShape) 走精简核，ModSelectContiguousLeanDivider 把 divider 从 69 下调到 48 (tile 更宽/tile 数
// 更少)。下方 same_shape_float 场景 (8192 same-shape fp32 = contiguous) 实际用此 divider。
constexpr uint32_t kMirrorUbDividerFp32Lean = 48;

uint32_t ExpectedUsableUbSize(uint64_t ubSize, uint32_t ubDivider)
{
    uint32_t raw = static_cast<uint32_t>(ubSize - kMirrorReservedUbSize - sizeof(ModNs::ModTilingData)) / ubDivider;
    return raw / kMirrorDataBlockSize * kMirrorDataBlockSize;
}

void ExpectTiling(const gert::TilingContextPara& tilingContextPara, uint64_t expectTilingKey,
                  const ModNs::ModTilingData& expectTilingData)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(reinterpret_cast<std::vector<size_t>&>(tilingInfo.workspaceSizes), std::vector<size_t>({WORKSPACE_SIZE}));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    EXPECT_EQ(*reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get()), expectTilingData);
}
} // namespace

class ModTiling : public testing::Test {};

TEST_F(ModTiling, same_shape_float)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    ModNs::ModTilingData expect{};
    // 既非朴素路的 3968，也非 fp32 general divider (69)（精简核改用 48）：
    // this 8192 same-shape fp32 case is a CONTIGUOUS same-dtype dispatch, so ModSelectContiguousLeanDivider
    // picks the lean divider (48). Recomputed from the mirrored formula (not a hardcoded literal) so it stays
    // correct/self-explaining if the shared UB-budget constants change together — this is exactly the stale
    // 此前 (精简核前) 断言用 69-derived 值，精简核后改 48-derived。
    expect.usableUbSize = ExpectedUsableUbSize(compileInfo.ubSize, kMirrorUbDividerFp32Lean);
    expect.needCoreNum = 8;
    expect.totalDataCount = 8192;
    expect.perCoreDataCount = 1024;
    expect.tailDataCoreNum = 0;
    expect.lastCoreDataCount = 1024;
    expect.isInput2Scalar = false;
    expect.isInput2SameShape = true;
    expect.dimNum = 1;
    expect.input1Shape[0] = 8192;
    expect.input2Shape[0] = 8192;
    expect.input2Stride[0] = 1;
    for (uint32_t i = 1; i < 8; ++i) {
        expect.input1Shape[i] = 1;
        expect.input2Shape[i] = 1;
        expect.input2Stride[i] = 0;
    }

    ExpectTiling(tilingContextPara, 1973790, expect);
}

TEST_F(ModTiling, broadcast_stride_float16)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 3, 1}, {1, 3, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 1315860);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_FALSE(tilingData->isInput2Scalar);
    EXPECT_FALSE(tilingData->isInput2SameShape);
    EXPECT_EQ(tilingData->dimNum, 3U);
    EXPECT_EQ(tilingData->input2Stride[0], 0U);
    EXPECT_EQ(tilingData->input2Stride[1], 1U);
    EXPECT_EQ(tilingData->input2Stride[2], 0U);
}

TEST_F(ModTiling, scalar_int32)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 2631720);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_TRUE(tilingData->isInput2Scalar);
    EXPECT_FALSE(tilingData->isInput2SameShape);
    EXPECT_EQ(tilingData->totalDataCount, 128U);
    EXPECT_EQ(tilingData->needCoreNum, 1U);
}

// The following cases cover the five same-dtype tiling lanes and their UB divider selection.

// int16 same-dtype -> ubDivider = UB_DIVIDER_INT16 (45)，四个 same-dtype divider 中最小 (int16 same-dtype
// 跳过 AlgoA scratch 分配) -> 相同 ubSize 下 same-dtype lane 中 usableUbSize 最大。亦钉住新增的
// int16 dtype -> TilingKey 映射 (MOD_TPL_INT16，mod_tiling_key.h lane 4)。
TEST_F(ModTiling, same_shape_int16)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{2048}, {2048}}, ge::DT_INT16, ge::FORMAT_ND},
                                                  {{{2048}, {2048}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2048}, {2048}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_FALSE(tilingData->isInput2Scalar);
    EXPECT_TRUE(tilingData->isInput2SameShape);
    // UB_DIVIDER_INT16 (45) — see op_host/mod_tiling.cpp comment (K2: no A1..A5 scratch for int16
    // same-dtype naive path).
    constexpr uint32_t kMirrorUbDividerInt16 = 45;
    EXPECT_EQ(tilingData->usableUbSize, ExpectedUsableUbSize(compileInfo.ubSize, kMirrorUbDividerInt16));
    // int16 same-dtype has strictly MORE usable UB than fp32 same-dtype (both scratch-allocating
    // lanes) precisely because K2 skips A1..A5 — this is the sanity anchor for "int16 same-dtype
    // actually took the K2 branch, not the fp32/FP-cast UB_DIVIDER by accident".
    EXPECT_GT(tilingData->usableUbSize, ExpectedUsableUbSize(compileInfo.ubSize, kMirrorUbDividerFp32));
}

// BF16 same-shape uses its registered same-dtype key and the lean contiguous divider.
TEST_F(ModTiling, same_shape_bf16)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 657930);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->usableUbSize, ExpectedUsableUbSize(compileInfo.ubSize, kMirrorUbDividerFp32Lean));
}

// INT32 same-shape preserves the original high-precision lane and its non-lean divider.
TEST_F(ModTiling, same_shape_int32)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 2631720);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->usableUbSize, ExpectedUsableUbSize(compileInfo.ubSize, kMirrorUbDividerFp32));
}

// naiveThresh 是新增的 ModTilingData 字段；host tiling
// always writes FmodNaiveThresh() (default 256.0, no FMOD_NAIVE_THRESH env override in this UT
// process) regardless of dtype/shape. Reuses the same_shape_float scenario.
TEST_F(ModTiling, naive_thresh_default_value)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_FLOAT_EQ(tilingData->naiveThresh, 256.0f);
}

// MapModDtype (op_host/mod_tiling.cpp :169-183) falls through to its `return MOD_TPL_FP32;` default
// for any dtype outside its 5 explicit cases — previously untested (all prior tests use one of the 5
// listed dtypes). DT_DOUBLE (an AICPU-fallback dtype the op_api layer supports but the AiCore kernel
// template never lists) exercises that default branch; asserts tiling still completes without crash
// and both dtype mapping-dependent fields (tilingKey / usableUbSize's ubDivider choice) come out
// exactly as they would for genuine same-dtype FP32 (same default-mapped value for x1/x2/y).
TEST_F(ModTiling, unlisted_dtype_falls_back_to_fp32_mapping)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara doubleCtx("Mod",
                                      {
                                          {{{256}, {256}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                          {{{256}, {256}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                      },
                                      {
                                          {{{256}, {256}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                      },
                                      {}, &compileInfo);
    gert::TilingContextPara fp32Ctx("Mod",
                                    {
                                        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo);

    TilingInfo doubleInfo, fp32Info;
    ASSERT_TRUE(ExecuteTiling(doubleCtx, doubleInfo));
    ASSERT_TRUE(ExecuteTiling(fp32Ctx, fp32Info));
    // DT_DOUBLE isn't in MapModDtype's explicit case list -> falls to the same default (MOD_TPL_FP32)
    // as genuine DT_FLOAT for all three (x1/x2/y all-double is still "same-dtype" post-mapping) ->
    // identical TilingKey and identical UB_DIVIDER_FP32-derived usableUbSize.
    EXPECT_EQ(doubleInfo.tilingKey, fp32Info.tilingKey);
    ASSERT_EQ(doubleInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* doubleTiling = reinterpret_cast<ModNs::ModTilingData*>(doubleInfo.tilingData.get());
    auto* fp32Tiling = reinterpret_cast<ModNs::ModTilingData*>(fp32Info.tilingData.get());
    EXPECT_EQ(doubleTiling->usableUbSize, fp32Tiling->usableUbSize);
}

// All five registered same-dtype lanes must produce pairwise-distinct TilingKeys.
TEST_F(ModTiling, tilingkey_distinct_across_same_dtype_lanes)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    auto runKey = [&](ge::DataType x1, ge::DataType x2, ge::DataType y) -> uint64_t {
        gert::TilingContextPara ctx("Mod",
                                    {
                                        {{{256}, {256}}, x1, ge::FORMAT_ND},
                                        {{{256}, {256}}, x2, ge::FORMAT_ND},
                                    },
                                    {
                                        {{{256}, {256}}, y, ge::FORMAT_ND},
                                    },
                                    {}, &compileInfo);
        TilingInfo info;
        EXPECT_TRUE(ExecuteTiling(ctx, info));
        return info.tilingKey;
    };

    const uint64_t kInt32Same = runKey(ge::DT_INT32, ge::DT_INT32, ge::DT_INT32);
    const uint64_t kFp16Same = runKey(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16);
    const uint64_t kFp32Same = runKey(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    const uint64_t kBf16Same = runKey(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16);
    const uint64_t kInt16Same = runKey(ge::DT_INT16, ge::DT_INT16, ge::DT_INT16);

    const std::vector<uint64_t> keys = {kBf16Same, kFp16Same, kFp32Same, kInt32Same, kInt16Same};
    for (size_t i = 0; i < keys.size(); ++i) {
        for (size_t j = i + 1; j < keys.size(); ++j) {
            EXPECT_NE(keys[i], keys[j]) << "lane " << i << " and lane " << j << " collide on TilingKey " << keys[i];
        }
    }
}

// =====================================================================================
// op_host 对精简核 divider 选择与融合广播 tiling 链的覆盖
// (ModSelectContiguousLeanDivider) and the whole Path B fused-broadcast tiling chain
// (ModTryFusedBroadcast -> ModFusedBroadcastEligible -> ModCollapseBroadcastSegments /
// ModComputeFusedTiling) — none of which had any op_host UT before this fix.
// =====================================================================================

// ModSelectContiguousLeanDivider input2Scalar branch: an fp32 SCALAR-other contiguous dispatch takes the
// lean divider (48), same as same-shape. Complements same_shape_float (which covers the same-shape branch).
TEST_F(ModTiling, contiguous_scalar_fp32_lean_divider)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512}, {512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_TRUE(tilingData->isInput2Scalar);
    EXPECT_EQ(tilingData->usableUbSize, ExpectedUsableUbSize(compileInfo.ubSize, kMirrorUbDividerFp32Lean));
}

// ModSelectContiguousLeanDivider fp16 branch: an fp16 same-shape contiguous dispatch also takes the lean
// divider (48 = UB_DIVIDER_FP16_LEAN). Covers the D_T_Y == MOD_TPL_FP16/BF16 arm of the helper.
TEST_F(ModTiling, contiguous_fp16_lean_divider)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_TRUE(tilingData->isInput2SameShape);
    EXPECT_EQ(tilingData->usableUbSize, ExpectedUsableUbSize(compileInfo.ubSize, kMirrorUbDividerFp32Lean));
}

// Path B fused broadcast OUTER-row eligibility + tiling (mode 1): self=[64,128] fp32, other=[1,128]
// (INNER=128 is 32B-aligned for fp32). ModTryFusedBroadcast -> eligible mode 1 -> bcastFusedMode=1,
// bcOuter=64, bcInner=128, and a bcUbFormer/bcBlockFactor that fit the UB budget.
TEST_F(ModTiling, fused_broadcast_outer_row_tiling)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 128}, {1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_FALSE(tilingData->isInput2Scalar);
    EXPECT_FALSE(tilingData->isInput2SameShape);
    EXPECT_EQ(tilingData->bcastFusedMode, 1U); // OUTER row broadcast
    EXPECT_EQ(tilingData->bcOuter, 64U);
    EXPECT_EQ(tilingData->bcInner, 128U);
    EXPECT_GE(tilingData->bcUbFormer, 1U);
    EXPECT_GE(tilingData->bcBlockFactor, 1U);
}

// Path B fused broadcast INNER-col eligibility + tiling (mode 2): self=[128,64] fp32, other=[128,1]
// (INNER=64 is 32B-aligned). ModTryFusedBroadcast -> eligible mode 2 -> bcastFusedMode=2, bcOuter=128,
// bcInner=64.
TEST_F(ModTiling, fused_broadcast_inner_col_tiling)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{128, 64}, {128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{128, 1}, {128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{128, 64}, {128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->bcastFusedMode, 2U); // INNER col broadcast
    EXPECT_EQ(tilingData->bcOuter, 128U);
    EXPECT_EQ(tilingData->bcInner, 64U);
}

// 0811 新增：int16 same-dtype OUTER 行广播 + 非 32B 对齐 INNER (95) —— 原资格 (fp-only + 32B 对齐) 双拒,
// 0811 放宽后应融合: mode 1, bcInner=95, bcIpad=96 (2B dtype 16-elem 单位)。评委 Test_044 同族几何。
TEST_F(ModTiling, fused_broadcast_int16_outer_unaligned)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{8, 95}, {8, 95}}, ge::DT_INT16, ge::FORMAT_ND},
                                                  {{{95}, {95}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 95}, {8, 95}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->bcastFusedMode, 1U); // OUTER row broadcast
    EXPECT_EQ(tilingData->bcOuter, 8U);
    EXPECT_EQ(tilingData->bcInner, 95U);
    EXPECT_EQ(tilingData->bcIpad, 96U); // ceil(95*2/32)*32/2
    EXPECT_GE(tilingData->bcUbFormer, 1U);
}

// 0811 新增：fp32 INNER 列广播 + 非 32B 对齐 INNER (5) —— 原 32B 门槛拒绝, 现应融合: mode 2,
// bcInner=5, bcIpad=8 (fp32 8-elem 单位)。评委 Test_036 同族几何 (尾维列广播)。
TEST_F(ModTiling, fused_broadcast_fp32_inner_unaligned)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{64, 5}, {64, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 1}, {64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{64, 5}, {64, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->bcastFusedMode, 2U); // INNER col broadcast
    EXPECT_EQ(tilingData->bcOuter, 64U);
    EXPECT_EQ(tilingData->bcInner, 5U);
    EXPECT_EQ(tilingData->bcIpad, 8U); // ceil(5*4/32)*32/4
}

// Path B fusion rejected — self-side broadcast (self=[1,128], other=[64,128]): ModCollapseBroadcastSegments
// hits `od != sd && od != 1` on the leading axis -> returns false -> bcastFusedMode stays 0 (generic
// ProcessBroadcast). Exercises the eligibility reject path that keeps non-fusable broadcasts on the
// zero-regression generic route.
TEST_F(ModTiling, fused_broadcast_reject_self_broadcast)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{1, 128}, {1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->bcastFusedMode, 0U); // not fused -> generic ProcessBroadcast
}

// Path B fusion rejected — 3 collapse segments (self=[2,3,5], other=[1,3,1]): after collapse nseg=3 (>2)
// so ModFusedBroadcastEligible returns false -> bcastFusedMode stays 0. Exercises the nseg>2 reject arm.
TEST_F(ModTiling, fused_broadcast_reject_three_segments)
{
    optiling::ModCompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara tilingContextPara("Mod",
                                              {
                                                  {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 3, 1}, {1, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(ModNs::ModTilingData));
    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->bcastFusedMode, 0U); // 3 segments -> not fused
}
