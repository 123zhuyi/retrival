/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/radix_top_k_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class RadixTopKTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RadixTopKTilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "RadixTopKTilingTest TearDown" << std::endl;
    }
};

namespace {

constexpr uint64_t LARGE_SORT_LEN = 1448600000;
constexpr uint64_t CORE_NUM = 48;
constexpr uint64_t UB_SIZE = 196608;

static thread_local int32_t g_kValue = 0;

gert::TilingContextPara MakeRadixTopKTilingContext(
    const gert::StorageShape &xShape,
    ge::DataType xDtype,
    int32_t kValue,
    bool sorted = true,
    int64_t dim = -1,
    bool largest = true,
    int64_t indicesDtype = 3)
{
    g_kValue = kValue;
    gert::StorageShape kShape({1}, {1});

    optiling::RadixTopKCompileInfo compileInfo;
    compileInfo.totalCoreNum = CORE_NUM;
    compileInfo.ubSizePlatForm = UB_SIZE;

    auto* compileInfoPtr = new optiling::RadixTopKCompileInfo(compileInfo);

    return gert::TilingContextPara(
        "RadixTopK",
        {
            {xShape, xDtype, ge::FORMAT_ND},
            {kShape, ge::DT_INT32, ge::FORMAT_ND, true, &g_kValue},
        },
        {
            {xShape, xDtype, ge::FORMAT_ND},
            {xShape, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(sorted)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(dim)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(largest)),
            gert::TilingContextPara::OpAttr("indices_dtype",
                Ops::Math::AnyValue::CreateFrom<int64_t>(indicesDtype)),
        },
        compileInfoPtr);
}

uint64_t ComputeExpectedTilingKey(bool sorted, bool largest, uint64_t sortLen)
{
    uint64_t key = 0;
    key |= (static_cast<uint64_t>(sorted) << 0);
    key |= (static_cast<uint64_t>(largest) << 1);
    bool isLargeShape = (sortLen > LARGE_SORT_LEN);
    key |= (static_cast<uint64_t>(isLargeShape) << 2);
    return key;
}

void CheckTilingKey(const gert::TilingContextPara &tilingCtx, uint64_t expectKey)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectKey);
    EXPECT_EQ(tilingInfo.workspaceSizes.size(), 1);
}

void CheckTilingWithWs(const gert::TilingContextPara &tilingCtx,
                       uint64_t expectKey, bool expectWsNonzero)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectKey);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    if (expectWsNonzero) {
        EXPECT_GT(tilingInfo.workspaceSizes[0], 0u)
            << "k 较小时应触发 workspace 分配 (needWorkspace=true)";
    } else {
        EXPECT_EQ(tilingInfo.workspaceSizes[0], 0u)
            << "k 较大时应复用 indices 内存 (needWorkspace=false)";
    }
}

}  // namespace

// ============================================================
// UB 变体 fp16 — 1D shapes
// ============================================================

TEST_F(RadixTopKTilingTest, ub_fp16_1d_128_k5)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{128}, {128}}, ge::DT_FLOAT16, 5);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 128));
}

TEST_F(RadixTopKTilingTest, ub_fp16_1d_4096_k500)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{4096}, {4096}}, ge::DT_FLOAT16, 500);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 4096));
}

TEST_F(RadixTopKTilingTest, ub_fp16_1d_65536_k100)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{65536}, {65536}}, ge::DT_FLOAT16, 100);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 65536));
}

// ============================================================
// UB 变体 fp16 — 2D/3D shapes
// ============================================================

TEST_F(RadixTopKTilingTest, ub_fp16_2d_32x512_k5)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{32, 512}, {32, 512}}, ge::DT_FLOAT16, 5);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 512));
}

TEST_F(RadixTopKTilingTest, ub_fp16_3d_4x128x256_k10)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{4, 128, 256}, {4, 128, 256}}, ge::DT_FLOAT16, 10);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 256));
}

TEST_F(RadixTopKTilingTest, ub_fp16_bert_512x768_k10)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{512, 768}, {512, 768}}, ge::DT_FLOAT16, 10);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 768));
}

// ============================================================
// UB 变体 bf16
// ============================================================

TEST_F(RadixTopKTilingTest, ub_bf16_1d_1024_k5)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{1024}, {1024}}, ge::DT_BF16, 5);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 1024));
}

TEST_F(RadixTopKTilingTest, ub_bf16_2d_128x1024_k50)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{128, 1024}, {128, 1024}}, ge::DT_BF16, 50);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 1024));
}

// ============================================================
// largest/sorted 参数组合验证 tiling key
// ============================================================

TEST_F(RadixTopKTilingTest, ub_fp16_largest_false_sorted_true)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{512}, {512}}, ge::DT_FLOAT16, 10, true, -1, false);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, false, 512));
}

TEST_F(RadixTopKTilingTest, ub_fp16_largest_true_sorted_false)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{512}, {512}}, ge::DT_FLOAT16, 10, false, -1, true);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(false, true, 512));
}

TEST_F(RadixTopKTilingTest, ub_fp16_largest_false_sorted_false)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{512}, {512}}, ge::DT_FLOAT16, 10, false, -1, false);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(false, false, 512));
}

// ============================================================
// 边界 K 值
// ============================================================

TEST_F(RadixTopKTilingTest, ub_fp16_k_equals_sortlen)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{256}, {256}}, ge::DT_FLOAT16, 256);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 256));
}

TEST_F(RadixTopKTilingTest, ub_fp16_k_exceeds_ws_threshold)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{4096}, {4096}}, ge::DT_FLOAT16, 385);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 4096));
}

// ============================================================
// 极小 shape 边界测试
// ============================================================

TEST_F(RadixTopKTilingTest, ub_fp16_tiny_1_elem)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{1}, {1}}, ge::DT_FLOAT16, 1);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 1));
}

TEST_F(RadixTopKTilingTest, ub_fp16_tiny_3_elems)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{3}, {3}}, ge::DT_FLOAT16, 2);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 3));
}

TEST_F(RadixTopKTilingTest, ub_fp16_tiny_1x1)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{1, 1}, {1, 1}}, ge::DT_FLOAT16, 1);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 1));
}

// ============================================================
// WS 变体 (sortLen > LARGE_SORT_LEN) — 需要 workspace
// ============================================================

TEST_F(RadixTopKTilingTest, ws_fp16_large_sort_len)
{
    uint64_t bigSortLen = LARGE_SORT_LEN + 1;
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{1, bigSortLen}, {1, bigSortLen}}, ge::DT_FLOAT16, 10);
    uint64_t expectKey = ComputeExpectedTilingKey(true, true, bigSortLen);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectKey);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_GT(tilingInfo.workspaceSizes[0], 0);
}

// ============================================================
// Workspace 验证：k≤coreNum×6 时需要 ws；k>coreNum×6 时不需要
// ============================================================

TEST_F(RadixTopKTilingTest, ws_verify_small_k_needs_ws)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{512}, {512}}, ge::DT_FLOAT16, 5);
    CheckTilingWithWs(tilingCtx, ComputeExpectedTilingKey(true, true, 512), true);
}

TEST_F(RadixTopKTilingTest, ws_verify_large_k_reuses_indices)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{4096}, {4096}}, ge::DT_FLOAT16, 500);
    CheckTilingWithWs(tilingCtx, ComputeExpectedTilingKey(true, true, 4096), false);
}

// ============================================================
// k=100, UB 变体 — sortLen 足够大使得 coreNum×6 > 100，触发 workspace
// ============================================================

TEST_F(RadixTopKTilingTest, k100_ub_needs_ws)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{65536}, {65536}}, ge::DT_FLOAT16, 100);
    CheckTilingWithWs(tilingCtx, ComputeExpectedTilingKey(true, true, 65536), true);
}
