/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <cstdint>
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "../../../op_kernel/mod_tiling_data.h"

using TestUtDefaultTilingStruct = ModNs::ModTilingData;

#include "../../../op_kernel/mod.cpp"

namespace {
void* GmAllocAlign(size_t size) { return GmAlloc((size + 31) >> 5 << 5); }

void FillFloat(void* gm, const std::vector<float>& data)
{
    auto* ptr = reinterpret_cast<float*>(gm);
    for (size_t i = 0; i < data.size(); ++i) {
        ptr[i] = data[i];
    }
}

// Generic fill/read helpers for the active integer, fp16, bf16, scalar and broadcast cases.
template <typename T>
void FillBuf(void* gm, const std::vector<T>& data)
{
    auto* ptr = reinterpret_cast<T*>(gm);
    for (size_t i = 0; i < data.size(); ++i) {
        ptr[i] = data[i];
    }
}

template <typename T>
std::vector<T> ReadBuf(void* gm, size_t count)
{
    auto* ptr = reinterpret_cast<T*>(gm);
    return std::vector<T>(ptr, ptr + count);
}

// Fills the [0..7] shape/stride padding tail the same way every test below needs it (dims >= dimNum
// are always {1, 1, 0} per SetInput2ShapeInfo in op_host/mod_tiling.cpp).
void PadShapeStride(ModNs::ModTilingData* t, uint32_t dimNum)
{
    for (uint32_t i = dimNum; i < 8; ++i) {
        t->input1Shape[i] = 1;
        t->input2Shape[i] = 1;
        t->input2Stride[i] = 0;
    }
}
} // namespace

class ModKernelTest : public testing::Test {};

TEST_F(ModKernelTest, float32_same_shape_one_core)
{
    constexpr size_t elemCount = 4096;
    constexpr size_t dataSize = elemCount * sizeof(float);
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    std::vector<float> self(elemCount);
    std::vector<float> other(elemCount);
    for (size_t i = 0; i < elemCount; ++i) {
        self[i] = static_cast<float>(i + 3);
        other[i] = static_cast<float>((i % 7) + 2);
    }
    FillFloat(x1, self);
    FillFloat(x2, other);

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = true;
    tilingData->dimNum = 1;
    tilingData->input1Shape[0] = elemCount;
    tilingData->input2Shape[0] = elemCount;
    tilingData->input2Stride[0] = 1;
    tilingData->naiveThresh = 256.0f; // 显式置默认阈值 (此前未初始化)
    PadShapeStride(tilingData, 1);

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<30, 30, 30>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);
    SUCCEED();

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

TEST_F(ModKernelTest, float32_same_shape_small_tail)
{
    constexpr size_t elemCount = 16;
    constexpr size_t dataSize = elemCount * sizeof(float);
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    std::vector<float> self = {5.5f,  -11.51f, 36.23f, 7.0f, -10.0f, -8.0f, -15.0f, -7.0f,
                               10.0f, 8.0f,    15.0f,  7.0f, -10.0f, -8.0f, -15.0f, -7.0f};
    std::vector<float> other = {2.0f,  3.0f,  -24.1f, 2.0f,  3.0f,  5.0f,  4.0f,  2.0f,
                                -3.0f, -5.0f, -4.0f,  -2.0f, -3.0f, -5.0f, -4.0f, -2.0f};
    FillFloat(x1, self);
    FillFloat(x2, other);

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = 0;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = true;
    tilingData->dimNum = 2;
    tilingData->input1Shape[0] = 4;
    tilingData->input1Shape[1] = 4;
    tilingData->input2Shape[0] = 4;
    tilingData->input2Shape[1] = 4;
    tilingData->input2Stride[0] = 4;
    tilingData->input2Stride[1] = 1;
    tilingData->naiveThresh = 256.0f; // 显式置默认阈值 (此前未初始化)
    PadShapeStride(tilingData, 2);

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<30, 30, 30>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);
    SUCCEED();

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// The following tests cover the active same-dtype lanes, AlgoA/K2, flat scalar dispatch and broadcast paths.

// int16 same-dtype -> USE_ALGO_A=false -> 恒走 naive 4-op 整数路 (永不 AlgoA，与 naiveThresh 无关)。
// self=[10,-10,7,-7], other=3 -> trunc(10/3)=3 r=1；trunc(-10/3)=-3 r=-1；trunc(7/3)=2 r=1；trunc(-7/3)=-2 r=-1
// (整数域精确)。
TEST_F(ModKernelTest, int16_same_dtype_naive)
{
    constexpr size_t elemCount = 4;
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(int16_t)));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(int16_t)));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(int16_t)));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillBuf<int16_t>(x1, {10, -10, 7, -7});
    FillBuf<int16_t>(x2, {3, 3, 3, 3});

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = true;
    tilingData->dimNum = 1;
    tilingData->input1Shape[0] = elemCount;
    tilingData->input2Shape[0] = elemCount;
    tilingData->input2Stride[0] = 1;
    tilingData->naiveThresh = 256.0f;
    PadShapeStride(tilingData, 1);

    ICPU_SET_TILING_KEY(1973850); // dtype-value only; kernel dispatch is via template args below
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_INT16, MOD_TPL_INT16, MOD_TPL_INT16>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<int16_t>(y, elemCount);
    const std::vector<int16_t> expect = {1, -1, 1, -1};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_EQ(out[i], expect[i]) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// fp32 same-dtype ComputeFPCore 自适应路由——per-tile max|a| >=
// naiveThresh(256) routes the WHOLE tile to RemainderAlgoA (32-op large-quotient compensation)
// instead of the naive 4-op path. self has |a| in the thousands (>> 256) so this specifically
// exercises the AlgoA branch (previously entirely untested at op_kernel UT level — the two
// pre-existing tests only ever used |a| <= 36.23, always naive). trunc(1000.5/7)=142 r=6.5;
// trunc(-1000.5/7)=-142 r=-6.5; trunc(500.25/3)=166 r=2.25; trunc(-500.25/3)=-166 r=-2.25.
TEST_F(ModKernelTest, float32_algoa_large_quotient)
{
    constexpr size_t elemCount = 4;
    constexpr size_t dataSize = elemCount * sizeof(float);
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillFloat(x1, {1000.5f, -1000.5f, 500.25f, -500.25f});
    FillFloat(x2, {7.0f, 7.0f, 3.0f, 3.0f});

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = true;
    tilingData->dimNum = 1;
    tilingData->input1Shape[0] = elemCount;
    tilingData->input2Shape[0] = elemCount;
    tilingData->input2Stride[0] = 1;
    tilingData->naiveThresh = 256.0f; // |a|=500/1000 >= 256 -> AlgoA route
    PadShapeStride(tilingData, 1);

    ICPU_SET_TILING_KEY(1973790);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<30, 30, 30>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<float>(y, elemCount);
    const std::vector<float> expect = {6.5f, -6.5f, 2.25f, -2.25f};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_NEAR(out[i], expect[i], 1e-2f) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// INT16 same-dtype general broadcast keeps the K2 dispatch while covering the legacy
// ProcessBroadcast path with a non-float storage type. x1=[2,4], x2=[1,4].
TEST_F(ModKernelTest, int16_general_broadcast)
{
    constexpr size_t rows = 2;
    constexpr size_t cols = 4;
    constexpr size_t elemCount = rows * cols;
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(int16_t)));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(cols * sizeof(int16_t)));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(int16_t)));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillBuf<int16_t>(x1, {10, -10, 7, -7, 20, -20, 14, -14});
    FillBuf<int16_t>(x2, {3, 3, 3, 3});

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = false;
    tilingData->dimNum = 2;
    tilingData->input1Shape[0] = rows;
    tilingData->input1Shape[1] = cols;
    tilingData->input2Shape[0] = 1;
    tilingData->input2Shape[1] = cols;
    tilingData->input2Stride[0] = 0;
    tilingData->input2Stride[1] = 1;
    tilingData->naiveThresh = 256.0f;
    PadShapeStride(tilingData, 2);

    ICPU_SET_TILING_KEY(1973850);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_INT16, MOD_TPL_INT16, MOD_TPL_INT16>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<int16_t>(y, elemCount);
    const std::vector<int16_t> expect = {1, -1, 1, -1, 2, -2, 2, -2};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_EQ(out[i], expect[i]) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// INT16 scalar other remains on the same-dtype K2 lane and exercises the flat scalar path.
TEST_F(ModKernelTest, int16_scalar_same_dtype)
{
    constexpr size_t elemCount = 4;
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(int16_t)));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(sizeof(int16_t)));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(int16_t)));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillBuf<int16_t>(x1, {10, -10, 7, -7});
    FillBuf<int16_t>(x2, {3});

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = true;
    tilingData->isInput2SameShape = false;
    tilingData->dimNum = 1;
    tilingData->input1Shape[0] = elemCount;
    tilingData->input2Shape[0] = 1;
    tilingData->input2Stride[0] = 0;
    tilingData->naiveThresh = 256.0f;
    PadShapeStride(tilingData, 1);

    ICPU_SET_TILING_KEY(1973850);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_INT16, MOD_TPL_INT16, MOD_TPL_INT16>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<int16_t>(y, elemCount);
    const std::vector<int16_t> expect = {1, -1, 1, -1};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_EQ(out[i], expect[i]) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// flat-buffer arch22 scalar dispatch（isInput2Scalar=true -> LoadScalarOtherFlat +
// CopyInFlat's isInput2Scalar-skip branch in mod_flat_impl.h), previously entirely uncovered (both
// pre-existing kernel tests use isInput2SameShape tensor-other, never a true scalar other).
// self=[5.5,-11.51,36.23,7.0], other scalar=2.0 -> trunc(5.5/2)=2 r=1.5; trunc(-11.51/2)=-5 r=-1.51;
// trunc(36.23/2)=18 r=0.23; trunc(7.0/2)=3 r=1.0.
TEST_F(ModKernelTest, flat_buffer_scalar_dispatch)
{
    constexpr size_t elemCount = 4;
    constexpr size_t dataSize = elemCount * sizeof(float);
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(sizeof(float))); // true scalar: 1 element
    auto y = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillFloat(x1, {5.5f, -11.51f, 36.23f, 7.0f});
    FillFloat(x2, {2.0f});

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = true;
    tilingData->isInput2SameShape = false;
    tilingData->dimNum = 1;
    tilingData->input1Shape[0] = elemCount;
    tilingData->input2Shape[0] = 1;
    tilingData->input2Stride[0] = 0;
    tilingData->naiveThresh = 256.0f;
    PadShapeStride(tilingData, 1);

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<30, 30, 30>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<float>(y, elemCount);
    const std::vector<float> expect = {1.5f, -1.51f, 0.23f, 1.0f};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_NEAR(out[i], expect[i], 1e-2f) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// Legacy TQue ProcessBroadcast (general broadcast, !isInput2Scalar && !isInput2SameShape),
// previously entirely uncovered at op_kernel UT level (both flat-buffer tests above take the
// isInput2SameShape/isInput2Scalar -> ProcessContiguousFlat branch instead). x1 shape [2,4], x2
// shape [1,4] broadcasts over dim0 (input2Stride=[0,1] per SetInput2ShapeInfo). Row0 self=
// [10,-10,7,-7], row1 self=[20,-20,14,-14], other (single row, reused for both)=[3,3,3,3].
// Row0 expect (same calc as int16_same_dtype_naive, fp32 here): [1,-1,1,-1].
// Row1: trunc(20/3)=6 r=2; trunc(-20/3)=-6 r=-2; trunc(14/3)=4 r=2; trunc(-14/3)=-4 r=-2 -> [2,-2,2,-2].
TEST_F(ModKernelTest, legacy_broadcast_dispatch)
{
    constexpr size_t rows = 2;
    constexpr size_t cols = 4;
    constexpr size_t elemCount = rows * cols;
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(float)));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(cols * sizeof(float))); // x2 shape [1,4]: only 4 elems
    auto y = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(float)));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillFloat(x1, {10.0f, -10.0f, 7.0f, -7.0f, 20.0f, -20.0f, 14.0f, -14.0f});
    FillFloat(x2, {3.0f, 3.0f, 3.0f, 3.0f});

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = false; // dimNum equal but dim0 differs (2 vs 1) -> general broadcast
    tilingData->dimNum = 2;
    tilingData->input1Shape[0] = rows;
    tilingData->input1Shape[1] = cols;
    tilingData->input2Shape[0] = 1;
    tilingData->input2Shape[1] = cols;
    tilingData->input2Stride[0] = 0; // broadcast dim (size 1)
    tilingData->input2Stride[1] = 1; // contiguous dim
    tilingData->naiveThresh = 256.0f;
    PadShapeStride(tilingData, 2);

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<30, 30, 30>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<float>(y, elemCount);
    const std::vector<float> expect = {1.0f, -1.0f, 1.0f, -1.0f, 2.0f, -2.0f, 2.0f, -2.0f};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_NEAR(out[i], expect[i], 1e-3f) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// =====================================================================================
// 精简连续 half/bf16 计算 (ComputeContigLean 的 NEED_FP32_IO_BUF cast 分支 + InitLeanWorkBuffers
// A5/x1F32/x2F32) 与整个融合广播 kernel (mod_bcast_impl.h: ProcessFusedBcast / BuildOuterOtherFused /
// CopyInFusedBcast / ComputeFusedBcast / CopyOutFusedBcast) 此前无 op_kernel UT 覆盖。下方四个用例
// 手算期望 trunc-mod 结果 (符号随 self)，是真实正确性检查而非 SUCCEED() 冒烟。
// =====================================================================================
namespace {
// Convert a float vector to a half/bf16 storage vector so the lean fp16/bf16 kernel path can be
// exercised. The CPU sim's half/bfloat16_t support static_cast to/from float (same pattern as arange's
// kernel UT test_arange_kernel.cpp) — exact widening on read-back for comparison.
template <typename T>
std::vector<T> AsDtypeVec(const std::vector<float>& f)
{
    std::vector<T> v;
    v.reserve(f.size());
    for (float x : f) {
        v.push_back(static_cast<T>(x));
    }
    return v;
}
} // namespace

// 精简 fp16 same-dtype 连续 -> ComputeContigLean NEED_FP32_IO_BUF 分支 (Cast fp16->fp32 ->
// RemainderAdaptive -> Cast(CAST_RINT) fp32->fp16) + InitLeanWorkBuffers cast buffer。
// self=[10,-10,7,-7,20,-20,14,-14] (fp16), other=[3,3,3,3,4,4,4,4] -> trunc-mod = [1,-1,1,-1,0,0,2,-2]
// (均 fp16 精确可表整数)。
TEST_F(ModKernelTest, fp16_same_shape_lean)
{
    constexpr size_t elemCount = 8;
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(half)));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(half)));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(half)));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillBuf<half>(x1, AsDtypeVec<half>({10, -10, 7, -7, 20, -20, 14, -14}));
    FillBuf<half>(x2, AsDtypeVec<half>({3, 3, 3, 3, 4, 4, 4, 4}));

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = true;
    tilingData->dimNum = 1;
    tilingData->input1Shape[0] = elemCount;
    tilingData->input2Shape[0] = elemCount;
    tilingData->input2Stride[0] = 1;
    tilingData->naiveThresh = 256.0f;
    PadShapeStride(tilingData, 1);

    ICPU_SET_TILING_KEY(1315860);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_FP16, MOD_TPL_FP16, MOD_TPL_FP16>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<half>(y, elemCount);
    const std::vector<float> expect = {1, -1, 1, -1, 0, 0, 2, -2};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_NEAR(static_cast<float>(out[i]), expect[i], 1e-2f) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// lean bf16 same-dtype contiguous -> same ComputeContigLean cast branch as fp16, bf16 dtype
// (mod<10,10,10> = Mod<bfloat16_t>). Integer results are exact in bf16; looser tol for bf16's 8-bit mantissa.
TEST_F(ModKernelTest, bf16_same_shape_lean)
{
    constexpr size_t elemCount = 8;
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(bfloat16_t)));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(bfloat16_t)));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(bfloat16_t)));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillBuf<bfloat16_t>(x1, AsDtypeVec<bfloat16_t>({10, -10, 7, -7, 20, -20, 14, -14}));
    FillBuf<bfloat16_t>(x2, AsDtypeVec<bfloat16_t>({3, 3, 3, 3, 4, 4, 4, 4}));

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = true;
    tilingData->dimNum = 1;
    tilingData->input1Shape[0] = elemCount;
    tilingData->input2Shape[0] = elemCount;
    tilingData->input2Stride[0] = 1;
    tilingData->naiveThresh = 256.0f;
    PadShapeStride(tilingData, 1);

    ICPU_SET_TILING_KEY(657930);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_BF16, MOD_TPL_BF16, MOD_TPL_BF16>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<bfloat16_t>(y, elemCount);
    const std::vector<float> expect = {1, -1, 1, -1, 0, 0, 2, -2};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_NEAR(static_cast<float>(out[i]), expect[i], 1e-1f) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// Path B fused broadcast mode 1 (OUTER row, other=[1,INNER]): the single INNER row is broadcast across all
// OUTER rows. Exercises BuildOuterOtherFused (build-once-per-core resident divisor) + ComputeFusedBcast
// fp32-native branch. self=[2,8] fp32 rows, other=[1,8] (8 = 32B-aligned INNER for fp32). bcastFusedMode=1.
// row0 self=[10,-10,7,-7,20,-20,14,-14] % other=[2,3,4,2,5,4,3,2] -> [0,-1,3,-1,0,0,2,0];
// row1 self=[100,-100,70,-70,200,-200,140,-140] % (same other) -> [0,-1,2,0,0,0,2,0].
TEST_F(ModKernelTest, fused_bcast_outer_row_fp32)
{
    constexpr size_t rows = 2;
    constexpr size_t inner = 8;
    constexpr size_t elemCount = rows * inner;
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(float)));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(inner * sizeof(float))); // one INNER row
    auto y = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(float)));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillFloat(x1, {10, -10, 7, -7, 20, -20, 14, -14, 100, -100, 70, -70, 200, -200, 140, -140});
    FillFloat(x2, {2, 3, 4, 2, 5, 4, 3, 2});

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = false; // general broadcast -> fused path
    tilingData->dimNum = 2;
    tilingData->input1Shape[0] = rows;
    tilingData->input1Shape[1] = inner;
    tilingData->input2Shape[0] = 1;
    tilingData->input2Shape[1] = inner;
    tilingData->input2Stride[0] = 0;
    tilingData->input2Stride[1] = 1;
    tilingData->naiveThresh = 256.0f;
    tilingData->bcastFusedMode = 1; // OUTER row broadcast
    tilingData->bcOuter = rows;
    tilingData->bcInner = inner;
    tilingData->bcUbFormer = rows; // one tile covers all rows
    tilingData->bcBlockFactor = rows;
    tilingData->bcIpad = inner; // 8 fp32 = 32B 对齐 -> padding 退化 1D
    PadShapeStride(tilingData, 2);

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_FP32, MOD_TPL_FP32, MOD_TPL_FP32>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<float>(y, elemCount);
    const std::vector<float> expect = {0, -1, 3, -1, 0, 0, 2, 0, 0, -1, 2, 0, 0, 0, 2, 0};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_NEAR(out[i], expect[i], 1e-3f) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// Path B fused broadcast mode 2 (INNER col, other=[OUTER,1]): each OUTER row is mod'd by its own per-row
// scalar broadcast across the INNER columns. Exercises CopyInFusedBcast's mode-2 per-row-scalar read +
// ComputeFusedBcast's per-row Duplicate. self=[2,8] fp32, other=[2,1] fp32 (per-row divisors [3,5]).
// row0 self=[10,-10,7,-7,9,-9,4,-4] % 3 -> [1,-1,1,-1,0,0,1,-1];
// row1 self=[20,-20,14,-14,11,-11,6,-6] % 5 -> [0,0,4,-4,1,-1,1,-1].
TEST_F(ModKernelTest, fused_bcast_inner_col_fp32)
{
    constexpr size_t rows = 2;
    constexpr size_t inner = 8;
    constexpr size_t elemCount = rows * inner;
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(float)));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(rows * sizeof(float))); // one scalar per OUTER row
    auto y = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(float)));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillFloat(x1, {10, -10, 7, -7, 9, -9, 4, -4, 20, -20, 14, -14, 11, -11, 6, -6});
    FillFloat(x2, {3, 5});

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = false; // general broadcast -> fused path
    tilingData->dimNum = 2;
    tilingData->input1Shape[0] = rows;
    tilingData->input1Shape[1] = inner;
    tilingData->input2Shape[0] = rows;
    tilingData->input2Shape[1] = 1;
    tilingData->input2Stride[0] = 1;
    tilingData->input2Stride[1] = 0;
    tilingData->naiveThresh = 256.0f;
    tilingData->bcastFusedMode = 2; // INNER col broadcast
    tilingData->bcOuter = rows;
    tilingData->bcInner = inner;
    tilingData->bcUbFormer = rows;
    tilingData->bcBlockFactor = rows;
    tilingData->bcIpad = inner; // 8 fp32 = 32B 对齐 -> padding 退化 1D
    PadShapeStride(tilingData, 2);

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_FP32, MOD_TPL_FP32, MOD_TPL_FP32>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<float>(y, elemCount);
    const std::vector<float> expect = {1, -1, 1, -1, 0, 0, 1, -1, 0, 0, 4, -4, 1, -1, 1, -1};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_NEAR(out[i], expect[i], 1e-3f) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// 0811 新增：int16 same-dtype OUTER 行广播 + 非 32B 对齐 INNER (5 -> bcIpad=16, 2B 16-elem 单位)。
// 走 0811 新代码：2D 自动 padding CopyIn + priming + Muls 行复制常驻 + int16 cast 路 (CAST_NONE widen /
// CAST_RINT 下行)。self=[2,5] int16, other=[5] int16=[2,3,4,2,5]。
// row0 [10,-10,7,-7,9] % [2,3,4,2,5] -> [0,-1,3,-1,4]; row1 [20,-20,14,-14,11] -> [0,-2,2,0,1]。
TEST_F(ModKernelTest, fused_bcast_outer_row_int16_unaligned)
{
    constexpr size_t rows = 2;
    constexpr size_t inner = 5;
    constexpr size_t elemCount = rows * inner;
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(int16_t)));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(inner * sizeof(int16_t))); // one INNER row
    auto y = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(int16_t)));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillBuf<int16_t>(x1, {10, -10, 7, -7, 9, 20, -20, 14, -14, 11});
    FillBuf<int16_t>(x2, {2, 3, 4, 2, 5});

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = false; // general broadcast -> fused path
    tilingData->dimNum = 2;
    tilingData->input1Shape[0] = rows;
    tilingData->input1Shape[1] = inner;
    tilingData->input2Shape[0] = 1;
    tilingData->input2Shape[1] = inner;
    tilingData->input2Stride[0] = 0;
    tilingData->input2Stride[1] = 1;
    tilingData->naiveThresh = 256.0f;
    tilingData->bcastFusedMode = 1; // OUTER row broadcast
    tilingData->bcOuter = rows;
    tilingData->bcInner = inner;
    tilingData->bcUbFormer = rows; // one tile covers all rows
    tilingData->bcBlockFactor = rows;
    tilingData->bcIpad = 16; // ceil(5*2/32)*32/2 — padding 行步长
    PadShapeStride(tilingData, 2);

    ICPU_SET_TILING_KEY(1973850); // dtype-value only; kernel dispatch is via template args below
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_INT16, MOD_TPL_INT16, MOD_TPL_INT16>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<int16_t>(y, elemCount);
    const std::vector<int16_t> expect = {0, -1, 3, -1, 4, 0, -2, 2, 0, 1};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_EQ(out[i], expect[i]) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// 0811 新增：fp32 INNER 列广播 + 非 32B 对齐 INNER (5 -> bcIpad=8, fp32 8-elem 单位)。走 0811 新代码：
// 2D 自动 padding CopyIn + 每 tile 预填 1.0 + 逐行 padding Duplicate。self=[2,5] fp32, other=[2,1]=[3,5]。
// row0 [10,-10,7,-7,9] % 3 -> [1,-1,1,-1,0]; row1 [20,-20,14,-14,11] % 5 -> [0,0,4,-4,1]。
TEST_F(ModKernelTest, fused_bcast_inner_col_fp32_unaligned)
{
    constexpr size_t rows = 2;
    constexpr size_t inner = 5;
    constexpr size_t elemCount = rows * inner;
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(float)));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(rows * sizeof(float))); // one scalar per OUTER row
    auto y = static_cast<GM_ADDR>(GmAllocAlign(elemCount * sizeof(float)));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    FillFloat(x1, {10, -10, 7, -7, 9, 20, -20, 14, -14, 11});
    FillFloat(x2, {3, 5});

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = false; // general broadcast -> fused path
    tilingData->dimNum = 2;
    tilingData->input1Shape[0] = rows;
    tilingData->input1Shape[1] = inner;
    tilingData->input2Shape[0] = rows;
    tilingData->input2Shape[1] = 1;
    tilingData->input2Stride[0] = 1;
    tilingData->input2Stride[1] = 0;
    tilingData->naiveThresh = 256.0f;
    tilingData->bcastFusedMode = 2; // INNER col broadcast
    tilingData->bcOuter = rows;
    tilingData->bcInner = inner;
    tilingData->bcUbFormer = rows;
    tilingData->bcBlockFactor = rows;
    tilingData->bcIpad = 8; // ceil(5*4/32)*32/4 — padding 行步长
    PadShapeStride(tilingData, 2);

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_FP32, MOD_TPL_FP32, MOD_TPL_FP32>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<float>(y, elemCount);
    const std::vector<float> expect = {1, -1, 1, -1, 0, 0, 0, 4, -4, 1};
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_NEAR(out[i], expect[i], 1e-3f) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// NOTE: no int32-same-dtype kernel UT here — ComputeInt32 (mod_int32_impl.h, inherited upstream
// code outside this change's scope) branches at COMPILE time on two different macros: InitBuffers/
// InitConstants allocate the FP32MaxValid/INT32MaxValid/Epsilon buffers under HIGH_PRECISION, while
// ComputeInt32 itself branches on HIGH_PERFORMANCE, and neither macro is defined in this UT's build
// config. Its #else branch would call FP32MaxValidBuff.Get<float>() on a TBuf that InitBuffers never
// allocated for T=int -> `buffer length is 0` abort. This latent upstream gap (production builds
// inject the macros via CMake/opbuild per SoC/TilingKey; the bare UT harness does not) predates
// this change; coverage can be added separately once the macro plumbing is addressed upstream.

// ---------------------------------------------------------------------------
// 0811 深夜 per-core 路由用例 (V5.1 实证收口后重定位)：usableUbSize=3648 -> kernel Init 后
//   maxDataCount=3648 (3648=57*64 已对齐)。V5.1 起 fp lane 的 per-core 预扫已删除 (真机 A/B 证伪
//   净亏)，fp 保持 per-tile RemainderAdaptive 现状 -> 这三例转而作为大 shape / 多 chunk 几何下
//   per-tile 路由的正确性回归用例 (防"首 tile 决定"类错误实现永远不再有藏身地：per-tile 本来
//   就逐 tile 探针)。
// ---------------------------------------------------------------------------

// 大 shape 连续路 per-tile 混合路由回归：前 2/3 数据全小值 (|a|<=7 < 256 -> naive tile)，最后一个
//   tile 藏大值 30000.5 (>=256 -> AlgoA tile)。V5.1 起 fp 的 per-core 预扫已删除 (真机 A/B 证伪净亏)，
//   fp 保持 per-tile RemainderAdaptive 现状，本例锁该路径在大 shape / 多 chunk 几何下数值不回归。
//   所有取值在两路重叠域结果一致 (naive/AlgoA 均精确) -> 路由无关、mock/真机皆稳健。
//   (V1-V3 曾在 i=0 埋 (12.0,0.4) 判别点锁"首 tile 决定"错误实现——该点在 per-tile 路由下走 naive，
//   fp32 除法舍入跨整数边界得 0.0 (C4-1 病态，mock/真机同)，与整核 AlgoA 期望 0.4 冲突，V5.1 已随
//   fp 预扫一并移除。)
//   期望值手算 trunc-mod：fmod(30000.5,7)=5.5 (trunc(30000.5/7)=4285, 4285*7=29995)；
//   v=(i%7)+1 对 7 取模 -> v==7 ? 0 : v。
TEST_F(ModKernelTest, core_route_prescan_big_value_in_last_tile)
{
    constexpr size_t elemCount = 8192;
    constexpr size_t dataSize = elemCount * sizeof(float);
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    std::vector<float> self(elemCount);
    std::vector<float> other(elemCount, 7.0f);
    std::vector<float> expect(elemCount);
    for (size_t i = 0; i < elemCount; ++i) {
        const float v = static_cast<float>((i % 7) + 1);
        self[i] = v;
        expect[i] = (v == 7.0f) ? 0.0f : v;
    }
    self[8000] = 30000.5f; // 最后一个 tile [7296,8192) 藏大值 -> 该 tile per-tile 路由 AlgoA
    expect[8000] = 5.5f;
    FillFloat(x1, self);
    FillFloat(x2, other);

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = true;
    tilingData->dimNum = 1;
    tilingData->input1Shape[0] = elemCount;
    tilingData->input2Shape[0] = elemCount;
    tilingData->input2Stride[0] = 1;
    tilingData->naiveThresh = 256.0f;
    tilingData->bcastFusedMode = 0;
    tilingData->bcIpad = 0;
    PadShapeStride(tilingData, 1);

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_FP32, MOD_TPL_FP32, MOD_TPL_FP32>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<float>(y, elemCount);
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_NEAR(out[i], expect[i], 1e-3f) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// 全小值 (|a|<=7 << 256) 大 shape 多 tile：per-tile 路由下全 tile 走 naive -> 结果与 naive 手算一致。
//   此例锁大 shape 连续路 naive 数值不回归 (V5.1 前它覆盖预扫 coreRoute_==1 分支，现转为 per-tile
//   大 shape 回归)。
TEST_F(ModKernelTest, core_route_prescan_all_small_stays_naive)
{
    constexpr size_t elemCount = 8192;
    constexpr size_t dataSize = elemCount * sizeof(float);
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    std::vector<float> self(elemCount);
    std::vector<float> other(elemCount, 7.0f);
    std::vector<float> expect(elemCount);
    for (size_t i = 0; i < elemCount; ++i) {
        const float v = static_cast<float>((i % 7) + 1);
        self[i] = v;
        expect[i] = (v == 7.0f) ? 0.0f : v;
    }
    FillFloat(x1, self);
    FillFloat(x2, other);

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = true;
    tilingData->dimNum = 1;
    tilingData->input1Shape[0] = elemCount;
    tilingData->input2Shape[0] = elemCount;
    tilingData->input2Stride[0] = 1;
    tilingData->naiveThresh = 256.0f;
    tilingData->bcastFusedMode = 0;
    tilingData->bcIpad = 0;
    PadShapeStride(tilingData, 1);

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_FP32, MOD_TPL_FP32, MOD_TPL_FP32>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<float>(y, elemCount);
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_NEAR(out[i], expect[i], 1e-3f) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

// 通用广播 (非融合) 正确性回归：nseg=3 几何 (self=[64,3,2], other=[1,3,1]) 融合资格不收 -> 走通用
//   ProcessBroadcast (尾维纯广播折叠 isConstantRun -> tile=2 元素常量段，192 个微 tile，
//   正是 064/079 类形态)。末尾 tile [382,384) 藏大值 self[382]=30000.5 (x2 索引 (382/2)%3=191%3=2 ->
//   除数 7 -> 该 tile per-tile 路由 AlgoA -> fmod=5.5)。
//   x2 索引 = GetInput2Offset(i) = (i/2)%3 (input2Stride={0,1,0})；期望值逐点整数手算
//   fmod(self[i], x2arr[(i/2)%3]) (v<=7、d∈{3,5,7} 正整数 trunc 精确，用整数除法避免浮点手算误差)。
//   规模说明 (cann9 mock 二分实证 2026-08-12)：本例初版用 [8192,3,2] (49152 elems) 在 tikicpulib 上
//   全零失败——usableUbSize=3648 时通用广播路 buffer 账面 ~246KB (queues 2*3648*4*3 + tmp + ResQuot/
//   ResRem + Zero/Inf/Nan/Mask + A1..A5 = 69 B/elem) 超 DAV_2201 UB 192KB，mock 退化空跑；小几何同族
//   折叠 ([8,3,2]x[1,3,1]) / 非折叠 ([8,3,2]x[1,3,2]) / 中规模小 buffer ([1024,3,2]+ub512, 3072 tile
//   全量诚实仿真) 对照全 PASS -> 失败与 buffer 账面超限挂钩、与折叠机制和迭代数无关，kernel 无 bug
//   (真机 200+200 精度含同类几何全过)。缩小到 384 元素保留同族回归价值且 mock 可消化。
TEST_F(ModKernelTest, core_route_prescan_generic_broadcast)
{
    constexpr size_t elemCount = 64 * 3 * 2; // 384
    constexpr size_t dataSize = elemCount * sizeof(float);
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(3 * sizeof(float))); // [1,3,1] 三个 per-mid-segment 标量
    auto y = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    const int x2int[3] = {3, 5, 7};
    std::vector<float> self(elemCount);
    std::vector<float> expect(elemCount);
    for (size_t i = 0; i < elemCount; ++i) {
        const int vi = static_cast<int>(i % 7) + 1;
        const int di = x2int[(i / 2) % 3];
        self[i] = static_cast<float>(vi);
        expect[i] = static_cast<float>(vi - (vi / di) * di); // trunc-mod (正整数域精确)
    }
    self[382] = 30000.5f; // 末尾 tile [382,384) 藏大值 -> 该 tile per-tile 路由 AlgoA
    expect[382] = 5.5f;   // trunc(30000.5/7)=4285, 4285*7=29995 -> r=5.5
    FillFloat(x1, self);
    FillFloat(x2, {3.0f, 5.0f, 7.0f});

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = false; // general broadcast (非融合, bcastFusedMode=0)
    tilingData->dimNum = 3;
    tilingData->input1Shape[0] = 64;
    tilingData->input1Shape[1] = 3;
    tilingData->input1Shape[2] = 2;
    tilingData->input2Shape[0] = 1;
    tilingData->input2Shape[1] = 3;
    tilingData->input2Shape[2] = 1;
    tilingData->input2Stride[0] = 0;
    tilingData->input2Stride[1] = 1;
    tilingData->input2Stride[2] = 0;
    tilingData->naiveThresh = 256.0f;
    tilingData->bcastFusedMode = 0;
    tilingData->bcIpad = 0;
    PadShapeStride(tilingData, 3);

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<MOD_TPL_FP32, MOD_TPL_FP32, MOD_TPL_FP32>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);

    auto out = ReadBuf<float>(y, elemCount);
    for (size_t i = 0; i < elemCount; ++i) {
        EXPECT_NEAR(out[i], expect[i], 1e-3f) << "i=" << i;
    }

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}
