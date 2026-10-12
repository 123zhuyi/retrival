/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_stateless_exponential.cpp
 * \brief StatelessExponential kernel UT (CPU simulation via ICPU_RUN_KF).
 */

#include <cstdint>
#include <cmath>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"

extern "C" __global__ __aicore__ void stateless_exponential(GM_ADDR self, GM_ADDR seed, GM_ADDR offset, GM_ADDR selfOut,
                                                            GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr uint64_t kTilingKeyFp32 = 3;
constexpr uint64_t kTilingKeyFp16 = 1;
constexpr uint64_t kTilingKeyBf16 = 2;
constexpr int64_t kElementCount = 256;
constexpr int64_t kSeed = 42;
constexpr int64_t kOffset = 0;
// SIMT_THREAD_GROUP_SIZE used by host-side execution policy; keep consistent so the
// CPU-simulated kernel maps all elements onto the launched threads.
constexpr uint32_t kSimtThreadGroupSize = 256;

inline size_t Align32(size_t size) { return (size + 31U) / 32U * 32U; }

// The kernel writes via ProcessWithSplitBlocks, which iterates splitBlocks[0..splitBlockCount).
// splitBlockCount must be >= 1, otherwise no element is written.
void FillTiling(RandomUnifiedSimtTilingDataStruct* tilingData, int64_t seed, int64_t offset)
{
    std::memset(tilingData, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = seed;
    tilingData->offset = offset;
    tilingData->prob = 1.0f; // lambd = 1.0
    tilingData->splitBlockCount = 1;
    tilingData->splitBlocks[0].numel = kElementCount;
    tilingData->splitBlocks[0].gmOffset = 0;
    tilingData->splitBlocks[0].grid = 1;
    tilingData->splitBlocks[0].totalThreads = kSimtThreadGroupSize;
    tilingData->splitBlocks[0].kernelOffset = offset;
}
} // namespace

class StatelessExponentialKernelTest : public testing::Test {};

TEST_F(StatelessExponentialKernelTest, smoke_float32)
{
    auto* self = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(self, 0, kElementCount * sizeof(float));
    *reinterpret_cast<int64_t*>(seed) = kSeed;
    *reinterpret_cast<int64_t*>(offset) = kOffset;
    FillTiling(reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling), kSeed, kOffset);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKeyFp32);
    // In-place: self is both the input and output tensor.
    ICPU_RUN_KF(stateless_exponential, kNumBlocks, self, seed, offset, self, workspace, tiling);

    auto* out = reinterpret_cast<float*>(self);
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_TRUE(std::isfinite(out[i])) << "Element " << i << " not finite: " << out[i];
        EXPECT_GT(out[i], 0.0f) << "Element " << i << " <= 0: " << out[i];
    }

    AscendC::GmFree(self);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(StatelessExponentialKernelTest, smoke_float16)
{
    auto* self = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(half))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(self, 0, kElementCount * sizeof(half));
    *reinterpret_cast<int64_t*>(seed) = kSeed;
    *reinterpret_cast<int64_t*>(offset) = kOffset;
    FillTiling(reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling), kSeed, kOffset);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKeyFp16);
    ICPU_RUN_KF(stateless_exponential, kNumBlocks, self, seed, offset, self, workspace, tiling);

    auto* out = reinterpret_cast<half*>(self);
    for (int64_t i = 0; i < kElementCount; ++i) {
        float v = static_cast<float>(out[i]);
        EXPECT_TRUE(std::isfinite(v)) << "Element " << i << " not finite";
        EXPECT_GT(v, 0.0f) << "Element " << i << " <= 0";
    }

    AscendC::GmFree(self);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(StatelessExponentialKernelTest, smoke_bfloat16)
{
    auto* self = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(bfloat16_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(self, 0, kElementCount * sizeof(bfloat16_t));
    *reinterpret_cast<int64_t*>(seed) = kSeed;
    *reinterpret_cast<int64_t*>(offset) = kOffset;
    FillTiling(reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling), kSeed, kOffset);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKeyBf16);
    ICPU_RUN_KF(stateless_exponential, kNumBlocks, self, seed, offset, self, workspace, tiling);

    auto* out = reinterpret_cast<bfloat16_t*>(self);
    for (int64_t i = 0; i < kElementCount; ++i) {
        float v = static_cast<float>(out[i]);
        EXPECT_TRUE(std::isfinite(v)) << "Element " << i << " not finite";
        EXPECT_GT(v, 0.0f) << "Element " << i << " <= 0";
    }

    AscendC::GmFree(self);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(StatelessExponentialKernelTest, determinism)
{
    auto* self1 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* self2 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    *reinterpret_cast<int64_t*>(seed) = 12345;
    *reinterpret_cast<int64_t*>(offset) = 4;
    FillTiling(reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling), 12345, 4);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKeyFp32);
    ICPU_RUN_KF(stateless_exponential, kNumBlocks, self1, seed, offset, self1, workspace, tiling);
    ICPU_RUN_KF(stateless_exponential, kNumBlocks, self2, seed, offset, self2, workspace, tiling);

    EXPECT_EQ(std::memcmp(self1, self2, kElementCount * sizeof(float)), 0);

    AscendC::GmFree(self1);
    AscendC::GmFree(self2);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
