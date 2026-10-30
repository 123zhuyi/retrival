/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"

extern "C" __global__ __aicore__ void stateless_uniform(
    GM_ADDR shape, GM_ADDR seed, GM_ADDR offset, GM_ADDR from, GM_ADDR to,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr int64_t kElementCount = 256;

constexpr int64_t GPU_BLOCK_SIZE = 256;
constexpr int64_t SM_COUNT = 78;
constexpr int64_t MAX_THREADS_PER_SM = 2048;
constexpr int64_t BLOCKS_PER_SM = MAX_THREADS_PER_SM / GPU_BLOCK_SIZE;
constexpr int64_t UNROLL_FACTOR = 4;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}

void FillSplitBlock(RandomUnifiedSimtTilingDataStruct* td, int64_t numel)
{
    int64_t grid = (numel + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
    if (grid > SM_COUNT * BLOCKS_PER_SM) {
        grid = SM_COUNT * BLOCKS_PER_SM;
    }
    int64_t totalThreads = grid * GPU_BLOCK_SIZE;
    int64_t denom = GPU_BLOCK_SIZE * grid * UNROLL_FACTOR;
    int64_t counterOffset = ((numel + denom - 1) / denom) * UNROLL_FACTOR;

    td->splitBlockCount = 1;
    td->splitBlocks[0].numel = numel;
    td->splitBlocks[0].gmOffset = 0;
    td->splitBlocks[0].grid = grid;
    td->splitBlocks[0].totalThreads = totalThreads;
    td->splitBlocks[0].kernelOffset = 0;
}
} // namespace

class StatelessUniformKernelTest : public testing::Test {
};

// Test 1: float32, from=0.0, to=1.0, seed=42, shape=[16,16]=256
TEST_F(StatelessUniformKernelTest, smoke_float32_default_range)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int64_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* from = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(double))));
    auto* to = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(double))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * sizeof(float));
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));

    reinterpret_cast<int64_t*>(shape)[0] = 16;
    reinterpret_cast<int64_t*>(shape)[1] = 16;
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(offset) = 0;
    *reinterpret_cast<double*>(from) = 0.0;
    *reinterpret_cast<double*>(to) = 1.0;

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 42;
    tilingData->offset = 0;
    tilingData->ubSize = 229376;
    tilingData->fromFp32 = 0.0f;
    tilingData->toFp32 = 1.0f;
    FillSplitBlock(tilingData, kElementCount);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(stateless_uniform, kNumBlocks, shape, seed, offset, from, to, y, workspace, tiling);

    auto* yData = reinterpret_cast<float*>(y);
    int nonZeroCount = 0;
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_GE(yData[i], 0.0f) << "Element " << i << " is below from (0.0)";
        EXPECT_LT(yData[i], 1.0f) << "Element " << i << " is >= to (1.0)";
        if (yData[i] != 0.0f) {
            nonZeroCount++;
        }
    }
    EXPECT_GT(nonZeroCount, 0) << "All output elements are zero";

    AscendC::GmFree(shape);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(from);
    AscendC::GmFree(to);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test 2: float32, from=-2.0, to=3.0, custom range
TEST_F(StatelessUniformKernelTest, smoke_float32_custom_range)
{
    constexpr int64_t kCount = 512;
    constexpr float kFrom = -2.0f;
    constexpr float kTo = 3.0f;

    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* from = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(double))));
    auto* to = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(double))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kCount * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kCount * sizeof(float));
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));

    reinterpret_cast<int64_t*>(shape)[0] = kCount;
    *reinterpret_cast<int64_t*>(seed) = 12345;
    *reinterpret_cast<int64_t*>(offset) = 0;
    *reinterpret_cast<double*>(from) = static_cast<double>(kFrom);
    *reinterpret_cast<double*>(to) = static_cast<double>(kTo);

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kCount;
    tilingData->seed = 12345;
    tilingData->offset = 0;
    tilingData->ubSize = 229376;
    tilingData->fromFp32 = kFrom;
    tilingData->toFp32 = kTo;
    FillSplitBlock(tilingData, kCount);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(stateless_uniform, kNumBlocks, shape, seed, offset, from, to, y, workspace, tiling);

    auto* yData = reinterpret_cast<float*>(y);
    for (int64_t i = 0; i < kCount; ++i) {
        EXPECT_GE(yData[i], kFrom) << "Element " << i << " is below from";
        EXPECT_LT(yData[i], kTo) << "Element " << i << " is >= to";
    }

    AscendC::GmFree(shape);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(from);
    AscendC::GmFree(to);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test 3: Determinism - same seed/offset produce same output
TEST_F(StatelessUniformKernelTest, determinism)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* from = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(double))));
    auto* to = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(double))));
    auto* y1 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* y2 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    reinterpret_cast<int64_t*>(shape)[0] = kElementCount;
    *reinterpret_cast<int64_t*>(seed) = 777;
    *reinterpret_cast<int64_t*>(offset) = 0;
    *reinterpret_cast<double*>(from) = 0.0;
    *reinterpret_cast<double*>(to) = 1.0;

    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 777;
    tilingData->offset = 0;
    tilingData->ubSize = 229376;
    tilingData->fromFp32 = 0.0f;
    tilingData->toFp32 = 1.0f;
    FillSplitBlock(tilingData, kElementCount);

    // Run 1
    std::memset(y1, 0, kElementCount * sizeof(float));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(stateless_uniform, kNumBlocks, shape, seed, offset, from, to, y1, workspace, tiling);

    // Run 2
    std::memset(y2, 0, kElementCount * sizeof(float));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(stateless_uniform, kNumBlocks, shape, seed, offset, from, to, y2, workspace, tiling);

    auto* y1Data = reinterpret_cast<float*>(y1);
    auto* y2Data = reinterpret_cast<float*>(y2);
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_EQ(y1Data[i], y2Data[i]) << "Mismatch at index " << i;
    }

    AscendC::GmFree(shape);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(from);
    AscendC::GmFree(to);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test 4: Different seed produces different output
TEST_F(StatelessUniformKernelTest, different_seed)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* from = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(double))));
    auto* to = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(double))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    reinterpret_cast<int64_t*>(shape)[0] = kElementCount;
    *reinterpret_cast<int64_t*>(offset) = 0;
    *reinterpret_cast<double*>(from) = 0.0;
    *reinterpret_cast<double*>(to) = 1.0;

    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->offset = 0;
    tilingData->ubSize = 229376;
    tilingData->fromFp32 = 0.0f;
    tilingData->toFp32 = 1.0f;
    FillSplitBlock(tilingData, kElementCount);

    // Run with seed=42
    float result1[kElementCount];
    *reinterpret_cast<int64_t*>(seed) = 42;
    tilingData->seed = 42;
    std::memset(y, 0, kElementCount * sizeof(float));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(stateless_uniform, kNumBlocks, shape, seed, offset, from, to, y, workspace, tiling);
    std::memcpy(result1, y, kElementCount * sizeof(float));

    // Run with seed=12345
    float result2[kElementCount];
    *reinterpret_cast<int64_t*>(seed) = 12345;
    tilingData->seed = 12345;
    std::memset(y, 0, kElementCount * sizeof(float));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(stateless_uniform, kNumBlocks, shape, seed, offset, from, to, y, workspace, tiling);
    std::memcpy(result2, y, kElementCount * sizeof(float));

    int diffCount = 0;
    for (int64_t i = 0; i < kElementCount; ++i) {
        if (result1[i] != result2[i]) {
            diffCount++;
        }
    }
    EXPECT_GT(diffCount, kElementCount / 2) << "Different seeds produced too few different values";

    AscendC::GmFree(shape);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(from);
    AscendC::GmFree(to);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
