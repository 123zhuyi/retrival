/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "stateless_truncated_normal_v2_tiling.h"

extern "C" __global__ __aicore__ void stateless_truncated_normal_v2(
    GM_ADDR shape, GM_ADDR key, GM_ADDR counter, GM_ADDR alg, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {

using OutputType = DTYPE_Y;
constexpr size_t kOutputElemSize = sizeof(OutputType);

constexpr uint32_t kNumBlocks = 1;
constexpr int64_t kElementCount = 256;
constexpr uint64_t kTilingKey = 100;

// GPU equivalent constants (must match base class constants in random_tiling_arch35.h)
constexpr int64_t GPU_BLOCK_SIZE = 256;
constexpr int64_t SM_COUNT = 78;
constexpr int64_t MAX_THREADS_PER_SM = 2048;
constexpr int64_t MAX_GENERATOR_OFFSETS = 4;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}

inline void CalcBlockPolicy(int64_t numel, int64_t& grid, int64_t& totalThreads, int64_t& counterOffset)
{
    grid = (numel + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
    int64_t blocksPerSM = MAX_THREADS_PER_SM / GPU_BLOCK_SIZE;
    if (grid > SM_COUNT * blocksPerSM) {
        grid = SM_COUNT * blocksPerSM;
    }
    totalThreads = grid * GPU_BLOCK_SIZE;
    int64_t unroll = 4;
    int64_t threadsPerRound = totalThreads * unroll;
    int64_t roundsNeeded = (numel + threadsPerRound - 1) / threadsPerRound;
    counterOffset = roundsNeeded * MAX_GENERATOR_OFFSETS;
}

inline void FillSingleBlock(RandomUnifiedSimtTilingDataStruct* tilingData, int64_t numel)
{
    tilingData->splitBlockCount = 1;
    tilingData->splitBlocks[0].numel = numel;
    tilingData->splitBlocks[0].gmOffset = 0;
    int64_t counterOffset = 0;
    CalcBlockPolicy(numel, tilingData->splitBlocks[0].grid,
                    tilingData->splitBlocks[0].totalThreads,
                    counterOffset);
    tilingData->splitBlocks[0].kernelOffset = 0;
}

inline float ReadAsFloat(const uint8_t* buf, int64_t idx)
{
    return static_cast<float>(reinterpret_cast<const OutputType*>(buf)[idx]);
}
} // namespace

class StatelessTruncatedNormalV2KernelTest : public testing::Test {
};

// Test 1: Basic truncated normal generation - verify output in (-2, 2)
TEST_F(StatelessTruncatedNormalV2KernelTest, smoke_truncated_normal)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* key = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(uint64_t))));
    auto* counter = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(uint64_t))));
    auto* alg = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int32_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kOutputElemSize)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * kOutputElemSize);
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));

    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    *reinterpret_cast<uint64_t*>(key) = 42ULL;
    reinterpret_cast<uint64_t*>(counter)[0] = 0ULL;
    reinterpret_cast<uint64_t*>(counter)[1] = 0ULL;
    *reinterpret_cast<int32_t*>(alg) = 1;

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 0;
    tilingData->offset = 0;
    FillSingleBlock(tilingData, kElementCount);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_truncated_normal_v2, kNumBlocks, shape, key, counter, alg, y, workspace, tiling);

    int nonZeroCount = 0;
    for (int64_t i = 0; i < kElementCount; ++i) {
        float val = ReadAsFloat(y, i);
        EXPECT_FALSE(std::isnan(val)) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(val)) << "Inf at index " << i;
        EXPECT_GT(val, -2.0f) << "Value <= -2.0 at index " << i;
        EXPECT_LT(val, 2.0f) << "Value >= 2.0 at index " << i;
        if (val != 0.0f) {
            nonZeroCount++;
        }
    }
    EXPECT_GT(nonZeroCount, kElementCount / 2) << "Too many zero values";

    AscendC::GmFree(shape);
    AscendC::GmFree(key);
    AscendC::GmFree(counter);
    AscendC::GmFree(alg);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test 2: Different key/counter produce different output
TEST_F(StatelessTruncatedNormalV2KernelTest, smoke_different_key)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(1 * sizeof(int32_t))));
    auto* key = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(uint64_t))));
    auto* counter = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(uint64_t))));
    auto* alg = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int32_t))));
    auto* y1 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kOutputElemSize)));
    auto* y2 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kOutputElemSize)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    reinterpret_cast<int32_t*>(shape)[0] = kElementCount;
    reinterpret_cast<uint64_t*>(counter)[0] = 0ULL;
    reinterpret_cast<uint64_t*>(counter)[1] = 0ULL;
    *reinterpret_cast<int32_t*>(alg) = 1;

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 0;
    tilingData->offset = 0;
    FillSingleBlock(tilingData, kElementCount);

    // Run with key=100
    *reinterpret_cast<uint64_t*>(key) = 100ULL;
    std::memset(y1, 0, kElementCount * kOutputElemSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_truncated_normal_v2, kNumBlocks, shape, key, counter, alg, y1, workspace, tiling);

    // Run with key=200
    *reinterpret_cast<uint64_t*>(key) = 200ULL;
    std::memset(y2, 0, kElementCount * kOutputElemSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_truncated_normal_v2, kNumBlocks, shape, key, counter, alg, y2, workspace, tiling);

    // Different keys should produce different outputs
    EXPECT_NE(std::memcmp(y1, y2, kElementCount * kOutputElemSize), 0)
        << "Different keys produced identical output";

    AscendC::GmFree(shape);
    AscendC::GmFree(key);
    AscendC::GmFree(counter);
    AscendC::GmFree(alg);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test 3: Determinism - same key/counter produce bit-exact same output
TEST_F(StatelessTruncatedNormalV2KernelTest, determinism)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(1 * sizeof(int32_t))));
    auto* key = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(uint64_t))));
    auto* counter = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(uint64_t))));
    auto* alg = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int32_t))));
    auto* y1 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kOutputElemSize)));
    auto* y2 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kOutputElemSize)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    reinterpret_cast<int32_t*>(shape)[0] = kElementCount;
    *reinterpret_cast<uint64_t*>(key) = 777ULL;
    reinterpret_cast<uint64_t*>(counter)[0] = 123ULL;
    reinterpret_cast<uint64_t*>(counter)[1] = 456ULL;
    *reinterpret_cast<int32_t*>(alg) = 1;

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 0;
    tilingData->offset = 0;
    FillSingleBlock(tilingData, kElementCount);

    // Run 1
    std::memset(y1, 0, kElementCount * kOutputElemSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_truncated_normal_v2, kNumBlocks, shape, key, counter, alg, y1, workspace, tiling);

    // Run 2 (same inputs)
    std::memset(y2, 0, kElementCount * kOutputElemSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_truncated_normal_v2, kNumBlocks, shape, key, counter, alg, y2, workspace, tiling);

    EXPECT_EQ(std::memcmp(y1, y2, kElementCount * kOutputElemSize), 0)
        << "Determinism check failed: same key/counter produced different output";

    AscendC::GmFree(shape);
    AscendC::GmFree(key);
    AscendC::GmFree(counter);
    AscendC::GmFree(alg);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test 4: Large key/counter values (boundary test)
TEST_F(StatelessTruncatedNormalV2KernelTest, smoke_large_key_counter)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(1 * sizeof(int32_t))));
    auto* key = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(uint64_t))));
    auto* counter = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(uint64_t))));
    auto* alg = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int32_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kOutputElemSize)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * kOutputElemSize);
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));

    reinterpret_cast<int32_t*>(shape)[0] = kElementCount;
    *reinterpret_cast<uint64_t*>(key) = 0xFFFFFFFFFFFFFFFFULL;
    reinterpret_cast<uint64_t*>(counter)[0] = 0xFFFFFFFFFFFFFFFFULL;
    reinterpret_cast<uint64_t*>(counter)[1] = 0xFFFFFFFFFFFFFFFFULL;
    *reinterpret_cast<int32_t*>(alg) = 1;

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 0;
    tilingData->offset = 0;
    FillSingleBlock(tilingData, kElementCount);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_truncated_normal_v2, kNumBlocks, shape, key, counter, alg, y, workspace, tiling);

    for (int64_t i = 0; i < kElementCount; ++i) {
        float val = ReadAsFloat(y, i);
        EXPECT_FALSE(std::isnan(val)) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(val)) << "Inf at index " << i;
        EXPECT_GT(val, -2.0f) << "Value <= -2.0 at index " << i;
        EXPECT_LT(val, 2.0f) << "Value >= 2.0 at index " << i;
    }

    AscendC::GmFree(shape);
    AscendC::GmFree(key);
    AscendC::GmFree(counter);
    AscendC::GmFree(alg);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
