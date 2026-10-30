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
#include "./stateless_normal_tiling.h"

extern "C" __global__ __aicore__ void stateless_normal(GM_ADDR shape, GM_ADDR seed, GM_ADDR offset, GM_ADDR mean,
                                                       GM_ADDR stdev, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {

using OutputType = DTYPE_Y;
using MeanType = DTYPE_MEAN;
using StdevType = DTYPE_STDEV;
constexpr size_t kOutputElemSize = sizeof(OutputType);
constexpr size_t kMeanElemSize = sizeof(MeanType);
constexpr size_t kStdevElemSize = sizeof(StdevType);

constexpr uint32_t kNumBlocks = 1;
constexpr int64_t kElementCount = 256;
constexpr uint64_t kTilingKey = 100; // unified: BothTensor only (matches base class DEFAULT_TILING_KEY)

// GPU equivalent constants (must match base class constants in random_tiling_arch35.h)
constexpr int64_t GPU_BLOCK_SIZE = 256;
constexpr int64_t SM_COUNT = 78;
constexpr int64_t MAX_THREADS_PER_SM = 2048;
constexpr int64_t BLOCKS_PER_SM = MAX_THREADS_PER_SM / GPU_BLOCK_SIZE;
constexpr int64_t MAX_GENERATOR_OFFSETS = 4;

inline size_t Align32(size_t size) { return (size + 31U) / 32U * 32U; }

// Compute execution policy for a split block (mirrors CalcExecutionPoliciesForBlocks)
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

// Fill a single split block with execution policy
inline void FillSingleBlock(RandomUnifiedSimtTilingDataStruct* tilingData, int64_t numel, int64_t seedVal,
                            int64_t offsetVal)
{
    tilingData->splitBlockCount = 1;
    tilingData->splitBlocks[0].numel = numel;
    tilingData->splitBlocks[0].gmOffset = 0;
    int64_t counterOffset = 0;
    CalcBlockPolicy(numel, tilingData->splitBlocks[0].grid, tilingData->splitBlocks[0].totalThreads, counterOffset);
    tilingData->splitBlocks[0].kernelOffset = offsetVal;
}

inline float ReadAsFloat(const uint8_t* buf, int64_t idx)
{
    return static_cast<float>(reinterpret_cast<const OutputType*>(buf)[idx]);
}
} // namespace

class StatelessNormalKernelTest : public testing::Test {};

// Test 1: standard normal N(0,1), mean/stdev broadcast to full tensor
TEST_F(StatelessNormalKernelTest, smoke_standard_normal)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int64_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* mean = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kMeanElemSize)));
    auto* stdev = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kStdevElemSize)));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kOutputElemSize)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * kOutputElemSize);
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));

    reinterpret_cast<int64_t*>(shape)[0] = 16;
    reinterpret_cast<int64_t*>(shape)[1] = 16;
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(offset) = 0;
    // mean=0, stdev=1 broadcast to full tensor (simulating L2 BroadcastTo)
    auto* meanData = reinterpret_cast<MeanType*>(mean);
    auto* stdevData = reinterpret_cast<StdevType*>(stdev);
    for (int64_t i = 0; i < kElementCount; ++i) {
        meanData[i] = static_cast<MeanType>(0.0f);
        stdevData[i] = static_cast<StdevType>(1.0f);
    }

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 42;
    tilingData->offset = 0;
    FillSingleBlock(tilingData, kElementCount, 42, 0);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_normal, kNumBlocks, shape, seed, offset, mean, stdev, y, workspace, tiling);

    int nonZeroCount = 0;
    for (int64_t i = 0; i < kElementCount; ++i) {
        float val = ReadAsFloat(y, i);
        EXPECT_FALSE(std::isnan(val)) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(val)) << "Inf at index " << i;
        EXPECT_GT(val, -10.0f) << "Value too small at index " << i;
        EXPECT_LT(val, 10.0f) << "Value too large at index " << i;
        if (val != 0.0f) {
            nonZeroCount++;
        }
    }
    EXPECT_GT(nonZeroCount, kElementCount / 2) << "Too many zero values in normal distribution";

    AscendC::GmFree(shape);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(mean);
    AscendC::GmFree(stdev);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test 2: shifted normal N(5,2), mean/stdev broadcast to full tensor
TEST_F(StatelessNormalKernelTest, smoke_shifted_normal)
{
    constexpr float kMean = 5.0f;
    constexpr float kStdev = 2.0f;

    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(1 * sizeof(int64_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* mean = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kMeanElemSize)));
    auto* stdev = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kStdevElemSize)));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kOutputElemSize)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * kOutputElemSize);
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));

    reinterpret_cast<int64_t*>(shape)[0] = kElementCount;
    *reinterpret_cast<int64_t*>(seed) = 12345;
    *reinterpret_cast<int64_t*>(offset) = 0;
    auto* meanData = reinterpret_cast<MeanType*>(mean);
    auto* stdevData = reinterpret_cast<StdevType*>(stdev);
    for (int64_t i = 0; i < kElementCount; ++i) {
        meanData[i] = static_cast<MeanType>(kMean);
        stdevData[i] = static_cast<StdevType>(kStdev);
    }

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 12345;
    tilingData->offset = 0;
    FillSingleBlock(tilingData, kElementCount, 12345, 0);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_normal, kNumBlocks, shape, seed, offset, mean, stdev, y, workspace, tiling);

    double sum = 0.0;
    for (int64_t i = 0; i < kElementCount; ++i) {
        float val = ReadAsFloat(y, i);
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
        sum += static_cast<double>(val);
    }
    double sampleMean = sum / kElementCount;
    EXPECT_NEAR(sampleMean, kMean, 1.5) << "Sample mean too far from expected";

    AscendC::GmFree(shape);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(mean);
    AscendC::GmFree(stdev);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test 3: mean=tensor(10.0), stdev=tensor(0.5) - per-element mean/stdev
TEST_F(StatelessNormalKernelTest, smoke_mean_tensor)
{
    constexpr float kStdev = 0.5f;
    constexpr float kMeanVal = 10.0f;

    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(1 * sizeof(int64_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* mean = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kMeanElemSize)));
    auto* stdev = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kStdevElemSize)));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kOutputElemSize)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * kOutputElemSize);
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));

    reinterpret_cast<int64_t*>(shape)[0] = kElementCount;
    *reinterpret_cast<int64_t*>(seed) = 99;
    *reinterpret_cast<int64_t*>(offset) = 0;
    auto* meanData = reinterpret_cast<MeanType*>(mean);
    auto* stdevData = reinterpret_cast<StdevType*>(stdev);
    for (int64_t i = 0; i < kElementCount; ++i) {
        meanData[i] = static_cast<MeanType>(kMeanVal);
        stdevData[i] = static_cast<StdevType>(kStdev);
    }

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 99;
    tilingData->offset = 0;
    FillSingleBlock(tilingData, kElementCount, 99, 0);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_normal, kNumBlocks, shape, seed, offset, mean, stdev, y, workspace, tiling);

    double sum = 0.0;
    for (int64_t i = 0; i < kElementCount; ++i) {
        float val = ReadAsFloat(y, i);
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
        sum += static_cast<double>(val);
    }
    double sampleMean = sum / kElementCount;
    EXPECT_NEAR(sampleMean, static_cast<double>(kMeanVal), 0.5)
        << "Sample mean too far from expected for mean tensor mode";

    AscendC::GmFree(shape);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(mean);
    AscendC::GmFree(stdev);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test 4: Determinism - same seed/offset should produce same output
TEST_F(StatelessNormalKernelTest, determinism)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(1 * sizeof(int64_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* mean = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kMeanElemSize)));
    auto* stdev = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kStdevElemSize)));
    auto* y1 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kOutputElemSize)));
    auto* y2 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * kOutputElemSize)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    reinterpret_cast<int64_t*>(shape)[0] = kElementCount;
    *reinterpret_cast<int64_t*>(seed) = 777;
    *reinterpret_cast<int64_t*>(offset) = 0;
    auto* meanData = reinterpret_cast<MeanType*>(mean);
    auto* stdevData = reinterpret_cast<StdevType*>(stdev);
    for (int64_t i = 0; i < kElementCount; ++i) {
        meanData[i] = static_cast<MeanType>(0.0f);
        stdevData[i] = static_cast<StdevType>(1.0f);
    }

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 777;
    tilingData->offset = 0;
    FillSingleBlock(tilingData, kElementCount, 777, 0);

    // Run 1
    std::memset(y1, 0, kElementCount * kOutputElemSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_normal, kNumBlocks, shape, seed, offset, mean, stdev, y1, workspace, tiling);

    // Run 2 (same inputs)
    std::memset(y2, 0, kElementCount * kOutputElemSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_normal, kNumBlocks, shape, seed, offset, mean, stdev, y2, workspace, tiling);

    EXPECT_EQ(std::memcmp(y1, y2, kElementCount * kOutputElemSize), 0)
        << "Determinism check failed: same seed/offset produced different output";

    AscendC::GmFree(shape);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(mean);
    AscendC::GmFree(stdev);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
