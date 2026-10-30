/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"

extern __global__ __aicore__ void stateless_sample_multinomial(
    GM_ADDR x, GM_ADDR seed, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {
using XType = DTYPE_X;
constexpr uint32_t kNumBlocks = 1;
constexpr uint64_t kTilingKey = 100;
constexpr int64_t kNumDist = 2;
constexpr int64_t kNumCat = 4;
constexpr int64_t kNumSamples = 8;
constexpr int64_t kElementCount = kNumDist * kNumSamples;
constexpr int64_t kCdfElementCount = kNumDist * kNumCat;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}

void FillTiling(RandomUnifiedSimtTilingDataStruct* tilingData, int64_t seed, int64_t offset)
{
    std::memset(tilingData, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = (kElementCount + 3) / 4;
    tilingData->seed = seed;
    tilingData->offset = offset;
    tilingData->extraInt64Param1 = kElementCount;
    tilingData->from = kNumSamples;
    tilingData->range = kNumCat;
    tilingData->splitBlockCount = 0;
}

void FillCdf(uint8_t* x)
{
    auto* xData = reinterpret_cast<XType*>(x);
    const float cdf[kCdfElementCount] = {
        0.10f, 0.30f, 0.60f, 1.00f,
        0.25f, 0.50f, 0.75f, 1.00f,
    };
    for (int64_t i = 0; i < kCdfElementCount; ++i) {
        xData[i] = static_cast<XType>(cdf[i]);
    }
}
} // namespace

class StatelessSampleMultinomialKernelTest : public testing::Test {
};

TEST_F(StatelessSampleMultinomialKernelTest, smoke_output_in_category_range)
{
    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kCdfElementCount * sizeof(XType))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(int64_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    FillCdf(x);
    std::memset(y, 0xFF, kElementCount * sizeof(int64_t));
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(offset) = 0;
    FillTiling(reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling), 42, 0);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_sample_multinomial, kNumBlocks, x, seed, offset, y, workspace, tiling);

    auto* yData = reinterpret_cast<int64_t*>(y);
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_GE(yData[i], 0);
        EXPECT_LT(yData[i], kNumCat);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(StatelessSampleMultinomialKernelTest, determinism)
{
    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kCdfElementCount * sizeof(XType))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y1 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(int64_t))));
    auto* y2 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(int64_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    FillCdf(x);
    *reinterpret_cast<int64_t*>(seed) = 12345;
    *reinterpret_cast<int64_t*>(offset) = 4;
    FillTiling(reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling), 12345, 4);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_sample_multinomial, kNumBlocks, x, seed, offset, y1, workspace, tiling);

    std::memset(y2, 0, kElementCount * sizeof(int64_t));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_sample_multinomial, kNumBlocks, x, seed, offset, y2, workspace, tiling);

    EXPECT_EQ(std::memcmp(y1, y2, kElementCount * sizeof(int64_t)), 0);

    AscendC::GmFree(x);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
