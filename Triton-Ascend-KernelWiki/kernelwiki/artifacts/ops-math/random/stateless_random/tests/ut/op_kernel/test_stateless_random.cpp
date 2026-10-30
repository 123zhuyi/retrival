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

extern "C" __global__ __aicore__ void stateless_random(
    GM_ADDR shape, GM_ADDR seed, GM_ADDR offset, GM_ADDR from, GM_ADDR to, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling);

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr uint64_t kTilingKey = 100;
constexpr int64_t kElementCount = 256;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}
} // namespace

class StatelessRandomKernelTest : public testing::Test {};

TEST_F(StatelessRandomKernelTest, smoke_int32_no_range)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* from = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* to = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(int32_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * sizeof(int32_t));
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(offset) = 0;
    *reinterpret_cast<int64_t*>(from) = 0;
    *reinterpret_cast<int64_t*>(to) = 100;

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 42;
    tilingData->offset = 0;
    tilingData->ubSize = 196608;
    tilingData->extraInt64Param1 = 4;
    tilingData->from = 0;
    tilingData->range = 100;
    tilingData->splitBlockCount = 1;
    tilingData->splitBlocks[0].numel = kElementCount;
    tilingData->splitBlocks[0].gmOffset = 0;
    tilingData->splitBlocks[0].grid = 1;
    tilingData->splitBlocks[0].totalThreads = 512;
    tilingData->splitBlocks[0].kernelOffset = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_random, kNumBlocks, shape, seed, offset, from, to, y, workspace, tiling);

    auto* yData = reinterpret_cast<int32_t*>(y);
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_GE(yData[i], 0);
        EXPECT_LT(yData[i], 100);
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

TEST_F(StatelessRandomKernelTest, smoke_int32_with_range)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* from = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* to = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(int32_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * sizeof(int32_t));
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(offset) = 0;
    *reinterpret_cast<int64_t*>(from) = 5;
    *reinterpret_cast<int64_t*>(to) = 15;

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 42;
    tilingData->offset = 0;
    tilingData->ubSize = 196608;
    tilingData->extraInt64Param1 = 4;
    tilingData->from = 5;
    tilingData->range = 10;
    tilingData->splitBlockCount = 1;
    tilingData->splitBlocks[0].numel = kElementCount;
    tilingData->splitBlocks[0].gmOffset = 0;
    tilingData->splitBlocks[0].grid = 1;
    tilingData->splitBlocks[0].totalThreads = 512;
    tilingData->splitBlocks[0].kernelOffset = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_random, kNumBlocks, shape, seed, offset, from, to, y, workspace, tiling);

    auto* yData = reinterpret_cast<int32_t*>(y);
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_GE(yData[i], 5);
        EXPECT_LT(yData[i], 15);
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
