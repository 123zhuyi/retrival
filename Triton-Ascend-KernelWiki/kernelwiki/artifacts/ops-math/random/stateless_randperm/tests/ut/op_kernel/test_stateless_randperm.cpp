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

inline int min(int x, int y) { return x < y ? x : y; }
inline long min(long x, long y) { return x < y ? x : y; }

#include "../../../op_kernel/stateless_randperm_apt.cpp"

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr int64_t kN = 8;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}
} // namespace

class StatelessRandpermKernelTest : public testing::Test {
};

TEST_F(StatelessRandpermKernelTest, smoke_int32)
{
    auto* n = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int32_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kN * sizeof(int32_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(StatelessRandpermTilingData))));

    *reinterpret_cast<int32_t*>(n) = kN;
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(offset) = 0;
    std::memset(y, 0, kN * sizeof(int32_t));
    std::memset(workspace, 0, Align32(16 * 1024 * 1024));

    auto* tilingData = reinterpret_cast<StatelessRandpermTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(StatelessRandpermTilingData));
    tilingData->n = kN;
    tilingData->randomBits = 32;
    tilingData->islandFactor = 1;
    tilingData->islandFactorTail = 1;
    tilingData->castFactor = 1;
    tilingData->castFactorTail = 1;
    tilingData->realCoreNum = kNumBlocks;
    tilingData->randomWkSizeByte = 1024;
    tilingData->subNTileCount = 1;
    tilingData->subNTile[0] = kN;
    tilingData->philoxKey[0] = 42;
    tilingData->philoxKey[1] = 0;
    tilingData->philoxOffset = 0;

    tilingData->sortTilingData.numTileDataSize = kN;
    tilingData->sortTilingData.unsortedDimParallel = 1;
    tilingData->sortTilingData.lastDimTileNum = 1;
    tilingData->sortTilingData.sortLoopTimes = 1;
    tilingData->sortTilingData.lastDimNeedCore = 1;
    tilingData->sortTilingData.keyParams0 = 1;
    tilingData->sortTilingData.keyParams1 = 256;
    tilingData->sortTilingData.tmpUbSize = 4096;
    tilingData->sortTilingData.lastAxisNum = kN;
    tilingData->sortTilingData.unsortedDimNum = 1;

    // Template params: randomType=2(int32), nIsInt32=1, schId=0, isInt32=1, isDescend=0
    auto func = stateless_randperm<2, 1, 0, 1, 0>;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(func, kNumBlocks, n, seed, offset, y, workspace, tiling);

    AscendC::GmFree(n);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
