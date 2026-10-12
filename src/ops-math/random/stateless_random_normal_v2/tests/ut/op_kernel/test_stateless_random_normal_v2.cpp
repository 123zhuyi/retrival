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

extern "C" __global__ __aicore__ void stateless_random_normal_v2(
    GM_ADDR shape, GM_ADDR key, GM_ADDR counter, GM_ADDR alg, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr uint64_t kTilingKey = 101;
constexpr int64_t kElementCount = 256;

struct StatelessRandomNormalV2TilingLayout {
    uint32_t blockNum;
    uint32_t blockTilingSize;
    uint32_t tailBlockTilingSize;
    uint32_t ubTilingSize;
    uint32_t alg;
    uint32_t key[2];
    uint32_t counter[4];
};

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}
} // namespace

class StatelessRandomNormalV2KernelTest : public testing::Test {
};

TEST_F(StatelessRandomNormalV2KernelTest, smoke_float)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* key = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(uint32_t))));
    auto* counter = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(4 * sizeof(uint32_t))));
    auto* alg = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int32_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(StatelessRandomNormalV2TilingLayout))));

    std::memset(y, 0, kElementCount * sizeof(float));
    std::memset(tiling, 0, sizeof(StatelessRandomNormalV2TilingLayout));
    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    reinterpret_cast<uint32_t*>(key)[0] = 42;
    reinterpret_cast<uint32_t*>(key)[1] = 0;
    reinterpret_cast<uint32_t*>(counter)[0] = 0;
    reinterpret_cast<uint32_t*>(counter)[1] = 0;
    reinterpret_cast<uint32_t*>(counter)[2] = 0;
    reinterpret_cast<uint32_t*>(counter)[3] = 0;
    *reinterpret_cast<int32_t*>(alg) = 1;

    auto* tilingData = reinterpret_cast<StatelessRandomNormalV2TilingLayout*>(tiling);
    tilingData->blockNum = kNumBlocks;
    tilingData->blockTilingSize = kElementCount;
    tilingData->tailBlockTilingSize = kElementCount;
    tilingData->ubTilingSize = kElementCount;
    tilingData->alg = 1;
    tilingData->key[0] = 42;
    tilingData->key[1] = 0;
    tilingData->counter[0] = 0;
    tilingData->counter[1] = 0;
    tilingData->counter[2] = 0;
    tilingData->counter[3] = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_random_normal_v2, kNumBlocks, shape, key, counter, alg, y, workspace, tiling);

    AscendC::GmFree(shape);
    AscendC::GmFree(key);
    AscendC::GmFree(counter);
    AscendC::GmFree(alg);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
