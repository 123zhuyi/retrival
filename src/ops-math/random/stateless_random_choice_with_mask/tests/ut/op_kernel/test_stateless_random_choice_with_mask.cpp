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
#include "../../../op_kernel/stateless_random_choice_with_mask.cpp"

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr uint32_t kSchMode = 0;
constexpr int64_t kM = 4;
constexpr int64_t kN = 4;
constexpr int64_t kInputSize = kM * kN;
constexpr int32_t kCount = 2;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}
} // namespace

class StatelessRandomChoiceWithMaskKernelTest : public testing::Test {
};

TEST_F(StatelessRandomChoiceWithMaskKernelTest, smoke_test)
{
    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kInputSize * sizeof(int32_t))));
    auto* count = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int32_t))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kCount * kN * sizeof(int32_t))));
    auto* mask = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kCount * sizeof(int32_t))));
    auto* shapeOut = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(
        AscendC::GmAlloc(Align32(sizeof(StatelessRandomChoiceWithMaskSimtTilingData))));

    auto* xData = reinterpret_cast<int32_t*>(x);
    for (int64_t i = 0; i < kInputSize; ++i) {
        xData[i] = 1;
    }
    *reinterpret_cast<int32_t*>(count) = kCount;
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(offset) = 0;
    std::memset(y, 0, Align32(kCount * kN * sizeof(int32_t)));
    std::memset(mask, 0, Align32(kCount * sizeof(int32_t)));
    std::memset(workspace, 0, Align32(16 * 1024 * 1024));

    auto* tilingData = reinterpret_cast<StatelessRandomChoiceWithMaskSimtTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(StatelessRandomChoiceWithMaskSimtTilingData));
    tilingData->blockNum = kNumBlocks;
    tilingData->normalCoreProNum = kInputSize;
    tilingData->m = kM;
    tilingData->n = kN;
    tilingData->seed = 42;
    tilingData->offset = 0;
    tilingData->inputSize = kInputSize;
    tilingData->noZeroCalcCount = kInputSize;
    tilingData->noZeroWorkspaceSize = 4096;
    tilingData->randomWorkspaceSize = 4096;
    tilingData->ubSize = 65536;
    tilingData->count = kCount;
    tilingData->inputDim = 2;
    tilingData->inputShape[0] = kM;
    tilingData->inputShape[1] = kN;

    auto func = stateless_random_choice_with_mask<kSchMode>;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(func, kNumBlocks, x, count, seed, offset, y, mask, shapeOut, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(count);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(shapeOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
