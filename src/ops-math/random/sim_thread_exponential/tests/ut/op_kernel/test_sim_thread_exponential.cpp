/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sim_thread_exponential.cpp
 * \brief
 */

#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"

extern "C" __global__ __aicore__ void sim_thread_exponential(
    GM_ADDR self, GM_ADDR self_ref, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr uint64_t kTilingKeyFp32 = 3;
constexpr uint64_t kTilingKeyFp16 = 1;
constexpr int64_t kElementCount = 256;
constexpr int64_t kSeed = 42;
constexpr int64_t kOffset = 0;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}
} // namespace

class SimThreadExponentialKernelTest : public testing::Test {
};

TEST_F(SimThreadExponentialKernelTest, smoke_float32)
{
    auto* self = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(self, 0, kElementCount * sizeof(float));
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = kSeed;
    tilingData->offset = kOffset;
    tilingData->prob = 1.0f;  // lambda = 1.0

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKeyFp32);
    ICPU_RUN_KF(sim_thread_exponential, kNumBlocks, self, self, workspace, tiling);

    // 基本合理性检查：所有输出应为有限且大于0
    auto* selfData = reinterpret_cast<float*>(self);
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_TRUE(std::isfinite(selfData[i])) << "Element " << i << " is not finite: " << selfData[i];
        EXPECT_GT(selfData[i], 0.0f) << "Element " << i << " <= 0: " << selfData[i];
    }

    AscendC::GmFree(self);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SimThreadExponentialKernelTest, smoke_float16)
{
    auto* self = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float) / 2)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(self, 0, kElementCount * sizeof(float) / 2);
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = kSeed;
    tilingData->offset = kOffset;
    tilingData->prob = 1.0f;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKeyFp16);
    ICPU_RUN_KF(sim_thread_exponential, kNumBlocks, self, self, workspace, tiling);

    AscendC::GmFree(self);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
