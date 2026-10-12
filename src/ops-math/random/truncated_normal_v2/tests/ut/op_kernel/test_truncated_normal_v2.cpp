/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"

extern "C" __global__ __aicore__ void truncated_normal_v2(
    GM_ADDR shape, GM_ADDR offset, GM_ADDR y, GM_ADDR offset_ref, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr uint64_t kTilingKey = 100;
constexpr int64_t kElementCount = 256;
constexpr int64_t kExpectedOffset = kElementCount * 256;
constexpr float kRandomThreadR = 2.0f;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}
} // namespace

class TruncatedNormalV2KernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TruncatedNormalV2KernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TruncatedNormalV2KernelTest TearDown" << std::endl;
    }
};

TEST_F(TruncatedNormalV2KernelTest, smoke_float)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* offsetRef = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * sizeof(float));
    std::memset(offsetRef, 0, sizeof(int64_t));
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    reinterpret_cast<int64_t*>(offset)[0] = 0;
    reinterpret_cast<int64_t*>(offsetRef)[0] = 0;

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 10;
    tilingData->offset = 5;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(truncated_normal_v2, kNumBlocks, shape, offset, y, offsetRef, workspace, tiling);

    EXPECT_EQ(reinterpret_cast<int64_t*>(offset)[0], kExpectedOffset);
    EXPECT_EQ(reinterpret_cast<int64_t*>(offsetRef)[0], 0);

    auto* yData = reinterpret_cast<float*>(y);
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_LT(std::abs(yData[i]), kRandomThreadR);
    }

    AscendC::GmFree(shape);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(offsetRef);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
