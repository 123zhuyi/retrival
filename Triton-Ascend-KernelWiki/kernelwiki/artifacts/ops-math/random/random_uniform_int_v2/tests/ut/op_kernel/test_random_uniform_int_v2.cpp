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
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/random_uniform_int_v2.cpp"

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr uint32_t kOpType = 0;
constexpr int64_t kElementCount = 256;
constexpr int64_t kExpectedOffset = kElementCount * 256;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}
} // namespace

class RandomUniformIntV2KernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RandomUniformIntV2KernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RandomUniformIntV2KernelTest TearDown" << std::endl;
    }
};

TEST_F(RandomUniformIntV2KernelTest, smoke_int32)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* min = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int32_t))));
    auto* max = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int32_t))));
    auto* inOffset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(int32_t))));
    auto* outOffset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUniformIntV2TilingData4RegBase))));

    std::memset(y, 0, kElementCount * sizeof(int32_t));
    std::memset(tiling, 0, sizeof(RandomUniformIntV2TilingData4RegBase));
    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    reinterpret_cast<int32_t*>(min)[0] = 2;
    reinterpret_cast<int32_t*>(max)[0] = 5;
    reinterpret_cast<int64_t*>(inOffset)[0] = 0;
    reinterpret_cast<int64_t*>(outOffset)[0] = 0;

    auto* tilingData = reinterpret_cast<RandomUniformIntV2TilingData4RegBase*>(tiling);
    tilingData->blockNum = kNumBlocks;
    tilingData->normalCoreProNum = kElementCount;
    tilingData->tailCoreProNum = kElementCount;
    tilingData->singleUbSize = kElementCount;
    tilingData->seed = 10;
    tilingData->seed2 = 5;
    tilingData->outputSize = kElementCount;
    tilingData->range = 3;
    tilingData->lo = 2;

    auto func = random_uniform_int_v2<kOpType>;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(func, kNumBlocks, shape, min, max, inOffset, y, outOffset, workspace, tiling);

    EXPECT_EQ(reinterpret_cast<int64_t*>(outOffset)[0], kExpectedOffset);

    AscendC::GmFree(shape);
    AscendC::GmFree(min);
    AscendC::GmFree(max);
    AscendC::GmFree(inOffset);
    AscendC::GmFree(y);
    AscendC::GmFree(outOffset);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
