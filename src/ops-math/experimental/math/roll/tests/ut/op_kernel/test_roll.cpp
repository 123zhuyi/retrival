/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "roll_tiling.h"
#include "../../../op_kernel/roll.cpp"

class RollKernelTest : public testing::Test {};

__global__ __aicore__ void roll_float_test(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    RunRollKernel<float>(x, y, tiling);
}

TEST_F(RollKernelTest, kernel_launch_smoke)
{
    constexpr size_t size = 6;
    constexpr int64_t ubElements = 64 * 1024 / static_cast<int64_t>(sizeof(float));
    constexpr uint32_t numBlocks = 1;

    std::vector<float> xHost = {0, 1, 2, 3, 4, 5};
    std::vector<float> yHost(size, 0);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(size * sizeof(float));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(size * sizeof(float));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(RollTilingData));

    memcpy(x, xHost.data(), size * sizeof(float));
    memcpy(y, yHost.data(), size * sizeof(float));

    auto* tilingData = reinterpret_cast<RollTilingData*>(tiling);
    memset(tilingData, 0, sizeof(RollTilingData));
    tilingData->totalNum = size;
    tilingData->dimNum = 1;
    tilingData->perCoreElements = size;
    tilingData->lastCoreElements = size;
    tilingData->usedCoreNum = 1;
    tilingData->ubElements = ubElements;
    tilingData->blockFactor = size;
    tilingData->ubFactor = ubElements;
    tilingData->activeDimCount = 1;
    tilingData->activeDim = 0;
    tilingData->dimSize = size;
    tilingData->innerSize = 1;
    tilingData->activeShift = 1;
    tilingData->shapes[0] = size;
    tilingData->strides[0] = 1;
    tilingData->shifts[0] = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(roll_float_test, numBlocks, x, y, workspace, tiling);

    memcpy(yHost.data(), y, size * sizeof(float));
    EXPECT_EQ(yHost, (std::vector<float>{5, 0, 1, 2, 3, 4}));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(RollKernelTest, kernel_complex64_moves_each_value_as_eight_bytes)
{
    constexpr size_t size = 6;
    constexpr int64_t ubElements = 64 * 1024 / static_cast<int64_t>(sizeof(uint64_t));
    constexpr uint32_t numBlocks = 1;

    std::vector<uint64_t> xHost = {
        0x1111111122222222ULL, 0x3333333344444444ULL, 0x5555555566666666ULL,
        0x7777777788888888ULL, 0x99999999AAAAAAAAULL, 0xBBBBBBBBCCCCCCCCULL,
    };
    std::vector<uint64_t> yHost(size, 0);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(size * sizeof(uint64_t));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(size * sizeof(uint64_t));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(RollTilingData));

    memcpy(x, xHost.data(), size * sizeof(uint64_t));
    memcpy(y, yHost.data(), size * sizeof(uint64_t));

    auto* tilingData = reinterpret_cast<RollTilingData*>(tiling);
    memset(tilingData, 0, sizeof(RollTilingData));
    tilingData->totalNum = size;
    tilingData->dimNum = 1;
    tilingData->perCoreElements = size;
    tilingData->lastCoreElements = size;
    tilingData->usedCoreNum = 1;
    tilingData->ubElements = ubElements;
    tilingData->blockFactor = size;
    tilingData->ubFactor = ubElements;
    tilingData->activeDimCount = 1;
    tilingData->activeDim = 0;
    tilingData->dimSize = size;
    tilingData->innerSize = 1;
    tilingData->activeShift = 1;
    tilingData->shapes[0] = size;
    tilingData->strides[0] = 1;
    tilingData->shifts[0] = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((roll<0>), numBlocks, x, y, workspace, tiling);

    memcpy(yHost.data(), y, size * sizeof(uint64_t));
    EXPECT_EQ(yHost, (std::vector<uint64_t>{
                         0xBBBBBBBBCCCCCCCCULL,
                         0x1111111122222222ULL,
                         0x3333333344444444ULL,
                         0x5555555566666666ULL,
                         0x7777777788888888ULL,
                         0x99999999AAAAAAAAULL,
                     }));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(RollKernelTest, kernel_complex64_large_non_last_dim_stays_within_ub)
{
    constexpr int64_t outerSize = 2500;
    constexpr int64_t dimSize = 2;
    constexpr int64_t innerSize = 2;
    constexpr int64_t blockSize = dimSize * innerSize;
    constexpr int64_t size = outerSize * blockSize;
    constexpr int64_t ubElements = 64 * 1024 / static_cast<int64_t>(sizeof(uint64_t));
    constexpr int64_t perCoreElements = size;
    constexpr uint32_t numBlocks = 1;

    std::vector<uint64_t> xHost(size);
    std::vector<uint64_t> yHost(size, 0);
    std::vector<uint64_t> expected(size);
    for (int64_t i = 0; i < size; ++i) {
        xHost[static_cast<size_t>(i)] = static_cast<uint64_t>(i);
    }
    for (int64_t block = 0; block < outerSize; ++block) {
        const int64_t base = block * blockSize;
        expected[static_cast<size_t>(base)] = static_cast<uint64_t>(base + innerSize);
        expected[static_cast<size_t>(base + 1)] = static_cast<uint64_t>(base + innerSize + 1);
        expected[static_cast<size_t>(base + innerSize)] = static_cast<uint64_t>(base);
        expected[static_cast<size_t>(base + innerSize + 1)] = static_cast<uint64_t>(base + 1);
    }

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(size * sizeof(uint64_t));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(size * sizeof(uint64_t));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(RollTilingData));

    memcpy(x, xHost.data(), size * sizeof(uint64_t));
    memcpy(y, yHost.data(), size * sizeof(uint64_t));

    auto* tilingData = reinterpret_cast<RollTilingData*>(tiling);
    memset(tilingData, 0, sizeof(RollTilingData));
    tilingData->totalNum = size;
    tilingData->dimNum = 3;
    tilingData->perCoreElements = perCoreElements;
    tilingData->lastCoreElements = size - (numBlocks - 1) * perCoreElements;
    tilingData->usedCoreNum = numBlocks;
    tilingData->ubElements = ubElements;
    tilingData->blockFactor = perCoreElements;
    tilingData->ubFactor = ubElements;
    tilingData->activeDimCount = 1;
    tilingData->activeDim = 1;
    tilingData->dimSize = dimSize;
    tilingData->innerSize = innerSize;
    tilingData->activeShift = 1;
    tilingData->shapes[0] = outerSize;
    tilingData->shapes[1] = dimSize;
    tilingData->shapes[2] = innerSize;
    tilingData->strides[0] = blockSize;
    tilingData->strides[1] = innerSize;
    tilingData->strides[2] = 1;
    tilingData->shifts[1] = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((roll<0>), numBlocks, x, y, workspace, tiling);

    memcpy(yHost.data(), y, size * sizeof(uint64_t));
    EXPECT_EQ(yHost, expected);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
