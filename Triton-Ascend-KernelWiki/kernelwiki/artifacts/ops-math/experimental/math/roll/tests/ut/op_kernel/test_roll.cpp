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

TEST_F(RollKernelTest, kernel_launch_smoke)
{
    constexpr size_t size = 6;
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
    tilingData->ubElements = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
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

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
