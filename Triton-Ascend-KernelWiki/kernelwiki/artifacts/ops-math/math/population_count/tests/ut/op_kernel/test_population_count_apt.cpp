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

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#endif

#include "../../../op_kernel/arch35/population_count_tiling_data.h"

// Skip tiling_key.h (host-only macros) when compiling kernel UT
#define POPULATION_COUNT_TILING_KEY_H_
#include "../../../op_kernel/population_count_apt.cpp"

using namespace std;

class PopulationCountKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "PopulationCountKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "PopulationCountKernelTest TearDown" << endl; }
};

// Smoke test: kernel compiles, links, and runs without crashing.
// Full accuracy verification is done via ST tests on real NPU hardware
// (tikicpulib CPU sim does not correctly emulate uint16 ShiftRight/Ands).
TEST_F(PopulationCountKernelTest, test_all_zeros)
{
    constexpr size_t numElements = 256;
    size_t inputSize = numElements * sizeof(int16_t);
    size_t outputSize = numElements * sizeof(uint8_t);
    size_t tilingSize = sizeof(PopulationCountTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    memset(x, 0, inputSize);

    PopulationCountTilingData* tilingData = reinterpret_cast<PopulationCountTilingData*>(tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = numElements;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((population_count<int16_t, 0>), blockDim, x, y, workspace, tiling);

    uint8_t* yData = reinterpret_cast<uint8_t*>(y);
    for (size_t i = 0; i < numElements; ++i) {
        EXPECT_EQ(yData[i], 0) << "Expected 0 at index " << i;
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Smoke test for empty tensor (totalNum=0) — verifies early exit path.
TEST_F(PopulationCountKernelTest, test_empty_tensor)
{
    size_t tilingSize = sizeof(PopulationCountTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    PopulationCountTilingData* tilingData = reinterpret_cast<PopulationCountTilingData*>(tiling);
    tilingData->totalNum = 0;
    tilingData->blockFactor = 0;
    tilingData->ubFactor = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((population_count<int16_t, 0>), blockDim, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Smoke test for double-buffer path — verifies kernel runs with BUFFER_MODE=1.
TEST_F(PopulationCountKernelTest, test_double_buffer_zeros)
{
    constexpr size_t numElements = 2048;
    size_t inputSize = numElements * sizeof(int16_t);
    size_t outputSize = numElements * sizeof(uint8_t);
    size_t tilingSize = sizeof(PopulationCountTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    memset(x, 0, inputSize);

    PopulationCountTilingData* tilingData = reinterpret_cast<PopulationCountTilingData*>(tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = 1024;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((population_count<int16_t, 1>), blockDim, x, y, workspace, tiling);

    uint8_t* yData = reinterpret_cast<uint8_t*>(y);
    for (size_t i = 0; i < numElements; ++i) {
        EXPECT_EQ(yData[i], 0) << "Expected 0 at index " << i;
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
