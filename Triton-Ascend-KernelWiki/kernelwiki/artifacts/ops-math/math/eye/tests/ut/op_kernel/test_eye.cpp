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
 * \file test_eye.cpp
 * \brief kernel UT for Eye operator
 */

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/eye_apt.cpp"

class EyeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "eye_test SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "eye_test TearDown" << std::endl;
    }
};

// ============================================================================
// SIMD path (tiling key 1000) - float32, small square matrix
// Eye(numRows=4, numColumns=4, batch=1) => output y: (4, 4)
// numPerBatch * typeSize = 16 * 4 = 64 <= 32796, use SIMD path
// ============================================================================
TEST_F(EyeTest, test_simd_float32_4x4)
{
    constexpr int64_t NUM_ROWS = 4;
    constexpr int64_t NUM_COLUMNS = 4;
    constexpr int64_t BATCH = 1;

    size_t outputByteSize = BATCH * NUM_ROWS * NUM_COLUMNS * sizeof(float);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    constexpr size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(EyeForAscendCTilingData));

    EyeForAscendCTilingData* tilingData = reinterpret_cast<EyeForAscendCTilingData*>(tiling);
    tilingData->usedCoreNum = 1;
    tilingData->normBlockData = 1;
    tilingData->tailBlockData = 1;
    tilingData->loopLength = 1;
    tilingData->numRows = NUM_ROWS;
    tilingData->numColumns = NUM_COLUMNS;
    tilingData->batch = BATCH;

    uint32_t numBlocks = 1;
    ICPU_SET_TILING_KEY(1000);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(eye, numBlocks, y, workspace, tiling);

    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// ============================================================================
// Large shape path (tiling key 2000) - float32, large matrix with uint32
// Eye(numRows=128, numColumns=128, batch=1) => output y: (128, 128)
// numPerBatch * typeSize = 16384 * 4 = 65536 > 32796, use large shape path
// allAxis = 16384, loopLength = 16384, 1 core
// ============================================================================
TEST_F(EyeTest, test_large_float32_128x128)
{
    constexpr int64_t NUM_ROWS = 128;
    constexpr int64_t NUM_COLUMNS = 128;
    constexpr int64_t BATCH = 1;

    size_t outputByteSize = BATCH * NUM_ROWS * NUM_COLUMNS * sizeof(float);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    constexpr size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(EyeForAscendCTilingData));

    EyeForAscendCTilingData* tilingData = reinterpret_cast<EyeForAscendCTilingData*>(tiling);
    tilingData->usedCoreNum = 1;
    tilingData->normBlockData = 16384;
    tilingData->tailBlockData = 16384;
    tilingData->loopLength = 16384;
    tilingData->numRows = NUM_ROWS;
    tilingData->numColumns = NUM_COLUMNS;
    tilingData->batch = BATCH;

    uint32_t numBlocks = 1;
    ICPU_SET_TILING_KEY(2000);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(eye, numBlocks, y, workspace, tiling);

    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
