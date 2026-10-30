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

#include "../../../op_kernel/arch35/atan_grad_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void atan_grad(
    GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);

class AtanGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "AtanGradKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "AtanGradKernelTest TearDown" << endl; }
};

TEST_F(AtanGradKernelTest, test_fp32_basic)
{
    constexpr size_t numElements = 256;
    size_t dataSize = numElements * sizeof(float);
    size_t tilingSize = sizeof(AtanGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AtanGradTilingData* tilingData = reinterpret_cast<AtanGradTilingData*>(tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = numElements;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(atan_grad, blockDim, y, dy, z, workspace, tiling);

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AtanGradKernelTest, test_fp32_large)
{
    constexpr size_t numElements = 2048;
    size_t dataSize = numElements * sizeof(float);
    size_t tilingSize = sizeof(AtanGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AtanGradTilingData* tilingData = reinterpret_cast<AtanGradTilingData*>(tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = 1024;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(atan_grad, blockDim, y, dy, z, workspace, tiling);

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
