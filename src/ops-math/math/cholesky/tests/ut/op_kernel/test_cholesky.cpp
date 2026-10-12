/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void cholesky(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);

class cholesky_test : public testing::Test {
   protected:
    static void SetUpTestCase() { cout << "cholesky_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "cholesky_test TearDown\n" << endl; }
};

TEST_F(cholesky_test, cholesky_kernel_test_1) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t selfSize = 32 * 32 * sizeof(float);
    size_t outSize = 32 * 32 * sizeof(float);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(CholeskyTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *selfGM = (uint8_t *)AscendC::GmAlloc(selfSize);
    uint8_t *outGM = (uint8_t *)AscendC::GmAlloc(outSize);
    
    memset(selfGM, 0, selfSize);
    memset(outGM, 0, outSize);

    CholeskyTilingData *tilingData = reinterpret_cast<CholeskyTilingData*>(tiling);
    tilingData->matSizeN = 32;
    tilingData->matrixNumCount = 1;

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(cholesky, 1, selfGM, outGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)selfGM);
    AscendC::GmFree((void*)outGM);
}

TEST_F(cholesky_test, cholesky_kernel_test_2) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t selfSize = 2 * 32 * 32 * sizeof(float);
    size_t outSize = 2 * 32 * 32 * sizeof(float);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(CholeskyTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *selfGM = (uint8_t *)AscendC::GmAlloc(selfSize);
    uint8_t *outGM = (uint8_t *)AscendC::GmAlloc(outSize);
    
    memset(selfGM, 0, selfSize);
    memset(outGM, 0, outSize);

    CholeskyTilingData *tilingData = reinterpret_cast<CholeskyTilingData*>(tiling);
    tilingData->matSizeN = 32;
    tilingData->matrixNumCount = 2;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(cholesky, 8, selfGM, outGM, workspace, (uint8_t*)(tilingData));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)selfGM);
    AscendC::GmFree((void*)outGM);
}