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

#include "../../../op_kernel/arch35/add_mat_mat_elements_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void add_mat_mat_elements(
    GM_ADDR c, GM_ADDR a, GM_ADDR b, GM_ADDR beta, GM_ADDR alpha,
    GM_ADDR cOut, GM_ADDR workspace, GM_ADDR tiling);

class AddMatMatElementsKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "AddMatMatElementsKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "AddMatMatElementsKernelTest TearDown" << endl; }
};

TEST_F(AddMatMatElementsKernelTest, test_fp32_basic)
{
    constexpr size_t numElements = 1024;
    size_t dataSize = numElements * sizeof(float);
    size_t scalarSize = sizeof(float);
    size_t tilingSize = sizeof(AddMatMatElementsTilingData);
    uint32_t blockDim = 1;

    uint8_t* c = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* a = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* b = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(scalarSize);
    uint8_t* alpha = (uint8_t*)AscendC::GmAlloc(scalarSize);
    uint8_t* cOut = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AddMatMatElementsTilingData* tilingData =
        reinterpret_cast<AddMatMatElementsTilingData*>(tiling);
    tilingData->totalLength = numElements;
    tilingData->tileLength = numElements;
    tilingData->blockNum = blockDim;
    tilingData->blockLength = numElements;
    tilingData->lastBlockLength = numElements;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_mat_mat_elements, blockDim, c, a, b, beta, alpha, cOut, workspace, tiling);

    AscendC::GmFree(c);
    AscendC::GmFree(a);
    AscendC::GmFree(b);
    AscendC::GmFree(beta);
    AscendC::GmFree(alpha);
    AscendC::GmFree(cOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AddMatMatElementsKernelTest, test_fp32_large)
{
    constexpr size_t numElements = 8192;
    size_t dataSize = numElements * sizeof(float);
    size_t scalarSize = sizeof(float);
    size_t tilingSize = sizeof(AddMatMatElementsTilingData);
    uint32_t blockDim = 1;

    uint8_t* c = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* a = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* b = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(scalarSize);
    uint8_t* alpha = (uint8_t*)AscendC::GmAlloc(scalarSize);
    uint8_t* cOut = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AddMatMatElementsTilingData* tilingData =
        reinterpret_cast<AddMatMatElementsTilingData*>(tiling);
    tilingData->totalLength = numElements;
    tilingData->tileLength = 1024;
    tilingData->blockNum = blockDim;
    tilingData->blockLength = numElements;
    tilingData->lastBlockLength = numElements;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_mat_mat_elements, blockDim, c, a, b, beta, alpha, cOut, workspace, tiling);

    AscendC::GmFree(c);
    AscendC::GmFree(a);
    AscendC::GmFree(b);
    AscendC::GmFree(beta);
    AscendC::GmFree(alpha);
    AscendC::GmFree(cOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
