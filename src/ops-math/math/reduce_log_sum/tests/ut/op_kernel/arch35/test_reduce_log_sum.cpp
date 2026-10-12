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
 * \file test_reduce_log_sum.cpp
 * \brief reduce_log_sum opkernel unit test
 */

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "reduce_log_sum_apt.cpp"

using namespace std;

class ReduceLogSumKernel : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ReduceLogSumKernel SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ReduceLogSumKernel TearDown\n" << endl;
    }
};

// float32, 4x64 reduce axis=1 -> output 4
TEST_F(ReduceLogSumKernel, test_case_float32_4x64)
{
    uint64_t tilingKey = 0;
    uint32_t numBlocks = 4;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t xSize = 4 * 64 * sizeof(float);
    size_t ySize = 4 * sizeof(float);
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* axes = (uint8_t*)AscendC::GmAlloc(sizeof(int32_t));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ReduceOpTilingData));

    Ops::Base::ReduceOpTilingData* tilingData = reinterpret_cast<Ops::Base::ReduceOpTilingData*>(tiling);
    tilingData->factorACntPerCore = 1;
    tilingData->factorATotalCnt = 4;
    tilingData->ubFactorA = 1;
    tilingData->factorRCntPerCore = 1;
    tilingData->factorRTotalCnt = 1;
    tilingData->ubFactorR = 1;
    tilingData->groupR = 1;
    tilingData->outSize = 4;
    tilingData->basicBlock = 51200;
    tilingData->coreNum = 64;
    tilingData->meanVar = 0.015625f;
    tilingData->shape[0] = 4;
    tilingData->shape[1] = 64;
    tilingData->stride[0] = 64;
    tilingData->stride[1] = 1;
    tilingData->dstStride[0] = 1;
    tilingData->dstStride[1] = 1;

    ICPU_SET_TILING_KEY(tilingKey);
    auto reduce_log_sum_func = [](GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        reduce_log_sum<true, 0, 1, 1>(x, axes, y, workspace, tiling);
    };
    ICPU_RUN_KF(reduce_log_sum_func, numBlocks, x, axes, y, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)axes);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// float16, 2x4 reduce axis=0 -> output 4 (keep_dims)
TEST_F(ReduceLogSumKernel, test_case_float16_2x4)
{
    uint64_t tilingKey = 0;
    uint32_t numBlocks = 2;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t xSize = 2 * 4 * sizeof(uint16_t);
    size_t ySize = 1 * 4 * sizeof(uint16_t);
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* axes = (uint8_t*)AscendC::GmAlloc(sizeof(int32_t));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ReduceOpTilingData));

    Ops::Base::ReduceOpTilingData* tilingData = reinterpret_cast<Ops::Base::ReduceOpTilingData*>(tiling);
    tilingData->factorACntPerCore = 1;
    tilingData->factorATotalCnt = 1;
    tilingData->ubFactorA = 1;
    tilingData->factorRCntPerCore = 1;
    tilingData->factorRTotalCnt = 1;
    tilingData->ubFactorR = 1;
    tilingData->groupR = 1;
    tilingData->outSize = 4;
    tilingData->basicBlock = 51200;
    tilingData->coreNum = 64;
    tilingData->meanVar = 0.015625f;
    tilingData->shape[0] = 2;
    tilingData->shape[1] = 4;
    tilingData->stride[0] = 4;
    tilingData->stride[1] = 1;
    tilingData->dstStride[0] = 1;
    tilingData->dstStride[1] = 1;

    ICPU_SET_TILING_KEY(tilingKey);
    auto reduce_log_sum_func = [](GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        reduce_log_sum<true, 0, 1, 1>(x, axes, y, workspace, tiling);
    };
    ICPU_RUN_KF(reduce_log_sum_func, numBlocks, x, axes, y, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)axes);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// float32, 16x256 reduce axis=0 -> output 256
TEST_F(ReduceLogSumKernel, test_case_float32_16x256)
{
    uint64_t tilingKey = 0;
    uint32_t numBlocks = 16;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t xSize = 16 * 256 * sizeof(float);
    size_t ySize = 1 * 256 * sizeof(float);
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* axes = (uint8_t*)AscendC::GmAlloc(sizeof(int32_t));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ReduceOpTilingData));

    Ops::Base::ReduceOpTilingData* tilingData = reinterpret_cast<Ops::Base::ReduceOpTilingData*>(tiling);
    tilingData->factorACntPerCore = 1;
    tilingData->factorATotalCnt = 1;
    tilingData->ubFactorA = 1;
    tilingData->factorRCntPerCore = 1;
    tilingData->factorRTotalCnt = 1;
    tilingData->ubFactorR = 1;
    tilingData->groupR = 1;
    tilingData->outSize = 256;
    tilingData->basicBlock = 51200;
    tilingData->coreNum = 64;
    tilingData->meanVar = 0.015625f;
    tilingData->shape[0] = 16;
    tilingData->shape[1] = 256;
    tilingData->stride[0] = 256;
    tilingData->stride[1] = 1;
    tilingData->dstStride[0] = 1;
    tilingData->dstStride[1] = 1;

    ICPU_SET_TILING_KEY(tilingKey);
    auto reduce_log_sum_func = [](GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        reduce_log_sum<true, 0, 1, 1>(x, axes, y, workspace, tiling);
    };
    ICPU_RUN_KF(reduce_log_sum_func, numBlocks, x, axes, y, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)axes);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}
