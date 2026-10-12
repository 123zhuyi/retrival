/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_unfold_grad.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/unfold_grad_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void unfold_grad(
    GM_ADDR grad_out, GM_ADDR input_sizes, GM_ADDR grad_in, GM_ADDR workspace, GM_ADDR tiling);

class unfold_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "unfold_grad_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "unfold_grad_test TearDown\n" << std::endl;
    }
};

TEST_F(unfold_grad_test, test_case_float32_outputshape_8_2_dim_0_size_3_step_2)
{
    size_t gradOutSize = 3 * 2 * 8 * sizeof(float);
    size_t inputSizeSize = 1 * 3 * sizeof(int64_t);
    size_t outputSize = 8 * 4 * sizeof(float);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 1;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 0;
    tilingData->size = 3;
    tilingData->step = 2;
    tilingData->loop = 0;
    tilingData->tail = 2;
    tilingData->sizeLevel = 0;
    tilingData->inputSizeLastDim = 2;
    tilingData->gradOutSizeDim = 3;
    tilingData->typeSizeT1 = 4;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 1;
    tilingData->batchNumPerCore = 1;
    tilingData->batchNumTailCore = 1;
    tilingData->useCoreNum = numBlocks;
    tilingData->ubSizeT1 = 0;
    tilingData->ubSizeT2 = 97280;
    tilingData->outputNumPerCore = 16;
    tilingData->inputNumPerCore = 18;
    tilingData->iterationNumPerCore = 3;
    tilingData->handleNUMOnceIterationPerCore = 2;
    tilingData->tasksOnceMaxPerCore = 2944;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 8;
    tilingData->lowestCommonMultiple = 48;
    tilingData->colOnceMaxPerUB = 184;
    tilingData->tailColLength = 1;

    ICPU_SET_TILING_KEY(212);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, numBlocks, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(unfold_grad_test, test_case_float32_outputshape_6_3_dim_1_size_2_step_1)
{
    size_t gradOutSize = 6 * 2 * 4 * sizeof(float);
    size_t inputSizeSize = 1 * 3 * sizeof(int64_t);
    size_t outputSize = 6 * 10 * sizeof(float);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 6;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 1;
    tilingData->size = 2;
    tilingData->step = 1;
    tilingData->loop = 0;
    tilingData->tail = 4;
    tilingData->sizeLevel = 0;
    tilingData->inputSizeLastDim = 3;
    tilingData->gradOutSizeDim = 2;
    tilingData->typeSizeT1 = 4;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 6;
    tilingData->batchNumPerCore = 1;
    tilingData->batchNumTailCore = 1;
    tilingData->useCoreNum = numBlocks;
    tilingData->ubSizeT1 = 0;
    tilingData->ubSizeT2 = 97280;
    tilingData->outputNumPerCore = 3;
    tilingData->inputNumPerCore = 4;
    tilingData->iterationNumPerCore = 0;
    tilingData->handleNUMOnceIterationPerCore = 4;
    tilingData->tasksOnceMaxPerCore = 24320;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 0;
    tilingData->lowestCommonMultiple = 1;
    tilingData->colOnceMaxPerUB = 0;
    tilingData->tailColLength = 8;

    ICPU_SET_TILING_KEY(211);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, numBlocks, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(unfold_grad_test, test_case_float16_outputshape_6_3_dim_1_size_2_step_1)
{
    size_t gradOutSize = 6 * 2 * 2 * sizeof(half);
    size_t inputSizeSize = 1 * 3 * sizeof(int64_t);
    size_t outputSize = 6 * 3 * sizeof(half);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 1;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 1;
    tilingData->size = 2;
    tilingData->step = 1;
    tilingData->loop = 0;
    tilingData->tail = 4;
    tilingData->sizeLevel = 0;
    tilingData->inputSizeLastDim = 3;
    tilingData->gradOutSizeDim = 2;
    tilingData->typeSizeT1 = 2;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 6;
    tilingData->batchNumPerCore = 6;
    tilingData->batchNumTailCore = 6;
    tilingData->useCoreNum = numBlocks;
    tilingData->ubSizeT1 = 38912;
    tilingData->ubSizeT2 = 77824;
    tilingData->outputNumPerCore = 3;
    tilingData->inputNumPerCore = 4;
    tilingData->iterationNumPerCore = 0;
    tilingData->handleNUMOnceIterationPerCore = 4;
    tilingData->tasksOnceMaxPerCore = 19456;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 0;
    tilingData->lowestCommonMultiple = 1;
    tilingData->colOnceMaxPerUB = 0;
    tilingData->tailColLength = 8;

    ICPU_SET_TILING_KEY(221);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, numBlocks, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(unfold_grad_test, test_case_float16_outputshape_8_2_dim_0_size_3_step_2)
{
    size_t gradOutSize = 3 * 2 * 3 * sizeof(half);
    size_t inputSizeSize = 1 * 3 * sizeof(int64_t);
    size_t outputSize = 8 * 2 * sizeof(half);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 1;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 0;
    tilingData->size = 3;
    tilingData->step = 2;
    tilingData->loop = 0;
    tilingData->tail = 2;
    tilingData->sizeLevel = 0;
    tilingData->inputSizeLastDim = 2;
    tilingData->gradOutSizeDim = 3;
    tilingData->typeSizeT1 = 2;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 1;
    tilingData->batchNumPerCore = 1;
    tilingData->batchNumTailCore = 1;
    tilingData->useCoreNum = numBlocks;
    tilingData->ubSizeT1 = 38912;
    tilingData->ubSizeT2 = 77824;
    tilingData->outputNumPerCore = 16;
    tilingData->inputNumPerCore = 18;
    tilingData->iterationNumPerCore = 3;
    tilingData->handleNUMOnceIterationPerCore = 2;
    tilingData->tasksOnceMaxPerCore = 2432;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 8;
    tilingData->lowestCommonMultiple = 48;
    tilingData->colOnceMaxPerUB = 152;
    tilingData->tailColLength = 1;

    ICPU_SET_TILING_KEY(222);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, numBlocks, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(unfold_grad_test, test_case_float32_outputshape_6_3_dim_1_size_1_step_2)
{
    size_t gradOutSize = 6 * 2 * 1 * sizeof(float);
    size_t inputSizeSize = 2 * sizeof(int64_t);
    size_t outputSize = 6 * 3 * sizeof(float);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 1;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 1;
    tilingData->size = 1;
    tilingData->step = 2;
    tilingData->loop = 0;
    tilingData->tail = 2;
    tilingData->sizeLevel = 0;
    tilingData->inputSizeLastDim = 3;
    tilingData->gradOutSizeDim = 2;
    tilingData->typeSizeT1 = 4;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 6;
    tilingData->batchNumPerCore = 6;
    tilingData->batchNumTailCore = 6;
    tilingData->useCoreNum = numBlocks;
    tilingData->ubSizeT1 = 0;
    tilingData->ubSizeT2 = 97280;
    tilingData->outputNumPerCore = 3;
    tilingData->inputNumPerCore = 2;
    tilingData->iterationNumPerCore = 0;
    tilingData->handleNUMOnceIterationPerCore = 2;
    tilingData->tasksOnceMaxPerCore = 12160;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 0;
    tilingData->lowestCommonMultiple = 1;
    tilingData->colOnceMaxPerUB = 0;
    tilingData->tailColLength = 2;

    ICPU_SET_TILING_KEY(311);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, numBlocks, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(unfold_grad_test, test_case_float32_outputshape_8_2_dim_0_size_2_step_3)
{
    size_t gradOutSize = 3 * 2 * 2 * sizeof(float);
    size_t inputSizeSize = 2 * sizeof(int64_t);
    size_t outputSize = 8 * 2 * sizeof(float);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 1;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 0;
    tilingData->size = 2;
    tilingData->step = 3;
    tilingData->loop = 0;
    tilingData->tail = 2;
    tilingData->sizeLevel = 0;
    tilingData->inputSizeLastDim = 2;
    tilingData->gradOutSizeDim = 3;
    tilingData->typeSizeT1 = 4;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 1;
    tilingData->batchNumPerCore = 1;
    tilingData->batchNumTailCore = 1;
    tilingData->useCoreNum = numBlocks;
    tilingData->ubSizeT1 = 0;
    tilingData->ubSizeT2 = 97280;
    tilingData->outputNumPerCore = 16;
    tilingData->inputNumPerCore = 12;
    tilingData->iterationNumPerCore = 3;
    tilingData->handleNUMOnceIterationPerCore = 2;
    tilingData->tasksOnceMaxPerCore = 3008;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 8;
    tilingData->lowestCommonMultiple = 16;
    tilingData->colOnceMaxPerUB = 376;
    tilingData->tailColLength = 1;

    ICPU_SET_TILING_KEY(312);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, numBlocks, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(unfold_grad_test, test_case_float16_outputshape_6_3_dim_1_size_1_step_2)
{
    size_t gradOutSize = 6 * 2 * 1 * sizeof(half);
    size_t inputSizeSize = 2 * sizeof(int64_t);
    size_t outputSize = 6 * 3 * sizeof(half);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 1;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 1;
    tilingData->size = 1;
    tilingData->step = 2;
    tilingData->loop = 0;
    tilingData->tail = 2;
    tilingData->sizeLevel = 0;
    tilingData->inputSizeLastDim = 3;
    tilingData->gradOutSizeDim = 2;
    tilingData->typeSizeT1 = 2;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 6;
    tilingData->batchNumPerCore = 6;
    tilingData->batchNumTailCore = 6;
    tilingData->useCoreNum = numBlocks;
    tilingData->ubSizeT1 = 38912;
    tilingData->ubSizeT2 = 77824;
    tilingData->outputNumPerCore = 3;
    tilingData->inputNumPerCore = 2;
    tilingData->iterationNumPerCore = 0;
    tilingData->handleNUMOnceIterationPerCore = 2;
    tilingData->tasksOnceMaxPerCore = 9728;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 0;
    tilingData->lowestCommonMultiple = 1;
    tilingData->colOnceMaxPerUB = 0;
    tilingData->tailColLength = 2;

    ICPU_SET_TILING_KEY(321);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, numBlocks, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(unfold_grad_test, test_case_float16_outputshape_8_2_dim_0_size_2_step_3)
{
    size_t gradOutSize = 3 * 2 * 2 * sizeof(half);
    size_t inputSizeSize = 2 * sizeof(int64_t);
    size_t outputSize = 8 * 2 * sizeof(half);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 1;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 0;
    tilingData->size = 2;
    tilingData->step = 3;
    tilingData->loop = 0;
    tilingData->tail = 2;
    tilingData->sizeLevel = 0;
    tilingData->inputSizeLastDim = 2;
    tilingData->gradOutSizeDim = 3;
    tilingData->typeSizeT1 = 2;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 1;
    tilingData->batchNumPerCore = 1;
    tilingData->batchNumTailCore = 1;
    tilingData->useCoreNum = numBlocks;
    tilingData->ubSizeT1 = 38912;
    tilingData->ubSizeT2 = 77824;
    tilingData->outputNumPerCore = 16;
    tilingData->inputNumPerCore = 12;
    tilingData->iterationNumPerCore = 3;
    tilingData->handleNUMOnceIterationPerCore = 2;
    tilingData->tasksOnceMaxPerCore = 2432;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 8;
    tilingData->lowestCommonMultiple = 16;
    tilingData->colOnceMaxPerUB = 304;
    tilingData->tailColLength = 1;

    ICPU_SET_TILING_KEY(322);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, numBlocks, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(unfold_grad_test, test_case_float16_outputshape_7_8192_dim_1_size_3958_step_3463) 
{
    size_t gradOutSize = 7 * 2 * 3958 * sizeof(half);
    size_t inputSizeSize = 2 * sizeof(int64_t);
    size_t outputSize = 7 * 8192 * sizeof(half);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 1;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 1;
    tilingData->size = 3958;
    tilingData->step = 3463;
    tilingData->loop = 0;
    tilingData->tail = 7916;
    tilingData->sizeLevel = 1;
    tilingData->inputSizeLastDim = 8192;
    tilingData->gradOutSizeDim = 2;
    tilingData->typeSizeT1 = 2;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 7;
    tilingData->batchNumPerCore = 1;
    tilingData->batchNumTailCore = 1;
    tilingData->useCoreNum = numBlocks;
    tilingData->ubSizeT1 = 87296;
    tilingData->ubSizeT2 = 174592;
    tilingData->outputNumPerCore = 8192;
    tilingData->inputNumPerCore = 7916;
    tilingData->iterationNumPerCore = 0;
    tilingData->handleNUMOnceIterationPerCore = 7916;
    tilingData->tasksOnceMaxPerCore = 43538;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 0;
    tilingData->lowestCommonMultiple = 1;
    tilingData->colOnceMaxPerUB = 0;
    tilingData->tailColLength = 31331528;

    ICPU_SET_TILING_KEY(1221);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, numBlocks, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(unfold_grad_test, test_case_float32_outputshape_7_8192_dim_1_size_3958_step_3463) {
    size_t gradOutSize = 7 * 2 * 3958 * sizeof(float);
    size_t inputSizeSize = 2 * sizeof(int64_t);
    size_t outputSize = 7 * 8192 * sizeof(float);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 1;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 1;
    tilingData->size = 3958;
    tilingData->step = 3463;
    tilingData->loop = 0;
    tilingData->tail = 7916;
    tilingData->sizeLevel = 1;
    tilingData->inputSizeLastDim = 8192;
    tilingData->gradOutSizeDim = 2;
    tilingData->typeSizeT1 = 4;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 7;
    tilingData->batchNumPerCore = 1;
    tilingData->batchNumTailCore = 1;
    tilingData->useCoreNum = numBlocks;
    tilingData->ubSizeT1 = 0;
    tilingData->ubSizeT2 = 261888;
    tilingData->outputNumPerCore = 8192;
    tilingData->inputNumPerCore = 7916;
    tilingData->iterationNumPerCore = 0;
    tilingData->handleNUMOnceIterationPerCore = 7916;
    tilingData->tasksOnceMaxPerCore = 63328;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 0;
    tilingData->lowestCommonMultiple = 1;
    tilingData->colOnceMaxPerUB = 0;
    tilingData->tailColLength = 31331528;

    ICPU_SET_TILING_KEY(1211);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, numBlocks, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

}

TEST_F(unfold_grad_test, test_case_float16_outputshape_8_32768_dim_1_size_3924_step_25617) {
    size_t gradOutSize = 8 * 2 * 3924 * sizeof(half);
    size_t inputSizeSize = 2 * sizeof(int64_t);
    size_t outputSize = 8 * 32768 * sizeof(half);
    size_t tilingSize = sizeof(UnfoldGradTilingData);

    uint8_t* gradOut = (uint8_t*)AscendC::GmAlloc(gradOutSize);
    uint8_t* inputSize = (uint8_t*)AscendC::GmAlloc(inputSizeSize);

    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t numBlocks = 1;

    UnfoldGradTilingData* tilingData = reinterpret_cast<UnfoldGradTilingData*>(tiling);
    tilingData->dim = 1;
    tilingData->size = 3958;
    tilingData->step = 3463;
    tilingData->loop = 0;
    tilingData->tail = 7916;
    tilingData->sizeLevel = 1;
    tilingData->inputSizeLastDim = 8192;
    tilingData->gradOutSizeDim = 2;
    tilingData->typeSizeT1 = 2;
    tilingData->typeSizeT2 = 4;
    tilingData->width = 8;

    tilingData->batchNum = 8;
    tilingData->batchNumPerCore = 1;
    tilingData->batchNumTailCore = 1;
    tilingData->useCoreNum = numBlocks;
    tilingData->ubSizeT1 = 87296;
    tilingData->ubSizeT2 = 174592;
    tilingData->outputNumPerCore = 32768;
    tilingData->inputNumPerCore = 7848;
    tilingData->iterationNumPerCore = 0;
    tilingData->handleNUMOnceIterationPerCore = 7848;
    tilingData->tasksOnceMaxPerCore = 43164;
    tilingData->inputSizeLength = 2;
    tilingData->rowAvailableLengthSrc = 0;
    tilingData->lowestCommonMultiple = 1;
    tilingData->colOnceMaxPerUB = 0;
    tilingData->tailColLength = 201042216;

    ICPU_SET_TILING_KEY(1321);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(unfold_grad, numBlocks, gradOut, inputSize, output, workspace, tiling);

    AscendC::GmFree(gradOut);
    AscendC::GmFree(inputSize);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

}
