/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <cstdint>
#include <iostream>
#include <string>

#include <vector>

#include "data_utils.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_data_def.h"

extern "C" __global__ __aicore__ void pad_v4_grad(
    GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
class pad_v4_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "pad_v4_grad SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "pad_v4_grad TearDown\n" << std::endl;
    }
};

TEST_F(pad_v4_grad_test, test_bfloat16_case1)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 64;
    uint32_t W = 64;
    uint32_t hPad1 = 0;
    uint32_t hPad2 = 0;
    uint32_t wPad1 = 1;
    uint32_t wPad2 = 1;
    uint32_t numBlocks = 8;
    uint32_t ubSize = 192 * 1024 - 11 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV4GradTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * sizeof(float);
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - wPad1 - wPad2) * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV4GradTilingData* tilingData = reinterpret_cast<PadV4GradTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 64;
    tilingData->width = 64;
    tilingData->alignHeight = 64;
    tilingData->alignWidth = 64;
    tilingData->outHeight = 64;
    tilingData->outWidth = 62;
    tilingData->alignOutHeight = 64;
    tilingData->alignOutWidth = 64;
    tilingData->hPad1 = 0;
    tilingData->hPad2 = 0;
    tilingData->wPad1 = 1;
    tilingData->wPad2 = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 112;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3000;
    tilingData->workspacePerCore = 0;
    tilingData->wPadCopyCount = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(3000);
    ICPU_RUN_KF(pad_v4_grad, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v4_grad_test, test_bfloat16_case2)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 16;
    uint32_t W = 530;
    uint32_t hPad1 = 0;
    uint32_t hPad2 = 0;
    uint32_t wPad1 = 1;
    uint32_t wPad2 = 1;
    uint32_t numBlocks = 48;
    uint32_t ubSize = 192 * 1024 - 11 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV4GradTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * 2;
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - wPad1 - wPad2) * 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV4GradTilingData* tilingData = reinterpret_cast<PadV4GradTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 16;
    tilingData->width = 530;
    tilingData->alignHeight = 16;
    tilingData->alignWidth = 544;
    tilingData->outHeight = 16;
    tilingData->outWidth = 528;
    tilingData->alignOutHeight = 16;
    tilingData->alignOutWidth = 528;
    tilingData->hPad1 = 0;
    tilingData->hPad2 = 0;
    tilingData->wPad1 = 1;
    tilingData->wPad2 = 1;
    tilingData->blockNum = 16;
    tilingData->ubFactorElement = 15360;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3101;
    tilingData->workspacePerCore = 256;
    tilingData->wPadCopyCount = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(3101);
    ICPU_RUN_KF(pad_v4_grad, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v4_grad_test, test_bfloat16_case3)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 200;
    uint32_t W = 200;
    uint32_t hPad1 = 1;
    uint32_t hPad2 = 1;
    uint32_t wPad1 = 0;
    uint32_t wPad2 = 0;
    uint32_t numBlocks = 48;
    uint32_t ubSize = 192 * 1024 - 11 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV4GradTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * 2;
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - wPad1 - wPad2) * 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV4GradTilingData* tilingData = reinterpret_cast<PadV4GradTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 200;
    tilingData->width = 200;
    tilingData->alignHeight = 208;
    tilingData->alignWidth = 208;
    tilingData->outHeight = 198;
    tilingData->outWidth = 200;
    tilingData->alignOutHeight = 208;
    tilingData->alignOutWidth = 208;
    tilingData->hPad1 = 1;
    tilingData->hPad2 = 1;
    tilingData->wPad1 = 0;
    tilingData->wPad2 = 0;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 7680;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3110;
    tilingData->workspacePerCore = 0;
    tilingData->wPadCopyCount = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(pad_v4_grad, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v4_grad_test, test_bfloat16_case4)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 150;
    uint32_t W = 30;
    uint32_t hPad1 = 3;
    uint32_t hPad2 = 5;
    uint32_t wPad1 = 3;
    uint32_t wPad2 = 1;
    uint32_t numBlocks = 48;
    uint32_t ubSize = 192 * 1024 - 11 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV4GradTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * 2;
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - wPad1 - wPad2) * 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV4GradTilingData* tilingData = reinterpret_cast<PadV4GradTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 150;
    tilingData->width = 30;
    tilingData->alignHeight = 160;
    tilingData->alignWidth = 32;
    tilingData->outHeight = 142;
    tilingData->outWidth = 26;
    tilingData->alignOutHeight = 144;
    tilingData->alignOutWidth = 32;
    tilingData->hPad1 = 3;
    tilingData->hPad2 = 5;
    tilingData->wPad1 = 3;
    tilingData->wPad2 = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 112;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3010;
    tilingData->workspacePerCore = 4096;
    tilingData->wPadCopyCount = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(pad_v4_grad, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(pad_v4_grad_test, test_bfloat16_case5)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t N = 1;
    uint32_t C = 1;
    uint32_t H = 16;
    uint32_t W = 530;
    uint32_t hPad1 = 1;
    uint32_t hPad2 = 1;
    uint32_t wPad1 = 1;
    uint32_t wPad2 = 1;
    uint32_t numBlocks = 48;
    uint32_t ubSize = 192 * 1024 - 11 * 1024;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(PadV4GradTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    size_t x_size = N * C * H * W * 2;
    size_t padding_size = 4 * sizeof(int32_t);
    size_t dx_size = N * C * H * (W - wPad1 - wPad2) * 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(padding_size);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dx_size);

    PadV4GradTilingData* tilingData = reinterpret_cast<PadV4GradTilingData*>(tiling);
    tilingData->batch = 1;
    tilingData->channel = 1;
    tilingData->height = 16;
    tilingData->width = 530;
    tilingData->alignHeight = 16;
    tilingData->alignWidth = 544;
    tilingData->outHeight = 14;
    tilingData->outWidth = 528;
    tilingData->alignOutHeight = 16;
    tilingData->alignOutWidth = 528;
    tilingData->hPad1 = 1;
    tilingData->hPad2 = 1;
    tilingData->wPad1 = 1;
    tilingData->wPad2 = 1;
    tilingData->blockNum = 1;
    tilingData->ubFactorElement = 240;
    tilingData->ncPerCore = 1;
    tilingData->tailNC = 0;
    tilingData->tilingKey = 3100;
    tilingData->workspacePerCore = 69632;
    tilingData->wPadCopyCount = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    ICPU_RUN_KF(pad_v4_grad, tilingData->blockNum, x, padding, dx, workspace, tiling); // use this macro for cpu debug
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)padding);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}