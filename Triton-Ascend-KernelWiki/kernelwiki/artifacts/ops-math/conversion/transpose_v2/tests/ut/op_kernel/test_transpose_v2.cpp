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
 * \file test_transpose_v2.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/transpose_v2_tiling.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void transpose_v2(
    GM_ADDR x, GM_ADDR perm, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class transpose_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "transpose_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "transpose_v2_test TearDown\n" << endl;
    }
};

TEST_F(transpose_v2_test, test_case_fp16_001)
{
    int64_t C = 1;
    int64_t H = 30;
    int64_t W = 68;

    size_t xSize = C * H * W * sizeof(half);
    size_t permSize = 3 * sizeof(int32_t);
    size_t ySize = C * W * H * sizeof(half);
    size_t tilingDataSize = sizeof(TransposeV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* perm = (uint8_t*)AscendC::GmAlloc(permSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t numBlocks = 1;

    TransposeV2TilingData* tilingData = reinterpret_cast<TransposeV2TilingData*>(tiling);
    tilingData->tasksPerCore = C;
    tilingData->tasksTail = 0;
    tilingData->inputH = H;
    tilingData->inputW = W;
    tilingData->inputH16Align = (H + 15) / 16;
    tilingData->inputWAlign = (W + 15) / 16;
    tilingData->hOnce = 128;
    tilingData->tasksOnceMax = 13;
    tilingData->repeatH = 2;
    tilingData->transLoop = 1;
    tilingData->doubleBuffer = 2;

    ICPU_SET_TILING_KEY(20);
    ICPU_RUN_KF(transpose_v2, numBlocks, x, perm, y, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(perm);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(transpose_v2_test, test_case_fp16_002)
{
    int64_t C = 1;
    int64_t H = 32;
    int64_t W = 68;

    size_t xSize = C * H * W * sizeof(half);
    size_t permSize = 3 * sizeof(int32_t);
    size_t ySize = C * W * H * sizeof(half);
    size_t tilingDataSize = sizeof(TransposeV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* perm = (uint8_t*)AscendC::GmAlloc(permSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t numBlocks = 1;

    TransposeV2TilingData* tilingData = reinterpret_cast<TransposeV2TilingData*>(tiling);
    tilingData->dim1Len = C;
    tilingData->dim2Len = H;
    tilingData->dim3Len = W;
    tilingData->dim3LenAlign = (W + 15) / 16;
    tilingData->tasksPerCore = C;
    tilingData->tailCore = 0;
    tilingData->dim1OnceMax = 13;
    tilingData->dim2OnceMax = 2;
    tilingData->subMode = 1;
    tilingData->doubleBuffer = 2;

    ICPU_SET_TILING_KEY(120);
    ICPU_RUN_KF(transpose_v2, numBlocks, x, perm, y, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(perm);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(transpose_v2_test, test_case_fp16_003)
{
    int64_t C = 1;
    int64_t H = 32;
    int64_t W = 64;

    size_t xSize = C * H * W * sizeof(half);
    size_t permSize = 3 * sizeof(int32_t);
    size_t ySize = C * W * H * sizeof(half);
    size_t tilingDataSize = sizeof(TransposeV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* perm = (uint8_t*)AscendC::GmAlloc(permSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t numBlocks = 1;

    TransposeV2TilingData* tilingData = reinterpret_cast<TransposeV2TilingData*>(tiling);
    tilingData->dim1Len = C;
    tilingData->dim2Len = H;
    tilingData->dim3Len = W;
    tilingData->dim3LenAlign = (W + 15) / 16;
    tilingData->tasksPerCore = C;
    tilingData->tailCore = 0;
    tilingData->dim1OnceMax = 13;
    tilingData->dim2OnceMax = 2;
    tilingData->subMode = 1;
    tilingData->doubleBuffer = 2;

    ICPU_SET_TILING_KEY(121);
    ICPU_RUN_KF(transpose_v2, numBlocks, x, perm, y, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(perm);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(transpose_v2_test, test_case_fp16_004)
{
    int64_t C = 1;
    int64_t H = 32;
    int64_t W = 64;

    size_t xSize = C * H * W * sizeof(half);
    size_t permSize = 3 * sizeof(int32_t);
    size_t ySize = C * W * H * sizeof(half);
    size_t tilingDataSize = sizeof(TransposeV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* perm = (uint8_t*)AscendC::GmAlloc(permSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t numBlocks = 1;

    TransposeV2TilingData* tilingData = reinterpret_cast<TransposeV2TilingData*>(tiling);
    tilingData->dim1Len = C;
    tilingData->dim2Len = H;
    tilingData->dim3Len = W;
    tilingData->dim3LenAlign = (W + 15) / 16;
    tilingData->tasksPerCore = C;
    tilingData->tailCore = 0;
    tilingData->dim1OnceMax = 13;
    tilingData->dim2OnceMax = 2;
    tilingData->subMode = 1;
    tilingData->doubleBuffer = 2;

    ICPU_SET_TILING_KEY(221);
    ICPU_RUN_KF(transpose_v2, numBlocks, x, perm, y, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(perm);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
