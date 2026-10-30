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
#include <vector>
#include "gtest/gtest.h"
#include "test_circular_pad.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void circular_pad(
    GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class circualr_pad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "circualr_pad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "circualr_pad_test TearDown\n" << endl;
    }
};

TEST_F(circualr_pad_test, test_case_fp32_smallshape_2d)
{
    uint32_t x_shape = 1 * 1 * 3 * 3;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 5 * 5;
    // inputs
    size_t x_size = x_shape * sizeof(float);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 3;
    tilingData->inputW = 3;
    tilingData->outputH = 5;
    tilingData->outputW = 5;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 0;
    tilingData->outputL = 0;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 72;

    ICPU_SET_TILING_KEY(241);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_fp32_bigshape_2d)
{
    uint32_t x_shape = 1 * 1 * 300 * 300;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 500 * 500;
    // inputs
    size_t x_size = x_shape * sizeof(float);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(62400 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 300;
    tilingData->inputW = 300;
    tilingData->outputH = 500;
    tilingData->outputW = 500;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 1;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 62400;

    ICPU_SET_TILING_KEY(242);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_fp32_smallshape_3d)
{
    uint32_t x_shape = 1 * 1 * 3 * 3;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 3 * 5 * 5;
    // inputs
    size_t x_size = x_shape * sizeof(float);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 3;
    tilingData->inputW = 3;
    tilingData->outputH = 5;
    tilingData->outputW = 5;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 1;
    tilingData->outputL = 3;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 72;

    ICPU_SET_TILING_KEY(341);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_fp32_bigshape_3d)
{
    uint32_t x_shape = 1 * 1 * 300 * 300;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 3 * 500 * 500;
    // inputs
    size_t x_size = x_shape * sizeof(float);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(62400 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 300;
    tilingData->inputW = 300;
    tilingData->outputH = 500;
    tilingData->outputW = 500;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 1;
    tilingData->outputL = 3;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 62400;

    ICPU_SET_TILING_KEY(342);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_fp16_smallshape_2d)
{
    uint32_t x_shape = 1 * 1 * 3 * 3;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 5 * 5;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 3;
    tilingData->inputW = 3;
    tilingData->outputH = 5;
    tilingData->outputW = 5;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 0;
    tilingData->outputL = 0;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 72;

    ICPU_SET_TILING_KEY(221);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_fp16_bigshape_2d)
{
    uint32_t x_shape = 1 * 1 * 300 * 300;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 500 * 500;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(62400 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 300;
    tilingData->inputW = 300;
    tilingData->outputH = 500;
    tilingData->outputW = 500;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 1;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 62400;

    ICPU_SET_TILING_KEY(222);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_fp16_smallshape_3d)
{
    uint32_t x_shape = 1 * 1 * 3 * 3;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 3 * 5 * 5;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 3;
    tilingData->inputW = 3;
    tilingData->outputH = 5;
    tilingData->outputW = 5;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 1;
    tilingData->outputL = 3;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 72;

    ICPU_SET_TILING_KEY(321);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_fp16_bigshape_3d)
{
    uint32_t x_shape = 1 * 1 * 300 * 300;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 3 * 500 * 500;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(62400 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 300;
    tilingData->inputW = 300;
    tilingData->outputH = 500;
    tilingData->outputW = 500;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 1;
    tilingData->outputL = 3;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 62400;

    ICPU_SET_TILING_KEY(322);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_bf16_smallshape_2d)
{
    uint32_t x_shape = 1 * 1 * 3 * 3;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 5 * 5;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 3;
    tilingData->inputW = 3;
    tilingData->outputH = 5;
    tilingData->outputW = 5;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 0;
    tilingData->outputL = 0;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 72;

    ICPU_SET_TILING_KEY(231);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_bf16_bigshape_2d)
{
    uint32_t x_shape = 1 * 1 * 300 * 300;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 500 * 500;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(62400 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 300;
    tilingData->inputW = 300;
    tilingData->outputH = 500;
    tilingData->outputW = 500;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 1;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 62400;

    ICPU_SET_TILING_KEY(232);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_bf16_smallshape_3d)
{
    uint32_t x_shape = 1 * 1 * 3 * 3;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 3 * 5 * 5;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 3;
    tilingData->inputW = 3;
    tilingData->outputH = 5;
    tilingData->outputW = 5;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 1;
    tilingData->outputL = 3;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 72;

    ICPU_SET_TILING_KEY(331);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_bf16_bigshape_3d)
{
    uint32_t x_shape = 1 * 1 * 300 * 300;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 3 * 500 * 500;
    // inputs
    size_t x_size = x_shape * sizeof(uint16_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(62400 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 300;
    tilingData->inputW = 300;
    tilingData->outputH = 500;
    tilingData->outputW = 500;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 1;
    tilingData->outputL = 3;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 62400;

    ICPU_SET_TILING_KEY(332);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_int8_smallshape_2d)
{
    uint32_t x_shape = 1 * 1 * 3 * 3;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 5 * 5;
    // inputs
    size_t x_size = x_shape * sizeof(uint8_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 3;
    tilingData->inputW = 3;
    tilingData->outputH = 5;
    tilingData->outputW = 5;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 0;
    tilingData->outputL = 0;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 72;

    ICPU_SET_TILING_KEY(211);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_int8_bigshape_2d)
{
    uint32_t x_shape = 1 * 1 * 300 * 300;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 500 * 500;
    // inputs
    size_t x_size = x_shape * sizeof(uint8_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(62400 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 300;
    tilingData->inputW = 300;
    tilingData->outputH = 500;
    tilingData->outputW = 500;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 1;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 62400;

    ICPU_SET_TILING_KEY(212);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_int8_smallshape_3d)
{
    uint32_t x_shape = 1 * 1 * 3 * 3;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 3 * 5 * 5;
    // inputs
    size_t x_size = x_shape * sizeof(uint8_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 3;
    tilingData->inputW = 3;
    tilingData->outputH = 5;
    tilingData->outputW = 5;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 1;
    tilingData->outputL = 3;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 72;

    ICPU_SET_TILING_KEY(311);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_int8_bigshape_3d)
{
    uint32_t x_shape = 1 * 1 * 300 * 300;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 3 * 500 * 500;
    // inputs
    size_t x_size = x_shape * sizeof(uint8_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(62400 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 300;
    tilingData->inputW = 300;
    tilingData->outputH = 500;
    tilingData->outputW = 500;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 1;
    tilingData->outputL = 3;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 62400;

    ICPU_SET_TILING_KEY(312);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_int32_smallshape_2d)
{
    uint32_t x_shape = 1 * 1 * 3 * 3;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 5 * 5;
    // inputs
    size_t x_size = x_shape * sizeof(uint32_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 3;
    tilingData->inputW = 3;
    tilingData->outputH = 5;
    tilingData->outputW = 5;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 0;
    tilingData->outputL = 0;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 72;

    ICPU_SET_TILING_KEY(251);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_int32_bigshape_2d)
{
    uint32_t x_shape = 1 * 1 * 300 * 300;
    uint32_t paddings_shape = 4;
    uint32_t y_shape = 1 * 1 * 500 * 500;
    // inputs
    size_t x_size = x_shape * sizeof(uint32_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(62400 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 300;
    tilingData->inputW = 300;
    tilingData->outputH = 500;
    tilingData->outputW = 500;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 0;
    tilingData->back = 0;
    tilingData->inputL = 1;
    tilingData->outputL = 1;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 62400;

    ICPU_SET_TILING_KEY(252);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_int32_smallshape_3d)
{
    uint32_t x_shape = 1 * 1 * 3 * 3;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 3 * 5 * 5;
    // inputs
    size_t x_size = x_shape * sizeof(uint32_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 3;
    tilingData->inputW = 3;
    tilingData->outputH = 5;
    tilingData->outputW = 5;
    tilingData->left = 1;
    tilingData->right = 1;
    tilingData->top = 1;
    tilingData->bottom = 1;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 1;
    tilingData->outputL = 3;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 72;

    ICPU_SET_TILING_KEY(351);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(circualr_pad_test, test_case_int32_bigshape_3d)
{
    uint32_t x_shape = 1 * 1 * 300 * 300;
    uint32_t paddings_shape = 6;
    uint32_t y_shape = 1 * 3 * 500 * 500;
    // inputs
    size_t x_size = x_shape * sizeof(uint32_t);
    size_t paddings_size = paddings_shape * sizeof(int64_t);
    size_t y_size = y_shape * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(CircularPadCommonTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* paddings = (uint8_t*)AscendC::GmAlloc(paddings_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(62400 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    CircularPadCommonTilingData* tilingData = reinterpret_cast<CircularPadCommonTilingData*>(tiling);
    tilingData->inputH = 300;
    tilingData->inputW = 300;
    tilingData->outputH = 500;
    tilingData->outputW = 500;
    tilingData->left = 100;
    tilingData->right = 100;
    tilingData->top = 100;
    tilingData->bottom = 100;
    tilingData->front = 1;
    tilingData->back = 1;
    tilingData->inputL = 1;
    tilingData->outputL = 3;
    tilingData->perCoreTaskNum = 0;
    tilingData->tailTaskNum = 1;
    tilingData->workspaceLen = 62400;

    ICPU_SET_TILING_KEY(352);
    ICPU_RUN_KF(circular_pad, numBlocks, x, paddings, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(paddings);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
