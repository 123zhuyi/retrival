/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "test_power.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>
#include <cmath>

using namespace std;

extern "C" __global__ __aicore__ void power(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class power_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "power_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "power_test TearDown\n" << endl;
    }
};

TEST_F(power_test, test_case_fp32_power_2_001)
{
    uint32_t x_shape = 256;
    uint32_t y_shape = 256;
    
    size_t x_size = x_shape * sizeof(float);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(PowerTestTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    PowerTestTilingData* tilingData = reinterpret_cast<PowerTestTilingData*>(tiling);
    tilingData->totalLength = x_shape;
    tilingData->blockLength = x_shape;
    tilingData->scale = 1.0f;
    tilingData->shift = 0.0f;
    tilingData->power = 2.0f;
    tilingData->negScalar = 1.0f;

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(power, numBlocks, x, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(power_test, test_case_fp32_power_3_002)
{
    uint32_t x_shape = 512;
    uint32_t y_shape = 512;
    
    size_t x_size = x_shape * sizeof(float);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(PowerTestTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    PowerTestTilingData* tilingData = reinterpret_cast<PowerTestTilingData*>(tiling);
    tilingData->totalLength = x_shape;
    tilingData->blockLength = x_shape;
    tilingData->scale = 1.0f;
    tilingData->shift = 0.0f;
    tilingData->power = 3.0f;
    tilingData->negScalar = 1.0f;

    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(power, numBlocks, x, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(power_test, test_case_fp32_linear_003)
{
    uint32_t x_shape = 1024;
    uint32_t y_shape = 1024;
    
    size_t x_size = x_shape * sizeof(float);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(PowerTestTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    PowerTestTilingData* tilingData = reinterpret_cast<PowerTestTilingData*>(tiling);
    tilingData->totalLength = x_shape;
    tilingData->blockLength = x_shape;
    tilingData->scale = 2.0f;
    tilingData->shift = 1.0f;
    tilingData->power = 1.0f;
    tilingData->negScalar = 1.0f;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(power, numBlocks, x, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(power_test, test_case_fp16_power_2_004)
{
    uint32_t x_shape = 512;
    uint32_t y_shape = 512;
    
    size_t x_size = x_shape * sizeof(half);
    size_t y_size = y_shape * sizeof(half);
    size_t tiling_data_size = sizeof(PowerTestTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    PowerTestTilingData* tilingData = reinterpret_cast<PowerTestTilingData*>(tiling);
    tilingData->totalLength = x_shape;
    tilingData->blockLength = x_shape;
    tilingData->scale = 1.0f;
    tilingData->shift = 0.0f;
    tilingData->power = 2.0f;
    tilingData->negScalar = 1.0f;

    ICPU_SET_TILING_KEY(100003);
    ICPU_RUN_KF(power, numBlocks, x, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(power_test, test_case_bf16_power_2_005)
{
    uint32_t x_shape = 512;
    uint32_t y_shape = 512;
    
    size_t x_size = x_shape * sizeof(bfloat16_t);
    size_t y_size = y_shape * sizeof(bfloat16_t);
    size_t tiling_data_size = sizeof(PowerTestTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    PowerTestTilingData* tilingData = reinterpret_cast<PowerTestTilingData*>(tiling);
    tilingData->totalLength = x_shape;
    tilingData->blockLength = x_shape;
    tilingData->scale = 1.0f;
    tilingData->shift = 0.0f;
    tilingData->power = 2.0f;
    tilingData->negScalar = 1.0f;

    ICPU_SET_TILING_KEY(200003);
    ICPU_RUN_KF(power, numBlocks, x, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(power_test, test_case_fp32_generic_pow_006)
{
    uint32_t x_shape = 1024;
    uint32_t y_shape = 1024;
    
    size_t x_size = x_shape * sizeof(float);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(PowerTestTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    PowerTestTilingData* tilingData = reinterpret_cast<PowerTestTilingData*>(tiling);
    tilingData->totalLength = x_shape;
    tilingData->blockLength = x_shape;
    tilingData->scale = 1.0f;
    tilingData->shift = 0.0f;
    tilingData->power = 2.5f;
    tilingData->negScalar = 1.0f;

    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF(power, numBlocks, x, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(power_test, test_case_fp32_negative_power_007)
{
    uint32_t x_shape = 256;
    uint32_t y_shape = 256;
    
    size_t x_size = x_shape * sizeof(float);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(PowerTestTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    PowerTestTilingData* tilingData = reinterpret_cast<PowerTestTilingData*>(tiling);
    tilingData->totalLength = x_shape;
    tilingData->blockLength = x_shape;
    tilingData->scale = 1.0f;
    tilingData->shift = 0.0f;
    tilingData->power = -2.0f;
    tilingData->negScalar = 1.0f;

    ICPU_SET_TILING_KEY(6);
    ICPU_RUN_KF(power, numBlocks, x, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(power_test, test_case_fp32_zero_power_008)
{
    uint32_t x_shape = 256;
    uint32_t y_shape = 256;
    
    size_t x_size = x_shape * sizeof(float);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(PowerTestTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    PowerTestTilingData* tilingData = reinterpret_cast<PowerTestTilingData*>(tiling);
    tilingData->totalLength = x_shape;
    tilingData->blockLength = x_shape;
    tilingData->scale = 1.0f;
    tilingData->shift = 0.0f;
    tilingData->power = 0.0f;
    tilingData->negScalar = 1.0f;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(power, numBlocks, x, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}