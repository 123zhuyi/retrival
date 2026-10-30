/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define __aicore__
#include "concat_dv2_tiling_data.h"
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>


#include <cstdint>

using namespace std;


extern "C" __global__ __aicore__ void concat_dv2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class concat_dv2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "concat_dv2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "concat_dv2_test TearDown\n" << endl;
    }
};

TEST_F(concat_dv2_test, test_case_fp32_smallshape_2d)
{
    uint32_t x1_shape = 1 * 32;
    uint32_t y_shape = 1 * 32;
    // inputs
    size_t x1_size = x1_shape * sizeof(float);
    size_t y_size = y_shape * sizeof(float);
    size_t tiling_data_size = sizeof(ConcatDV2TilingDataUT);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(72 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 1;

    ConcatDV2TilingDataUT* tilingData = reinterpret_cast<ConcatDV2TilingDataUT*>(tiling);
    tilingData->elePerLoop = 32;
    tilingData->elePercore = 32;
    tilingData->ubLoop = 1;
    tilingData->eleTailCore = 32;
    tilingData->ubLoopTail = 1;
    tilingData->sameDimSize = 32;
    tilingData->endTensorIdx[0] = 0;
    tilingData->endTensorOffset[0] = 0;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(concat_dv2, numBlocks, x1, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x1);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}