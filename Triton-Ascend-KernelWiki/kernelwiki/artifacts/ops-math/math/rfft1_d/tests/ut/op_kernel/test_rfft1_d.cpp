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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/rfft1_d_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void rfft1_d(GM_ADDR x, GM_ADDR dft, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class rfft1d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rfft1d_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rfft1d_test TearDown\n" << endl;
    }
};

TEST_F(rfft1d_test, test_case_DFT)
{
    uint32_t numBlocks = 24;
    uint32_t fftLength = 64;
    size_t inputByteSize = numBlocks * fftLength * sizeof(float);
    size_t dftByteSize = (64 * 72 * 2) * sizeof(float);
    size_t outputByteSize = numBlocks * ((fftLength / 2) + 1) * 2 * sizeof(float);
    size_t tilingDataSize = sizeof(Rfft1DTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* dft = (uint8_t*)AscendC::GmAlloc(dftByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    Rfft1DTilingData* tilingDatafromBin = reinterpret_cast<Rfft1DTilingData*>(tiling);

    tilingDatafromBin->length = fftLength;
    tilingDatafromBin->batchesPerCore = 0;
    tilingDatafromBin->leftOverBatches = 1;
    tilingDatafromBin->normal = 1;
    tilingDatafromBin->factors[0] = 1;
    tilingDatafromBin->factors[1] = 1;
    tilingDatafromBin->factors[2] = 1;
    tilingDatafromBin->dftRealOverallSize = 64 * 72;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(rfft1_d, numBlocks, x, dft, y, workspace, (uint8_t*)(tilingDatafromBin));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    AscendC::GmFree(x);
    AscendC::GmFree(dft);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}