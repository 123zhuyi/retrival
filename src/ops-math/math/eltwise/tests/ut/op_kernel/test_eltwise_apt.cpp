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

#include "../../../op_kernel/arch35/eltwise_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void eltwise(
    GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class EltwiseKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "EltwiseKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "EltwiseKernelTest TearDown" << endl; }
};

TEST_F(EltwiseKernelTest, test_fp32_empty_tensor)
{
    // Empty tensor path: totalNum=0, blockFactor=0, inputNum=0 -> kernel early-exit.
    // This validates the early-return branch in Init() without needing to fake
    // the AscendC::ListTensorDesc folded-input format.
    size_t tilingSize = sizeof(EltwiseTilingData);
    uint32_t blockDim = 1;
    constexpr size_t kStubInputsBufBytes = 1024;
    constexpr size_t kStubOutputBytes = 1024;

    uint8_t* inputs = (uint8_t*)AscendC::GmAlloc(kStubInputsBufBytes);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(kStubOutputBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    EltwiseTilingData* tilingData = reinterpret_cast<EltwiseTilingData*>(tiling);
    memset(tilingData, 0, tilingSize);
    tilingData->totalNum = 0;
    tilingData->blockFactor = 0;
    tilingData->ubFactor = 0;
    tilingData->inputNum = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(eltwise, blockDim, inputs, output, workspace, tiling);

    AscendC::GmFree(inputs);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
