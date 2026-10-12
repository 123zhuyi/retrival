/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define DTYPE_X float
#include "../../../op_kernel/rsqrt.cpp"
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "rsqrt_test_base.h"

using namespace std;

class RsqrtTest : public testing::Test {
    RSQRT_SETUP_TEARDOWN(RsqrtTest)
};

// 单流水 (schMode=1) 测试：shape (32, 32, 32, 17)，40核
TEST_F(RsqrtTest, fp32_single_buffer)
{
    constexpr size_t elemCnt = 32 * 32 * 32 * 17;
    size_t byteSize = elemCnt * sizeof(float);
    size_t tilingSize = sizeof(RsqrtTilingData);
    uint32_t blockDim = 40;

    int ret = system("cd ./rsqrt_data/ && python3 gen_data.py '(32, 32, 32, 17)' 'float32'");
    EXPECT_EQ(ret, 0);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(byteSize);
    ReadFile("./rsqrt_data/float32_input_rsqrt.bin", byteSize, x, byteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(byteSize);
    uint8_t* wk = (uint8_t*)AscendC::GmAlloc(RSQRT_WORKSPACE_SIZE);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(tilingSize);

    RsqrtTilingData* td = reinterpret_cast<RsqrtTilingData*>(tl);
    td->smallCoreDataNum = 13920;
    td->bigCoreDataNum = 13928;
    td->finalBigTileNum = 1;
    td->finalSmallTileNum = 1;
    td->tileDataNum = 16296;
    td->smallTailDataNum = 13920;
    td->bigTailDataNum = 13928;
    td->tailBlockNum = 32;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(rsqrt<1>, blockDim, x, y, wk, (uint8_t*)td);

    WriteFile("./rsqrt_data/float32_output_rsqrt.bin", y, byteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(wk);
    AscendC::GmFree(tl);

    ret = system("cd ./rsqrt_data/ && python3 compare_data.py 'float32'");
    EXPECT_EQ(ret, 0);
}

// 双流水 (schMode=0) 测试：大规模 shape 迫使 tiling 选择双流水
TEST_F(RsqrtTest, fp32_double_buffer)
{
    constexpr size_t elemCnt = 4096 * 4096;
    size_t byteSize = elemCnt * sizeof(float);
    size_t tilingSize = sizeof(RsqrtTilingData);
    uint32_t blockDim = 40;

    int ret = system("cd ./rsqrt_data/ && python3 gen_data.py '(4096, 4096)' 'float32'");
    EXPECT_EQ(ret, 0);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(byteSize);
    ReadFile("./rsqrt_data/float32_input_rsqrt.bin", byteSize, x, byteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(byteSize);
    uint8_t* wk = (uint8_t*)AscendC::GmAlloc(RSQRT_WORKSPACE_SIZE);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(tilingSize);

    RsqrtTilingData* td = reinterpret_cast<RsqrtTilingData*>(tl);
    td->smallCoreDataNum = 419424;
    td->bigCoreDataNum = 419432;
    td->finalBigTileNum = 52;
    td->finalSmallTileNum = 52;
    td->tileDataNum = 8144;
    td->smallTailDataNum = 4080;
    td->bigTailDataNum = 4088;
    td->tailBlockNum = 32;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(rsqrt<0>, blockDim, x, y, wk, (uint8_t*)td);

    WriteFile("./rsqrt_data/float32_output_rsqrt.bin", y, byteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(wk);
    AscendC::GmFree(tl);

    ret = system("cd ./rsqrt_data/ && python3 compare_data.py 'float32'");
    EXPECT_EQ(ret, 0);
}
