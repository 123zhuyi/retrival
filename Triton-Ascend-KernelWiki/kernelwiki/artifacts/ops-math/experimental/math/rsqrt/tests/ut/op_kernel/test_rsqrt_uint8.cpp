/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define DTYPE_X uint8_t
#define RSQRT_KERNEL_SUFFIX uint8
#include "rsqrt_test_entry.h"
#include <string>
#include <cstdlib>
#include "rsqrt_test_base.h"

using namespace std;

class RsqrtUint8Test : public testing::Test {
    RSQRT_SETUP_TEARDOWN(RsqrtUint8Test)
};

TEST_F(RsqrtUint8Test, uint8_0)
{
    constexpr size_t elemCnt = 128;
    size_t byteSize = elemCnt * sizeof(uint8_t);
    size_t tilingSize = sizeof(RsqrtTilingData);
    uint32_t blockDim = 1;

    int ret = system("cd ./rsqrt_data/ && python3 gen_data.py '(128,)' 'uint8'");
    EXPECT_EQ(ret, 0);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(byteSize);
    ReadFile("./rsqrt_data/uint8_input_rsqrt.bin", byteSize, x, byteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(byteSize);
    uint8_t* wk = (uint8_t*)AscendC::GmAlloc(RSQRT_WORKSPACE_SIZE);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(tilingSize);

    RsqrtTilingData* td = reinterpret_cast<RsqrtTilingData*>(tl);
    td->smallCoreDataNum = 128;
    td->bigCoreDataNum = 160;
    td->finalBigTileNum = 1;
    td->finalSmallTileNum = 1;
    td->tileDataNum = 48896;
    td->smallTailDataNum = 128;
    td->bigTailDataNum = 160;
    td->tailBlockNum = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(rsqrt_uint8<1>, blockDim, x, y, wk, (uint8_t*)td);

    WriteFile("./rsqrt_data/uint8_output_rsqrt.bin", y, byteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(wk);
    AscendC::GmFree(tl);

    ret = system("cd ./rsqrt_data/ && python3 compare_data.py 'uint8'");
    EXPECT_EQ(ret, 0);
}
