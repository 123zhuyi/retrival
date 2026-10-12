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
 * \file test_pows.cpp
 * \brief
*/

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/pows_tiling.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void pows(GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, GM_ADDR tiling);

class pows_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "pows_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "pows_test TearDown\n" << endl;
    }
};

TEST_F(pows_test, test_case_101)
{
    size_t input1ByteSize = 256 * sizeof(int16_t);
    size_t input2ByteSize = 1 * sizeof(int16_t);
    size_t outputByteSize = 256 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(PowsTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t numBlocks = 40;
    system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/pows/pows_data ./");
    system("chmod -R 755 ./pows_data/");
    system("cd ./pows_data/ && rm -rf ./*bin");
    system("cd ./pows_data/ && python3 gen_data.py 1 1 256 float16");
    system("cd ./pows_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    PowsTilingData* tilingDatafromBin = reinterpret_cast<PowsTilingData*>(tiling);

    tilingDatafromBin->mainCoreLoopNum = 1;
    tilingDatafromBin->mainCoreTailLength = 0;
    tilingDatafromBin->tailCoreLoopNum = 0;
    tilingDatafromBin->tailCoreTailLength = 0;
    tilingDatafromBin->realCoreNum = 1;
    tilingDatafromBin->numPerCore = 1;
    tilingDatafromBin->tilingKey = 101;
    tilingDatafromBin->bufSize = 4096;
    tilingDatafromBin->dataLength = 256;
    tilingDatafromBin->blockSize = 16;

    ReadFile(path + "/pows_data/input_x1.bin", input1ByteSize, x1, input1ByteSize);
    ReadFile(path + "/pows_data/input_x2.bin", input2ByteSize, x2, input2ByteSize);
    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(pows, numBlocks, x1, x2, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
