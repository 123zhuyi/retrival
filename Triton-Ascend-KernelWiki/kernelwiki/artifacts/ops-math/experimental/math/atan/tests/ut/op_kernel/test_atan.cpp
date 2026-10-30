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
 * \file test_atan.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/atan.cpp"

using namespace std;

extern "C" __global__ __aicore__ void atan(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class AtanTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "atan_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./atan_data/");
    }
    static void TearDownTestCase() { std::cout << "atan_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string AtanTest::rootPath = "../../../../";
const std::string AtanTest::dataPath = rootPath + "math/atan/tests/ut/op_kernel/atan_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(AtanTest, test_case_float16_1)
{
    uint32_t blockDim = 40;
    system("cd ./atan_data/ && python3 gen_data.py '(128, 64)' 'float16'");
    uint32_t dataCount = 128 * 64;
    size_t inputCondByteSize = dataCount * sizeof(uint8_t);
    size_t inputByteSize = dataCount * sizeof(half);
    std::string x_fileName = "./atan_data/float16_input_t_atan.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 256));
    ReadFile(x_fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 256));

    size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AtanTilingData));

    AtanTilingData* tilingData = reinterpret_cast<AtanTilingData*>(tiling);

    tilingData->smallCoreDataNum = 128;
    tilingData->bigCoreDataNum = 256;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->smallTailDataNum = 128;
    tilingData->bigTailDataNum = 256;
    tilingData->tileDataNum = 5120;
    tilingData->tailBlockNum = 24;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = atan<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./atan_data/float16_output_t_atan.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./atan_data/ && python3 compare_data.py 'float16'");
}