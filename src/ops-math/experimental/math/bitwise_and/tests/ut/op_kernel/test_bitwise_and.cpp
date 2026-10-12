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

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/bitwise_and.cpp"

using namespace std;

constexpr uint32_t smallCoreDataNum = 1024;
constexpr uint32_t bigCoreDataNum = 1040;
constexpr uint32_t tileDataNum = 2048;
constexpr uint32_t smallTailDataNum = 1024;
constexpr uint32_t bigTailDataNum = 1040;
constexpr uint32_t tmpTileDataNum = 4096;
constexpr uint32_t tmpSmallTailDataNum = 2048;
constexpr uint32_t tmpBigTailDataNum = 2080;

extern "C" __global__ __aicore__ void bitwise_and(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class BitwiseAndTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "bitwise_and_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./bitwise_and_data/");
    }
    static void TearDownTestCase() { std::cout << "bitwise_and_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string BitwiseAndTest::rootPath = "../../../../experimental/";
const std::string BitwiseAndTest::dataPath = rootPath + "math/bitwise_and/tests/ut/op_kernel/bitwise_and_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(BitwiseAndTest, test_case_int16_1)
{
    uint32_t blockDim = 1;
    system("cd ./bitwise_and_data/ && python3 gen_data.py '(1024)' 'int16'");
    uint32_t dataCount = 1024;
    size_t inputByteSize = dataCount * sizeof(int16_t);

    std::string x1_fileName = "./bitwise_and_data/int16_input_t1_bitwise_and.bin";
    std::string x2_fileName = "./bitwise_and_data/int16_input_t2_bitwise_and.bin";

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));

    ReadFile(x1_fileName, inputByteSize, x1, inputByteSize);
    ReadFile(x2_fileName, inputByteSize, x2, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(int16_t);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(BitwiseAndTilingData));

    BitwiseAndTilingData* tilingData = reinterpret_cast<BitwiseAndTilingData*>(tiling);

    tilingData->smallCoreDataNum = smallCoreDataNum;
    tilingData->bigCoreDataNum = bigCoreDataNum;
    tilingData->tileDataNum = tileDataNum;
    tilingData->smallTailDataNum = smallTailDataNum;
    tilingData->bigTailDataNum = bigTailDataNum;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;
    tilingData->tmpTileDataNum = tmpTileDataNum;
    tilingData->tmpSmallTailDataNum = tmpSmallTailDataNum;
    tilingData->tmpBigTailDataNum = tmpBigTailDataNum;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = bitwise_and<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, x1, x2, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./bitwise_and_data/int16_output_t_bitwise_and.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./bitwise_and_data/ && python3 compare_data.py 'int16'");
}