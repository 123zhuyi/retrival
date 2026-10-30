/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_cross.cpp
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

#include "../../../op_kernel/cross.cpp"

using namespace std;

extern "C" __global__ __aicore__ void cross(
    GM_ADDR c, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class CrossTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "cross_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system("echo \"here is \" && pwd");
        system(cmd.c_str());
        system("chmod -R 755 ./cross_data/");
    }
    static void TearDownTestCase() { std::cout << "cross_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string CrossTest::rootPath = "../../../../";
const std::string CrossTest::dataPath = rootPath + "experimental/math/cross/tests/ut/op_kernel/cross_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(CrossTest, test_case_float16_1)
{
    uint32_t blockDim = 1;
    // cross 算子需要有一个维度为3
    system("cd ./cross_data/ && python3 gen_data.py '(2, 3)' 'float16'");
    uint32_t dataCount = 2 * 3;
    size_t inputByteSize = dataCount * sizeof(half);

    std::string x1_fileName = "./cross_data/float16_input_t1_cross.bin";
    std::string x2_fileName = "./cross_data/float16_input_t2_cross.bin";

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));

    ReadFile(x1_fileName, inputByteSize, x1, inputByteSize);
    ReadFile(x2_fileName, inputByteSize, x2, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(CrossTilingData));

    CrossTilingData* tilingData = reinterpret_cast<CrossTilingData*>(tiling);

    tilingData->smallCoreDataNum = 8192;
    tilingData->bigCoreDataNum = 8208;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = 8192;
    tilingData->smallTailDataNum = 8192;
    tilingData->bigTailDataNum = 8208;
    tilingData->tailBlockNum = 0;
    tilingData->intervalNum = 1;
    tilingData->loopTimes = 2;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = cross<ELEMENTWISE_TPL_SCH_MODE_3>;
    ICPU_RUN_KF(func, blockDim, x1, x2, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./cross_data/float16_output_t_cross.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./cross_data/ && python3 compare_data.py 'float16'");
}

TEST_F(CrossTest, test_case_float32_1)
{
    uint32_t blockDim = 1;
    // cross 算子需要有一个维度为3
    system("cd ./cross_data/ && python3 gen_data.py '(2, 3)' 'float32'");
    uint32_t dataCount = 2 * 3;
    size_t inputByteSize = dataCount * sizeof(float);

    std::string x1_fileName = "./cross_data/float32_input_t1_cross.bin";
    std::string x2_fileName = "./cross_data/float32_input_t2_cross.bin";

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));

    ReadFile(x1_fileName, inputByteSize, x1, inputByteSize);
    ReadFile(x2_fileName, inputByteSize, x2, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(CrossTilingData));

    CrossTilingData* tilingData = reinterpret_cast<CrossTilingData*>(tiling);

    tilingData->smallCoreDataNum = 8192;
    tilingData->bigCoreDataNum = 8208;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = 8192;
    tilingData->smallTailDataNum = 8192;
    tilingData->bigTailDataNum = 8200;
    tilingData->tailBlockNum = 0;
    tilingData->intervalNum = 1;
    tilingData->loopTimes = 2;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = cross<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, x1, x2, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./cross_data/float32_output_t_cross.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./cross_data/ && python3 compare_data.py 'float32'");
}
