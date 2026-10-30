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
 * \file test_real.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <complex>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/real.cpp"

using namespace std;

class RealTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "real_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./real_data/");
    }
    static void TearDownTestCase() { std::cout << "real_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string RealTest::rootPath = "../../../../experimental/";
const std::string RealTest::dataPath = rootPath + "math/real/tests/ut/op_kernel/real_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

// Test case for complex32 -> float16
TEST_F(RealTest, test_case_complex32)
{
    system("pwd");
    system("ls -la real_data 2>/dev/null || echo 'real_data not found'");

    uint32_t blockDim = 1;
    system("cd ./real_data/ && python3 gen_data.py '(1, 32)' 'complex32'");

    uint32_t dataCount = 32;
    size_t inputByteSize = dataCount * sizeof(int32_t);
    std::string xfileName = "./real_data/complex32_input_real.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(xfileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(RealTilingData));

    RealTilingData* tilingData = reinterpret_cast<RealTilingData*>(tiling);

    tilingData->totalUsedCoreNum = 1;
    tilingData->tailBlockNum = 0;
    tilingData->ubPartDataNum = dataCount;
    tilingData->smallCoreDataNum = dataCount;
    tilingData->smallCoreLoopNum = 1;
    tilingData->smallCoreTailDataNum = dataCount;
    tilingData->bigCoreDataNum = 0;
    tilingData->bigCoreLoopNum = 0;
    tilingData->bigCoreTailDataNum = 0;
    tilingData->tilingKey = 1;     // COMPLEX32_MODE
    tilingData->useNonInplace = 1; // 32 elements: 32*2*2=128 < 256, non-inplace

    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    using RealFuncType = void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
    RealFuncType func = reinterpret_cast<RealFuncType>(::real);
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./real_data/complex32_output_real.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./real_data/ && python3 compare_data.py 'complex32'");
}

// Test case for complex64 -> float32
TEST_F(RealTest, test_case_complex64)
{
    system("pwd");
    system("ls -la real_data 2>/dev/null || echo 'real_data not found'");

    uint32_t blockDim = 1;
    system("cd ./real_data/ && python3 gen_data.py '(1, 32)' 'complex64'");

    uint32_t dataCount = 32;
    size_t inputByteSize = dataCount * sizeof(std::complex<float>);
    std::string xfileName = "./real_data/complex64_input_real.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(xfileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(RealTilingData));

    RealTilingData* tilingData = reinterpret_cast<RealTilingData*>(tiling);

    tilingData->totalUsedCoreNum = 1;
    tilingData->tailBlockNum = 0;
    tilingData->ubPartDataNum = dataCount;
    tilingData->smallCoreDataNum = dataCount;
    tilingData->smallCoreLoopNum = 1;
    tilingData->smallCoreTailDataNum = dataCount;
    tilingData->bigCoreDataNum = 0;
    tilingData->bigCoreLoopNum = 0;
    tilingData->bigCoreTailDataNum = 0;
    tilingData->tilingKey = 2; // COMPLEX64_MODE

    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    using RealFuncType = void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
    RealFuncType func = reinterpret_cast<RealFuncType>(::real);
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./real_data/complex64_output_real.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./real_data/ && python3 compare_data.py 'complex64'");
}

// Test case for float16 -> float16 (identity)
TEST_F(RealTest, test_case_float16)
{
    system("pwd");
    system("ls -la real_data 2>/dev/null || echo 'real_data not found'");

    uint32_t blockDim = 1;
    system("cd ./real_data/ && python3 gen_data.py '(1, 32)' 'float16'");

    uint32_t dataCount = 32;
    size_t inputByteSize = dataCount * sizeof(half);
    std::string xfileName = "./real_data/float16_input_real.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(xfileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(RealTilingData));

    RealTilingData* tilingData = reinterpret_cast<RealTilingData*>(tiling);

    tilingData->totalUsedCoreNum = 1;
    tilingData->tailBlockNum = 0;
    tilingData->ubPartDataNum = dataCount;
    tilingData->smallCoreDataNum = dataCount;
    tilingData->smallCoreLoopNum = 1;
    tilingData->smallCoreTailDataNum = dataCount;
    tilingData->bigCoreDataNum = 0;
    tilingData->bigCoreLoopNum = 0;
    tilingData->bigCoreTailDataNum = 0;
    tilingData->tilingKey = 4; // FLOAT16_MODE

    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    using RealFuncType = void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
    RealFuncType func = reinterpret_cast<RealFuncType>(::real);
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./real_data/float16_output_real.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./real_data/ && python3 compare_data.py 'float16'");
}

// Test case for float32 -> float32 (identity)
TEST_F(RealTest, test_case_float32)
{
    system("pwd");
    system("ls -la real_data 2>/dev/null || echo 'real_data not found'");

    uint32_t blockDim = 1;
    system("cd ./real_data/ && python3 gen_data.py '(1, 32)' 'float32'");

    uint32_t dataCount = 32;
    size_t inputByteSize = dataCount * sizeof(float);
    std::string xfileName = "./real_data/float32_input_real.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(xfileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(RealTilingData));

    RealTilingData* tilingData = reinterpret_cast<RealTilingData*>(tiling);

    tilingData->totalUsedCoreNum = 1;
    tilingData->tailBlockNum = 0;
    tilingData->ubPartDataNum = dataCount;
    tilingData->smallCoreDataNum = dataCount;
    tilingData->smallCoreLoopNum = 1;
    tilingData->smallCoreTailDataNum = dataCount;
    tilingData->bigCoreDataNum = 0;
    tilingData->bigCoreLoopNum = 0;
    tilingData->bigCoreTailDataNum = 0;
    tilingData->tilingKey = 5; // FLOAT_MODE

    ICPU_SET_TILING_KEY(tilingData->tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    using RealFuncType = void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
    RealFuncType func = reinterpret_cast<RealFuncType>(::real);
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./real_data/float32_output_real.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./real_data/ && python3 compare_data.py 'float32'");
}
