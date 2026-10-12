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
 * \file test_axpy_v2.cpp
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

#include "../../../op_kernel/axpy_v2.cpp"

using namespace std;

class AxpyV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "axpy_v2_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./axpy_v2_data/");
    }
    static void TearDownTestCase() { std::cout << "axpy_v2_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string AxpyV2Test::rootPath = "../../../../experimental/";
const std::string AxpyV2Test::dataPath = rootPath + "math/axpy_v2/tests/ut/op_kernel/axpy_v2_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(AxpyV2Test, test_case_float16_1)
{
    uint32_t blockDim = 1;
    system("cd ./axpy_v2_data/ && python3 gen_data.py '(128, 64)' 'float16'");
    uint32_t dataCount = 128 * 64;
    size_t inputAlphaByteSize = sizeof(half);
    size_t inputByteSize = dataCount * sizeof(half);

    std::string x1_fileName = "./axpy_v2_data/float16_input_t1_axpy_v2.bin";
    std::string x2_fileName = "./axpy_v2_data/float16_input_t2_axpy_v2.bin";
    std::string alpha_fileName = "./axpy_v2_data/float16_input_alpha_axpy_v2.bin";

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* alpha = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputAlphaByteSize, 32));

    ReadFile(x1_fileName, inputByteSize, x1, inputByteSize);
    ReadFile(x2_fileName, inputByteSize, x2, inputByteSize);
    ReadFile(alpha_fileName, inputAlphaByteSize, alpha, inputAlphaByteSize);

    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AxpyV2TilingData));

    AxpyV2TilingData* tilingData = reinterpret_cast<AxpyV2TilingData*>(tiling);

    tilingData->smallCoreDataNum = 8192;
    tilingData->bigCoreDataNum = 8320;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = 16384;
    tilingData->smallTailDataNum = 8192;
    tilingData->bigTailDataNum = 8320;
    tilingData->tailBlockNum = 0;
    tilingData->bufferNum = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = axpy_v2<ELEMENTWISE_TPL_SCH_MODE_1>;
    ICPU_RUN_KF(func, blockDim, x1, x2, alpha, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./axpy_v2_data/float16_output_t_axpy_v2.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(alpha));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./axpy_v2_data/ && python3 compare_data.py 'float16'");
}