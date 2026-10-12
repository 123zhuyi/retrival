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
 * \file test_real_div.cpp
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

#include "../../../op_kernel/real_div.cpp"

using namespace std;

class RealDivTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "real_div_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./real_div_data/");
    }
    static void TearDownTestCase() { std::cout << "real_div_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string RealDivTest::rootPath = "../../../../experimental/";
const std::string RealDivTest::dataPath = rootPath + "math/real_div/tests/ut/op_kernel/real_div_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(RealDivTest, test_case_float16_1)
{
    system("pwd");
    system("ls -la real_div_data 2>/dev/null || echo 'real_div_data not found'");

    uint32_t blockDim = 1;
    system("cd ./real_div_data/ && python3 gen_data.py '(1, 32)' 'float16'");

    uint32_t dataCount = 32;
    size_t inputByteSize = dataCount * sizeof(half);
    std::string x1fileName = "./real_div_data/float16_input_real_div_x1.bin";
    std::string x2fileName = "./real_div_data/float16_input_real_div_x2.bin";

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(x1fileName, inputByteSize, x1, inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(x2fileName, inputByteSize, x2, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(RealDivTilingData));

    RealDivTilingData* tilingData = reinterpret_cast<RealDivTilingData*>(tiling);

    tilingData->smallCoreDataNum = 32;
    tilingData->bigCoreDataNum = 48;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = 4912;
    tilingData->smallTailDataNum = 32;
    tilingData->bigTailDataNum = 48;
    tilingData->tailBlockNum = 0;
    tilingData->bufferNum = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = real_div<ELEMENTWISE_TPL_SCH_MODE_1>;
    ICPU_RUN_KF(func, blockDim, x1, x2, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./real_div_data/float16_output_real_div.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./real_div_data/ && python3 compare_data.py 'float16'");
}