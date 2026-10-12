/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FOR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/diag_part.cpp"
#include "../../../op_kernel/diag_part_tiling_key.h"

using namespace std;

// Default macro values (can be overridden by compile flags)
#ifndef DTYPE_X
#define DTYPE_X half
#endif
#ifndef DTYPE_VALUE
#define DTYPE_VALUE 1 // DIAG_PART_TPL_DTYPE_FLOAT16
#endif
#ifndef ALIGN_SIZE
#define ALIGN_SIZE 16 // float16 alignment
#endif

extern "C" __global__ __aicore__ void diag_part(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class DiagPartTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "diag_part_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./diag_part_data/");
    }
    static void TearDownTestCase() { std::cout << "diag_part_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string DiagPartTest::rootPath = "../../../../experimental/";
const std::string DiagPartTest::dataPath = rootPath + "math/diag_part/tests/ut/op_kernel/diag_part_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

// Helper: run one diag_part kernel UT case
static void RunDiagPartTest(uint32_t sideLen)
{
    uint32_t blockDim = 1;
    system(("cd ./diag_part_data/ && python3 gen_data.py '(" + to_string(sideLen) + "," + to_string(sideLen) +
            ")' 'float16'")
               .c_str());

    uint32_t inputCount = sideLen * sideLen;
    uint32_t outputCount = sideLen;
    size_t inputByteSize = inputCount * sizeof(DTYPE_X);
    size_t outputByteSize = outputCount * sizeof(DTYPE_X);

    std::string x_fileName = "./diag_part_data/float16_input_x_diag_part.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    ReadFile(x_fileName, inputByteSize, x, inputByteSize);

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(DiagPartTilingData));

    // Tiling matches diag_part_tiling.cpp logic
    uint32_t tailNum = sideLen % ALIGN_SIZE;
    if (tailNum == 0) {
        tailNum = ALIGN_SIZE; // exact fit: last block is full
    }

    DiagPartTilingData* tilingData = reinterpret_cast<DiagPartTilingData*>(tiling);
    tilingData->sideLength = sideLen;
    tilingData->dtype = DTYPE_VALUE;
    tilingData->realCoreNum = 1;
    tilingData->numPerCore = ALIGN_SIZE;
    tilingData->tailNum = tailNum;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = diag_part<ELEMENTWISE_TPL_SCH_MODE_0, DTYPE_VALUE>;
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./diag_part_data/float16_output_diag_part.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./diag_part_data/ && python3 compare_data.py 'float16'");
}

TEST_F(DiagPartTest, test_case_4x4) { RunDiagPartTest(4); }

TEST_F(DiagPartTest, test_case_8x8) { RunDiagPartTest(8); }
