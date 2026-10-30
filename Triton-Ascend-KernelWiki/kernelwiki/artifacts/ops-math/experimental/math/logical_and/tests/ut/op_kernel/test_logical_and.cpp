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
 * \file test_logical_and.cpp
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

#include "../../../op_kernel/logical_and.cpp"

class LogicalAndTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "logical_and_test SetUp" << std::endl;
        const std::string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./logical_and_data/");
    }
    static void TearDownTestCase() { std::cout << "logical_and_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string LogicalAndTest::rootPath = "../../../../experimental/";
const std::string LogicalAndTest::dataPath = rootPath + "math/logical_and/tests/ut/op_kernel/logical_and_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(LogicalAndTest, test_case_bool_1)
{
    uint32_t blockDim = 1;
    system("cd ./logical_and_data/ && python3 gen_data.py '(128, 64)' 'bool'");
    uint32_t dataCount = 128 * 64;
    size_t inputByteSize = dataCount * sizeof(uint8_t);

    std::string x1_fileName = "./logical_and_data/bool_input_t1_logical_and.bin";
    std::string x2_fileName = "./logical_and_data/bool_input_t2_logical_and.bin";

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));

    ReadFile(x1_fileName, inputByteSize, x1, inputByteSize);
    ReadFile(x2_fileName, inputByteSize, x2, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(uint8_t);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(LogicalAndTilingData));

    LogicalAndTilingData* tilingData = reinterpret_cast<LogicalAndTilingData*>(tiling);

    tilingData->smallCoreDataNum = 8192;
    tilingData->bigCoreDataNum = 8224;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = 49152;
    tilingData->smallTailDataNum = 8192;
    tilingData->bigTailDataNum = 8224;
    tilingData->tailBlockNum = 0;
    tilingData->bufferNum = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = logical_and<ELEMENTWISE_TPL_SCH_MODE_1>;
    ICPU_RUN_KF(func, blockDim, x1, x2, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./logical_and_data/bool_output_t_logical_and.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./logical_and_data/ && python3 compare_data.py 'bool'");
}