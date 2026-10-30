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
 * \file test_acos.cpp
 * \brief
 */

#include "../../../op_kernel/acos.cpp"
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

class AcosTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "AcosTest SetUp\n" << endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./acos_data/");
    }
    static void TearDownTestCase() { cout << "AcosTest TearDown\n" << endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string AcosTest::rootPath = "../../../../";
const std::string AcosTest::dataPath = rootPath + "experimental/math/acos/tests/ut/op_kernel/acos_data";

TEST_F(AcosTest, test_case_0)
{
    size_t xByteSize = 32 * 32 * 32 * 17 * sizeof(float);
    size_t yByteSize = 32 * 32 * 32 * 17 * sizeof(float);
    size_t tiling_data_size = sizeof(AcosTilingData);
    uint32_t blockDim = 40;

    system("cd ./acos_data/ && python3 gen_data.py '(32, 32, 32, 17)' 'float32'");
    std::string fileName = "./acos_data/float32_input_acos.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    ReadFile(fileName, xByteSize, x, xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AcosTilingData* tilingDatafromBin = reinterpret_cast<AcosTilingData*>(tiling);
    tilingDatafromBin->formerCoreNum = 16;
    tilingDatafromBin->tailCoreNum = 24;
    tilingDatafromBin->formerCoreDataNum = 13927;
    tilingDatafromBin->tailCoreDataNum = 13926;
    tilingDatafromBin->formerCoreLoopCount = 3;
    tilingDatafromBin->formerCoreFormerDataNum = 4928;
    tilingDatafromBin->formerCoreTailDataNum = 4071;
    tilingDatafromBin->tailCoreLoopCount = 3;
    tilingDatafromBin->tailCoreFormerDataNum = 4928;
    tilingDatafromBin->tailCoreTailDataNum = 4070;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(acos<float>, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    fileName = "./acos_data/float32_output_acos.bin";
    WriteFile(fileName, y, yByteSize);

    // 释放资源
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./acos_data/ && python3 compare_data.py 'float32'");
    free(path_);
}

TEST_F(AcosTest, test_case_1)
{
    size_t xByteSize = 32 * 32 * 16 * 17 * sizeof(short);
    size_t yByteSize = 32 * 32 * 16 * 17 * sizeof(short);
    size_t tiling_data_size = sizeof(AcosTilingData);
    uint32_t blockDim = 40;

    system("cd ./acos_data/ && python3 gen_data.py '(32, 32, 16, 17)' 'float16'");
    std::string fileName = "./acos_data/float16_input_acos.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    ReadFile(fileName, xByteSize, x, xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AcosTilingData* tilingDatafromBin = reinterpret_cast<AcosTilingData*>(tiling);
    tilingDatafromBin->formerCoreNum = 8;
    tilingDatafromBin->tailCoreNum = 32;
    tilingDatafromBin->formerCoreDataNum = 6964;
    tilingDatafromBin->tailCoreDataNum = 6963;
    tilingDatafromBin->formerCoreLoopCount = 2;
    tilingDatafromBin->formerCoreFormerDataNum = 6144;
    tilingDatafromBin->formerCoreTailDataNum = 820;
    tilingDatafromBin->tailCoreLoopCount = 2;
    tilingDatafromBin->tailCoreFormerDataNum = 6144;
    tilingDatafromBin->tailCoreTailDataNum = 819;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(acos<half>, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    fileName = "./acos_data/float16_output_acos.bin";
    WriteFile(fileName, y, yByteSize);

    // 释放资源
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./acos_data/ && python3 compare_data.py 'float16'");
    free(path_);
}

TEST_F(AcosTest, test_case_2)
{
    size_t xByteSize = 32 * 32 * 16 * 17 * sizeof(short);
    size_t yByteSize = 32 * 32 * 16 * 17 * sizeof(short);
    size_t tiling_data_size = sizeof(AcosTilingData);
    uint32_t blockDim = 40;

    system("cd ./acos_data/ && python3 gen_data.py '(32, 32, 16, 17)' 'bfloat16'");
    std::string fileName = "./acos_data/bfloat16_input_acos.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    ReadFile(fileName, xByteSize, x, xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AcosTilingData* tilingDatafromBin = reinterpret_cast<AcosTilingData*>(tiling);
    tilingDatafromBin->formerCoreNum = 8;
    tilingDatafromBin->tailCoreNum = 32;
    tilingDatafromBin->formerCoreDataNum = 6964;
    tilingDatafromBin->tailCoreDataNum = 6963;
    tilingDatafromBin->formerCoreLoopCount = 2;
    tilingDatafromBin->formerCoreFormerDataNum = 6144;
    tilingDatafromBin->formerCoreTailDataNum = 820;
    tilingDatafromBin->tailCoreLoopCount = 2;
    tilingDatafromBin->tailCoreFormerDataNum = 6144;
    tilingDatafromBin->tailCoreTailDataNum = 819;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(acos<bfloat16_t>, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    fileName = "./acos_data/bfloat16_output_acos.bin";
    WriteFile(fileName, y, yByteSize);

    // 释放资源
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./acos_data/ && python3 compare_data.py 'bfloat16'");
    free(path_);
}