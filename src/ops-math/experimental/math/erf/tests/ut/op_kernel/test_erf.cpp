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
 * \file test_erf.cpp
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

#include "../../../op_kernel/erf.cpp"

using namespace std;

constexpr uint32_t INPUT_DATA_NUM = 1024;

class ErfTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "erf_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./erf_data/");
    }
    static void TearDownTestCase() { std::cout << "erf_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string ErfTest::rootPath = "../../../../experimental/";
const std::string ErfTest::dataPath = rootPath + "math/erf/tests/ut/op_kernel/erf_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

TEST_F(ErfTest, test_case_float32_1)
{
    system("pwd");
    system("ls -la erf_data 2>/dev/null || echo 'erf_data not found'");

    uint32_t blockDim = 1;
    system("cd ./erf_data/ && python3 gen_data.py '(1024)' 'float32'");
    uint32_t dataCount = INPUT_DATA_NUM;
    size_t inputByteSize = dataCount * sizeof(float);

    std::string self_fileName = "./erf_data/float32_input_self_erf.bin";

    uint8_t* self = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));

    ReadFile(self_fileName, inputByteSize, self, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ErfTilingData));

    ErfTilingData* tilingData = reinterpret_cast<ErfTilingData*>(tiling);

    tilingData->smallCoreDataNum = 0;
    tilingData->bigCoreDataNum = INPUT_DATA_NUM;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 0;
    tilingData->tileDataNum = INPUT_DATA_NUM;
    tilingData->smallTailDataNum = 0;
    tilingData->bigTailDataNum = INPUT_DATA_NUM;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = erf<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, self, out, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./erf_data/float32_output_out_erf.bin";
    WriteFile(fileName, out, outputByteSize);

    AscendC::GmFree((void*)(self));
    AscendC::GmFree((void*)(out));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./erf_data/ && python3 compare_data.py 'float32'");
}