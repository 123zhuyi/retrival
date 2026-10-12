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
 * \file test_floor_mod.cpp
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

#include "../../../op_kernel/floor_mod.cpp"

using namespace std;
using namespace FloorModNs;

constexpr uint32_t USABLE_UB_SIZE = 2048;
constexpr uint32_t ALIGNMENT = 32;
constexpr size_t WORKSPACE_SIZE = 32 * 1024 * 1024;

class FloorModTest : public testing::Test {
public:
    static std::string srcDir;

protected:
    static void SetUpTestCase()
    {
        std::cout << "floor_mod_test SetUp" << std::endl;

        std::string fullPath = __FILE__;
        size_t lastSlash = fullPath.find_last_of("/");
        if (lastSlash != std::string::npos) {
            srcDir = fullPath.substr(0, lastSlash);
        } else {
            srcDir = ".";
        }
    }

    static void TearDownTestCase() { std::cout << "floor_mod_test TearDown" << std::endl; }
};

std::string FloorModTest::srcDir = "";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

#define TEST_FP16 20

TEST_F(FloorModTest, test_case_float16_1)
{
    uint32_t blockDim = 1;

    std::string shape_str = "(32, 32)";
    uint32_t dataCount = 32 * 32;

    std::string scriptPath = srcDir + "/floor_mod_data/gen_data.py";

    std::string cmd = "python3 " + scriptPath + " '" + shape_str + "' 'float16'";
    int gen_ret = system(cmd.c_str());
    EXPECT_EQ(gen_ret, 0) << "Failed to generate data using script: " << scriptPath;

    size_t inputByteSize = dataCount * sizeof(half);
    size_t outputByteSize = dataCount * sizeof(half);

    // 读取当前目录下的 bin 文件
    std::string x1_fileName = "float16_input_x1.bin";
    std::string x2_fileName = "float16_input_x2.bin";

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, ALIGNMENT));
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, ALIGNMENT));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, ALIGNMENT));

    ReadFile(x1_fileName, inputByteSize, x1, inputByteSize);
    ReadFile(x2_fileName, inputByteSize, x2, inputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(WORKSPACE_SIZE);

    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FloorModTilingData));
    FloorModTilingData* tilingData = reinterpret_cast<FloorModTilingData*>(tiling);

    // 配置 Tiling 数据
    tilingData->usableUbSize = USABLE_UB_SIZE;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = dataCount;
    tilingData->perCoreDataCount = dataCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = dataCount;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    auto func = floor_mod<TEST_FP16, TEST_FP16, TEST_FP16>;

    ICPU_RUN_KF(func, blockDim, x1, x2, y, workspace, tiling);

    std::string out_fileName = "float16_output_y.bin";
    WriteFile(out_fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    // 调用比对脚本
    std::string cmpPath = srcDir + "/floor_mod_data/compare_data.py";
    std::string cmp_cmd = "python3 " + cmpPath + " 'float16'";
    int ret = system(cmp_cmd.c_str());
    EXPECT_EQ(ret, 0) << "Data comparison failed using script: " << cmpPath;
}