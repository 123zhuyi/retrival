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
 * \file test_ones_like.cpp
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

#include "../../../op_kernel/ones_like.cpp"

using namespace std;

extern "C" __global__ __aicore__ void ones_like(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class OnesLikeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ones_like_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./ones_like_data/");
    }
    static void TearDownTestCase() { std::cout << "ones_like_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string OnesLikeTest::rootPath = "../../../../";
const std::string OnesLikeTest::dataPath = rootPath + "math/ones_like/tests/ut/op_kernel/ones_like_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(OnesLikeTest, test_case_float16_1)
{
    uint32_t blockDim = 1;
    system("cd ./ones_like_data/ && python3 gen_data.py '(128, 64)' 'float16'");
    uint32_t dataCount = 128 * 64;
    size_t inputCondByteSize = dataCount * sizeof(uint8_t);
    size_t inputByteSize = dataCount * sizeof(half);
    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(OnesLikeTilingData));

    OnesLikeTilingData* tilingData = reinterpret_cast<OnesLikeTilingData*>(tiling);

    tiling->formerNum = 1;
    tiling->tailNum = 0;
    tiling->tailLength = 0;
    tiling->formerLength = 8192;
    tiling->formerTileLength = 8192;
    tiling->formerTileNum = 1;
    tiling->formerLastTileLength = 8192;
    tiling->tailTileLength = 8192;
    tiling->tailTileNum = 1;
    tiling->tailLastTileLength = 8192;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = ones_like<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./ones_like_data/float16_output_t_ones_like.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./ones_like_data/ && python3 compare_data.py 'float16'");
}