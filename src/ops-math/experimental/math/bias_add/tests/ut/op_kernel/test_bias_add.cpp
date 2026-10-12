/**
 * This file is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
 * All Rights Reserved.
 *
 * Author (account):
 * - Yang Zhenze <@gcw_5x5Ew5Ms>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_bias_add.cpp
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

#include "../../../op_kernel/bias_add.cpp"

using namespace std;

class BiasAddTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "bias_add_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./bias_add_data/");
    }
    static void TearDownTestCase() { std::cout << "bias_add_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string BiasAddTest::rootPath = "../../../../experimental/";
const std::string BiasAddTest::dataPath = rootPath + "math/bias_add/tests/ut/op_kernel/bias_add_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

// Build a single-core BiasAddTilingData and run one schMode against the float golden,
// then assert bit/tol-close output via compare_data.py. total must be a multiple of channel.
static void RunFloatGoldenCase(uint32_t schModeKey, uint32_t total, uint32_t channel)
{
    const string genCmd = "cd ./bias_add_data/ && python3 gen_data.py " + std::to_string(total) + " " +
                          std::to_string(channel) + " float32";
    system(genCmd.c_str());

    size_t inputByteSize = total * sizeof(float);
    size_t biasByteSize = channel * sizeof(float);
    size_t outputByteSize = total * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(CeilAlign(biasByteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    ReadFile("./bias_add_data/float32_input_bias_add_x.bin", inputByteSize, x, inputByteSize);
    ReadFile("./bias_add_data/float32_input_bias_add_bias.bin", biasByteSize, bias, biasByteSize);

    size_t workspaceSize = 32;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(BiasAddTilingData));

    BiasAddTilingData* tilingData = reinterpret_cast<BiasAddTilingData*>(tiling);
    memset(tilingData, 0, sizeof(BiasAddTilingData));
    tilingData->totalElements = total;
    tilingData->channelSize = channel;
    tilingData->innerSize = 1;
    tilingData->smallCoreDataNum = total;
    tilingData->bigCoreDataNum = total;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = total;
    tilingData->biasCacheElems = 0;
    tilingData->brcTmpBytes = 0;
    tilingData->useFastPath = (schModeKey == BIAS_ADD_TPL_SCH_MODE_TINY_NOQUEUE) ? 1 : 0;
    tilingData->superCycleSize = 0;
    tilingData->kCycleCount = 0;
    tilingData->smallTailDataNum = total;
    tilingData->bigTailDataNum = total;
    tilingData->tailBlockNum = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if (schModeKey == BIAS_ADD_TPL_SCH_MODE_TINY_NOQUEUE) {
        ICPU_RUN_KF((bias_add<BIAS_ADD_TPL_SCH_MODE_TINY_NOQUEUE>), 1, x, bias, y, workspace, tiling);
    } else {
        ICPU_RUN_KF((bias_add<BIAS_ADD_TPL_SCH_MODE_BASE>), 1, x, bias, y, workspace, tiling);
    }

    WriteFile("./bias_add_data/float32_output_bias_add.bin", y, outputByteSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    int ret = system("cd ./bias_add_data/ && python3 compare_data.py float32");
    EXPECT_EQ(ret, 0);
}

TEST_F(BiasAddTest, test_tiny_noqueue_float32) { RunFloatGoldenCase(BIAS_ADD_TPL_SCH_MODE_TINY_NOQUEUE, 24, 6); }

TEST_F(BiasAddTest, test_base_float32) { RunFloatGoldenCase(BIAS_ADD_TPL_SCH_MODE_BASE, 48, 6); }
