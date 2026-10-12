/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

#include "../../../op_kernel/round.cpp"

extern "C" __global__ __aicore__ void round(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class RoundTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "round_test SetUp" << std::endl;
        const std::string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./round_data/");
    }

    static void TearDownTestCase() { std::cout << "round_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string RoundTest::rootPath = "../../../../";
const std::string RoundTest::dataPath = rootPath + "math/round/tests/ut/op_kernel/round_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

// 定义 tiling 结构体（需与 kernel 中 RoundTilingData 一致）
struct RoundTilingData {
    uint64_t smallCoreDataNum;
    uint64_t bigCoreDataNum;
    uint64_t finalBigTileNum;
    uint64_t finalSmallTileNum;
    uint64_t tileDataNum;
    uint64_t smallTailDataNum;
    uint64_t bigTailDataNum;
    uint64_t tailBlockNum;
    float decimals; // 新增参数：保留小数位数（如 1 表示不缩放，10 表示保留1位小数等）
};

TEST_F(RoundTest, test_case_float16_1)
{
    uint32_t blockDim = 1;
    system("cd ./round_data/ && python3 gen_data.py '(2,8)' 'float32'");

    uint32_t dataCount = 2 * 8;
    size_t inputByteSize = dataCount * sizeof(float);
    size_t outputByteSize = inputByteSize;

    std::string x_fileName = "./round_data/float16_input_round.bin";
    std::string y_fileName = "./round_data/float16_output_round.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    ReadFile(x_fileName, inputByteSize, x, inputByteSize);

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(RoundTilingData));

    RoundTilingData* tilingData = reinterpret_cast<RoundTilingData*>(tiling);

    // 配置 tiling 参数（与 select_v2 类似，按实际数据量调整）
    tilingData->smallCoreDataNum = 16;
    tilingData->bigCoreDataNum = 0;
    tilingData->finalBigTileNum = 0;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = 4904;
    tilingData->smallTailDataNum = 16;
    tilingData->bigTailDataNum = 0;
    tilingData->tailBlockNum = 0;
    tilingData->decimals = 1.0f; // 不进行缩放，直接四舍五入

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = round_kernel;
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    WriteFile(y_fileName, y, outputByteSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./round_data/ && python3 compare_data.py 'float16'");
}