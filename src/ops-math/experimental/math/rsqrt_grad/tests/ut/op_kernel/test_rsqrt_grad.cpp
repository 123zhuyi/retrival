/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/rsqrt_grad.cpp"

using namespace std;

constexpr uint32_t smallCoreDataNum = 1024;
constexpr uint32_t bigCoreDataNum = 1040;
constexpr uint32_t tileDataNum = 2048;
constexpr uint32_t smallTailDataNum = 1024;
constexpr uint32_t bigTailDataNum = 1040;

extern "C" __global__ __aicore__ void rsqrt_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);

class RsqrtGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "rsqrt_grad_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./rsqrt_grad_data/");
    }
    static void TearDownTestCase() { std::cout << "rsqrt_grad_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string RsqrtGradTest::rootPath = "../../../../experimental/";
const std::string RsqrtGradTest::dataPath = rootPath + "math/rsqrt_grad/tests/ut/op_kernel/rsqrt_grad_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(RsqrtGradTest, test_case_float_1)
{
    uint32_t blockDim = 1;
    system("cd ./rsqrt_grad_data/ && python3 gen_data.py '(1024)' 'float'");
    uint32_t dataCount = 1024;
    size_t inputByteSize = dataCount * sizeof(float_t);

    std::string y_fileName = "./rsqrt_grad_data/float_input_y_rsqrt_grad.bin";
    std::string dy_fileName = "./rsqrt_grad_data/float_input_dy_rsqrt_grad.bin";

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));

    ReadFile(y_fileName, inputByteSize, y, inputByteSize);
    ReadFile(dy_fileName, inputByteSize, dy, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(float_t);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(RsqrtGradTilingData));

    RsqrtGradTilingData* tilingData = reinterpret_cast<RsqrtGradTilingData*>(tiling);

    tilingData->smallCoreDataNum = smallCoreDataNum;
    tilingData->bigCoreDataNum = bigCoreDataNum;
    tilingData->tileDataNum = tileDataNum;
    tilingData->smallTailDataNum = smallTailDataNum;
    tilingData->bigTailDataNum = bigTailDataNum;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;
    tilingData->bufferOpen = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = rsqrt_grad<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, y, dy, z, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./rsqrt_grad_data/float_output_t_rsqrt_grad.bin";
    WriteFile(fileName, z, outputByteSize);

    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)(dy));
    AscendC::GmFree((void*)(z));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./rsqrt_grad_data/ && python3 compare_data.py 'float'");
}