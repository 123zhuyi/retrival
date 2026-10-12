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

#include "../../../op_kernel/sign_bits_unpack.cpp"

using namespace std;

constexpr uint32_t smallCoreDataNum = 128;
constexpr uint32_t bigCoreDataNum = 160;
constexpr uint32_t tileDataNum = 2048;
constexpr uint32_t smallTailDataNum = 128;
constexpr uint32_t bigTailDataNum = 160;

extern "C" __global__ __aicore__ void sign_bits_unpack(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);

class SignBitsUnpackTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "sign_bits_unpack_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./sign_bits_unpack_data/");
    }
    static void TearDownTestCase() { std::cout << "sign_bits_unpack_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string SignBitsUnpackTest::rootPath = "../../../../experimental/";
const std::string SignBitsUnpackTest::dataPath =
    rootPath + "math/sign_bits_unpack/tests/ut/op_kernel/sign_bits_unpack_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(SignBitsUnpackTest, test_case_float_1)
{
    uint32_t blockDim = 1;
    system("cd ./sign_bits_unpack_data/ && python3 gen_data.py '(128)' 'uint8'");
    uint32_t dataCount = 1024;
    uint32_t dataCountIn = 128;
    size_t inputByteSize = dataCountIn * sizeof(uint8_t);

    std::string self_fileName = "./sign_bits_unpack_data/uint8_input_self_sign_bits_unpack.bin";

    uint8_t* self = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));

    ReadFile(self_fileName, inputByteSize, self, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(SignBitsUnpackTilingData));

    SignBitsUnpackTilingData* tilingData = reinterpret_cast<SignBitsUnpackTilingData*>(tiling);

    tilingData->smallCoreDataNum = smallCoreDataNum;
    tilingData->bigCoreDataNum = bigCoreDataNum;
    tilingData->tileDataNum = tileDataNum;
    tilingData->smallTailDataNum = smallTailDataNum;
    tilingData->bigTailDataNum = bigTailDataNum;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;
    tilingData->bufferOpen = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = sign_bits_unpack<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, self, out, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./sign_bits_unpack_data/float_output_t_sign_bits_unpack.bin";
    WriteFile(fileName, out, outputByteSize);

    AscendC::GmFree((void*)(self));
    AscendC::GmFree((void*)(out));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./sign_bits_unpack_data/ && python3 compare_data.py 'float'");
}