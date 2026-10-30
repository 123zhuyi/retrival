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

#include "../../../op_kernel/sign_bits_pack.cpp"

using namespace std;

constexpr uint32_t smallCoreDataNum = 128;
constexpr uint32_t bigCoreDataNum = 256;
constexpr uint32_t tileDataNum = 128;
constexpr uint32_t smallTailDataNum = 1024;
constexpr uint32_t bigTailDataNum = 1040;

extern "C" __global__ __aicore__ void sign_bits_pack(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class SignBitsPackTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "sign_bits_pack_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./sign_bits_pack_data/");
    }
    static void TearDownTestCase() { std::cout << "sign_bits_pack_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string SignBitsPackTest::rootPath = "../../../../experimental/";
const std::string SignBitsPackTest::dataPath = rootPath + "math/sign_bits_pack/tests/ut/op_kernel/sign_bits_pack_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0)
        return 0;
    return (a + b - 1) / b * b;
}

TEST_F(SignBitsPackTest, test_case_float_1)
{
    size_t inputXByteSize = 14 * sizeof(float);
    size_t outputYByteSize = 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(SignBitsPackTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    SignBitsPackTilingData* tilingDatafromBin = reinterpret_cast<SignBitsPackTilingData*>(tiling);
    tilingDatafromBin->smallCoreDataNum = 128;
    tilingDatafromBin->bigCoreDataNum = 256;
    tilingDatafromBin->finalBigTileNum = 2;
    tilingDatafromBin->finalSmallTileNum = 1;
    tilingDatafromBin->tileDataNum = 128;
    tilingDatafromBin->smallTailDataNum = 128;
    tilingDatafromBin->bigTailDataNum = 128;
    tilingDatafromBin->tailBlockNum = 0;
    tilingDatafromBin->usedDb = 0;
    tilingDatafromBin->lastCopyLength = 14;
    tilingDatafromBin->rightPaddingElemNums = 2;
    tilingDatafromBin->lastCalcLength = 64;

    auto KernelSignBitsPack = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::sign_bits_pack<0>(x, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(sign_bits_pack<0>, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}