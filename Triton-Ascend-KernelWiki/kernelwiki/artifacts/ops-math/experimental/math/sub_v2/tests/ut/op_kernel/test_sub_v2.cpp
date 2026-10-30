/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua<@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file test_sub_v2.cpp
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
#include "tiling_case_executor.h"
#include "../op_host/sub_v2_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void sub_v2(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);

class SubV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "sub_v2_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./sub_v2_data/");
    }
    static void TearDownTestCase() { std::cout << "sub_v2_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string SubV2Test::rootPath = "../../../../";
const std::string SubV2Test::dataPath = rootPath + "math/sub_v2/tests/ut/op_kernel/sub_v2_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(SubV2Test, test_case_float16_1)
{
    optiling::SubV2CompileInfo compileInfo = {64, 262144, false};
    gert::TilingContextPara tilingContextPara(
        "SubV2",
        {{{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./sub_v2_data/ && python3 gen_data.py '(128, 64)' 'float16'");
    uint32_t dataCount = 128 * 64;
    size_t byteSize = dataCount * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ReadFile("./sub_v2_data/float16_input_x_sub_v2.bin", byteSize, x, byteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ReadFile("./sub_v2_data/float16_input_y_sub_v2.bin", byteSize, y, byteSize);

    uint8_t* z = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(sub_v2, tilingInfo.blockNum, x, y, z, workspace, tiling);

    WriteFile("./sub_v2_data/float16_output_sub_v2.bin", z, byteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)(z));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./sub_v2_data/ && python3 compare_data.py 'float16'");
}

TEST_F(SubV2Test, test_case_float32_1)
{
    optiling::SubV2CompileInfo compileInfo = {64, 262144, false};
    gert::TilingContextPara tilingContextPara(
        "SubV2",
        {// 两个输入
         {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            // 一个输出
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./sub_v2_data/ && python3 gen_data.py '(256, 33)' 'float32'");
    uint32_t dataCount = 256 * 33;
    size_t byteSize = dataCount * sizeof(float);

    // 分配和读取输入X
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ReadFile("./sub_v2_data/float32_input_x_sub_v2.bin", byteSize, x, byteSize);

    // 分配和读取输入Y
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ReadFile("./sub_v2_data/float32_input_y_sub_v2.bin", byteSize, y, byteSize);

    // 分配输出Z
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(sub_v2, tilingInfo.blockNum, x, y, z, workspace, tiling);

    WriteFile("./sub_v2_data/float32_output_sub_v2.bin", z, byteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)(z));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./sub_v2_data/ && python3 compare_data.py 'float32'");
}
