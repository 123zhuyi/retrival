/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
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

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "tiling_case_executor.h"
#include "../op_host/lin_space_d_tiling.h"

using namespace std;

// 声明核函数
extern "C" __global__ __aicore__ void lin_space_d(
    GM_ADDR start, GM_ADDR stop, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class LinSpaceDTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LinSpaceDTest SetUp" << std::endl;
        // 复制测试数据目录
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./lin_space_d_data/");
    }
    static void TearDownTestCase() { std::cout << "LinSpaceDTest TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string LinSpaceDTest::rootPath = "../../../../";
const std::string LinSpaceDTest::dataPath = rootPath + "math/lin_space_d/tests/ut/op_kernel/lin_space_d_data";

// 对齐内存分配大小（32字节对齐）
template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

// 测试float32类型输入（start=0, stop=10, num=5）
TEST_F(LinSpaceDTest, test_case_float32)
{
    optiling::LinSpaceDCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LinSpaceD",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // start
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // stop
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}  // num=5
        },
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND} // 输出形状[5]
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    // 生成输入数据和黄金数据（start=0, stop=10, num=5）
    system("cd ./lin_space_d_data/ && python3 gen_data.py 0 10 5 'float32'");

    // 读取输入数据（start和stop各1个元素，float32）
    size_t startStopSize = sizeof(float);
    uint8_t* start = (uint8_t*)AscendC::GmAlloc(CeilAlign(startStopSize, 32));
    uint8_t* stop = (uint8_t*)AscendC::GmAlloc(CeilAlign(startStopSize, 32));
    ReadFile("./lin_space_d_data/float32_start.bin", startStopSize, start, startStopSize);
    ReadFile("./lin_space_d_data/float32_stop.bin", startStopSize, stop, startStopSize);

    // 分配输出内存（5个float32元素）
    size_t outputSize = 5 * sizeof(float);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputSize, 32));

    // 分配工作空间和tiling内存
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    // 执行核函数
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(lin_space_d, tilingInfo.blockNum, start, stop, output, workspace, tiling);

    // 保存输出结果
    WriteFile("./lin_space_d_data/float32_output.bin", output, outputSize);

    // 释放内存
    AscendC::GmFree(start);
    AscendC::GmFree(stop);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    // 比较输出与黄金数据
    system("cd ./lin_space_d_data/ && python3 compare_data.py 'float32'");
}

// 测试int32类型输入（start=2, stop=10, num=4）
TEST_F(LinSpaceDTest, test_case_int32)
{
    optiling::LinSpaceDCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "LinSpaceD",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}, // start=2
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}, // stop=10
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}  // num=4
        },
        {
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND} // 输出[2,4,6,8]
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./lin_space_d_data/ && python3 gen_data.py 2 10 4 'int32'");

    // 读取输入和分配内存（流程同float32测试）
    size_t startStopSize = sizeof(int32_t);
    uint8_t* start = (uint8_t*)AscendC::GmAlloc(CeilAlign(startStopSize, 32));
    uint8_t* stop = (uint8_t*)AscendC::GmAlloc(CeilAlign(startStopSize, 32));
    ReadFile("./lin_space_d_data/int32_start.bin", startStopSize, start, startStopSize);
    ReadFile("./lin_space_d_data/int32_stop.bin", startStopSize, stop, startStopSize);

    size_t outputSize = 4 * sizeof(int32_t);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(lin_space_d, tilingInfo.blockNum, start, stop, output, workspace, tiling);

    WriteFile("./lin_space_d_data/int32_output.bin", output, outputSize);

    AscendC::GmFree(start);
    AscendC::GmFree(stop);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    system("cd ./lin_space_d_data/ && python3 compare_data.py 'int32'");
}