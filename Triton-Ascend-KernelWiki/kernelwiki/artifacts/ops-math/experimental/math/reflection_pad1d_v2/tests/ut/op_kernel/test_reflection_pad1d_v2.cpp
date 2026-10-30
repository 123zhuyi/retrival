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
#include "../op_host/reflection_pad1d_v2_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void reflection_pad1d_v2(
    GM_ADDR x, GM_ADDR padding, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class ReflectionPad1dV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReflectionPad1dV2Test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./reflection_pad1d_v2_data/");
    }
    static void TearDownTestCase() { std::cout << "ReflectionPad1dV2Test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

// 静态路径定义：对应ReflectionPad1dV2数据目录
const std::string ReflectionPad1dV2Test::rootPath = "../../../../";
const std::string ReflectionPad1dV2Test::dataPath =
    rootPath + "math/reflection_pad1d_v2/tests/ut/op_kernel/reflection_pad1d_v2_data";

// 对齐内存分配大小
template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

// 测试用例1：float32类型 + 左3右4填充 + 1D输入[8]
TEST_F(ReflectionPad1dV2Test, test_case_float32)
{
    // 1. 构造分块上下文参数
    optiling::ReflectionPad1dV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（1D形状[8]，float32类型，ND格式）
         {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         // 输入1：padding（1D形状[2]，int32类型，ND格式，对应[3,4]）
         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {// 输出0：形状[15]（8+3+4），float32类型，ND格式
         {{{15}, {15}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    // 2. 调用Python脚本生成输入数据和黄金数据（参数：padLeft=3, padRight=4, wSize=8, dtype=float32）
    system("cd ./reflection_pad1d_v2_data/ && python3 gen_data.py 3 4 8 'float32'");

    // 3. 分配并读取输入x的GM内存（与LinSpaceD一致的内存操作）
    size_t xSize = 8 * sizeof(float);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(xSize, 32));
    ReadFile("./reflection_pad1d_v2_data/float32_x.bin", xSize, x, xSize);

    // 4. 分配并读取输入padding的GM内存
    size_t paddingSize = 2 * sizeof(int32_t);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(CeilAlign(paddingSize, 32));
    ReadFile("./reflection_pad1d_v2_data/float32_padding.bin", paddingSize, padding, paddingSize);

    // 5. 分配输出GM内存
    size_t outputSize = 15 * sizeof(float);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputSize, 32));

    // 6. 分配工作空间和tiling GM内存
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    // 7. 执行核函数
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(reflection_pad1d_v2, tilingInfo.blockNum, x, padding, output, workspace, tiling);

    // 8. 保存输出结果
    WriteFile("./reflection_pad1d_v2_data/float32_output.bin", output, outputSize);

    // 9. 释放GM内存
    AscendC::GmFree(x);
    AscendC::GmFree(padding);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    // 10. 比较输出结果与黄金数据
    system("cd ./reflection_pad1d_v2_data/ && python3 compare_data.py 'float32'");
}

// 测试用例2：bf16类型 + 左2右3填充 + 1D输入[5]
TEST_F(ReflectionPad1dV2Test, test_case_bf16)
{
    // 1. 构造分块上下文参数
    optiling::ReflectionPad1dV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReflectionPad1dV2",
        {// 输入0：x（1D形状[5]，bf16类型，ND格式）
         {{{5}, {5}}, ge::DT_BF16, ge::FORMAT_ND},
         // 输入1：padding（1D形状[2]，int64类型，ND格式，对应[2,3]）
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND}},
        {// 输出0：形状[10]（5+2+3），bf16类型，ND格式
         {{{10}, {10}}, ge::DT_BF16, ge::FORMAT_ND}},
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    // 2. 调用Python脚本生成输入数据和黄金数据（参数：padLeft=2, padRight=3, wSize=5, dtype=bf16）
    system("cd ./reflection_pad1d_v2_data/ && python3 gen_data.py 2 3 5 'bf16'");

    // 3. 分配并读取输入x的GM内存
    size_t xSize = 5 * sizeof(bfloat16);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(xSize, 32));
    ReadFile("./reflection_pad1d_v2_data/bf16_x.bin", xSize, x, xSize);

    // 4. 分配并读取输入padding的GM内存
    size_t paddingSize = 2 * sizeof(int64_t);
    uint8_t* padding = (uint8_t*)AscendC::GmAlloc(CeilAlign(paddingSize, 32));
    ReadFile("./reflection_pad1d_v2_data/bf16_padding.bin", paddingSize, padding, paddingSize);

    // 5. 分配输出GM内存
    size_t outputSize = 10 * sizeof(bfloat16);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputSize, 32));

    // 6. 分配工作空间和tiling GM内存
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    // 7. 执行核函数
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(reflection_pad1d_v2, tilingInfo.blockNum, x, padding, output, workspace, tiling);

    // 8. 保存输出结果
    WriteFile("./reflection_pad1d_v2_data/bf16_output.bin", output, outputSize);

    // 9. 释放GM内存
    AscendC::GmFree(x);
    AscendC::GmFree(padding);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    // 10. 比较输出结果与黄金数据
    system("cd ./reflection_pad1d_v2_data/ && python3 compare_data.py 'bf16'");
}