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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/bitwise_not.cpp"

using namespace std;

extern "C" __global__ __aicore__ void bitwise_not(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

// 说明（op_kernel UT 的 dtype 约束）：
//   kernel UT 单个二进制由 CMakeLists 用 `-DDTYPE_X=int16_t` 编译，kernel 内 dtype 分支均为
//   `if constexpr (std::is_same_v<TYPE_X, ...>)`，因此本二进制**只实例化 int16 主路径**。
//   其余 dtype（int8/int32/int64/uint8/bool 含 BOOL 逻辑非链路）的逐元素数值正确性由 ST（NPU）与
//   op_host tiling UT（isBool 标志 / 多 dtype TilingKey 分支）覆盖；本 kernel UT 聚焦 int16 主路径
//   计算正确性 + 尾块 DataCopyPad 路径无回归。golden = numpy.invert（bitwise exact，atol=0/rtol=0）。

class BitwiseNotTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "bitwise_not_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./bitwise_not_data/");
    }
    static void TearDownTestCase()
    {
        std::cout << "bitwise_not_test TearDown" << std::endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string BitwiseNotTest::rootPath = "../../../../experimental/";
const std::string BitwiseNotTest::dataPath = rootPath + "math/bitwise_not/tests/ut/op_kernel/bitwise_not_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

// 单核、单 tile 的 tiling 配置助手（TilingKey 0，仅 small core）。
static void FillSingleCoreTiling(BitwiseNotTilingData* t, uint32_t dataCount, uint32_t isBool)
{
    t->smallCoreDataNum = dataCount;
    t->bigCoreDataNum = 0;
    t->bigCoreLoopNum = 0;
    t->smallCoreLoopNum = 1;
    t->ubPartDataNum = dataCount;
    t->smallCoreTailDataNum = dataCount;
    t->bigCoreTailDataNum = 0;
    t->tailBlockNum = 0;
    t->isBool = isBool;
    // 单核即 GM 序最后一个核，末核尾 tile 真实剩余元素数 = dataCount（无对齐填充 pad）。
    t->lastCoreId = 0;
    t->lastCoreTailDataNum = dataCount;
}

// int16 主路径单核执行体：跑 kernel，golden = numpy.invert，bitwise exact 比对。
// shapeStr 形如 "(1024)"；dataCount 为元素数（int16 元素）。
static void RunInt16SingleCore(const std::string& shapeStr, uint32_t dataCount)
{
    uint32_t blockDim = 1;
    const std::string genCmd = "cd ./bitwise_not_data/ && python3 gen_data.py '" + shapeStr + "' 'int16'";
    system(genCmd.c_str());

    size_t inputByteSize = static_cast<size_t>(dataCount) * sizeof(int16_t);
    size_t outputByteSize = static_cast<size_t>(dataCount) * sizeof(int16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));
    ReadFile("./bitwise_not_data/int16_input_bitwise_not.bin", inputByteSize, x, inputByteSize);

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(BitwiseNotTilingData));
    BitwiseNotTilingData* tilingData = reinterpret_cast<BitwiseNotTilingData*>(tiling);
    FillSingleCoreTiling(tilingData, dataCount, 0);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = bitwise_not<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    WriteFile("./bitwise_not_data/int16_output_bitwise_not.bin", y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    int ret = system("cd ./bitwise_not_data/ && python3 compare_data.py 'int16'");
    EXPECT_EQ(ret, 0) << "int16 bitwise-not compare FAILED for shape " << shapeStr;
}

// 整型 bitwise-not 主路径：32B 对齐大 tile（标准 elementwise 正确性）。
TEST_F(BitwiseNotTest, test_case_int16)
{
    RunInt16SingleCore("(1024)", 1024);
}

// 非 32B 对齐尾块（int16 一块=16 元素，17 非对齐）：单核单 tile，lastCoreTailDataNum=17，
// 走 DataCopyPad（blockLen=17*2 Byte）安全写回，验证尾块路径无回归。
TEST_F(BitwiseNotTest, test_case_int16_tail_17)
{
    RunInt16SingleCore("(17)", 17);
}

// 另一非对齐尾块（33 = 2*16+1）。
TEST_F(BitwiseNotTest, test_case_int16_tail_33)
{
    RunInt16SingleCore("(33)", 33);
}
