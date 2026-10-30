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

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "tikicpulib.h"
#include "data_utils.h"
#include "../op_kernel/masked_fill_tiling_data.h"

using namespace std;
using namespace NsMaskedFill;

// 声明核函数
extern "C" __global__ __aicore__ void masked_fill(
    GM_ADDR x, GM_ADDR mask, GM_ADDR value, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class MaskedFillKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // 复制测试数据目录
        system("cp -rf ./data ./masked_fill_data");
        system("chmod -R 755 ./masked_fill_data");
    }
    static void TearDownTestCase() { system("rm -rf ./masked_fill_data"); }

    // 内存对齐（32字节）
    template <typename T>
    size_t AlignSize(size_t size)
    {
        return (size + 31) / 32 * 32;
    }
};

// 测试1：float32类型，mask全为true（所有元素填充为value）
TEST_F(MaskedFillKernelTest, float32_mask_all_true)
{
    // 1. 分块配置
    MaskedFillCompileInfo compile_info;
    gert::TilingContextPara tiling_para(
        "MaskedFill",
        {
            {{{2, 3}, {2, 3}}, DT_FLOAT, FORMAT_ND}, // x: (2,3)
            {{{2, 3}, {2, 3}}, DT_BOOL, FORMAT_ND},  // mask: (2,3)（全true）
            {{{}, {}}, DT_FLOAT, FORMAT_ND}          // value: 10.0
        },
        {
            {{{2, 3}, {2, 3}}, DT_FLOAT, FORMAT_ND} // 输出：(2,3)
        },
        &compile_info);
    TilingInfo tiling_info;
    ASSERT_TRUE(ExecuteTiling(tiling_para, tiling_info));

    // 2. 生成测试数据（x: 1-6，mask: 全true，value:10.0）
    system("cd ./masked_fill_data && python3 gen_data.py float32 2 3 all_true 10.0");

    // 3. 分配GM内存
    size_t x_size = 2 * 3 * sizeof(float);
    size_t mask_size = 2 * 3 * sizeof(bool);
    size_t value_size = sizeof(float);
    size_t output_size = x_size;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(AlignSize(x_size));
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(AlignSize(mask_size));
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(AlignSize(value_size));
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(AlignSize(output_size));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tiling_info.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_info.tilingDataSize);

    // 4. 读取输入数据
    ReadFile("./masked_fill_data/float32_x.bin", x_size, x, AlignSize(x_size));
    ReadFile("./masked_fill_data/float32_mask.bin", mask_size, mask, AlignSize(mask_size));
    ReadFile("./masked_fill_data/float32_value.bin", value_size, value, AlignSize(value_size));
    memcpy(tiling, tiling_info.tilingData.get(), tiling_info.tilingDataSize);

    // 5. 执行核函数
    ICPU_SET_TILING_KEY(tiling_info.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(masked_fill, tiling_info.blockNum, x, mask, value, output, workspace, tiling);

    // 6. 保存输出并比较
    WriteFile("./masked_fill_data/float32_output.bin", output, output_size);
    int ret = system("cd ./masked_fill_data && python3 compare_data.py float32");
    ASSERT_EQ(ret, 0); // 比较成功返回0

    // 7. 释放内存
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(value);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 测试2：int32类型，mask全为false（输出与x一致）
TEST_F(MaskedFillKernelTest, int32_mask_all_false)
{
    // 分块配置
    MaskedFillCompileInfo compile_info;
    gert::TilingContextPara tiling_para(
        "MaskedFill",
        {
            {{{3}, {3}}, DT_INT32, FORMAT_ND}, // x: (3)
            {{{3}, {3}}, DT_BOOL, FORMAT_ND},  // mask: (3)（全false）
            {{{}, {}}, DT_INT32, FORMAT_ND}    // value: 100
        },
        {
            {{{3}, {3}}, DT_INT32, FORMAT_ND} // 输出：(3)
        },
        &compile_info);
    TilingInfo tiling_info;
    ASSERT_TRUE(ExecuteTiling(tiling_para, tiling_info));

    // 生成数据（x: 1,2,3；mask: 全false；value:100）
    system("cd ./masked_fill_data && python3 gen_data.py int32 3 1 all_false 100");

    // 分配内存
    size_t x_size = 3 * sizeof(int32_t);
    size_t mask_size = 3 * sizeof(bool);
    size_t value_size = sizeof(int32_t);
    size_t output_size = x_size;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(AlignSize(x_size));
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(AlignSize(mask_size));
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(AlignSize(value_size));
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(AlignSize(output_size));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tiling_info.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_info.tilingDataSize);

    // 读取数据
    ReadFile("./masked_fill_data/int32_x.bin", x_size, x, AlignSize(x_size));
    ReadFile("./masked_fill_data/int32_mask.bin", mask_size, mask, AlignSize(mask_size));
    ReadFile("./masked_fill_data/int32_value.bin", value_size, value, AlignSize(value_size));
    memcpy(tiling, tiling_info.tilingData.get(), tiling_info.tilingDataSize);

    // 执行核函数
    ICPU_SET_TILING_KEY(tiling_info.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(masked_fill, tiling_info.blockNum, x, mask, value, output, workspace, tiling);

    // 比较结果
    WriteFile("./masked_fill_data/int32_output.bin", output, output_size);
    int ret = system("cd ./masked_fill_data && python3 compare_data.py int32");
    ASSERT_EQ(ret, 0);

    // 释放内存
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(value);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 测试3：bfloat16类型，mask部分true（广播场景：mask(1,3) → x(2,3)）
TEST_F(MaskedFillKernelTest, bfloat16_broadcast_mask)
{
    // 分块配置
    MaskedFillCompileInfo compile_info;
    gert::TilingContextPara tiling_para(
        "MaskedFill",
        {
            {{{2, 3}, {2, 3}}, DT_BF16, FORMAT_ND}, // x: (2,3)
            {{{1, 3}, {1, 3}}, DT_BOOL, FORMAT_ND}, // mask: (1,3)（广播）
            {{{}, {}}, DT_BF16, FORMAT_ND}          // value: 5.0（bfloat16）
        },
        {
            {{{2, 3}, {2, 3}}, DT_BF16, FORMAT_ND} // 输出：(2,3)
        },
        &compile_info);
    TilingInfo tiling_info;
    ASSERT_TRUE(ExecuteTiling(tiling_para, tiling_info));

    // 生成数据（x: 1-6；mask: [true, false, true]；value:5.0）
    system("cd ./masked_fill_data && python3 gen_data.py bfloat16 2 3 broadcast 5.0");

    // 分配内存（bfloat16用uint16_t存储）
    size_t x_size = 2 * 3 * sizeof(uint16_t);
    size_t mask_size = 1 * 3 * sizeof(bool);
    size_t value_size = sizeof(uint16_t);
    size_t output_size = x_size;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(AlignSize(x_size));
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(AlignSize(mask_size));
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(AlignSize(value_size));
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(AlignSize(output_size));
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tiling_info.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_info.tilingDataSize);

    // 读取数据
    ReadFile("./masked_fill_data/bfloat16_x.bin", x_size, x, AlignSize(x_size));
    ReadFile("./masked_fill_data/bfloat16_mask.bin", mask_size, mask, AlignSize(mask_size));
    ReadFile("./masked_fill_data/bfloat16_value.bin", value_size, value, AlignSize(value_size));
    memcpy(tiling, tiling_info.tilingData.get(), tiling_info.tilingDataSize);

    // 执行核函数
    ICPU_SET_TILING_KEY(tiling_info.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(masked_fill, tiling_info.blockNum, x, mask, value, output, workspace, tiling);

    // 比较结果
    WriteFile("./masked_fill_data/bfloat16_output.bin", output, output_size);
    int ret = system("cd ./masked_fill_data && python3 compare_data.py bfloat16");
    ASSERT_EQ(ret, 0);

    // 释放内存
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(value);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}