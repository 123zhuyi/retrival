/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_assign_sub.cpp
 * \brief AssignSub 算子 kernel UT 测试
 *
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "../../../op_kernel/assign_sub_tiling_data.h"
#include "../../../op_kernel/assign_sub.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0)
        return sign;
    if (exp >= 31)
        return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

class AssignSubKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "AssignSubKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "AssignSubKernelTest TearDown" << endl; }
};

TEST_F(AssignSubKernelTest, test_kernel_run)
{
    constexpr size_t size = 945;
    constexpr size_t tilingDataSize = sizeof(AssignSubTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t varByteSize = 945 * 1;
    constexpr size_t valueByteSize = 945 * 1;
    constexpr size_t var_outByteSize = 945 * 1;
    std::vector<int8_t> varHost(945, 1);
    std::vector<int8_t> valueHost(945, 1);
    std::vector<int8_t> var_outHost(945, 0);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(varByteSize);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(valueByteSize);
    uint8_t* var_out = (uint8_t*)AscendC::GmAlloc(var_outByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    ASSERT_NE(var, nullptr);
    ASSERT_NE(value, nullptr);
    ASSERT_NE(var_out, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    memcpy(var, varHost.data(), varByteSize);
    memcpy(value, valueHost.data(), valueByteSize);

    // 直接构造 tilingData（固定值，生成时确定）
    AssignSubTilingData* tilingData = reinterpret_cast<AssignSubTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF((assign_sub<1>), numBlocks, var, value, var_out, workspace, tiling);

    // 将动态输出的 packed buffer 拆回 individual buffers

    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(var_outHost.data(), var_out, var_outByteSize);
    {
        std::ofstream _ofs("int8_output_assign_sub_0.bin", std::ios::binary);
        _ofs.write(reinterpret_cast<const char*>(var_outHost.data()), var_outByteSize);
    }

    AscendC::GmFree(var);
    AscendC::GmFree(value);
    AscendC::GmFree(var_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
