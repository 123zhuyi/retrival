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
 * \file test_kl_div_v2.cpp
 * \brief KLDivV2 算子 kernel UT 测试
 *
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "../../../op_kernel/kl_div_v2_tiling_data.h"
#include "../../../op_kernel/kl_div_v2.cpp"

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

class KLDivV2KernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "KLDivV2KernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "KLDivV2KernelTest TearDown" << endl; }
};

TEST_F(KLDivV2KernelTest, test_kernel_run)
{
    constexpr size_t size = 15;
    constexpr size_t tilingDataSize = sizeof(KLDivV2TilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 15 * 2;
    constexpr size_t targetByteSize = 15 * 2;
    constexpr size_t yByteSize = 1 * 2;
    std::vector<float> xHost(15, 1);
    std::vector<float> targetHost(15, 1);
    std::vector<float> yHost(1, 0);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    for (size_t _i = 0; _i < 15; _i++) {
        uint16_t _h = FloatToHalf(xHost[_i]);
        memcpy(x + _i * 2, &_h, 2);
    }
    for (size_t _i = 0; _i < 15; _i++) {
        uint16_t _h = FloatToHalf(targetHost[_i]);
        memcpy(target + _i * 2, &_h, 2);
    }

    // 直接构造 tilingData（固定值，生成时确定）
    KLDivV2TilingData* tilingData = reinterpret_cast<KLDivV2TilingData*>(tiling);
    tilingData->smallCoreDataNum = 15;
    tilingData->bigCoreDataNum = 15;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = 15;
    tilingData->smallTailDataNum = 15;
    tilingData->bigTailDataNum = 15;
    tilingData->tailBlockNum = 0;
    tilingData->reduction = 1;
    tilingData->logTarget = 0;
    tilingData->cof = 1.0f / 15.0f;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF((kl_div_v2<0>), numBlocks, x, target, y, workspace, tiling);

    // 将动态输出的 packed buffer 拆回 individual buffers

    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    {
        std::ofstream _ofs("float16_output_kl_div_v2_0.bin", std::ios::binary);
        _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(target);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
