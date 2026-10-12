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
 * \file test_squared_difference.cpp
 * \brief SquaredDifference 算子 kernel UT 测试
 *
 * 使用统一 ops-math tikicpulib UT 构建，直接构造 tilingData。
 */

#include "squared_difference_tiling.h"
#include "../../../op_kernel/squared_difference.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToBFloat16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

static void InitOneDimTiling(SquaredDifferenceTilingData* tiling, int64_t length, int32_t dtypeKey,
                             int64_t blockNum = 1)
{
    std::memset(tiling, 0, sizeof(*tiling));
    tiling->mode = SD_MODE_ONEDIM;
    tiling->dtypeKey = dtypeKey;
    tiling->shapeLen = 1;
    tiling->outDims[0] = length;
    tiling->x1Strides[0] = 1;
    tiling->x2Strides[0] = 1;
    tiling->outStrides[0] = 1;
    tiling->totalLength = length;
    tiling->ubFormer = length;
    tiling->ubOuter = 1;
    tiling->ubTail = length;
    tiling->innerDim = 1;
    tiling->alignInner = 1;
    tiling->maxTileElem = length;
    tiling->blockNum = blockNum;
    tiling->blockBase = 1;
    tiling->blockRemainder = 0;
    tiling->fusedProduct = 1;
}

static void InitBrcTiling(SquaredDifferenceTilingData* tiling)
{
    std::memset(tiling, 0, sizeof(*tiling));
    tiling->mode = SD_MODE_BRC;
    tiling->dtypeKey = SD_DT_FP32;
    tiling->shapeLen = 2;
    tiling->ubSplitAxis = 0;
    tiling->outDims[0] = 2;
    tiling->outDims[1] = 3;
    tiling->x1Strides[0] = 3;
    tiling->x1Strides[1] = 1;
    tiling->x2Strides[0] = 0;
    tiling->x2Strides[1] = 1;
    tiling->outStrides[0] = 3;
    tiling->outStrides[1] = 1;
    tiling->totalLength = 6;
    tiling->ubFormer = 2;
    tiling->ubOuter = 1;
    tiling->ubTail = 2;
    tiling->innerDim = 3;
    // BRC rows are padded to a 32-byte boundary in the local tensor.
    tiling->alignInner = 8;
    tiling->maxTileElem = 16;
    tiling->nFormer = 3;
    tiling->nOuter = 1;
    tiling->nTail = 3;
    tiling->blockNum = 1;
    tiling->blockBase = 1;
    tiling->blockRemainder = 0;
    tiling->fusedProduct = 1;
}

class SquaredDifferenceKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SquaredDifferenceKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "SquaredDifferenceKernelTest TearDown" << endl; }
};

TEST_F(SquaredDifferenceKernelTest, test_kernel_run)
{
    constexpr size_t size = 19;
    constexpr size_t tilingDataSize = sizeof(SquaredDifferenceTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t x1ByteSize = 19 * 2;
    constexpr size_t x2ByteSize = 19 * 2;
    constexpr size_t yByteSize = 19 * 2;
    std::vector<float> x1Host(size, 3.0F);
    std::vector<float> x2Host(size, 1.0F);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    for (size_t i = 0; i < size; i++) {
        uint16_t x1Value = FloatToBFloat16(x1Host[i]);
        uint16_t x2Value = FloatToBFloat16(x2Host[i]);
        memcpy(x1 + i * sizeof(uint16_t), &x1Value, sizeof(x1Value));
        memcpy(x2 + i * sizeof(uint16_t), &x2Value, sizeof(x2Value));
    }

    SquaredDifferenceTilingData* tilingData = reinterpret_cast<SquaredDifferenceTilingData*>(tiling);
    InitOneDimTiling(tilingData, size, SD_DT_BF16);

    ICPU_SET_TILING_KEY(SD_KEY_BF16_ONEDIM);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF((squared_difference<SD_KEY_BF16_ONEDIM>), numBlocks, x1, x2, y, workspace, tiling);

    std::vector<uint16_t> output(size, 0);
    memcpy(output.data(), y, yByteSize);
    const uint16_t expected = FloatToBFloat16(4.0F);
    for (uint16_t value : output) {
        EXPECT_EQ(value, expected);
    }

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SquaredDifferenceKernelTest, test_kernel_broadcast_fp32)
{
    constexpr size_t inputSize = 6;
    constexpr size_t outputByteSize = inputSize * sizeof(float);
    constexpr uint32_t numBlocks = 1;

    std::array<float, inputSize> x1Host = {3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::array<float, 3> x2Host = {1.0F, 2.0F, 3.0F};
    uint8_t* x1 = static_cast<uint8_t*>(AscendC::GmAlloc(outputByteSize));
    uint8_t* x2 = static_cast<uint8_t*>(AscendC::GmAlloc(3 * sizeof(float)));
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(outputByteSize));
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SquaredDifferenceTilingData)));
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);
    memcpy(x1, x1Host.data(), outputByteSize);
    memcpy(x2, x2Host.data(), 3 * sizeof(float));
    auto* tilingData = reinterpret_cast<SquaredDifferenceTilingData*>(tiling);
    InitBrcTiling(tilingData);

    ICPU_SET_TILING_KEY(SD_KEY_FP32_BRC);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((squared_difference<SD_KEY_FP32_BRC>), numBlocks, x1, x2, y, workspace, tiling);

    std::array<float, inputSize> output{};
    memcpy(output.data(), y, outputByteSize);
    const std::array<float, inputSize> expected = {4.0F, 4.0F, 4.0F, 25.0F, 25.0F, 25.0F};
    for (size_t i = 0; i < inputSize; i++) {
        EXPECT_FLOAT_EQ(output[i], expected[i]);
    }

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
