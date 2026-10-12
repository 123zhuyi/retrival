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

#include "../../../op_kernel/trilu.cpp"

using namespace std;

constexpr uint32_t BLOCK_DIM = 1;

extern "C" __global__ __aicore__ void trilu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

class TriluKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TriluKernelTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "TriluKernelTest TearDown" << std::endl; }
};

TEST_F(TriluKernelTest, test_case_triu_int32)
{
    constexpr int64_t H = 4;
    constexpr int64_t W = 4;
    constexpr uint32_t dataCount = H * W;
    size_t byteSize = dataCount * sizeof(int32_t);

    int32_t xHostData[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int32_t expectData[16] = {1, 2, 3, 4, 0, 6, 7, 8, 0, 0, 11, 12, 0, 0, 0, 16};

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    memcpy(x, xHostData, byteSize);

    uint8_t* tilingBuf = (uint8_t*)AscendC::GmAlloc(sizeof(TriluTilingData));
    TriluTilingData* tilingData = reinterpret_cast<TriluTilingData*>(tilingBuf);
    memset(tilingData, 0, sizeof(TriluTilingData));
    tilingData->totalElements = dataCount;
    tilingData->perCoreElements = dataCount;
    tilingData->needCoreNum = 1;
    tilingData->lastCoreElements = dataCount;
    tilingData->diagonal = 0;
    tilingData->upper = 1;
    tilingData->h = H;
    tilingData->w = W;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = trilu<TRILU_TPL_SCH_MODE_2>;
    ICPU_RUN_KF(func, BLOCK_DIM, x, y, nullptr, (uint8_t*)(tilingData));

    int32_t* result = reinterpret_cast<int32_t*>(y);
    for (uint32_t i = 0; i < dataCount; i++) {
        EXPECT_EQ(result[i], expectData[i]) << "Mismatch at index " << i;
    }

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)tilingBuf);
}

TEST_F(TriluKernelTest, test_case_tril_int32_diagonal1)
{
    constexpr int64_t H = 4;
    constexpr int64_t W = 4;
    constexpr uint32_t dataCount = H * W;
    size_t byteSize = dataCount * sizeof(int32_t);

    int32_t xHostData[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int32_t expectData[16] = {1, 2, 0, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 16};

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    memcpy(x, xHostData, byteSize);

    uint8_t* tilingBuf = (uint8_t*)AscendC::GmAlloc(sizeof(TriluTilingData));
    TriluTilingData* tilingData = reinterpret_cast<TriluTilingData*>(tilingBuf);
    memset(tilingData, 0, sizeof(TriluTilingData));
    tilingData->totalElements = dataCount;
    tilingData->perCoreElements = dataCount;
    tilingData->needCoreNum = 1;
    tilingData->lastCoreElements = dataCount;
    tilingData->diagonal = 1;
    tilingData->upper = 0;
    tilingData->h = H;
    tilingData->w = W;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = trilu<TRILU_TPL_SCH_MODE_2>;
    ICPU_RUN_KF(func, BLOCK_DIM, x, y, nullptr, (uint8_t*)(tilingData));

    int32_t* result = reinterpret_cast<int32_t*>(y);
    for (uint32_t i = 0; i < dataCount; i++) {
        EXPECT_EQ(result[i], expectData[i]) << "Mismatch at index " << i;
    }

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)tilingBuf);
}
