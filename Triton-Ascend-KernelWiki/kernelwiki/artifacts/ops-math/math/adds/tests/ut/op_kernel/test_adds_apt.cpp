/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/arch35/adds_tiling_data.h"
#include "../../../op_kernel/arch35/adds_struct.h"
#include "adds_kernel_ut_adapter.h"

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr int64_t kElementCount = 256;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}
} // namespace

using namespace AddsOp;

#include "../../../op_kernel/adds_apt.cpp"

class AddsKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddsKernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddsKernelTest TearDown" << std::endl;
    }
};

TEST_F(AddsKernelTest, test_fp32_basic)
{
    size_t inputByteSize = Align32(kElementCount * sizeof(float));
    size_t outputByteSize = Align32(kElementCount * sizeof(float));
    size_t tilingSize = Align32(sizeof(AddsTilingData));

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(Align32(1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto* xData = reinterpret_cast<float*>(x);
    for (int64_t i = 0; i < kElementCount; ++i) {
        xData[i] = static_cast<float>(i % 100) * 0.01f;
    }
    std::memset(y, 0, outputByteSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AddsTilingData*>(tiling);
    tilingData->baseTiling.blockNum = kNumBlocks;
    tilingData->baseTiling.scheMode = 0;
    tilingData->scalarValue = 1.5f;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF((adds<0, 3>), kNumBlocks, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AddsKernelTest, test_fp16_basic)
{
    size_t inputByteSize = Align32(kElementCount * sizeof(half));
    size_t outputByteSize = Align32(kElementCount * sizeof(half));
    size_t tilingSize = Align32(sizeof(AddsTilingData));

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(Align32(1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto* xData = reinterpret_cast<half*>(x);
    for (int64_t i = 0; i < kElementCount; ++i) {
        xData[i] = static_cast<half>(static_cast<float>(i % 100) * 0.01f);
    }
    std::memset(y, 0, outputByteSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AddsTilingData*>(tiling);
    tilingData->baseTiling.blockNum = kNumBlocks;
    tilingData->baseTiling.scheMode = 0;
    tilingData->scalarValue = 2.0f;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF((adds<0, 1>), kNumBlocks, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AddsKernelTest, test_bf16_basic)
{
    size_t inputByteSize = Align32(kElementCount * sizeof(bfloat16_t));
    size_t outputByteSize = Align32(kElementCount * sizeof(bfloat16_t));
    size_t tilingSize = Align32(sizeof(AddsTilingData));

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(Align32(1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    std::memset(y, 0, outputByteSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AddsTilingData*>(tiling);
    tilingData->baseTiling.blockNum = kNumBlocks;
    tilingData->baseTiling.scheMode = 0;
    tilingData->scalarValue = 0.5f;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF((adds<0, 2>), kNumBlocks, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AddsKernelTest, test_int32_basic)
{
    size_t inputByteSize = Align32(kElementCount * sizeof(int32_t));
    size_t outputByteSize = Align32(kElementCount * sizeof(int32_t));
    size_t tilingSize = Align32(sizeof(AddsTilingData));

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(Align32(1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto* xData = reinterpret_cast<int32_t*>(x);
    for (int64_t i = 0; i < kElementCount; ++i) {
        xData[i] = i % 100;
    }
    std::memset(y, 0, outputByteSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AddsTilingData*>(tiling);
    tilingData->baseTiling.blockNum = kNumBlocks;
    tilingData->baseTiling.scheMode = 0;
    tilingData->scalarValue = 10.0f;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF((adds<0, 5>), kNumBlocks, x, y, workspace, tiling);


    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AddsKernelTest, test_int16_basic)
{
    size_t inputByteSize = Align32(kElementCount * sizeof(int16_t));
    size_t outputByteSize = Align32(kElementCount * sizeof(int16_t));
    size_t tilingSize = Align32(sizeof(AddsTilingData));

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(Align32(1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto* xData = reinterpret_cast<int16_t*>(x);
    for (int64_t i = 0; i < kElementCount; ++i) {
        xData[i] = static_cast<int16_t>(i % 100);
    }
    std::memset(y, 0, outputByteSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AddsTilingData*>(tiling);
    tilingData->baseTiling.blockNum = kNumBlocks;
    tilingData->baseTiling.scheMode = 0;
    tilingData->scalarValue = 5.0f;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF((adds<0, 4>), kNumBlocks, x, y, workspace, tiling);


    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AddsKernelTest, test_int64_basic)
{
    size_t inputByteSize = Align32(kElementCount * sizeof(int64_t));
    size_t outputByteSize = Align32(kElementCount * sizeof(int64_t));
    size_t tilingSize = Align32(sizeof(AddsTilingData));

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(Align32(1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto* xData = reinterpret_cast<int64_t*>(x);
    for (int64_t i = 0; i < kElementCount; ++i) {
        xData[i] = static_cast<int64_t>(i % 100);
    }
    std::memset(y, 0, outputByteSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AddsTilingData*>(tiling);
    tilingData->baseTiling.blockNum = kNumBlocks;
    tilingData->baseTiling.scheMode = 0;
    tilingData->scalarValue = 20.0f;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(6);
    ICPU_RUN_KF((adds<0, 6>), kNumBlocks, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}