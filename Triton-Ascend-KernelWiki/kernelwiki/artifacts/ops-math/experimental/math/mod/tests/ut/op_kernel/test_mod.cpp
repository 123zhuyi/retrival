/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "../../../op_kernel/mod_tiling_data.h"

using TestUtDefaultTilingStruct = ModNs::ModTilingData;

#include "../../../op_kernel/mod.cpp"

namespace {
void* GmAllocAlign(size_t size)
{
    return GmAlloc((size + 31) >> 5 << 5);
}

void FillFloat(void* gm, const std::vector<float>& data)
{
    auto* ptr = reinterpret_cast<float*>(gm);
    for (size_t i = 0; i < data.size(); ++i) {
        ptr[i] = data[i];
    }
}

} // namespace

class ModKernelTest : public testing::Test {};

TEST_F(ModKernelTest, float32_same_shape_one_core)
{
    constexpr size_t elemCount = 4096;
    constexpr size_t dataSize = elemCount * sizeof(float);
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    std::vector<float> self(elemCount);
    std::vector<float> other(elemCount);
    for (size_t i = 0; i < elemCount; ++i) {
        self[i] = static_cast<float>(i + 3);
        other[i] = static_cast<float>((i % 7) + 2);
    }
    FillFloat(x1, self);
    FillFloat(x2, other);

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = elemCount;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = true;
    tilingData->dimNum = 1;
    tilingData->input1Shape[0] = elemCount;
    tilingData->input2Shape[0] = elemCount;
    tilingData->input2Stride[0] = 1;
    for (uint32_t i = 1; i < 8; ++i) {
        tilingData->input1Shape[i] = 1;
        tilingData->input2Shape[i] = 1;
        tilingData->input2Stride[i] = 0;
    }

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<30, 30, 30>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);
    SUCCEED();

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

TEST_F(ModKernelTest, float32_same_shape_small_tail)
{
    constexpr size_t elemCount = 16;
    constexpr size_t dataSize = elemCount * sizeof(float);
    auto x1 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto x2 = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(32));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(ModNs::ModTilingData)));

    std::vector<float> self = {5.5f, -11.51f, 36.23f, 7.0f, -10.0f, -8.0f, -15.0f, -7.0f,
                               10.0f, 8.0f, 15.0f, 7.0f, -10.0f, -8.0f, -15.0f, -7.0f};
    std::vector<float> other = {2.0f, 3.0f, -24.1f, 2.0f, 3.0f, 5.0f, 4.0f, 2.0f,
                                -3.0f, -5.0f, -4.0f, -2.0f, -3.0f, -5.0f, -4.0f, -2.0f};
    FillFloat(x1, self);
    FillFloat(x2, other);

    auto* tilingData = reinterpret_cast<ModNs::ModTilingData*>(tiling);
    tilingData->usableUbSize = 3648;
    tilingData->needCoreNum = 1;
    tilingData->totalDataCount = elemCount;
    tilingData->perCoreDataCount = 0;
    tilingData->tailDataCoreNum = 0;
    tilingData->lastCoreDataCount = elemCount;
    tilingData->isInput2Scalar = false;
    tilingData->isInput2SameShape = true;
    tilingData->dimNum = 2;
    tilingData->input1Shape[0] = 4;
    tilingData->input1Shape[1] = 4;
    tilingData->input2Shape[0] = 4;
    tilingData->input2Shape[1] = 4;
    tilingData->input2Stride[0] = 4;
    tilingData->input2Stride[1] = 1;
    for (uint32_t i = 2; i < 8; ++i) {
        tilingData->input1Shape[i] = 1;
        tilingData->input2Shape[i] = 1;
        tilingData->input2Stride[i] = 0;
    }

    ICPU_SET_TILING_KEY(30);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = &mod<30, 30, 30>;
    ICPU_RUN_KF(kernelFunc, 1, x1, x2, y, workspace, tiling);
    SUCCEED();

    GmFree(x1);
    GmFree(x2);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}
