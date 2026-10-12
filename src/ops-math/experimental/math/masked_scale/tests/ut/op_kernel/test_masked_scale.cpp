// ----------------------------------------------------------------------------
// Copyright (c) Huawei Device Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include "tikicpulib.h"
#include "../../../op_kernel/arch35/masked_scale_tiling_data.h"
#include "../../../op_kernel/arch35/masked_scale_tiling_key.h"

using TestUtDefaultTilingStruct = MaskedScaleTilingData;

#include "../../../op_kernel/masked_scale_apt.cpp"

namespace {
void* GmAllocAlign(size_t size) { return GmAlloc((size + 31U) >> 5U << 5U); }

void FillFloat(void* gm, const std::vector<float>& data)
{
    auto* ptr = reinterpret_cast<float*>(gm);
    for (size_t i = 0; i < data.size(); ++i) {
        ptr[i] = data[i];
    }
}

void ExpectFloatNear(const void* gm, const std::vector<float>& expected)
{
    const auto* ptr = reinterpret_cast<const float*>(gm);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(ptr[i], expected[i], 1e-5F);
    }
}

void InitTiling(MaskedScaleTilingData* tilingData, uint32_t dim0, uint32_t ubFormer, float scale, uint32_t selfDtype,
                uint32_t maskDtype, uint32_t tilingKey)
{
    tilingData->Init();
    tilingData->dim0 = dim0;
    tilingData->coreNum = 1U;
    tilingData->blockFormer = dim0;
    tilingData->blockNum = 1U;
    tilingData->ubFormer = ubFormer;
    tilingData->selfDtype = selfDtype;
    tilingData->maskDtype = maskDtype;
    tilingData->branchKey = tilingKey;
    tilingData->subCaseKey = tilingKey;
    tilingData->scaleFloat = scale;
    tilingData->bufferNum = 1U;
}
} // namespace

class MaskedScaleKernelTest : public testing::Test {};

TEST_F(MaskedScaleKernelTest, fp32_one_core_single_loop)
{
    constexpr uint32_t elemCount = 128U;
    constexpr float scale = 0.5F;
    constexpr size_t dataSize = elemCount * sizeof(float);
    auto self = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto mask = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(16U * 1024U * 1024U));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(MaskedScaleTilingData)));

    std::vector<float> selfData(elemCount);
    std::vector<float> maskData(elemCount);
    std::vector<float> expected(elemCount);
    for (uint32_t i = 0U; i < elemCount; ++i) {
        selfData[i] = static_cast<float>(i) - 64.0F;
        maskData[i] = static_cast<float>((i % 5U) + 1U);
        expected[i] = selfData[i] * maskData[i] * scale;
    }
    FillFloat(self, selfData);
    FillFloat(mask, maskData);

    auto* tilingData = reinterpret_cast<MaskedScaleTilingData*>(tiling);
    InitTiling(tilingData, elemCount, elemCount, scale, MASKED_SCALE_TPL_FP32, MASKED_SCALE_TPL_FP32,
               MASKED_SCALE_KEY_FP32_FP32);

    ICPU_SET_TILING_KEY(MASKED_SCALE_KEY_FP32_FP32);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = masked_scale<MASKED_SCALE_TPL_FP32, MASKED_SCALE_TPL_FP32>;
    ICPU_RUN_KF(kernelFunc, 1, self, mask, y, workspace, tiling);
    ExpectFloatNear(y, expected);

    GmFree(self);
    GmFree(mask);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}

TEST_F(MaskedScaleKernelTest, fp32_one_core_multi_loop_tail)
{
    constexpr uint32_t elemCount = 130U;
    constexpr uint32_t ubFormer = 64U;
    constexpr float scale = -2.0F;
    constexpr size_t dataSize = elemCount * sizeof(float);
    auto self = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto mask = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(dataSize));
    auto workspace = static_cast<GM_ADDR>(GmAllocAlign(16U * 1024U * 1024U));
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(sizeof(MaskedScaleTilingData)));

    std::vector<float> selfData(elemCount);
    std::vector<float> maskData(elemCount);
    std::vector<float> expected(elemCount);
    for (uint32_t i = 0U; i < elemCount; ++i) {
        selfData[i] = static_cast<float>(i % 17U) + 0.25F;
        maskData[i] = (i % 2U == 0U) ? 1.0F : 0.25F;
        expected[i] = selfData[i] * maskData[i] * scale;
    }
    FillFloat(self, selfData);
    FillFloat(mask, maskData);

    auto* tilingData = reinterpret_cast<MaskedScaleTilingData*>(tiling);
    InitTiling(tilingData, elemCount, ubFormer, scale, MASKED_SCALE_TPL_FP32, MASKED_SCALE_TPL_FP32,
               MASKED_SCALE_KEY_FP32_FP32);

    ICPU_SET_TILING_KEY(MASKED_SCALE_KEY_FP32_FP32);
    SetKernelMode(KernelMode::AIV_MODE);
    auto kernelFunc = masked_scale<MASKED_SCALE_TPL_FP32, MASKED_SCALE_TPL_FP32>;
    ICPU_RUN_KF(kernelFunc, 1, self, mask, y, workspace, tiling);
    ExpectFloatNear(y, expected);

    GmFree(self);
    GmFree(mask);
    GmFree(y);
    GmFree(workspace);
    GmFree(tiling);
}
