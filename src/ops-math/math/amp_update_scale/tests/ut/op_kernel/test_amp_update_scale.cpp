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
#include "../../../op_host/amp_update_scale_tiling.h"

namespace {
constexpr uint32_t kNumBlocks = 1;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}
} // namespace

using namespace optiling;

#include "../../../op_kernel/amp_update_scale.cpp"

class AmpUpdateScaleKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AmpUpdateScaleKernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AmpUpdateScaleKernelTest TearDown" << std::endl;
    }
};

TEST_F(AmpUpdateScaleKernelTest, test_fp32_found_inf)
{
    size_t elementByteSize = Align32(sizeof(float));
    size_t growthByteSize = Align32(sizeof(int32_t));
    size_t tilingSize = Align32(sizeof(AmpUpdateScaleTilingData));
    size_t workspaceSize = Align32(1024);

    uint8_t* currentScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* growthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* foundInf = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedGrowthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    *reinterpret_cast<float*>(currentScale) = 65536.0f;
    *reinterpret_cast<int32_t*>(growthTracker) = 3;
    *reinterpret_cast<float*>(foundInf) = 1.0f;
    std::memset(updatedScale, 0, elementByteSize);
    std::memset(updatedGrowthTracker, 0, growthByteSize);
    std::memset(workspace, 0, workspaceSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AmpUpdateScaleTilingData*>(tiling);
    tilingData->set_growthFactor(2.0f);
    tilingData->set_backoffFactor(0.5f);
    tilingData->set_growthInterval(5);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(amp_update_scale, kNumBlocks, currentScale, growthTracker, foundInf,
                updatedScale, updatedGrowthTracker, workspace, tiling);

    EXPECT_EQ(*reinterpret_cast<float*>(updatedScale), 32768.0f);
    EXPECT_EQ(*reinterpret_cast<int32_t*>(updatedGrowthTracker), 0);

    AscendC::GmFree(currentScale);
    AscendC::GmFree(growthTracker);
    AscendC::GmFree(foundInf);
    AscendC::GmFree(updatedScale);
    AscendC::GmFree(updatedGrowthTracker);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AmpUpdateScaleKernelTest, test_fp32_growth_triggered)
{
    size_t elementByteSize = Align32(sizeof(float));
    size_t growthByteSize = Align32(sizeof(int32_t));
    size_t tilingSize = Align32(sizeof(AmpUpdateScaleTilingData));
    size_t workspaceSize = Align32(1024);

    uint8_t* currentScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* growthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* foundInf = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedGrowthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    *reinterpret_cast<float*>(currentScale) = 100.0f;
    *reinterpret_cast<int32_t*>(growthTracker) = 4;
    *reinterpret_cast<float*>(foundInf) = 0.0f;
    std::memset(updatedScale, 0, elementByteSize);
    std::memset(updatedGrowthTracker, 0, growthByteSize);
    std::memset(workspace, 0, workspaceSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AmpUpdateScaleTilingData*>(tiling);
    tilingData->set_growthFactor(2.0f);
    tilingData->set_backoffFactor(0.5f);
    tilingData->set_growthInterval(5);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(amp_update_scale, kNumBlocks, currentScale, growthTracker, foundInf,
                updatedScale, updatedGrowthTracker, workspace, tiling);

    EXPECT_EQ(*reinterpret_cast<float*>(updatedScale), 200.0f);
    EXPECT_EQ(*reinterpret_cast<int32_t*>(updatedGrowthTracker), 0);

    AscendC::GmFree(currentScale);
    AscendC::GmFree(growthTracker);
    AscendC::GmFree(foundInf);
    AscendC::GmFree(updatedScale);
    AscendC::GmFree(updatedGrowthTracker);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AmpUpdateScaleKernelTest, test_fp32_no_growth)
{
    size_t elementByteSize = Align32(sizeof(float));
    size_t growthByteSize = Align32(sizeof(int32_t));
    size_t tilingSize = Align32(sizeof(AmpUpdateScaleTilingData));
    size_t workspaceSize = Align32(1024);

    uint8_t* currentScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* growthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* foundInf = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedGrowthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    *reinterpret_cast<float*>(currentScale) = 100.0f;
    *reinterpret_cast<int32_t*>(growthTracker) = 2;
    *reinterpret_cast<float*>(foundInf) = 0.0f;
    std::memset(updatedScale, 0, elementByteSize);
    std::memset(updatedGrowthTracker, 0, growthByteSize);
    std::memset(workspace, 0, workspaceSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AmpUpdateScaleTilingData*>(tiling);
    tilingData->set_growthFactor(2.0f);
    tilingData->set_backoffFactor(0.5f);
    tilingData->set_growthInterval(5);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(amp_update_scale, kNumBlocks, currentScale, growthTracker, foundInf,
                updatedScale, updatedGrowthTracker, workspace, tiling);

    EXPECT_EQ(*reinterpret_cast<float*>(updatedScale), 100.0f);
    EXPECT_EQ(*reinterpret_cast<int32_t*>(updatedGrowthTracker), 3);

    AscendC::GmFree(currentScale);
    AscendC::GmFree(growthTracker);
    AscendC::GmFree(foundInf);
    AscendC::GmFree(updatedScale);
    AscendC::GmFree(updatedGrowthTracker);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AmpUpdateScaleKernelTest, test_fp32_scale_inf_check)
{
    size_t elementByteSize = Align32(sizeof(float));
    size_t growthByteSize = Align32(sizeof(int32_t));
    size_t tilingSize = Align32(sizeof(AmpUpdateScaleTilingData));
    size_t workspaceSize = Align32(1024);

    uint8_t* currentScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* growthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* foundInf = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedGrowthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    *reinterpret_cast<float*>(currentScale) = 3.4e38f;
    *reinterpret_cast<int32_t*>(growthTracker) = 4;
    *reinterpret_cast<float*>(foundInf) = 0.0f;
    std::memset(updatedScale, 0, elementByteSize);
    std::memset(updatedGrowthTracker, 0, growthByteSize);
    std::memset(workspace, 0, workspaceSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AmpUpdateScaleTilingData*>(tiling);
    tilingData->set_growthFactor(2.0f);
    tilingData->set_backoffFactor(0.5f);
    tilingData->set_growthInterval(5);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(amp_update_scale, kNumBlocks, currentScale, growthTracker, foundInf,
                updatedScale, updatedGrowthTracker, workspace, tiling);

    EXPECT_EQ(*reinterpret_cast<float*>(updatedScale), 3.4e38f);
    EXPECT_EQ(*reinterpret_cast<int32_t*>(updatedGrowthTracker), 4);

    AscendC::GmFree(currentScale);
    AscendC::GmFree(growthTracker);
    AscendC::GmFree(foundInf);
    AscendC::GmFree(updatedScale);
    AscendC::GmFree(updatedGrowthTracker);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AmpUpdateScaleKernelTest, test_fp16_found_inf)
{
    size_t elementByteSize = Align32(sizeof(half));
    size_t growthByteSize = Align32(sizeof(int32_t));
    size_t tilingSize = Align32(sizeof(AmpUpdateScaleTilingData));
    size_t workspaceSize = Align32(1024);

    uint8_t* currentScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* growthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* foundInf = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedGrowthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    *reinterpret_cast<half*>(currentScale) = static_cast<half>(100.0f);
    *reinterpret_cast<int32_t*>(growthTracker) = 4;
    *reinterpret_cast<half*>(foundInf) = static_cast<half>(0.0f);
    std::memset(updatedScale, 0, elementByteSize);
    std::memset(updatedGrowthTracker, 0, growthByteSize);
    std::memset(workspace, 0, workspaceSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AmpUpdateScaleTilingData*>(tiling);
    tilingData->set_growthFactor(2.0f);
    tilingData->set_backoffFactor(0.5f);
    tilingData->set_growthInterval(5);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(amp_update_scale, kNumBlocks, currentScale, growthTracker, foundInf,
                updatedScale, updatedGrowthTracker, workspace, tiling);

    EXPECT_EQ(static_cast<float>(*reinterpret_cast<half*>(updatedScale)), 200.0f);
    EXPECT_EQ(*reinterpret_cast<int32_t*>(updatedGrowthTracker), 0);

    AscendC::GmFree(currentScale);
    AscendC::GmFree(growthTracker);
    AscendC::GmFree(foundInf);
    AscendC::GmFree(updatedScale);
    AscendC::GmFree(updatedGrowthTracker);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AmpUpdateScaleKernelTest, test_bf16_basic)
{
    size_t elementByteSize = Align32(sizeof(bfloat16_t));
    size_t growthByteSize = Align32(sizeof(int32_t));
    size_t tilingSize = Align32(sizeof(AmpUpdateScaleTilingData));
    size_t workspaceSize = Align32(1024);

    uint8_t* currentScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* growthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* foundInf = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedScale = (uint8_t*)AscendC::GmAlloc(elementByteSize);
    uint8_t* updatedGrowthTracker = (uint8_t*)AscendC::GmAlloc(growthByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    *reinterpret_cast<bfloat16_t*>(currentScale) = static_cast<bfloat16_t>(100.0f);
    *reinterpret_cast<int32_t*>(growthTracker) = 4;
    *reinterpret_cast<bfloat16_t*>(foundInf) = static_cast<bfloat16_t>(0.0f);
    std::memset(updatedScale, 0, elementByteSize);
    std::memset(updatedGrowthTracker, 0, growthByteSize);
    std::memset(workspace, 0, workspaceSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<AmpUpdateScaleTilingData*>(tiling);
    tilingData->set_growthFactor(2.0f);
    tilingData->set_backoffFactor(0.5f);
    tilingData->set_growthInterval(5);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(amp_update_scale, kNumBlocks, currentScale, growthTracker, foundInf,
                updatedScale, updatedGrowthTracker, workspace, tiling);

    EXPECT_EQ(static_cast<float>(*reinterpret_cast<bfloat16_t*>(updatedScale)), 200.0f);
    EXPECT_EQ(*reinterpret_cast<int32_t*>(updatedGrowthTracker), 0);

    AscendC::GmFree(currentScale);
    AscendC::GmFree(growthTracker);
    AscendC::GmFree(foundInf);
    AscendC::GmFree(updatedScale);
    AscendC::GmFree(updatedGrowthTracker);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}