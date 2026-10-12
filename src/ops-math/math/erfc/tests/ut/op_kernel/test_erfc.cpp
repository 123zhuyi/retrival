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
 * \file test_erfc.cpp
 * \brief kernel UT for Erfc operator
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "atvoss/elewise/elewise_base_struct.h"

#ifndef DTYPE_X
#define DTYPE_X float
#endif

#include "../../../op_kernel/arch35/erfc.cpp"

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr int64_t kElementCount = 256;

inline size_t Align32(size_t size) { return (size + 31U) / 32U * 32U; }
} // namespace

class ErfcKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ErfcKernelTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ErfcKernelTest TearDown" << std::endl; }
};

TEST_F(ErfcKernelTest, test_fp32_basic)
{
    size_t inputByteSize = Align32(kElementCount * sizeof(float));
    size_t outputByteSize = Align32(kElementCount * sizeof(float));
    size_t tilingSize = Align32(sizeof(Ops::Base::EleBaseTilingDataV2));

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(Align32(1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto* xData = reinterpret_cast<float*>(x);
    for (int64_t i = 0; i < kElementCount; ++i) {
        xData[i] = static_cast<float>(i % 100) * 0.01f - 0.5f;
    }
    std::memset(y, 0, outputByteSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<Ops::Base::EleBaseTilingDataV2*>(tiling);
    tilingData->dim0 = kElementCount;
    tilingData->coreNum = 1;
    tilingData->ubFormer = kElementCount;
    tilingData->blockFormer = kElementCount;
    tilingData->blockNum = kNumBlocks;
    tilingData->ubLoopOfFormerBlock = 1;
    tilingData->ubLoopOfTailBlock = 1;
    tilingData->ubTailOfFormerBlock = kElementCount;
    tilingData->ubTailOfTailBlock = kElementCount;
    tilingData->elemNum = kElementCount;
    tilingData->scheMode = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(6);
    ICPU_RUN_KF((AscendC::erfc<0>), kNumBlocks, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
