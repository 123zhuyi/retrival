/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/arch35/drop_out_do_mask_tiling_arch35.h"

extern "C" __global__ __aicore__ void drop_out_do_mask(
    GM_ADDR x, GM_ADDR mask, GM_ADDR prob, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint64_t kTilingKey = 100;
constexpr uint32_t kNumBlocks = 1;
constexpr int64_t kElementCount = 256;
constexpr size_t kMaskBytes = 32;
constexpr size_t kWorkspaceBytes = 32;

size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}

uint8_t *AllocGm(size_t size)
{
    return reinterpret_cast<uint8_t *>(AscendC::GmAlloc(Align32(size)));
}

void InitInput(float *data)
{
    for (int64_t i = 0; i < kElementCount; ++i) {
        data[i] = static_cast<float>(i) + 0.5f;
    }
}
} // namespace

class DropOutDoMaskKernelUT : public testing::Test {
};

TEST_F(DropOutDoMaskKernelUT, keep_prob_one_copies_input_to_output)
{
    const size_t dataBytes = static_cast<size_t>(kElementCount) * sizeof(float);

    optiling::DropOutDoMaskForAscendCTilingData tilingData;
    tilingData.set_usedCoreNum(1);
    tilingData.set_normBlockData(kElementCount);
    tilingData.set_tailBlockData(kElementCount);
    tilingData.set_ubFactor(kElementCount);
    tilingData.set_normBlockLoop(1);
    tilingData.set_normBlockTail(kElementCount);
    tilingData.set_tailBlockLoop(1);
    tilingData.set_tailBlockTail(kElementCount);
    tilingData.set_epsilon(1.0e-6f);
    const size_t tilingBytes = static_cast<size_t>(tilingData.GetDataSize());

    uint8_t *x = AllocGm(dataBytes);
    uint8_t *mask = AllocGm(kMaskBytes);
    uint8_t *prob = AllocGm(sizeof(float));
    uint8_t *y = AllocGm(dataBytes);
    uint8_t *workspace = AllocGm(kWorkspaceBytes);
    uint8_t *tiling = AllocGm(tilingBytes);

    InitInput(reinterpret_cast<float *>(x));
    std::memset(mask, 0xFF, Align32(kMaskBytes));
    *reinterpret_cast<float *>(prob) = 1.0f;
    std::memset(y, 0, Align32(dataBytes));
    std::memset(tiling, 0, Align32(tilingBytes));
    tilingData.SaveToBuffer(tiling, tilingBytes);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(drop_out_do_mask, kNumBlocks, x, mask, prob, y, workspace, tiling);

    EXPECT_EQ(0, std::memcmp(y, x, dataBytes));

    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(prob);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
