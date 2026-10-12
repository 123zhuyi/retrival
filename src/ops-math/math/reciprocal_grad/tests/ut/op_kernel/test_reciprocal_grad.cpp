/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "atvoss/elewise/elewise_base_struct.h"
#include "atvoss/elewise/elewise_sch.h"
#include "kernel_operator.h"
#include "../../../op_kernel/arch35/reciprocal_grad_dag.h"
#include "../../../op_kernel/arch35/reciprocal_grad_tiling_data.h"
#include "../../../op_kernel/arch35/reciprocal_grad_struct.h"

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr int64_t kElementCount = 256;

inline size_t Align32(size_t size) { return (size + 31U) / 32U * 32U; }

inline void InitEleBaseTiling(Ops::Base::EleBaseTilingData& t, int64_t elemNum)
{
    t.dim0 = elemNum;
    t.coreNum = 1;
    t.ubFormer = static_cast<int32_t>(elemNum);
    t.blockFormer = elemNum;
    t.blockNum = kNumBlocks;
    t.ubLoopOfFormerBlock = 1;
    t.ubLoopOfTailBlock = 1;
    t.ubTailOfFormerBlock = elemNum;
    t.ubTailOfTailBlock = elemNum;
    t.elemNum = elemNum;
    t.scheMode = 0;
}

// UT 用模板 wrapper 显式实例化 kernel 逻辑（Kernel 入口依赖编译期宏 DTYPE_X，
// UT 中改为按 T 显式指定 dtype，与 Kernel 入口行为一致）
template <typename T, uint64_t schMode>
void RunReciprocalGrad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReciprocalGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReciprocalGradTilingData, tilingData, tiling);
    TPipe pipe;
    using OpDag = typename NsReciprocalGrad::ReciprocalGradCompute<T>::OpDag;
    ElementwiseSch<schMode, OpDag> sch(&(tilingData.baseTiling), &pipe);
    sch.Init(y, dy, z);
    sch.Process();
}
} // namespace

class ReciprocalGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReciprocalGradKernelTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReciprocalGradKernelTest TearDown" << std::endl; }
};

TEST_F(ReciprocalGradKernelTest, test_fp32_basic)
{
    size_t inputByteSize = Align32(kElementCount * sizeof(float));
    size_t outputByteSize = Align32(kElementCount * sizeof(float));
    size_t tilingSize = Align32(sizeof(ReciprocalGradTilingData));

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(Align32(1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto* yData = reinterpret_cast<float*>(y);
    auto* dyData = reinterpret_cast<float*>(dy);
    for (int64_t i = 0; i < kElementCount; ++i) {
        yData[i] = static_cast<float>(i % 100 + 1) * 0.01f;
        dyData[i] = static_cast<float>(i % 50) * 0.02f;
    }
    std::memset(z, 0, outputByteSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<ReciprocalGradTilingData*>(tiling);
    InitEleBaseTiling(tilingData->baseTiling, kElementCount);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF((RunReciprocalGrad<float, 0>), kNumBlocks, y, dy, z, workspace, tiling);

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ReciprocalGradKernelTest, test_fp16_basic)
{
    size_t inputByteSize = Align32(kElementCount * sizeof(half));
    size_t outputByteSize = Align32(kElementCount * sizeof(half));
    size_t tilingSize = Align32(sizeof(ReciprocalGradTilingData));

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(Align32(1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto* yData = reinterpret_cast<half*>(y);
    auto* dyData = reinterpret_cast<half*>(dy);
    for (int64_t i = 0; i < kElementCount; ++i) {
        yData[i] = static_cast<half>(static_cast<float>(i % 100 + 1) * 0.01f);
        dyData[i] = static_cast<half>(static_cast<float>(i % 50) * 0.02f);
    }
    std::memset(z, 0, outputByteSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<ReciprocalGradTilingData*>(tiling);
    InitEleBaseTiling(tilingData->baseTiling, kElementCount);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF((RunReciprocalGrad<half, 0>), kNumBlocks, y, dy, z, workspace, tiling);

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ReciprocalGradKernelTest, test_bf16_basic)
{
    size_t inputByteSize = Align32(kElementCount * sizeof(bfloat16_t));
    size_t outputByteSize = Align32(kElementCount * sizeof(bfloat16_t));
    size_t tilingSize = Align32(sizeof(ReciprocalGradTilingData));

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(Align32(1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto* yData = reinterpret_cast<bfloat16_t*>(y);
    auto* dyData = reinterpret_cast<bfloat16_t*>(dy);
    for (int64_t i = 0; i < kElementCount; ++i) {
        yData[i] = static_cast<bfloat16_t>(static_cast<float>(i % 100 + 1) * 0.01f);
        dyData[i] = static_cast<bfloat16_t>(static_cast<float>(i % 50) * 0.02f);
    }
    std::memset(z, 0, outputByteSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<ReciprocalGradTilingData*>(tiling);
    InitEleBaseTiling(tilingData->baseTiling, kElementCount);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF((RunReciprocalGrad<bfloat16_t, 0>), kNumBlocks, y, dy, z, workspace, tiling);

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ReciprocalGradKernelTest, test_fp32_dy_zero)
{
    size_t inputByteSize = Align32(kElementCount * sizeof(float));
    size_t outputByteSize = Align32(kElementCount * sizeof(float));
    size_t tilingSize = Align32(sizeof(ReciprocalGradTilingData));

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(Align32(1024));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto* yData = reinterpret_cast<float*>(y);
    auto* dyData = reinterpret_cast<float*>(dy);
    for (int64_t i = 0; i < kElementCount; ++i) {
        yData[i] = static_cast<float>(i % 100 + 1) * 0.01f;
        dyData[i] = (i % 2 == 0) ? 0.0f : static_cast<float>(i % 50) * 0.02f;
    }
    std::memset(z, 0, outputByteSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<ReciprocalGradTilingData*>(tiling);
    InitEleBaseTiling(tilingData->baseTiling, kElementCount);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF((RunReciprocalGrad<float, 0>), kNumBlocks, y, dy, z, workspace, tiling);

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
