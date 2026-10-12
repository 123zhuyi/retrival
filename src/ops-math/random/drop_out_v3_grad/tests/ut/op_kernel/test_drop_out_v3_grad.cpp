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
 * \file test_drop_out_v3_grad.cpp
 * \brief DropOutV3Grad kernel UT —— 对标 golden 校验 grad_x = (mask==1) ? scale*grad_y : 0
 *        仅编译 float 路径（-DDTYPE_GRAD_Y=float），覆盖三分支 / mask 各形态 / 尾块非对齐 / 多核。
 */

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/arch35/drop_out_v3_grad_tiling_arch35.h"

extern "C" __global__ __aicore__ void drop_out_v3_grad(GM_ADDR grad_y, GM_ADDR mask, GM_ADDR scale, GM_ADDR grad_x,
                                                       GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint64_t kTilingKey = 100;
constexpr size_t kWorkspaceBytes = 32;

size_t Align32(size_t size) { return (size + 31U) / 32U * 32U; }

uint8_t* AllocGm(size_t size) { return reinterpret_cast<uint8_t*>(AscendC::GmAlloc(Align32(size))); }

// 递增填充 grad_y，值域带小数与负数，暴露截断/符号问题
void InitGradY(float* data, int64_t count)
{
    for (int64_t i = 0; i < count; ++i) {
        data[i] = static_cast<float>(i) * 0.5f - 3.0f;
    }
}

// 取 mask 第 i 位（bit-packed，1=保留）
bool MaskBit(const uint8_t* mask, int64_t i) { return (mask[i / 8] >> (i % 8)) & 0x1U; }

// golden：逐元素 grad_x = (mask==1) ? scale*grad_y : 0
void ComputeGolden(const float* gradY, const uint8_t* mask, float scale, float* golden, int64_t count)
{
    for (int64_t i = 0; i < count; ++i) {
        golden[i] = MaskBit(mask, i) ? scale * gradY[i] : 0.0f;
    }
}

// 单核、单 UB 循环的常规 tiling（count 需 <= ubFactor）
// 注意：DropOutV3GradForAscendCTilingData（TilingDef 派生）内部持有堆缓冲且无安全的拷贝语义，
// 复制该对象会导致析构时 double free。故一律通过引用就地填充，禁止按值返回/拷贝。
void FillSingleCoreTiling(optiling::DropOutV3GradForAscendCTilingData& t, int64_t count)
{
    // 真实 tiling 保证 ubFactor 恒为 256 的倍数（见 tiling FloorAlign(ubFactor, 256)），
    // kernel 的 mask UB buffer 按 ubFactor/8 分配依赖该对齐前提。测试须同构，否则小 count 会 buffer 不足。
    const int64_t ubFactor = (count + 255) / 256 * 256;
    t.set_usedCoreNum(1);
    t.set_normBlockData(count);
    t.set_tailBlockData(count);
    t.set_ubFactor(ubFactor);
    t.set_normBlockLoop(1);
    t.set_normBlockTail(count);
    t.set_tailBlockLoop(1);
    t.set_tailBlockTail(count);
    t.set_epsilon(std::numeric_limits<float>::epsilon());
}

// 通用执行器：按给定 mask 字节 / scale 跑 kernel，返回 grad_x 到 out
void RunKernel(int64_t count, uint32_t numBlocks, optiling::DropOutV3GradForAscendCTilingData& tilingData,
               const std::vector<uint8_t>& maskBytes, float scaleVal, std::vector<float>& out)
{
    const size_t dataBytes = static_cast<size_t>(count) * sizeof(float);
    const size_t maskBytesLen = maskBytes.size();
    const size_t tilingBytes = static_cast<size_t>(tilingData.GetDataSize());

    uint8_t* gradY = AllocGm(dataBytes);
    uint8_t* mask = AllocGm(maskBytesLen);
    uint8_t* scale = AllocGm(sizeof(float));
    uint8_t* gradX = AllocGm(dataBytes);
    uint8_t* workspace = AllocGm(kWorkspaceBytes);
    uint8_t* tiling = AllocGm(tilingBytes);

    InitGradY(reinterpret_cast<float*>(gradY), count);
    std::memcpy(mask, maskBytes.data(), maskBytesLen);
    *reinterpret_cast<float*>(scale) = scaleVal;
    std::memset(gradX, 0, Align32(dataBytes));
    std::memset(tiling, 0, Align32(tilingBytes));

    tilingData.SaveToBuffer(tiling, tilingBytes);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(drop_out_v3_grad, numBlocks, gradY, mask, scale, gradX, workspace, tiling);

    out.resize(count);
    std::memcpy(out.data(), gradX, dataBytes);

    AscendC::GmFree(gradY);
    AscendC::GmFree(mask);
    AscendC::GmFree(scale);
    AscendC::GmFree(gradX);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 校验 out 与 golden 逐元素相等（float 位精确，scale*grad_y 在 kernel/golden 均 fp32 单乘）
void ExpectEqGolden(int64_t count, const std::vector<uint8_t>& maskBytes, float scaleVal, const std::vector<float>& out)
{
    std::vector<float> gradY(count);
    InitGradY(gradY.data(), count);
    std::vector<float> golden(count);
    ComputeGolden(gradY.data(), maskBytes.data(), scaleVal, golden.data(), count);
    for (int64_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(out[i], golden[i]) << "mismatch at index " << i;
    }
}

std::vector<uint8_t> MaskAllOnes(int64_t count)
{
    return std::vector<uint8_t>(Align32(static_cast<size_t>(count + 7) / 8), 0xFFU);
}

std::vector<uint8_t> MaskAllZeros(int64_t count)
{
    return std::vector<uint8_t>(Align32(static_cast<size_t>(count + 7) / 8), 0x00U);
}
} // namespace

class DropOutV3GradKernelUT : public testing::Test {};

// ================= 分支 B：scale==1（全保留），grad_x == grad_y =================
TEST_F(DropOutV3GradKernelUT, scale_one_copies_input_to_output)
{
    constexpr int64_t count = 256;
    optiling::DropOutV3GradForAscendCTilingData tiling;
    FillSingleCoreTiling(tiling, count);
    auto mask = MaskAllOnes(count); // scale==1 分支不读 mask，填 1 便于同时对 golden
    std::vector<float> out;
    RunKernel(count, 1, tiling, mask, 1.0f, out);
    ExpectEqGolden(count, mask, 1.0f, out);
}

// ================= 分支 A：scale==0（全丢弃），grad_x == 0 =================
TEST_F(DropOutV3GradKernelUT, scale_zero_outputs_all_zero)
{
    constexpr int64_t count = 256;
    optiling::DropOutV3GradForAscendCTilingData tiling;
    FillSingleCoreTiling(tiling, count);
    auto mask = MaskAllOnes(count); // 即便 mask 全 1，scale==0 也应全 0
    std::vector<float> out;
    RunKernel(count, 1, tiling, mask, 0.0f, out);
    for (int64_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f) << "index " << i;
    }
}

// ================= 分支 C：正常 scale，mask 全 1 =================
TEST_F(DropOutV3GradKernelUT, normal_scale_mask_all_ones)
{
    constexpr int64_t count = 256;
    optiling::DropOutV3GradForAscendCTilingData tiling;
    FillSingleCoreTiling(tiling, count);
    auto mask = MaskAllOnes(count);
    std::vector<float> out;
    RunKernel(count, 1, tiling, mask, 2.0f, out);
    ExpectEqGolden(count, mask, 2.0f, out);
}

// ================= 分支 C：正常 scale，mask 全 0（结果应全 0） =================
TEST_F(DropOutV3GradKernelUT, normal_scale_mask_all_zeros)
{
    constexpr int64_t count = 256;
    optiling::DropOutV3GradForAscendCTilingData tiling;
    FillSingleCoreTiling(tiling, count);
    auto mask = MaskAllZeros(count);
    std::vector<float> out;
    RunKernel(count, 1, tiling, mask, 2.0f, out);
    ExpectEqGolden(count, mask, 2.0f, out);
    for (int64_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(out[i], 0.0f) << "index " << i;
    }
}

// ================= 分支 C：正常 scale，mask 交替 0/1，校验 Select 精确对位 =================
TEST_F(DropOutV3GradKernelUT, normal_scale_mask_alternating)
{
    constexpr int64_t count = 256;
    optiling::DropOutV3GradForAscendCTilingData tiling;
    FillSingleCoreTiling(tiling, count);
    std::vector<uint8_t> mask(Align32(count / 8), 0xAAU); // 10101010，偶数位丢弃奇数位保留
    std::vector<float> out;
    RunKernel(count, 1, tiling, mask, 1.5f, out);
    ExpectEqGolden(count, mask, 1.5f, out);
}

// ================= 分支 C：非规则 scale（含小数），验证一次乘法精度 =================
TEST_F(DropOutV3GradKernelUT, normal_scale_fractional_value)
{
    constexpr int64_t count = 256;
    optiling::DropOutV3GradForAscendCTilingData tiling;
    FillSingleCoreTiling(tiling, count);
    std::vector<uint8_t> mask(Align32(count / 8), 0xF0U);
    std::vector<float> out;
    RunKernel(count, 1, tiling, mask, 1.0f / 0.7f, out); // scale=1/(1-0.3)
    ExpectEqGolden(count, mask, 1.0f / 0.7f, out);
}

// ================= 边界：尾块非 256 对齐（count 不是向量寄存器整数倍） =================
TEST_F(DropOutV3GradKernelUT, tail_count_not_aligned)
{
    constexpr int64_t count = 300; // 非 256 对齐，触发 UpdateMask 尾处理
    optiling::DropOutV3GradForAscendCTilingData tiling;
    FillSingleCoreTiling(tiling, count);
    std::vector<uint8_t> mask(Align32(static_cast<size_t>(count + 7) / 8), 0xCCU);
    std::vector<float> out;
    RunKernel(count, 1, tiling, mask, 2.5f, out);
    ExpectEqGolden(count, mask, 2.5f, out);
}

// ================= 边界：极小规模 count=1 =================
TEST_F(DropOutV3GradKernelUT, single_element)
{
    constexpr int64_t count = 1;
    optiling::DropOutV3GradForAscendCTilingData tiling;
    FillSingleCoreTiling(tiling, count);
    std::vector<uint8_t> mask(Align32(1), 0x01U);
    std::vector<float> out;
    RunKernel(count, 1, tiling, mask, 3.0f, out);
    ExpectEqGolden(count, mask, 3.0f, out);
}

// ================= 多核 + 多 UB 循环：单核内 count 超 ubFactor，需循环 =================
TEST_F(DropOutV3GradKernelUT, multi_core_multi_loop)
{
    constexpr int64_t count = 1024;
    constexpr int64_t ubFactor = 256;
    constexpr int64_t coreNum = 2;
    constexpr int64_t normBlockData = 512; // 每核 512
    optiling::DropOutV3GradForAscendCTilingData tiling;
    tiling.set_usedCoreNum(coreNum);
    tiling.set_normBlockData(normBlockData);
    tiling.set_tailBlockData(normBlockData); // 整除，尾核同普通核
    tiling.set_ubFactor(ubFactor);
    tiling.set_normBlockLoop(normBlockData / ubFactor); // 2 次
    tiling.set_normBlockTail(ubFactor);
    tiling.set_tailBlockLoop(normBlockData / ubFactor);
    tiling.set_tailBlockTail(ubFactor);
    tiling.set_epsilon(std::numeric_limits<float>::epsilon());

    auto mask = MaskAllOnes(count);
    std::vector<float> out;
    RunKernel(count, coreNum, tiling, mask, 2.0f, out);
    ExpectEqGolden(count, mask, 2.0f, out);
}

// ================= scale 接近 1 但不等于 1（走正常分支而非全保留） =================
TEST_F(DropOutV3GradKernelUT, scale_near_one_uses_normal_branch)
{
    constexpr int64_t count = 256;
    optiling::DropOutV3GradForAscendCTilingData tiling;
    FillSingleCoreTiling(tiling, count);
    auto mask = MaskAllOnes(count);
    std::vector<float> out;
    RunKernel(count, 1, tiling, mask, 1.0001f, out);
    ExpectEqGolden(count, mask, 1.0001f, out); // 结果应为 1.0001*grad_y，而非原样拷贝
}
