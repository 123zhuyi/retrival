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
 * \file test_arange_kernel.cpp
 * \brief Arange op_kernel UT（CPU 孪生 / tikicpulib）：CopyOut 尾轴 OOB 防护覆盖
 *
 * 本 UT 用 tikicpulib CPU 孪生实跑 kernel，专项验证 CopyOut 末块 OOB 防护：
 *   realNum = min(num, totalNum - globalOffset)；realNum 满 32B 对齐走 DataCopy 快路径，
 *   否则末块走 DataCopyPad 按真实字节精确写 realNum 个元素。
 *   该逻辑全部在 op_kernel/arange.h 的 CopyOut() 内，op_host UT（tiling/infershape）无法覆盖。
 *
 * 验证点：
 *   (1) 末块非 32B 对齐时 out[0..N-1] 逐元素正确（realNum 截断不丢真实元素）；
 *   (2) out[N..] guard 区不被写（旧实现 DataCopy(ALIGN_UP_32B(num)) 会越界写 → guard 被破坏）；
 *   (3) 窄整型 1B(int8)/2B(int16) 尾轴（alignNum=32/16，最易越界）+ FP32 快路径无回归。
 *
 * 框架：直接 #include op_kernel/arange.h（头文件含 KernelArange / KernelArange_Cast 模板类），
 *   在本文件内为每个待测 dtype 写一个 extern "C" __global__ 薄包装核函数，避免依赖
 *   -DDTYPE_OUT 单 dtype 编译宏，从而在一次编译内覆盖 fp32 / int8 / int16 三类尾轴场景。
 *
 * golden：arange 为闭式 out[i]=start+i*step，golden 在 host 侧 C++ 直算（不依赖 python 数据文件）。
 *   整型出口语义 = CAST_ROUND 取整 + 硬件饱和(clamp)，与 ST golden 一致。
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>
#include <limits>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

// 直接引入 kernel 头（KernelArange / KernelArange_Cast 模板类 + ArangeTilingData + SCH_MODE 宏）
#include "../../../op_kernel/arange.h"

using namespace std;

namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024;
// guard 哨兵：out[N..] 区填充该值，kernel 跑后该区必须保持不变（证明无 OOB 写）
constexpr uint8_t GUARD_BYTE = 0xAB;

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

// 单核（blockDim=1）切分：totalNum 全归 core0 的 tail 段（formerNum=0），
// UB 子循环按 unitNum 切。该构造与 host tiling 在单核下产出一致（小 shape / 强制单核）。
static void FillSingleCoreTiling(ArangeTilingData& t, uint32_t totalNum, uint32_t dtypeSize, uint32_t unitNum)
{
    uint32_t alignNum = BLOCK_SIZE / dtypeSize;
    if (alignNum == 0) {
        alignNum = 1;
    }
    uint32_t totalBlocks = (totalNum + alignNum - 1) / alignNum;
    if (totalBlocks == 0) {
        totalBlocks = 1;
    }
    uint32_t tailLength = totalBlocks * alignNum; // 32B 对齐放大后的名义长度（≥ totalNum）

    t.dtypeSize = dtypeSize;
    t.totalNum = totalNum;
    t.unitNum = unitNum;
    t.coreNum = 1;
    t.formerNum = 0;
    t.formerLength = 0;
    t.tailLength = tailLength;
    t.formerUnitLoops = 0;
    t.formerTailNum = 0;
    // tail UB 子循环
    uint32_t loops = tailLength / unitNum;
    uint32_t tail = tailLength - unitNum * loops;
    if (tail > 0) {
        loops += 1;
    }
    t.tailUnitLoops = loops;
    t.tailTailNum = tail;
}

// ---- 待测薄包装核函数（test 内实例化 arange.h 模板类，规避 -DDTYPE 单 dtype 编译限制）----
// FP32 直算路径（MODE_1）
extern "C" __global__ __aicore__ void arange_ut_fp32(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ArangeTilingData);
    GET_TILING_DATA_WITH_STRUCT(ArangeTilingData, tilingData, tiling);
    NsArange::KernelArange<float, float, float> op;
    op.Init(start, end, step, out, tilingData);
    op.Process();
}

// int8 Cast 路径（MODE_0）：1B 尾轴 alignNum=32
extern "C" __global__ __aicore__ void arange_ut_int8(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ArangeTilingData);
    GET_TILING_DATA_WITH_STRUCT(ArangeTilingData, tilingData, tiling);
    NsArange::KernelArange_Cast<int8_t, int8_t, int8_t> op;
    op.Init(start, end, step, out, tilingData);
    op.Process();
}

// int16 Cast 路径（MODE_0）—— 2B 尾轴 alignNum=16
extern "C" __global__ __aicore__ void arange_ut_int16(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ArangeTilingData);
    GET_TILING_DATA_WITH_STRUCT(ArangeTilingData, tilingData, tiling);
    NsArange::KernelArange_Cast<int16_t, int16_t, int16_t> op;
    op.Init(start, end, step, out, tilingData);
    op.Process();
}

// ---- golden ----
static float GoldenAt(float start, float step, uint32_t i) { return start + static_cast<float>(i) * step; }

template <typename T>
static T SaturateRound(float v)
{
    // CAST_ROUND 就近取整 + 硬件饱和（clamp）—— 与 ST golden 语义一致
    float r = std::nearbyint(v);
    float lo = static_cast<float>(std::numeric_limits<T>::lowest());
    float hi = static_cast<float>(std::numeric_limits<T>::max());
    if (r < lo)
        r = lo;
    if (r > hi)
        r = hi;
    return static_cast<T>(r);
}

// 通用执行 + 断言：分配 out = N 真实元素 + guard 区，跑 kernel，校验 out[0..N-1] 与 guard 区
template <typename TOUT>
static void RunAndCheck(void (*func)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR), uint32_t totalNum,
                        uint32_t dtypeSize, uint32_t unitNum, float startVal, float stepVal, bool isFloat)
{
    // 标量输入（按 32B 对齐分配，单元素）
    uint8_t* start = (uint8_t*)AscendC::GmAlloc(CeilAlign(sizeof(TOUT), (size_t)BLOCK_SIZE));
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(CeilAlign(sizeof(TOUT), (size_t)BLOCK_SIZE));
    uint8_t* endg = (uint8_t*)AscendC::GmAlloc(CeilAlign(sizeof(TOUT), (size_t)BLOCK_SIZE));
    *reinterpret_cast<TOUT*>(start) = static_cast<TOUT>(startVal);
    *reinterpret_cast<TOUT*>(step) = static_cast<TOUT>(stepVal);
    *reinterpret_cast<TOUT*>(endg) = static_cast<TOUT>(0);

    // out 区：真实 N 元素 + 一段 guard（32B），用于检测 OOB 越界写。
    //   kernel CopyOut 末块若按 32B 对齐放大写，会写入 guard 区 → 被破坏。
    const uint32_t guardElems = BLOCK_SIZE / dtypeSize > 0 ? BLOCK_SIZE / dtypeSize : 1; // 至少 1 个 32B 块的元素
    const uint32_t allocElems = totalNum + guardElems + guardElems;                      // N + 2 个 32B 块裕量
    size_t outBytes = (size_t)allocElems * sizeof(TOUT);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(CeilAlign(outBytes, (size_t)BLOCK_SIZE));
    // 全 guard 填充，跑后只有 [0,N) 应被覆盖
    std::memset(out, GUARD_BYTE, CeilAlign(outBytes, (size_t)BLOCK_SIZE));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(WORKSPACE_SIZE);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(ArangeTilingData));
    ArangeTilingData* t = reinterpret_cast<ArangeTilingData*>(tiling);
    FillSingleCoreTiling(*t, totalNum, dtypeSize, unitNum);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // 核函数签名固定为 (start, end, step, out, workspace, tiling)，实参必须按此【位置顺序】传：
    //   arg1=start, arg2=end(endg, 不参与计算/占位), arg3=step, arg4=out。
    ICPU_RUN_KF(func, 1, start, endg, step, out, workspace, tiling);

    TOUT* outData = reinterpret_cast<TOUT*>(out);

    // (1) out[0..N-1] 逐元素正确
    uint32_t mismatch = 0;
    for (uint32_t i = 0; i < totalNum; i++) {
        float g = GoldenAt(startVal, stepVal, i);
        if (isFloat) {
            float got = static_cast<float>(outData[i]);
            if (std::fabs(got - g) > 1e-4f * (std::fabs(g) + 1.0f)) {
                if (mismatch < 8) {
                    std::cout << "  [FP mismatch] i=" << i << " got=" << got << " exp=" << g << std::endl;
                }
                mismatch++;
            }
        } else {
            TOUT exp = SaturateRound<TOUT>(g);
            if (outData[i] != exp) {
                if (mismatch < 8) {
                    std::cout << "  [INT mismatch] i=" << i << " got=" << (int64_t)outData[i] << " exp=" << (int64_t)exp
                              << std::endl;
                }
                mismatch++;
            }
        }
    }
    EXPECT_EQ(mismatch, 0u) << "out[0.." << (totalNum - 1) << "] 逐元素与 golden 不一致";

    // (2) guard 区 out[N..] 必须仍为 GUARD_BYTE（无 OOB 越界写）
    const uint8_t* guardStart = out + (size_t)totalNum * sizeof(TOUT);
    size_t guardBytes = CeilAlign(outBytes, (size_t)BLOCK_SIZE) - (size_t)totalNum * sizeof(TOUT);
    uint32_t corrupted = 0;
    for (size_t b = 0; b < guardBytes; b++) {
        if (guardStart[b] != GUARD_BYTE) {
            corrupted++;
        }
    }
    EXPECT_EQ(corrupted, 0u) << "guard 区 out[N..] 被写（" << corrupted << "/" << guardBytes
                             << " byte 被破坏）—— CopyOut 尾轴 OOB 硬化失效（realNum/DataCopyPad 未生效）";

    AscendC::GmFree((void*)start);
    AscendC::GmFree((void*)step);
    AscendC::GmFree((void*)endg);
    AscendC::GmFree((void*)out);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}
} // namespace

class ArangeKernel : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ArangeKernel SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ArangeKernel TearDown" << std::endl; }
};

// ============================================================================
// CopyOut 尾轴 OOB 防护（realNum + DataCopyPad）专项
// 构造「N 非 32B 对齐」使末块走 DataCopyPad 分支；guard 区检测越界写。
// ============================================================================

// ---- int8（1B，alignNum=32，尾轴最易越界）----

// int8 N=33：totalBlocks=ceil(33/32)=2，末块真实 1 元素（非 32B 对齐）→ DataCopyPad 路径
TEST_F(ArangeKernel, int8_tail_unaligned_n33_no_oob)
{
    // unitNum 取较大值使单 UB 块覆盖 tailLength（33→放大到 64），单 loop 内末块=realNum 截断
    RunAndCheck<int8_t>(arange_ut_int8, /*N=*/33, /*dtypeSize=*/1, /*unitNum=*/256,
                        /*start=*/0.0f, /*step=*/1.0f, /*isFloat=*/false);
}

// int8 N=100：totalBlocks=ceil(100/32)=4，名义 128，末块真实 100-? 非对齐 → DataCopyPad
TEST_F(ArangeKernel, int8_tail_unaligned_n100_no_oob)
{
    RunAndCheck<int8_t>(arange_ut_int8, 100, 1, 256, -50.0f, 1.0f, false);
}

// int8 N=32：恰 32B 对齐 → DataCopy 快路径（不应走 Pad），guard 仍须不破坏
TEST_F(ArangeKernel, int8_tail_aligned_n32_fast_path)
{
    RunAndCheck<int8_t>(arange_ut_int8, 32, 1, 256, 0.0f, 1.0f, false);
}

// int8 越界饱和：start=120 step=1 N=20 末值 139 → clamp 127（golden 同饱和）
TEST_F(ArangeKernel, int8_saturate_clamp_with_tail)
{
    RunAndCheck<int8_t>(arange_ut_int8, 20, 1, 256, 120.0f, 1.0f, false);
}

// ---- int16（2B，alignNum=16）----

// int16 N=17：totalBlocks=ceil(17/16)=2，名义 32，末块非对齐 → DataCopyPad
TEST_F(ArangeKernel, int16_tail_unaligned_n17_no_oob)
{
    RunAndCheck<int16_t>(arange_ut_int16, 17, 2, 256, -8.0f, 1.0f, false);
}

// int16 N=16：恰 32B 对齐（16*2B=32B）→ 快路径，guard 不破坏
TEST_F(ArangeKernel, int16_tail_aligned_n16_fast_path)
{
    RunAndCheck<int16_t>(arange_ut_int16, 16, 2, 256, 100.0f, 5.0f, false);
}

// int16 N=300 多 UB 块 + 末块非对齐（300%16!=0）：unitNum 较小迫使多次 UB 循环，尾块走 Pad
TEST_F(ArangeKernel, int16_multi_ub_tail_unaligned_no_oob)
{
    RunAndCheck<int16_t>(arange_ut_int16, 300, 2, 64, 0.0f, 2.0f, false);
}

// ---- FP32（4B，alignNum=8）快路径无回归 ----

// fp32 N=10：totalBlocks=ceil(10/8)=2，名义 16，末块非对齐 → DataCopyPad（FP32 也走防护路径）
TEST_F(ArangeKernel, fp32_tail_unaligned_n10_no_oob)
{
    RunAndCheck<float>(arange_ut_fp32, 10, 4, 256, 0.5f, 0.25f, true);
}

// fp32 N=8：恰 32B 对齐 → 快路径，guard 不破坏
TEST_F(ArangeKernel, fp32_tail_aligned_n8_fast_path)
{
    RunAndCheck<float>(arange_ut_fp32, 8, 4, 256, -1.0f, 0.5f, true);
}

// fp32 N=1 单元素：极小 shape，末块=1（非 8 对齐）→ DataCopyPad，guard 不破坏
TEST_F(ArangeKernel, fp32_n1_single_element_no_oob)
{
    RunAndCheck<float>(arange_ut_fp32, 1, 4, 256, 3.14f, 1.0f, true);
}
