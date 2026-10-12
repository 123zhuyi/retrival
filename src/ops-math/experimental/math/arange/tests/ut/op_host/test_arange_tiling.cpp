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
 * \file test_arange_tiling.cpp
 * \brief Arange op_host tiling UT：多核 former/tail 切分 + 全 dtype TilingKey 映射覆盖
 *
 * 覆盖：
 *  - 多核 tiling former/tail 块切分正确性
 *  - formerNum = totalBlocks % coreNum、former 上取整 / tail 下取整
 *  - 小 shape 退化（coreNum = min(maxCore, totalBlocks)、N=1 单核）
 *  - 每段 UB 子循环（CalcUnitLoops：unitLoops / tailNum）
 *  - 从 out shape 读 N（不读 start/end/step 数值）
 *  - 全 dtype（fp32/fp16/bf16/int8/uint8/int16/int32/int64）的 TilingKey 映射（FP32→MODE_1，其余→MODE_0）
 *  - 窄整型 dtype_size：int8=1、uint8=1、int16=2，且真实影响 32B 对齐块切分
 *    （alignNum=32/dtype_size：int8=32、uint8=32、int16=16）
 *  - UB 红线断言：unitNum = ub_unit_size / max(dtypeSize, 4)，验证 int8/uint8/int16
 *    下 unitNum 按 FP32(4B) 封顶（与 fp32 同值），不会因 dtype_size=1/2 放大到爆 UB
 *  - 窄整型下多核 former/tail 切分仍正确
 *
 * 设计说明：tiling 内部用 GetCoreMemSize(UB) / GetCoreNum() 读真实平台规格（UB 字节数随
 * 910b 平台保留区而非测试传入值），故 unitNum / 真实 coreNum 上限不在 UT 中硬编码。
 * UT 改为「读回 tiling 实际产出的 coreNum / unitNum，再按设计公式复算并逐字段断言」，
 * 这样既严格验证多核 former/tail 切分与 CalcUnitLoops 逻辑，又不耦合平台 UB 细节。
 *
 * UB 红线断言策略：ub_unit_size 是平台固定值（10 等份后 32B 对齐），与 dtype 无关。
 * 公式 unitNum = ub_unit_size / max(dtypeSize, 4) 下，dtypeSize≤4 的所有 dtype（int8/uint8/
 * int16/fp16/bf16/fp32/int32）的 unitNum 必然【全部相等】（分母都是 4）。若 bug 复现
 * （unitNum = ub_unit_size / dtypeSize），int8 的 unitNum 会是 fp32 的 4 倍、int16 是 2 倍。
 * 因此「跨 dtype 比对 unitNum 相等」这一断言可在不硬编码平台 UB 的前提下精确捕获回归。
 */

#include <iostream>
#include <gtest/gtest.h>
#include "arange_tiling.h"
#include "../../../op_kernel/arange_tiling_data.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

namespace {
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DTYPE_SIZE1 = 1; // int8 / uint8
constexpr uint32_t DTYPE_SIZE2 = 2;
constexpr uint32_t DTYPE_SIZE4 = 4;
constexpr uint32_t DTYPE_SIZE8 = 8;
constexpr uint32_t FP32_SIZE = 4; // unitNum 按 FP32 字节统一切（UB 红线基准）

// 与 arange_tiling.cpp 的 switch 一致（含窄整型 int8/uint8=1、int16=2）
static uint32_t DtypeSizeOf(ge::DataType dt)
{
    switch (dt) {
        case ge::DataType::DT_INT8:
        case ge::DataType::DT_UINT8:
            return DTYPE_SIZE1;
        case ge::DataType::DT_FLOAT16:
        case ge::DataType::DT_BF16:
        case ge::DataType::DT_INT16:
            return DTYPE_SIZE2;
        case ge::DataType::DT_FLOAT:
        case ge::DataType::DT_INT32:
            return DTYPE_SIZE4;
        case ge::DataType::DT_INT64:
            return DTYPE_SIZE8;
        default:
            return DTYPE_SIZE2;
    }
}

// 块数上取整（ceil(blocks / coreNum)），用于窄整型 former 段块数复算
static uint32_t ceil_div_blocks(uint32_t totalBlocks, uint32_t coreNum)
{
    if (coreNum == 0) {
        return totalBlocks;
    }
    return (totalBlocks + coreNum - 1) / coreNum;
}

// 独立复算 CalcUnitLoops（与 arange_tiling.cpp 同算法），用于断言 host tiling 输出
static void ExpectUnitLoops(uint32_t segLength, uint32_t unitNum, uint32_t& unitLoops, uint32_t& tailNum)
{
    if (segLength == 0 || unitNum == 0) {
        unitLoops = 0;
        tailNum = 0;
        return;
    }
    unitLoops = segLength / unitNum;
    tailNum = segLength - unitNum * unitLoops;
    if (tailNum > 0) {
        unitLoops += 1;
    }
}

// 跑一次 tiling 并取回 TilingInfo（含 raw tiling data + blockNum + tilingKey + workspace）
static bool RunTiling(uint32_t totalNum, ge::DataType dtype, uint64_t maxCoreNum, TilingInfo& info)
{
    ArangeCompileInfo compileInfo;
    // start/end/step 为标量（shape {1}），out 为一维 [totalNum]
    gert::TilingContextPara para(
        "Arange",
        {
            {{{1}, {1}}, dtype, ge::FORMAT_ND}, // start
            {{{1}, {1}}, dtype, ge::FORMAT_ND}, // end
            {{{1}, {1}}, dtype, ge::FORMAT_ND}, // step
        },
        {
            {{{static_cast<int64_t>(totalNum)}, {static_cast<int64_t>(totalNum)}}, dtype, ge::FORMAT_ND}, // out [N]
        },
        &compileInfo,
        maxCoreNum); // coreNum -> compileInfo CORE_NUM -> GetCoreNum() 上限（UB 由真实平台决定）
    return ExecuteTiling(para, info);
}

static const ArangeTilingData* AsTiling(const TilingInfo& info)
{
    EXPECT_GE(info.tilingDataSize, sizeof(ArangeTilingData));
    return reinterpret_cast<const ArangeTilingData*>(info.tilingData.get());
}

// 核心：读回 tiling 实际产出的 coreNum / unitNum / totalNum / dtypeSize，
// 再用设计公式复算 former/tail/formerNum/CalcUnitLoops 并逐字段断言。
static void CheckMulticoreInvariants(const TilingInfo& info, uint32_t expectTotalNum, ge::DataType dtype)
{
    const ArangeTilingData* t = AsTiling(info);

    // —— 基础字段 ——
    EXPECT_EQ(t->totalNum, expectTotalNum);
    EXPECT_EQ(t->dtypeSize, DtypeSizeOf(dtype));
    EXPECT_GT(t->unitNum, 0u);
    EXPECT_GT(t->coreNum, 0u);
    EXPECT_EQ(info.blockNum, static_cast<size_t>(t->coreNum)); // SetBlockDim(coreNum)
    ASSERT_FALSE(info.workspaceSizes.empty());
    EXPECT_EQ(info.workspaceSizes[0], 0); // workspace = 0

    // —— 按实际 coreNum / dtypeSize 复算块切分 ——
    uint32_t dtypeSize = t->dtypeSize;
    uint32_t unitNum = t->unitNum;
    uint32_t coreNum = t->coreNum;

    uint32_t alignNum = BLOCK_SIZE / dtypeSize;
    if (alignNum == 0) {
        alignNum = 1;
    }
    uint32_t totalBlocks = (expectTotalNum + alignNum - 1) / alignNum;
    if (totalBlocks == 0) {
        totalBlocks = 1;
    }

    // 小 shape 退化：coreNum = min(maxCore, totalBlocks)。无法预知真实 maxCore，
    // 但 coreNum 必须 <= totalBlocks（不会开超过块数的核），且 >= 1。
    EXPECT_LE(coreNum, totalBlocks);
    EXPECT_GE(coreNum, 1u);

    // formerNum = totalBlocks % coreNum
    uint32_t expFormerNum = totalBlocks % coreNum;
    EXPECT_EQ(t->formerNum, expFormerNum);

    // former 上取整 / tail 下取整
    uint32_t formerBlocks = (totalBlocks + coreNum - 1) / coreNum; // ceil
    uint32_t tailBlocks = totalBlocks / coreNum;                   // floor
    uint32_t expFormerLength = (expFormerNum != 0) ? (formerBlocks * alignNum) : 0;
    uint32_t expTailLength = tailBlocks * alignNum;
    EXPECT_EQ(t->formerLength, expFormerLength);
    EXPECT_EQ(t->tailLength, expTailLength);

    // 均衡性：当存在 former 段时，former 恰好比 tail 多 1 个 32B 对齐块
    if (expFormerNum != 0) {
        EXPECT_EQ(t->formerLength - t->tailLength, alignNum);
    }

    // 全核覆盖：former 段总元素 + tail 段总元素 >= totalNum（32B 对齐放大后必不漏）
    uint64_t covered = static_cast<uint64_t>(t->formerLength) * expFormerNum +
                       static_cast<uint64_t>(t->tailLength) * (coreNum - expFormerNum);
    EXPECT_GE(covered, static_cast<uint64_t>(expectTotalNum));

    // —— 每段 UB 子循环（CalcUnitLoops）——
    uint32_t expFormerLoops = 0, expFormerTail = 0;
    uint32_t expTailLoops = 0, expTailTail = 0;
    if (expFormerNum != 0) {
        ExpectUnitLoops(expFormerLength, unitNum, expFormerLoops, expFormerTail);
    }
    ExpectUnitLoops(expTailLength, unitNum, expTailLoops, expTailTail);
    EXPECT_EQ(t->formerUnitLoops, expFormerLoops);
    EXPECT_EQ(t->formerTailNum, expFormerTail);
    EXPECT_EQ(t->tailUnitLoops, expTailLoops);
    EXPECT_EQ(t->tailTailNum, expTailTail);
}
} // namespace

class ArangeTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ArangeTiling SetUp" << endl; }
    static void TearDownTestCase() { cout << "ArangeTiling TearDown" << endl; }
};

// ===== 1) 多核 former/tail 块切分 + formerNum=totalBlocks%coreNum + 上/下取整 =====

// 大 N 全核：totalBlocks 远大于核数，不整除 -> 必有 former 段，验证 formerNum / 上取整 / 下取整
TEST_F(ArangeTiling, multicore_fp32_large_not_divisible)
{
    const uint32_t N = 100000; // fp32 alignNum=8 -> totalBlocks=12500
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    // 大 N 下应开多核
    EXPECT_GT(t->coreNum, 1u);
    CheckMulticoreInvariants(info, N, ge::DT_FLOAT);
}

// 整除场景：构造 N 使 totalBlocks 整除实际 coreNum 时 formerNum=0；这里直接验证 formerNum 与
// formerLength 的强关联（formerNum==0 <=> formerLength==0）在任意 N 下成立
TEST_F(ArangeTiling, multicore_former_zero_iff_length_zero)
{
    const uint32_t N = 1920;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    if (t->formerNum == 0) {
        EXPECT_EQ(t->formerLength, 0u);
        EXPECT_EQ(t->formerUnitLoops, 0u);
        EXPECT_EQ(t->formerTailNum, 0u);
    } else {
        EXPECT_GT(t->formerLength, 0u);
    }
    CheckMulticoreInvariants(info, N, ge::DT_FLOAT);
}

// 大 N int64（dtypeSize=8, alignNum=4）多核切分
TEST_F(ArangeTiling, multicore_int64_large)
{
    const uint32_t N = 65536;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_INT64, 24, info));
    CheckMulticoreInvariants(info, N, ge::DT_INT64);
}

// 大 N fp16（dtypeSize=2, alignNum=16）多核切分
TEST_F(ArangeTiling, multicore_fp16_large)
{
    const uint32_t N = 50000;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT16, 24, info));
    CheckMulticoreInvariants(info, N, ge::DT_FLOAT16);
}

// ===== 2) 小 shape 退化：coreNum = min(maxCore, totalBlocks) =====

// totalBlocks < maxCore：只开 totalBlocks 个核（coreNum 受块数限制）
TEST_F(ArangeTiling, small_shape_core_capped_by_blocks)
{
    // fp32 alignNum=8。N=40 -> totalBlocks=ceil(40/8)=5。maxCore=24 -> coreNum=min(24,5)=5
    const uint32_t N = 40;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 5u);    // 受 totalBlocks=5 限制（< 24）
    EXPECT_EQ(t->formerNum, 0u);  // 5 % 5 == 0
    EXPECT_EQ(t->tailLength, 8u); // 每核 1 个 32B 块 = 8 fp32
    CheckMulticoreInvariants(info, N, ge::DT_FLOAT);
}

// N=1 单核
TEST_F(ArangeTiling, n_equals_1_single_core)
{
    const uint32_t N = 1;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 1u); // totalBlocks=1 -> min(24,1)=1
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(t->formerNum, 0u);
    EXPECT_EQ(t->tailLength, 8u);    // 1 个 32B 块（fp32 对齐放大）
    EXPECT_EQ(t->tailUnitLoops, 1u); // 1 块一次 UB 循环
    CheckMulticoreInvariants(info, N, ge::DT_FLOAT);
}

// 小 shape 不整除 -> formerNum=1（maxCore=4 时 9 块 % 4 = 1）
TEST_F(ArangeTiling, small_shape_not_divisible_former_one)
{
    // fp32 alignNum=8。N=72 -> totalBlocks=9。maxCore=4 -> coreNum=min(4,9)=4，9%4=1
    const uint32_t N = 72;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT, 4, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 4u);
    EXPECT_EQ(t->formerNum, 1u);     // 9 % 4
    EXPECT_EQ(t->formerLength, 24u); // ceil(9/4)=3 块 * 8
    EXPECT_EQ(t->tailLength, 16u);   // 9/4=2 块 * 8
    CheckMulticoreInvariants(info, N, ge::DT_FLOAT);
}

// N 极小但 maxCore=1 强制单核（验证 maxCore 上限 + 单核全量切分）
TEST_F(ArangeTiling, force_single_core_max1)
{
    const uint32_t N = 100;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT, 1, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 1u);      // maxCore=1 -> coreNum=1
    EXPECT_EQ(t->formerNum, 0u);    // 单核无 former
    EXPECT_EQ(t->tailLength, 104u); // ceil(100/8)=13 块 * 8 = 104
    CheckMulticoreInvariants(info, N, ge::DT_FLOAT);
}

// ===== 3) 每段 UB 子循环（CalcUnitLoops）正确性：unitLoops>1、有尾块 / 无尾块 =====

// 强制单核 + 大 N，使 tail 段元素数 >> unitNum -> tailUnitLoops > 1（CalcUnitLoops 多块路径）
TEST_F(ArangeTiling, ub_subloop_multi_with_tail_single_core)
{
    const uint32_t N = 200000; // 单核（maxCore=1）下 tailLength 远大于 unitNum
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT, 1, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 1u);
    EXPECT_GT(t->tailUnitLoops, 1u); // 必然多次 UB 循环
    // CalcUnitLoops 关系：tailUnitLoops = ceil(tailLength/unitNum)，
    // tailTailNum = tailLength - (tailUnitLoops-1)*unitNum（CheckMulticoreInvariants 已逐字段断言）
    CheckMulticoreInvariants(info, N, ge::DT_FLOAT);
}

// tail 段恰为 unitNum 整数倍 -> tailTailNum=0（尾块对齐边界）：
// 用实际 unitNum 反推一个使 tailLength 恰为 unitNum 整数倍的 N（单核）
TEST_F(ArangeTiling, ub_subloop_exact_no_tail_single_core)
{
    // 先跑一次拿到实际 unitNum（与平台 UB 相关），再用 N = unitNum*2 构造整除场景
    TilingInfo probe;
    ASSERT_TRUE(RunTiling(800, ge::DT_FLOAT, 1, probe));
    const ArangeTilingData* p = AsTiling(probe);
    uint32_t unitNum = p->unitNum;
    ASSERT_GT(unitNum, 0u);
    // unitNum 来自 ub_unit_size/4，且 ub_unit_size 32B 对齐 -> unitNum 是 8 的倍数（fp32 alignNum=8）
    // 故 N=unitNum*2 时 totalBlocks=N/8 整除，tailLength=N，tailLength%unitNum==0
    ASSERT_EQ(unitNum % 8u, 0u);
    const uint32_t N = unitNum * 2;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT, 1, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 1u);
    EXPECT_EQ(t->tailLength, N);     // 单核全量，N 已 8 对齐
    EXPECT_EQ(t->tailUnitLoops, 2u); // N/unitNum = 2 整除
    EXPECT_EQ(t->tailTailNum, 0u);   // 恰好对齐，无尾块
    CheckMulticoreInvariants(info, N, ge::DT_FLOAT);
}

// ===== 4) 从 out shape 读 N（不读 start/end/step 数值）=====

TEST_F(ArangeTiling, totalnum_from_out_shape_only)
{
    const uint32_t N = 333;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->totalNum, N); // totalNum 严格等于 out shape size（与标量数值无关）
}

// out shape 改变 -> totalNum 随之改变（再次印证 N 来源是 out shape）
TEST_F(ArangeTiling, totalnum_tracks_out_shape)
{
    TilingInfo a, b;
    ASSERT_TRUE(RunTiling(123, ge::DT_FLOAT, 24, a));
    ASSERT_TRUE(RunTiling(4567, ge::DT_FLOAT, 24, b));
    EXPECT_EQ(AsTiling(a)->totalNum, 123u);
    EXPECT_EQ(AsTiling(b)->totalNum, 4567u);
}

// ===== 5) dtype -> TilingKey 映射（FP32→MODE_1，其余→MODE_0）+ dtype_size =====

TEST_F(ArangeTiling, tilingkey_fp32_is_mode1)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling(800, ge::DT_FLOAT, 24, info));
    EXPECT_NE(info.tilingKey, 0); // FP32 -> MODE_1（非 0）
    EXPECT_EQ(AsTiling(info)->dtypeSize, 4u);
}

TEST_F(ArangeTiling, tilingkey_fp16_is_mode0)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling(800, ge::DT_FLOAT16, 24, info));
    EXPECT_EQ(info.tilingKey, 0); // 其余 -> MODE_0（0）
    EXPECT_EQ(AsTiling(info)->dtypeSize, 2u);
}

TEST_F(ArangeTiling, tilingkey_bf16_is_mode0)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling(800, ge::DT_BF16, 24, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_EQ(AsTiling(info)->dtypeSize, 2u);
}

TEST_F(ArangeTiling, tilingkey_int32_is_mode0)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling(800, ge::DT_INT32, 24, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_EQ(AsTiling(info)->dtypeSize, 4u);
}

TEST_F(ArangeTiling, tilingkey_int64_is_mode0)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling(800, ge::DT_INT64, 24, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_EQ(AsTiling(info)->dtypeSize, 8u);
}

// dtype_size 真正影响切分块数：同 N 下 alignNum 不同 -> totalBlocks 不同 -> coreNum(=min) 不同
TEST_F(ArangeTiling, dtype_size_affects_block_split)
{
    // N=160：fp32 alignNum=8 -> totalBlocks=20；int64 alignNum=4 -> totalBlocks=40；fp16 alignNum=16 -> totalBlocks=10
    // maxCore=64（足够大），coreNum=min(64,totalBlocks)=totalBlocks
    TilingInfo infoFp32, infoInt64, infoFp16;
    ASSERT_TRUE(RunTiling(160, ge::DT_FLOAT, 64, infoFp32));
    ASSERT_TRUE(RunTiling(160, ge::DT_INT64, 64, infoInt64));
    ASSERT_TRUE(RunTiling(160, ge::DT_FLOAT16, 64, infoFp16));
    EXPECT_EQ(infoFp32.blockNum, 20u);
    EXPECT_EQ(infoInt64.blockNum, 40u);
    EXPECT_EQ(infoFp16.blockNum, 10u);
}

// =====================================================================================
// 窄整型 Tiling 分支 UT 覆盖（int8 / uint8 / int16）
// =====================================================================================

// ===== 6) 窄整型 dtype -> TilingKey 映射 + dtype_size（int8=1 / uint8=1 / int16=2）=====
// int8/uint8/int16 全部走 MODE_0(Cast 路径，tilingKey=0)，仅 FP32 走 MODE_1(非 0)。

TEST_F(ArangeTiling, tilingkey_int8_is_mode0)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling(800, ge::DT_INT8, 24, info));
    EXPECT_EQ(info.tilingKey, 0);             // int8 -> MODE_0（Cast 路径）
    EXPECT_EQ(AsTiling(info)->dtypeSize, 1u); // int8 dtype_size = 1
}

TEST_F(ArangeTiling, tilingkey_uint8_is_mode0)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling(800, ge::DT_UINT8, 24, info));
    EXPECT_EQ(info.tilingKey, 0);             // uint8 -> MODE_0（Cast 路径）
    EXPECT_EQ(AsTiling(info)->dtypeSize, 1u); // uint8 dtype_size = 1
}

TEST_F(ArangeTiling, tilingkey_int16_is_mode0)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling(800, ge::DT_INT16, 24, info));
    EXPECT_EQ(info.tilingKey, 0);             // int16 -> MODE_0（Cast 路径）
    EXPECT_EQ(AsTiling(info)->dtypeSize, 2u); // int16 dtype_size = 2
}

// 对照锚点：唯独 FP32 走 MODE_1（非 0），窄整型均不应误落 MODE_1
TEST_F(ArangeTiling, only_fp32_is_mode1_narrow_int_all_mode0)
{
    TilingInfo infoFp32, infoI8, infoU8, infoI16;
    ASSERT_TRUE(RunTiling(800, ge::DT_FLOAT, 24, infoFp32));
    ASSERT_TRUE(RunTiling(800, ge::DT_INT8, 24, infoI8));
    ASSERT_TRUE(RunTiling(800, ge::DT_UINT8, 24, infoU8));
    ASSERT_TRUE(RunTiling(800, ge::DT_INT16, 24, infoI16));
    EXPECT_NE(infoFp32.tilingKey, 0); // 仅 FP32 = MODE_1
    EXPECT_EQ(infoI8.tilingKey, 0);
    EXPECT_EQ(infoU8.tilingKey, 0);
    EXPECT_EQ(infoI16.tilingKey, 0);
}

// dtype_size 真实影响 32B 对齐块切分：同 N 下 alignNum 不同 -> totalBlocks 不同 -> coreNum(=min) 不同
// int8/uint8 alignNum=32（32B 装 32 个 1B 元素）；int16 alignNum=16；对照 fp32 alignNum=8。
TEST_F(ArangeTiling, narrow_int_dtype_size_affects_block_split)
{
    // N=960：int8 alignNum=32 -> totalBlocks=30；int16 alignNum=16 -> totalBlocks=60；fp32 alignNum=8 ->
    // totalBlocks=120 maxCore=128（足够大，> 各 totalBlocks），coreNum=min(128,totalBlocks)=totalBlocks
    const uint32_t N = 960;
    TilingInfo infoI8, infoU8, infoI16, infoFp32;
    ASSERT_TRUE(RunTiling(N, ge::DT_INT8, 128, infoI8));
    ASSERT_TRUE(RunTiling(N, ge::DT_UINT8, 128, infoU8));
    ASSERT_TRUE(RunTiling(N, ge::DT_INT16, 128, infoI16));
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT, 128, infoFp32));
    EXPECT_EQ(infoI8.blockNum, 30u);    // ceil(960/32)=30
    EXPECT_EQ(infoU8.blockNum, 30u);    // uint8 同 int8（alignNum=32）
    EXPECT_EQ(infoI16.blockNum, 60u);   // ceil(960/16)=60
    EXPECT_EQ(infoFp32.blockNum, 120u); // ceil(960/8)=120
    // 块数与 dtype_size 强关联：dtype_size 越小，单块装的元素越多，块数越少
    EXPECT_EQ(AsTiling(infoI8)->dtypeSize, 1u);
    EXPECT_EQ(AsTiling(infoI16)->dtypeSize, 2u);
}

// ===== 7) UB 红线断言：unitNum = ub_unit_size / max(dtypeSize, 4) =====
// 核心：dtypeSize ≤ 4 的所有 dtype，分母都被 max(.,4) 钳到 4，故 unitNum 必【全部相等】。
// 若红线 bug 复现（unitNum=ub_unit_size/dtypeSize），int8 会是 fp32 的 4 倍、int16 是 2 倍，断言失败。

// int8 / uint8 / int16 的 unitNum 必须等于 fp32 的 unitNum（按 FP32 字节统一切，不随 1B/2B 放大）
TEST_F(ArangeTiling, ub_redline_unitnum_capped_at_fp32_narrow_int)
{
    TilingInfo infoFp32, infoI8, infoU8, infoI16;
    ASSERT_TRUE(RunTiling(100000, ge::DT_FLOAT, 24, infoFp32));
    ASSERT_TRUE(RunTiling(100000, ge::DT_INT8, 24, infoI8));
    ASSERT_TRUE(RunTiling(100000, ge::DT_UINT8, 24, infoU8));
    ASSERT_TRUE(RunTiling(100000, ge::DT_INT16, 24, infoI16));

    uint32_t unitFp32 = AsTiling(infoFp32)->unitNum;
    ASSERT_GT(unitFp32, 0u);
    // 红线核心断言：窄整型 unitNum 与 fp32 完全相等（max(dtypeSize,4)=4 统一分母）
    EXPECT_EQ(AsTiling(infoI8)->unitNum, unitFp32)
        << "int8 unitNum 应按 FP32(4B) 封顶，与 fp32 相等；若放大说明 UB 红线修正回归";
    EXPECT_EQ(AsTiling(infoU8)->unitNum, unitFp32) << "uint8 unitNum 应按 FP32(4B) 封顶，与 fp32 相等";
    EXPECT_EQ(AsTiling(infoI16)->unitNum, unitFp32) << "int16 unitNum 应按 FP32(4B) 封顶，与 fp32 相等";
}

// 反例守护：若曾用 unitNum=ub_unit_size/dtypeSize，int8 会是 fp32 的约 4 倍。
// 这里显式断言 int8 unitNum 严格【不】等于 fp32 的 4 倍（修正后不会放大）。
TEST_F(ArangeTiling, ub_redline_unitnum_not_blown_up_int8)
{
    TilingInfo infoFp32, infoI8;
    ASSERT_TRUE(RunTiling(100000, ge::DT_FLOAT, 24, infoFp32));
    ASSERT_TRUE(RunTiling(100000, ge::DT_INT8, 24, infoI8));
    uint32_t unitFp32 = AsTiling(infoFp32)->unitNum;
    uint32_t unitI8 = AsTiling(infoI8)->unitNum;
    ASSERT_GT(unitFp32, 0u);
    EXPECT_NE(unitI8, unitFp32 * 4u); // bug 复现时 int8 = fp32*4，正常时相等
    EXPECT_EQ(unitI8, unitFp32);      // 正常：相等
}

// UB 预算回归：4 份 FP32 中间 + 2 份 outQueue(双缓冲) 占用必须 <= 184KB（DAV_2201 向量可用）。
// 用实际产出的 unitNum + dtypeSize 复算占用上界，逐 dtype 验证不爆 UB。
TEST_F(ArangeTiling, ub_budget_within_184kb_all_narrow_int)
{
    constexpr uint64_t UB_VEC_AVAIL = 184ull * 1024; // DAV_2201 向量可用上界
    const ge::DataType dtypes[] = {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_FLOAT};
    for (ge::DataType dt : dtypes) {
        TilingInfo info;
        ASSERT_TRUE(RunTiling(100000, dt, 24, info));
        const ArangeTilingData* t = AsTiling(info);
        uint32_t unitNum = t->unitNum;
        uint32_t dtypeSize = t->dtypeSize;
        ASSERT_GT(unitNum, 0u);
        // Cast 路径 UB 占用上界：4 份 FP32 中间(calc_init/step/temp/out) + 2 份 outQueue 双缓冲
        uint64_t fp32Mid = 4ull * unitNum * FP32_SIZE;
        uint64_t outBuf = 2ull * unitNum * dtypeSize;
        uint64_t total = fp32Mid + outBuf;
        EXPECT_LE(total, UB_VEC_AVAIL) << "dtype=" << static_cast<int>(dt) << " unitNum=" << unitNum << " UB 占用 "
                                       << total << " 超 184KB（红线修正未生效会在此爆）";
    }
}

// ===== 8) 窄整型下多核 former/tail 切分仍正确 =====

// int8 大 N 全核：alignNum=32，多核 former/tail 切分 + formerNum + 上/下取整 + CalcUnitLoops 全字段
TEST_F(ArangeTiling, multicore_int8_large)
{
    const uint32_t N = 100000; // int8 alignNum=32 -> totalBlocks=ceil(100000/32)=3125
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_INT8, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_GT(t->coreNum, 1u); // 大 N 下应开多核
    CheckMulticoreInvariants(info, N, ge::DT_INT8);
}

// uint8 大 N 全核：与 int8 同 alignNum=32，独立验证 uint8 分支不漏
TEST_F(ArangeTiling, multicore_uint8_large)
{
    const uint32_t N = 80000; // uint8 alignNum=32 -> totalBlocks=2500
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_UINT8, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_GT(t->coreNum, 1u);
    CheckMulticoreInvariants(info, N, ge::DT_UINT8);
}

// int16 大 N 全核：alignNum=16
TEST_F(ArangeTiling, multicore_int16_large)
{
    const uint32_t N = 65536; // int16 alignNum=16 -> totalBlocks=4096
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_INT16, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_GT(t->coreNum, 1u);
    CheckMulticoreInvariants(info, N, ge::DT_INT16);
}

// 窄整型不整除场景：构造 totalBlocks 不整除 coreNum -> 必有 former 段，验证 formerNum/上取整/下取整
// int8 alignNum=32。N=2336 -> totalBlocks=ceil(2336/32)=73；maxCore=8 -> coreNum=8，73%8=1
TEST_F(ArangeTiling, multicore_int8_not_divisible_former)
{
    const uint32_t N = 2336;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_INT8, 8, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 8u);
    EXPECT_EQ(t->formerNum, 1u);                              // 73 % 8 = 1
    EXPECT_EQ(t->formerLength, ceil_div_blocks(73, 8) * 32u); // ceil(73/8)=10 块 * 32 = 320
    EXPECT_EQ(t->tailLength, (73u / 8u) * 32u);               // 73/8=9 块 * 32 = 288
    CheckMulticoreInvariants(info, N, ge::DT_INT8);
}

// int16 不整除场景：alignNum=16。N=1168 -> totalBlocks=ceil(1168/16)=73；maxCore=8 -> 73%8=1
TEST_F(ArangeTiling, multicore_int16_not_divisible_former)
{
    const uint32_t N = 1168;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_INT16, 8, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 8u);
    EXPECT_EQ(t->formerNum, 1u);                              // 73 % 8 = 1
    EXPECT_EQ(t->formerLength, ceil_div_blocks(73, 8) * 16u); // ceil(73/8)=10 块 * 16 = 160
    EXPECT_EQ(t->tailLength, (73u / 8u) * 16u);               // 73/8=9 块 * 16 = 144
    CheckMulticoreInvariants(info, N, ge::DT_INT16);
}

// 窄整型小 shape 退化：totalBlocks < maxCore -> coreNum=totalBlocks（int8 1B 下尤其少块）
// int8 alignNum=32。N=96 -> totalBlocks=ceil(96/32)=3；maxCore=24 -> coreNum=min(24,3)=3
TEST_F(ArangeTiling, narrow_int_small_shape_core_capped_by_blocks)
{
    const uint32_t N = 96;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_INT8, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 3u);     // 受 totalBlocks=3 限制（< 24）
    EXPECT_EQ(t->formerNum, 0u);   // 3 % 3 == 0
    EXPECT_EQ(t->tailLength, 32u); // 每核 1 个 32B 块 = 32 int8
    CheckMulticoreInvariants(info, N, ge::DT_INT8);
}

// 窄整型 N=1 单核：int8 一块 32B 放大到 32 元素
TEST_F(ArangeTiling, narrow_int_n_equals_1_single_core)
{
    const uint32_t N = 1;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_INT8, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 1u); // totalBlocks=1 -> min(24,1)=1
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(t->formerNum, 0u);
    EXPECT_EQ(t->tailLength, 32u); // int8 1 个 32B 块 = 32 元素
    EXPECT_EQ(t->tailUnitLoops, 1u);
    CheckMulticoreInvariants(info, N, ge::DT_INT8);
}

// =====================================================================================
// op_host 全覆盖补缺：补齐未覆盖分支（无回归）
//   补：① switch default 分支（非枚举 dtype → dtype_size=2）；② N=0 兜底（totalBlocks=0→1）；
//      ③ bf16/int32/int64 的 not-divisible former 段 + 小 shape 退化；
//      ④ 全 8 dtype 的 dtypeSize/TilingKey 矩阵逐类断言；⑤ 极大 N 全核。
// =====================================================================================

// ===== 9) switch default 分支：非枚举 dtype（如 DT_DOUBLE）落 default → dtype_size=2、MODE_0 =====
// arange_tiling.cpp switch 对未列出的 dtype 走 default: dtype_size=DTYPE_SIZE2, tilingkey=0(MODE_0)。
// 用 DT_DOUBLE（不在 switch case 列表）触发 default 分支。
TEST_F(ArangeTiling, switch_default_branch_unlisted_dtype)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling(800, ge::DT_DOUBLE, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(info.tilingKey, 0); // default → MODE_0
    EXPECT_EQ(t->dtypeSize, 2u);  // default → DTYPE_SIZE2
    // default 分支下多核切分仍须自洽（alignNum=32/2=16，按 dtypeSize=2 复算）
    CheckMulticoreInvariants(info, 800, ge::DT_DOUBLE);
}

// ===== 10) N=0 兜底：out shape size=0 时 totalBlocks=0→1，coreNum>=1，不崩、字段自洽 =====
// totalBlocks==0 时强制为 1（caller 理论不传 0，但 tiling 须健壮）。
TEST_F(ArangeTiling, n_equals_0_bottom_guard)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling(0, ge::DT_FLOAT, 24, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->totalNum, 0u); // totalNum 仍如实记 0
    EXPECT_GE(t->coreNum, 1u);  // 兜底至少 1 核
    EXPECT_EQ(info.blockNum, static_cast<size_t>(t->coreNum));
    ASSERT_FALSE(info.workspaceSizes.empty());
    EXPECT_EQ(info.workspaceSizes[0], 0);
    // totalBlocks 兜底为 1：单核 tailLength = 1 个 32B 块 = 8 fp32
    EXPECT_EQ(t->coreNum, 1u);
    EXPECT_EQ(t->formerNum, 0u);
    EXPECT_EQ(t->tailLength, 8u);
}

// ===== 11) bf16 not-divisible former 段 =====
// bf16 dtypeSize=2, alignNum=16。N=1168 → totalBlocks=ceil(1168/16)=73；maxCore=8 → 73%8=1
TEST_F(ArangeTiling, multicore_bf16_not_divisible_former)
{
    const uint32_t N = 1168;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_BF16, 8, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 8u);
    EXPECT_EQ(t->formerNum, 1u);                              // 73 % 8 = 1
    EXPECT_EQ(t->formerLength, ceil_div_blocks(73, 8) * 16u); // ceil(73/8)=10 块 * 16 = 160
    EXPECT_EQ(t->tailLength, (73u / 8u) * 16u);               // 73/8=9 块 * 16 = 144
    CheckMulticoreInvariants(info, N, ge::DT_BF16);
}

// ===== 12) int32 not-divisible former 段 =====
// int32 dtypeSize=4, alignNum=8。N=584 → totalBlocks=ceil(584/8)=73；maxCore=8 → 73%8=1
TEST_F(ArangeTiling, multicore_int32_not_divisible_former)
{
    const uint32_t N = 584;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_INT32, 8, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 8u);
    EXPECT_EQ(t->formerNum, 1u);
    EXPECT_EQ(t->formerLength, ceil_div_blocks(73, 8) * 8u); // ceil(73/8)=10 块 * 8 = 80
    EXPECT_EQ(t->tailLength, (73u / 8u) * 8u);               // 9 块 * 8 = 72
    CheckMulticoreInvariants(info, N, ge::DT_INT32);
}

// ===== 13) int64 not-divisible former 段 =====
// int64 dtypeSize=8, alignNum=4。N=292 → totalBlocks=ceil(292/4)=73；maxCore=8 → 73%8=1
TEST_F(ArangeTiling, multicore_int64_not_divisible_former)
{
    const uint32_t N = 292;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_INT64, 8, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_EQ(t->coreNum, 8u);
    EXPECT_EQ(t->formerNum, 1u);
    EXPECT_EQ(t->formerLength, ceil_div_blocks(73, 8) * 4u); // 10 块 * 4 = 40
    EXPECT_EQ(t->tailLength, (73u / 8u) * 4u);               // 9 块 * 4 = 36
    CheckMulticoreInvariants(info, N, ge::DT_INT64);
}

// ===== 14) 全 8 dtype 的 dtypeSize / TilingKey 矩阵逐类汇总断言 =====
// 把分散在多个 tilingkey_* 用例的「dtype→(dtypeSize, mode)」契约收敛为一张矩阵表逐类断言，
// 防止任一 dtype 的 dtype_size 或 mode 回归（全覆盖审视点）。
TEST_F(ArangeTiling, full_dtype_size_and_tilingkey_matrix)
{
    struct Row {
        ge::DataType dt;
        uint32_t expSize;
        bool expMode1;
    };
    const Row rows[] = {
        {ge::DT_FLOAT, 4, true}, // 唯一 MODE_1（纯 FP32 直算）
        {ge::DT_FLOAT16, 2, false}, {ge::DT_BF16, 2, false},  {ge::DT_INT8, 1, false},  {ge::DT_UINT8, 1, false},
        {ge::DT_INT16, 2, false},   {ge::DT_INT32, 4, false}, {ge::DT_INT64, 8, false},
    };
    for (const Row& r : rows) {
        TilingInfo info;
        ASSERT_TRUE(RunTiling(800, r.dt, 24, info)) << "dtype=" << static_cast<int>(r.dt);
        const ArangeTilingData* t = AsTiling(info);
        EXPECT_EQ(t->dtypeSize, r.expSize) << "dtype=" << static_cast<int>(r.dt) << " dtypeSize 回归";
        if (r.expMode1) {
            EXPECT_NE(info.tilingKey, 0) << "dtype=" << static_cast<int>(r.dt) << " 应为 MODE_1(非0)";
        } else {
            EXPECT_EQ(info.tilingKey, 0) << "dtype=" << static_cast<int>(r.dt) << " 应为 MODE_0(0)";
        }
    }
}

// ===== 15) 极大 N 全核：接近 2^23 元素，验证多核切分在大规模下不溢出、字段自洽 =====
// fp32 alignNum=8。N=8000000 → totalBlocks=1000000，远超核数 → 满核 + former/tail 均衡。
TEST_F(ArangeTiling, multicore_fp32_extreme_large_n)
{
    const uint32_t N = 8000000;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_FLOAT, 48, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_GT(t->coreNum, 1u);  // 大 N 必满核
    EXPECT_LE(t->coreNum, 48u); // 不超过传入 maxCore
    // 极大 N 下 unitLoops 必然 > 1（单 UB 块装不下每核元素）
    EXPECT_GT(t->tailUnitLoops, 1u);
    CheckMulticoreInvariants(info, N, ge::DT_FLOAT);
}

// ===== 16) int8 极大 N（1B 尾轴 + 满核 + 多 UB 循环）=====
// int8 alignNum=32。N=4000000 → totalBlocks=125000，满核。
TEST_F(ArangeTiling, multicore_int8_extreme_large_n)
{
    const uint32_t N = 4000000;
    TilingInfo info;
    ASSERT_TRUE(RunTiling(N, ge::DT_INT8, 48, info));
    const ArangeTilingData* t = AsTiling(info);
    EXPECT_GT(t->coreNum, 1u);
    EXPECT_GT(t->tailUnitLoops, 1u);
    CheckMulticoreInvariants(info, N, ge::DT_INT8);
}
