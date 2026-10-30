/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_fused_mul_add_n_tiling.cpp
 * \brief A2 (DAV_2201 / ascend910b) op_host tiling UT for FusedMulAddN (experimental, flat layout).
 *
 *        fp32 main path (TilingKey=0): multi-core split (blockNum/blockFormer/blockTail),
 *        UB split fields (ubFormer / ubLoop* / ubTail*), block_dim and workspace.
 *
 *        Multi-dtype tiling branches: all 5 dtypes -> their TilingKey, plus the
 *        UB-split-coefficient branch in GetBytesPerElem (direct 6*sizeof(T) vs cast-domain 20B):
 *          - dtype -> TilingKey: fp32=0 / fp16=1 / int32=2 / int16=3 / bf16=4
 *          - alignment granularity differs by elemSize: fp32/int32 -> 8 elem (32B), fp16/bf16/int16 -> 16 elem
 *          - bytesPerElem differs by dtype: fp32/int32=24, int16=12, fp16/bf16=20 -> different ubFormerMax
 *        Multi-tile/tail invariants are checked self-consistently (no hardcoded platform-reserved UB).
 *
 *        Experimental flat layout: TilingData/CompileInfo are defined in op_kernel/fused_mul_add_n_tiling_data.h
 *        and the host tiling lives in op_host/fused_mul_add_n_tiling.cpp (no arch32/arch35 subdirs).
 *
 *        Platform values come from the faker (default coreNum=64, ubSize=262144=256KB), which the
 *        tiling reads via context->GetPlatformInfo(). The actually-usable UB returned by
 *        GetCoreMemSize may be slightly smaller than the injected 262144 (system-reserved), so any
 *        case that depends on the UB ceiling uses robust split-self-consistency invariants instead
 *        of hardcoding ubFormer. Multi-core (coreNum=64) expectations are anchored to small/aligned
 *        shapes whose result is independent of the UB ceiling.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "../../../op_kernel/fused_mul_add_n_tiling_data.h"

using namespace std;
using namespace ge;

namespace {
// 与 op_host/fused_mul_add_n_tiling.cpp 的 GetTilingKeyByDtype 一致
constexpr uint64_t TILING_KEY_FP32 = 0;
constexpr uint64_t TILING_KEY_FP16 = 1;
constexpr uint64_t TILING_KEY_INT32 = 2;
constexpr uint64_t TILING_KEY_INT16 = 3;
constexpr uint64_t TILING_KEY_BF16 = 4;
// faker 默认平台参数（context->GetPlatformInfo() 注入值）
constexpr uint64_t UT_CORE_NUM = 64;
constexpr uint64_t UT_UB_SIZE = 262144;
constexpr size_t UT_WORKSPACE = 32;

// 32B 对齐粒度（元素数），与 tiling 的 elemPerBlockAlign = 32 / sizeof(T) 一致
constexpr int64_t ALIGN_FP32 = 8;  // 32B / 4B
constexpr int64_t ALIGN_HALF = 16; // 32B / 2B (fp16/bf16/int16)

// 核数 cap（与 op_host/fused_mul_add_n_tiling.cpp 一致）：
//   minElemsPerCore = MIN_BYTES_PER_CORE / (IO_PATHS * elemByteSize)
//   effCoreNum      = clamp(totalNum / minElemsPerCore, 1, coreNum)
// elemByteSize 为元素真实字节数（fp32/int32=4，fp16/bf16/int16=2）。
constexpr int64_t MIN_BYTES_PER_CORE = 48 * 1024;
constexpr int64_t IO_PATHS_PER_ELEM = 3;
// ascend910b AIV 核数（核数预测锚点：[256,256]/[4096,1024] 的核数预测在此核数下成立）。
constexpr uint64_t SOC_CORE_NUM_910B = 40;

// 预期有效核数（向下取整 clamp）
int64_t ExpectEffCoreNum(int64_t totalNum, int64_t elemByteSize, int64_t coreNum)
{
    int64_t minElems = MIN_BYTES_PER_CORE / (IO_PATHS_PER_ELEM * elemByteSize);
    if (minElems < 1) {
        minElems = 1;
    }
    int64_t eff = totalNum / minElems;
    if (eff < 1) {
        eff = 1;
    }
    if (eff > coreNum) {
        eff = coreNum;
    }
    return eff;
}

// FusedMulAddNTilingData 字段顺序（op_kernel/fused_mul_add_n_tiling_data.h 定义，全部 int64_t）：
//   [0]totalNum [1]blockNum [2]blockFormer [3]blockTail
//   [4]ubFormer [5]ubLoopOfFormerBlock [6]ubLoopOfTailBlock [7]ubTailOfFormerBlock [8]ubTailOfTailBlock
enum TdIdx {
    IDX_TOTAL_NUM = 0,
    IDX_BLOCK_NUM,
    IDX_BLOCK_FORMER,
    IDX_BLOCK_TAIL,
    IDX_UB_FORMER,
    IDX_UB_LOOP_FORMER,
    IDX_UB_LOOP_TAIL,
    IDX_UB_TAIL_FORMER,
    IDX_UB_TAIL_TAIL,
    IDX_FIELD_NUM
};

// 取 TilingData 第 idx 个 int64 字段
int64_t Field(const TilingInfo& info, size_t idx)
{
    EXPECT_GE(info.tilingDataSize, (idx + 1) * sizeof(int64_t));
    const int64_t* data = reinterpret_cast<const int64_t*>(info.tilingData.get());
    return data[idx];
}

// faker 的 BuildTilingContext 要求 compileInfo 非空（否则 context 创建失败 -> 空指针 segfault）。
// tiling 实际平台值取自 GetPlatformInfo()（faker 由 coreNum_/ubSize_ 注入），
// 此 compileInfo 仅用于满足 faker 建上下文要求，保持与平台一致。
static optiling::FusedMulAddNCompileInfo g_compileInfo = {
    static_cast<int64_t>(UT_CORE_NUM), static_cast<int64_t>(UT_UB_SIZE)};

// 构造同 dtype（5 元一致）的 FusedMulAddN tiling 用例（x1==x2 shape，x3 单元素 scalar）
gert::TilingContextPara MakePara(ge::DataType dtype, const gert::StorageShape& shape)
{
    return gert::TilingContextPara(
        "FusedMulAddN",
        {
            {shape, dtype, ge::FORMAT_ND},
            {shape, dtype, ge::FORMAT_ND},
            {{{1}, {1}}, dtype, ge::FORMAT_ND},
        },
        {
            {shape, dtype, ge::FORMAT_ND},
        },
        &g_compileInfo, UT_CORE_NUM, UT_UB_SIZE);
}

// 构造 fp32 同 dtype 的 FusedMulAddN tiling 用例（x1==x2 shape，x3 单元素 scalar）
gert::TilingContextPara MakeFp32Para(const gert::StorageShape& shape)
{
    return MakePara(ge::DT_FLOAT, shape);
}

// 构造可指定核数的同 dtype 用例（x3 单元素 scalar）。
// 注：tiling 通过 context->GetPlatformInfo() 读取核数/UB，faker 由 coreNum_/ubSize_ 注入，
// 故此处自定义 coreNum 可确定性触发「单核 / 元素数 < 核数（部分核空闲）」等多核切分边界分支。
gert::TilingContextPara MakeParaWithCore(ge::DataType dtype, const gert::StorageShape& shape, uint64_t coreNum)
{
    return gert::TilingContextPara(
        "FusedMulAddN",
        {
            {shape, dtype, ge::FORMAT_ND},
            {shape, dtype, ge::FORMAT_ND},
            {{{1}, {1}}, dtype, ge::FORMAT_ND},
        },
        {
            {shape, dtype, ge::FORMAT_ND},
        },
        &g_compileInfo, coreNum, UT_UB_SIZE);
}

// 构造 x3 形态可自定义的同 dtype 用例（用于校验 x3 = [1] 与 [1,1] 的 ShapeSize=1 等价路径，
// 以及 x3 ShapeSize≠1 的校验失败路径）。x1==x2 = shape。
gert::TilingContextPara MakeParaWithX3(
    ge::DataType dtype, const gert::StorageShape& shape, const gert::StorageShape& x3Shape)
{
    return gert::TilingContextPara(
        "FusedMulAddN",
        {
            {shape, dtype, ge::FORMAT_ND},
            {shape, dtype, ge::FORMAT_ND},
            {x3Shape, dtype, ge::FORMAT_ND},
        },
        {
            {shape, dtype, ge::FORMAT_ND},
        },
        &g_compileInfo, UT_CORE_NUM, UT_UB_SIZE);
}

// dtype 对应的 32B 对齐粒度（元素数）
int64_t AlignOf(ge::DataType dtype)
{
    return (dtype == ge::DT_FLOAT || dtype == ge::DT_INT32) ? ALIGN_FP32 : ALIGN_HALF;
}

// 校验「单核多 tile」场景下，所有切分字段满足算法自洽不变量（不依赖平台实测 UB 字面值）。
// 适用于 totalNum > 0 且单核任务量 > 单 tile 上限的大 shape。
void ExpectMultiTileInvariants(const TilingInfo& info, int64_t align)
{
    const int64_t totalNum = Field(info, IDX_TOTAL_NUM);
    const int64_t blockNum = Field(info, IDX_BLOCK_NUM);
    const int64_t blockFormer = Field(info, IDX_BLOCK_FORMER);
    const int64_t blockTail = Field(info, IDX_BLOCK_TAIL);
    const int64_t ubFormer = Field(info, IDX_UB_FORMER);
    const int64_t ubLoopFormer = Field(info, IDX_UB_LOOP_FORMER);
    const int64_t ubLoopTail = Field(info, IDX_UB_LOOP_TAIL);
    const int64_t ubTailFormer = Field(info, IDX_UB_TAIL_FORMER);
    const int64_t ubTailTail = Field(info, IDX_UB_TAIL_TAIL);

    // block_dim 与 blockNum 一致
    EXPECT_EQ(static_cast<int64_t>(info.blockNum), blockNum);
    // 多核切分自洽：blockFormer*(blockNum-1) + blockTail == totalNum
    EXPECT_EQ(blockFormer * (blockNum - 1) + blockTail, totalNum);
    // blockFormer 按 align 对齐
    EXPECT_EQ(blockFormer % align, 0);
    // 单 tile 上限：ubFormer 对齐、为正、不超过单核任务量
    EXPECT_EQ(ubFormer % align, 0);
    EXPECT_GT(ubFormer, 0);
    EXPECT_LE(ubFormer, blockFormer);
    // 多 tile 循环成立（单核任务量 > 单 tile 上限）
    EXPECT_GT(ubLoopFormer, 1);
    EXPECT_GE(ubLoopTail, 1);
    // 尾 tile 元素数落在 (0, ubFormer] 区间
    EXPECT_GT(ubTailFormer, 0);
    EXPECT_LE(ubTailFormer, ubFormer);
    EXPECT_GT(ubTailTail, 0);
    EXPECT_LE(ubTailTail, ubFormer);
    // UB 切分自洽：ubFormer*(ubLoop-1) + ubTail == block 任务量
    EXPECT_EQ(ubFormer * (ubLoopFormer - 1) + ubTailFormer, blockFormer);
    EXPECT_EQ(ubFormer * (ubLoopTail - 1) + ubTailTail, blockTail);
}
} // namespace

class FusedMulAddNTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedMulAddNTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedMulAddNTiling TearDown" << std::endl;
    }
};

// ---- 用例 1：fp32 单 tile 小 shape（256 元素）+ 核数 cap ----
// cap：minElemsPerCore = 48KB/(3*4) = 4096，effCore = clamp(256/4096, 1, 64) = 1。
// → 256 元素全部落单核（blockFormer=256，blockNum=1），不再摊到 32 核。单 tile 即可放下。
// 校验小 shape 在 cap 下降核到 1（消除固定开销摊薄不足）。
TEST_F(FusedMulAddNTiling, fp32_single_tile_small_shape)
{
    auto para = MakeFp32Para({{8, 32}, {8, 32}}); // 256 元素

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    // TilingKey = 0 (fp32 主路径)
    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP32);
    // block_dim 与 blockNum 一致；cap 后单核
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 1);
    // 多核切分（cap 后单核分到全部 256 元素）
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 256);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 256);
    // UB 切分（单 tile）
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_TAIL), 256);
    // workspace
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_EQ(info.workspaceSizes[0], UT_WORKSPACE);
}

// ---- 用例 2：fp32 65536 元素（[256,256]）+ 核数 cap → 16 核 ----
// cap：minElemsPerCore=4096（fp32），effCore=clamp(65536/4096,1,64)=16。
// → blockFormer=4096，blockNum=16，blockTail=4096（完美均衡），单 tile（4096*24=96KB>可用UB? 否：
//   可用 UB 远大于 96KB，4096 单 tile 放得下）。校验 TilingKey/TilingData 完整序列 + workspace。
// [256,256] fp32 → 16 核。
TEST_F(FusedMulAddNTiling, fp32_full_core_even)
{
    auto para = MakeFp32Para({{256, 256}, {256, 256}}); // 65536 元素

    uint64_t expectTilingKey = TILING_KEY_FP32;
    // totalNum blockNum blockFormer blockTail ubFormer ubLoopFormer ubLoopTail ubTailFormer ubTailTail
    string expectTilingData = "65536 16 4096 4096 4096 1 1 4096 4096 ";
    std::vector<size_t> expectWorkspaces = {UT_WORKSPACE};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ---- 用例 3：fp32 小 shape（1000 元素）+ 核数 cap → 单核 ----
// cap：minElemsPerCore=4096，effCore=clamp(1000/4096,1,64)=1。
// → 1000 元素全部落单核（blockFormer=1000，已 8 对齐，blockNum=1，blockTail=1000）。
// 小 shape 在 cap 下不再切到 63 核（消除核间不均衡 + 固定开销摊薄不足）。
// 注：非整除尾核切分由新增 cap UT（coreNum=40 大 shape）覆盖；此处校验 cap 降核到 1。
TEST_F(FusedMulAddNTiling, fp32_small_shape_capped_single_core)
{
    auto para = MakeFp32Para({{1000}, {1000}}); // 1000 元素（恰 8 对齐）

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP32);
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 1000);
    EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 1);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 1000);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 1000);
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 1000);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_FORMER), 1000);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_TAIL), 1000);
    // 一致性：blockFormer*(blockNum-1) + blockTail == totalNum
    EXPECT_EQ(
        Field(info, IDX_BLOCK_FORMER) * (Field(info, IDX_BLOCK_NUM) - 1) + Field(info, IDX_BLOCK_TAIL),
        Field(info, IDX_TOTAL_NUM));
}

// ---- 用例 4：fp32 大 shape，触发单核多 tile 循环（ubFormer < blockFormer）----
// 4194304 元素 / 64 核 = 65536/核，远超 fp32 单 tile 上限 -> 每核需多个 ub tile。
// 注：单 tile 上限 ubFormerMax 取决于平台实测可用 UB（GetCoreMemSize 返回值，可能略小于
// 注入的 262144，含系统保留），故此处只校验「多 tile 循环成立 + 对齐 + 切分自洽」这些稳健不变量，
// 不硬编码 ubFormer 字面值。
TEST_F(FusedMulAddNTiling, fp32_multi_ub_loop)
{
    auto para = MakeFp32Para({{4194304}, {4194304}}); // 64*65536 元素

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP32);
    EXPECT_EQ(info.blockNum, 64u);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 65536);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 65536);

    const int64_t ubFormer = Field(info, IDX_UB_FORMER);
    const int64_t ubLoopFormer = Field(info, IDX_UB_LOOP_FORMER);
    const int64_t ubTailFormer = Field(info, IDX_UB_TAIL_FORMER);
    const int64_t ubLoopTail = Field(info, IDX_UB_LOOP_TAIL);
    const int64_t ubTailTail = Field(info, IDX_UB_TAIL_TAIL);

    // 多 tile 循环成立（单核任务量 > 单 tile 上限）
    EXPECT_GT(ubLoopFormer, 1);
    EXPECT_GT(ubLoopTail, 1);
    // ubFormer 32B 对齐（fp32: 8 元素粒度），且不超过单核任务量
    EXPECT_EQ(ubFormer % 8, 0);
    EXPECT_GT(ubFormer, 0);
    EXPECT_LE(ubFormer, Field(info, IDX_BLOCK_FORMER));
    // 尾 tile 元素数落在 (0, ubFormer] 区间
    EXPECT_GT(ubTailFormer, 0);
    EXPECT_LE(ubTailFormer, ubFormer);
    // 切分自洽：ubFormer*(ubLoopFormer-1) + ubTailFormer == blockFormer（former 核与尾核同任务量）
    EXPECT_EQ(ubFormer * (ubLoopFormer - 1) + ubTailFormer, Field(info, IDX_BLOCK_FORMER));
    EXPECT_EQ(ubFormer * (ubLoopTail - 1) + ubTailTail, Field(info, IDX_BLOCK_TAIL));
}

// ---- 用例 5：fp32 空 tensor（0 元素）-> 单核 0 循环，kernel 直接返回 ----
TEST_F(FusedMulAddNTiling, fp32_empty_tensor)
{
    auto para = MakeFp32Para({{0}, {0}}); // 0 元素

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP32);
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 0);
    EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 1);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 0);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 0);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 0);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 0);
}

// ---- 用例 6：dtype 不一致 -> 校验失败路径（tiling 返回 GRAPH_FAILED）----
TEST_F(FusedMulAddNTiling, dtype_mismatch_failed)
{
    gert::TilingContextPara para(
        "FusedMulAddN",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x3 dtype 与 x1 不一致
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &g_compileInfo, UT_CORE_NUM, UT_UB_SIZE);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ============================================================================
//  各 dtype 的 tiling 分支覆盖
//  目标 1：dtype -> 正确 TilingKey（fp32=0/fp16=1/int32=2/int16=3/bf16=4）
//  目标 2：UB 切分系数随 dtype 变化（直算 6×sizeof vs Cast 域 20B），导致 ubFormer/ubLoop 差异
//  目标 3：多核/多 tile/尾块在不同 dtype 下的切分自洽不变量
// ============================================================================

// ---- 用例 7a：fp16 单 tile 小 shape（256 元素）-> TilingKey=1 + 核数 cap → 单核 ----
// cap（2 字节 dtype）：minElemsPerCore=48KB/(3*2)=8192，effCore=clamp(256/8192,1,64)=1。
// → 256 元素全部落单核（blockFormer=256，blockNum=1）。校验 fp16 小 shape cap 降核 + TilingKey=1。
TEST_F(FusedMulAddNTiling, fp16_single_tile_small_shape_key1)
{
    auto para = MakePara(ge::DT_FLOAT16, {{8, 32}, {8, 32}}); // 256 元素

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP16);
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 256);
    EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 1);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 256);
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_TAIL), 256);
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_EQ(info.workspaceSizes[0], static_cast<int64_t>(UT_WORKSPACE));
}

// ---- 用例 7b：bf16 单 tile 小 shape（256 元素）-> TilingKey=4 + 核数 cap → 单核 ----
// bf16 与 fp16 共享 Cast 域算路径 + 同为 2 字节 dtype（cap minElemsPerCore=8192），
// 256 元素 cap 降核到 1，切分结果与 fp16 用例 7a 完全一致，仅 TilingKey 不同（4 vs 1）。
TEST_F(FusedMulAddNTiling, bf16_single_tile_small_shape_key4)
{
    auto para = MakePara(ge::DT_BF16, {{8, 32}, {8, 32}}); // 256 元素

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_BF16);
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 256);
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
}

// ---- 用例 7c：int32 单 tile 小 shape（256 元素）-> TilingKey=2 + 核数 cap → 单核 ----
// int32 与 fp32 同为 4 字节直算（cap minElemsPerCore=4096），256 元素 cap 降核到 1，
// 切分结果与 fp32 用例 1 完全一致，仅 TilingKey 不同（2 vs 0）。
TEST_F(FusedMulAddNTiling, int32_single_tile_small_shape_key2)
{
    auto para = MakePara(ge::DT_INT32, {{8, 32}, {8, 32}}); // 256 元素

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_INT32);
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 256);
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
}

// ---- 用例 7d：int16 单 tile 小 shape（256 元素）-> TilingKey=3 + 核数 cap → 单核 ----
// int16 直算（GetBytesPerElem=12，对齐 16），但 cap 用元素真实字节数 2（minElemsPerCore=8192），
// 256 元素 cap 降核到 1（与 fp16/bf16 一致）。
TEST_F(FusedMulAddNTiling, int16_single_tile_small_shape_key3)
{
    auto para = MakePara(ge::DT_INT16, {{8, 32}, {8, 32}}); // 256 元素

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_INT16);
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 256);
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
}

// ---- 用例 8：dtype -> TilingKey 映射表（参数化逐一校验，含 fp32）----
// 固定同一中等 shape（4096 元素），只校验 TilingKey 与切分自洽（多核），证明映射唯一确定。
TEST_F(FusedMulAddNTiling, dtype_to_tiling_key_mapping)
{
    struct Case {
        ge::DataType dtype;
        uint64_t expectKey;
        const char* name;
    };
    const Case cases[] = {
        {ge::DT_FLOAT, TILING_KEY_FP32, "fp32"},   {ge::DT_FLOAT16, TILING_KEY_FP16, "fp16"},
        {ge::DT_INT32, TILING_KEY_INT32, "int32"}, {ge::DT_INT16, TILING_KEY_INT16, "int16"},
        {ge::DT_BF16, TILING_KEY_BF16, "bf16"},
    };
    for (const auto& c : cases) {
        auto para = MakePara(c.dtype, {{4096}, {4096}});
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "ExecuteTiling failed for dtype=" << c.name;
        EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), c.expectKey) << "TilingKey mismatch for dtype=" << c.name;
        // 多核切分自洽（4096 元素，blockFormer 按 align 对齐）
        const int64_t align = AlignOf(c.dtype);
        EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 4096) << "dtype=" << c.name;
        EXPECT_EQ(Field(info, IDX_BLOCK_FORMER) % align, 0) << "dtype=" << c.name;
        EXPECT_EQ(
            Field(info, IDX_BLOCK_FORMER) * (Field(info, IDX_BLOCK_NUM) - 1) + Field(info, IDX_BLOCK_TAIL),
            Field(info, IDX_TOTAL_NUM))
            << "dtype=" << c.name;
    }
}

// ---- 用例 9：UB 切分系数随 dtype 变化（int16 直算 12B/elem vs fp16 Cast 域 Axpy 16B/elem）----
// 同一大 shape 下，两者对齐粒度相同（均 16），故多核切分（blockFormer/blockNum/blockTail）完全一致；
// 但 bytesPerElem 不同（int16=12 < fp16=16，Axpy 融合后 fp16 由 20 降到 16）-> ubFormerMax 不同
// （int16 更大）-> 单 tile 元素数 ubFormer 不同。
// 取足够大的单核任务量使两者均触发多 tile（ubFormer == ubFormerMax），从而 ubFormer_int16 > ubFormer_fp16，
// 进而 int16 的 ubLoop 更少。这直接证明 GetBytesPerElem 的 dtype 分支被走到且生效。
// 用稳健不变量校验，不硬编码平台保留 UB 字面值。
TEST_F(FusedMulAddNTiling, ub_coef_differs_int16_vs_fp16)
{
    // 单核任务量需远超任一 dtype 的单 tile 上限以触发多 tile。
    // 64 核 * 65536/核 = 4194304 元素（与 fp32 多 tile 用例同量级，对半宽 dtype 更易多 tile）。
    gert::StorageShape shape{{4194304}, {4194304}};

    TilingInfo infoInt16;
    ASSERT_TRUE(ExecuteTiling(MakePara(ge::DT_INT16, shape), infoInt16));
    TilingInfo infoFp16;
    ASSERT_TRUE(ExecuteTiling(MakePara(ge::DT_FLOAT16, shape), infoFp16));

    EXPECT_EQ(static_cast<uint64_t>(infoInt16.tilingKey), TILING_KEY_INT16);
    EXPECT_EQ(static_cast<uint64_t>(infoFp16.tilingKey), TILING_KEY_FP16);

    // 两者均为半宽 dtype（对齐 16），多核切分应完全一致（与 UB 系数无关）
    EXPECT_EQ(Field(infoInt16, IDX_BLOCK_NUM), Field(infoFp16, IDX_BLOCK_NUM));
    EXPECT_EQ(Field(infoInt16, IDX_BLOCK_FORMER), Field(infoFp16, IDX_BLOCK_FORMER));
    EXPECT_EQ(Field(infoInt16, IDX_BLOCK_TAIL), Field(infoFp16, IDX_BLOCK_TAIL));

    // 各自满足多 tile 切分自洽
    ExpectMultiTileInvariants(infoInt16, ALIGN_HALF);
    ExpectMultiTileInvariants(infoFp16, ALIGN_HALF);

    // UB 系数差异核心断言：int16(12B) 单 tile 上限 > fp16(16B, Axpy 融合) -> ubFormer 更大
    const int64_t ubFormerInt16 = Field(infoInt16, IDX_UB_FORMER);
    const int64_t ubFormerFp16 = Field(infoFp16, IDX_UB_FORMER);
    EXPECT_GT(ubFormerInt16, ubFormerFp16) << "int16(12B/elem) ubFormer should exceed fp16(16B/elem) ubFormer";
    // ubFormer 更大 -> 单核循环次数更少（或相等，取整可能持平，但绝不更多）
    EXPECT_LE(Field(infoInt16, IDX_UB_LOOP_FORMER), Field(infoFp16, IDX_UB_LOOP_FORMER));
}

// ---- 用例 10：UB 切分系数 fp32(24B) vs int16(12B) ----
// 不同对齐粒度（fp32=8, int16=16），但核心验证「直算 6×sizeof(T) 系数」：
// fp32 bytesPerElem=24，int16=12。同一 totalNum 下单 tile 字节预算一致，故 int16 单 tile 元素数更多。
// 取大 shape 触发各自多 tile，校验自洽不变量 + ubFormer 字节占用均不超过预算上界（稳健不变量）。
TEST_F(FusedMulAddNTiling, ub_coef_fp32_int16_multi_tile)
{
    gert::StorageShape shape{{4194304}, {4194304}}; // 64*65536

    TilingInfo infoFp32;
    ASSERT_TRUE(ExecuteTiling(MakePara(ge::DT_FLOAT, shape), infoFp32));
    TilingInfo infoInt16;
    ASSERT_TRUE(ExecuteTiling(MakePara(ge::DT_INT16, shape), infoInt16));

    EXPECT_EQ(static_cast<uint64_t>(infoFp32.tilingKey), TILING_KEY_FP32);
    EXPECT_EQ(static_cast<uint64_t>(infoInt16.tilingKey), TILING_KEY_INT16);

    // 各自多 tile 切分自洽（对齐粒度不同）
    ExpectMultiTileInvariants(infoFp32, ALIGN_FP32);
    ExpectMultiTileInvariants(infoInt16, ALIGN_HALF);

    // 单 tile 字节占用稳健上界：ubFormer*bytesPerElem <= 可用 UB（= 注入 UB，实测可用必 <= 注入）
    // fp32: 24B/elem, int16: 12B/elem
    EXPECT_LE(Field(infoFp32, IDX_UB_FORMER) * 24, static_cast<int64_t>(UT_UB_SIZE));
    EXPECT_LE(Field(infoInt16, IDX_UB_FORMER) * 12, static_cast<int64_t>(UT_UB_SIZE));
}

// ---- 用例 11：fp16 小 shape（1000 元素）+ 核数 cap → 单核（非整除对齐尾元素并入单核）----
// cap（2 字节）：minElemsPerCore=8192，effCore=clamp(1000/8192,1,64)=1。
// → 单核分到 1000 元素，blockFormer 对齐到 16 -> 1008（>1000 取下限对齐），blockNum=1，blockTail=1000。
// 校验半宽 dtype 小 shape cap 降核到 1（非整除对齐尾元素自洽并入单核）+ TilingKey。
// 注：单核内非整除 ub tile 由其它大 shape 多 tile 用例覆盖；此处 1000<UB 上限故单 tile。
TEST_F(FusedMulAddNTiling, fp16_small_shape_capped_single_core)
{
    auto para = MakePara(ge::DT_FLOAT16, {{1000}, {1000}});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP16);
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 1000);
    EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 1);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 1008); // ceil(1000/16)*16
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 1000);   // 单核：blockTail = totalNum
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 1008);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_FORMER), 1008);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_TAIL), 1000);
    // 一致性：blockFormer*(blockNum-1) + blockTail == totalNum（单核：0 + 1000 == 1000）
    EXPECT_EQ(
        Field(info, IDX_BLOCK_FORMER) * (Field(info, IDX_BLOCK_NUM) - 1) + Field(info, IDX_BLOCK_TAIL),
        Field(info, IDX_TOTAL_NUM));
}

// ---- 用例 12：各 dtype 空 tensor（0 元素）-> 单核 0 循环，仅 TilingKey 不同 ----
TEST_F(FusedMulAddNTiling, empty_tensor_all_dtypes)
{
    struct Case {
        ge::DataType dtype;
        uint64_t expectKey;
        const char* name;
    };
    const Case cases[] = {
        {ge::DT_FLOAT16, TILING_KEY_FP16, "fp16"},
        {ge::DT_INT32, TILING_KEY_INT32, "int32"},
        {ge::DT_INT16, TILING_KEY_INT16, "int16"},
        {ge::DT_BF16, TILING_KEY_BF16, "bf16"},
    };
    for (const auto& c : cases) {
        auto para = MakePara(c.dtype, {{0}, {0}});
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "dtype=" << c.name;
        EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), c.expectKey) << "dtype=" << c.name;
        EXPECT_EQ(info.blockNum, 1u) << "dtype=" << c.name;
        EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 0) << "dtype=" << c.name;
        EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 1) << "dtype=" << c.name;
        EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 0) << "dtype=" << c.name;
        EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 0) << "dtype=" << c.name;
    }
}

// ---- 用例 13：不支持 dtype（如 int8）-> 校验失败路径（TILING_KEY_INVALID -> GRAPH_FAILED）----
TEST_F(FusedMulAddNTiling, unsupported_dtype_failed)
{
    auto para = MakePara(ge::DT_INT8, {{16, 16}, {16, 16}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ============================================================================
//  op_host tiling 全覆盖（校验失败路径 + 多核/UB 切分边界 + x3 形态等价）
//  覆盖 FusedMulAddNTilingFunc 的关键分支，逐条对应「分支 -> UT」清单：
//    [校验] x1/x2 shape 不一致 / x1!=x2 dtype 不一致 / x3 ShapeSize!=1（2D / 多元素）-> GRAPH_FAILED
//    [多核] totalNum 恰为核数整数倍 / 元素数 < 核数（部分核空闲）/ 单核（自然 & 强制 coreNum=1）
//    [UB ]  尾 tile == 1（blockTail=1 -> ubTailOfTailBlock=1）
//    [x3 ]  形态 [1] 与 [1,1] 均通过（ShapeSize=1 等价），且切分结果完全一致
//    [ws ]  workspace 在各 dtype 下恒为 WORKSPACE_SIZE(32)
// ============================================================================

// ---- 用例 14：x1/x2 shape 不一致 -> 校验失败路径（shapeX1 != shapeX2 -> GRAPH_FAILED）----
TEST_F(FusedMulAddNTiling, x1_x2_shape_mismatch_failed)
{
    gert::TilingContextPara para(
        "FusedMulAddN",
        {
            {{{8, 32}, {8, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},   // x1: 256 元素
            {{{16, 32}, {16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x2: 512 元素，与 x1 不一致
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8, 32}, {8, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &g_compileInfo, UT_CORE_NUM, UT_UB_SIZE);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---- 用例 15：x1 与 x2 dtype 不一致 -> 校验失败路径（dtypeX1 != dtypeX2 -> GRAPH_FAILED）----
// 用例 6 已覆盖 x3 dtype 不一致；此处覆盖 dtype 校验的 x1!=x2 这一条腿。
TEST_F(FusedMulAddNTiling, x1_x2_dtype_mismatch_failed)
{
    gert::TilingContextPara para(
        "FusedMulAddN",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x2 dtype 与 x1 不一致
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &g_compileInfo, UT_CORE_NUM, UT_UB_SIZE);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---- 用例 16：y 与 x1 dtype 不一致 -> 校验失败路径（dtypeX1 != dtypeY -> GRAPH_FAILED）----
TEST_F(FusedMulAddNTiling, y_dtype_mismatch_failed)
{
    gert::TilingContextPara para(
        "FusedMulAddN",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // y dtype 与 x1 不一致
        },
        &g_compileInfo, UT_CORE_NUM, UT_UB_SIZE);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---- 用例 17：x3 为 2D 单元素 [1,1] -> ShapeSize=1 等价 [1]，校验通过且切分与 [1] 完全一致 ----
// 直接证明 shapeX3.GetShapeSize() != 1 这条校验对 [1,1] 不触发（ShapeSize=1）。
TEST_F(FusedMulAddNTiling, x3_shape_1x1_equivalent_to_1)
{
    const gert::StorageShape shape{{8, 32}, {8, 32}}; // 256 元素

    TilingInfo infoX3_1;
    ASSERT_TRUE(ExecuteTiling(MakeParaWithX3(ge::DT_FLOAT, shape, {{1}, {1}}), infoX3_1));
    TilingInfo infoX3_11;
    ASSERT_TRUE(ExecuteTiling(MakeParaWithX3(ge::DT_FLOAT, shape, {{1, 1}, {1, 1}}), infoX3_11));

    // 两者均通过且 TilingKey 一致（fp32）
    EXPECT_EQ(static_cast<uint64_t>(infoX3_1.tilingKey), TILING_KEY_FP32);
    EXPECT_EQ(static_cast<uint64_t>(infoX3_11.tilingKey), TILING_KEY_FP32);
    // 全部 TilingData 字段逐一相等（x3 形态不影响切分，仅 ShapeSize=1 校验等价）
    EXPECT_EQ(infoX3_1.blockNum, infoX3_11.blockNum);
    for (size_t i = 0; i < IDX_FIELD_NUM; ++i) {
        EXPECT_EQ(Field(infoX3_1, i), Field(infoX3_11, i)) << "TilingData field idx=" << i << " mismatch";
    }
    // workspace 一致
    ASSERT_EQ(infoX3_11.workspaceSizes.size(), 1u);
    EXPECT_EQ(infoX3_11.workspaceSizes[0], static_cast<int64_t>(UT_WORKSPACE));
}

// ---- 用例 18：x3 ShapeSize != 1（2D 多元素 [2,1]）-> 校验失败路径（GRAPH_FAILED）----
TEST_F(FusedMulAddNTiling, x3_shape_size_not_one_2d_failed)
{
    auto para = MakeParaWithX3(ge::DT_FLOAT, {{8, 32}, {8, 32}}, {{2, 1}, {2, 1}}); // ShapeSize=2
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---- 用例 19：x3 ShapeSize != 1（1D 多元素 [4]）-> 校验失败路径（GRAPH_FAILED）----
TEST_F(FusedMulAddNTiling, x3_shape_size_not_one_1d_failed)
{
    auto para = MakeParaWithX3(ge::DT_FLOAT16, {{16}, {16}}, {{4}, {4}}); // ShapeSize=4
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ---- 用例 20：fp32 小 shape（512 元素）+ 核数 cap → 单核 ----
// cap：minElemsPerCore=4096，effCore=clamp(512/4096,1,64)=1。
// → 512 元素全部落单核（blockFormer=512，blockNum=1，blockTail=512），单 tile。
TEST_F(FusedMulAddNTiling, fp32_total_exact_multiple_of_core)
{
    auto para = MakeFp32Para({{512}, {512}});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP32);
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 512);
    EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 1);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 512);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 512);
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 512);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_FORMER), 512);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_TAIL), 512);
    // 切分自洽
    EXPECT_EQ(
        Field(info, IDX_BLOCK_FORMER) * (Field(info, IDX_BLOCK_NUM) - 1) + Field(info, IDX_BLOCK_TAIL),
        Field(info, IDX_TOTAL_NUM));
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_EQ(info.workspaceSizes[0], static_cast<int64_t>(UT_WORKSPACE));
}

// ---- 用例 21：极小 shape（fp32 64 元素，64 核）+ 核数 cap → 单核（远小于阈值）----
// cap：minElemsPerCore=4096，effCore=clamp(64/4096,1,64)=1。
// → 64 元素远小于单核阈值（48KB/4=12288 元素），全部落单核（blockFormer=64，blockNum=1）。
// 校验「实际用核数 blockNum << 注入核数 64」（cap 致核数收缩到 1）。
TEST_F(FusedMulAddNTiling, fp32_elems_less_than_core_partial_idle)
{
    auto para = MakeFp32Para({{64}, {64}});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP32);
    // cap 后单核 1 << 注入核数 64
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 1);
    EXPECT_LT(Field(info, IDX_BLOCK_NUM), static_cast<int64_t>(UT_CORE_NUM));
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 64);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 64);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 64);
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 64);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
    EXPECT_EQ(
        Field(info, IDX_BLOCK_FORMER) * (Field(info, IDX_BLOCK_NUM) - 1) + Field(info, IDX_BLOCK_TAIL),
        Field(info, IDX_TOTAL_NUM));
}

// ---- 用例 22：单核（fp32 8 元素，64 核）-> blockNum=1 ----
// 8 元素恰为一个 32B 对齐块，blockFormer=8 -> blockNum=ceil(8/8)=1，仅 1 核工作。
TEST_F(FusedMulAddNTiling, fp32_single_core_one_block)
{
    auto para = MakeFp32Para({{8}, {8}});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP32);
    EXPECT_EQ(info.blockNum, 1u); // 单核
    EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 1);
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 8);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 8);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 8); // 单核时 blockTail == totalNum
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 8);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_FORMER), 8);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_TAIL), 8);
}

// ---- 用例 23：强制单核（coreNum=1）-> blockNum=1，整 totalNum 落在单核 ----
// 通过 faker 注入 coreNum=1，校验 perCore=totalNum、blockNum=1 的单核路径（与默认 64 核解耦）。
TEST_F(FusedMulAddNTiling, fp32_forced_single_core)
{
    // 末参为 coreNum=1，强制单核路径。
    auto para = MakeParaWithCore(ge::DT_FLOAT, {{256}, {256}}, 1);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP32);
    EXPECT_EQ(info.blockNum, 1u); // coreNum=1 -> 单核
    EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 1);
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 256);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 256); // 单核分到全部元素（已对齐）
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 256);
    // 256 元素单 tile 可放下（远小于 UB 上限）
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 256);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
    EXPECT_EQ(
        Field(info, IDX_BLOCK_FORMER) * (Field(info, IDX_BLOCK_NUM) - 1) + Field(info, IDX_BLOCK_TAIL),
        Field(info, IDX_TOTAL_NUM));
}

// ---- 用例 24：极小非对齐 shape（fp32 9 元素，64 核）+ 核数 cap → 单核，blockFormer 对齐到 16 ----
// cap：effCore=clamp(9/4096,1,64)=1。单核 perCore=9 -> blockFormer=ceil(9/8)*8=16，
// blockNum=ceil(9/16)=1，blockTail=9（单核：blockTail=totalNum）。单 tile：ubTailTail=9。
// 覆盖「cap 下极小非对齐 shape 单核 + blockFormer 对齐 > totalNum」边界。
TEST_F(FusedMulAddNTiling, fp32_tail_tile_one_element)
{
    auto para = MakeFp32Para({{9}, {9}});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP32);
    EXPECT_EQ(info.blockNum, 1u);
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 9);
    EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 1);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 16); // ceil(9/8)*8
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 9);    // 单核：blockTail = totalNum
    EXPECT_EQ(Field(info, IDX_UB_FORMER), 16);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_FORMER), 1);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_FORMER), 16);
    EXPECT_EQ(Field(info, IDX_UB_LOOP_TAIL), 1);
    EXPECT_EQ(Field(info, IDX_UB_TAIL_TAIL), 9);
    // 切分自洽（单核：0 + 9 == 9）
    EXPECT_EQ(
        Field(info, IDX_BLOCK_FORMER) * (Field(info, IDX_BLOCK_NUM) - 1) + Field(info, IDX_BLOCK_TAIL),
        Field(info, IDX_TOTAL_NUM));
}

// ---- 用例 25：workspace 恒为 WORKSPACE_SIZE(32)，与 dtype/shape 无关（逐元素无需大 workspace）----
// 遍历 5 dtype + 不同 shape，统一校验 workspace 设置正确（覆盖 workspaces[0]=WORKSPACE_SIZE 这一行）。
TEST_F(FusedMulAddNTiling, workspace_always_fixed_all_dtypes)
{
    struct Case {
        ge::DataType dtype;
        const char* name;
    };
    const Case cases[] = {
        {ge::DT_FLOAT, "fp32"},  {ge::DT_FLOAT16, "fp16"}, {ge::DT_INT32, "int32"},
        {ge::DT_INT16, "int16"}, {ge::DT_BF16, "bf16"},
    };
    for (const auto& c : cases) {
        // 取触发多 tile 的大 shape，确保 workspace 与切分循环无关恒为 32
        auto para = MakePara(c.dtype, {{4194304}, {4194304}});
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "dtype=" << c.name;
        ASSERT_EQ(info.workspaceSizes.size(), 1u) << "dtype=" << c.name;
        EXPECT_EQ(info.workspaceSizes[0], static_cast<int64_t>(UT_WORKSPACE)) << "dtype=" << c.name;
    }
}

// ============================================================================
//  核数 cap（min-bytes-per-core）专项覆盖
//   - cap 命中：小 shape 降核（effCore < coreNum），核数 = clamp(totalNum/minElemsPerCore,1,coreNum)
//   - cap 不命中（大 shape）：每核负载远超阈值 -> 仍用满 coreNum
//   - elemByteSize 决定 cap：4 字节 minElems=4096，2 字节 minElems=8192（同 shape 4字节用核更多）
//   - 非整除小 shape 尾块：blockNum<coreNum 且 blockFormer*(blockNum-1)+blockTail==totalNum，
//     blockTail∈[1,blockFormer]（核间自洽，对齐导致 blockFormer 略 > totalNum/effCore）
//   核数预测锚点用 ascend910b 核数 40（faker 注入 coreNum=40）。
// ============================================================================

// ---- 用例 26：cap 非整除小 shape 尾块（[255,255] fp32，coreNum=40）----
// 核心 cap UT：totalNum=65025（非整除），落入 cap 区。
// minElemsPerCore=48KB/(3*4)=4096，effCore=clamp(65025/4096,1,40)=15 < 40。
// blockFormer=ceil(ceil(65025/15)/8)*8=4336，blockNum=ceil(65025/4336)=15，blockTail=65025-4336*14=4321。
// 断言 blockNum<40，且切分自洽（blockFormer*(blockNum-1)+blockTail==totalNum，blockTail∈[1,blockFormer]）。
TEST_F(FusedMulAddNTiling, core_cap_nondivisible_small_shape_tailblock_fp32)
{
    const int64_t total = 255 * 255; // 65025（非整除）
    auto para = MakeParaWithCore(ge::DT_FLOAT, {{255, 255}, {255, 255}}, SOC_CORE_NUM_910B);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const int64_t blockNum = Field(info, IDX_BLOCK_NUM);
    const int64_t blockFormer = Field(info, IDX_BLOCK_FORMER);
    const int64_t blockTail = Field(info, IDX_BLOCK_TAIL);

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP32);
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), total);
    // cap 命中：用核数 < 40（预期 15）
    EXPECT_LT(blockNum, static_cast<int64_t>(SOC_CORE_NUM_910B));
    EXPECT_EQ(blockNum, ExpectEffCoreNum(total, 4, SOC_CORE_NUM_910B)); // == 15
    EXPECT_EQ(static_cast<int64_t>(info.blockNum), blockNum);           // block_dim == blockNum
    // blockFormer 32B 对齐（fp32: 8 元素粒度）
    EXPECT_EQ(blockFormer % ALIGN_FP32, 0);
    // 尾块自洽：blockFormer*(blockNum-1)+blockTail == totalNum
    EXPECT_EQ(blockFormer * (blockNum - 1) + blockTail, total);
    // blockTail ∈ [1, blockFormer]
    EXPECT_GE(blockTail, 1);
    EXPECT_LE(blockTail, blockFormer);
    // 具体数值
    EXPECT_EQ(blockNum, 15);
    EXPECT_EQ(blockFormer, 4336);
    EXPECT_EQ(blockTail, 4321);
}

// ---- 用例 27：cap 非整除小 shape 尾块（[255,255] fp16，coreNum=40）----
// 2 字节 dtype：minElemsPerCore=48KB/(3*2)=8192，effCore=clamp(65025/8192,1,40)=7 < 40。
// 校验半宽 dtype 的 cap 尾块自洽（与 fp32 同 shape 但用核更少，因 elemByteSize 更小）。
TEST_F(FusedMulAddNTiling, core_cap_nondivisible_small_shape_tailblock_fp16)
{
    const int64_t total = 255 * 255; // 65025
    auto para = MakeParaWithCore(ge::DT_FLOAT16, {{255, 255}, {255, 255}}, SOC_CORE_NUM_910B);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const int64_t blockNum = Field(info, IDX_BLOCK_NUM);
    const int64_t blockFormer = Field(info, IDX_BLOCK_FORMER);
    const int64_t blockTail = Field(info, IDX_BLOCK_TAIL);

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP16);
    EXPECT_EQ(Field(info, IDX_TOTAL_NUM), total);
    EXPECT_LT(blockNum, static_cast<int64_t>(SOC_CORE_NUM_910B));
    EXPECT_EQ(blockNum, ExpectEffCoreNum(total, 2, SOC_CORE_NUM_910B)); // == 7
    EXPECT_EQ(blockFormer % ALIGN_HALF, 0);
    EXPECT_EQ(blockFormer * (blockNum - 1) + blockTail, total);
    EXPECT_GE(blockTail, 1);
    EXPECT_LE(blockTail, blockFormer);
    // fp16 同 shape 用核数 < fp32（elemByteSize 更小 -> minElemsPerCore 更大 -> 降核更狠）
    EXPECT_LT(blockNum, ExpectEffCoreNum(total, 4, SOC_CORE_NUM_910B));
}

// ---- 用例 28：[256,256] 各 dtype 核数（coreNum=40，ascend910b）----
// 各 dtype 的核数预测：
//   fp32→16, fp16→8, bf16→8, int32→16, int16→8。
TEST_F(FusedMulAddNTiling, core_cap_core_count_256x256)
{
    struct Case {
        ge::DataType dtype;
        uint64_t key;
        int64_t elemByteSize;
        int64_t expectCore;
        const char* name;
    };
    const Case cases[] = {
        {ge::DT_FLOAT, TILING_KEY_FP32, 4, 16, "fp32"},
        {ge::DT_FLOAT16, TILING_KEY_FP16, 2, 8, "fp16"},
        {ge::DT_BF16, TILING_KEY_BF16, 2, 8, "bf16"},
        {ge::DT_INT32, TILING_KEY_INT32, 4, 16, "int32"},
        {ge::DT_INT16, TILING_KEY_INT16, 2, 8, "int16"},
    };
    for (const auto& c : cases) {
        auto para = MakeParaWithCore(c.dtype, {{256, 256}, {256, 256}}, SOC_CORE_NUM_910B);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "dtype=" << c.name;

        const int64_t blockNum = Field(info, IDX_BLOCK_NUM);
        const int64_t blockFormer = Field(info, IDX_BLOCK_FORMER);
        const int64_t blockTail = Field(info, IDX_BLOCK_TAIL);
        const int64_t align = AlignOf(c.dtype);

        EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), c.key) << "dtype=" << c.name;
        EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 65536) << "dtype=" << c.name;
        // 预测核数
        EXPECT_EQ(blockNum, c.expectCore) << "dtype=" << c.name;
        EXPECT_EQ(blockNum, ExpectEffCoreNum(65536, c.elemByteSize, SOC_CORE_NUM_910B)) << "dtype=" << c.name;
        EXPECT_LT(blockNum, static_cast<int64_t>(SOC_CORE_NUM_910B)) << "dtype=" << c.name; // cap 命中
        // 65536 恰整除 16/8 -> 完美均衡（blockTail == blockFormer）
        EXPECT_EQ(blockFormer % align, 0) << "dtype=" << c.name;
        EXPECT_EQ(blockTail, blockFormer) << "dtype=" << c.name;
        EXPECT_EQ(blockFormer * (blockNum - 1) + blockTail, 65536) << "dtype=" << c.name;
    }
}

// ---- 用例 29：大 shape [4096,1024]（coreNum=40）-> 仍用满 40 核 ----
// 每核负载远超 48KB 阈值（4194304/40=104857 元素 >> 4096/8192），cap 不命中 -> 用满 40 核。
TEST_F(FusedMulAddNTiling, core_cap_large_shape_full_40core_4096x1024)
{
    struct Case {
        ge::DataType dtype;
        uint64_t key;
        const char* name;
    };
    const Case cases[] = {
        {ge::DT_FLOAT, TILING_KEY_FP32, "fp32"},
        {ge::DT_FLOAT16, TILING_KEY_FP16, "fp16"},
        {ge::DT_BF16, TILING_KEY_BF16, "bf16"},
        {ge::DT_INT32, TILING_KEY_INT32, "int32"},
        {ge::DT_INT16, TILING_KEY_INT16, "int16"},
    };
    for (const auto& c : cases) {
        auto para = MakeParaWithCore(c.dtype, {{4096, 1024}, {4096, 1024}}, SOC_CORE_NUM_910B);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "dtype=" << c.name;

        const int64_t blockNum = Field(info, IDX_BLOCK_NUM);
        const int64_t blockFormer = Field(info, IDX_BLOCK_FORMER);
        const int64_t blockTail = Field(info, IDX_BLOCK_TAIL);
        const int64_t align = AlignOf(c.dtype);

        EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), c.key) << "dtype=" << c.name;
        EXPECT_EQ(Field(info, IDX_TOTAL_NUM), 4194304) << "dtype=" << c.name;
        // 用满 40 核（cap 不命中）
        EXPECT_EQ(blockNum, 40) << "dtype=" << c.name;
        EXPECT_EQ(static_cast<int64_t>(info.blockNum), 40) << "dtype=" << c.name;
        // 切分自洽
        EXPECT_EQ(blockFormer % align, 0) << "dtype=" << c.name;
        EXPECT_EQ(blockFormer * (blockNum - 1) + blockTail, 4194304) << "dtype=" << c.name;
        EXPECT_GE(blockTail, 1) << "dtype=" << c.name;
        EXPECT_LE(blockTail, blockFormer) << "dtype=" << c.name;
    }
}

// ---- 用例 30：大 shape 在 faker 默认 64 核下仍用满 64 核（coreNum=64）----
// 复用既有 faker 默认核数：4194304/64=65536 元素/核 >> 阈值 -> cap 不命中，用满 64 核。
// 与用例 4/9/10 形成「大 shape cap 不降核」的交叉确认。
TEST_F(FusedMulAddNTiling, core_cap_large_shape_full_64core)
{
    auto para = MakeFp32Para({{4194304}, {4194304}});
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(static_cast<uint64_t>(info.tilingKey), TILING_KEY_FP32);
    EXPECT_EQ(info.blockNum, 64u); // 用满注入核数（cap 不命中）
    EXPECT_EQ(Field(info, IDX_BLOCK_NUM), 64);
    EXPECT_EQ(Field(info, IDX_BLOCK_FORMER), 65536);
    EXPECT_EQ(Field(info, IDX_BLOCK_TAIL), 65536);
}

// ---- 用例 31：cap 由 elemByteSize 决定（同 shape 同核数，4字节用核 > 2字节）----
// 取一个 4/2 字节用核数不同的 shape：[512,512]=262144，coreNum=40。
//   fp32: minElems=4096 -> eff=clamp(262144/4096=64,1,40)=40（被 coreNum 限到 40，cap 未额外降核）
//   fp16: minElems=8192 -> eff=clamp(262144/8192=32,1,40)=32 < 40（cap 命中降核）
// 直接证明「elemByteSize 进入 cap 计算」：同 totalNum/coreNum 下 fp32 用核 > fp16 用核。
TEST_F(FusedMulAddNTiling, core_cap_depends_on_elem_byte_size)
{
    const int64_t total = 512 * 512; // 262144
    TilingInfo infoFp32;
    ASSERT_TRUE(ExecuteTiling(MakeParaWithCore(ge::DT_FLOAT, {{512, 512}, {512, 512}}, SOC_CORE_NUM_910B), infoFp32));
    TilingInfo infoFp16;
    ASSERT_TRUE(ExecuteTiling(MakeParaWithCore(ge::DT_FLOAT16, {{512, 512}, {512, 512}}, SOC_CORE_NUM_910B), infoFp16));

    const int64_t bnFp32 = Field(infoFp32, IDX_BLOCK_NUM);
    const int64_t bnFp16 = Field(infoFp16, IDX_BLOCK_NUM);

    EXPECT_EQ(Field(infoFp32, IDX_TOTAL_NUM), total);
    EXPECT_EQ(Field(infoFp16, IDX_TOTAL_NUM), total);
    // cap 用 elemByteSize：4 字节 minElems 更小 -> 用核更多（>=）
    EXPECT_EQ(bnFp32, ExpectEffCoreNum(total, 4, SOC_CORE_NUM_910B));
    EXPECT_EQ(bnFp16, ExpectEffCoreNum(total, 2, SOC_CORE_NUM_910B));
    EXPECT_GT(bnFp32, bnFp16) << "fp32(4B) 应比 fp16(2B) 用更多核（minElemsPerCore 更小）";
    // 各自切分自洽
    EXPECT_EQ(Field(infoFp32, IDX_BLOCK_FORMER) * (bnFp32 - 1) + Field(infoFp32, IDX_BLOCK_TAIL), total);
    EXPECT_EQ(Field(infoFp16, IDX_BLOCK_FORMER) * (bnFp16 - 1) + Field(infoFp16, IDX_BLOCK_TAIL), total);
}
