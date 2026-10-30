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
 * \file test_split_v_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/split_v_tiling_arch35.h"

using namespace std;
using namespace ge;
class SplitVTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SplitVTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SplitVTiling TearDown" << std::endl; }
};

TEST_F(SplitVTiling, SplitV_test_tiling_001)
{
    optiling::SplitVCompileInfo compileInfo = {32, 0, 253952};
    int32_t size_splits = 1820;
    int32_t split_dim = 0;
    gert::TilingContextPara tilingContextPara(
        "SplitV",
        {
            {{{1820, 232}, {1820, 232}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &size_splits},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
        },
        {
            {{{1820, 232}, {1820, 232}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(1))}, &compileInfo);
    uint64_t expectTilingKey = 104;
    string expectTilingData = "253952 0 0 1 1 1 1 0 0 1 0 13195 13216 13195 13216 32 32 32 0 0 422240 422240 0"
                              " -1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

namespace {
// SplitVTilingData 全为 int64 字段, 按声明顺序紧密排布。
// 索引: mBlockCount=6, gSize=9, nBlockCount=16, realCoreNum=17。
constexpr size_t IDX_M_BLOCK_COUNT = 6;
constexpr size_t IDX_N_BLOCK_COUNT = 16;
constexpr size_t IDX_REAL_CORE_NUM = 17;

int64_t GetTilingField(const TilingInfo& info, size_t idx)
{
    const int64_t* data = reinterpret_cast<const int64_t*>(info.tilingData.get());
    return data[idx];
}
} // namespace

// 高 split 块数 + N 轴为奇数: 旧规则下只有 m=coreNum,n=1 能取 delta=0, 会把所有核压到 M 轴。
// 新的 computeCost 模型应允许切 N (nBlockCount > 1), 降低单核 split 处理次数。
// 构造条件: numSplit_ > MAX_COL_OFFSET_COUNT(128) 以绕过 SIMT 分支, 且非等长以绕过 SameLen 分支,
// 从而进入 CalBlockTilingParams 双轴切核逻辑。
TEST_F(SplitVTiling, SplitV_test_tiling_split_balance_odd_n)
{
    optiling::SplitVCompileInfo compileInfo = {32, 0, 253952};
    // split_dim=1, 沿 dim1 切成 130 个非等长块: 129 个 64 + 1 个 65 = 8321 (奇数)
    const int32_t splitCnt = 130;
    std::vector<int32_t> sizeSplits(splitCnt, 64);
    sizeSplits[splitCnt - 1] = 65;      // 合轴后 N=8321 为奇数
    const int32_t nDim = 129 * 64 + 65; // 8321
    int32_t split_dim = 1;
    gert::TilingContextPara tilingContextPara(
        "SplitV",
        {
            {{{64, nDim}, {64, nDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{splitCnt}, {splitCnt}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeSplits.data()},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
        },
        {
            {{{64, nDim}, {64, nDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(splitCnt))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    int64_t nBlockCount = GetTilingField(tilingInfo, IDX_N_BLOCK_COUNT);
    int64_t mBlockCount = GetTilingField(tilingInfo, IDX_M_BLOCK_COUNT);
    // 关键断言: 不再被 delta=0 锁死在 n=1
    EXPECT_GT(nBlockCount, 1);
    // 不应退化为纯 M 轴全切
    EXPECT_LT(mBlockCount, 32);
}

// 大 M 小 N + split 块少: 切 N 会抬高 mTimes 而 maxSplitCnt 已经很小,
// computeCost 模型应保持少切 N (nBlockCount 接近 1), 避免过度切分。
TEST_F(SplitVTiling, SplitV_test_tiling_split_balance_small_split)
{
    optiling::SplitVCompileInfo compileInfo = {32, 0, 253952};
    // split_dim=1, 输入 [8192, 4], 沿 dim1 切成 2 块 (2 + 2)
    int32_t sizeSplits[2] = {2, 2};
    int32_t split_dim = 1;
    gert::TilingContextPara tilingContextPara(
        "SplitV",
        {
            {{{8192, 4}, {8192, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeSplits},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
        },
        {
            {{{8192, 4}, {8192, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    int64_t nBlockCount = GetTilingField(tilingInfo, IDX_N_BLOCK_COUNT);
    int64_t realCoreNum = GetTilingField(tilingInfo, IDX_REAL_CORE_NUM);
    // split 块少时不应为了均衡而过度切 N
    EXPECT_LE(nBlockCount, 2);
    // 仍应充分利用多核
    EXPECT_GT(realCoreNum, 1);
}

// 诊断: 大量小 split 块 (每块 n 很小) 场景下, CalBlockSplitTwoAxis 第 517 行的
// N = CeilDiv(colN*dtype, 512) 把 N 方向切核数硬性压到极小 (与 numSplit_ 脱钩)。
// 构造: numSplit_=200, 每块 size_splits=2, DT_FLOAT(4B) => colN=400,
// N 上界 = CeilDiv(400*4, 512) = CeilDiv(1600, 512) = 4, 远小于 200 个 split 块。
//
// 用例 A「小 M + 多小 split」: M=4, coreNum=32。
// 已知边界: 仅放宽第 517 行 N 上界无法救此场景, 因为第 518 行
// cores = CeilDiv(4*401*4, 8K) = 1 在候选枚举前已把总核数砍到 1。
// 此处记录残留边界, realCoreNum 仍为 1, 非回归失败。放宽 cores 口径超出「只改 517」范围。
TEST_F(SplitVTiling, SplitV_test_tiling_many_small_split_small_m)
{
    optiling::SplitVCompileInfo compileInfo = {32, 0, 253952};
    const int32_t splitCnt = 200;
    std::vector<int32_t> sizeSplits(splitCnt, 2);
    sizeSplits[splitCnt - 1] = 3;     // 非等长, 绕过 SameLen 分支; 合轴后 N = 199*2 + 3 = 401
    const int32_t nDim = 199 * 2 + 3; // 401
    const int32_t mDim = 4;           // 小 M
    int32_t split_dim = 1;
    gert::TilingContextPara tilingContextPara(
        "SplitV",
        {
            {{{mDim, nDim}, {mDim, nDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{splitCnt}, {splitCnt}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeSplits.data()},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
        },
        {
            {{{mDim, nDim}, {mDim, nDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(splitCnt))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    int64_t nBlockCount = GetTilingField(tilingInfo, IDX_N_BLOCK_COUNT);
    int64_t mBlockCount = GetTilingField(tilingInfo, IDX_M_BLOCK_COUNT);
    int64_t realCoreNum = GetTilingField(tilingInfo, IDX_REAL_CORE_NUM);
    std::cout << "[diag small_m] nBlockCount=" << nBlockCount << " mBlockCount=" << mBlockCount
              << " realCoreNum=" << realCoreNum << std::endl;
    // 核心证据: 尽管有 200 个 split 块, 512B 上界 + 小 M 使核严重用不满。
    EXPECT_LT(realCoreNum, 32);
}

// 用例 B「大 M + 多小 split」: M=8192, coreNum=32。
// 平均 split 字节 = 401*4/200 = 8B < 128B(cacheline), 触发第 517 行放宽:
// 对齐粒度从 512B 降到 128B, N 切核上界从 CeilDiv(401*4,512)=4 提升到 CeilDiv(401,128/4)=13。
// 预期 nBlockCount > 4, 验证放宽生效, split 块可分到更多 N 核。
TEST_F(SplitVTiling, SplitV_test_tiling_many_small_split_large_m)
{
    optiling::SplitVCompileInfo compileInfo = {32, 0, 253952};
    const int32_t splitCnt = 200;
    std::vector<int32_t> sizeSplits(splitCnt, 2);
    sizeSplits[splitCnt - 1] = 3;
    const int32_t nDim = 199 * 2 + 3; // 401
    const int32_t mDim = 8192;        // 大 M
    int32_t split_dim = 1;
    gert::TilingContextPara tilingContextPara(
        "SplitV",
        {
            {{{mDim, nDim}, {mDim, nDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{splitCnt}, {splitCnt}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeSplits.data()},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
        },
        {
            {{{mDim, nDim}, {mDim, nDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(splitCnt))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    int64_t nBlockCount = GetTilingField(tilingInfo, IDX_N_BLOCK_COUNT);
    int64_t mBlockCount = GetTilingField(tilingInfo, IDX_M_BLOCK_COUNT);
    int64_t realCoreNum = GetTilingField(tilingInfo, IDX_REAL_CORE_NUM);
    std::cout << "[diag large_m] nBlockCount=" << nBlockCount << " mBlockCount=" << mBlockCount
              << " realCoreNum=" << realCoreNum << std::endl;
    // 放宽后 N 切核上界 = CeilDiv(401, 128/4) = 13, 应突破原 512B 对齐的上界 4。
    EXPECT_GT(nBlockCount, 4);
    // computeCost 模型据此在 M/N 间重新权衡, 应充分利用多核 (具体核数由模型决定, 不强绑 32)。
    EXPECT_GT(realCoreNum, 1);
    (void)mBlockCount;
}

// huge split(numSplit > HUGE_SPLIT_NUM=4096) + 大 M(>= HUGE_M=64) 回归:
// computeCost 乘法模型在大 M 下会偏向切 M 压低 mTimes, 导致 N 轴分核骤减性能回退。
// 硬覆盖兜底应强制选 m==SPLIT_DIM_INDEX(=2) 的候选, M 只切 2 份其余核全给 N。
// 构造: numSplit_=4097, M=64, 前 4096 块 size_splits=2 + 尾块 3, FLOAT, colN=8195。
// avgSplitBytes = 8195*4/4097 ≈ 8B < 128 触发放宽, N 上界 = CeilDiv(8195, 128/4) = 257;
// cores = min(32, CeilDiv(64*8195*4, 8K)) = 32, FindAllPossibleCutCnt(32) 含 m=2 => m=2,n=16 候选存在。
TEST_F(SplitVTiling, SplitV_test_tiling_huge_split_large_m_force_m2)
{
    optiling::SplitVCompileInfo compileInfo = {32, 0, 253952};
    const int32_t splitCnt = 4097;
    std::vector<int32_t> sizeSplits(splitCnt, 2);
    sizeSplits[splitCnt - 1] = 3;      // 非等长, 绕过 SameLen 分支; colN = 4096*2 + 3 = 8195
    const int32_t nDim = 4096 * 2 + 3; // 8195
    const int32_t mDim = 64;           // >= HUGE_M
    int32_t split_dim = 1;
    gert::TilingContextPara tilingContextPara(
        "SplitV",
        {
            {{{mDim, nDim}, {mDim, nDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{splitCnt}, {splitCnt}}, ge::DT_INT32, ge::FORMAT_ND, true, sizeSplits.data()},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
        },
        {
            {{{mDim, nDim}, {mDim, nDim}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(splitCnt))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    int64_t nBlockCount = GetTilingField(tilingInfo, IDX_N_BLOCK_COUNT);
    int64_t mBlockCount = GetTilingField(tilingInfo, IDX_M_BLOCK_COUNT);
    int64_t realCoreNum = GetTilingField(tilingInfo, IDX_REAL_CORE_NUM);
    std::cout << "[diag huge_split] nBlockCount=" << nBlockCount << " mBlockCount=" << mBlockCount
              << " realCoreNum=" << realCoreNum << std::endl;
    // 硬覆盖强制 M 只切 2 份。
    EXPECT_EQ(mBlockCount, 2);
    // 其余核全给 N 轴, 充分分核。
    EXPECT_GT(nBlockCount, 4);
    (void)realCoreNum;
}
