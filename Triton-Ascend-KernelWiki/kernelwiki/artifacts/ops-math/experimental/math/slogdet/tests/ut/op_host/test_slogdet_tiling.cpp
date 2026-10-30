/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file test_slogdet_tiling.cpp
 * \brief Slogdet op_host Tiling UT（迭代一核心路径 + 迭代二 Tiling 分支覆盖；FULL 路径 MEM_STRATEGY=0）。
 *
 * 覆盖点（迭代一）：
 *   - FULL 路径（n 小，全驻留）：matSizeN / matrixNumCount / blockSize=n / blockNum=1；
 *   - batch 按核切分：needCoreNum = min(coreNum, matrixNumCount)（SetBlockDim）；
 *   - workspace = 16MB（WS_SYS_SIZE）；
 *   - TilingKey 选择（ASCENDC_TPL_SEL_PARAM，fp32 + MEM_STRATEGY=0）；
 *   - 非法 shape（rank<2 / n=0）→ GRAPH_FAILED。
 *
 * 覆盖点（迭代二 新增 Tiling 分支覆盖，P0）：
 *   - FULL 路径不同 n 规模分支（小 n / 中 n 16/32/64 / n=1 边界）；
 *   - 不同 batch 分布（单矩阵 / batch / 多维 batch / batch>核数封顶 / 核数=batch 恰好）；
 *   - N_RESIDENT_MAX 计算：n <= residentMax 走 FULL（if 分支） vs n > residentMax（else 分支），
 *     用受控 ubSize 精确触发两个分支；
 *   - MEM_STRATEGY 选择逻辑、epsSingular n 无关性。
 *
 * 覆盖点（迭代三 全 TilingKey 分支覆盖，P0）：
 *   - **BLOCKED 路径转正（MEM_STRATEGY=1）**：n > N_RESIDENT_MAX 真正下发 BLOCKED（不回退 FULL）：
 *       · TilingKey = 256（MEM_STRATEGY 编码第 8 位 1<<8），与 FULL(0) 不同（不回退 FULL 实证）；
 *       · blockSize = COL_BLOCK(64)，与 n 解耦（FULL 路径 blockSize=n）；
 *       · workspace = max(16MB, needCore*n*n*4)：覆盖系统预留主导 与 BLOCKED slot 需求主导 两分支；
 *       · BLOCKED + batch 按核切分；
 *   - MEM_STRATEGY 选择逻辑参数化（同 n=128 在大/小 UB 下分别命中 FULL/BLOCKED 两 TilingKey）；
 *   - epsSingular **最终锁定 1e-30**（小绝对阈值，移除 kernel 相对阈值 maxAbs 扫描；host ComputeEps 返回 1e-30），
 *     断言与 n 无关、与路径无关。
 *
 * TilingData 字段断言使用底层 ExecuteTiling（拿到 raw bytes 后按 SlogdetTilingData 解读），
 * 避免比对含 float(epsSingular) 的字符串脆弱性。
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "../../../op_kernel/slogdet_tiling_data.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace gert;

namespace SlogdetUT {

static const std::string OP_NAME = "Slogdet";

// FULL 路径下 16MB 系统 workspace（与 op_host/arch32/slogdet_tiling.cpp WS_SYS_SIZE 一致）
constexpr size_t WS_SYS_SIZE = 16U * 1024U * 1024U;

// FULL 路径（fp32 + MEM_STRATEGY=0）在本 UT 框架下观测到的稳定 TilingKey。
// ASCENDC_TPL_SEL_PARAM 的键编码在 UT host 框架下解析为 0（实测锁定，见 slogdet_tiling_key_full_stable）；
// 作为回归守卫：FULL 路径下不同 batch/n 的 TilingKey 必须一致且等于该锁定值。
constexpr int64_t SLOGDET_FULL_TILING_KEY = 0;
// BLOCKED 路径（fp32 + MEM_STRATEGY=1）TilingKey：MEM_STRATEGY 编码在第 8 位 → 1<<8 = 256（实测上板 _256）。
constexpr int64_t SLOGDET_BLOCKED_TILING_KEY = 256;

// epsSingular 锁定值（迭代三据全量 ST 对照 torch fp32 最终锁定为小绝对阈值 1e-30，与 n 无关）：
// 见 op_host/arch32/slogdet_tiling.cpp ComputeEps（穿刺 slogdet_illcond_precision 实证 1e-12/相对阈值过激）。
constexpr float SLOGDET_EPS_SINGULAR = 1e-30f;

// 默认 UB（262144=256KB）下 ResolveResidentMax 推导的 N_RESIDENT_MAX（实测：248）。
// 受控小 UB 下的 residentMax：ub=32768→72，ub=16384→40（用于精确触发 FULL/large-n 两分支）。
constexpr uint32_t RESIDENT_MAX_DEFAULT_UB = 248U;

struct SlogdetCompileInfo {};

class SlogdetTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SlogdetTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SlogdetTiling TearDown" << std::endl;
    }
};

// 从 vector<int64_t> 构造 StorageShape（origin == storage），规避只接受 initializer_list 的构造限制。
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (int64_t d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

// 构造单输入(self) + 双输出(signOut/logOut) 的 TilingContextPara。
// self.shape = batchShape + [n, n]；输出 shape = batchShape。
static gert::TilingContextPara MakePara(SlogdetCompileInfo* compileInfo,
                                        const std::vector<int64_t>& selfShape,
                                        const std::vector<int64_t>& outShape,
                                        uint64_t coreNum = 64,
                                        uint64_t ubSize = 262144)
{
    gert::StorageShape self = MakeStorageShape(selfShape);
    gert::StorageShape sign = MakeStorageShape(outShape);
    gert::StorageShape log = MakeStorageShape(outShape);
    std::vector<gert::TilingContextPara::TensorDescription> inputs(
        {{self, ge::DT_FLOAT, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::TensorDescription> outputs(
        {{sign, ge::DT_FLOAT, ge::FORMAT_ND}, {log, ge::DT_FLOAT, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    return gert::TilingContextPara(OP_NAME, inputs, outputs, attrs, compileInfo, coreNum, ubSize, 4096);
}

// FULL 路径单矩阵（无 batch）：self=[6,6] ⇒ matrixNumCount=1，needCore=1
TEST_F(SlogdetTiling, slogdet_tiling_full_single_matrix)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {6, 6}, {});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    ASSERT_GE(info.tilingDataSize, sizeof(SlogdetTilingData));
    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 6U);
    EXPECT_EQ(td->matrixNumCount, 1UL);
    EXPECT_EQ(td->blockSize, 6U);   // FULL: blockSize = n
    EXPECT_EQ(td->blockNum, 1U);    // FULL: blockNum = 1
    EXPECT_GT(td->epsSingular, 0.0f);
    // 迭代三：epsSingular 最终锁定为小绝对阈值 1e-30（host ComputeEps 返回常量，kernel 直接 |piv|<=eps 判奇异）
    EXPECT_FLOAT_EQ(td->epsSingular, SLOGDET_EPS_SINGULAR);

    // batch=1 ⇒ needCoreNum = min(64, 1) = 1
    EXPECT_EQ(info.blockNum, 1U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(static_cast<size_t>(info.workspaceSizes[0]), WS_SYS_SIZE);
}

// FULL 路径 batch（核数充足）：self=[3,6,6] ⇒ matrixNumCount=3，needCore=3
TEST_F(SlogdetTiling, slogdet_tiling_full_batch_core_enough)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {3, 6, 6}, {3});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 6U);
    EXPECT_EQ(td->matrixNumCount, 3UL);
    EXPECT_EQ(td->blockSize, 6U);
    EXPECT_EQ(td->blockNum, 1U);

    // matrixNumCount=3 < coreNum=64 ⇒ needCoreNum = 3
    EXPECT_EQ(info.blockNum, 3U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(static_cast<size_t>(info.workspaceSizes[0]), WS_SYS_SIZE);
}

// batch > coreNum：matrixNumCount=100，coreNum=8 ⇒ needCoreNum = min(8,100) = 8（核数封顶）
TEST_F(SlogdetTiling, slogdet_tiling_batch_exceeds_core)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {100, 4, 4}, {100}, /*coreNum=*/8, /*ubSize=*/262144);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 4U);
    EXPECT_EQ(td->matrixNumCount, 100UL);
    EXPECT_EQ(td->blockNum, 1U);

    // needCoreNum = min(coreNum=8, matrixNumCount=100) = 8
    EXPECT_EQ(info.blockNum, 8U);
}

// 多维 batch：self=[2,3,5,5] ⇒ matrixNumCount = 2*3 = 6
TEST_F(SlogdetTiling, slogdet_tiling_multi_dim_batch)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {2, 3, 5, 5}, {2, 3});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 5U);
    EXPECT_EQ(td->matrixNumCount, 6UL);  // 2 * 3
    EXPECT_EQ(td->blockSize, 5U);
    EXPECT_EQ(td->blockNum, 1U);
    EXPECT_EQ(info.blockNum, 6U);
}

// n=1 边界（reduce 轴长度为 1）：self=[1,1] ⇒ matSizeN=1, matrixNumCount=1
TEST_F(SlogdetTiling, slogdet_tiling_n1_boundary)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {1, 1}, {});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 1U);
    EXPECT_EQ(td->matrixNumCount, 1UL);
    EXPECT_EQ(td->blockSize, 1U);
    EXPECT_EQ(td->blockNum, 1U);
    EXPECT_EQ(info.blockNum, 1U);
}

// 大 n（n > residentMax，触发 else 分支）：迭代二仍 FULL-only，回退 FULL（blockNum=1）。
// 受控 ubSize=16384(16KB) → ResolveResidentMax=40；n=64 > 40 精确命中 large-n else 分支。
// 迭代三：large-n（n > residentMax）真正下发 BLOCKED（MEM_STRATEGY=1），不再回退 FULL。
TEST_F(SlogdetTiling, slogdet_tiling_large_n_blocked)
{
    SlogdetCompileInfo compileInfo;
    // residentMax(ub=16384)=40 < n=64 → 进入 else（large-n）分支 → BLOCKED。
    auto para = MakePara(&compileInfo, {64, 64}, {}, /*coreNum=*/64, /*ubSize=*/16384);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 64U);
    EXPECT_EQ(td->matrixNumCount, 1UL);
    // 迭代三：BLOCKED 路径 blockSize = COL_BLOCK(64)，blockNum=1
    EXPECT_EQ(td->blockSize, 64U);
    EXPECT_EQ(td->blockNum, 1U);
    EXPECT_EQ(info.blockNum, 1U);
    // large-n → MEM_STRATEGY=1 → BLOCKED TilingKey（与 FULL 路径不同）
    EXPECT_EQ(info.tilingKey, SLOGDET_BLOCKED_TILING_KEY);
    // BLOCKED workspace = needCoreNum * n*n fp32（≥ 64*64*4 = 16384B；本例 needCore=1）
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_GE(static_cast<size_t>(info.workspaceSizes[0]),
              static_cast<size_t>(1U) * 64U * 64U * sizeof(float));
}

// TilingKey 选择：FULL 路径（fp32 + MEM_STRATEGY=0）应生成一个稳定的 TilingKey。
// ASCENDC_TPL_SEL_PARAM 内部编码不手算，此处断言其 == 首个 FULL 用例发现的稳定值。
TEST_F(SlogdetTiling, slogdet_tiling_key_full_stable)
{
    SlogdetCompileInfo compileInfo;
    auto para1 = MakePara(&compileInfo, {6, 6}, {});
    auto para2 = MakePara(&compileInfo, {3, 6, 6}, {3});

    TilingInfo info1;
    TilingInfo info2;
    ASSERT_TRUE(ExecuteTiling(para1, info1));
    ASSERT_TRUE(ExecuteTiling(para2, info2));

    // FULL 路径（MEM_STRATEGY=0）下，不同 batch / n 的 TilingKey 应一致（仅由模板参数决定）。
    EXPECT_EQ(info1.tilingKey, info2.tilingKey);
    // 锁定为 FULL 路径稳定键值（由构建实测确认，见下方常量）。
    EXPECT_EQ(info1.tilingKey, SLOGDET_FULL_TILING_KEY);
}

// 非法 shape：rank<2（self=[4]）⇒ Tiling 返回 GRAPH_FAILED
TEST_F(SlogdetTiling, slogdet_tiling_invalid_rank_lt2)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {4}, {});
    // 期望 Tiling 失败
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// MED-1（review 4.2）：n 上界校验。BLOCKED 列 gather DataCopyPad blockCount(uint16,≤4095)=n-k≤n，
//   n>4095 静默越界 → host tiling 显式返回 GRAPH_FAILED（不静默错误，不崩溃）。
//   n=4096（恰超上界 SLOGDET_MAX_N=4095）⇒ 期望 Tiling 返回 GRAPH_FAILED。
TEST_F(SlogdetTiling, slogdet_tiling_n_over_upper_bound_rejected)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {4096, 4096}, {});
    ExecuteTestCase(para, ge::GRAPH_FAILED);  // n=4096 > 4095 → 拒绝（正确返回错误而非崩溃）
}

// MED-1 边界（in）：n=4095（恰为安全上界）⇒ tiling 接受（GRAPH_SUCCESS），matSizeN 正确写入。
//   注：UT 仅跑 tiling（不下发 kernel），故 BLOCKED workspace/选路成立即通过；功能由 ST 验证至 n=512。
TEST_F(SlogdetTiling, slogdet_tiling_n_at_upper_bound_accepted)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {4095, 4095}, {}, /*coreNum=*/1, /*ubSize=*/262144);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 4095U);
    EXPECT_EQ(td->matrixNumCount, 1UL);
}

// ============================================================================
// 迭代二：Tiling 分支覆盖扩充（FULL 路径不同 n 规模 / batch 分布 / N_RESIDENT_MAX / eps floor）
// ============================================================================

// FULL 路径中 n 规模分支：默认 UB（residentMax=248）下 n=16/32/64 均 <= residentMax → FULL（blockNum=1）。
// 参数化覆盖三个中等规模，确认 blockSize=n、blockNum=1、MEM_STRATEGY=0（TilingKey 稳定）。
class SlogdetTilingMidN : public testing::TestWithParam<uint32_t> {};

TEST_P(SlogdetTilingMidN, full_mid_n_scales)
{
    const uint32_t n = GetParam();
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {static_cast<int64_t>(n), static_cast<int64_t>(n)}, {});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    // n <= residentMax(248) → FULL 路径
    ASSERT_LE(n, RESIDENT_MAX_DEFAULT_UB);
    EXPECT_EQ(td->matSizeN, n);
    EXPECT_EQ(td->matrixNumCount, 1UL);
    EXPECT_EQ(td->blockSize, n);   // FULL: blockSize = n
    EXPECT_EQ(td->blockNum, 1U);   // FULL: blockNum = 1
    EXPECT_FLOAT_EQ(td->epsSingular, SLOGDET_EPS_SINGULAR);
    EXPECT_EQ(info.blockNum, 1U);  // 单矩阵 → needCore=1
    EXPECT_EQ(info.tilingKey, SLOGDET_FULL_TILING_KEY);  // MEM_STRATEGY=0
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(static_cast<size_t>(info.workspaceSizes[0]), WS_SYS_SIZE);
}

INSTANTIATE_TEST_SUITE_P(MidNScales, SlogdetTilingMidN, testing::Values(16U, 32U, 64U));

// N_RESIDENT_MAX if 分支边界（n 恰好 <= residentMax）：受控 ubSize=16384 → residentMax=40。
// n=40 恰好命中 if 分支（n <= residentMax），FULL（blockNum=1）。
TEST_F(SlogdetTiling, slogdet_tiling_resident_boundary_in)
{
    SlogdetCompileInfo compileInfo;
    // residentMax(ub=16384)=40；n=40 <= 40 → if 分支 FULL。
    auto para = MakePara(&compileInfo, {40, 40}, {}, /*coreNum=*/64, /*ubSize=*/16384);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 40U);
    EXPECT_EQ(td->matrixNumCount, 1UL);
    EXPECT_EQ(td->blockSize, 40U);
    EXPECT_EQ(td->blockNum, 1U);
    EXPECT_EQ(info.blockNum, 1U);
    EXPECT_EQ(info.tilingKey, SLOGDET_FULL_TILING_KEY);
}

// N_RESIDENT_MAX else 分支（n 略 > residentMax）：受控 ubSize=16384 → residentMax=40。
// 迭代三：n=41 > 40 → else（large-n）分支 → BLOCKED（MEM_STRATEGY=1，blockSize=COL_BLOCK=64）。
TEST_F(SlogdetTiling, slogdet_tiling_resident_boundary_out)
{
    SlogdetCompileInfo compileInfo;
    // residentMax(ub=16384)=40；n=41 > 40 → else 分支 → BLOCKED。
    auto para = MakePara(&compileInfo, {41, 41}, {}, /*coreNum=*/64, /*ubSize=*/16384);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 41U);
    EXPECT_EQ(td->matrixNumCount, 1UL);
    // 迭代三 BLOCKED：blockSize = COL_BLOCK(64)，blockNum=1
    EXPECT_EQ(td->blockSize, 64U);
    EXPECT_EQ(td->blockNum, 1U);
    EXPECT_EQ(info.blockNum, 1U);
    EXPECT_EQ(info.tilingKey, SLOGDET_BLOCKED_TILING_KEY);
}

// 核数 == batch 恰好相等：matrixNumCount=8，coreNum=8 ⇒ needCoreNum=8（边界，不封顶不留空）。
TEST_F(SlogdetTiling, slogdet_tiling_core_equals_batch)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {8, 5, 5}, {8}, /*coreNum=*/8, /*ubSize=*/262144);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matrixNumCount, 8UL);
    // needCoreNum = min(8, 8) = 8
    EXPECT_EQ(info.blockNum, 8U);
}

// 多维 batch（3 维 batch）：self=[2,2,2,3,3] ⇒ matrixNumCount = 2*2*2 = 8，n=3。
TEST_F(SlogdetTiling, slogdet_tiling_three_dim_batch)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {2, 2, 2, 3, 3}, {2, 2, 2});

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 3U);
    EXPECT_EQ(td->matrixNumCount, 8UL);  // 2*2*2
    EXPECT_EQ(td->blockSize, 3U);
    EXPECT_EQ(td->blockNum, 1U);
    EXPECT_EQ(info.blockNum, 8U);
}

// epsSingular n 无关性：迭代三最终锁定为小绝对阈值 1e-30（host 不再下发 n·FLT_EPS 相对阈值）。
// 多个 n（1 / 5 / 64 / 200）下 epsSingular 必须完全一致 == SLOGDET_EPS_SINGULAR（1e-30）。
TEST_F(SlogdetTiling, slogdet_tiling_eps_floor_n_independent)
{
    SlogdetCompileInfo compileInfo;
    const int64_t ns[] = {1, 5, 64, 200};
    for (int64_t n : ns) {
        auto para = MakePara(&compileInfo, {n, n}, {});
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "n=" << n;
        const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
        EXPECT_FLOAT_EQ(td->epsSingular, SLOGDET_EPS_SINGULAR) << "n=" << n;
    }
}

// 非法 shape：n=0（self=[3,0,0]）⇒ Tiling 返回 GRAPH_FAILED（matSizeN==0 校验）。
TEST_F(SlogdetTiling, slogdet_tiling_invalid_n_zero)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {3, 0, 0}, {3});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// ============================================================================
// 迭代三：BLOCKED 路径（MEM_STRATEGY=1）全分支覆盖 + 不回退 FULL 实证
// ============================================================================

// BLOCKED 不回退 FULL 的实证：同一 n（大 n）在不同 UB 下分别命中 FULL / BLOCKED，
// 两者 TilingKey 必须不同（BLOCKED=256 ≠ FULL=0），且 blockSize 不同（FULL=n vs BLOCKED=COL_BLOCK=64），
// 证明 large-n 走 BLOCKED 是真实分发而非静默回退 FULL。
TEST_F(SlogdetTiling, slogdet_tiling_blocked_not_fallback_full)
{
    SlogdetCompileInfo compileInfo;
    // n=64：大 UB（默认 262144，residentMax=248）→ FULL；小 UB（16384，residentMax=40）→ BLOCKED。
    auto paraFull = MakePara(&compileInfo, {64, 64}, {}, /*coreNum=*/64, /*ubSize=*/262144);
    auto paraBlocked = MakePara(&compileInfo, {64, 64}, {}, /*coreNum=*/64, /*ubSize=*/16384);

    TilingInfo full;
    TilingInfo blocked;
    ASSERT_TRUE(ExecuteTiling(paraFull, full));
    ASSERT_TRUE(ExecuteTiling(paraBlocked, blocked));

    const auto* tdFull = reinterpret_cast<const SlogdetTilingData*>(full.tilingData.get());
    const auto* tdBlocked = reinterpret_cast<const SlogdetTilingData*>(blocked.tilingData.get());

    // FULL 路径：blockSize=n=64，TilingKey=0
    EXPECT_EQ(tdFull->blockSize, 64U);
    EXPECT_EQ(full.tilingKey, SLOGDET_FULL_TILING_KEY);
    // BLOCKED 路径：blockSize=COL_BLOCK=64（恰好与 n 同值，但语义为列块宽度），TilingKey=256
    EXPECT_EQ(tdBlocked->blockSize, 64U);
    EXPECT_EQ(blocked.tilingKey, SLOGDET_BLOCKED_TILING_KEY);
    // 关键：两路径 TilingKey 不同 → BLOCKED 是真实分发，非回退 FULL
    EXPECT_NE(blocked.tilingKey, full.tilingKey);
}

// BLOCKED blockSize 语义实证：n 显著大于 COL_BLOCK 时，BLOCKED blockSize 锁定为 COL_BLOCK(64)，
// 不随 n 变化（FULL 路径 blockSize=n 会变）。用 n=200（>residentMax(16384)=40）验证。
TEST_F(SlogdetTiling, slogdet_tiling_blocked_blocksize_is_col_block)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {200, 200}, {}, /*coreNum=*/64, /*ubSize=*/16384);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 200U);
    EXPECT_EQ(td->matrixNumCount, 1UL);
    // BLOCKED：blockSize = COL_BLOCK(64)，与 n(200) 解耦
    EXPECT_EQ(td->blockSize, 64U);
    EXPECT_EQ(td->blockNum, 1U);
    EXPECT_EQ(info.tilingKey, SLOGDET_BLOCKED_TILING_KEY);
    EXPECT_FLOAT_EQ(td->epsSingular, SLOGDET_EPS_SINGULAR);  // eps 锁定不随路径变化
}

// BLOCKED workspace = 16MB 系统预留主导分支（needCore*n*n*4 <= 16MB）：
// n=64, needCore=1 → blockedSlots=16384 < 16MB → workspace 取系统预留 16MB（精确值）。
TEST_F(SlogdetTiling, slogdet_tiling_blocked_workspace_sys_dominant)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {64, 64}, {}, /*coreNum=*/64, /*ubSize=*/16384);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    EXPECT_EQ(info.tilingKey, SLOGDET_BLOCKED_TILING_KEY);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    // blockedSlots = 1 * 64*64*4 = 16384 < 16MB → 取 max → 16MB
    EXPECT_EQ(static_cast<size_t>(info.workspaceSizes[0]), WS_SYS_SIZE);
}

// BLOCKED workspace = BLOCKED slot 需求主导分支（needCore*n*n*4 > 16MB）：
// self=[17,512,512]，coreNum=64 → needCore=min(64,17)=17；residentMax(16384)=40 < 512 → BLOCKED。
// blockedSlots = 17 * 512*512*4 = 17,825,792 B > 16MB → workspace 取 BLOCKED slot 需求（精确值）。
TEST_F(SlogdetTiling, slogdet_tiling_blocked_workspace_slot_dominant)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {17, 512, 512}, {17}, /*coreNum=*/64, /*ubSize=*/16384);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 512U);
    EXPECT_EQ(td->matrixNumCount, 17UL);
    EXPECT_EQ(td->blockSize, 64U);   // COL_BLOCK
    EXPECT_EQ(info.tilingKey, SLOGDET_BLOCKED_TILING_KEY);
    // needCore = min(64, 17) = 17
    EXPECT_EQ(info.blockNum, 17U);
    // workspace = needCore * n*n*4 = 17 * 512*512*4 = 17825792 > 16MB
    const size_t expectWs = static_cast<size_t>(17U) * 512U * 512U * sizeof(float);
    ASSERT_GT(expectWs, WS_SYS_SIZE);  // 自检：确为 slot 主导分支
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(static_cast<size_t>(info.workspaceSizes[0]), expectWs);
}

// BLOCKED + batch 按核切分：self=[8,100,100]（n=100>residentMax(16384)=40 → BLOCKED），
// coreNum=4 → needCore=min(4,8)=4。验证 BLOCKED 路径下 batch 切分正常 + workspace 按 needCore 计。
TEST_F(SlogdetTiling, slogdet_tiling_blocked_batch_core_split)
{
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {8, 100, 100}, {8}, /*coreNum=*/4, /*ubSize=*/16384);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 100U);
    EXPECT_EQ(td->matrixNumCount, 8UL);
    EXPECT_EQ(td->blockSize, 64U);   // BLOCKED COL_BLOCK
    EXPECT_EQ(info.tilingKey, SLOGDET_BLOCKED_TILING_KEY);
    // needCore = min(coreNum=4, batch=8) = 4
    EXPECT_EQ(info.blockNum, 4U);
    // workspace = max(16MB, needCore*n*n*4) = max(16MB, 4*100*100*4=160000) = 16MB
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(static_cast<size_t>(info.workspaceSizes[0]), WS_SYS_SIZE);
}

// MEM_STRATEGY 选择逻辑全覆盖（参数化）：同一 n=128，
//   - 大 UB（262144, residentMax=248 >= 128）→ FULL（key=0, blockSize=128）；
//   - 小 UB（16384, residentMax=40 < 128）→ BLOCKED（key=256, blockSize=64）。
// 验证 MEM_STRATEGY 完全由「n vs residentMax」决定，两个 TilingKey 分支都被覆盖。
struct MemStrategyCase {
    uint64_t ubSize;
    int64_t expectTilingKey;
    uint32_t expectBlockSize;
};
class SlogdetMemStrategy : public testing::TestWithParam<MemStrategyCase> {};

TEST_P(SlogdetMemStrategy, n128_strategy_by_ub)
{
    const auto& c = GetParam();
    SlogdetCompileInfo compileInfo;
    auto para = MakePara(&compileInfo, {128, 128}, {}, /*coreNum=*/64, c.ubSize);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    const auto* td = reinterpret_cast<const SlogdetTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->matSizeN, 128U);
    EXPECT_EQ(td->blockSize, c.expectBlockSize);
    EXPECT_EQ(info.tilingKey, c.expectTilingKey);
    EXPECT_FLOAT_EQ(td->epsSingular, SLOGDET_EPS_SINGULAR);
}

INSTANTIATE_TEST_SUITE_P(
    Strategy, SlogdetMemStrategy,
    testing::Values(
        MemStrategyCase{262144U, SLOGDET_FULL_TILING_KEY, 128U},     // FULL（n<=residentMax 248）
        MemStrategyCase{16384U, SLOGDET_BLOCKED_TILING_KEY, 64U}));  // BLOCKED（n>residentMax 40）

}  // namespace SlogdetUT
