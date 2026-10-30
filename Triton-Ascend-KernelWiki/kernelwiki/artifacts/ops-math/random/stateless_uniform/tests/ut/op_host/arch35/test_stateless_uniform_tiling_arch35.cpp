/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_stateless_uniform_tiling_arch35.cpp
 * \brief StatelessUniform Tiling UT（arch35）
 *
 * TilingData 布局（RandomUnifiedSimtTilingDataStruct，68 个 int64_t）：
 *   [0]  usedCoreNum
 *   [1]  outputSize
 *   [2]  seed                    -- getSeedAndOffset 固定返回 0（kernel 从 GM 直接读取）
 *   [3]  offset                  -- 框架计算后的 counter offset 总量
 *   [4]  ubSize                  -- UniqueProcess 设置（ubSize - DcacheSize）
 *   [5]  extraInt64Param1        -- 不使用，默认 0
 *   [6]  prob(float)|extraFloat32Param1(float) -- 不使用，默认 0
 *   [7]  from (int64)            -- 公共字段，StatelessUniform 不使用，默认 0
 *   [8]  range (uint64)          -- 公共字段，StatelessUniform 不使用，默认 0
 *   [9]  fromFp32(float)|toFp32(float) -- from/to（double→float），little-endian float pair
 *   [10] splitBlockCount         -- CalcSplitBlocks 填充（小 tensor 为 1）
 *   [11..15] splitBlocks[0]      -- {numel, gmOffset, grid, totalThreads, kernelOffset}
 *   [16..20] splitBlocks[1]      -- 全 0（未使用）
 *   ...
 *   [46..50] splitBlocks[7]      -- 全 0（未使用）
 *
 * 平台参数（UT 默认值）：
 *   coreNum  = 64
 *   ubSize   = 262144 (256 KB)
 *   DcacheSize = 32768 → 有效 ubSize = 229376
 *
 * GPU 仿真常量（CalcExecutionPoliciesForBlocks）：
 *   BLOCK_SIZE = 256, MULTI_PROCESSOR_COUNT = 78, blocks_per_sm = 8
 *   MAX_GRID = 78 * 8 = 624
 *   grid = min(ceil(numel/256), 624)
 *   totalThreads = grid * 256
 *   counterOffset = ceil(numel / (256 * grid * 4)) * 4
 *
 * tilingKey 统一为 100（不再按 dtype 分发）
 *
 * SIMT 分核计算（DoSimtBlockTiling）：
 *   avgPerCore   = CeilDiv(outputSize, coreNum)
 *   numOfPerCore = CeilAlign(avgPerCore, coreAlignSize=256)
 *   usedCoreNum  = min(coreNum, CeilDiv(outputSize, numOfPerCore))
 *
 * 输入布局：
 *   [0] shape:  DT_INT64, 1D, const（shape 值由 constValue 传入）
 *   [1] seed:   DT_INT64, scalar, const（getSeedAndOffset 读取）
 *   [2] offset: DT_INT64, scalar, const（getSeedAndOffset 读取）
 *   [3] from:   DT_DOUBLE, scalar, const（UniqueProcess 读取）
 *   [4] to:     DT_DOUBLE, scalar, const（UniqueProcess 读取）
 *
 * 属性：
 *   dtype: int64（0=float32, 2=float16, 3=bfloat16）
 *
 * [9] fromFp32|toFp32 编码（little-endian float pair → int64）：
 *   fromFp32=0.0f, toFp32=1.0f → 0x3F80000000000000 = 4575657221408423936
 *
 * 测试覆盖矩阵：
 *   dtype 覆盖：
 *     float32:  test_0
 *     float16:  test_1
 *     bfloat16: test_2
 *
 *   shape 维度覆盖：
 *     1D: test_3
 *     2D: test_0~2
 *     3D: test_4
 *     4D: test_5
 *
 *   边界覆盖：
 *     极小 shape（单核）:              test_6
 *     全核（usedCoreNum=64）:          test_7
 *     非对齐尾块:                      test_8
 *
 *   非法用例：
 *     非法输出 dtype（DT_INT32）:      test_invalid_dtype
 *     非法输出 dtype（DT_DOUBLE）:     test_invalid_float64
 *     非法输入 shape dtype（DT_INT32）: test_invalid_shape_dtype
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/stateless_uniform_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// splitBlocks[1..7] 全零后缀（7 个空 SplitBlockInfo × 5 个 int64 = 35 个零）
#define SPLIT_BLOCKS_TAIL "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "

class StatelessUniformTilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "StatelessUniformTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StatelessUniformTilingTest TearDown" << std::endl;
    }
};

// ===== dtype 覆盖 =====

// case 0: float32 output, 2D shape [32,512], from=0.0, to=1.0
// outputSize=16384, usedCoreNum=64
// grid=min(ceil(16384/256),624)=64, totalThreads=16384, counterOffset=ceil(16384/65536)*4=4
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_0)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {32, 512};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{32,512},{32,512}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    // [0..6] 64 16384 0 0 229376 0 0  [7..8] from|range  [9] fromFp32|toFp32  [10] splitBlockCount  [11..15] splitBlocks[0]
    string expectTilingData = "64 16384 0 0 229376 0 0 0 0 4575657221408423936 1 16384 0 64 16384 0 " SPLIT_BLOCKS_TAIL;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 1: float16 output, 2D shape [64,256], from=0.0, to=1.0
// outputSize=16384, usedCoreNum=64
// grid=64, totalThreads=16384, counterOffset=4
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_1)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {64, 256};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{64,256},{64,256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 16384 0 0 229376 0 0 0 0 4575657221408423936 1 16384 0 64 16384 0 " SPLIT_BLOCKS_TAIL;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 2: bfloat16 output, 2D shape [32,512], from=0.0, to=1.0
// outputSize=16384, usedCoreNum=64
// grid=64, totalThreads=16384, counterOffset=4
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_2)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {32, 512};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{32,512},{32,512}}, ge::DT_BF16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 16384 0 0 229376 0 0 0 0 4575657221408423936 1 16384 0 64 16384 0 " SPLIT_BLOCKS_TAIL;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ===== shape 维度覆盖 =====

// case 3: 1D shape [16384], float32, from=0.0, to=1.0
// outputSize=16384, usedCoreNum=64
// grid=64, totalThreads=16384, counterOffset=4
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_3)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {16384};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{16384},{16384}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 16384 0 0 229376 0 0 0 0 4575657221408423936 1 16384 0 64 16384 0 " SPLIT_BLOCKS_TAIL;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 4: 3D shape [4,64,64]=16384, float16, from=0.0, to=1.0
// outputSize=16384, usedCoreNum=64
// grid=64, totalThreads=16384, counterOffset=4
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_4)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {4, 64, 64};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{3},{3}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{4,64,64},{4,64,64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 16384 0 0 229376 0 0 0 0 4575657221408423936 1 16384 0 64 16384 0 " SPLIT_BLOCKS_TAIL;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 5: 4D shape [2,8,32,32]=16384, bfloat16, from=0.0, to=1.0
// outputSize=16384, usedCoreNum=64
// grid=64, totalThreads=16384, counterOffset=4
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_5)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {2, 8, 32, 32};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{4},{4}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{2,8,32,32},{2,8,32,32}}, ge::DT_BF16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 16384 0 0 229376 0 0 0 0 4575657221408423936 1 16384 0 64 16384 0 " SPLIT_BLOCKS_TAIL;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ===== 边界 shape 覆盖 =====

// case 6: 极小 shape，1D [16]，float32，from=0.0, to=1.0
// outputSize=16 < coreAlignSize=512 → usedCoreNum=1
// grid=min(ceil(16/256),624)=1, totalThreads=256, counterOffset=ceil(16/1024)*4=4
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_6)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {16};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{16},{16}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "1 16 0 0 229376 0 0 0 0 4575657221408423936 1 16 0 1 256 0 " SPLIT_BLOCKS_TAIL;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 7: 全核场景，2D [64,512]=32768，float32，from=0.0, to=1.0
// usedCoreNum=64
// grid=min(ceil(32768/256),624)=128, totalThreads=32768, counterOffset=ceil(32768/131072)*4=4
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_7)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {64, 512};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{64,512},{64,512}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 32768 0 0 229376 0 0 0 0 4575657221408423936 1 32768 0 128 32768 0 " SPLIT_BLOCKS_TAIL;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// case 8: 非对齐尾块，2D [100,200]=20000，float16，from=0.0, to=1.0
// usedCoreNum=40
// grid=min(ceil(20000/256),624)=79, totalThreads=20224, counterOffset=ceil(20000/80896)*4=4
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_8)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {100, 200};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{100,200},{100,200}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
    },
    &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "40 20000 0 0 229376 0 0 0 0 4575657221408423936 1 20000 0 79 20224 0 " SPLIT_BLOCKS_TAIL;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ===== 非法用例 =====

// case 9: 不支持的输出 dtype（DT_INT32）→ GRAPH_FAILED
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_invalid_dtype)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {32, 512};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{32,512},{32,512}}, ge::DT_INT32, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// case 10: 不支持的输出 dtype（DT_DOUBLE）→ GRAPH_FAILED
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_invalid_float64)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int64_t> shapeValue = {32, 512};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{2},{2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{32,512},{32,512}}, ge::DT_DOUBLE, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// case 11: 不支持的输入 shape dtype（DT_INT32）→ GRAPH_FAILED
// StatelessUniform 只接受 DT_INT64 shape（V3 支持 INT32+INT64）
TEST_F(StatelessUniformTilingTest, stateless_uniform_test_invalid_shape_dtype)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    vector<int32_t> shapeValue = {32, 512};
    int64_t seedVal = 0;
    int64_t offsetVal = 0;
    double fromVal = 0.0;
    double toVal = 1.0;

    gert::TilingContextPara tilingContextPara(
        "StatelessUniform",
    {
        {{{2},{2}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedVal},
        {{{1},{1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &fromVal},
        {{{1},{1}}, ge::DT_DOUBLE, ge::FORMAT_ND, true, &toVal},
    },
    {
        {{{32,512},{32,512}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
    },
    &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
