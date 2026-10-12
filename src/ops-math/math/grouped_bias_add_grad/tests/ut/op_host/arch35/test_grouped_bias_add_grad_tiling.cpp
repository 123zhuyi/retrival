/**

Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/grouped_bias_add_grad_tiling_arch35.h"

using namespace std;

class TilingGroupedBiasAddGrad : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TilingGroupedBiasAddGrad SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TilingGroupedBiasAddGrad TearDown" << std::endl;
    }
};

/*

============================================
测试用例设计思路
============================================
模式选择逻辑:
CUT_G_MODE (tilingKey=67108865): 当 hBlockCount <= 32 时触发
CUT_H_MODE (tilingKey=33554433): 当 hBlockCount > 32 时触发
ARA模式 (tilingKey根据reduce模版): 当 group_idx 为空（3D输入）时触发
CUT_G_MODE tiling data 格式（12个字段）:
cutGDim cutHDim blockFactor blockTailFactor blockTailStartIndex inputShape[0] inputShape[1]
ubHTailFactor useUbSize groupedIdxSize outputSize groupIdxType
CUT_H_MODE tiling data 格式（11个字段）:
blockFactor blockTailFactor groupIdxDim inputShape[0] inputShape[1]
hFactorCount hTailFactor useUbSize groupedIdxSize outputSize groupIdxType
平台参数:
coreNum = 64
ubSize = 253952
cacheLineSize = 128 (256/2)
hPerBlock(FP32) = 128/4 = 32
hPerBlock(FP16/BF16) = 128/2 = 64
*/
// ============================================
// CUT_G_MODE 测试用例 (hBlockCount <= 32)
// ============================================

// 1. 基本功能：小shape，FP32，groupIdxType=0，无workspace
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_FP32_groupIdxType0)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 32}, {3, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865; // CUT_G_MODE = 2
    string expectTilingData = "3 1 1 1 3 10 32 32 56800 32 512 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 2. groupIdxType=1，需要分配workspace
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_FP32_groupIdxType1)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 32}, {3, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(1))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865;
    string expectTilingData = "3 1 1 1 3 10 32 32 56800 32 512 2048 20480 1 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 3. FP16类型，H=64 (hPerBlock=64, 1块)
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_FP16_groupIdxType0)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 64}, {10, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 64}, {3, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865;
    // FP16: hPerBlock=64, cutHDim=1, outputSize=256, useUbSize=126752
    string expectTilingData = "3 1 1 1 3 10 64 64 55776 32 1024 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 4. BF16类型
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_BF16_groupIdxType0)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 64}, {10, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 64}, {3, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865;
    // BF16: 与FP16相同计算方式
    string expectTilingData = "3 1 1 1 3 10 64 64 55776 32 1024 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 5. 大G维度 (G=10, totalBlocks=10, usedCoreNum=10)
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_LargeG)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{100, 32}, {100, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865;
    // G=10, cutHDim=1, groupedIdxSize=64
    string expectTilingData = "10 1 1 1 10 100 32 32 56800 64 512 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 6. 分核余数场景 (G=5, H=1块, totalBlocks=5, 每核1块)
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_CoreSplitRemainder)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{20, 32}, {20, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{5, 32}, {5, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865;
    // G=5, cutHDim=1, groupedIdxSize=32
    string expectTilingData = "5 1 1 1 5 20 32 32 56800 32 512 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 7. group_idx为INT64类型，groupIdxType=1触发workspace分配
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_INT64GroupIdx)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{3, 32}, {3, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(1))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 603979777;
    // INT64 groupIdx: groupedIdxSize=AlignUp(3*8,32)=32
    string expectTilingData = "3 1 1 1 3 10 32 32 56800 32 512 2048 20480 1 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================
// CUT_H_MODE 测试用例 (hBlockCount > 32)
// ============================================

// 8. 大H轴，FP32，触发CUT_H_MODE (H=2048, hPerBlock=32, hBlockCount=64 > 32)
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutH_FP32_LargeH)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{100, 2048}, {100, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4, 2048}, {4, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865; // CUT_H_MODE = 1
    // CUT_H: hFactorCount=64, blockFactor=1
    string expectTilingData = "4 16 1 1 64 100 2048 128 56800 32 512 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 9. CUT_H_MODE + FP16 (H=4096, hPerBlock=64, hBlockCount=64 > 32)
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutH_FP16_LargeH)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{100, 4096}, {100, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4, 4096}, {4, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865;
    // FP16 CUT_H: hPerBlock=64, hFactorCount=64, outputSize=256
    string expectTilingData = "4 16 1 1 64 100 4096 256 55776 32 1024 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================
// 边界/特殊场景测试用例
// ============================================

// 10. H轴刚好32个元素（FP32, hPerBlock=32, 单块）, G=2
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_MinH)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 32}, {2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865;
    string expectTilingData = "2 1 1 1 2 10 32 32 56800 32 512 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 11. G=1 单组场景, H=64 (cutHDim=2)
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_SingleGroup)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 64}, {10, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64}, {1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865;
    // G=1, cutHDim=2, totalBlocks=2
    string expectTilingData = "1 1 1 1 1 10 64 64 56800 32 512 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 12. 多H块场景 (H=128, hPerBlock=32, cutHDim=4)，测试maxHBlocks计算
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_MultiHBlocks)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{20, 128}, {20, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4, 128}, {4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865; // GcutHDim = 44 = 16 <= 32, 仍为CUT_G_MODE
    // G=4, cutHDim=4, totalBlocks=16
    string expectTilingData = "4 1 1 1 4 20 128 128 56800 32 512 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 13. 临界场景：hBlockCount刚好=32 (H=1024, hPerBlock=32, hBlockCount=32)
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_HBlockCountEqual32)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{50, 1024}, {50, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 1024}, {2, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865; // hBlockCount=32 <= 32, 仍为CUT_G_MODE
    // G=2, cutHDim=32, totalBlocks=64
    string expectTilingData = "2 8 1 1 16 50 1024 128 56800 32 512 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 14. 临界场景：hBlockCount刚好=33，触发CUT_H_MODE
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutH_HBlockCountEqual33)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{50, 1056}, {50, 1056}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 1056}, {2, 1056}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865; // hBlockCount=33 > 32, CUT_H_MODE
    // CUT_H: hFactorCount=33
    string expectTilingData = "2 9 1 1 18 50 1056 32 56800 32 512 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================
// group_idx 维度边界测试用例
// ============================================

// 15. G=2048 边界值（最大合法值）
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_MaxGroupIdx2048)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{4096, 32}, {4096, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2048}, {2048}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2048, 32}, {2048, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    uint64_t expectTilingKey = 67108865; // CUT_G_MODE
    // G=2048, cutHDim=1, totalBlocks=2048, usedCoreNum=64
    // blockFactor=32, blockTailFactor=32, tailStartIndex=64
    // groupedIdxSize = AlignUp(2048*4, 32) = 8192
    // outputSize = 128
    // useUbSize = AlignDown((253952-128-8192-128)/2, 32) = AlignDown(245504/2, 32) = 122752
    string expectTilingData = "2048 1 32 32 64 4096 32 32 54784 8192 512 2048 20480 0 ";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 16. G=2049 超出边界（应该失败）
TEST_F(TilingGroupedBiasAddGrad, ascend950_CutG_ExceedMaxGroupIdx2049)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{4098, 32}, {4098, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2049}, {2049}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2049, 32}, {2049, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    // 预期返回失败，group_idx维度超过2048限制
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {4096};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ============================================
// 错误路径测试用例（覆盖OP_LOGE_FOR_*日志修改）
// ============================================

// 17. grad_y dtype不支持（INT32），期望报错 (EZ0020)
TEST_F(TilingGroupedBiasAddGrad, ascend950_Failed_InvalidGradYDtype)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 32}, {10, 32}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 32}, {3, 32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 18. grad_y不是2D tensor（3D输入带group_idx），期望报错 (EZ0012)
TEST_F(TilingGroupedBiasAddGrad, ascend950_Failed_GradYNot2D)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{2, 5, 32}, {2, 5, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 32}, {3, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 19. group_idx dtype不支持（FP32），期望报错 (EZ0020)
TEST_F(TilingGroupedBiasAddGrad, ascend950_Failed_InvalidGroupIdxDtype)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 32}, {3, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 20. group_idx_type属性非法（值为2），期望报错 (EZ0024)
TEST_F(TilingGroupedBiasAddGrad, ascend950_Failed_InvalidGroupIdxType)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 32}, {3, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(2))}, &compileInfo,
        64, 253952);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 21. 输出grad_bias shape不匹配（dim0不等于dimG），期望报错 (EZ0008)
TEST_F(TilingGroupedBiasAddGrad, ascend950_Failed_OutputShapeMismatch)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4, 32}, {4, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ============================================
// ARA模式 错误路径测试用例（覆盖OP_LOGE_FOR_*整改）
// ============================================

// 22. ARA模式：grad_y不是3D tensor（2D输入无group_idx），期望报错 OP_LOGE_FOR_INVALID_SHAPEDIM
TEST_F(TilingGroupedBiasAddGrad, ascend950_ARA_Failed_GradYNot3D)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 32}, {1, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 23. ARA模式：grad_y为4D tensor（无group_idx），期望报错 OP_LOGE_FOR_INVALID_SHAPEDIM
TEST_F(TilingGroupedBiasAddGrad, ascend950_ARA_Failed_GradY4D)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{2, 5, 10, 32}, {2, 5, 10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 32}, {1, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 24. RA模式：输出grad_bias维度不为2（1D），期望报错 OP_LOGE_FOR_INVALID_SHAPEDIM
TEST_F(TilingGroupedBiasAddGrad, ascend950_Failed_OutputShapeDimNot2)
{
    optiling::GroupedBiasAddGradCompileInfoArch35 compileInfo = {253952, 64, 32, 128, 256};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{96}, {96}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        64, 253952);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
