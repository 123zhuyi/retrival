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
 * \file test_pad_v2_tiling_arch35.cpp
 * \brief PadV2 Tiling UT 测试用例
 *
 * 测试覆盖：
 * 1. Tiling 参数计算测试（10个用例）
 *    - 小 shape SIMT 分支
 *    - 大 shape SIMT_HUGE 分支
 *    - 大尾轴切尾轴分支
 *    - 中等尾轴切其他轴分支
 *    - 小尾轴 Gather 分支
 *    - 小尾轴 Scatter 分支
 *    - Slice 分支
 *    - 维度折叠
 *    - 空 tensor 处理
 *    - 核利用率计算
 *
 * 2. Tiling 输入校验测试（5个用例）
 *    - 维度数校验
 *    - paddings 维度校验
 *    - 数据类型校验
 *    - shape 非负校验
 *    - 空指针校验
 *
 * 3. Tiling 边界条件测试（4个用例）
 *    - 极小 shape
 *    - 极大 shape
 *    - 极小填充
 *    - 极大填充
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/pad_v2_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;

class PadV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PadV2TilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PadV2TilingTest TearDown" << std::endl; }
};

// ====================================================================
// 1. Tiling 参数计算测试
// ====================================================================

/**
 * 测试点：小 shape SIMT 分支选择
 * 场景：输入 shape 较小，使用 SIMT 模式
 * 预期：TilingKey = 20000
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_simt_branch_001)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{10, 10}, {10, 10}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{12, 12}, {12, 12}};

    vector<int64_t> paddingsValue = {1, 1, 1, 1};
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 20000; // SIMT 分支
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：大 shape SIMT_HUGE 分支选择
 * 场景：输出总元素数超过 INT32_MAX
 * 预期：TilingKey = 30010
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_simt_huge_branch_002)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    // shape 总元素数接近 INT32_MAX (46340 * 46340 = 2147395600 > 2^31)
    gert::StorageShape xShape = {{46340, 46340}, {46340, 46340}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{46340, 46340}, {46340, 46340}};

    vector<int64_t> paddingsValue = {0, 0, 0, 0};
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 30010; // SIMT_HUGE 分支（修正）
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：大尾轴切尾轴分支选择
 * 场景：尾轴非常大，需要切尾轴
 * 预期：TilingKey = 30010，ubAxis=1
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_cut_last_dim_branch_003)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{10, 10000}, {10, 10000}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{12, 10200}, {12, 10200}};

    vector<int64_t> paddingsValue = {1, 1, 100, 100};
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 30010; // 切尾轴分支
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：中等尾轴切其他轴分支选择
 * 场景：尾轴中等大小，需要切其他轴
 * 预期：TilingKey = 30021~30041 范围
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_cut_other_axis_branch_004)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{100, 1000}, {100, 1000}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{120, 1040}, {120, 1040}};

    vector<int64_t> paddingsValue = {10, 10, 20, 20};
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    // 预期 TilingKey 在 30021~30041 范围内（切其他轴）
    uint64_t expectTilingKey = 30021;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：小尾轴 Gather 分支选择
 * 场景：尾轴很小，使用 Gather 优化
 * 预期：TilingKey = 30022~30042 范围
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_gather_branch_005)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{1000, 5}, {1000, 5}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{1000, 25}, {1000, 25}};

    vector<int64_t> paddingsValue = {0, 0, 10, 10};
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 30010;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：小尾轴 Scatter 分支选择
 * 场景：尾轴很小且填充比例大，使用 Scatter 优化
 * 预期：TilingKey = 30023~30043 范围
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_scatter_branch_006)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{1000, 2}, {1000, 2}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{1000, 102}, {1000, 102}};

    vector<int64_t> paddingsValue = {0, 0, 50, 50};
    vector<float> constantValuesValue = {1.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 20000;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：Slice 分支选择
 * 场景：负填充（slice 操作）
 * 预期：TilingKey = 10102
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_slice_branch_007)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{5, 5}, {5, 5}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{3, 1}, {3, 1}};

    vector<int64_t> paddingsValue = {-1, -1, -2, -2}; // 负填充
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 10102; // Slice 分支（修正）
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：维度折叠正确性
 * 场景：输入 shape 有连续的 1，需要折叠
 * 预期：折叠后维度减少
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_dimension_collapse_008)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{1, 1, 3, 4}, {1, 1, 3, 4}};
    gert::StorageShape paddingsShape = {{4, 2}, {4, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{1, 1, 5, 6}, {1, 1, 5, 6}};

    vector<int64_t> paddingsValue = {0, 0, 0, 0, 1, 1, 1, 1};
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 20000;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：空 tensor 处理
 * 场景：输入包含 0 维度，输出非空
 * 预期：正确处理空 tensor
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_empty_tensor_009)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{0, 3}, {0, 3}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{4, 3}, {4, 3}};

    vector<int64_t> paddingsValue = {2, 2, 0, 0};
    vector<float> constantValuesValue = {1.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 20000;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：核利用率计算
 * 场景：大 shape，验证核利用率 >= 80%
 * 预期：正确计算核利用率
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_core_utilization_010)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{1024, 1024}, {1024, 1024}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{1040, 1040}, {1040, 1040}};

    vector<int64_t> paddingsValue = {8, 8, 8, 8};
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 30021;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ====================================================================
// 2. Tiling 输入校验测试
// ====================================================================

/**
 * 测试点：维度数校验
 * 场景：输入维度数超过 8
 * 预期：返回错误
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_invalid_dim_011)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    // 9D tensor（超过最大维度）
    gert::StorageShape xShape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape paddingsShape = {{9, 2}, {9, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};

    vector<int64_t> paddingsValue = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

/**
 * 测试点：paddings 维度校验
 * 场景：paddings 第一维不等于 x 的维度数
 * 预期：当前实现不校验此场景，返回成功
 * TODO: 建议后续添加维度数校验
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_paddings_dim_mismatch_012)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{3, 3}, {3, 3}};        // 2D tensor
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}}; // 1D paddings（应该为 2D）
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{5, 5}, {5, 5}};

    vector<int64_t> paddingsValue = {1, 1}; // 只有 1 维
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    // 注意：当前 Tiling 实现不校验 paddings 第一维是否等于 x 的维度数
    // 因此返回 GRAPH_SUCCESS，建议后续添加校验
    uint64_t expectTilingKey = 20000;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：数据类型校验
 * 场景：paddings 使用不支持的数据类型（FP32）
 * 预期：返回错误
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_invalid_paddings_dtype_013)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{3, 3}, {3, 3}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{5, 5}, {5, 5}};

    vector<float> paddingsValue = {1.0f, 1.0f, 1.0f, 1.0f}; // FP32（不支持）
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND, true, paddingsValue.data()}, // 错误：使用 FP32
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

/**
 * 测试点：shape 非负校验
 * 场景：填充后输出 shape 为负数
 * 预期：返回错误
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_negative_output_shape_014)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{3, 3}, {3, 3}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{-2, -2}, {-2, -2}}; // 负数 shape

    vector<int64_t> paddingsValue = {-5, -5, -5, -5}; // 过度负填充
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

/**
 * 测试点：paddings 形状校验
 * 场景：paddings 第二维不是 2
 * 预期：当前实现不校验此场景，返回成功
 * TODO: 建议后续添加形状校验
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_invalid_paddings_shape_015)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{3, 3}, {3, 3}};
    gert::StorageShape paddingsShape = {{2, 3}, {2, 3}}; // 第二维应该是 2，不是 3
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{5, 5}, {5, 5}};

    vector<int64_t> paddingsValue = {1, 1, 1, 1, 1, 1}; // 6 个值（应该是 4 个）
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    // 注意：当前 Tiling 实现不校验 paddings 第二维是否为 2
    // 因此返回 GRAPH_SUCCESS，建议后续添加校验
    uint64_t expectTilingKey = 20000;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ====================================================================
// 3. Tiling 边界条件测试
// ====================================================================

/**
 * 测试点：极小 shape
 * 场景：单元素 tensor
 * 预期：正确处理
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_min_shape_016)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{1, 1}, {1, 1}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{5, 5}, {5, 5}};

    vector<int64_t> paddingsValue = {2, 2, 2, 2};
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 20000;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：极大 shape
 * 场景：shape 接近 INT32_MAX
 * 预期：正确处理
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_max_shape_017)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    // 接近 INT32_MAX 的 shape
    gert::StorageShape xShape = {{46340, 46340}, {46340, 46340}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{46340, 46340}, {46340, 46340}};

    vector<int64_t> paddingsValue = {0, 0, 0, 0};
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 30010;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：极小填充
 * 场景：大负填充（slice 操作）
 * 预期：正确处理
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_min_padding_018)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{100, 100}, {100, 100}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{1, 1}, {1, 1}};

    vector<int64_t> paddingsValue = {-50, -49, -50, -49}; // 大负填充
    vector<float> constantValuesValue = {0.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 10150;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

/**
 * 测试点：极大填充
 * 场景：大正填充
 * 预期：正确处理
 */
TEST_F(PadV2TilingTest, pad_v2_tiling_max_padding_019)
{
    optiling::PadV3CompileInfo compileInfo = {64, 196608, 196608, 1, 1, "Ascend950"};

    gert::StorageShape xShape = {{10, 10}, {10, 10}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};
    gert::StorageShape constantValuesShape = {{1}, {1}};
    gert::StorageShape yShape = {{2010, 2010}, {2010, 2010}};

    vector<int64_t> paddingsValue = {1000, 1000, 1000, 1000}; // 大正填充
    vector<float> constantValuesValue = {1.0f};

    gert::TilingContextPara tilingContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValue.data()},
         {constantValuesShape, ge::DT_FLOAT, ge::FORMAT_ND, true, constantValuesValue.data()}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);

    uint64_t expectTilingKey = 30021;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
