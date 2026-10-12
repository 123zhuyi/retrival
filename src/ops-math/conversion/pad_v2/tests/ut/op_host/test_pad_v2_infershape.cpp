/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class PadV2InferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PadV2InferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PadV2InferShapeTest TearDown" << std::endl; }
};

// ========== 功能测试 (12个用例) ==========

// OP-PadV2-L2-infershape-001: 验证 2D 基础 shape 计算
TEST_F(PadV2InferShapeTest, TestBasic2DShapeCalculation)
{
    gert::StorageShape xShape = {{4, 5}, {4, 5}};
    std::vector<int32_t> paddingsValues = {1, 2, 3, 4};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{7, 12}, {7, 12}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{7, 12}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-002: 验证 3D 多维度 shape 计算
TEST_F(PadV2InferShapeTest, Test3DShapeCalculation)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    std::vector<int32_t> paddingsValues = {1, 1, 0, 2, 2, 0};
    gert::StorageShape paddingsShape = {{3, 2}, {3, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{4, 5, 6}, {4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 5, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-003: 验证 1D 单维度 shape 计算
TEST_F(PadV2InferShapeTest, Test1DShapeCalculation)
{
    gert::StorageShape xShape = {{10}, {10}};
    std::vector<int32_t> paddingsValues = {5, 5};
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{20}, {20}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{20}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-004: 验证 4D 高维 shape 计算
TEST_F(PadV2InferShapeTest, Test4DShapeCalculation)
{
    gert::StorageShape xShape = {{1, 2, 3, 4}, {1, 2, 3, 4}};
    std::vector<int32_t> paddingsValues = {0, 0, 1, 1, 2, 2, 3, 3};
    gert::StorageShape paddingsShape = {{4, 2}, {4, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 4, 7, 10}, {1, 4, 7, 10}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 4, 7, 10}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-005: 验证 8D 最大维度 shape 计算
TEST_F(PadV2InferShapeTest, Test8DShapeCalculation)
{
    gert::StorageShape xShape = {{1, 1, 1, 1, 1, 2, 2, 2}, {1, 1, 1, 1, 1, 2, 2, 2}};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    gert::StorageShape paddingsShape = {{8, 2}, {8, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 1, 1, 1, 2, 2, 2}, {1, 1, 1, 1, 1, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 1, 1, 1, 2, 2, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-006: 验证 paddings 为 INT64
TEST_F(PadV2InferShapeTest, TestPaddingsINT64)
{
    gert::StorageShape xShape = {{3, 3}, {3, 3}};
    std::vector<int64_t> paddingsValues = {1, 1, 1, 1};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{5, 5}, {5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-007: 验证负填充（slice）
TEST_F(PadV2InferShapeTest, TestNegativePadding)
{
    gert::StorageShape xShape = {{5, 5}, {5, 5}};
    std::vector<int32_t> paddingsValues = {-1, -1, -2, -2};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{3, 1}, {3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-008: 验证零填充（恒等映射）
TEST_F(PadV2InferShapeTest, TestZeroPadding)
{
    gert::StorageShape xShape = {{3, 4}, {3, 4}};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-009: 验证混合填充
TEST_F(PadV2InferShapeTest, TestMixedPadding)
{
    gert::StorageShape xShape = {{5, 5}, {5, 5}};
    std::vector<int32_t> paddingsValues = {-1, 2, 3, -2};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{6, 6}, {6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-010: 验证动态 shape（-1）
TEST_F(PadV2InferShapeTest, TestDynamicShape)
{
    gert::StorageShape xShape = {{-1, 5}, {-1, 5}};
    std::vector<int32_t> paddingsValues = {1, 1, 2, 2};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{-1, 9}, {-1, 9}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 9}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-011: 验证全动态 shape
TEST_F(PadV2InferShapeTest, TestAllDynamicShape)
{
    gert::StorageShape xShape = {{-1, -1}, {-1, -1}};
    std::vector<int32_t> paddingsValues = {1, 1, 2, 2};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-012: 验证未知 rank（-2）
TEST_F(PadV2InferShapeTest, TestUnknownRank)
{
    gert::StorageShape xShape = {{-2}, {-2}};
    std::vector<int32_t> paddingsValues = {1, 1};
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ========== 边界条件测试 (6个用例) ==========

// OP-PadV2-L2-infershape-013: 空 tensor 输入
TEST_F(PadV2InferShapeTest, TestEmptyTensorInput)
{
    gert::StorageShape xShape = {{0, 3}, {0, 3}};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{0, 3}, {0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-014: 空 tensor 输入（输出非空）
TEST_F(PadV2InferShapeTest, TestEmptyTensorInputOutputNonEmpty)
{
    gert::StorageShape xShape = {{0, 3}, {0, 3}};
    std::vector<int32_t> paddingsValues = {2, 2, 0, 0};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-015: 输出为空 tensor
TEST_F(PadV2InferShapeTest, TestOutputEmptyTensor)
{
    gert::StorageShape xShape = {{5, 5}, {5, 5}};
    std::vector<int32_t> paddingsValues = {-3, -2, -3, -2};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{0, 0}, {0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-016: 单元素 tensor
TEST_F(PadV2InferShapeTest, TestSingleElementTensor)
{
    gert::StorageShape xShape = {{1, 1}, {1, 1}};
    std::vector<int32_t> paddingsValues = {2, 2, 2, 2};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{5, 5}, {5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// OP-PadV2-L2-infershape-017: 0D tensor（标量）- 不支持
TEST_F(PadV2InferShapeTest, Test0DTensor)
{
    gert::StorageShape xShape = {{}, {}};
    std::vector<int32_t> paddingsValues = {};
    gert::StorageShape paddingsShape = {{0, 2}, {0, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    // 0D tensor 不支持，预期返回 GRAPH_FAILED
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// OP-PadV2-L2-infershape-018: 非常量 paddings
TEST_F(PadV2InferShapeTest, TestNonConstPaddings)
{
    gert::StorageShape xShape = {{3, 3}, {3, 3}};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara("PadV2",
                                                      {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {paddingsShape, ge::DT_INT32, ge::FORMAT_ND}, // 非常量 tensor
                                                       {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ========== 异常场景测试 (6个用例) ==========

// OP-PadV2-L2-infershape-019: paddings 维度不匹配
TEST_F(PadV2InferShapeTest, TestPaddingsDimensionMismatch)
{
    gert::StorageShape xShape = {{3, 3}, {3, 3}};
    std::vector<int32_t> paddingsValues = {1, 1};
    gert::StorageShape paddingsShape = {{1, 2}, {1, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// OP-PadV2-L2-infershape-020: paddings 数量错误
TEST_F(PadV2InferShapeTest, TestPaddingsCountError)
{
    gert::StorageShape xShape = {{3, 3}, {3, 3}};
    std::vector<int32_t> paddingsValues = {1, 1, 1};
    gert::StorageShape paddingsShape = {{3}, {3}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// OP-PadV2-L2-infershape-021: 输出 shape 为负数
TEST_F(PadV2InferShapeTest, TestOutputShapeNegative)
{
    gert::StorageShape xShape = {{3, 3}, {3, 3}};
    std::vector<int32_t> paddingsValues = {-5, -5, -5, -5};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// OP-PadV2-L2-infershape-022: paddings 类型不支持
TEST_F(PadV2InferShapeTest, TestUnsupportedPaddingsType)
{
    gert::StorageShape xShape = {{3, 3}, {3, 3}};
    std::vector<float> paddingsValues = {1.0f, 1.0f, 1.0f, 1.0f};
    gert::StorageShape paddingsShape = {{2, 2}, {2, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// OP-PadV2-L2-infershape-023: 空 paddings
TEST_F(PadV2InferShapeTest, TestEmptyPaddings)
{
    gert::StorageShape xShape = {{3, 3}, {3, 3}};
    std::vector<int32_t> paddingsValues = {};
    gert::StorageShape paddingsShape = {{0, 2}, {0, 2}};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV2",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND},
         {paddingsShape, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
         {paddingsShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
