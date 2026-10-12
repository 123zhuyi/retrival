/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "op_infer_datatype_context_builder.h"
#include "op_infer_shape_range_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class ReduceLogSumExpTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceLogSumExpTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceLogSumExpTest TearDown" << std::endl; }
};

// ==================== 非空axes, DT_INT32, keep_dims=true ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_const_infer_1)
{
    std::vector<int32_t> axesValue = {1, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 1, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 非空axes, DT_INT32, keep_dims=false, 包含负轴 ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_const_infer_2)
{
    std::vector<int32_t> axesValue = {1, -2};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 标量输入, DT_INT32, keep_dims=false ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_const_infer_scalar)
{
    std::vector<int32_t> axesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 标量输入, keep_dims=true ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_scalar_keepdims_true)
{
    std::vector<int32_t> axesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 空axes (dimNum==1, dim0==0), noop_with_empty_axes=true(默认) ====================
// 输出shape与输入相同
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_empty_axes_noop_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5, 16, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 空axes (axesDimNum=0), noop_with_empty_axes=false, keep_dims=true ====================
// 所有轴归约为1
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_empty_axes_noop_false_keepdims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 空axes (axesDimNum=0), noop_with_empty_axes=false, keep_dims=false ====================
// 输出为标量
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_empty_axes_noop_false_keepdims_false)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 空axes (axesDimNum=0), 标量输入, noop_with_empty_axes=true ====================
// 标量输入保持不变
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_empty_axes_noop_true_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 空axes (axesDimNum=0), 标量输入, noop_with_empty_axes=false, keep_dims=false
// ==================== 标量继续保持标量
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_empty_axes_noop_false_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== axes维度为1且大小为0, noop_with_empty_axes=true ====================
// shape[0] axes, 等同于空axes, noop=true, 输出shape与输入相同
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_zero_dim_axes_noop_true)
{
    std::vector<int32_t> axesValue = {};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5, 16, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== axes维度为1且大小为0, noop_with_empty_axes=false, keep_dims=false ====================
// shape[0] axes, 等同于空axes, 输出为标量
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_zero_dim_axes_noop_false_keepdims_false)
{
    std::vector<int32_t> axesValue = {};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 非空axes, DT_INT64, keep_dims=true ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_int64_keepdims_true)
{
    std::vector<int64_t> axesValue = {1, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 1, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 非空axes, DT_INT64, keep_dims=false, 包含负轴 ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_int64_keepdims_false)
{
    std::vector<int64_t> axesValue = {1, -2};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 非空axes, DT_INT64, keep_dims=true, 全负轴 ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_int64_neg_axes_keepdims_true)
{
    std::vector<int64_t> axesValue = {-1, -2};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5, 1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 非空axes, DT_INT64, keep_dims=false, 全负轴 ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_int64_neg_axes_keepdims_false)
{
    std::vector<int64_t> axesValue = {-1, -3};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 非int32/int64 axes dtype → 视为非const，保守推导 ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_invalid_axes_dtype)
{
    std::vector<float> axesValue = {1.0f, 2.0f};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, -1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 无效axes范围 (axis超出维度范围) → GRAPH_FAILED ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_invalid_axes_bounds)
{
    std::vector<int32_t> axesValue = {4}; // shape dimNum=4, 有效范围[-4,3], 4超出范围
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// ==================== 无效axes范围 (负轴超出范围) → GRAPH_FAILED ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_invalid_axes_bounds_neg)
{
    std::vector<int32_t> axesValue = {-5}; // shape dimNum=4, 有效范围[-4,3], -5超出范围
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// ==================== 无效axes范围 (标量输入, axis!=0) → GRAPH_FAILED ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_invalid_axes_bounds_scalar)
{
    std::vector<int32_t> axesValue = {1}; // 标量输入只允许axis=0
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// ==================== 归约所有轴, keep_dims=true ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_all_dims_keepdims_true)
{
    std::vector<int32_t> axesValue = {0, 1, 2, 3};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 归约所有轴, keep_dims=false → 标量输出 ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_all_dims_keepdims_false)
{
    std::vector<int32_t> axesValue = {0, 1, 2, 3};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 1D输入, 归约唯一轴, keep_dims=true ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_1d_keepdims_true)
{
    std::vector<int32_t> axesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 1D输入, 归约唯一轴, keep_dims=false → 标量 ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_1d_keepdims_false)
{
    std::vector<int32_t> axesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 单轴归约, keep_dims=true ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_single_axis_keepdims_true)
{
    std::vector<int32_t> axesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 16, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 单轴归约, keep_dims=false ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_single_axis_keepdims_false)
{
    std::vector<int32_t> axesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 16, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 动态shape(-1), keep_dims=true ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_dynamic_keepdims_true)
{
    std::vector<int32_t> axesValue = {2};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, -1, 16, 16}, {3, -1, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, -1, 1, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 动态shape(-1), keep_dims=false ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_dynamic_keepdims_false)
{
    std::vector<int32_t> axesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, -1, 16, 16}, {3, -1, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 16, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== DT_INT32, 负轴, keep_dims=true ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_neg_axes_keepdims_true)
{
    std::vector<int32_t> axesValue = {-1, -2};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5, 1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== DT_INT64, 多轴归约, keep_dims=true, 5D输入 ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_int64_5d_keepdims_true)
{
    std::vector<int64_t> axesValue = {2, 4};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 1, 4, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== DT_INT64, 多轴归约, keep_dims=false, 5D输入 ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_int64_5d_keepdims_false)
{
    std::vector<int64_t> axesValue = {2, 4};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== DT_INT64, 标量输入, keep_dims=true ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_int64_scalar_keepdims_true)
{
    std::vector<int64_t> axesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== axes维度大于1 (2D tensor) → 铺平为1D, 正常常量推导 ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_multi_dim_axes_flatten)
{
    std::vector<int32_t> axesValue = {1, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 1, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 未知秩输入 ({-2}) → 输出 {-2} ====================
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_unknown_rank)
{
    std::vector<int32_t> axesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::InfershapeContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== InferDataType: 输出dtype与输入dtype一致 ====================
// 覆盖 InferDataType4ReduceLogSumExp + InferDataType4ReduceCommon
// TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_infer_datatype)
// {
//     auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
//     ASSERT_NE(spaceRegistry, nullptr);
//     auto opImpl = spaceRegistry->GetOpImpl("ReduceLogSumExp");
//     ASSERT_NE(opImpl, nullptr);
//     ASSERT_NE(opImpl->infer_datatype, NULL);

//     gert::OpInferDataTypeContextBuilder builder;
//     builder.OpType("ReduceLogSumExp").OpName("ReduceLogSumExp");
//     builder.IONum(2, 1);
//     builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
//     builder.InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
//     builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
//     auto contextHolder = builder.Build();
//     auto* context = contextHolder.GetContext();
//     ASSERT_NE(context, nullptr);

//     auto ret = opImpl->infer_datatype(context);
//     EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//     EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
// }

// ==================== InferShapeRange: 返回GRAPH_SUCCESS ====================
// 覆盖 InferShapeRange4ReduceLogSumExp + InferShapeRange4ReduceCommon
TEST_F(ReduceLogSumExpTest, ReduceLogSumExp_infer_shape_range)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("ReduceLogSumExp");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("ReduceLogSumExp").OpName("ReduceLogSumExp");
    builder.IONum(2, 1);
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_shape_range(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
