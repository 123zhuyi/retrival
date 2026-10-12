/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_common.h"
#include "util/util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

using namespace ge;

class squared_difference : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "squared_difference SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "squared_difference TearDown" << std::endl;
    }
};

TEST_F(squared_difference, squared_difference_infer_shape_fp16)
{
    ge::op::SquaredDifference op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
    auto tensor_desc = create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", tensor_desc);
    op.UpdateInputDesc("x2", tensor_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
        {2, 100},
    };
    EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(squared_difference, squared_difference_infer_shape_1)
{
    ge::op::SquaredDifference op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
    auto tensor_desc = create_desc_shape_range({60}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", tensor_desc);
    op.UpdateInputDesc("x2", tensor_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {60};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(squared_difference, squared_difference_infer_shape_2)
{
    ge::op::SquaredDifference op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
    auto tensor_desc = create_desc_shape_range({60}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", tensor_desc);
    std::vector<std::pair<int64_t, int64_t>> shape_range_x2 = {{2, 100}, {2, 100}, {2, 100}, {2, 100}};
    auto tensor_desc_x2 =
        create_desc_shape_range({60, 60, 60, 60}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::FORMAT_ND, shape_range_x2);
    op.UpdateInputDesc("x2", tensor_desc_x2);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {60, 60, 60, 60};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(squared_difference, squared_difference_infer_shape_3)
{
    ge::op::SquaredDifference op;
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{2, 100}};
    auto tensor_desc = create_desc_shape_range({-1}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x1", tensor_desc);
    std::vector<std::pair<int64_t, int64_t>> shape_range_x2 = {{2, 100}, {2, 100}, {2, 100}, {2, 100}};
    auto tensor_desc_x2 =
        create_desc_shape_range({-1, -1, -1, 60}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::FORMAT_ND, shape_range_x2);
    op.UpdateInputDesc("x2", tensor_desc_x2);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1, -1, -1, 60};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
        {2, 100},
        {2, 100},
        {2, 100},
        {60, 60},
    };
    EXPECT_EQ(output_shape_range, expected_shape_range);
}

static ge::Operator BuildGraph4InferAxisType(
    const std::initializer_list<int64_t>& dims1, const std::initializer_list<int64_t>& dims2, const ge::Format& format,
    const ge::DataType& dataType)
{
    auto apply_op = op::SquaredDifference("SquaredDifference");
    TENSOR_INPUT_WITH_SHAPE(apply_op, x1, dims1, dataType, format, {});
    TENSOR_INPUT_WITH_SHAPE(apply_op, x2, dims2, dataType, format, {});
    apply_op.InferShapeAndType();
    return apply_op;
}

TEST_F(squared_difference, infer_axis_type_with_input_dim_num_great_than_1)
{
    auto op1 = BuildGraph4InferAxisType({-1, -1}, {-1, -1}, ge::FORMAT_ND, ge::DT_FLOAT16);

    ge::InferAxisTypeInfoFunc infer_func = GetInferAxisTypeFunc("SquaredDifference");
    EXPECT_NE(infer_func, nullptr);

    std::vector<ge::AxisTypeInfo> infos;
    const uint32_t infer_axis_ret = static_cast<uint32_t>(infer_func(op1, infos));
    EXPECT_EQ(infer_axis_ret, ge::GRAPH_SUCCESS);

    std::vector<ge::AxisTypeInfo> expect_infos = {
        ge::AxisTypeInfoBuilder()
            .AxisType(ge::AxisType::ELEMENTWISE)
            .AddInputCutInfo({0, {0}})
            .AddInputCutInfo({1, {0}})
            .AddOutputCutInfo({0, {0}})
            .Build(),
        ge::AxisTypeInfoBuilder()
            .AxisType(ge::AxisType::ELEMENTWISE)
            .AddInputCutInfo({0, {1}})
            .AddInputCutInfo({1, {1}})
            .AddOutputCutInfo({0, {1}})
            .Build(),
    };

    EXPECT_STREQ(AxisTypeInfoToString(infos).c_str(), AxisTypeInfoToString(expect_infos).c_str());
}

TEST_F(squared_difference, infer_axis_type_case2)
{
    auto op1 = BuildGraph4InferAxisType({-1, -1, -1, -1}, {-1, -1}, ge::FORMAT_ND, ge::DT_FLOAT16);

    ge::InferAxisTypeInfoFunc infer_func = GetInferAxisTypeFunc("SquaredDifference");
    EXPECT_NE(infer_func, nullptr);

    std::vector<ge::AxisTypeInfo> infos;
    const uint32_t infer_axis_ret = static_cast<uint32_t>(infer_func(op1, infos));
    EXPECT_EQ(infer_axis_ret, ge::GRAPH_SUCCESS);

    std::vector<ge::AxisTypeInfo> expect_infos = {
        ge::AxisTypeInfoBuilder()
            .AxisType(ge::AxisType::ELEMENTWISE)
            .AddInputCutInfo({0, {0}})
            .AddOutputCutInfo({0, {0}})
            .Build(),
        ge::AxisTypeInfoBuilder()
            .AxisType(ge::AxisType::ELEMENTWISE)
            .AddInputCutInfo({0, {1}})
            .AddOutputCutInfo({0, {1}})
            .Build(),
        ge::AxisTypeInfoBuilder()
            .AxisType(ge::AxisType::ELEMENTWISE)
            .AddInputCutInfo({0, {2}})
            .AddInputCutInfo({1, {0}})
            .AddOutputCutInfo({0, {2}})
            .Build(),
        ge::AxisTypeInfoBuilder()
            .AxisType(ge::AxisType::ELEMENTWISE)
            .AddInputCutInfo({0, {3}})
            .AddInputCutInfo({1, {1}})
            .AddOutputCutInfo({0, {3}})
            .Build(),
    };

    EXPECT_STREQ(AxisTypeInfoToString(infos).c_str(), AxisTypeInfoToString(expect_infos).c_str());
}

TEST_F(squared_difference, infer_axis_type_case3)
{
    auto op1 = BuildGraph4InferAxisType({-1, -1, -1, -1}, {}, ge::FORMAT_ND, ge::DT_FLOAT16);

    ge::InferAxisTypeInfoFunc infer_func = GetInferAxisTypeFunc("SquaredDifference");
    EXPECT_NE(infer_func, nullptr);

    std::vector<ge::AxisTypeInfo> infos;
    const uint32_t infer_axis_ret = static_cast<uint32_t>(infer_func(op1, infos));
    EXPECT_EQ(infer_axis_ret, ge::GRAPH_SUCCESS);

    std::vector<ge::AxisTypeInfo> expect_infos = {
        ge::AxisTypeInfoBuilder()
            .AxisType(ge::AxisType::ELEMENTWISE)
            .AddInputCutInfo({0, {0}})
            .AddOutputCutInfo({0, {0}})
            .Build(),
        ge::AxisTypeInfoBuilder()
            .AxisType(ge::AxisType::ELEMENTWISE)
            .AddInputCutInfo({0, {1}})
            .AddOutputCutInfo({0, {1}})
            .Build(),
        ge::AxisTypeInfoBuilder()
            .AxisType(ge::AxisType::ELEMENTWISE)
            .AddInputCutInfo({0, {2}})
            .AddOutputCutInfo({0, {2}})
            .Build(),
        ge::AxisTypeInfoBuilder()
            .AxisType(ge::AxisType::ELEMENTWISE)
            .AddInputCutInfo({0, {3}})
            .AddOutputCutInfo({0, {3}})
            .Build(),
    };

    EXPECT_STREQ(AxisTypeInfoToString(infos).c_str(), AxisTypeInfoToString(expect_infos).c_str());
}