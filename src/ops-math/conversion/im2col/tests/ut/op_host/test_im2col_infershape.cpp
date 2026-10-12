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
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class Im2colInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Im2colInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Im2colInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(Im2colInfershapeTest, im2col_infer_shape_unknow_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_input_shape_lt_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{1, 2, 3}, {1, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_input_shape_gt_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_input_format_invalid)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_no_ksizes)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_ksizes_size_lt_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_ksizes_size_gt_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_ksizes_value_le_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 0})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_strides_size_lt_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_strides_size_gt_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_strides_value_le_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 0})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_dilations_size_lt_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_dilations_size_gt_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_dilations_value_le_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 0})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_padding_mode_invalid)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("xxx")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_pads_size_lt_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_pads_size_eq_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_pads_size_gt_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_pads_value_lt_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-1})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_effect_shape_gt_padding_shape_with_CALCULATED_mode)
{
    // padded input = (6, 15)
    // effect = (5, 16)
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{1, 2, 6, 12}, {1, 2, 6, 12}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 6})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 1, 2})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_ParamCheck_effect_shape_gt_padding_shape_with_VALID_mode)
{
    // padded input = (3, 8)
    // effect = (3, 9)
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{1, 3, 8, 2}, {1, 3, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 5})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("VALID")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({10})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_getInput_succ_with_attrSize1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{10, 20, 30, 40}, {10, 20, 30, 40}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({4})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({5})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10, 80, 12, 16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_getInput_succ_with_attrSize2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{10, 20, 30, 40}, {10, 20, 30, 40}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({4, 5})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({6, 7})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({8, 9})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10, 120, 10, 9},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_getInput_succ_with_attrSize4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3, 4})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 4, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_succ_calcpad)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{1, 2, 6, 19}, {1, 2, 6, 12}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 6})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 1, 2})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 36, 2, 7},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_succ_samepad_stride1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{29, 53, 37, 1}, {29, 53, 37, 1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({14, 22})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {29, 53, 37, 308},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_succ_samepad_stride_not1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{29, 53, 37, 1}, {29, 53, 37, 1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({14, 22})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {29, 27, 13, 308},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_succ_validpad)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{1, 3, 8, 2}, {1, 3, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 4})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("VALID")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({10})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 2, 24},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_succ_calcpad_unknow_n)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{-1, 2, 6, 19}, {-1, 2, 6, 12}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 6})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 1, 2})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 36, 2, 7},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_succ_calcpad_unknow_c)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{1, -1, 6, 19}, {1, -1, 6, 12}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 6})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 1, 2})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, -1, 2, 7},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_succ_calcpad_unknow_h)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{1, 2, -1, 19}, {1, 2, -1, 12}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 6})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 1, 2})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 36, -1, 7},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_succ_validpad_unkown_h)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{1, -1, 8, 2}, {1, -1, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 4})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("VALID")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({10})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, -1, 2, 24},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Im2colInfershapeTest, im2col_infer_shape_succ_samepad_unknow_w)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Im2col",
        {
            {{{29, 53, -1, 1}, {29, 53, -1, 1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({14, 22})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {29, 27, -1, 308},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
