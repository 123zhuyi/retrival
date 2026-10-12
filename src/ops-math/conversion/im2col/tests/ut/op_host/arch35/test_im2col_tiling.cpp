/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/im2col/op_kernel/arch35/im2col_tilingdata.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class Im2colTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Im2colTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Im2colTilingTest TearDown" << std::endl;
    }
};

namespace {
Im2ColCompileInfo compileInfo;
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_input_shape_lt_4)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_input_shape_gt_4)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_input_format_invalid)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_no_ksizes)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_ksizes_size_lt_2)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2})},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_ksizes_size_gt_2)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_ksizes_value_le_0)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_strides_size_lt_1)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_strides_size_gt_2)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_strides_value_le_0)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_dilations_size_lt_1)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_dilations_size_gt_2)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_dilations_value_le_0)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_padding_mode_invalid)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_pads_size_lt_1)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_pads_size_eq_3)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_pads_size_gt_4)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_pads_value_lt_0)
{
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_effect_shape_gt_padding_shape_with_CALCULATED_mode)
{
    // padded input = (6, 15)
    // effect = (5, 16)
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, ImIm2colTilingTest_ParamCheck_effect_shape_gt_padding_shape_with_VALID_mode)
{
    // padded input = (3, 8)
    // effect = (3, 9)
    gert::TilingContextPara tilingContextPara(
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
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(Im2colTilingTest, Im2colTilingTest_GetInputShape_Succ_NCHW)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{10, 20, 30, 40}, {10, 20, 30, 40}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    Im2ColInputInfo* input = reinterpret_cast<Im2ColInputInfo*>(tilingInfo.tilingData.get());
    EXPECT_EQ(input->N, 10);
    EXPECT_EQ(input->C, 20);
    EXPECT_EQ(input->H, 30);
    EXPECT_EQ(input->W, 40);
}

TEST_F(Im2colTilingTest, Im2colTilingTest_GetInputShape_Succ_NHWC)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{10, 20, 30, 40}, {10, 20, 30, 40}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    Im2ColInputInfo* input = reinterpret_cast<Im2ColInputInfo*>(tilingInfo.tilingData.get());
    EXPECT_EQ(input->N, 10);
    EXPECT_EQ(input->H, 20);
    EXPECT_EQ(input->W, 30);
    EXPECT_EQ(input->C, 40);
}

TEST_F(Im2colTilingTest, Im2colTilingTest_GetInput_Succ_with_attrSize1)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{100, 200, 300, 400}, {100, 200, 300, 400}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 3})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({4})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({5})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({6})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    Im2ColInputInfo* input = reinterpret_cast<Im2ColInputInfo*>(tilingInfo.tilingData.get());
    EXPECT_EQ(input->hKernelSize, 3);
    EXPECT_EQ(input->wKernelSize, 3);
    EXPECT_EQ(input->hStride, 4);
    EXPECT_EQ(input->wStride, 4);
    EXPECT_EQ(input->hDilation, 5);
    EXPECT_EQ(input->wDilation, 5);
    EXPECT_EQ(input->hPaddingBefore, 6);
    EXPECT_EQ(input->hPaddingAfter, 6);
    EXPECT_EQ(input->wPaddingBefore, 6);
    EXPECT_EQ(input->wPaddingAfter, 6);
}

TEST_F(Im2colTilingTest, Im2colTilingTest_GetInput_Succ_with_attrSize2)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{100, 200, 300, 400}, {100, 200, 300, 400}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({4, 5})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({6, 7})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    Im2ColInputInfo* input = reinterpret_cast<Im2ColInputInfo*>(tilingInfo.tilingData.get());
    EXPECT_EQ(input->hKernelSize, 2);
    EXPECT_EQ(input->wKernelSize, 3);
    EXPECT_EQ(input->hStride, 4);
    EXPECT_EQ(input->wStride, 5);
    EXPECT_EQ(input->hDilation, 6);
    EXPECT_EQ(input->wDilation, 7);
    EXPECT_EQ(input->hPaddingBefore, 1);
    EXPECT_EQ(input->hPaddingAfter, 1);
    EXPECT_EQ(input->wPaddingBefore, 2);
    EXPECT_EQ(input->wPaddingAfter, 2);
}

TEST_F(Im2colTilingTest, Im2colTilingTest_GetInput_Succ_with_attrSize4)
{
    gert::TilingContextPara tilingContextPara(
        "Im2col",
        {
            {{{100, 200, 300, 400}, {100, 200, 300, 400}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {"ksizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
            {"strides", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("CALCULATED")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3, 4})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    Im2ColInputInfo* input = reinterpret_cast<Im2ColInputInfo*>(tilingInfo.tilingData.get());
    EXPECT_EQ(input->hPaddingBefore, 1);
    EXPECT_EQ(input->hPaddingAfter, 2);
    EXPECT_EQ(input->wPaddingBefore, 3);
    EXPECT_EQ(input->wPaddingAfter, 4);
}

TEST_F(Im2colTilingTest, Im2colTilingTest_GetInput_Succ_with_SAME_mode)
{
    gert::TilingContextPara tilingContextPara(
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
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("SAME")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    Im2ColInputInfo* input = reinterpret_cast<Im2ColInputInfo*>(tilingInfo.tilingData.get());
    EXPECT_EQ(input->hPaddingBefore, 6);
    EXPECT_EQ(input->hPaddingAfter, 7);
    EXPECT_EQ(input->wPaddingBefore, 10);
    EXPECT_EQ(input->wPaddingAfter, 11);
}

TEST_F(Im2colTilingTest, Im2colTilingTest_GetInput_Succ_with_VALID_mode)
{
    gert::TilingContextPara tilingContextPara(
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
            {"dilations", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"padding_mode", Ops::Math::AnyValue::CreateFrom<std::string>("VALID")},
            {"pads", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({10})},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    Im2ColInputInfo* input = reinterpret_cast<Im2ColInputInfo*>(tilingInfo.tilingData.get());
    EXPECT_EQ(input->hPaddingBefore, 0);
    EXPECT_EQ(input->hPaddingAfter, 0);
    EXPECT_EQ(input->wPaddingBefore, 0);
    EXPECT_EQ(input->wPaddingAfter, 0);
}
