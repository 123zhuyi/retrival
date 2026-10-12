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

class PadV3GradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
        std::cout << "PadV3GradInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
        std::cout << "PadV3GradInfershapeTest TearDown" << std::endl;
    }
};

using namespace ge;

TEST_F(PadV3GradInfershapeTest, pad_v3_grad_infer_1)
{
    std::vector<int32_t> values = {1, 1, 2, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "PadV3Grad",
        {{{{5, 9}, {5, 9}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")},
         {"paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PadV3GradInfershapeTest, pad_v3_grad_infer_2_error_paddings_dtype)
{
    std::vector<int32_t> values = {1, 1, 2, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "PadV3Grad",
        {{{{5, 9}, {5, 9}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_UINT64, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")},
         {"paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(PadV3GradInfershapeTest, pad_v3_grad_infer_3_error_paddings_num)
{
    std::vector<int32_t> values = {1, 1, 2, 2, 0, 0};
    gert::InfershapeContextPara infershapeContextPara(
        "PadV3Grad",
        {{{{5, 9}, {5, 9}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3, 2}, {3, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")},
         {"paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(PadV3GradInfershapeTest, pad_v3_grad_infer_4_error_output)
{
    std::vector<int32_t> values = {6, 7, 2, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "PadV3Grad",
        {{{{5, 9}, {5, 9}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")},
         {"paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-8, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test scenario: non-const paddings tensor, expect all dims set to -1 (unknown)
TEST_F(PadV3GradInfershapeTest, pad_v3_grad_infer_5_unknown)
{
    std::vector<int32_t> values = {};
    gert::InfershapeContextPara infershapeContextPara(
        "PadV3Grad",
        {{{{5, 9}, {5, 9}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")},
         {"paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: unknown rank input shape, expect output set to unknown rank
TEST_F(PadV3GradInfershapeTest, pad_v3_grad_infer_6_unknown_rank)
{
    gert::StorageShape xShape = {{-2}, {-2}};
    gert::StorageShape padShape = {{2, 2}, {2, 2}};
    std::vector<int32_t> values = {1, 1, 2, 2};

    gert::InfershapeContextPara infershapeContextPara(
        "PadV3Grad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {padShape, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")},
         {"paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: empty tensor input (0-dim shape), expect infer shape to fail
TEST_F(PadV3GradInfershapeTest, pad_v3_grad_infer_7_empty_tensor)
{
    std::vector<int32_t> values = {};
    gert::InfershapeContextPara infershapeContextPara(
        "PadV3Grad",
        {{{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0, 2}, {0, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")},
         {"paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test scenario: INT64 paddings dtype with contiguous=false, expect correct output shape
// Non-contiguous layout: paddings = [front0, front1, back0, back1], so front=[1,2], back=[3,4]
TEST_F(PadV3GradInfershapeTest, pad_v3_grad_infer_8_int64_non_contiguous)
{
    std::vector<int64_t> values = {1, 2, 3, 4};
    gert::InfershapeContextPara infershapeContextPara(
        "PadV3Grad",
        {{{{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")},
         {"paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(false)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 14}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: x shape with unknown dim value (-1) and valid paddings, expect unknown dim in output
TEST_F(PadV3GradInfershapeTest, pad_v3_grad_infer_9_unknown_dim)
{
    std::vector<int32_t> values = {1, 1, 2, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "PadV3Grad",
        {{{{-1, 9}, {-1, 9}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{"mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")},
         {"paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
