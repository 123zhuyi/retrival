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

class Pad : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
        std::cout << "Pad SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
        std::cout << "Pad TearDown" << std::endl;
    }
};

using namespace ge;

TEST_F(Pad, Pad_const_infer_1)
{
    std::vector<int32_t> values = {1, 1, 2, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "Pad",
        {{{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 9}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(Pad, Pad_const_infer_2_error_paddings_num)
{
    std::vector<int32_t> values = {1, 1, 2, 2, 0, 0};
    gert::InfershapeContextPara infershapeContextPara(
        "Pad",
        {{{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3, 2}, {3, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 9}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(Pad, Pad_const_infer_3_error_paddings_dtype)
{
    std::vector<int32_t> values = {1, 1, 2, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "Pad",
        {{{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_UINT64, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 9}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(Pad, Pad_const_infer_4_error_paddings_num)
{
    std::vector<int32_t> values = {1, 1, 2, 2, 0, 0};
    gert::InfershapeContextPara infershapeContextPara(
        "Pad",
        {{{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{3, 2}, {3, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 9}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test scenario: pad with INT64 const paddings and valid shape, expect infershape to succeed with correct output dims
TEST_F(Pad, Pad_const_infer_int64_success)
{
    std::vector<int64_t> values = {1LL, 1LL, 2LL, 2LL};
    gert::InfershapeContextPara infershapeContextPara(
        "Pad",
        {{{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 9}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: pad with INT32 negative paddings causing negative output dimension, expect infershape to fail
TEST_F(Pad, Pad_const_infer_negative_dim_int32)
{
    std::vector<int32_t> values = {-10, -1, 0, 0};
    gert::InfershapeContextPara infershapeContextPara(
        "Pad",
        {{{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test scenario: pad with INT64 negative paddings causing negative output dimension, expect infershape to fail
TEST_F(Pad, Pad_const_infer_negative_dim_int64)
{
    std::vector<int64_t> values = {-10LL, -1LL, 0LL, 0LL};
    gert::InfershapeContextPara infershapeContextPara(
        "Pad",
        {{{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, values.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// Test scenario: pad with unknown rank input shape, expect infershape to succeed with unknown rank output
TEST_F(Pad, Pad_infer_unknown_rank)
{
    std::vector<int32_t> values = {1, 1};
    gert::StorageShape xShape = {{-2}, {-2}};
    gert::InfershapeContextPara infershapeContextPara(
        "Pad",
        {{xShape, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, values.data()}},
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test scenario: pad with non-const paddings (dynamic shape), expect infershape to succeed with all unknown dims
TEST_F(Pad, Pad_infer_dynamic_paddings)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pad", {{{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
