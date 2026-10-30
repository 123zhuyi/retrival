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

class HistogramFixedWidthTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HistogramFixedWidthTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "HistogramFixedWidthTest TearDown" << std::endl; }
};

TEST_F(HistogramFixedWidthTest, infershape_success_fp32)
{
    int32_t nbinsValue = 50;
    gert::InfershapeContextPara infershapeContextPara(
        "HistogramFixedWidth",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsValue},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_INT32))},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{50}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(HistogramFixedWidthTest, infershape_success_fp16)
{
    int32_t nbinsValue = 100;
    gert::InfershapeContextPara infershapeContextPara(
        "HistogramFixedWidth",
        {
            {{{100}, {100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsValue},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_INT32))},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{100}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(HistogramFixedWidthTest, infershape_fail_range_shape_not_2)
{
    int32_t nbinsValue = 50;
    gert::InfershapeContextPara infershapeContextPara(
        "HistogramFixedWidth",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsValue},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_INT32))},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(HistogramFixedWidthTest, infershape_fail_invalid_dtype)
{
    int32_t nbinsValue = 50;
    gert::InfershapeContextPara infershapeContextPara(
        "HistogramFixedWidth",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsValue},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT))},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
