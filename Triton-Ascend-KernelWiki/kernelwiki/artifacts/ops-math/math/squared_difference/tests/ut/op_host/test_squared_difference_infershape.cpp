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
    gert::InfershapeContextPara infershapeContextPara(
        "SquaredDifference",
        {
            {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(squared_difference, squared_difference_infer_shape_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquaredDifference",
        {
            {{{60}, {60}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{60}, {60}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {60},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(squared_difference, squared_difference_infer_shape_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquaredDifference",
        {
            {{{60}, {60}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{60, 60, 60, 60}, {60, 60, 60, 60}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {60, 60, 60, 60},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(squared_difference, squared_difference_infer_shape_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquaredDifference",
        {
            {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, -1, -1, 60}, {-1, -1, -1, 60}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1, 60},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(squared_difference, infer_axis_type_with_input_dim_num_great_than_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquaredDifference",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}