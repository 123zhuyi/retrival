/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_one_hot_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "any_value.h"

class OneHotInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "OneHotInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "OneHotInfershape TearDown" << std::endl;
    }
};

// Helper function to create OpAttr
static gert::InfershapeContextPara::OpAttr MakeAxisAttr(int64_t axis)
{
    return gert::InfershapeContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom(axis));
}

// Test axis=-1 (default), depth=5
TEST_F(OneHotInfershape, test_infershape_axis_default)
{
    int32_t depth_val = 5;
    gert::InfershapeContextPara infershapeContextPara(
        "OneHot",
        {
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &depth_val},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            MakeAxisAttr(-1),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test axis=0
TEST_F(OneHotInfershape, test_infershape_axis_0)
{
    int32_t depth_val = 5;
    gert::InfershapeContextPara infershapeContextPara(
        "OneHot",
        {
            {{{4, 3}, {4, 3}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &depth_val},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            MakeAxisAttr(0),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 4, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test axis=1
TEST_F(OneHotInfershape, test_infershape_axis_1)
{
    int32_t depth_val = 10;
    gert::InfershapeContextPara infershapeContextPara(
        "OneHot",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &depth_val},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            MakeAxisAttr(1),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 10, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test axis=2
TEST_F(OneHotInfershape, test_infershape_axis_2)
{
    int32_t depth_val = 10;
    gert::InfershapeContextPara infershapeContextPara(
        "OneHot",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &depth_val},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            MakeAxisAttr(2),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 10, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test axis=3 (last dim)
TEST_F(OneHotInfershape, test_infershape_axis_3)
{
    int32_t depth_val = 10;
    gert::InfershapeContextPara infershapeContextPara(
        "OneHot",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &depth_val},
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            MakeAxisAttr(3),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 10}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test 1D input with axis=-1
TEST_F(OneHotInfershape, test_infershape_1d_input)
{
    int32_t depth_val = 5;
    gert::InfershapeContextPara infershapeContextPara(
        "OneHot",
        {
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &depth_val},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            MakeAxisAttr(-1),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test uint8 indices type
TEST_F(OneHotInfershape, test_infershape_uint8_indices)
{
    int32_t depth_val = 8;
    gert::InfershapeContextPara infershapeContextPara(
        "OneHot",
        {
            {{{4, 3}, {4, 3}}, ge::DT_UINT8, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &depth_val},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            MakeAxisAttr(-1),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test failed case: axis out of range
TEST_F(OneHotInfershape, test_infershape_failed_axis_out_of_range)
{
    int32_t depth_val = 5;
    gert::InfershapeContextPara infershapeContextPara(
        "OneHot",
        {
            {{{4, 3}, {4, 3}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &depth_val},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            MakeAxisAttr(3), // axis=3 is out of range for 2D input
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
