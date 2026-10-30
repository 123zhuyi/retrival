/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_strided_slice_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include <vector>

using namespace std;

class StridedSliceV3Infershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StridedSliceV3 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StridedSliceV3 TearDown" << std::endl;
    }
};

TEST_F(StridedSliceV3Infershape, strided_slice_infershape_test1)
{
    vector<int64_t> beginValue = {0};
    vector<int64_t> endValue = {1};
    vector<int64_t> axesValue = {0};
    vector<int64_t> stridesValue = {1};

    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV3",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceV3Infershape, strided_slice_infershape_test2)
{
    vector<int64_t> beginValue = {7};
    vector<int64_t> endValue = {9};
    vector<int64_t> axesValue = {1};
    vector<int64_t> stridesValue = {2};

    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV3",
        {
            {{{10, 10}, {10, 10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceV3Infershape, strided_slice_infershape_test3)
{
    vector<int64_t> endValue = {9};
    vector<int64_t> axesValue = {1};
    vector<int64_t> stridesValue = {2};

    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV3",
        {
            {{{10, 10}, {10, 10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, false},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceV3Infershape, strided_slice_infershape_test4)
{
    vector<int64_t> beginValue = {7};
    vector<int64_t> endValue = {9};
    vector<int64_t> stridesValue = {2};

    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV3",
        {
            {{{10, 10}, {10, 10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, false},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceV3Infershape, strided_slice_infershape_test5)
{
    vector<int64_t> beginValue = {7};
    vector<int64_t> endValue = {9};
    vector<int64_t> stridesValue = {2};

    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV3",
        {
            {{{10, 10}, {10, 10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr}, // shape为0表示axes取所有值
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 10}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceV3Infershape, strided_slice_infershape_test6)
{
    vector<int64_t> beginValue = {7};
    vector<int64_t> endValue = {9};
    vector<int64_t> axesValue = {1};
    vector<int64_t> stridesValue = {2};

    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV3",
        {
            {{{10, 10}, {10, 10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()}, // shape为空表示一个scalar
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceV3Infershape, strided_slice_infershape_test7)
{
    vector<int64_t> beginValue = {7};
    vector<int64_t> endValue = {9};
    vector<int64_t> stridesValue = {2};

    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV3",
        {
            {{{10, 10}, {10, 10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr}, // shape为空表示一个scalar,但没有赋值，动态流程
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceV3Infershape, strided_slice_infershape_test8)
{
    vector<int64_t> beginValue = {9};
    vector<int64_t> endValue = {-5};
    vector<int64_t> axesValue = {0};
    vector<int64_t> stridesValue = {17};

    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV3",
        {
            {{{2147483649}, {2147483649}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{126322567}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceV3Infershape, strided_slice_infershape_test9)
{
    vector<int64_t> beginValue = {9};
    vector<int64_t> endValue = {0};
    vector<int64_t> axesValue = {0};
    vector<int64_t> stridesValue = {-3};

    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV3",
        {
            {{{10}, {10}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
