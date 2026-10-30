/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

class StridedSliceInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StridedSlice SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StridedSlice TearDown" << std::endl;
    }
};

TEST_F(StridedSliceInfershape, strided_slice_infershape_test1)
{
    vector<int64_t> beginValue = {0};
    vector<int64_t> endValue = {1};
    vector<int64_t> stridesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_negative_index)
{
    vector<int64_t> beginValue = {-1};
    vector<int64_t> endValue = {10};
    vector<int64_t> stridesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_begin_mask)
{
    vector<int64_t> beginValue = {0};
    vector<int64_t> endValue = {5};
    vector<int64_t> stridesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_end_mask)
{
    vector<int64_t> beginValue = {5};
    vector<int64_t> endValue = {10};
    vector<int64_t> stridesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_negative_stride)
{
    vector<int64_t> beginValue = {9};
    vector<int64_t> endValue = {0};
    vector<int64_t> stridesValue = {-1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{9}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_multi_dim)
{
    vector<int64_t> beginValue = {1, 2};
    vector<int64_t> endValue = {3, 5};
    vector<int64_t> stridesValue = {1, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{4, 6}, {4, 6}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_shrink_axis)
{
    vector<int64_t> beginValue = {2};
    vector<int64_t> endValue = {3};
    vector<int64_t> stridesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_new_axis)
{
    vector<int64_t> beginValue = {0};
    vector<int64_t> endValue = {1};
    vector<int64_t> stridesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_ellipsis)
{
    vector<int64_t> beginValue = {0, 0};
    vector<int64_t> endValue = {2, 3};
    vector<int64_t> stridesValue = {1, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_int32)
{
    vector<int32_t> beginValue = {0};
    vector<int32_t> endValue = {5};
    vector<int32_t> stridesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_stride2)
{
    vector<int64_t> beginValue = {0};
    vector<int64_t> endValue = {10};
    vector<int64_t> stridesValue = {2};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_invalid_strides)
{
    vector<int64_t> beginValue = {0};
    vector<int64_t> endValue = {5};
    vector<int64_t> stridesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_begin_greater_than_end)
{
    vector<int64_t> beginValue = {5};
    vector<int64_t> endValue = {2};
    vector<int64_t> stridesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceInfershape, strided_slice_infershape_5d)
{
    vector<int64_t> beginValue = {1, 2, 3};
    vector<int64_t> endValue = {3, 5, 7};
    vector<int64_t> stridesValue = {1, 1, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSlice",
        {
            {{{4, 6, 8, 10, 12}, {4, 6, 8, 10, 12}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 10, 12}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}