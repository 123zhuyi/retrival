/**
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

class StridedSliceV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StridedSliceV2 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StridedSliceV2 TearDown" << std::endl; }
};

TEST_F(StridedSliceV2Infershape, strided_slice_infershape_test1)
{
    vector<int64_t> beginValue = {0};
    vector<int64_t> endValue = {1};
    vector<int64_t> stridesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV2",
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

// 带 axes 和 strides，DT_INT32，多轴含负轴
TEST_F(StridedSliceV2Infershape, strided_slice_v2_int32_with_axes_strides)
{
    vector<int32_t> beginValue = {0, 1};
    vector<int32_t> endValue = {2, 3};
    vector<int32_t> axesValue = {0, -1};
    vector<int32_t> stridesValue = {1, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV2",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, beginValue.data()},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, endValue.data()},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, stridesValue.data()},
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
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 带 axes 和 strides，DT_INT64，多轴含负轴
TEST_F(StridedSliceV2Infershape, strided_slice_v2_int64_with_axes_strides)
{
    vector<int64_t> beginValue = {0, 1};
    vector<int64_t> endValue = {2, 3};
    vector<int64_t> axesValue = {0, -1};
    vector<int64_t> stridesValue = {1, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV2",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
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
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 无 axes 和 strides，走 ConstructValidAxis 的 resize+iota 路径
TEST_F(StridedSliceV2Infershape, strided_slice_v2_no_axes_no_strides)
{
    vector<int64_t> beginValue = {0, 1};
    vector<int64_t> endValue = {2, 3};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV2",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
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
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 未知秩输入 {-2}
TEST_F(StridedSliceV2Infershape, strided_slice_v2_unknown_rank)
{
    vector<int64_t> beginValue = {0};
    vector<int64_t> endValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV2",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
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
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// begin/end shape dim0 为 -1，走 shape_max == -1 分支
TEST_F(StridedSliceV2Infershape, strided_slice_v2_unknown_shape_size)
{
    vector<int64_t> beginValue = {0};
    vector<int64_t> endValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV2",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
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
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

// axes/begin/end/strides 为非const (data为nullptr)
TEST_F(StridedSliceV2Infershape, strided_slice_v2_non_const_inputs)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceV2",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}
