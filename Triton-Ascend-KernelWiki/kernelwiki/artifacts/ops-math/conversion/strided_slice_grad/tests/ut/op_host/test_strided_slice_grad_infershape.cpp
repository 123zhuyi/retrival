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

class StridedSliceGradInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StridedSliceGrad SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StridedSliceGrad TearDown" << std::endl;
    }
};

TEST_F(StridedSliceGradInfershape, strided_slice_grad_infershape_test1)
{
    vector<int32_t> shape = {1, 96};
    vector<int32_t> begin = {0, 0};
    vector<int32_t> end = {1, 91};
    vector<int32_t> strides = {1, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceGrad",
        {
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, shape.data()},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, begin.data()},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, end.data()},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, strides.data()},
            {{{1, 96}, {1, 96}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 96}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceGradInfershape, strided_slice_grad_dynamic_shape_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceGrad",
        {
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, 96}, {-1, 96}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        {5}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(StridedSliceGradInfershape, strided_slice_grad_unknown_rank_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceGrad",
        {
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        {5}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceGradInfershape, strided_slice_grad_special_branch_test)
{
    vector<int32_t> begin = {0};
    vector<int32_t> end = {1};
    vector<int32_t> strides = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceGrad",
        {
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, begin.data()},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, end.data()},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, strides.data()},
            {{{-1, 96}, {-1, 96}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        {5}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(StridedSliceGradInfershape, strided_slice_grad_error_empty_shape_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceGrad",
        {
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, 96}, {-1, 96}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        {5}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(StridedSliceGradInfershape, strided_slice_grad_error_rank_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceGrad",
        {
            {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, 96}, {-1, 96}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        {5}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}