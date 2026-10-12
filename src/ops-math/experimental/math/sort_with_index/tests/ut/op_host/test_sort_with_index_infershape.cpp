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
 * \file test_sort_with_index_infershape.cpp
 * \brief Core-path infershape UT for SortWithIndex (ascend910b).
 *        Covers: y.shape = x.shape, sorted_index.shape = index.shape;
 *        static-shape constraint x.shape == index.shape (mismatch -> GRAPH_FAILED).
 *        Main-line dtype pair is (x = float16, index = int32).
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SortWithIndexInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "SortWithIndexInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SortWithIndexInfershape TearDown" << std::endl;
    }
};

// Main path: 2-D, x.shape == index.shape -> y.shape = x.shape, sorted_index.shape = index.shape.
TEST_F(SortWithIndexInfershape, sortwithindex_infershape_same_shape_2d)
{
    gert::InfershapeContextPara infershapeContextPara("SortWithIndex",
        {
            {{{3, 8}, {3, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 8}, {3, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 8}, {3, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 1-D, x.shape == index.shape (single-row sort slice; k = N path).
TEST_F(SortWithIndexInfershape, sortwithindex_infershape_same_shape_1d)
{
    gert::InfershapeContextPara infershapeContextPara("SortWithIndex",
        {
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4}, {4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// sort-axis length == 1 (k = N = 1, single-element slice -> direct copy).
TEST_F(SortWithIndexInfershape, sortwithindex_infershape_axis_len_one)
{
    gert::InfershapeContextPara infershapeContextPara("SortWithIndex",
        {
            {{{4, 1}, {4, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 1}, {4, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 1}, {4, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Static-shape constraint: x.shape != index.shape -> shape_mismatch -> GRAPH_FAILED.
TEST_F(SortWithIndexInfershape, sortwithindex_infershape_shape_mismatch_fail)
{
    gert::InfershapeContextPara infershapeContextPara("SortWithIndex",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
