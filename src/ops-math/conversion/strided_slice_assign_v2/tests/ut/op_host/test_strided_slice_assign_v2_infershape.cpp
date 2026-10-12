/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_strided_slice_assign_v2_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class StridedSliceAssignV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StridedSliceAssignV2Infershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StridedSliceAssignV2Infershape TearDown" << std::endl;
    }
};

TEST_F(StridedSliceAssignV2Infershape, strided_slice_assign_v2_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceAssignV2",
        {
            {{{4, 6, 8}, {4, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 2, 4}, {2, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 6, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceAssignV2Infershape, strided_slice_assign_v2_infershape_test2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceAssignV2",
        {
            {{{10, 20, 30}, {10, 20, 30}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 10, 15}, {5, 10, 15}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10, 20, 30},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceAssignV2Infershape, strided_slice_assign_v2_infershape_test3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceAssignV2",
        {
            {{{100, 200}, {100, 200}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{50, 100}, {50, 100}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {100, 200},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceAssignV2Infershape, strided_slice_assign_v2_infershape_test4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceAssignV2",
        {
            {{{-1, 6, 8}, {-1, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, 2, 4}, {-1, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 6, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceAssignV2Infershape, strided_slice_assign_v2_infershape_test5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceAssignV2",
        {
            {{{5, 10, 15, 20}, {5, 10, 15, 20}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 5, 8, 10}, {2, 5, 8, 10}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 10, 15, 20},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StridedSliceAssignV2Infershape, strided_slice_assign_v2_infershape_test6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StridedSliceAssignV2",
        {
            {{{1000}, {1000}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{500}, {500}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1000},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
