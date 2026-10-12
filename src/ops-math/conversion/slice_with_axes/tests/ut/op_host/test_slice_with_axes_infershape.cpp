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

class SliceWithAxesInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SliceWithAxesInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SliceWithAxesInfershape TearDown" << std::endl;
    }
};

TEST_F(SliceWithAxesInfershape, slice_with_axes_1d_basic)
{
    int32_t offsets[] = {2};
    int32_t sizes[] = {5};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{10}, {10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceWithAxesInfershape, slice_with_axes_2d_single_axis)
{
    int32_t offsets[] = {0};
    int32_t sizes[] = {4};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceWithAxesInfershape, slice_with_axes_3d_multi_axes)
{
    int32_t offsets[] = {1, 5};
    int32_t sizes[] = {5, 10};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{10, 20, 30}, {10, 20, 30}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 2})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 20, 10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceWithAxesInfershape, slice_with_axes_size_minus1)
{
    int32_t offsets[] = {30};
    int32_t sizes[] = {-1};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{100}, {100}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {70},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceWithAxesInfershape, slice_with_axes_int64_index)
{
    int64_t offsets[] = {10};
    int64_t sizes[] = {20};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{50, 60}, {50, 60}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsets},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {20, 60},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
