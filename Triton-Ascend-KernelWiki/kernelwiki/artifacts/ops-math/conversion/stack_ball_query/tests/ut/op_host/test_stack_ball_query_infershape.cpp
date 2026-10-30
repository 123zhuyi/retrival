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

class StackBallQueryInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StackBallQueryInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StackBallQueryInfershape TearDown" << std::endl;
    }
};

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
            gert::InfershapeContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_test2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{5, 30}, {5, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{15, 3}, {15, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(0.5)),
            gert::InfershapeContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(10)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_test3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{10, 50}, {10, 50}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{20, 3}, {20, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(2.0)),
            gert::InfershapeContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(16)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_test4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{8, 40}, {8, 40}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{12, 3}, {12, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.5)),
            gert::InfershapeContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(8)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_test5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{-1, 20}, {-1, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, 3}, {-1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
            gert::InfershapeContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(32)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 32},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
