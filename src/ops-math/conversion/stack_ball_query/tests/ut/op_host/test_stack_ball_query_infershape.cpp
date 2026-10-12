/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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
    static void SetUpTestCase() { std::cout << "StackBallQueryInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StackBallQueryInfershape TearDown" << std::endl; }
};

// ===== 正常路径 =====

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_fp32_int32)
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
        {10, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_fp16_int64)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{5, 30}, {5, 30}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{15, 3}, {15, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(0.5)),
            gert::InfershapeContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(10)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {15, 10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ===== Unknown Rank（-2）正例 =====

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_unknown_rank_xyz)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_unknown_rank_center_xyz)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_unknown_rank_batch_cnt)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
            gert::InfershapeContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

// ===== Unknown Shape（-1）正例 =====

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_unknown_shape_xyz)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{-1, 20}, {-1, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_unknown_shape_center_xyz)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, 3}, {-1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_unknown_shape_batch_cnt)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
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
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ===== 维度校验失败用例 =====

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_neg_xyz_not_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{20}, {20}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_neg_center_xyz_not_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 3, 1}, {10, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
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
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_neg_xyz_batch_cnt_not_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
            gert::InfershapeContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(StackBallQueryInfershape, stack_ball_query_infershape_neg_center_batch_cnt_not_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "StackBallQuery",
        {
            {{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
            gert::InfershapeContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
