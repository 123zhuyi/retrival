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
#include <string>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class TopKPQDistanceV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TopKPQDistanceV2Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TopKPQDistanceV2Infershape TearDown" << std::endl; }
};

TEST_F(TopKPQDistanceV2Infershape, topk_pq_distance_v2_infershape_asc)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TopKPQDistanceV2",
        {
            {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("ASC")},
            {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
            {"group_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4},
        {2, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TopKPQDistanceV2Infershape, topk_pq_distance_v2_infershape_des)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TopKPQDistanceV2",
        {
            {{{3, 16}, {3, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("DES")},
            {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
            {"group_size", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 8},
        {3, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TopKPQDistanceV2Infershape, topk_pq_distance_v2_infershape_dynamic_batch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TopKPQDistanceV2",
        {
            {{{-1, 8}, {-1, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1, 4}, {-1, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("ASC")},
            {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
            {"group_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 4},
        {-1, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TopKPQDistanceV2Infershape, topk_pq_distance_v2_infershape_order_invalid)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TopKPQDistanceV2",
        {
            {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("asc")},
            {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
            {"group_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(TopKPQDistanceV2Infershape, topk_pq_distance_v2_infershape_k_not_positive)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TopKPQDistanceV2",
        {
            {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("ASC")},
            {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"group_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(TopKPQDistanceV2Infershape, topk_pq_distance_v2_infershape_k_greater_than_element_num)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TopKPQDistanceV2",
        {
            {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("ASC")},
            {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(16)},
            {"group_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(TopKPQDistanceV2Infershape, topk_pq_distance_v2_infershape_group_size_not_positive)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TopKPQDistanceV2",
        {
            {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("ASC")},
            {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
            {"group_size", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(TopKPQDistanceV2Infershape, topk_pq_distance_v2_infershape_element_num_not_multiple_of_group_size)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TopKPQDistanceV2",
        {
            {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("ASC")},
            {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
            {"group_size", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(TopKPQDistanceV2Infershape, topk_pq_distance_v2_infershape_input0_not_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TopKPQDistanceV2",
        {
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("ASC")},
            {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
            {"group_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(TopKPQDistanceV2Infershape, topk_pq_distance_v2_infershape_input1_not_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TopKPQDistanceV2",
        {
            {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("ASC")},
            {"k", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
            {"group_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
