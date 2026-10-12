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

// InplaceTopKDistance declares no output and updates its inputs in place, so its infershape is an empty
// registration (mirroring the empty COMMON_INFER_FUNC in the canndev source). These cases only assert that the
// infershape is registered and reports success; there is no output shape to check.
class InplaceTopKDistanceInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InplaceTopKDistanceInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "InplaceTopKDistanceInfershape TearDown" << std::endl; }
};

TEST_F(InplaceTopKDistanceInfershape, inplace_top_k_distance_infershape_float)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InplaceTopKDistance",
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {},
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("asc")},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {});
}

TEST_F(InplaceTopKDistanceInfershape, inplace_top_k_distance_infershape_float16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InplaceTopKDistance",
        {
            {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {},
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("desc")},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {});
}

TEST_F(InplaceTopKDistanceInfershape, inplace_top_k_distance_infershape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara(
        "InplaceTopKDistance",
        {
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {},
        {
            {"order", Ops::Math::AnyValue::CreateFrom<std::string>("asc")},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {});
}
