/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "gtest/gtest.h"
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SquareSumV1Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "square_sum_v1_proto Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "square_sum_v1_proto Proto Test TearDown" << std::endl;
    }
};

TEST_F(SquareSumV1Test, square_sum_v1_infershape_test_01)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SquareSumV1Test, square_sum_v1_infershape_test_02)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{-1, 8, -1, -1}, {-1, 8, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 2})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 8, 1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SquareSumV1Test, static_square_sum_v1_infershape_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{16, 48, 16, 32}, {16, 48, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // attr
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 2})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 48, 1, 32},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SquareSumV1Test, InfershapeSquareSumV1_001)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SquareSumV1",
        {
            {{{4, 3, 1}, {4, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            // attr
            {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-1, 2, 1})},
            {"keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 1, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
