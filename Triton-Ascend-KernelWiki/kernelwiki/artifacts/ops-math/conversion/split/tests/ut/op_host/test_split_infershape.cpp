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
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SplitInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SplitInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SplitInfershape TearDown" << std::endl;
    }
};

// Test: Split infershape with same shape
TEST_F(SplitInfershape, split_infershape_same_shape)
{
    int32_t split_dim = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "Split",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
            {{{11, 16}, {11, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{11, 8}, {11, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with invalid shape
TEST_F(SplitInfershape, split_infershape_invalid_xshape)
{
    int32_t split_dim = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "Split",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with invalid shape
TEST_F(SplitInfershape, split_infershape_invalid_split_dim)
{
    int32_t split_dim = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "Split",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with invalid dtype
TEST_F(SplitInfershape, split_infershape_invalid_dtype)
{
    int32_t split_dim = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "Split",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, &split_dim},
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with negative split_dim (relative indexing)
TEST_F(SplitInfershape, split_infershape_negative_split_dim)
{
    int32_t split_dim = -1;
    gert::InfershapeContextPara infershapeContextPara(
        "Split",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4}, {4, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with split_dim not get const (unknown split_dim)
TEST_F(SplitInfershape, split_infershape_unknown_split_dim)
{
    std::vector<int64_t> splitDimValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "Split",
        {
            {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND, true, splitDimValue.data()},
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 8}, {2, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with dynamic dim (-1) in split dimension
TEST_F(SplitInfershape, split_infershape_dynamic_dim)
{
    int32_t split_dim = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "Split",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with num_split equals 1
TEST_F(SplitInfershape, split_infershape_single_split)
{
    int32_t split_dim = 0;
    gert::InfershapeContextPara infershapeContextPara(
        "Split",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: Split infershape with 3D tensor
TEST_F(SplitInfershape, split_infershape_3d_tensor)
{
    int32_t split_dim = 0;
    gert::InfershapeContextPara infershapeContextPara(
        "Split",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
            {{{6, 4, 8}, {6, 4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 8}, {2, 4, 8}, {2, 4, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
