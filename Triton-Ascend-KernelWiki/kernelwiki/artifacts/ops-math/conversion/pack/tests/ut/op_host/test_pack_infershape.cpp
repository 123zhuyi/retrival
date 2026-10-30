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
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class PackInferTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PackInferTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PackInferTest TearDown" << std::endl;
    }
};

TEST_F(PackInferTest, pack_infer_test_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pack",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        },
        {2}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PackInferTest, pack_infer_test_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pack",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        },
        {2}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 2, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PackInferTest, pack_infer_test_3rt)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pack",
        {
            {{{3, -1, -1, 5}, {3, -1, -1, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, -1, -1, 5}, {3, -1, -1, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, -1, -1, 5}, {3, -1, -1, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 3, -1, -1, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PackInferTest, pack_infer_test_4rt)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pack",
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PackInferTest, pack_infer_test_6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pack",
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        },
        {2}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PackInferTest, pack_infer_test_7)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pack",
        {
            {{{3, -1, -1, 5}, {3, -1, -1, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, -1, -1, 5}, {3, -1, -1, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(PackInferTest, pack_infer_test_8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pack",
        {
            {{{3, 2, 2, 5}, {3, 2, 2, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(10)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        },
        {1}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
