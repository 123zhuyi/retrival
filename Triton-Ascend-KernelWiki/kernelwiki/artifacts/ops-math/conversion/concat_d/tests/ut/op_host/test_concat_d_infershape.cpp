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

class ConcatDInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ConcatDInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ConcatDInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(ConcatDInfershapeTest, concat_d_infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatD",
        {
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 12},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatDInfershapeTest, concat_d_infer_shape_fp16_n1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatD",
        {
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        },
        {1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatDInfershapeTest, concat_d_infer_shape_fp16_shape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatD",
        {
            {{{2, 100, 1}, {2, 100, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 24}, {2, 100, 24}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 34}, {2, 100, 34}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 59},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatDInfershapeTest, concat_d_infer_shape_fp16_errorshape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatD",
        {
            {{{2, 100, 1}, {2, 100, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 24}, {2, 100, 24}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 34}, {2, 100, 34}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatDInfershapeTest, concat_d_infer_shape_fp16_errordim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatD",
        {
            {{{2, 100, 1}, {2, 100, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 24}, {2, 100, 24}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 34}, {2, 100, 34}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(5)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatDInfershapeTest, concat_d_infer_shape_fp16_errorshapdim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatD",
        {
            {{{2, 100, 1}, {2, 100, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 2, 4}, {2, 100, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 34}, {2, 100, 34}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(ConcatDInfershapeTest, concat_d_infer_shape_fp16_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatD",
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatDInfershapeTest, concat_d_infer_shape_no_shape_range_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatD",
        {
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 12},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatDInfershapeTest, concat_d_infer_shape_no_shape_range_mix_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatD",
        {
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 12},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatDInfershapeTest, concat_d_infer_shape_dynamic_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatD",
        {
            {{{2, 100, 4}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 12},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConcatDInfershapeTest, concat_d_infer_shape_dynamic2_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConcatD",
        {
            {{{2, 100, 4}, {-1, -1, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"N", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        {3}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100, 12},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
