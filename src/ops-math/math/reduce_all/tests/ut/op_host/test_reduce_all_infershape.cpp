/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_reduce_all_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ReduceAllInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceAllInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceAllInfershape TearDown" << std::endl; }
};

TEST_F(ReduceAllInfershape, reduce_all_infershape_float_test)
{
    std::vector<int64_t> axesValue = {0, 1, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceAll",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceAllInfershape, reduce_all_infershape_float16_test)
{
    std::vector<int64_t> axesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceAll",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceAllInfershape, reduce_all_infershape_int32_test)
{
    std::vector<int64_t> axesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceAll",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceAllInfershape, reduce_all_infershape_bool_test)
{
    std::vector<int64_t> axesValue = {0, 1, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceAll",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceAllInfershape, reduce_all_infershape_1d_test)
{
    std::vector<int64_t> axesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceAll",
        {
            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
