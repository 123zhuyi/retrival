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

class SliceLastDimInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SliceLastDimInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SliceLastDimInfershape TearDown" << std::endl;
    }
};

TEST_F(SliceLastDimInfershape, slice_last_dim_basic_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{10}, {10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(7)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceLastDimInfershape, slice_last_dim_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(4)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceLastDimInfershape, slice_last_dim_3d_with_stride)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{2, 3, 10}, {2, 3, 10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(10)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceLastDimInfershape, slice_last_dim_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceLastDimInfershape, slice_last_dim_full_slice)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{5, 100}, {5, 100}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(100)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 100},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
