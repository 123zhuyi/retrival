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

class BiasAddGradProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BiasAddGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BiasAddGrad Proto Test TearDown" << std::endl;
    }
};

TEST_F(BiasAddGradProtoTest, BiasAddGrad_infershape_diff_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BiasAddGrad",
        {
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BiasAddGradProtoTest, BiasAddGrad_infershape_diff_test_rt2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BiasAddGrad",
        {
            {{{5, 3, 9}, {5, 3, 9}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {9},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BiasAddGradProtoTest, infer_axis_type_test_01)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BiasAddGrad",
        {
            {{{3, 4, 5, 6}, {3, 4, 5, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // attr
            {"data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {6},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
