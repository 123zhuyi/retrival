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

class PolarInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Polar InferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Polar InferShape TearDown" << std::endl;
    }
};

TEST_F(PolarInferShape, polar_infershape_same_shape)
{
    gert::InfershapeContextPara infershapeContextPara("Polar",
    {
        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{}, {}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PolarInferShape, polar_infershape_broadcast)
{
    gert::InfershapeContextPara infershapeContextPara("Polar",
    {
        {{{60}, {60}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{60, 60, 60, 60}, {60, 60, 60, 60}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{}, {}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{60, 60, 60, 60}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PolarInferShape, polar_infershape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara("Polar",
    {
        {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{-1, -1, -1, 60}, {-1, -1, -1, 60}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{}, {}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, -1, 60}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PolarInferShape, polar_infershape_scalar_broadcast)
{
    gert::InfershapeContextPara infershapeContextPara("Polar",
    {
        {{{1, 200}, {1, 200}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{2, 200}, {2, 200}}, ge::DT_FLOAT, ge::FORMAT_ND},
    },
    {
        {{{}, {}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
    });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 200}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
