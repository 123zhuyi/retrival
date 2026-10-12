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
 * \file test_coordinates_1d_to_2d_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;
class Coordinates1DTo2DInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Coordinates1DTo2DInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Coordinates1DTo2DInferShapeTest TearDown" << std::endl; }
};

// x shape {100}, DT_INT32. Expect all 3 outputs (row/col/n) inherit x shape {100}.
TEST_F(Coordinates1DTo2DInferShapeTest, coordinates_1d_to_2d_infershape_success)
{
    gert::InfershapeContextPara::TensorDescription x({{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shape({{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription row({{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription col({{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription n({{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Coordinates1DTo2D", {x, shape}, {row, col, n});
    std::vector<std::vector<int64_t>> expectOutputShape = {{100}, {100}, {100}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// x DT_INT64: verify outputs follow x dtype.
TEST_F(Coordinates1DTo2DInferShapeTest, coordinates_1d_to_2d_infershape_int64)
{
    gert::InfershapeContextPara::TensorDescription x({{50}, {50}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription shape({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription row({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription col({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription n({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Coordinates1DTo2D", {x, shape}, {row, col, n});
    std::vector<std::vector<int64_t>> expectOutputShape = {{50}, {50}, {50}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
