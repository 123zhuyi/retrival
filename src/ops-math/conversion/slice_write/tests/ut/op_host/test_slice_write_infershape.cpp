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
 * \file test_slice_write_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;
class SliceWriteInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SliceWriteInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SliceWriteInferShapeTest TearDown" << std::endl; }
};

// x {100, 200}, output shape should equal x shape.
TEST_F(SliceWriteInferShapeTest, slice_write_infershape_success)
{
    gert::InfershapeContextPara::TensorDescription x({{100, 200}, {100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription begin({{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription value({{50, 50}, {50, 50}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("SliceWrite", {x, begin, value}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{100, 200}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// x DT_INT32: verify output dtype follows x.
TEST_F(SliceWriteInferShapeTest, slice_write_infershape_int32)
{
    gert::InfershapeContextPara::TensorDescription x({{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription begin({{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription value({{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("SliceWrite", {x, begin, value}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// x is DT_FLOAT, value is DT_INT32: dtype mismatch, expect failure.
TEST_F(SliceWriteInferShapeTest, slice_write_infershape_dtype_mismatch)
{
    gert::InfershapeContextPara::TensorDescription x({{100, 200}, {100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription begin({{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription value({{50, 50}, {50, 50}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("SliceWrite", {x, begin, value}, {out});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
