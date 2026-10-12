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
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;

class TransDataInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TransDataInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TransDataInfershapeTest TearDown" << std::endl; }
};

// TransData infershape uses InferShape4Elewise: output shape = input shape.
TEST_F(TransDataInfershapeTest, infer_2d_nd_shape)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    gert::InfershapeContextPara para(std::string("TransData"), inputs, outputs);
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TransDataInfershapeTest, infer_4d_nchw_shape)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
    };
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
    };
    gert::InfershapeContextPara para(std::string("TransData"), inputs, outputs);
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TransDataInfershapeTest, infer_5d_ncdhw_shape)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_INT8, ge::FORMAT_NCDHW},
    };
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {{{}, {}}, ge::DT_INT8, ge::FORMAT_NCDHW},
    };
    gert::InfershapeContextPara para(std::string("TransData"), inputs, outputs);
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 2, 3, 4, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TransDataInfershapeTest, infer_4d_fractal_nz_shape)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {{{1, 1, 16, 16}, {1, 1, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},
    };
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},
    };
    gert::InfershapeContextPara para(std::string("TransData"), inputs, outputs);
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 16, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TransDataInfershapeTest, infer_unknown_rank)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    gert::InfershapeContextPara para(std::string("TransData"), inputs, outputs);
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TransDataInfershapeTest, infer_scalar_shape)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    gert::InfershapeContextPara para(std::string("TransData"), inputs, outputs);
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
