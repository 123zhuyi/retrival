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
 * \file test_complex_abs_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ComplexAbsInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ComplexAbsInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ComplexAbsInfershape TearDown" << std::endl; }
};

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex64_test)
{
    gert::InfershapeContextPara infershapeContextPara("ComplexAbs",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex32_test)
{
    gert::InfershapeContextPara infershapeContextPara("ComplexAbs",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex64_1d_test)
{
    gert::InfershapeContextPara infershapeContextPara("ComplexAbs",
                                                      {
                                                          {{{10}, {10}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex32_1d_test)
{
    gert::InfershapeContextPara infershapeContextPara("ComplexAbs",
                                                      {
                                                          {{{10}, {10}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex64_4d_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ComplexAbs",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex32_4d_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ComplexAbs",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex64_5d_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ComplexAbs",
        {
            {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex32_5d_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ComplexAbs",
        {
            {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex64_nchw_format_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ComplexAbs",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_COMPLEX64, ge::FORMAT_NCHW},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex32_nhwc_format_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ComplexAbs",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_COMPLEX32, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex64_empty_tensor_test)
{
    gert::InfershapeContextPara infershapeContextPara("ComplexAbs",
                                                      {
                                                          {{{0, 3, 4}, {0, 3, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {0, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex32_empty_tensor_test)
{
    gert::InfershapeContextPara infershapeContextPara("ComplexAbs",
                                                      {
                                                          {{{0, 3, 4}, {0, 3, 4}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {0, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex64_scalar_test)
{
    gert::InfershapeContextPara infershapeContextPara("ComplexAbs",
                                                      {
                                                          {{{1}, {1}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ComplexAbsInfershape, complex_abs_infershape_complex32_scalar_test)
{
    gert::InfershapeContextPara infershapeContextPara("ComplexAbs",
                                                      {
                                                          {{{1}, {1}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
