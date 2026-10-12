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
 * \file test_cdist_grad_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class CdistGradInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CdistGradInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CdistGradInfershape TearDown" << std::endl; }
};

// CdistGrad has 4 inputs (grad, x1, x2, cdist) and reduces along axis -2.
// Input grad shape: [..., P, R, M], output shape: [..., P, M].
// grad {2, 3, 4, 5} (dimNum=4) -> output {2, 3, 5}
TEST_F(CdistGradInfershape, cdist_grad_infershape_float_test)
{
    gert::InfershapeContextPara infershapeContextPara("CdistGrad",
                                                      {
                                                          {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// grad {4, 3, 2} (dimNum=3) -> output {4, 2}
TEST_F(CdistGradInfershape, cdist_grad_infershape_float16_test)
{
    gert::InfershapeContextPara infershapeContextPara("CdistGrad",
                                                      {
                                                          {{{4, 3, 2}, {4, 3, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{4, 2}, {4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{4, 2}, {4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{4, 3, 2}, {4, 3, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// grad {2, 4, 5, 6} (dimNum=4) -> output {2, 4, 6}
TEST_F(CdistGradInfershape, cdist_grad_infershape_bf16_test)
{
    gert::InfershapeContextPara infershapeContextPara("CdistGrad",
                                                      {
                                                          {{{2, 4, 5, 6}, {2, 4, 5, 6}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{2, 4, 6}, {2, 4, 6}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{2, 4, 6}, {2, 4, 6}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{2, 4, 5, 6}, {2, 4, 5, 6}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 6},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// grad {3, 5} (dimNum=2, minimum supported) -> output {5}
TEST_F(CdistGradInfershape, cdist_grad_infershape_int32_test)
{
    gert::InfershapeContextPara infershapeContextPara("CdistGrad",
                                                      {
                                                          {{{3, 5}, {3, 5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// grad {2, 3, 4, 5, 6} (dimNum=5) -> output {2, 3, 4, 6}
TEST_F(CdistGradInfershape, cdist_grad_infershape_5d_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "CdistGrad",
        {
            {{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3, 4, 6}, {2, 3, 4, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3, 4, 6}, {2, 3, 4, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 6},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// grad {2, 0, 4, 5} with empty dim -> output {2, 0, 5}
TEST_F(CdistGradInfershape, cdist_grad_infershape_empty_tensor_test)
{
    gert::InfershapeContextPara infershapeContextPara("CdistGrad",
                                                      {
                                                          {{{2, 0, 4, 5}, {2, 0, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 0, 5}, {2, 0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 0, 5}, {2, 0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 0, 4, 5}, {2, 0, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 0, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// grad {5} (dimNum=1, less than 2) -> expect GRAPH_FAILED
TEST_F(CdistGradInfershape, cdist_grad_infershape_1d_failed_test)
{
    gert::InfershapeContextPara infershapeContextPara("CdistGrad",
                                                      {
                                                          {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
