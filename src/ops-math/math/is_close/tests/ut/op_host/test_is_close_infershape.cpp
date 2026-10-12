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
 * \file test_is_close_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class IsCloseInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IsCloseInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "IsCloseInfershape TearDown" << std::endl; }
};

// IsClose has 2 inputs and 1 output. Output is the broadcast of the two inputs.
// With same shapes, output shape equals input shape.
TEST_F(IsCloseInfershape, is_close_infershape_float_test)
{
    gert::InfershapeContextPara infershapeContextPara("IsClose",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IsCloseInfershape, is_close_infershape_float16_test)
{
    gert::InfershapeContextPara infershapeContextPara("IsClose",
                                                      {
                                                          {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IsCloseInfershape, is_close_infershape_int32_test)
{
    gert::InfershapeContextPara infershapeContextPara("IsClose",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IsCloseInfershape, is_close_infershape_bf16_test)
{
    gert::InfershapeContextPara infershapeContextPara("IsClose",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Broadcast case: {2, 3, 4} and {3, 4} -> output {2, 3, 4}
TEST_F(IsCloseInfershape, is_close_infershape_broadcast_test)
{
    gert::InfershapeContextPara infershapeContextPara("IsClose",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Broadcast case with scalar: {2, 3, 4} and {1} -> output {2, 3, 4}
TEST_F(IsCloseInfershape, is_close_infershape_broadcast_scalar_test)
{
    gert::InfershapeContextPara infershapeContextPara("IsClose",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IsCloseInfershape, is_close_infershape_1d_tensor_test)
{
    gert::InfershapeContextPara infershapeContextPara("IsClose",
                                                      {
                                                          {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IsCloseInfershape, is_close_infershape_empty_tensor_test)
{
    gert::InfershapeContextPara infershapeContextPara("IsClose",
                                                      {
                                                          {{{0, 3, 4}, {0, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{0, 3, 4}, {0, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {0, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
