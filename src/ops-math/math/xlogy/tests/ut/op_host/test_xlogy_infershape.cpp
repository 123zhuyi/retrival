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
 * \file test_xlogy_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class XlogyInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "XlogyInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "XlogyInfershape TearDown" << std::endl; }
};

TEST_F(XlogyInfershape, xlogy_infershape_float_test)
{
    gert::InfershapeContextPara infershapeContextPara("Xlogy",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(XlogyInfershape, xlogy_infershape_float16_test)
{
    gert::InfershapeContextPara infershapeContextPara("Xlogy",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(XlogyInfershape, xlogy_infershape_bf16_test)
{
    gert::InfershapeContextPara infershapeContextPara("Xlogy",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(XlogyInfershape, xlogy_infershape_broadcast_test)
{
    gert::InfershapeContextPara infershapeContextPara("Xlogy",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1, 3, 4}, {1, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(XlogyInfershape, xlogy_infershape_1d_test)
{
    gert::InfershapeContextPara infershapeContextPara("Xlogy",
                                                      {
                                                          {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
