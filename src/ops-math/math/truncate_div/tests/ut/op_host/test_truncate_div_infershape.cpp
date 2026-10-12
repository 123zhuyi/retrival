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
 * \file test_truncate_div_infershape.cpp
 * \brief TruncateDiv infershape test
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;

class TruncateDivInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TruncateDivInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TruncateDivInfershape TearDown" << std::endl; }
};

TEST_F(TruncateDivInfershape, truncate_div_infershape_test_0)
{
    gert::InfershapeContextPara infershapeContextPara("TruncateDiv",
                                                      {
                                                          {{{2, 2, 1}, {2, 2, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 2, 3}, {2, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TruncateDivInfershape, truncate_div_infershape_test_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TruncateDiv",
        {
            {{{3, 4, 5, 6, -1}, {3, 4, 5, 6, -1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 4, 5, 6, 1}, {3, 4, 5, 6, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 5, 6, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TruncateDivInfershape, truncate_div_infershape_test_2)
{
    gert::InfershapeContextPara infershapeContextPara("TruncateDiv",
                                                      {
                                                          {{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TruncateDivInfershape, truncate_div_infershape_test_3)
{
    gert::InfershapeContextPara infershapeContextPara("TruncateDiv",
                                                      {
                                                          {{{8, 16, 32}, {8, 16, 32}}, ge::DT_INT8, ge::FORMAT_ND},
                                                          {{{8, 16, 32}, {8, 16, 32}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 16, 32},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TruncateDivInfershape, truncate_div_infershape_test_4)
{
    gert::InfershapeContextPara infershapeContextPara("TruncateDiv",
                                                      {
                                                          {{{1, 128, 256}, {1, 128, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                          {{{1, 128, 256}, {1, 128, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 128, 256},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(TruncateDivInfershape, truncate_div_infershape_test_5)
{
    gert::InfershapeContextPara infershapeContextPara("TruncateDiv",
                                                      {
                                                          {{{1024}, {1024}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{1024}, {1024}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1024},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
