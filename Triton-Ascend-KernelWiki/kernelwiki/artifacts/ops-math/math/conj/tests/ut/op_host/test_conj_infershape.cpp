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

class ConjInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ConjInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ConjInfershape TearDown" << std::endl; }
};

TEST_F(ConjInfershape, conj_infershape_complex128)
{
    gert::InfershapeContextPara infershapeContextPara("Conj",
                                                      {
                                                          {{{2, 3}, {2, 3}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConjInfershape, conj_infershape_complex64)
{
    gert::InfershapeContextPara infershapeContextPara("Conj",
                                                      {
                                                          {{{8}, {8}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConjInfershape, conj_infershape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara("Conj",
                                                      {
                                                          {{{-1, 4}, {-1, 4}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
