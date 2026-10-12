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
 * \file test_cdist_infershape.cpp
 * \brief Cdist InferShape UT
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class CdistInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CdistInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CdistInfershape TearDown" << std::endl; }
};

TEST_F(CdistInfershape, infershape_2d_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Cdist", {{{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{6, 8}, {6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(CdistInfershape, infershape_3d_fp32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Cdist",
        {{{{2, 4, 8}, {2, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{2, 6, 8}, {2, 6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(CdistInfershape, infershape_4d_bf16)
{
    gert::InfershapeContextPara infershapeContextPara("Cdist",
                                                      {{{{3, 2, 4, 8}, {3, 2, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                       {{{3, 2, 6, 8}, {3, 2, 6, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 2, 4, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(CdistInfershape, infershape_1d_invalid)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Cdist", {{{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
