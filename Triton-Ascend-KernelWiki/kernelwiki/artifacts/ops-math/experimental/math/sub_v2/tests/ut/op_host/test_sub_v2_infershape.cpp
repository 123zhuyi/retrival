/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua<@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SubV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SubV2Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SubV2Infershape TearDown" << std::endl; }
};

TEST_F(SubV2Infershape, sub_v2_infershape_test1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SubV2", {{{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SubV2Infershape, sub_v2_infershape_test2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SubV2", {{{{5, -1}, {5, -1}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{5, -1}, {5, -1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}