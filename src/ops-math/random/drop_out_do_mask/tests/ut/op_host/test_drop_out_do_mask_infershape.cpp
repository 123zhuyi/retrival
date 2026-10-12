/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sim_random_uniform_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class DropOutDoMaskInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DropOutDoMaskInferShapeTest Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DropOutDoMaskInferShapeTest Proto Test TearDown" << std::endl;
    }
};

TEST_F(DropOutDoMaskInferShapeTest, drop_out_do_mask_test_case_1)
{
    gert::StorageShape x = {{4097, 1}, {4097, 1}};
    gert::StorageShape mask = {{528}, {528}};
    gert::StorageShape keep_prob = {{1}, {1}};
    gert::StorageShape y = {{}, {}};
    gert::InfershapeContextPara infershapeContextPara(
        "DropOutDoMask",
        {{x, ge::DT_FLOAT, ge::FORMAT_ND},
         {mask, ge::DT_UINT8, ge::FORMAT_ND},
         {keep_prob, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{y, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{4097, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
