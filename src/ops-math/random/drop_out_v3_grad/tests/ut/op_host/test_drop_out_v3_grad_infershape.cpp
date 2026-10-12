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
 * \file test_drop_out_v3_grad_infershape.cpp
 * \brief DropOutV3Grad infershape UT —— grad_x shape 应与 grad_y 一致。
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class DropOutV3GradInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DropOutV3GradInferShapeTest Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DropOutV3GradInferShapeTest Proto Test TearDown" << std::endl; }
};

// case 1: 二维 grad_y，grad_x 继承其 shape
TEST_F(DropOutV3GradInferShapeTest, drop_out_v3_grad_infershape_case_1)
{
    gert::StorageShape grad_y = {{4097, 1}, {4097, 1}};
    gert::StorageShape mask = {{528}, {528}};
    gert::StorageShape scale = {{1}, {1}};
    gert::StorageShape grad_x = {{}, {}};
    gert::InfershapeContextPara infershapeContextPara("DropOutV3Grad",
                                                      {{grad_y, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {mask, ge::DT_UINT8, ge::FORMAT_ND},
                                                       {scale, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{grad_x, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{4097, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// case 2: 多维 grad_y（bf16 + uint1），grad_x 继承 shape
TEST_F(DropOutV3GradInferShapeTest, drop_out_v3_grad_infershape_case_2)
{
    gert::StorageShape grad_y = {{2, 3, 16}, {2, 3, 16}};
    gert::StorageShape mask = {{2, 3, 16}, {2, 3, 16}};
    gert::StorageShape scale = {{1}, {1}};
    gert::StorageShape grad_x = {{}, {}};
    gert::InfershapeContextPara infershapeContextPara("DropOutV3Grad",
                                                      {{grad_y, ge::DT_BF16, ge::FORMAT_ND},
                                                       {mask, ge::DT_UINT1, ge::FORMAT_ND},
                                                       {scale, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{grad_x, ge::DT_BF16, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
