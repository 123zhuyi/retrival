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
 * \file test_sinc_infershape.cpp
 * \brief Sinc算子形状推导测试
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class sinc : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "sinc Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "sinc Proto Test TearDown" << std::endl; }
};

// ========== 正常场景测试 ==========

TEST_F(sinc, sinc_infershape_same_test_fp16)
{
    gert::StorageShape shape = {{4, 3, 4}, {4, 3, 4}};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Sinc", {x}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(sinc, sinc_infershape_same_test_fp32)
{
    gert::StorageShape shape = {{4, 3, 4}, {4, 3, 4}};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Sinc", {x}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(sinc, sinc_infershape_same_test_bf16)
{
    gert::StorageShape shape = {{4, 3, 4}, {4, 3, 4}};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_BF16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_BF16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Sinc", {x}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(sinc, sinc_infershape_1dim_test)
{
    gert::StorageShape shape = {{16}, {16}};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Sinc", {x}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(sinc, sinc_infershape_2dim_test)
{
    gert::StorageShape shape = {{8, 16}, {8, 16}};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Sinc", {x}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(sinc, sinc_infershape_4dim_test)
{
    gert::StorageShape shape = {{2, 4, 8, 16}, {2, 4, 8, 16}};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Sinc", {x}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 8, 16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(sinc, sinc_infershape_8dim_test)
{
    gert::StorageShape shape = {{1, 2, 2, 2, 2, 2, 2, 2}, {1, 2, 2, 2, 2, 2, 2, 2}};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Sinc", {x}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 2, 2, 2, 2, 2, 2, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ========== 空Tensor测试 ==========

TEST_F(sinc, sinc_infershape_empty_test)
{
    gert::StorageShape shape = {{0}, {0}};
    gert::InfershapeContextPara::TensorDescription x(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(shape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Sinc", {x}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {0},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
