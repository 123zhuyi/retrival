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

class EltwiseInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EltwiseInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EltwiseInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(EltwiseInfershapeTest, eltwise_infershape_2d)
{
    gert::StorageShape shape = {{32, 32}, {32, 32}};

    gert::InfershapeContextPara infershapeContextPara(
        "Eltwise",
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(EltwiseInfershapeTest, eltwise_infershape_1d)
{
    gert::StorageShape shape = {{128}, {128}};

    gert::InfershapeContextPara infershapeContextPara(
        "Eltwise",
        {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(EltwiseInfershapeTest, eltwise_infershape_3d)
{
    gert::StorageShape shape = {{4, 8, 16}, {4, 8, 16}};

    gert::InfershapeContextPara infershapeContextPara(
        "Eltwise",
        {{shape, ge::DT_BF16, ge::FORMAT_ND}, {shape, ge::DT_BF16, ge::FORMAT_ND}},
        {{shape, ge::DT_BF16, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(EltwiseInfershapeTest, eltwise_infershape_4d)
{
    gert::StorageShape shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};

    gert::InfershapeContextPara infershapeContextPara(
        "Eltwise",
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(EltwiseInfershapeTest, eltwise_infershape_single_input)
{
    gert::StorageShape shape = {{64, 64}, {64, 64}};

    gert::InfershapeContextPara infershapeContextPara(
        "Eltwise",
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{64, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
