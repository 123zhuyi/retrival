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
 * \file test_mod_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ModInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ModInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ModInfershape TearDown" << std::endl;
    }
};

// Test: mod infershape with same shape
TEST_F(ModInfershape, mod_infershape_same_shape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mod",
        {
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: mod infershape with broadcast
TEST_F(ModInfershape, mod_infershape_broadcast)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mod",
        {
            {{{1, 3, 4}, {1, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: mod infershape with broadcast different dims
TEST_F(ModInfershape, mod_infershape_broadcast_diff_dims)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mod",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 3}, {1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: mod infershape with int32 dtype
TEST_F(ModInfershape, mod_infershape_int32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mod",
        {
            {{{10, 20}, {10, 20}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{10, 20}, {10, 20}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10, 20}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: mod infershape with float32 dtype
TEST_F(ModInfershape, mod_infershape_float32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mod",
        {
            {{{5, 6, 7}, {5, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5, 6, 7}, {5, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, 6, 7}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: mod infershape with float16 dtype
TEST_F(ModInfershape, mod_infershape_float16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mod",
        {
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: mod infershape with scalar divisor
TEST_F(ModInfershape, mod_infershape_scalar_divisor)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mod",
        {
            {{{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10, 20}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: mod infershape with 1D tensors
TEST_F(ModInfershape, mod_infershape_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mod",
        {
            {{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{100}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: mod infershape with double dtype
TEST_F(ModInfershape, mod_infershape_double)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mod",
        {
            {{{10, 20}, {10, 20}}, ge::DT_DOUBLE, ge::FORMAT_ND},
            {{{10, 20}, {10, 20}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10, 20}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: mod infershape with NHWC format
TEST_F(ModInfershape, mod_infershape_nhwc)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Mod",
        {
            {{{1, 16, 16, 3}, {1, 16, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{1, 16, 16, 3}, {1, 16, 16, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 16, 16, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
