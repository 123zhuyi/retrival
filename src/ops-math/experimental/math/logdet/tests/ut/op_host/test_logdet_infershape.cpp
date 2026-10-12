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

class LogdetInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "LogdetInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LogdetInfershape TearDown" << std::endl;
    }
};

TEST_F(LogdetInfershape, single_matrix_3x3_output_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Logdet",
        {
            {{{3, 3}, {3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LogdetInfershape, single_matrix_2x2_output_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Logdet",
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LogdetInfershape, batch_matrix_4x3x3_output_batch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Logdet",
        {
            {{{4, 3, 3}, {4, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LogdetInfershape, batch_matrix_2x5x5_output_batch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Logdet",
        {
            {{{2, 5, 5}, {2, 5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LogdetInfershape, multi_batch_3x4x3x3_output_multi_batch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Logdet",
        {
            {{{3, 4, 3, 3}, {3, 4, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 0}, {0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LogdetInfershape, matrix_1x1_output_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Logdet",
        {
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LogdetInfershape, batch_1x1_matrices)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Logdet",
        {
            {{{6, 1, 1}, {6, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {6},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LogdetInfershape, dynamic_batch_shape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Logdet",
        {
            {{{-1, 3, 3}, {-1, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
