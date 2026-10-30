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
 * \file test_fused_mul_add_nl2loss_infershape.cpp
 * \brief FusedMulAddNL2loss infershape UT（infershape 无芯片依赖，放 op_host 根目录）
 *        y1 shape = x1 shape；y2 = 0 维标量（空 dims）
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;

class FusedMulAddNL2lossInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FusedMulAddNL2lossInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FusedMulAddNL2lossInfershape TearDown" << std::endl; }
};

// 2D fp32：y1 同 x1 shape，y2 为 0 维标量
TEST_F(FusedMulAddNL2lossInfershape, shape_2d_fp32)
{
    gert::InfershapeContextPara para("FusedMulAddNL2loss",
                                     {
                                         {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 16}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 1D fp16
TEST_F(FusedMulAddNL2lossInfershape, shape_1d_fp16)
{
    gert::InfershapeContextPara para("FusedMulAddNL2loss",
                                     {
                                         {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 单元素
TEST_F(FusedMulAddNL2lossInfershape, shape_single_element)
{
    gert::InfershapeContextPara para("FusedMulAddNL2loss",
                                     {
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 3D
TEST_F(FusedMulAddNL2lossInfershape, shape_3d_fp32)
{
    gert::InfershapeContextPara para("FusedMulAddNL2loss",
                                     {
                                         {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 4, 4}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 8D 高秩
TEST_F(FusedMulAddNL2lossInfershape, shape_8d_fp16)
{
    gert::InfershapeContextPara para(
        "FusedMulAddNL2loss",
        {
            {{{2, 3, 2, 3, 2, 3, 2, 3}, {2, 3, 2, 3, 2, 3, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 3, 2, 3, 2, 3, 2, 3}, {2, 3, 2, 3, 2, 3, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 2, 3, 2, 3, 2, 3}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态 shape（-1）：y1 透传 -1，y2 仍为 0 维标量
TEST_F(FusedMulAddNL2lossInfershape, dynamic_shape)
{
    gert::InfershapeContextPara para("FusedMulAddNL2loss",
                                     {
                                         {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态 rank（-2）：y1 透传 -2，y2 仍为 0 维标量
TEST_F(FusedMulAddNL2lossInfershape, dynamic_rank)
{
    gert::InfershapeContextPara para("FusedMulAddNL2loss",
                                     {
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
