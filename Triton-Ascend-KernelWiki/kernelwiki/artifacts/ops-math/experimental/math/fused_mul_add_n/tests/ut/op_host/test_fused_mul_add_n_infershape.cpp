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
 * \file test_fused_mul_add_n_infershape.cpp
 * \brief A2 op_host infershape UT（fp32 核心路径）。
 *        FusedMulAddN: y = x1 * x3[0] + x2（逐元素），输出 shape = x1.shape（InferShape4Elewise）。
 *        本 UT 收敛到 fp32 核心路径：
 *          - 动态 shape（-1,-1）-> 输出 -1,-1（透传 x1 占位）
 *          - 标量（0 维）-> 输出标量
 *          - 1D -> 输出同 1D
 *        infershape 仅推导输出 shape（dtype 由图模式 InferDataType 处理），故此处只校验 shape。
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class FusedMulAddNInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedMulAddNInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedMulAddNInfershape TearDown" << std::endl;
    }
};

// ---- 用例 1：fp32 动态 shape -> 输出 shape = x1.shape（-1,-1 透传）----
TEST_F(FusedMulAddNInfershape, fused_mul_add_n_case)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedMulAddN",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ---- 用例 2：fp32 标量（0 维）-> 输出标量 ----
TEST_F(FusedMulAddNInfershape, VerifyFusedMulAddN_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedMulAddN",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ---- 用例 3：fp32 1D -> 输出 shape = x1.shape（[1]）----
TEST_F(FusedMulAddNInfershape, VerifyFusedMulAddN_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "FusedMulAddN",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
