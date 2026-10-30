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

using namespace ge;
class ConjugateTransposeInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ConjugateTransposeInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ConjugateTransposeInferShapeTest TearDown" << std::endl; }
};

TEST_F(ConjugateTransposeInferShapeTest, conjugate_transpose_int32_perm)
{
    int32_t perm_value[3] = {0, 2, 1};
    gert::InfershapeContextPara::TensorDescription x({{3, 2, 4}, {3, 2, 4}}, ge::DT_COMPLEX128, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_COMPLEX128, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("ConjugateTranspose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ConjugateTransposeInferShapeTest, conjugate_transpose_int64_perm)
{
    int64_t perm_value[2] = {1, 0};
    gert::InfershapeContextPara::TensorDescription x({{2, 3}, {2, 3}}, ge::DT_COMPLEX64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_COMPLEX64, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("ConjugateTranspose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// negative perm is not supported by ConjugateTranspose (perm must be in [0, rank-1],
// aligned with the AICPU kernel and the canndev rt1 contract) -> failure
TEST_F(ConjugateTransposeInferShapeTest, conjugate_transpose_negative_perm_rejected)
{
    int64_t perm_value[3] = {-1, -2, 0};
    gert::InfershapeContextPara::TensorDescription x({{10, 20, 30}, {10, 20, 30}}, ge::DT_DOUBLE, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_DOUBLE, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("ConjugateTranspose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// perm size mismatch with input dims -> failure
TEST_F(ConjugateTransposeInferShapeTest, conjugate_transpose_perm_size_mismatch)
{
    int64_t perm_value[3] = {0, 1, 2};
    gert::InfershapeContextPara::TensorDescription x({{10, 20, 30, 40}, {10, 20, 30, 40}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("ConjugateTranspose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

// perm value out of valid range (perm[i] >= dimSize) -> failure
TEST_F(ConjugateTransposeInferShapeTest, conjugate_transpose_perm_out_of_range)
{
    int32_t perm_value[3] = {0, 3, 1};
    gert::InfershapeContextPara::TensorDescription x({{3, 2, 4}, {3, 2, 4}}, ge::DT_COMPLEX128, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, &perm_value);
    gert::InfershapeContextPara::TensorDescription out({{1}, {1}}, ge::DT_COMPLEX128, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("ConjugateTranspose", {x, perm}, {out});
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}
