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
 * \file test_matrix_diag_part_v3_infershape.cpp
 * \brief InferShape cases for MatrixDiagPartV3.
 *
 * Inputs are x, k and padding_value; k is a data dependency, so the faker feeds it as a const tensor.
 * The output keeps the batch dims of x, inserts a num_diags dim only when k selects a band, and ends with the
 * longest diagonal length min(num_rows + min(upper, 0), num_cols - max(lower, 0)).
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class MatrixDiagPartV3Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MatrixDiagPartV3Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MatrixDiagPartV3Infershape TearDown" << std::endl; }
};

namespace {
// k is a scalar here; pass an empty StorageShape to build a rank 0 tensor.
gert::InfershapeContextPara MakePara(const gert::StorageShape& xShape, ge::DataType xDtype,
                                     const gert::StorageShape& kShape, void* kValue,
                                     const gert::StorageShape& paddingValueShape)
{
    return gert::InfershapeContextPara("MatrixDiagPartV3",
                                       {
                                           {xShape, xDtype, ge::FORMAT_ND},
                                           {kShape, ge::DT_INT32, ge::FORMAT_ND, kValue != nullptr, kValue},
                                           {paddingValueShape, xDtype, ge::FORMAT_ND},
                                       },
                                       {
                                           {{{}, {}}, xDtype, ge::FORMAT_ND},
                                       },
                                       {
                                           {"align", Ops::Math::AnyValue::CreateFrom<std::string>("RIGHT_LEFT")},
                                       });
}
} // namespace

// Single diagonal on a plain matrix: k == 0 keeps the main diagonal, so y is [min(4, 4)] = [4].
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_main_diagonal)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{4, 4}, {4, 4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), {{}, {}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Batch dims are kept. k == 1 is the first superdiagonal: max_diag_len = min(4 + 0, 4 - 1) = 3, so y is [2, 3].
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_superdiagonal_with_batch)
{
    std::vector<int32_t> kValue = {1};
    auto para = MakePara({{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, {{1}, {1}}, kValue.data(), {{}, {}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// k == {-1, 1} selects a band of 3 diagonals, which adds a num_diags dim: y is [3, 4].
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_diagonal_band)
{
    std::vector<int32_t> kValue = {-1, 1};
    auto para = MakePara({{4, 4}, {4, 4}}, ge::DT_FLOAT, {{2}, {2}}, kValue.data(), {{}, {}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// A non square matrix: k == 0 gives max_diag_len = min(5 + 0, 3 - 0) = 3.
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_non_square)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{5, 3}, {5, 3}}, ge::DT_INT32, {{}, {}}, kValue.data(), {{}, {}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// k is not a compile-time constant, so the concrete band is unknown and the output falls back to a 1-D unknown shape.
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_k_not_const)
{
    auto para = MakePara({{4, 4}, {4, 4}}, ge::DT_FLOAT, {{1}, {1}}, nullptr, {{}, {}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// x has an unknown rank (-2), so the output falls back to a 1-D unknown shape.
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_x_unknown_rank)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{-2}, {-2}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), {{}, {}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// k has an unknown dim, so its own shape is not fully defined and the output falls back.
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_k_shape_not_fully_defined)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{4, 4}, {4, 4}}, ge::DT_FLOAT, {{-1}, {-1}}, kValue.data(), {{}, {}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// The matrix dims of x are dynamic, so max_diag_len stays unknown while the batch dim is still resolved.
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_dynamic_matrix_dims)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{2, -1, 4}, {2, -1, 4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), {{}, {}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// The lower diagonal index must not exceed the upper one.
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_lower_greater_than_upper)
{
    std::vector<int32_t> kValue = {2, 1};
    auto para = MakePara({{4, 4}, {4, 4}}, ge::DT_FLOAT, {{2}, {2}}, kValue.data(), {{}, {}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// k must hold at most two elements.
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_k_too_many_elements)
{
    std::vector<int32_t> kValue = {-1, 0, 1};
    auto para = MakePara({{4, 4}, {4, 4}}, ge::DT_FLOAT, {{3}, {3}}, kValue.data(), {{}, {}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// x must have rank at least 2.
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_x_rank_too_small)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), {{}, {}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// k must have rank at most 1.
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_k_rank_too_large)
{
    std::vector<int32_t> kValue = {0, 0};
    auto para = MakePara({{4, 4}, {4, 4}}, ge::DT_FLOAT, {{1, 2}, {1, 2}}, kValue.data(), {{}, {}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// padding_value must be a scalar.
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_padding_value_not_scalar)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{4, 4}, {4, 4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), {{1}, {1}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// The diagonal index must be 0 or fall in the open range (-num_rows, num_cols); 4 is out of range for a 4x4 matrix.
TEST_F(MatrixDiagPartV3Infershape, matrix_diag_part_v3_infershape_diag_index_out_of_range)
{
    std::vector<int32_t> kValue = {4};
    auto para = MakePara({{4, 4}, {4, 4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), {{}, {}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
