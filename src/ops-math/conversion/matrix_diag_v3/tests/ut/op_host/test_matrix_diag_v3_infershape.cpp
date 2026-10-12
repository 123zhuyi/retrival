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
 * \file test_matrix_diag_v3_infershape.cpp
 * \brief InferShape cases for MatrixDiagV3.
 *
 * Inputs are x (the diagonal values), k, num_rows, num_cols and padding_value; k, num_rows and num_cols are data
 * dependencies, so the faker feeds them as const tensors. A single diagonal grows the rank by one, a band of
 * diagonals keeps it. When num_rows / num_cols are absent they are derived from the longest diagonal length.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class MatrixDiagV3Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MatrixDiagV3Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MatrixDiagV3Infershape TearDown" << std::endl; }
};

namespace {
// Pass a null value pointer for an input that is not a compile-time constant.
gert::InfershapeContextPara MakePara(const gert::StorageShape& xShape, ge::DataType xDtype,
                                     const gert::StorageShape& kShape, void* kValue, void* numRowsValue,
                                     void* numColsValue, const gert::StorageShape& numRowsShape = {{}, {}},
                                     const gert::StorageShape& numColsShape = {{}, {}},
                                     const gert::StorageShape& paddingValueShape = {{}, {}})
{
    return gert::InfershapeContextPara(
        "MatrixDiagV3",
        {
            {xShape, xDtype, ge::FORMAT_ND},
            {kShape, ge::DT_INT32, ge::FORMAT_ND, kValue != nullptr, kValue},
            {numRowsShape, ge::DT_INT32, ge::FORMAT_ND, numRowsValue != nullptr, numRowsValue},
            {numColsShape, ge::DT_INT32, ge::FORMAT_ND, numColsValue != nullptr, numColsValue},
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

// Single main diagonal with explicit sizes: 4 diagonal values become a 4x4 matrix, so the rank grows from 1 to 2.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_main_diagonal_explicit_size)
{
    std::vector<int32_t> kValue = {0};
    std::vector<int32_t> numRows = {4};
    std::vector<int32_t> numCols = {4};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), numRows.data(), numCols.data());
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// num_rows and num_cols are absent, so both are derived as max(min_num_rows, min_num_cols) = max(4, 4) = 4.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_main_diagonal_inferred_size)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), nullptr, nullptr);
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// A subdiagonal needs one extra row: min_num_rows = 3 - min(-1, 0) = 4, min_num_cols = 3 + max(-1, 0) = 3,
// so both sizes become max(4, 3) = 4.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_subdiagonal_inferred_size)
{
    std::vector<int32_t> kValue = {-1};
    auto para = MakePara({{3}, {3}}, ge::DT_FLOAT, {{1}, {1}}, kValue.data(), nullptr, nullptr);
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// A band of 3 diagonals: the num_diags dim of x is consumed by num_rows, so the rank stays 2.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_diagonal_band)
{
    std::vector<int32_t> kValue = {-1, 1};
    auto para = MakePara({{3, 4}, {3, 4}}, ge::DT_FLOAT, {{2}, {2}}, kValue.data(), nullptr, nullptr);
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Batch dims are kept ahead of the band: x is [2, 3, 4] and num_rows is given as 5.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_diagonal_band_with_batch)
{
    std::vector<int32_t> kValue = {-1, 1};
    std::vector<int32_t> numRows = {5};
    std::vector<int32_t> numCols = {4};
    auto para = MakePara({{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, {{2}, {2}}, kValue.data(), numRows.data(),
                         numCols.data());
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 5, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// k is not a compile-time constant, so the output falls back to a 1-D unknown shape.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_k_not_const)
{
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{1}, {1}}, nullptr, nullptr, nullptr);
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// x has an unknown rank (-2), so the output falls back to a 1-D unknown shape.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_x_unknown_rank)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{-2}, {-2}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), nullptr, nullptr);
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// k has an unknown dim, so its own shape is not fully defined and the output falls back.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_k_shape_not_fully_defined)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{-1}, {-1}}, kValue.data(), nullptr, nullptr);
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// num_rows must not be smaller than the minimum implied by the diagonal length.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_num_rows_too_small)
{
    std::vector<int32_t> kValue = {0};
    std::vector<int32_t> numRows = {2};
    std::vector<int32_t> numCols = {4};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), numRows.data(), numCols.data());
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// num_cols must not be smaller than the minimum implied by the diagonal length.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_num_cols_too_small)
{
    std::vector<int32_t> kValue = {0};
    std::vector<int32_t> numRows = {4};
    std::vector<int32_t> numCols = {2};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), numRows.data(), numCols.data());
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// Both sizes exceed their minimums at the same time, which the source side rejects as inconsistent.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_both_sizes_exceed_minimum)
{
    std::vector<int32_t> kValue = {0};
    std::vector<int32_t> numRows = {5};
    std::vector<int32_t> numCols = {5};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), numRows.data(), numCols.data());
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// The num_diags dim of x must match the band implied by k: x holds 2 diagonals but k asks for 3.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_num_diags_mismatch)
{
    std::vector<int32_t> kValue = {-1, 1};
    auto para = MakePara({{2, 4}, {2, 4}}, ge::DT_FLOAT, {{2}, {2}}, kValue.data(), nullptr, nullptr);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// A band of diagonals needs x to have rank at least 2.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_band_x_rank_too_small)
{
    std::vector<int32_t> kValue = {-1, 1};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{2}, {2}}, kValue.data(), nullptr, nullptr);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// The lower diagonal index must not exceed the upper one.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_lower_greater_than_upper)
{
    std::vector<int32_t> kValue = {2, 1};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{2}, {2}}, kValue.data(), nullptr, nullptr);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// k must hold at most two elements.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_k_too_many_elements)
{
    std::vector<int32_t> kValue = {-1, 0, 1};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{3}, {3}}, kValue.data(), nullptr, nullptr);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// x must have rank at least 1.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_x_scalar)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{}, {}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), nullptr, nullptr);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// num_rows must be a scalar.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_num_rows_not_scalar)
{
    std::vector<int32_t> kValue = {0};
    std::vector<int32_t> numRows = {4};
    std::vector<int32_t> numCols = {4};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), numRows.data(), numCols.data(), {{1}, {1}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// num_cols must be a scalar.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_num_cols_not_scalar)
{
    std::vector<int32_t> kValue = {0};
    std::vector<int32_t> numRows = {4};
    std::vector<int32_t> numCols = {4};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), numRows.data(), numCols.data(), {{}, {}},
                         {{1}, {1}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// padding_value must be a scalar.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_padding_value_not_scalar)
{
    std::vector<int32_t> kValue = {0};
    auto para = MakePara({{4}, {4}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), nullptr, nullptr, {{}, {}}, {{}, {}},
                         {{1}, {1}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// x has a dynamic dim and k selects a band: num_diags still matches, but the diagonal length is unknown, so both
// derived matrix dims stay unknown.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_dynamic_diag_len_band)
{
    std::vector<int32_t> kValue = {-1, 1};
    auto para = MakePara({{3, -1}, {3, -1}}, ge::DT_FLOAT, {{2}, {2}}, kValue.data(), nullptr, nullptr);
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// x has a dynamic dim but num_rows / num_cols are given, so the output is fully resolved from the const values.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_dynamic_diag_len_explicit_size)
{
    std::vector<int32_t> kValue = {0};
    std::vector<int32_t> numRows = {4};
    std::vector<int32_t> numCols = {4};
    auto para = MakePara({{-1}, {-1}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), numRows.data(), numCols.data());
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Known source-side limitation, reproduced here so the behaviour is visible and any change is caught:
// when the diagonal length is dynamic and neither num_rows nor num_cols is const, the min sizes are derived from
// maxDiagLen == -1 and the output degenerates to a static zero-sized shape instead of staying unknown.
// canndev RT1 (matrix_calculation_ops.cc GetRowsAndCols) behaves identically; tracked for a coordinated fix.
TEST_F(MatrixDiagV3Infershape, matrix_diag_v3_infershape_dynamic_diag_len_degenerates_to_zero)
{
    std::vector<int32_t> kValue = {1};
    auto para = MakePara({{-1}, {-1}}, ge::DT_FLOAT, {{}, {}}, kValue.data(), nullptr, nullptr);
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 0}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
