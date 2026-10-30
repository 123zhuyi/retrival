/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_kernel/arch35/matrix_set_diag_tilingdata.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class MatrixSetDiagTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatrixSetDiagTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatrixSetDiagTilingTest TearDown" << std::endl;
    }
};

TEST_F(MatrixSetDiagTilingTest, test_tiling_int16)
{
    MatrixSetDiagCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiag",
        {
            {{{3, 3}, {3, 3}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            {{{3, 3}, {3, 3}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 2;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagTilingTest, test_tiling_uint8)
{
    MatrixSetDiagCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiag",
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{3, 4}, {3, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 2;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagTilingTest, test_tiling_float)
{
    MatrixSetDiagCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiag",
        {
            {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 2;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagTilingTest, test_tiling_bool)
{
    MatrixSetDiagCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiag",
        {
            {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{2, 2, 2}, {2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 2;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MatrixSetDiagTilingTest, test_tiling_failed_should_have_same_type)
{
    MatrixSetDiagCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiag",
        {
            {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagTilingTest, test_tiling_failed_diag_dim_should_less_input_dim)
{
    MatrixSetDiagCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiag",
        {
            {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{1, 2, 2}, {1, 2, 2}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagTilingTest, test_tiling_failed_input_dim_from_2)
{
    MatrixSetDiagCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiag",
        {
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagTilingTest, test_tiling_failed_diag_len_invalid)
{
    MatrixSetDiagCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiag",
        {
            {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(MatrixSetDiagTilingTest, test_tiling_failed_batch_dim_invalid)
{
    MatrixSetDiagCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "MatrixSetDiag",
        {
            {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 3}, {3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}