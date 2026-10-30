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
 * \file test_complex_abs_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/complex_abs_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class ComplexAbsTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ComplexAbsTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ComplexAbsTilingTest TearDown" << std::endl; }
};

TEST_F(ComplexAbsTilingTest, test_tiling_complex64_001)
{
    optiling::ComplexAbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ComplexAbs",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "8192 35184372088840 1024 8 1 1 1024 1024 8192 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ComplexAbsTilingTest, test_tiling_complex32_002)
{
    optiling::ComplexAbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ComplexAbs",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "8192 70368744177668 2048 4 1 1 2048 2048 16384 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ComplexAbsTilingTest, test_tiling_failed_complex64_output_fp16_003)
{
    optiling::ComplexAbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ComplexAbs",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ComplexAbsTilingTest, test_tiling_failed_complex32_output_fp32_004)
{
    optiling::ComplexAbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ComplexAbs",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ComplexAbsTilingTest, test_tiling_failed_empty_tensor_005)
{
    optiling::ComplexAbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ComplexAbs",
                                              {
                                                  {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 0, 2, 64}, {1, 0, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ComplexAbsTilingTest, test_tiling_failed_unsupported_dtype_006)
{
    optiling::ComplexAbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ComplexAbs",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ComplexAbsTilingTest, test_tiling_complex64_small_shape_007)
{
    optiling::ComplexAbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ComplexAbs",
                                              {
                                                  {{{1, 16}, {1, 16}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 16}, {1, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "16 35184372088833 512 1 1 1 512 16 8192 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ComplexAbsTilingTest, test_tiling_complex32_small_shape_008)
{
    optiling::ComplexAbsCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("ComplexAbs",
                                              {
                                                  {{{1, 16}, {1, 16}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 16}, {1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "16 70368744177665 512 1 1 1 512 16 16384 1 ";
    std::vector<size_t> expectWorkspaces = {8192};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
