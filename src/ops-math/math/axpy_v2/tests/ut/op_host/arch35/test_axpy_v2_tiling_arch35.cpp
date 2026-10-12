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
 * \file test_axpy_v2_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/axpy_v2_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class AxpyV2Tiling : public testing::Test {
   protected:
    static void SetUpTestCase()
    {
        std::cout << "AxpyV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AxpyV2Tiling TearDown" << std::endl;
    }
};

TEST_F(AxpyV2Tiling, axpy_v2_test_float16)
{
    optiling::AxpyV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AxpyV2",
                                              {
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 1407392063422720 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AxpyV2Tiling, axpy_v2_test_float32)
{
    optiling::AxpyV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AxpyV2",
                                              {
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 1407409243291776 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AxpyV2Tiling, axpy_v2_test_int32)
{
    optiling::AxpyV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AxpyV2",
                                              {
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 1407409243291776 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AxpyV2Tiling, axpy_v2_test_int64)
{
    optiling::AxpyV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AxpyV2",
                                              {
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT64, ge::FORMAT_ND},
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 1407443603030080 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AxpyV2Tiling, axpy_v2_test_int8)
{
    optiling::AxpyV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AxpyV2",
                                              {
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 1407383473488384 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AxpyV2Tiling, axpy_v2_test_uint8)
{
    optiling::AxpyV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AxpyV2",
                                              {
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_UINT8, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 1407383473488384 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AxpyV2Tiling, axpy_v2_test_bool)
{
    optiling::AxpyV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AxpyV2",
                                              {
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 1407383473488384 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AxpyV2Tiling, axpy_v2_test_dtype_not_same)
{
    optiling::AxpyV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AxpyV2",
                                              {
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AxpyV2Tiling, axpy_v2_test_not_support_dtype)
{
    optiling::AxpyV2CompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("AxpyV2",
                                              {
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT16, ge::FORMAT_ND},
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT16, ge::FORMAT_ND},
                                                {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}