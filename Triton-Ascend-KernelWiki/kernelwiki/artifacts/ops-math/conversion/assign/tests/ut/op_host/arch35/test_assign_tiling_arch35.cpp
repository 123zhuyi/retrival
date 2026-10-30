/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/assign_tiling_arch35.h"

using namespace std;

class AssignTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AssignTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AssignTilingTest TearDown" << std::endl;
  }
};

TEST_F(AssignTilingTest, Assign_tiling_float)
{
    optiling::AssignCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Assign",
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "64 1 1 1 512 256 4 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AssignTilingTest, Assign_tiling_float16)
{
    optiling::AssignCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Assign",
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "64 1 1 1 1024 256 2 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AssignTilingTest, Assign_tiling_int8)
{
    optiling::AssignCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Assign",
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 1 1 1 2048 256 1 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AssignTilingTest, Assign_tiling_uint8)
{
    optiling::AssignCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Assign",
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "64 1 1 1 2048 256 1 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AssignTilingTest, Assign_tiling_int32)
{
    optiling::AssignCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Assign",
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "64 1 1 1 512 256 4 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AssignTilingTest, Assign_tiling_int64)
{
    optiling::AssignCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Assign",
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 1 1 1 256 256 8 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AssignTilingTest, Assign_tiling_uint64)
{
    optiling::AssignCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Assign",
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 1 1 1 256 256 8 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AssignTilingTest, Assign_tiling_invalid_dtype)
{
    optiling::AssignCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "Assign",
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            {{{4, 4, 4, 4}, {4, 4, 4, 4}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "64 1 1 1 512 256 4 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
