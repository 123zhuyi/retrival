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
 * \file test_addcdiv_tiling.cpp
 * \brief
 */
#include "../../../../op_host/arch35/addcdiv_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
class AddcdivTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddcdivTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddcdivTiling TearDown" << std::endl;
    }
};

TEST_F(AddcdivTiling, addcdiv_test_0)
{
    optiling::AddcdivCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "Addcdiv",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 7;
    string expectTilingData = "11 549755814016 16 1 1 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AddcdivTiling, addcdiv_test_1)
{
    optiling::AddcdivCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "Addcdiv",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 7;
    string expectTilingData = "11 549755814048 13 1 1 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AddcdivTiling, addcdiv_test_2)
{
    optiling::AddcdivCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "Addcdiv",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 7;
    string expectTilingData = "11 549755814016 16 1 1 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AddcdivTiling, addcdiv_test_3)
{
    optiling::AddcdivCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "Addcdiv",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 7;
    string expectTilingData = "11 549755814016 16 1 1 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AddcdivTiling, addcdiv_test_failed_0)
{
    optiling::AddcdivCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "Addcdiv",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100000001000100;
    string expectTilingData =
        "1 512 1 64 1 3 0 1 16384 1 1 1 0 0 0 0 0 1 4 1 0 0 0 0 0 64 4 8 0 0 0 0 0 64 4 8 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 32 8 1 0 0 0 0 0 32 8 1 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AddcdivTiling, addcdiv_test_failed_1)
{
    optiling::AddcdivCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "Addcdiv",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100000001000100;
    string expectTilingData =
        "1 512 1 64 1 3 0 1 16384 1 1 1 0 0 0 0 0 1 4 1 0 0 0 0 0 64 4 8 0 0 0 0 0 64 4 8 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 32 8 1 0 0 0 0 0 32 8 1 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}