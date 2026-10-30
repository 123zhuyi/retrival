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
 * \file test_as_strided_tiling.cpp
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/as_strided_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class AsStridedTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AsStridedTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AsStridedTiling TearDown" << std::endl;
    }
};

TEST_F(AsStridedTiling, as_strided_tiling_test_case1)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2};
    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {{{{4, 4, 3, 3}, {4, 4, 3, 3}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {
                {{{3}, {3}}, ge::DT_INT8, ge::FORMAT_ND},
            },
        },
        &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case2)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2};
    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {{{{4, 4, 3, 3}, {4, 4, 3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {
                {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        },
        &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case3)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2};
    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {{{{4, 4, 3, 3}, {4, 4, 3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {
                {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        },
        &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case4)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2, 6, 6};
    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {{{{4, 4, 3, 3}, {4, 4, 3, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {
                {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        },
        &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case5)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2, 6, 6};
    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {{{{146, 43, 142, 344}, {146, 43, 142, 344}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{7}, {7}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {
                {{{146, 43, 142, 344}, {146, 43, 142, 344}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        },
        &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case6)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3, 2, 6, 6};
    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {{{{146, 43, 142, 344}, {146, 43, 142, 344}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {
                {{{146, 43, 142, 344}, {146, 43, 142, 344}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        },
        &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case7)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3};
    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {{{{1000, 1000, 512}, {1000, 1000, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {
                {{{124, 2, 80}, {124, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        },
        &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case8)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {3};
    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {{{{1000, 1000, 512}, {1000, 1000, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {
                {{{124, 2, 80}, {124, 2, 80}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            },
        },
        &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AsStridedTiling, as_strided_tiling_test_case9)
{
    optiling::AsStridedCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {1};
    gert::TilingContextPara tilingContextPara(
        "AsStrided",
        {{{{3, 3, 3, 3}, {3, 3, 3, 3}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {
                {{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
            },
        },
        &compileInfo);
    uint64_t expectTilingKey = 400;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
