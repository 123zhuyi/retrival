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
#include <gtest/gtest.h>
#include "math/bincount/op_host/arch35/bincount_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
class BincountTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BincountTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BincountTilingTest TearDown" << std::endl;
    }
};

TEST_F(BincountTilingTest, bincount_tiling_test_case1)
{
    optiling::AscendCBincountCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {11};
    gert::TilingContextPara tilingContextPara(
        "Bincount",
        {
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
           {{{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND},},
        },
        &compileInfo);
    uint64_t expectTilingKey = 65792;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(BincountTilingTest, bincount_tiling_test_case2)
{
    optiling::AscendCBincountCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {11};
    gert::TilingContextPara tilingContextPara(
        "Bincount",
        {
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{{11}, {11}}, ge::DT_FLOAT, ge::FORMAT_ND},},
        },
        &compileInfo);
    uint64_t expectTilingKey = 65536;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(BincountTilingTest, bincount_tiling_test_case3)
{
    optiling::AscendCBincountCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {11};
    gert::TilingContextPara tilingContextPara(
        "Bincount",
        {
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
            {{{10}, {10}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
           {{{{11}, {11}}, ge::DT_INT64, ge::FORMAT_ND},},
        },
        &compileInfo);
    uint64_t expectTilingKey = 66050;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(BincountTilingTest, bincount_tiling_test_case4)
{
    optiling::AscendCBincountCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {11};
    gert::TilingContextPara tilingContextPara(
        "Bincount",
        {
            {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
           {{{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND},},
        },
        &compileInfo);
    uint64_t expectTilingKey = 256;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(BincountTilingTest, bincount_tiling_test_case5)
{
    optiling::AscendCBincountCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {11};
    gert::TilingContextPara tilingContextPara(
        "Bincount",
        {
            {{{60000}, {60000}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
            {{{60000}, {60000}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
           {{{{11}, {11}}, ge::DT_FLOAT, ge::FORMAT_ND},},
        },
        &compileInfo);
    uint64_t expectTilingKey = 65536;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(BincountTilingTest, bincount_tiling_test_case6)
{
    optiling::AscendCBincountCompileInfo compileInfo = {64, 262144};
    std::vector<int64_t> inputSizeValues = {11};
    gert::TilingContextPara tilingContextPara(
        "Bincount",
        {
            {{{60000}, {60000}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
            {{{60000}, {60000}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
           {{{{11}, {11}}, ge::DT_INT32, ge::FORMAT_ND},},
        },
        &compileInfo);
    uint64_t expectTilingKey = 65792;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}