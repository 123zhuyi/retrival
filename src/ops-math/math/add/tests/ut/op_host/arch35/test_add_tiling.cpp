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
 * \file test_add_tiling.cpp
 * \brief
 */

#include "math/add/op_host/arch35/add_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class AddTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddTiling TearDown" << std::endl;
    }
};

TEST_F(AddTiling, add_test_0)
{
    optiling::AddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "Add",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 281492156580096 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AddTiling, add_test_2)
{
    optiling::AddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "Add",
        {
            {{{9, 8, 1}, {9, 8, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{9, 8, 1}, {9, 8, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{9, 8, 1}, {9, 8, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "72 4294967552 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
