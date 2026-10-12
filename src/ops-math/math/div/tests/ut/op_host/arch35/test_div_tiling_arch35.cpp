
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
 * \file test_div_tiling_arch35.cpp
 * \brief
 */

#include "math/div/op_host/arch35/div_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

struct DivCompileInfo{
    uint64_t coreNum;
    uint64_t ubSize;
};


class DivTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DivTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DivTiling TearDown" << std::endl;
    }
};

TEST_F(DivTiling, div_test_0)
{
    DivCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "Div",
        {
            {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "204800 137438956672 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(DivTiling, div_test_1)
{
    DivCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "Div",
        {
            {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{5, 5, 64, 128}, {5, 5, 64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "204800 137438956672 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
