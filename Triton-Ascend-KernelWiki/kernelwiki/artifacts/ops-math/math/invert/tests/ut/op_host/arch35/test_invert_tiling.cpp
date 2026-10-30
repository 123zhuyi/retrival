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
 * \file test_invert_tiling.cpp
 * \brief invert tiling ut test
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../../../op_host/arch35/invert_tiling_arch35.h"

using namespace std;

class InvertTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InvertTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InvertTilingTest TearDown" << std::endl;
    }
};

TEST_F(InvertTilingTest, InvertTiling_001)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Invert",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 101;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(InvertTilingTest, InvertTiling_002)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Invert",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 101;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(InvertTilingTest, InvertTiling_003)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Invert",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 101;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
TEST_F(InvertTilingTest, InvertTiling_004)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Invert",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 101;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(InvertTilingTest, InvertTiling_005)
{
    Ops::Base::ElewiseCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Invert",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
