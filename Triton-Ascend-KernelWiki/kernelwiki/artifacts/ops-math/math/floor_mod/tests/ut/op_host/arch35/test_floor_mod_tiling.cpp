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
 * \file test_floor_mod_tiling.cpp
 * \brief
 */

#include "math/floor_mod/op_host/arch35/floor_mod_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class FloorModTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "FloorModTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FloorModTiling TearDown" << std::endl;
    }    
};

TEST_F(FloorModTiling, floor_mod_test_0)
{
    optiling::FloorModCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FloorMod",
        {
            {{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "2048 281492156580096 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FloorModTiling, floor_mod_test_float16)
{
    optiling::FloorModCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FloorMod",
        {
            {{{1, 32, 4, 32}, {1, 32, 4, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 32, 4, 32}, {1, 32, 4, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 32, 4, 32}, {1, 32, 4, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

TEST_F(FloorModTiling, floor_mod_test_int64)
{
    optiling::FloorModCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "FloorMod",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}
