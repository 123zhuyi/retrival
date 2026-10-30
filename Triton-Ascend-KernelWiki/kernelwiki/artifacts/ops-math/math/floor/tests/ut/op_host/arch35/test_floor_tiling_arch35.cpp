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
 * \file test_floor_tiling_arch35.cpp
 * \brief
 */

#include "math/floor/op_host/arch35/floor_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class FloorTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FloorTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FloorTilingTest TearDown" << std::endl;
    }
};

TEST_F(FloorTilingTest, floor_test_0)
{
    optiling::FloorCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "Floor",
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100000000000100;
    string expectTilingData = "2048 2048 32768 1 1 2048 2048 32768 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}