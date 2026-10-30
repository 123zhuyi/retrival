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
 * \file test_pow_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/pow_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class PowTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PowTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PowTilingTest TearDown" << std::endl;
    }
};
/*
TEST_F(PowTilingTest, test0_tiling)
{
    optiling::PowCompileInfo compileInfo = {64, 253952, false, 0, 0};

    int8_t x1 = 1;
    int8_t x2 = 1;

    gert::TilingContextPara tilingContextPara(
        "Pow",
        {
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_NCHW, true, &x1},
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_NCHW, true, &x2},
        },
        {
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_NCHW},
        },
        &compileInfo);

    uint64_t expectTilingKey = 5001;
    string expectTilingData =
        " ";
    std::vector<size_t> expectWorkspaces = {16777216};

    // ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, false, 0, true, expectTilingData, expectWorkspaces);
}
*/