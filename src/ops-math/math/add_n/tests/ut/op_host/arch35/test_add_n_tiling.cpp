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
 * \file test_add_n_tiling.cpp
 * \brief
 */
#include "../../../../op_host/arch35/add_n_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
class AddNTiling : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddNTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddNTiling TearDown" << std::endl;
    }
};

TEST_F(AddNTiling, add_n_tiling)
{
    optiling::AddNCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "AddN",
        {
            {{{1, 1024, 6912}, {1, 1024, 6912}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1024, 6912}, {1, 1024, 6912}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{16, 1, 4, 4, 8}, {16, 1, 4, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 7;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}
