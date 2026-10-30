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
 * \file test_pows_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/pows_tiling.h"

using namespace std;
using namespace ge;

class PowsTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "PowsTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PowsTiling TearDown" << std::endl;
    }
};

TEST_F(PowsTiling, pows_tiling_001) 
{
    int32_t start = 0;
    int32_t stop = 8;
    int32_t num = 10;
    optiling::PowsCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara(
        "Pows",
        {
            {{{2, 2560}, {2, 2560}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 2560}, {2, 2560}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "1 0 0 80 46 112 112 101 16 13824 ";
    std::vector<size_t> expectWorkspaces = {1568};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
