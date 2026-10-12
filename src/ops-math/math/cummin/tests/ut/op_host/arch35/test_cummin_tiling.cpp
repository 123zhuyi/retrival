/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/cummin_tiling.h"

using namespace std;
using namespace ge;

class CumminTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CumminTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CumminTiling TearDown" << std::endl;
    }
};

TEST_F(CumminTiling, Cummin_test_tiling_001)
{
    optiling::CumminCompileInfo compileInfo = {56, 253952, 32, 256, 256};
    int32_t axis = 0;

    gert::TilingContextPara tilingContextPara(
        "Cummin",
        {
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData =
        "1 2 1 0 0 1 2 1 1 1 1 0 1951 0 0 0 0 0 0 0 0 0 0 1 1 2 3903 0 0 0 2 8 1 1 2 3903 0 0 0 1 8 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}