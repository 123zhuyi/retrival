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
#include "logical_and_tiling.h"
#include "../../../op_kernel/logical_and_tiling_data.h"
#include "../../../op_kernel/logical_and_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class LogicalAndTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "LogicalAndTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "LogicalAndTiling TearDown " << endl; }
};

TEST_F(LogicalAndTiling, ascend9101_test_tiling_bool_001)
{
    optiling::LogicalAndCompileInfo compileInfo = {40, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "LogicalAnd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "2048 2080 1 1 65472 2048 2080 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
