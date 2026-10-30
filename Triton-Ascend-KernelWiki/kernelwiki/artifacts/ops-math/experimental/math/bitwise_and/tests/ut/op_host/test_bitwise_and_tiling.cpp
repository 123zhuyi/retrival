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
#include "bitwise_and_tiling.h"
#include "../../../op_kernel/bitwise_and_tiling_data.h"
#include "../../../op_kernel/bitwise_and_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class BitwiseAndTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "BitwiseAndTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "BitwiseAndTiling TearDown " << endl; }
};

TEST_F(BitwiseAndTiling, ascend910b_test_tiling_INT16_001)
{
    optiling::BitwiseAndCompileInfo compileInfo = {40, 196608, true};
    gert::TilingContextPara tilingContextPara(
        "BitwiseAnd",
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1024, 1024}, {1024, 1024}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "16384 16400 1 1 21824 16384 16400 0 43648 32768 32800 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
