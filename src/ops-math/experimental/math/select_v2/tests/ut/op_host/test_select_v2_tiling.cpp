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
#include "select_v2_tiling.h"
#include "../../../op_kernel/select_v2_tiling_data.h"
#include "../../../op_kernel/select_v2_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class SelectV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SelectV2Tiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "SelectV2Tiling TearDown " << endl; }
};

TEST_F(SelectV2Tiling, ascend9101_test_tiling_fp16_001)
{
    optiling::SelectV2CompileInfo compileInfo = {40, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "SelectV2",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "8192 8208 1 1 15856 8192 8208 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
