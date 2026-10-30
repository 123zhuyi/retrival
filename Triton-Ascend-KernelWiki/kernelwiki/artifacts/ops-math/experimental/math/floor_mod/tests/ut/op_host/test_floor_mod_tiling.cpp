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
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "floor_mod_tiling.h"
#include "../../../op_kernel/floor_mod_tiling_data.h"
#include "../../../op_kernel/floor_mod_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class FloorModTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "FloorModTilingTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "FloorModTilingTest TearDown " << endl; }
};

TEST_F(FloorModTilingTest, test_tiling_fp16_basic)
{
    optiling::FloorModCompileInfo compileInfo = {20, 192 * 1024, false};

    gert::TilingContextPara tilingContextPara(
        "FloorMod",
        {
            // 输入 x1, x2: shape 128*64, float16
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 输出 y: shape 128*64, float16
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    // Tiling Key
    uint64_t expectTilingKey = 1315860;

    // 预期 Tiling Data
    string expectTilingData = "34359742592 8192 1024 0 1024 ";

    // 预期 Workspace 大小
    std::vector<size_t> expectWorkspaces = {32 * 1024 * 1024};

    // 执行测试用例
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}