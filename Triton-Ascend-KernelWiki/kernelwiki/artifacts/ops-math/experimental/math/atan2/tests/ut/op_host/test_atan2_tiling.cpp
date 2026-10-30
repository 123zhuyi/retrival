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
#include "../../../op_kernel/atan2_tiling_data.h"
#include "../../../op_kernel/atan2_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// 1. 先定义 namespace 和 mock 的 CompileInfo 结构体
namespace optiling {
struct Atan2CompileInfo {};
} // namespace optiling

// 2. 然后再使用 using namespace
using namespace optiling;

class Atan2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Atan2Tiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "Atan2Tiling TearDown " << endl; }
};

TEST_F(Atan2Tiling, ascend910b_test_tiling_fp16_001)
{
    optiling::Atan2CompileInfo compileInfo;

    // atan2 是双输入 (x1, x2)，单输出 (y)
    gert::TilingContextPara tilingContextPara(
        "Atan2",
        {
            // Input 0: x1
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // Input 1: x2
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // Output 0: y
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 0; // ELEMENTWISE_TPL_SCH_MODE_0

    string expectTilingData = "128 144 1 1 4080 128 144 0 130560 ";

    std::vector<size_t> expectWorkspaces = {16777728};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}