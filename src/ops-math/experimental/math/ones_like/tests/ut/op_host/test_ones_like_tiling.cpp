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
#include "ones_like_tiling.h"
#include "../../../op_kernel/ones_like_tiling_data.h"
#include "../../../op_kernel/ones_like_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class OnesLikeTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "OnesLikeTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "OnesLikeTiling TearDown " << endl; }
};

TEST_F(OnesLikeTiling, ascend910B_test_tiling_fp16_001)
{
    optiling::OnesLikeCompileInfo compileInfo = {40, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "OnesLike",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    // string expectTilingData = "4 2048 1 2048 2048 0 1 2048 2048 2048";
    string expectTilingData = "8796093022212 8796093022209 2048 4294969344 8796093024256 "; // uint64
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
