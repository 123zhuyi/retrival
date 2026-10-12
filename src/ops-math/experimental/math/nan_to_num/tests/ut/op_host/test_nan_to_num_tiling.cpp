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
#include "nan_to_num_tiling.h"
#include "../../../op_kernel/nan_to_num_tiling_data.h"
#include "../../../op_kernel/nan_to_num_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class NanToNumTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "NanToNumTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "NanToNumTiling TearDown " << endl; }
};

TEST_F(NanToNumTiling, ascend910_test_tiling_FLOAT_001)
{
    optiling::NanToNumCompileInfo compileInfo = {40, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "NanToNum",
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "16384 16448 2 2 13056 3328 3392 0 9187061764859101184 4286513152 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
