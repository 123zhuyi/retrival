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
#include "axpy_v2_tiling.h"
#include "../../../op_kernel/axpy_v2_tiling_data.h"
#include "../../../op_kernel/axpy_v2_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class AxpyV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "AxpyV2Tiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "AxpyV2Tiling TearDown " << endl; }
};

TEST_F(AxpyV2Tiling, ascend9101_test_tiling_fp16_001)
{
    optiling::AxpyV2CompileInfo compileInfo = {40, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "AxpyV2",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "1024 1152 1 1 21760 1024 1152 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
