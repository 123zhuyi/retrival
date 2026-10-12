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
#include "atan_tiling.h"
#include "../../../op_kernel/atan_tiling_data.h"
#include "../../../op_kernel/atan_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class AtanTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "AtanTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "AtanTiling TearDown " << endl; }
};

TEST_F(AtanTiling, ascend9101_test_tiling_fp16_001)
{
    optiling::AtanCompileInfo compileInfo = {64, 262144, false};
    gert::TilingContextPara tilingContextPara(
        "Atan",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1099511627904 4294967297 1099511627904 6784 "; // test框架是uint64_t读取的，读了2个uint32_t,"128
                                                                              // 256 1 1 128 256 6784 0";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
