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
#include "sign_bits_pack_tiling.h"
#include "../../../op_kernel/sign_bits_pack_tiling_data.h"
#include "../../../op_kernel/sign_bits_pack_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class SignBitsPackTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SignBitsPackTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "SignBitsPackTiling TearDown " << endl; }
};

TEST_F(SignBitsPackTiling, ascend910_test_tiling_FLOAT_001)
{
    optiling::SignBitsPackCompileInfo compileInfo = {40, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "SignBitsPack",
        {
            {{{14}, {14}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 1}, {2, 1}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "128 256 2 1 128 128 128 0 60129542144 274877906946 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
