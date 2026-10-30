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
#include "sign_tiling.h"
#include "../../../op_kernel/sign_tiling_data.h"
#include "../../../op_kernel/sign_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class SignTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SignTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "SignTiling TearDown " << endl; }
};

// TEST_F(SignTiling, ascend9101_test_tiling_fp16_001)
// {
//     optiling::SignCompileInfo compileInfo = {64, 262144, true};
//     gert::TilingContextPara tilingContextPara(
//         "Sign",
//         {
//             {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         {
//             {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);
//     uint64_t expectTilingKey = 0;
//     string expectTilingData = "8192 8208 1 1 21840 8192 8208 0 ";
//     std::vector<size_t> expectWorkspaces = {16777216};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }

// TEST_F(SignTiling, ascend9101_test_tiling_bf16_002)
// {
//     optiling::SignCompileInfo compileInfo = {64, 262144, true};
//     gert::TilingContextPara tilingContextPara(
//         "Sign",
//         {
//             {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
//         },
//         {
//             {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
//         },
//         &compileInfo);
//     uint64_t expectTilingKey = 0;
//     string expectTilingData = "8192 8208 1 1 21840 8192 8208 0 ";
//     std::vector<size_t> expectWorkspaces = {16777216};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }

// TEST_F(SignTiling, ascend9101_test_tiling_fp32_003)
// {
//     optiling::SignCompileInfo compileInfo = {64, 262144, true};
//     gert::TilingContextPara tilingContextPara(
//         "Sign",
//         {
//             {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
//         },
//         {
//             {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
//         },
//         &compileInfo);
//     uint64_t expectTilingKey = 0;
//     string expectTilingData = "8192 8200 1 1 16384 8192 8200 0 ";
//     std::vector<size_t> expectWorkspaces = {16777216};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }
