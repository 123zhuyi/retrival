/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "im2_col_tiling.h"
#include "../../../op_kernel/im2_col_tiling_data.h"
#include "../../../op_kernel/im2_col_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class Im2ColTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Im2ColTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "Im2ColTiling TearDown " << endl; }
};

// TEST_F(Im2ColTiling, ascend9101_test_tiling_fp16_001)
// {
//     optiling::Im2ColCompileInfo compileInfo = {40, 196352, true};
//     gert::TilingContextPara tilingContextPara(
//         "Im2Col",
//         {
//             {{{9, 10, 23, 1971}, {9, 10, 23, 1971}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         {
//             {{{9, 120, 9850}, {9, 120, 9850}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         &compileInfo);
//     uint64_t expectTilingKey = 0;
//     string expectTilingData = "9 10 23 1971 4 3 2 2 0 0 1 1 10 985 9850 120 4079970 10638000 10638000 265950 0 2048
//     2048"; std::vector<size_t> expectWorkspaces = {16777728}; ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS,
//     expectTilingKey, expectTilingData, expectWorkspaces);
// }