/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liang Yanglin <@liang-yanglin>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "cast_v2_tiling.h"
#include "../../../op_kernel/cast_v2_tiling_data.h"
#include "../../../op_kernel/cast_v2_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class CastV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "CastV2Tiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "CastV2Tiling TearDown " << endl; }
};

// TEST_F(CastV2Tiling, ascend9101_test_tiling_i32_f32_001)
// {
//     optiling::CastV2CompileInfo compileInfo = {64, 262144, true};
//     gert::TilingContextPara tilingContextPara(
//         "CastV2",
//         {
//             {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
//         },
//         {
//             {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
//         },
//         &compileInfo);
//     uint64_t expectTilingKey = 0;
//     string expectTilingData = "8192 8208 1 1 32704 8192 8208 0 ";
//     std::vector<size_t> expectWorkspaces = {16777216};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }

// TEST_F(CastV2Tiling, ascend9101_test_tiling_f16_f32_001)
// {
//     optiling::CastV2CompileInfo compileInfo = {64, 262144, true};
//     gert::TilingContextPara tilingContextPara(
//         "CastV2",
//         {
//             {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//         },
//         {
//             {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
//         },
//         &compileInfo);
//     uint64_t expectTilingKey = 0;
//     string expectTilingData = "8192 8208 1 1 43605 8192 8208 0 ";
//     std::vector<size_t> expectWorkspaces = {16777216};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }