/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_transpose_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/transpose_v2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class TransposeV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "TransposeV2 Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "TransposeV2 Tiling TearDown" << std::endl;
    }
};

TEST_F(TransposeV2Tiling, transpose_v2_float16_021_success_case) {
    optiling::Tiling4TransposeV2CompileInfo compileInfo = {48, 196608, 16777216};
    std::vector<int64_t> permValue = {0, 2, 1};  // constValue
    gert::TilingContextPara tilingContextPara("TransposeV2",
                                              {{{{1, 30, 68}, {1, 30, 68}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, permValue.data()},},
                                              {{{{1, 68, 30}, {1, 68, 30}}, ge::DT_INT32, ge::FORMAT_ND},},
                                                &compileInfo);
    uint64_t expectTilingKey = 20;
    string expectTilingData = "0 0 0 0 0 0 0 0 0 2 1 30 68 32 80 256 8 1 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// TEST_F(TransposeV2Tiling, transpose_v2_float16_102_success_case) {
//     optiling::Tiling4TransposeV2CompileInfo compileInfo = {48, 196608, 16777216};
//     std::vector<int64_t> permValue = {1, 0, 2};  // constValue
//     gert::TilingContextPara tilingContextPara("TransposeV2",
//                                               {{{{1, 30, 64}, {1, 30, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//                                                {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, permValue.data()},},
//                                               {{{{30, 1, 64}, {30, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},},
//                                                 &compileInfo);
//     uint64_t expectTilingKey = 121;
//     string expectTilingData = "1 30 64 64 0 30 1 1 511 2 0 0 0 0 0 0 0 0 0 ";
//     std::vector<size_t> expectWorkspaces = {16777216};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }

// TEST_F(TransposeV2Tiling, transpose_v2_float16_0213_success_case) {
//     optiling::Tiling4TransposeV2CompileInfo compileInfo = {48, 196608, 16777216};
//     std::vector<int64_t> permValue = {0, 2, 1, 3};  // constValue
//     gert::TilingContextPara tilingContextPara("TransposeV2",
//                                               {{{{1, 1, 32, 64}, {1, 1, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
//                                                {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, permValue.data()},},
//                                               {{{{1, 32, 1, 64}, {1, 32, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND},},
//                                                 &compileInfo);
//     uint64_t expectTilingKey = 221;
//     string expectTilingData = "1 32 64 64 0 32 1 1 511 2 0 0 0 0 0 0 0 0 0 ";
//     std::vector<size_t> expectWorkspaces = {16777216};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }
