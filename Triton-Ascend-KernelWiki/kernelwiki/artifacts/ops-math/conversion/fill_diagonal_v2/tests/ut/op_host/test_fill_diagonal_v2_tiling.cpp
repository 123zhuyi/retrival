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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/fill_diagonal_v2_tiling.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class FillDiagonalV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "FillDiagonalV2 Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "FillDiagonalV2 Tiling TearDown" << std::endl;
    }
};

TEST_F(FillDiagonalV2Tiling, test_fill_diagonal_v2_float) {
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};
    
    gert::TilingContextPara tilingContextPara("FillDiagonalV2",
                                              {{{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))},
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 17 256 196608 5 21 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillDiagonalV2Tiling, test_fill_diagonal_v2_float_cube) {
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};
    
    gert::TilingContextPara tilingContextPara("FillDiagonalV2",
                                              {{{{16, 16, 16}, {16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{16, 16, 16}, {16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{16, 16, 16}, {16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))},
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "4096 273 4096 196608 85 101 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillDiagonalV2Tiling, test_fill_diagonal_v2_float16) {
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};
    
    gert::TilingContextPara tilingContextPara("FillDiagonalV2",
                                              {{{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))},
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 17 256 196608 5 21 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}