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
#include "../../../op_host/stack_ball_query_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class StackBallQueryTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "StackBallQuery Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "StackBallQuery Tiling TearDown" << std::endl;
    }
};

TEST_F(StackBallQueryTiling, test_case_1) {
    optiling::StackBallQueryCompileInfo compileInfo = {0, 0};
    
    gert::TilingContextPara tilingContextPara("StackBallQuery",
                                              {{{{3, 20}, {3, 20}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},},
                                              {{{{10, 5}, {10, 5}}, ge::DT_INT32, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("max_radius", Ops::Math::AnyValue::CreateFrom<float>(1.0)),
                                               gert::TilingContextPara::OpAttr("sample_num", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
                                               &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "42949672962 42949672980 34359738370 4575657221408423938 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}