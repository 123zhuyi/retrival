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
#include "../../../op_host/coalesce_sparse_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class CoalesceSparseTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "CoalesceSparse Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "CoalesceSparse Tiling TearDown" << std::endl;
    }
};

struct CoalesceSparseCompileInfo {};

TEST_F(CoalesceSparseTiling, test_coalesce_sparse_success) {
    CoalesceSparseCompileInfo compileInfo = {};
    
    gert::TilingContextPara tilingContextPara("CoalesceSparse",
                                              {{{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{10, 2}, {10, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{10, 3}, {10, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{{{7, 2}, {7, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{7, 3}, {7, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                                &compileInfo);
    uint64_t expectTilingKey = 9;
    string expectTilingData = "1 2 3 256 10 4064 0 256 0 10 1 3 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}