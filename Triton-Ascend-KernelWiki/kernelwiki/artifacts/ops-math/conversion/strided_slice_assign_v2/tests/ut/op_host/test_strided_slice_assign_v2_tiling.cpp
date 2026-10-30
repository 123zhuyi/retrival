/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file test_strided_slice_assign_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/strided_slice_assign_v2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class StridedSliceAssignV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "StridedSliceAssignV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StridedSliceAssignV2Tiling TearDown" << std::endl;
    }
};

TEST_F(StridedSliceAssignV2Tiling, strided_slice_assing_v2_tiling_001) {
    optiling::StridedSliceAssignV2CompileInfo compileInfo = {48, 196608};
    std::vector<int64_t> beginValue = {1, 2, 3};
    std::vector<int64_t> endValue = {4, 6, 7};
    std::vector<int64_t> stridesValue = {2, 3, 1};
    std::vector<int64_t> axesValue = {0, 1, 2};
    gert::TilingContextPara tilingContextPara("StridedSliceAssignV2",
                                              {{{{4, 6, 8}, {4, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2, 2, 4}, {2, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},},
                                              {{{{2, 2, 4}, {2, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "3 4 6 8 0 0 0 0 0 2 2 4 0 0 0 0 0 1 2 3 0 0 0 0 0 2 3 1 0 0 0 0 0 0 48 8 0 0 0 0 0 0 8 4 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StridedSliceAssignV2Tiling, strided_slice_assing_v2_tiling_002) {
    optiling::StridedSliceAssignV2CompileInfo compileInfo = {48, 196608};
    std::vector<int64_t> beginValue = {5, 2, 3};
    std::vector<int64_t> endValue = {-100, 6, 7};
    std::vector<int64_t> stridesValue = {2, 3, 1};
    std::vector<int64_t> axesValue = {0, 1, 2};
    gert::TilingContextPara tilingContextPara("StridedSliceAssignV2",
                                              {{{{4, 6, 8}, {4, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{0, 2, 4}, {0, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
                                               {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},},
                                              {{{{0, 2, 4}, {0, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                                &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "3 4 6 8 0 0 0 0 0 0 2 4 0 0 0 0 0 4 2 3 0 0 0 0 0 2 3 1 0 0 0 0 0 0 48 8 0 0 0 0 0 0 8 4 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
