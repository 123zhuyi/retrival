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
 * \file test_adjacent_difference_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/adjacent_difference_tiling.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "any_value.h"

using namespace std;

class AdjacentDifferenceTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdjacentDifferenceTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdjacentDifferenceTilingTest TearDown" << std::endl;
    }
};

// Helper function to create OpAttr
static gert::TilingContextPara::OpAttr MakeYdtypeAttr(int64_t yDtype) {
    return gert::TilingContextPara::OpAttr("yDtype", Ops::Math::AnyValue::CreateFrom(yDtype));
}

// Test: adjacent_difference tiling with int64
TEST_F(AdjacentDifferenceTilingTest, test_tiling_int64) {
    optiling::AdjacentDifferenceCompileInfo compileInfo;
    compileInfo.aivCoreNum_ = 8;
    compileInfo.ubSize_ = 262144;
    // adjacent_difference with int64 output, yDtype should be DT_INT64 (6)
    gert::TilingContextPara tilingContextPara("AdjacentDifference",
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                MakeYdtypeAttr(static_cast<int64_t>(ge::DT_INT64)),
                                              },
                                              &compileInfo);
    // Workspace size is 1 (WORKSPACE_SIZE = 1 in tiling implementation)
    std::vector<size_t> expectWorkspaces = {1};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 1, "8192 128 64 ", expectWorkspaces);
}
