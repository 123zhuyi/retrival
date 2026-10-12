/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_chunk_cat_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/chunk_cat_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class ChunkCatTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "ChunkCat Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ChunkCat Tiling TearDown" << std::endl;
    }
};

TEST_F(ChunkCatTilingTest, chunk_cat_float16_021_success_case) {
    optiling::ChunkCatCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ChunkCat",
                                              {{{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{4, 16}, {4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{4, 12}, {4, 12}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                              {{{{4, 36}, {4, 36}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {{"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                               {"num_chunks", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},},
                                               {3}, {1},
                                                &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
