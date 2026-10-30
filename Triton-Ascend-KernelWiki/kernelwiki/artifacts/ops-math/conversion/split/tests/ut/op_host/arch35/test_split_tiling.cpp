/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../../split_v/op_host/arch35/split_v_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class SplitTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SplitTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SplitTilingTest TearDown" << std::endl;
    }
};

TEST_F(SplitTilingTest, test_tiling_int32) {
    optiling::SplitVCompileInfo compileInfo;
    compileInfo.core_num = 64;
    compileInfo.maxCoreNum = 64;
    compileInfo.ubSizePlatform = 253952;
    int32_t split_dim = 1;
    gert::TilingContextPara tilingContextPara("Split",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &split_dim},
            {{{11, 16}, {11, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{11, 8}, {11, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{11, 8}, {11, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 101;
    string expectTilingData = "253952 1 0 11 11 11 1 2 2 2 1 8 8 8 8 8 1 1 1 0 8 8 -1 -1 0 0 11 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                              " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
