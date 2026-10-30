/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "../../../../op_host/arch35/trunc_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class TruncTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TruncTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TruncTilingTest TearDown" << std::endl;
    }
};

TEST_F(TruncTilingTest, test_tiling_int32) {
    optiling::TruncCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;

    gert::TilingContextPara tilingContextPara("Trunc",
        {
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 13;
    string expectTilingData = "4 140737488355329 512 1 1 1 512 4 32768 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TruncTilingTest, test_tiling_int8) {
    optiling::TruncCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;

    gert::TilingContextPara tilingContextPara("Trunc",
        {
            {{{39, 15, 40, 14}, {39, 15, 40, 14}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{39, 15, 40, 14}, {39, 15, 40, 14}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 9;
    string expectTilingData = "327600 562949953421376 5120 64 1 1 5120 5040 131072 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}