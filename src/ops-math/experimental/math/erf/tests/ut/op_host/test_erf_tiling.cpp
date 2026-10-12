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
#include <gtest/gtest.h>
#include "erf_tiling.h"
#include "../../../op_kernel/erf_tiling_data.h"
#include "../../../op_kernel/erf_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

const std::string EXPECT_TILING_DATA_TEST2 = "16384 16384 12884901891 17592186050560 4096 ";

class ErfTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ErfTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "ErfTiling TearDown " << endl; }
};

TEST_F(ErfTiling, ascend9101_test_tiling_fp16_001)
{
    optiling::ErfCompileInfo compileInfo = {40, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "Erf",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0 1024 1 1024 1024 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ErfTiling, ascend9101_test_tiling_fp32_002)
{
    optiling::ErfCompileInfo compileInfo = {40, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "Erf",
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = EXPECT_TILING_DATA_TEST2;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
