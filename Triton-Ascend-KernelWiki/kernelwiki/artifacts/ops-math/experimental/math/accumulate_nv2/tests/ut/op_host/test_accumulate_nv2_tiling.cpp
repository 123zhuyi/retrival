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
 * \file accumulate_nv2.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_kernel/accumulate_nv2_tiling_data.h"
#include "../../../op_kernel/accumulate_nv2_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class AccumulateNv2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AccumulateNv2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AccumulateNv2Tiling TearDown" << std::endl; }
};

struct AccumulateNv2CompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isRegbase = false;
};

TEST_F(AccumulateNv2Tiling, accumulate_nv2_test_tiling_case0)
{
    AccumulateNv2CompileInfo compileInfo = {40, 196608, false};
    gert::TilingContextPara tilingContextPara(
        "AccumulateNv2",
        {
            {{{3, 6, 5}, {3, 6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 6, 5}, {3, 6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "446676598880 4294967297 412316893152 104 "; //"96 104 1 1 32736 96 104 0 2 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}