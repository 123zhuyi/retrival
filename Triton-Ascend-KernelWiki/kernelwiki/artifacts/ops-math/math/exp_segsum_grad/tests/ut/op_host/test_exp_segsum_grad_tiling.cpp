/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/exp_segsum_grad_tiling.h"

using namespace std;
using namespace ge;
using namespace gert;

class ExpSegsumGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ExpSegsumGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ExpSegsumGradTiling TearDown" << std::endl;
    }
};

TEST_F(ExpSegsumGradTiling, exp_segsum_grad_tiling_001) {
    optiling::ExpSegsumGradCompileInfo compileInfo = {1, 10};
    gert::TilingContextPara tilingContextPara(
        "ExpSegsumGrad",
        {
            {{{1, 1, 12, 12}, {1, 1, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 12, 12}, {1, 1, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
            {{{1, 1, 12, 12}, {1, 1, 12, 12}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "1 1 12 192 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

