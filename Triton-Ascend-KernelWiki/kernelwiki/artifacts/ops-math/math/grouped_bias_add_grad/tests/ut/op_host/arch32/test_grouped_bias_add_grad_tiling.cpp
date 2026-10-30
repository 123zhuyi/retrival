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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "../../../../op_host/arch32/grouped_bias_add_grad_tiling.h"

using namespace std;

class TilingGroupedBiasAddGrad : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TilingGroupedBiasAddGrad SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TilingGroupedBiasAddGrad TearDown" << std::endl;
    }
};

struct GroupedBiasAddGradCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
};

TEST_F(TilingGroupedBiasAddGrad, ascend910B1_test_tiling__001)
{
    optiling::GroupedBiasAddGradCompileInfo compileInfo = {196608, 48};
    gert::TilingContextPara tilingContextPara(
        "GroupedBiasAddGrad",
        {
            {{{10, 32}, {10, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 32}, {3, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("group_idx_type", Ops::Math::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo);
    uint64_t expectTilingKey = 1000111;
    string expectTilingData = "12884901891 1 1 3 0 32 10 511101108352 0 ";
    std::vector<size_t> expectWorkspaces = {33555968};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}