/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_range_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/range_tiling_arch35.h"
#include "../../../../op_host/arch35/range_tiling.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class RangeTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RangeTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RangeTilingTest TearDown" << std::endl;
    }
};

TEST_F(RangeTilingTest, test_tiling_int32)
{
    optiling::RangeCompileInfo compileInfo;
    compileInfo.running_core_num = 8;
    compileInfo.totalCoreNum = 8;
    compileInfo.ubSize = 262144;
    gert::TilingContextPara tilingContextPara(
        "Range",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, false, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, false, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, false, nullptr},
        },
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 110018589944848;
    string expectTilingData = "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 32 8 1 0 0 0 0 0 32 8 1 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
