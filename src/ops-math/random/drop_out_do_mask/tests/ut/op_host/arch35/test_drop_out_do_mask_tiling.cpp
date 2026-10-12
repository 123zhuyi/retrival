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
 * \file test_sim_thread_exponential_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../../op_host/arch35/drop_out_do_mask_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class DropOutDoMaskTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DropOutDoMaskTilingTest  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "DropOutDoMaskTilingTest  TearDown" << std::endl;
    }
};

TEST_F(DropOutDoMaskTilingTest, drop_out_do_mask_tiling_ascendc_01)
{
    optiling::DropOutDoMaskCompileInfo compileInfo = {64, 0, 262144};
    gert::TilingContextPara::TensorDescription x({{4095, 1, 3}, {4095, 1, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription mask({{1536}, {1536}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription keep_prob({{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription y({{4095, 1, 3}, {4095, 1, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("DropOutDoMask", {x, mask, keep_prob}, {y}, &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "6 2048 2045 30720 1 2048 1 2045 629145600 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}