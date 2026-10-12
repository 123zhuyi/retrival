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
 * \file test_drop_out_do_mask_v3_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/drop_out_do_mask_v3_d_tiling_arch35.h"

class DropOutDoMaskV3DTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DropOutDoMaskV3DTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DropOutDoMaskV3DTest TearDown" << std::endl;
  }
};

TEST_F(DropOutDoMaskV3DTiling, drop_out_do_mask_v3_test_0)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "DropOutDoMaskV3D",
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND,},
            {{{64}, {64}}, ge::DT_UINT8, ge::FORMAT_ND,},
            // {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND,},
        },
        {
            {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"keep_prob",Ops::Math::AnyValue::CreateFrom<float>(0.5)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}