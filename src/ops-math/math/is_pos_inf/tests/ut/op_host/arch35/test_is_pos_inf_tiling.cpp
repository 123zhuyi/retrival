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
 * \file test_is_pos_inf_tiling.cpp
 * \brief
 */

#include "math/is_pos_inf/op_host/arch35/is_pos_inf_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class IsPosInfTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "IsPosInfTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "IsPosInfTiling TearDown" << std::endl;
    }
};

TEST_F(IsPosInfTiling, test_tiling_fp16_001)
{
    optiling::IsPosInfCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara(
        "IsPosInf",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 203UL;
    string expectTilingData = "8192 24189255811074 4096 2 1 1 4096 4096 5632 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
