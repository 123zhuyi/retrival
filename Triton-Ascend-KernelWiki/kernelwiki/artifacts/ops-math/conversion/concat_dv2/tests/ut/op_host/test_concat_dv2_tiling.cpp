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
* \file test_concat_dv2_tiling.cpp
* \brief
*/

#include <iostream>
#include <gtest/gtest.h>
#include "conversion/concat_dv2/op_host/concat_dv2_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class ConcatDV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ConcatDV2Tiling  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ConcatDV2Tiling  TearDown" << std::endl;
    }
};

TEST_F(ConcatDV2Tiling, concat_dv2_tiling_test_success)
{
    optiling::Tiling4ConcatDV2CompileInfo compileInfo = {48, 196608, 16777216};
    gert::TilingContextPara tilingContextPara(
        "ConcatDV2",
        {
            {{{1, 32}, {1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 32}, {1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("concat_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
