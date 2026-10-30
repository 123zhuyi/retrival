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
 * \file test_circular_pad_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch32/circular_pad_grad_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class CircularPadGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CircularPadGradTiling  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "CircularPadGradTiling  TearDown" << std::endl;
    }
};

TEST_F(CircularPadGradTiling, circular_pad_grad_tiling_test_success)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {0, 0, 100, 100, 100, 100};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{{{1, 1, 500, 500}, {1, 1, 500, 500}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 300, 300}, {1, 1, 300, 300}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);

    uint64_t expectTilingKey = 222;
    std::string expectTilingData = "500 500 300 300 100 100 100 100 0 0 1 1 0 1 158400 ";
    std::vector<size_t> expectWorkspaces = {17410816};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CircularPadGradTiling, circular_pad_grad_tiling_test_failed)
{
    optiling::Tiling4CircularPadCommonCompileInfo compileInfo = {64, 262144, 16777216};
    std::vector<int64_t> constValue = {1, 1, 100, 100, 100, 100};
    gert::TilingContextPara tilingContextPara(
        "CircularPadGrad",
        {{{{1, 1, 500, 500}, {1, 1, 500, 500}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, constValue.data()}},
        {
            {{{1, 1, 300, 300}, {1, 1, 300, 300}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}