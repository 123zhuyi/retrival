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
 * \file test_assign_sub_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/assign_sub_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/elewise/elewise_tiling.h"

using namespace std;

class AssignSubTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AssignSubTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AssignSubTilingTest TearDown" << std::endl;
    }
};

TEST_F(AssignSubTilingTest, test_assign_sub_tiling_exception_dtype_001)
{
    optiling::AssignSubCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignSub",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 105;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignSubTilingTest, test_assign_sub_tiling_exception_dtype_0002)
{
    optiling::AssignSubCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignSub",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 104;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignSubTilingTest, test_assign_sub_tiling_exception_dtype_0003)
{
    optiling::AssignSubCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignSub",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 32, 2, 32}, {1, 32, 2, 32}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 2660;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignSubTilingTest, test_assign_sub_tiling_exception_dtype_0004)
{
    optiling::AssignSubCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignSub",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 32, 2, 32}, {1, 32, 2, 32}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 2660;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}
