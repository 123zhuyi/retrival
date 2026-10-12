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
 * \file test_assign_add_tiling.cpp
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../../../op_host/arch35/assign_add_tiling_arch35.h"

using namespace std;

class AssignAddTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AssignAddTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AssignAddTiling TearDown" << std::endl;
    }
};

TEST_F(AssignAddTiling, BiasAdd_tiling1)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignAddTiling, test_assign_add_tiling_bf16_002)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignAddTiling, test_assign_add_tiling_fp32_003)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignAddTiling, test_assign_add_tiling_fp32_004)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignAddTiling, test_assign_add_tiling_int8_005)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignAddTiling, test_assign_add_tiling_int32_006)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignAddTiling, test_assign_add_tiling_int64_007)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignAddTiling, test_assign_add_tiling_uint8_008)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignAddTiling, test_assign_add_tiling_exception_dtype_009)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT32, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 2660;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignAddTiling, test_assign_add_tiling_exception_dtype_010)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(AssignAddTiling, test_assign_add_tiling_exception_shape_011)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
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

TEST_F(AssignAddTiling, test_assign_add_tiling_mix_fp32_fp16_001)
{
    optiling::AssignAddCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "AssignAdd",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("use_locking", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 3;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}