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
 * \file test_one_hot_tiling_arch35.cpp
 * \brief
 */

#include "../../../../op_host/arch35/one_hot_tiling_arch35.h"
#include "../../../../op_host/arch35/one_hot_tiling.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "any_value.h"

using namespace std;

class OneHotTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "OneHotTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "OneHotTilingTest TearDown" << std::endl;
    }
};

// Helper function to create OpAttr
static gert::TilingContextPara::OpAttr MakeAxisAttr(int64_t axis)
{
    return gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom(axis));
}

// Test: one_hot tiling with basic parameters
TEST_F(OneHotTilingTest, test_tiling_basic)
{
    optiling::OneHotCompileInfo compileInfo;
    compileInfo.core_num = 8;
    compileInfo.ub_size = 262144;
    compileInfo.is_regbase_soc_version = false;

    int32_t depth = 10;

    // Use the test framework's ExecuteTiling function
    gert::TilingContextPara tilingContextPara(
        "OneHot",
        {
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &depth},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{4, 3, 4, 10}, {4, 3, 4, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            MakeAxisAttr(-1),
        },
        &compileInfo);

    uint64_t expectTilingKey = 1000;
    string expectTilingData = "65536 48 0 0 0 48 1 64 202752 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(OneHotTilingTest, test0_tiling)
{
    optiling::OneHotCompileInfo compileInfo;
    compileInfo.core_num = 8;
    compileInfo.ub_size = 262144;
    compileInfo.is_regbase_soc_version = false;

    int32_t depth = 10;

    // Use the test framework's ExecuteTiling function
    gert::TilingContextPara tilingContextPara(
        "OneHot",
        {
            {{{4, 1024, 4}, {4, 1024, 4}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &depth},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{4, 1024, 4, 10}, {4, 1024, 4, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            MakeAxisAttr(-1),
        },
        &compileInfo);

    uint64_t expectTilingKey = 1001;
    string expectTilingData = "18014398510530560 1024 2560 0 0 16384 1 64 202752 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(OneHotTilingTest, test1_tiling)
{
    optiling::OneHotCompileInfo compileInfo;
    compileInfo.core_num = 8;
    compileInfo.ub_size = 262144;
    compileInfo.is_regbase_soc_version = false;

    int32_t depth = 10;

    // Use the test framework's ExecuteTiling function
    gert::TilingContextPara tilingContextPara(
        "OneHot",
        {
            {{{4, 1024, 4}, {4, 1024, 4}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &depth},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{4, 1024, 4, 10}, {4, 1024, 4, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            MakeAxisAttr(-1),
        },
        &compileInfo);

    uint64_t expectTilingKey = 1001;
    string expectTilingData = "18014398510530560 1024 2560 0 0 16384 1 64 202752 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(OneHotTilingTest, test2_tiling)
{
    optiling::OneHotCompileInfo compileInfo;
    compileInfo.core_num = 8;
    compileInfo.ub_size = 262144;
    compileInfo.is_regbase_soc_version = false;

    int32_t depth = 1;

    // Use the test framework's ExecuteTiling function
    gert::TilingContextPara tilingContextPara(
        "OneHot",
        {
            {{{1, 7, 1}, {1, 7, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &depth},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{1, 7, 1, 1}, {1, 7, 1, 1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 1000;
    string expectTilingData = "65536 7 0 0 0 7 1 64 202752 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(OneHotTilingTest, test3_tiling)
{
    optiling::OneHotCompileInfo compileInfo = {64, 262144, true};
    compileInfo.is_regbase_soc_version = false;

    int32_t depth = 1;

    // Use the test framework's ExecuteTiling function
    gert::TilingContextPara tilingContextPara(
        "OneHot",
        {
            {{{1, 7, 1}, {1, 7, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &depth},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{1, 7, 1, 1}, {1, 7, 1, 1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 1000;
    string expectTilingData = "65536 7 0 0 0 7 1 64 202752 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(OneHotTilingTest, test4_tiling)
{
    optiling::OneHotCompileInfo compileInfo = {64, 262144, true};
    compileInfo.is_regbase_soc_version = false;

    int32_t depth = 4;

    // Use the test framework's ExecuteTiling function
    gert::TilingContextPara tilingContextPara(
        "OneHot",
        {
            {{{2147483649, 1, 1}, {2147483649, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &depth},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr},
        },
        {
            {{{4, 2147483649, 1, 1}, {4, 2147483649, 1, 1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 1001;
    string expectTilingData = "18014402808643585 33554432 134217728 0 0 1 2147483649 64 202752 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}