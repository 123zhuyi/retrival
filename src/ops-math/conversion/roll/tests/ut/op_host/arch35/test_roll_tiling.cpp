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
 * \file test_roll_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "conversion/roll/op_kernel/arch35/roll_struct.h"
#include "conversion/roll/op_host/arch35/roll_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class RollTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RollSubTiling2 SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "RollSubTiling2 TearDown" << std::endl;
    }
};

TEST_F(RollTiling, ascend910D1_test_tiling_int32_success_001)
{

    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1}))},
        &compileInfo);
    uint64_t expectTilingKey = 50001;
    string expectTilingData =
        "0 1 2 2 0 0 1 2 2 0 1 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16304 3 2 3 4 0 0 0 0 0 12 4 1 0 "
        "0 0 0 0 1 2 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
TEST_F(RollTiling, ascend910D1_test_tiling_int32_success_002)
{
    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{48974125}, {48974125}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{48974125}, {48974125}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({12578})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0}))},
        &compileInfo);
    uint64_t expectTilingKey = 10000;
    string expectTilingData =
        "0 0 0 0 -1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 765221 765202 1 32768 1 48974125 0 0 "
        "0 0 0 0 0 1 0 0 0 0 0 0 0 12578 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
TEST_F(RollTiling, ascend910D1_test_tiling_int32_success_003)
{
    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{7, 5, 9, 17, 21, 34, 41}, {7, 5, 9, 17, 21, 34, 41}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{7, 5, 9, 17, 21, 34, 41}, {7, 5, 9, 17, 21, 34, 41}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3, 4})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3, 5}))},
        &compileInfo);
    uint64_t expectTilingKey = 20000;
    string expectTilingData =
        "0 63 5 5 2 4 3 8 5 4 3 8 5 2 0 1440 0 0 30 4 0 0 41 41 0 0 48 48 0 0 164 0 0 0 0 0 0 0 32768 7 7 5 9 17 21 34 "
        "41 0 22394610 4478922 497658 29274 1394 41 1 0 0 2 0 3 0 4 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
TEST_F(RollTiling, ascend910D1_test_tiling_int32_success_004)
{
    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{33, 54, 71}, {33, 54, 71}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{33, 54, 71}, {33, 54, 71}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 1, 4})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 2}))},
        &compileInfo);
    uint64_t expectTilingKey = 30001;
    string expectTilingData =
        "0 64 28 18 1 1 1 28 28 1 1 18 18 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 1977 1971 1 32768 3 33 54 71 0 "
        "0 0 0 0 3834 71 1 0 0 0 0 0 2 1 4 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
TEST_F(RollTiling, ascend910D1_test_tiling_int32_success_005)
{
    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{17, 64}, {17, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{17, 64}, {17, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0}))},
        &compileInfo);
    uint64_t expectTilingKey = 30000;
    string expectTilingData =
        "0 17 1 1 0 0 1 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32768 2 17 64 0 0 0 0 0 0 64 1 0 "
        "0 0 0 0 0 3 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
TEST_F(RollTiling, ascend910D1_test_tiling_int32_success_006)
{
    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{1247, 9}, {1247, 9}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1247, 9}, {1247, 9}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 1})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1}))},
        &compileInfo);
    uint64_t expectTilingKey = 50000;
    string expectTilingData =
        "0 63 20 7 0 0 1 20 20 0 1 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16304 2 1247 9 0 0 0 0 0 0 9 "
        "1 0 0 0 0 0 0 3 1 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
TEST_F(RollTiling, ascend910D1_test_tiling_int32_success_007)
{
    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{48, 4, 48, 384}, {48, 4, 48, 384}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{48, 4, 48, 384}, {48, 4, 48, 384}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-6})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0}))},
        &compileInfo);
    uint64_t expectTilingKey = 40000;
    string expectTilingData =
        "0 64 55296 55296 1 1 2 32768 22528 1 2 32768 22528 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 55296 55296 0 "
        "32768 2 48 73728 0 0 0 0 0 0 73728 1 0 0 0 0 0 0 42 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RollTiling, ascend910D1_test_tiling_int32_failed_001)
{
    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 7}))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(RollTiling, ascend910D1_test_tiling_int32_failed_002)
{
    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 8})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 7}))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// x shape != y shape, expect GRAPH_FAILED (EZ0010)
TEST_F(RollTiling, ascend910D1_test_tiling_int32_failed_003)
{
    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 5}, {2, 3, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0}))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// dimNum > 8 (9 dims), expect GRAPH_FAILED (EZ0012)
TEST_F(RollTiling, ascend910D1_test_tiling_int32_failed_004)
{
    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0}))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// shiftsSize > 1 when dims is empty, expect GRAPH_FAILED (EZ0025)
TEST_F(RollTiling, ascend910D1_test_tiling_int32_failed_005)
{
    optiling::RollCompileInfoArch35 compileInfo = {64, 262144, 0, true, platform_ascendc::SocVersion::ASCEND950};
    gert::TilingContextPara tilingContextPara(
        "Roll",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("shifts", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
         gert::TilingContextPara::OpAttr("dims", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({}))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
