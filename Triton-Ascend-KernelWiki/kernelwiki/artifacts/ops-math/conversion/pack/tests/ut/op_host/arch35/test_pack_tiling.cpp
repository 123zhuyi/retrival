/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_host/arch35/pack_tiling_arch35.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class PackTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PackTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PackTilingTest TearDown" << std::endl;
    }
};

TEST_F(PackTilingTest, Pack_asc_tiling_UT_notfirst_axis_align_bitwidth4_01)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "Pack",
        {
            {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 12, 4}, {2, 12, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 12114;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PackTilingTest, Pack_asc_tiling_UT_first_axis_align_bitwidth2_02)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "Pack",
        {
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{24576}, {24576}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 2112;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PackTilingTest, Pack_asc_tiling_UT_notfirst_align_same_bitwidth2_03)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "Pack",
        {
            {{{31, 3, 2560}, {31, 3, 2560}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{31, 3, 2560}, {31, 3, 2560}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{31, 3, 2560}, {31, 3, 2560}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{31, 3, 7680}, {31, 3, 7680}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 2112;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PackTilingTest, Pack_asc_tiling_UT_notfirst_axis_align_same_bitwidth8_04)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "Pack",
        {
            {{{16384, 2, 512}, {16384, 2, 512}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{16384, 2, 512}, {16384, 2, 512}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{16384, 4, 512}, {16384, 4, 512}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        },
        {2}, {1}, &compileInfo);
    uint64_t expectTilingKey = 20002;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PackTilingTest, Pack_asc_tiling_UT_notfirst_axis_align_same_bitwidth1_05)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "Pack",
        {
            {{{12, 1, 8, 128}, {12, 1, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{12, 1, 8, 128}, {12, 1, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{12, 1, 16, 128}, {12, 1, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
        },
        {2}, {1}, &compileInfo);
    uint64_t expectTilingKey = 2111;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PackTilingTest, Pack_asc_tiling_UT_first_axis_align_same_bitwidth4_06)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "Pack",
        {
            {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{24576}, {24576}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 2114;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PackTilingTest, Pack_asc_tiling_UT_first_axis_align_same_bitwidth1_07)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "Pack",
        {
            {{{12, 1, 8, 128}, {12, 1, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{12, 1, 8, 128}, {12, 1, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{24, 1, 8, 128}, {24, 1, 8, 128}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        {2}, {1}, &compileInfo);
    uint64_t expectTilingKey = 2111;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(PackTilingTest, Pack_asc_tiling_UT_first_axis_align_same_bitwidth8_08)
{
    optiling::ConcatDCompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "Pack",
        {
            {{{16384, 512, 2}, {16384, 512, 2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{16384, 512, 2}, {16384, 512, 2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{32768, 512, 2}, {32768, 512, 2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        {2}, {1}, &compileInfo);
    uint64_t expectTilingKey = 20002;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
