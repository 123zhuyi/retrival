/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "math/right_shift/op_host/arch35/right_shift_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace std;

class RightShiftTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RightShiftTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RightShiftTilingTest TearDown" << std::endl;
    }
};

TEST_F(RightShiftTilingTest, right_shift_test_tiling_int8)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "RightShift",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 65544;
    string expectTilingData = "8192 34359738880 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RightShiftTilingTest, right_shift_test_tiling_uint8)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "RightShift",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 131080;
    string expectTilingData = "8192 34359738880 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RightShiftTilingTest, right_shift_test_tiling_int16)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "RightShift",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 196616;
    string expectTilingData = "8192 68719476992 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RightShiftTilingTest, right_shift_test_tiling_uint16)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "RightShift",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT16, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT16, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 262152;
    string expectTilingData = "8192 68719476992 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RightShiftTilingTest, right_shift_test_tiling_int32)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "RightShift",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 327688;
    string expectTilingData = "8192 137438953600 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RightShiftTilingTest, right_shift_test_tiling_uint32)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "RightShift",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT32, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT32, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT32, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 393224;
    string expectTilingData = "8192 137438953600 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RightShiftTilingTest, right_shift_test_tiling_int64)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "RightShift",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 458760;
    string expectTilingData = "8192 137438953600 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RightShiftTilingTest, right_shift_test_tiling_uint64)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "RightShift",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 524296;
    string expectTilingData = "8192 137438953600 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RightShiftTilingTest, right_shift_test_tiling_invalid_dtype)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "RightShift",
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {}, &compileInfo);

    uint64_t expectTilingKey = 65536;
    string expectTilingData = "";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
