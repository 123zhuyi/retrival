/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../../../op_host/arch35/greater_equal_tiling_arch35.h"

using namespace std;
using namespace ge;

class GreaterEqualTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GreaterEqualTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GreaterEqualTilingTest TearDown" << std::endl;
  }
};

TEST_F(GreaterEqualTilingTest, greater_equal_test_tiling_float)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("GreaterEqual",
        {{{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},},
        {{{{4, 2}, {4, 2}}, ge::DT_BOOL, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "8 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GreaterEqualTilingTest, greater_equal_tiling_float16)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("GreaterEqual",
        {{{{4, 2}, {4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 2}, {4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{{{4, 2}, {4, 2}}, ge::DT_BOOL, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "8 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GreaterEqualTilingTest, greater_equal_tiling_int64)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("GreaterEqual",
        {{{{4, 2}, {4, 2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{4, 2}, {4, 2}}, ge::DT_INT64, ge::FORMAT_ND},},
        {{{{4, 2}, {4, 2}}, ge::DT_BOOL, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "8 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GreaterEqualTilingTest, greater_equal_tiling_int32)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("GreaterEqual",
        {{{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND},},
        {{{{4, 2}, {4, 2}}, ge::DT_BOOL, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "8 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GreaterEqualTilingTest, greater_equal_tiling_int8)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("GreaterEqual",
        {{{{4, 2}, {4, 2}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{4, 2}, {4, 2}}, ge::DT_INT8, ge::FORMAT_ND},},
        {{{{4, 2}, {4, 2}}, ge::DT_BOOL, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "8 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GreaterEqualTilingTest, greater_equal_tiling_uint8)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("GreaterEqual",
        {{{{4, 2}, {4, 2}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{4, 2}, {4, 2}}, ge::DT_UINT8, ge::FORMAT_ND},},
        {{{{4, 2}, {4, 2}}, ge::DT_BOOL, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "8 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GreaterEqualTilingTest, greater_equal_tiling_uint64)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("GreaterEqual",
        {{{{4, 2}, {4, 2}}, ge::DT_UINT64, ge::FORMAT_ND},
         {{{4, 2}, {4, 2}}, ge::DT_UINT64, ge::FORMAT_ND},},
        {{{{4, 2}, {4, 2}}, ge::DT_BOOL, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "8 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GreaterEqualTilingTest, greater_equal_tiling_valid_dtype)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("GreaterEqual",
        {{{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{4, 2}, {4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
        {{{{4, 2}, {4, 2}}, ge::DT_BOOL, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "8 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectTilingData, expectWorkspaces);
}
