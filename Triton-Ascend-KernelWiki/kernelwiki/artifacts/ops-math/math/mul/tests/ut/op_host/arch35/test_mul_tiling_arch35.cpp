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
#include "../../../../op_host/arch35/mul_tiling_arch35.h"

using namespace std;
using namespace ge;

class MulTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MulTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MulTiling TearDown" << std::endl;
  }
};

TEST_F(MulTiling, mul_test_tiling_001)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967552 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MulTiling, mul_test_tiling_002)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967424 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MulTiling, mul_test_tiling_003)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967552 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MulTiling, mul_test_tiling_004)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_INT8, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MulTiling, mul_test_tiling_005)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_INT16, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_INT16, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_INT16, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967552 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MulTiling, mul_test_tiling_006)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967424 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MulTiling, mul_test_tiling_007)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_BF16, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967552 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MulTiling, mul_test_tiling_008)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_UINT8, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_UINT8, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967808 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MulTiling, mul_test_tiling_009)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967360 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MulTiling, mul_test_tiling_010)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_DOUBLE, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967360 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MulTiling, mul_test_tiling_011)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_COMPLEX32, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_COMPLEX32, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967424 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(MulTiling, mul_test_tiling_012)
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara("Mul",
        {{{{8, 8}, {8, 8}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
         {{{8, 8}, {8, 8}}, ge::DT_COMPLEX64, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_COMPLEX64, ge::FORMAT_ND},},
         &compileInfo);
    uint64_t expectTilingKey = 8;
    string expectTilingData = "64 4294967360 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
